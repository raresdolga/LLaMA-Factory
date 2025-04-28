import random

import numpy as np
import torch
from omegaconf import DictConfig
from torch import nn
from torch.distributions import Categorical
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import Qwen2Config, Qwen2Model


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class PairedTensorDataset(Dataset):
    def __init__(self, tensor1, tensor2):
        assert tensor1.shape[0] == tensor2.shape[0], "Tensors must have the same number of samples"
        self.tensor1 = tensor1.to(torch.long)
        self.tensor2 = tensor2.to(torch.long)

    def __len__(self):
        return self.tensor1.shape[0]

    def __getitem__(self, idx):
        promt = self.tensor1[idx]
        conv = self.tensor2[idx]
        data = torch.cat([promt, conv], dim=0)
        return data, conv

    def __getitems__(self, idx):
        promt = self.tensor1[idx]
        conv = self.tensor2[idx]
        data = torch.cat([promt, conv], dim=1)
        return data, conv


def loss_clip(
    mb_oldlogporb,  # old logprob of mini batch actions collected during the rollout
    mb_newlogprob,  # new logprob of mini batch actions created by the new policy
    mb_advantages,  # mini batch of advantages collected during the the rollout
    configs,
):
    """Policy loss with clipping to control gradients"""
    ratio = torch.exp(mb_newlogprob - mb_oldlogporb)
    policy_loss = -mb_advantages * ratio
    # clipped policy gradient loss enforces closeness
    clipped_loss = -mb_advantages * torch.clamp(ratio, 1 - configs.clip_epsilon, 1 + configs.clip_epsilon)
    pessimistic_loss = torch.max(policy_loss, clipped_loss).mean()
    return pessimistic_loss


def loss_vf(
    mb_oldreturns,  # mini batch of old returns collected during the rollout
    mb_newvalues,  # minibach of values calculated by the new value function
):
    """Enforcing the value function to give more accurate estimates of returns"""
    mb_newvalues = mb_newvalues.view(-1)
    loss = 0.5 * ((mb_newvalues - mb_oldreturns) ** 2).mean()
    return loss


def get_qwen_model(V):
    config = Qwen2Config(
        vocab_size=V + 3,  # same as Qwen2 base vocab
        hidden_size=256,  # much smaller than default
        intermediate_size=512,  # 4x hidden size (approx)
        num_attention_heads=4,  # must divide hidden_size
        num_key_value_heads=4,
        num_hidden_layers=4,  # number of transformer blocks
        max_position_embeddings=512,
        rms_norm_eps=1e-5,
        use_cache=False,
        pad_token_id=V,
        bos_token_id=V + 1,
        eos_token_id=V + 2,
        use_sliding_window=False,
        # attn_implementation="eager",
    )

    # Initialize model from scratch
    model = Qwen2Model(config)
    return model, config


class Agent(nn.Module):
    """an agent that creates actions and estimates values"""

    def __init__(self, V):
        super().__init__()
        self.base_model, self.config = get_qwen_model(V)
        self.value_head = nn.Linear(self.config.hidden_size, 1)
        self.policy_head = nn.Linear(self.config.hidden_size, V)
        # orthogonal initialization with a hi entropy for more exploration at the start
        torch.nn.init.orthogonal_(self.policy_head.weight, 0.01)

    def value_func(self, state):
        outputs = self.base_model(input_ids=state)
        value = self.value_head(outputs.last_hidden_state[:, -1, :])
        return value

    def policy(self, state, action=None):
        outputs = self.base_model(input_ids=state)
        last_hidden_state = outputs.last_hidden_state
        logits = self.policy_head(last_hidden_state[:, -1, :])
        # PyTorch categorical class helpful for sampling and probability calculations
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.value_head(last_hidden_state[:, -1, :])


def create_rollout(
    cur_observation,  # starting observation of shape [num_trajcts, observation_dim]
    labels,  # labels of shape [num_trajcts, T]
    configs,
    agent,
    device,
    all_returns,
):
    """Rollout phase: create parallel trajectories and store them in the rollout storage"""
    # curr_observation: B,M
    # cache empty tensors to store the rollouts
    # Batch, T for rollout, M+T for dim of rolout - artial sequence
    observations = []
    cur_observation = cur_observation.to(device)
    labels = labels.to(device)
    actions = torch.zeros((configs.T, configs.num_trajcts)).to(device)
    logprobs = torch.zeros((configs.T, configs.num_trajcts)).to(device)
    rewards = torch.zeros((configs.T, configs.num_trajcts)).to(device)
    values = torch.zeros((configs.T, configs.num_trajcts)).to(device)

    for t in range(configs.T):
        observations.append(cur_observation)

        # give observation to the model and collect action, logprobs of actions, entropy and value
        with torch.no_grad():
            action, logprob, entropy, value = agent.policy(cur_observation)
        values[t, :] = value.squeeze(dim=-1)
        actions[t, :] = action
        logprobs[t, :] = logprob
        # apply the action to the env and collect observation and reward
        cur_observation = torch.cat([cur_observation, action.unsqueeze(dim=1)], dim=-1)
        reward = (labels[:, t] == action).float()
        rewards[t, :] = reward.to(device)

    all_returns.append(rewards.mean().item())
    # create the rollout storage
    rollout = {
        "cur_observation": cur_observation,
        "observations": observations,
        "actions": actions,
        "logprobs": logprobs,
        "values": values,
        "rewards": rewards,
    }

    return rollout


def gae(
    cur_observation,  # the current state when advantages will be calculated
    rewards,  # rewards collected from trajectories of shape [num_trajcts, max_trajects_length]
    values,  # value estimates collected over trajectories of shape [num_trajcts, max_trajects_length]
    configs,
    agent,
    device,
):
    """Generalized Advantage Estimation (gae) estimating advantage of a particular trajecotry
    vs the expected return starting from a state
    """
    advantages = torch.zeros((configs.num_trajcts, configs.T))
    last_advantage = 0

    # the value after the last step
    with torch.no_grad():
        last_value = agent.value_func(cur_observation).reshape(1, -1)

    # reverse recursive to calculate advantages based on the delta formula
    for t in reversed(range(configs.T)):
        # mask if episode completed after step t
        last_advantage = last_advantage
        delta = rewards[t, :] + configs.gamma * last_value - values[t, :]
        last_advantage = delta + configs.gamma * configs.gae_lambda * last_advantage
        advantages[:, t] = last_advantage
        last_value = values[t, :]

    advantages = advantages.to(device)
    returns = advantages + values.permute(1, 0)

    return advantages, returns


class Storage(Dataset):
    def __init__(self, rollout, advantages, returns):
        # fill in the storage and flatten the parallel trajectories
        self.observations = rollout["observations"]
        self.logprobs = rollout["logprobs"]
        self.actions = rollout["actions"].long()
        self.advantages = advantages
        self.returns = returns

    def __getitem__(self, ix: int):
        item = [self.observations[ix], self.logprobs[ix], self.actions[ix], self.advantages[ix], self.returns[ix]]
        return item

    def __len__(self) -> int:
        return len(self.observations)


def collate_fn(batch):
    data, conv = batch
    return data, conv


############### PPO training ####################################
def train_ppo(
    dataset,
    model: Agent,
    config: Qwen2Config,
    epochs=3,
    learning_rate=1e-4,
    batch_size=16,
    internal_batch_size=32,
):
    device = "cuda:0"

    configs = {
        # experiment arguments
        "exp_name": "cartpole",
        "gym_id": "CartPole-v1",  # the id of from OpenAI gym
        # training arguments
        "learning_rate": 1e-3,  # the learning rate of the optimizer
        "num_epochs": 10000,  # total timesteps of the training
        "max_grad_norm": 0.5,  # the maximum norm allowed for the gradient
        # PPO parameters
        "num_trajcts": 10,  # N
        "T": 4,  # T
        "gamma": 0.99,  # gamma
        "gae_lambda": 0.95,  # lambda for the generalized advantage estimation
        "num_minibatches": 2,  # number of mibibatches used in each gradient
        "update_epochs": 2,  # number of full rollout storage creations
        "clip_epsilon": 0.2,  # the surrogate clipping coefficient
        "ent_coef": 0.01,  # entroy coefficient controlling the exploration factor C2
        "vf_coef": 0.5,  # value function controlling value estimation importance C1
        # visualization and print parameters
        "num_returns_to_average": 3,  # how many episodes to use for printing average return
        "num_episodes_to_average": 23,  # how many episodes to use for smoothing of the return diagram
    }

    # batch_size is the size of the flatten sequences when trajcts are flatten
    configs["batch_size"] = int(configs["num_trajcts"] * configs["T"])
    # number of samples used in each gradient
    configs["minibatch_size"] = int(configs["batch_size"] // configs["num_minibatches"])

    configs = DictConfig(configs)

    loader = DataLoader(dataset, batch_size=configs.num_trajcts, shuffle=True, collate_fn=collate_fn)

    model.to(device)
    agent = model
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    agent.train()
    # track returns
    all_returns = []

    # initialize the game
    # progress bar
    num_updates = configs.num_epochs
    progress_bar = tqdm(total=num_updates)

    for update in range(1, num_updates + 1):
        all_losses = []
        ##############################################
        # Phase 1: rollout creation
        for promt, conv in loader:
            # parallel envs creating trajectories
            rollout = create_rollout(promt, conv, configs, agent, device, all_returns)

            observations = rollout["observations"]
            cur_observation = rollout["cur_observation"]
            rewards = rollout["rewards"]
            values = rollout["values"]

            # print(
            #     f"cur_observation: {cur_observation.shape}, rewards: {rewards.shape}, dones: {dones.shape}, values: {values.shape}"
            # )
            # calculating advantages
            advantages, returns = gae(cur_observation, rewards, values, configs, agent, device)

            ##############################################
            # Phase 2: model update

            # linearly shrink the lr from the initial lr to zero
            frac = 1.0 - (update - 1.0) / num_updates
            optimizer.param_groups[0]["lr"] = frac * configs.learning_rate

            # training loop
            for epoch in range(configs.update_epochs):
                run_loss = 0
                for t in range(configs.T):
                    mb_observations, mb_logprobs, mb_actions, mb_advantages, mb_returns = (
                        observations[t],
                        rollout["logprobs"][t],
                        rollout["actions"][t],
                        advantages[:, t],
                        returns[:, t],
                    )
                    # print(
                    #     f"mb_observations: {mb_observations.shape}, mb_logprobs: {mb_logprobs.shape}, mb_actions: {mb_actions.shape}, mb_advantages: {mb_advantages.shape}, mb_returns: {mb_returns.shape}"
                    # )
                    # we calculate the distribution of actions through the updated model revisiting the old trajectories
                    _, mb_newlogprob, mb_entropy, mb_newvalues = agent.policy(mb_observations, mb_actions)

                    policy_loss = loss_clip(mb_logprobs, mb_newlogprob, mb_advantages, configs)

                    value_loss = loss_vf(mb_returns, mb_newvalues)

                    # average entory of the action space
                    entropy_loss = mb_entropy.mean()

                    # full weighted loss
                    loss = policy_loss - configs.ent_coef * entropy_loss + configs.vf_coef * value_loss

                    optimizer.zero_grad()
                    loss.backward()
                    run_loss += loss.item()

                    # extra clipping of the gradients to avoid overshoots
                    nn.utils.clip_grad_norm_(agent.parameters(), configs.max_grad_norm)
                    optimizer.step()

            all_losses.append(run_loss)
            run_loss = 0
            # progress bar
            if len(all_returns) > configs.num_returns_to_average:
                progress_bar.set_description(
                    f"episode return: {np.mean(all_returns[-configs.num_returns_to_average :]):.2f}"
                )
                progress_bar.refresh()
                progress_bar.update()

            print(f"Data epoch loss: {np.mean(all_losses):.2f}")

    return model


def generate_data(V, N=100, T=128, M=32):
    set_seed(42)
    B = N
    tensor1 = torch.randint(0, V, (B, M))
    tensor2 = torch.randint(0, V, (B, T))
    return tensor1, tensor2


def main():
    V = 100
    T = 4
    N = 10
    M = 32
    tensor1, tensor2 = generate_data(V, N=N, T=T, M=M)
    dataset = PairedTensorDataset(tensor1, tensor2)
    print(len(dataset))
    model = Agent(V)
    config = model.config
    train_ppo(dataset, model, config)


if __name__ == "__main__":
    """
    This is a toy example to test the variational training on a toy dataset. Tries to fixx errors in dummy script (the original version)
    python -m src.quick_test.toy_data2
    """
    main()
