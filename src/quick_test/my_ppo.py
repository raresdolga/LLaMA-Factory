import random

import numpy as np
import torch
from omegaconf import DictConfig
from torch import nn
from torch.distributions import Categorical
from torch.utils.data import DataLoader, Dataset
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


def collate_fn(batch):
    data, conv = batch
    return data, conv


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


def generate_trajectory(model, promt, generations):
    trajectory = []
    curr_gen = promt

    for t in range(generations.shape[1]):
        # trajectory.append( (state, token, logpi_old, reward, V(state)) )
        with torch.no_grad():
            action, log_prob, entropy, value_fn = model.policy(state=curr_gen)
        rewards = (action == generations[:, t]).float()
        for i in range(len(action)):
            trajectory.append((curr_gen[i], action[i], log_prob[i], entropy[i], value_fn[i], rewards[i]))
        curr_gen = torch.cat([curr_gen, action.unsqueeze(1)], dim=1)
    return trajectory


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
    def __init__(self, all_trajectories):
        self.all_trajectories = all_trajectories

    def __len__(self):
        return len(self.all_trajectories)

    def __getitem__(self, idx):
        return self.all_trajectories[idx]

    def __getitems__(self, idx):
        return [self.all_trajectories[i] for i in idx]


def collate_storage(batch):
    print("collate_storage", batch)
    curr_gen, action, log_prob, entropy, value_fn, rewards = [], [], [], [], [], []
    for i in range(len(batch)):
        curr_gen.append(batch[i][0])
        action.append(batch[i][1])
        log_prob.append(batch[i][2])
        entropy.append(batch[i][3])
        value_fn.append(batch[i][4])
        rewards.append(batch[i][5])
    return (
        torch.stack(curr_gen),
        torch.stack(action),
        torch.stack(log_prob),
        torch.stack(entropy),
        torch.stack(value_fn),
        torch.stack(rewards),
    )


def train_ppo(model, train_configs, dataset):
    epochs = 100
    batch_size = 10
    learning_rate = 1e-4
    weight_decay = 0.01
    betas = (0.9, 0.999)
    eps = 1e-8
    device = "cuda:0"

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay, betas=betas, eps=eps
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

    for epoch in range(epochs):
        all_trajectories = []
        for promt, contin in loader:
            trajectory = generate_trajectory(model, promt, contin)
            all_trajectories.append(trajectory)

        # storage = Storage(all_trajectories)
        # storage_loader = DataLoader(storage, batch_size=batch_size, shuffle=True, collate_fn=collate_storage)
        for ue in range(train_configs.update_epochs):
            for t in range(train_configs.T):
                for k in range(0, len(all_trajectories), batch_size):
                    curr_gen, action, log_prob, entropy, value_fn, rewards = [], [], [], [], [], []
                    batch_trajectories
                    for j in range(0, len(batch_trajectories)):
                        batch = all_trajectories[j]
                        curr_gen.append(batch[0])
                        action.append(batch[1])
                        log_prob.append(batch[2])
                        entropy.append(batch[3])
                        value_fn.append(batch[4])
                        rewards.append(batch[5])

                curr_gen, action, log_prob, entropy, value_fn, rewards = batch
                logpi_current = model.policy(curr_gen, action)
                r_t = torch.exp(logpi_current - log_prob)
                At = gae(curr_gen, rewards, value_fn, train_configs, model, device)
                loss_unclipped = r_t * At
                loss_clipped = torch.clip(r_t, 1 - eps, 1 + eps) * At
                loss_token = -torch.min(loss_unclipped, loss_clipped)
                print(loss_token.shape)
                loss_token = loss_token.mean()
                optimizer.zero_grad()
                loss_token.backward()
                optimizer.step()


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
    configs = {
        # experiment arguments
        "exp_name": "cartpole",
        "gym_id": "CartPole-v1",  # the id of from OpenAI gym
        # training arguments
        "learning_rate": 1e-3,  # the learning rate of the optimizer
        "num_epochs": 10000,  # total timesteps of the training
        "max_grad_norm": 0.5,  # the maximum norm allowed for the gradient
        # PPO parameters
        "num_trajcts": N,
        "T": T,
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
    configs = DictConfig(configs)
    model = Agent(V)
    train_ppo(model, configs, dataset)


if __name__ == "__main__":
    main()
