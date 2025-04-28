import random

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import Qwen2Config, Qwen2ForCausalLM


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


class Storage(Dataset):
    N_SAMPLES = 3

    def __init__(self, promt, neg_samples, old_log_probs, q):
        self.promt = promt
        self.samples = neg_samples.permute(1, 0, 2)
        self.old_log_probs = old_log_probs.permute(1, 0, 2)
        self.q = q.permute(1, 0)

    @classmethod
    def generate_samples(cls, input_ids, conv, model, device):
        prompts = input_ids[:, : -conv.shape[1]]
        with torch.no_grad():
            generated_samples, log_probs_samples, _ = sample_with_probsV2(
                model,
                {"input_ids": prompts},
                conv,
                n=cls.N_SAMPLES,
                max_new_tokens=conv.shape[1],
                temperature=1.0,
                device=device,
            )

        generated_samples = generated_samples.to(device)
        log_probs_samples = log_probs_samples.to(device)
        # add the true label
        with torch.no_grad():
            outputs = model(input_ids=input_ids)
            log_probs = F.log_softmax(outputs.logits, dim=-1)
            log_probs = log_probs[:, prompts.shape[-1] - 1 : -1, :]

        log_probs = log_probs.gather(dim=-1, index=conv.unsqueeze(-1))
        log_probs = log_probs.squeeze(-1)
        # The true label is always the last sample
        generated_samples = torch.cat([generated_samples, conv.unsqueeze(0)], dim=0)
        log_probs_samples = torch.cat([log_probs_samples, log_probs.unsqueeze(0)], dim=0)
        q, _ = compute_variationalOld(generated_samples, log_probs_samples)
        return cls(prompts, generated_samples, log_probs_samples, q)

    def __len__(self):
        return self.promt.shape[0]

    def __getitem__(self, idx):
        return self.promt[idx], self.samples[idx], self.old_log_probs[idx], self.q[idx]

    def __getitems__(self, idx):
        return self.promt[idx], self.samples[idx], self.old_log_probs[idx], self.q[idx]


def collate_fn(batch):
    data, conv = batch
    return data, conv


def collate_fn_storage(batch):
    return batch


def generate_data(V, N=100):
    set_seed(42)

    B = N
    T = 128
    M = 32
    tensor1 = torch.randint(0, V, (B, M))
    tensor2 = torch.randint(0, V, (B, T))
    return tensor1, tensor2


def get_model(V):
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
    model = Qwen2ForCausalLM(config)
    model.eval()
    return model, config


def train_likelihood(dataset, model: Qwen2ForCausalLM, epochs=3, learning_rate=1e-4, batch_size=2):
    device = "cuda:0"
    model.to(device)

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs * len(loader))

    loss_fn = nn.CrossEntropyLoss()

    model.train()
    for epoch in tqdm(range(epochs)):
        total_loss = 0
        for step, (input_ids, conv) in enumerate(loader):
            all_data, conv = input_ids.to(device), conv.to(device)

            outputs = model(input_ids=all_data)
            logits = outputs.logits  # shape: [batch, seq_len, vocab]
            target = conv
            logits = logits[:, -target.shape[1] - 1 : -1, :]
            loss = loss_fn(logits.reshape(-1, logits.size(-1)), target.reshape(-1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(loader)
        print(f"Epoch {epoch + 1} | Loss: {avg_loss:.4f}")


############### Variational Training ####################################
def top_k_top_p_filtering(logits, top_k=0, top_p=1.0, filter_value=-9e15):
    """Apply top-k and/or nucleus (top-p) filtering to logits.

    Args:
        logits (torch.Tensor): Logits distribution of shape (vocab_size).
        top_k (int): Keep only top k tokens with highest logits (top-k filtering).
        top_p (float): Keep the top tokens with cumulative probability >= top_p (nucleus filtering).
        filter_value (float): Value to assign to filtered logits.

    Returns:
        torch.Tensor: Filtered logits.
    """
    logits = logits.clone()
    batch_size = logits.shape[0]
    vocab_size = logits.shape[-1]
    # Top-K filtering
    # if top_k > 0:
    #     top_k = min(top_k, logits.size(-1))  # Safety check
    #     indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
    #     logits[indices_to_remove] = filter_value

    if top_k > 0:
        top_k = min(top_k, vocab_size)
        topk_vals, _ = torch.topk(logits, top_k, dim=-1)
        kth_vals = topk_vals[:, -1].unsqueeze(1)  # shape: [batch_size, 1]
        logits = torch.where(logits < kth_vals, torch.full_like(logits, filter_value), logits)

    # Top-P (nucleus) filtering
    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.softmax(sorted_logits, dim=-1).cumsum(dim=-1)

        # Remove tokens with cumulative prob above threshold
        sorted_indices_to_remove = cumulative_probs > top_p

        # Shift the mask right to keep at least one token
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        # Scatter to original indexing
        # indices_to_remove = sorted_indices[sorted_indices_to_remove]
        # logits[indices_to_remove] = filter_value
        for i in range(batch_size):
            indices = sorted_indices[i][sorted_indices_to_remove[i]]
            logits[i, indices] = filter_value

    return logits


def sample_with_probsV2(
    model: Qwen2ForCausalLM,
    tokenized_prompt,
    conv,
    n=1,
    max_new_tokens=50,
    temperature=1.0,
    top_k=50,
    top_p=0.95,
    device="cuda" if torch.cuda.is_available() else "cpu",
):
    """Batched N sampling.

    Args:
        model (Qwen2ForCausalLM): _description_
        tokenized_prompt (_type_): _description_
        n (int, optional): _description_. Defaults to 1.
        max_new_tokens (int, optional): _description_. Defaults to 50.
        temperature (float, optional): _description_. Defaults to 1.0.
        top_k (int, optional): _description_. Defaults to 50.
        top_p (float, optional): _description_. Defaults to 0.95.
        device (str, optional): _description_. Defaults to "cuda"iftorch.cuda.is_available()else"cpu".
        rewards (torch.Tensor, optional): _description_. The log probs of the true label at posiiton t. Used for a specific type of reard (log prob reward)

    Returns:
        _type_: _description_
    """
    model.eval()

    input_ids = tokenized_prompt["input_ids"].to(device)
    batch_size = input_ids.shape[0]
    input_ids = input_ids.repeat(n, 1)
    conv = conv.to(device)
    conv = conv.repeat(n, 1)
    curr_input = input_ids.clone()
    probs = []
    rewards = []
    # Initialize past_key_values (cache)
    past_key_values = None
    feed_input = curr_input
    for i in range(max_new_tokens):
        outputs = model(input_ids=feed_input, past_key_values=past_key_values, use_cache=True)
        logits = outputs.logits  # shape: [1, seq_len, vocab_size]
        # Update past_key_values for the next step
        past_key_values = outputs.past_key_values
        next_token_logits = logits[:, -1, :]  # take last token

        old_log_probs = next_token_logits.log_softmax(dim=-1)
        reward_logits = next_token_logits.gather(dim=-1, index=conv[:, i].unsqueeze(-1)).squeeze(-1)
        # Apply temperature
        next_token_logits = next_token_logits / temperature
        next_token_logits = top_k_top_p_filtering(next_token_logits, top_k=top_k, top_p=top_p)
        # Convert to probabilities
        log_probs = F.log_softmax(next_token_logits, dim=-1)
        # Sample next token
        dist = torch.distributions.Categorical(logits=log_probs)
        next_token = dist.sample()  # shape: (batch_size,)
        # Get prob of selected token
        # prob = log_probs[torch.arange(log_probs.shape[0], device=device), next_token]
        prob = old_log_probs[torch.arange(log_probs.shape[0], device=device), next_token]
        next_token = next_token.unsqueeze(1)
        probs.append(prob)
        rewards.append(reward_logits)
        # Append token to input
        curr_input = torch.cat([curr_input, next_token], dim=1)
        feed_input = next_token

    probs = torch.stack(probs, dim=-1)
    rewards = torch.stack(rewards, dim=-1)
    # get rid of the promt
    curr_input = curr_input[:, input_ids.shape[-1] :]

    generated_samples = curr_input.reshape(n, batch_size, -1)
    probs_samples = probs.reshape(n, batch_size, -1)
    rewards = rewards.reshape(n, batch_size, -1)
    return generated_samples, probs_samples, rewards


def compute_variationalOld(samples, log_probs):
    T = log_probs.shape[1]
    target = samples[-1]
    eps = 1e-30
    R = (samples == target).float().mean(dim=-1)  # .all(dim=-1)
    log_R = torch.log(R + eps)
    # log_R = torch.where(R == 0, -9e20, log_R)
    log_prob_seq = log_probs.sum(dim=-1)
    unnormalized_q = T * log_R + log_prob_seq
    dtype = unnormalized_q.dtype
    mini = torch.finfo(dtype).min
    unnormalized_q = torch.where(R == 0, mini, unnormalized_q)
    q = torch.softmax(unnormalized_q, dim=0)
    print(R)
    print(log_R)
    print(q)
    print(log_prob_seq)
    print("DONE")
    return q, log_prob_seq


def train_variational(
    dataset,
    model: Qwen2ForCausalLM,
    config: Qwen2Config,
    epochs=3,
    learning_rate=1e-4,
    batch_size=2,
    internal_batch_size=32,
):
    internal_updates = 100
    device = "cuda:0"
    pad_id = config.pad_token_id

    model.to(device)
    # target_model.to(device2)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs * len(loader))
    # loss_fn = nn.CrossEntropyLoss()
    model.train()
    for epoch in tqdm(range(epochs)):
        total_loss = 0
        loss_iterative = []
        it = 0
        for step, (input_ids, conv) in enumerate(loader):
            all_data, conv = input_ids.to(device), conv.to(device)
            storage = Storage.generate_samples(all_data, conv, model, device)

            storage_dl = DataLoader(
                storage, batch_size=internal_batch_size, shuffle=True, collate_fn=collate_fn_storage
            )
            for l in range(internal_updates):
                for promt, samples, old_log_probs, q in storage_dl:
                    B, S, T = samples.shape
                    samples = samples.reshape(-1, samples.shape[-1])
                    outputs = model(input_ids=samples)
                    log_probs = F.log_softmax(outputs.logits, dim=-1)
                    log_probs = log_probs[:, promt.shape[-1] - 1 : -1, :]
                    completion = samples[:, promt.shape[-1] :]
                    log_probs = log_probs.gather(dim=-1, index=completion.unsqueeze(-1)).squeeze(-1)
                    log_probs = log_probs.reshape(B, S, log_probs.shape[-1])

                    log_prob_seq = log_probs.sum(dim=-1)
                    loss = -torch.mean(q * log_prob_seq)  # - 100 * log_probs.mean()
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    scheduler.step()
                    it += 1
                    loss_iterative.append(loss.item())

                    if it % 10 == 0:
                        print(f"Step {it} | Loss: {np.mean(loss_iterative):.4f}")
                        loss_iterative = []

            total_loss += loss.item()

        avg_loss = total_loss / len(loader)
        print(f"Epoch {epoch + 1} | Loss: {avg_loss:.4f}")
    return model


def main():
    V = 100
    tensor1, tensor2 = generate_data(V, N=10)
    dataset = PairedTensorDataset(tensor1, tensor2)
    print(len(dataset))
    model, config = get_model(V)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params / 10**6}M")
    # train_likelihood(dataset, model, epochs=100, learning_rate=1e-3, batch_size=32)
    model = train_variational(dataset, model, config=config, epochs=70, learning_rate=1e-3, batch_size=20)
    model.eval()
    model.to("cuda:0")
    promt = tensor1[0:32, :]
    conv = tensor2[0:32, :]
    promt = promt.to("cuda:0")
    conv = conv.to("cuda:0")
    # outputs = model.generate(
    #     inputs=promt,
    #     max_new_tokens=conv.shape[1],
    #     temperature=1,
    #     do_sample=True,
    #     top_k=50,
    #     top_p=0.95,
    #     pad_token_id=V,
    # )

    generated_samples, log_sample_prob, _ = sample_with_probsV2(
        model, {"input_ids": promt}, n=1, max_new_tokens=conv.shape[1], conv=conv
    )
    generated_samples = generated_samples[0]
    print(generated_samples.shape, log_sample_prob.shape)
    print(generated_samples.shape, conv.shape)
    print((generated_samples == conv).float().mean())


if __name__ == "__main__":
    """
    This is a toy example to test the variational training on a toy dataset. Tries to fixx errors in dummy script (the original version)
    python -m src.quick_test.toy_data2
    """
    main()
