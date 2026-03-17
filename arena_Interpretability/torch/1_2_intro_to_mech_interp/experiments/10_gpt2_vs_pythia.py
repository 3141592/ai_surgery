import torch as t
from transformer_lens import HookedTransformer
import transformer_lens.utils as utils

device = utils.get_device()
prompt = "Q: The capital of France is?\nA:"

gpt2 = HookedTransformer.from_pretrained("gpt2-small", device=device)
pythia = HookedTransformer.from_pretrained("pythia-1b", device=device)

def show_top_tokens(model, prompt, k=10):
    tokens = model.to_tokens(prompt)
    logits = model(tokens)
    last_logits = logits[0, -1]
    topk = t.topk(last_logits, k)

    print(f"\nModel: {model.cfg.model_name}")
    for rank, token_id in enumerate(topk.indices):
        token_str = model.to_string(token_id.unsqueeze(0))
        logit_val = topk.values[rank].item()
        print(f"{rank:2d}  id={token_id.item():>6}  logit={logit_val:8.3f}  token={repr(token_str)}")

show_top_tokens(gpt2, prompt)
show_top_tokens(pythia, prompt)

import torch as t

def show_token_stats(model, prompt, target_str=" Paris"):
    tokens = model.to_tokens(prompt)
    logits = model(tokens)
    last_logits = logits[0, -1]
    probs = t.softmax(last_logits, dim=-1)

    target_id = model.to_single_token(target_str)
    target_logit = last_logits[target_id].item()
    target_prob = probs[target_id].item()

    rank = (last_logits > last_logits[target_id]).sum().item() + 1

    print(f"\nModel: {model.cfg.model_name}")
    print(f"Target token: {repr(target_str)}")
    print(f"Token id: {target_id}")
    print(f"Logit: {target_logit:.4f}")
    print(f"Probability: {target_prob:.6%}")
    print(f"Rank: {rank}")

show_token_stats(gpt2, prompt)
show_token_stats(pythia, prompt)