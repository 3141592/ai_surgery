import torch as t
import math
import circuitsvis as cv
import matplotlib.pyplot as plt
from torch import Tensor
from circuitsvis.attention import attention_heads

from transformer_lens import (
    ActivationCache,
    FactoredMatrix,
    HookedTransformer,
    HookedTransformerConfig,
    utils,
)
from transformer_lens.hook_points import HookPoint

print("------------------------")
print("Visualizing Attention Heads")

device = t.device("mps" if t.backends.mps.is_available() else "cuda" if t.cuda.is_available() else "cpu")

gpt2_small: HookedTransformer = HookedTransformer.from_pretrained("gpt2-small")

#gpt2_text = "Natural language processing tasks, such as question answering, machine translation, reading comprehension, and summarization, are typically approached with supervised learning on task-specific datasets."
gpt2_text = "The capital of France is Paris."

gpt_tokens = gpt2_small.to_tokens(gpt2_text)
gpt2_logits, gpt2_cache = gpt2_small.run_with_cache(gpt_tokens, remove_batch_dim=True)

print()
print("type(gpt2_cache):", type(gpt2_cache))

print() 
attention_pattern = gpt2_cache["pattern", 0]
print("attention_pattern shape:", attention_pattern.shape)

print()
gpt2_str_tokens = gpt2_small.to_str_tokens(gpt_tokens)
print("gpt2_str_tokens:", gpt2_str_tokens)

print()
n_layers = gpt2_cache.model.cfg.n_layers
n_heads = gpt2_cache.model.cfg.n_heads

fig, axes = plt.subplots(n_layers, n_heads, figsize=(12, 12))
axes = axes.flatten()
for j in range(n_layers):
    attention_pattern = gpt2_cache["pattern", j]
    for i in range(n_heads):
        head = attention_pattern[i].cpu()
        idx = j * n_heads + i
        ax = axes[idx]
        im = ax.imshow(head, aspect="auto",  vmin=0, vmax=1)
        ax.set_xticks([])
        ax.set_yticks([])
        # top row → head labels
        if j == 0:
            ax.set_title(f"H{i}", fontsize=8)

        # left column → layer labels
        if i == 0:
            ax.set_ylabel(f"L{j}", fontsize=8)

plt.tight_layout()
plt.show()

# Isolate one head across all layers
head_i = 11  # try 1, 3, 7, etc.

fig, axes = plt.subplots(n_layers, 1, figsize=(1, 12))

for j in range(n_layers):
    head = gpt2_cache["pattern", j][head_i].cpu()
    ax = axes[j]
    ax.imshow(head, aspect="auto", vmin=0, vmax=1)
    ax.set_title(f"Layer {j}")
    ax.set_xticks([])
    ax.set_yticks([])

plt.tight_layout()
plt.show()

for layer in [0, 10]:
    head = gpt2_cache["pattern", layer][head_i].cpu()
    print(f"\n--- Layer {layer} --- Head {head_i} ---")
    for q in range(len(gpt2_str_tokens)):
        top_k = head[q].topk(2)
        print(gpt2_str_tokens[q], "->", [gpt2_str_tokens[idx] for idx in top_k.indices])