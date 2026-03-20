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
head_i = 0
head = attention_pattern[head_i].cpu()

tokens = gpt2_str_tokens

for q in range(len(tokens)):
    top_k = head[q].topk(3)
    print(f"\nQuery: {tokens[q]}")
    for idx, val in zip(top_k.indices, top_k.values):
        print(f"  attends to: {tokens[idx]}  ({val.item():.3f})")

fig, ax = plt.subplots(figsize=(10, 8))
im = ax.imshow(head, aspect="auto")

ax.set_title(f"Layer 0 Head {head_i} Attention Pattern")
ax.set_xlabel("Key token")
ax.set_ylabel("Query token")

ax.set_xticks(range(len(tokens)))
ax.set_yticks(range(len(tokens)))
ax.set_xticklabels(tokens, rotation=90)
ax.set_yticklabels(tokens)

fig.colorbar(im, ax=ax)
plt.tight_layout()
plt.show()