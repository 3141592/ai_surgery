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

gpt2_text = "Natural language processing tasks, such as question answering, machine translation, reading comprehension, and summarization, are typically approached with supervised learning on task-specific datasets."

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
print("Layer 0 Head Attention Patterns:")

head0 = attention_pattern[0].cpu()

plt.figure(figsize=(8, 6))
plt.imshow(head0, aspect="auto")
plt.colorbar()
plt.xlabel("Key position")
plt.ylabel("Query position")
plt.title("Layer 0 Head 0 Attention Pattern")
plt.show()