# Transformer architecture
#   The weights W_K, W_Q, W_V mapping the residual stream to queries, keys and values are 3 separate matrices, rather than big concatenated one.
#   The weight matrices W_K, W_Q, W_V, W_O and activations have separate head_index and d_head axes, rather than flattening them into one big axis.
#   The activations all have shape [batch, position, head_index, d_head].
#   W_K, W_Q, W_V have shape [head_index, d_model, d_head] and W_O has shape [head_index, d_head, d_model]
print("------------------------")
print("Running your model")
import torch as t

from transformer_lens import (
    ActivationCache,
    FactoredMatrix,
    HookedTransformer,
    HookedTransformerConfig,
    utils,
)
from transformer_lens.hook_points import HookPoint

device = t.device("mps" if t.backends.mps.is_available() else "cuda" if t.cuda.is_available() else "cpu")

gpt2_small: HookedTransformer = HookedTransformer.from_pretrained("gpt2-small")

prompt = "The capital of France is "

print()
print("Showing logits:")

print()
logits = gpt2_small(prompt, return_type="logits")
print(f"Logits: {logits.shape}")
logits = gpt2_small(prompt, return_type="logits")
print(f"Logits: {logits.shape}")

print()
print("Showing logit example:")
print(f"Logits for the last token: {logits[0, -1, :10]}")

print()
print("Showing logit example:")
print(f"Logits for the last token: {logits[0, -1, :10]}")
