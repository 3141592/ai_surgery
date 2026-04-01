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

print("Tokenization Investigations")
print()
print("to_str_tokens:")
print(gpt2_small.to_str_tokens("gpt2"))

print()
print("to_str_tokens:")
print(gpt2_small.to_str_tokens(["gpt2","gpt2"]))

print()
print("to_tokens")
tokens = gpt2_small.to_tokens("gpt2")
print(tokens)

print()
print("to_string:")
print(gpt2_small.to_string(tokens))