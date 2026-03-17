import torch as t
import math
from torch import Tensor

from transformer_lens import (
    ActivationCache,
    FactoredMatrix,
    HookedTransformer,
    HookedTransformerConfig,
    utils,
)
from transformer_lens.hook_points import HookPoint

print("------------------------")
print("Caching all Activations")
print()

device = t.device("mps" if t.backends.mps.is_available() else "cuda" if t.cuda.is_available() else "cpu")

gpt2_small: HookedTransformer = HookedTransformer.from_pretrained("gpt2-small")

gpt2_text = "Natural language processing tasks, such as question answering, machine translation, reading comprehension, and summarization, are typically approached with supervised learning on task-specific datasets."

gpt_tokens = gpt2_small.to_tokens(gpt2_text)
print()
print("Tokens shape:", gpt_tokens.shape)
print("Tokens:", gpt_tokens)

print()
print("Tokenized gpt2_text length:", len(gpt2_small.to_str_tokens(gpt_tokens)))
print("Tokenized gpt2_text:", gpt2_small.to_str_tokens(gpt_tokens))

gpt2_logits, gpt2_cache = gpt2_small.run_with_cache(gpt_tokens, remove_batch_dim=True)
print()
print(type(gpt2_logits), type(gpt2_cache))

print()
print("Exercise - verify activations")

layer0_pattern_from_cache = gpt2_cache["pattern", 0]

# YOUR CODE HERE - define `layer0_pattern_from_q_and_k` manually, by manually performing the
# steps of the attention calculation (dot product, masking, scaling, softmax)
print()
print("layer0_pattern_from_cache:")
print("dim 0 → head")
print("dim 1 → query position (who is looking)")
print("dim 2 → key position (who is being looked at)")
print("layer0_pattern_from_cache:",layer0_pattern_from_cache.shape)
print()
print("cache q, k:")
print("dim 0 → token position (query position)")
print("dim 1 → head")
print("dim 2 → vector (d_head)")
print("gpt2_cache['q', 0]:", gpt2_cache["q", 0].shape)
print("gpt2_cache['k', 0]:", gpt2_cache["k", 0].shape)

# Perform matrix multiplication for QK^T
print()
print("Perform matrix multiplication for QK^T")
Q_h = gpt2_cache["q", 0][:, 0, :]
K_h = gpt2_cache["k", 0][:, 0, :]

print()
print("type(Q_h): ", type(Q_h))
print("type(K_h.transpose): ", type(K_h.t()))
scores = t.matmul(Q_h, K_h.t())
print("scores: ", scores)

print()
print("Shape of QK^T:", scores.shape)

print()
d_head = Q_h.shape[-1]
print("type(d_head): ", type(d_head))
print("d_head:", d_head)

print()
print("Apply scaling to QK^T by sqrt(d_head)")
scaled_scores = scores/math.sqrt(d_head)

print()
print("scaled_scores.shape: ", scaled_scores.shape)
print("scaled_scores: ", scaled_scores)

print()
print("Apply softmax to get attention pattern")
probabilities = t.softmax(scaled_scores, dim=-1)

print()
print("probabilities.shape: ", probabilities.shape)
print("probabilities: ", probabilities)

layer0_pattern_from_q_and_k = probabilities
t.testing.assert_close(layer0_pattern_from_cache, layer0_pattern_from_q_and_k)
print("Tests passed!")
