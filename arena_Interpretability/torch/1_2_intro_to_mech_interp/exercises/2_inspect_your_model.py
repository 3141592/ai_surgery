# 2 Inspect your model
# Number of layers
# Number of heads per layer
# Maximum context window
print("------------------------")
print("Load and run a HookedTransformer model")
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

print
print("Inspecting the model:")
print("gpt2_small: ", gpt2_small)

print()
print("Number of layers:", gpt2_small.cfg.n_layers)
print("Number of heads per layer:", gpt2_small.cfg.n_heads)
print("Maximum context window:", gpt2_small.cfg.n_ctx)

print()
print("gpt2_small.cfg: ", gpt2_small.cfg)
