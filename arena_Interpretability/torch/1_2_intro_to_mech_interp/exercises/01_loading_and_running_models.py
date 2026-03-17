# 1_Loading and running models
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

prompt = "The capital of France is "

print("Prompt: ", prompt)

print("Model answer: ", gpt2_small.generate(prompt, max_new_tokens=5))
