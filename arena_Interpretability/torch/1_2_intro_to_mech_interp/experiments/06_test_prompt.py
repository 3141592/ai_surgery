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
prompt = "The capital of France is"
answer = "Paris"
utils.test_prompt(prompt, answer, gpt2_small)

print()
print("Rephrased prompt:")
prompt = "What is the capital of France?"
utils.test_prompt(prompt, answer, gpt2_small)

print()
print("Rephrased prompt again:")
utils.test_prompt("The capital of France is the city of", " Paris", gpt2_small)
