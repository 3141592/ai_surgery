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

print()
prompt = "The capital of France is "
print("Prompt: ", prompt)

print()
loss = gpt2_small(prompt, return_type="loss")
print("Model loss: ", loss)
print("Model loss.item(): ", loss.item())

print()
logits = gpt2_small(prompt, return_type="logits")
print("Model logits: ", logits)
print("Model logits shape: ", logits.shape)

