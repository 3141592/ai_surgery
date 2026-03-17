import torch as t
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
print("Exercise: How many tokens does your model guess correctly?")
print()

device = t.device("mps" if t.backends.mps.is_available() else "cuda" if t.cuda.is_available() else "cpu")

gpt2_small: HookedTransformer = HookedTransformer.from_pretrained("gpt2-small")

model_description_text = "The capital of France is"

logits: Tensor = gpt2_small(model_description_text, return_type="logits")
print(f"Logits shape: {logits.shape}")
prediction = logits.argmax(dim=-1).squeeze()[:-1]

# YOUR CODE HERE - get the model's prediction on the text
print()
print("Model's prediction:", prediction)

print()
print("Prompt tokens:", gpt2_small.to_str_tokens(model_description_text))
print("Model's predicted tokens:", gpt2_small.to_str_tokens(prediction))

true_tokens = gpt2_small.to_tokens(model_description_text).squeeze()[1:]
is_correct = prediction == true_tokens

print()
print("to_tokens(model_description_text) ", gpt2_small.to_tokens(model_description_text))

print()
print("to_tokens(model_description_text).squeeze() ", gpt2_small.to_tokens(model_description_text).squeeze())
print(gpt2_small.to_tokens(model_description_text) == gpt2_small.to_tokens(model_description_text).squeeze())

print()
print("to_tokens(model_description_text).squeeze()[1:] ", gpt2_small.to_tokens(model_description_text).squeeze()[1:])

print()
print("to_tokens(model_description_text)[1:] ", gpt2_small.to_tokens(model_description_text)[1:])

print()
print(f"Model accuracy: {is_correct.sum()}/{len(true_tokens)}")
print()
print(f"Correct tokens: {gpt2_small.to_str_tokens(prediction[is_correct])}")
