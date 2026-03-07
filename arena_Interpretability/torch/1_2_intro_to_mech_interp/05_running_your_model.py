# 1_load_hooked_transformer.py
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

print()
prompt = "The capital of France is "
print("Prompt: ", prompt)

answer = gpt2_small.generate(prompt, max_new_tokens=5)
print()
print("Model answer: ", answer)

prompt = "The capital of France is "
tokens = gpt2_small.to_tokens(prompt, prepend_bos=False)
print("Prompt token ids:", tokens.tolist()[0])
print("Prompt tokens:", gpt2_small.to_str_tokens(tokens))

out = gpt2_small.generate(
    prompt,
    max_new_tokens=5,
    do_sample=False,
    return_type="tokens",
    prepend_bos=False,
)

print("All token ids:", out.tolist()[0])
print("All tokens:", gpt2_small.to_str_tokens(out))
print("Decoded:", gpt2_small.to_string(out))

loss = gpt2_small(prompt, return_type="loss")
print("Model loss:", loss)

import torch
print(torch.__version__)
print(torch.cuda.is_available())