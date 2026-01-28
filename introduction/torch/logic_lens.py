import torch
from transformer_lens import HookedTransformer

model = HookedTransformer.from_pretrained("gpt2-small")
prompt = "The capital of France is"

tokens = model.to_tokens(prompt)

# helper to show top next tokens
def top_next(logits, k=10):
    next_logits = logits[0, -1]  # last position
    topv, topi = torch.topk(next_logits, k)
    return [(model.to_string(i), float(v)) for v, i in zip(topv, topi)]

# BASE
base_logits = model(tokens)
print("BASE top next tokens:")
for tok, val in top_next(base_logits, 10):
    print(f"{tok!r}\t{val:.2f}")

# EDIT: ablate one head in block 0
HEAD = 9
def zero_head(z, hook):
    z[:, :, HEAD, :] = 0
    return z

with model.hooks(fwd_hooks=[("blocks.0.attn.hook_z", zero_head)]):
    edit_logits = model(tokens)

print(f"\nEDIT (zero block0 head{HEAD}) top next tokens:")
for tok, val in top_next(edit_logits, 10):
    print(f"{tok!r}\t{val:.2f}")
