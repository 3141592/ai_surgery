import torch
from transformer_lens import HookedTransformer

model = HookedTransformer.from_pretrained("gpt2-small")
prompt = "The capital of France is"
tokens = model.to_tokens(prompt)

# Run model and capture cache
logits, cache = model.run_with_cache(tokens)

# Token id for " now" (note leading space)
now_id = model.to_single_token(" now")

# Unembedding weights
W_U = model.W_U  # [d_model, vocab]

# Get attention result vectors from block 0 at final position
# shape: [batch, pos, head, d_model]
res = cache["blocks.0.attn.hook_result"]

head_resid = res[0, -1]  # [n_heads, d_model]

# Contribution of each head to the " now" logit
contrib = head_resid @ W_U[:, now_id]

vals, idx = torch.sort(contrib, descending=True)
for v, h in zip(vals.tolist(), idx.tolist()):
    print(f"head {h:2d}: {v: .3f}")

print("\nhead 9 contribution to ' now':", float(contrib[9]))

