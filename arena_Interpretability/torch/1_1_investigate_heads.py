# 1_1 Investigate heads
print("------------------------")
print("Investigate model heads")
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

model_description_text = """## Loading Models

HookedTransformer comes loaded with >40 open source GPT-style models. You can load any of them in with `HookedTransformer.from_pretrained(MODEL_NAME)`. Each model is loaded into the consistent HookedTransformer architecture, designed to be clean, consistent and interpretability-friendly.

For this demo notebook we'll look at GPT-2 Small, an 80M parameter model. To try the model out, let's find the loss on this paragraph!"""

print("------------------------")
print("Find top attention values for head 0 in the final layer, for the last token")
logits, cache = gpt2_small.run_with_cache(model_description_text)
pat = cache["blocks.11.attn.hook_pattern"]   # [1, 12, 110, 110]

head = 0
qpos = -1  # last token
attn = pat[0, head, qpos]                    # [110] over key positions

vals, idx = attn.topk(10)
print("------------------------")
print("Top 10 attention values for head", head, "query position", qpos)
print(list(zip(idx.tolist(), vals.tolist())))

print("------------------------")
print("Top 10 attention string tokens")
tokens = gpt2_small.to_str_tokens(model_description_text)

top_positions = [109, 73, 94, 72, 22, 47, 0, 108, 1, 4]
for i in top_positions:
    print(i, repr(tokens[i]))

print("------------------------")
print("Scan for heads with peaky attention patterns")
pat = cache["blocks.11.attn.hook_pattern"][0]  # [12, 110, 110]
entropy = -(pat * pat.clamp(min=1e-9).log()).sum(dim=-1)  # [12, 110]
print(entropy[:, -1])  # entropy per head at last token

print("------------------------")
print("Head 6 has the lowest entropy, so let's look at its attention pattern")
pat = cache["blocks.11.attn.hook_pattern"]
head = 6
qpos = -1
attn = pat[0, head, qpos]
vals, idx = attn.topk(10)
print(list(zip(idx.tolist(), vals.tolist())))

print("------------------------")
print("Top 10 attention string tokens for head 6")
tokens = gpt2_small.to_str_tokens(model_description_text)

top_positions = [0, 55, 42, 87, 77, 105, 1, 7, 8, 78]
for i in top_positions:
    print(i, repr(tokens[i]))

print("------------------------")
print("Remove BOS and see how that affects head 6 attention")
tokens2 = gpt2_small.to_str_tokens(model_description_text, prepend_bos=False)
logits2, cache2 = gpt2_small.run_with_cache(model_description_text, prepend_bos=False)
pat2 = cache2["blocks.11.attn.hook_pattern"]   # [1, 12, 110, 110]

head = 6
qpos = -1  # last token
attn2 = pat2[0, head, qpos]                    # [110] over key positions

vals, idx = attn.topk(10)
print("------------------------")
print("Top 10 attention values for head ", head, " without BOS, query position", qpos)
print(list(zip(idx.tolist(), vals.tolist())))
top_positions = idx
print("top positions", top_positions)

print("------------------------")
for i in top_positions:
    print(i, repr(tokens2[i]))

print("------------------------")
print("Check if this head is generally a “start-of-sequence anchor” head, look at a few different query positions")
pat = cache2["blocks.11.attn.hook_pattern"]
head = 6

for qpos in [10, 30, 60, -1]:
    attn = pat[0, head, qpos]
    v0 = attn[0].item()
    topv, topi = attn.topk(1)
    print(qpos, "attn_to_pos0=", v0, "top=", (topi.item(), topv.item()))

print("------------------------")
print("Try ablating just this head and see if the next-token distribution changes:")
# zero out head 6's contribution in block 11
def zero_head6(z, hook):
    z[:, :, 6, :] = 0
    return z

logits_ablated = gpt2_small.run_with_hooks(
    model_description_text,
    prepend_bos=False,
    fwd_hooks=[("blocks.11.attn.hook_z", zero_head6)]
)

# Compare top next-token predictions before vs after
topk = 10
print("orig:", logits2[0, -1].topk(topk).indices)
print("ablt:", logits_ablated[0, -1].topk(topk).indices)

print("------------------------")
print("Compute a numeric difference on the final-position logits.")
import torch
import torch.nn.functional as F

# logits2: original from run_with_cache(prepend_bos=False)
# logits_ablated: from run_with_hooks(prepend_bos=False)

delta = (logits_ablated[0, -1] - logits2[0, -1]).abs()
print("max |Δlogit|:", delta.max().item())
print("mean |Δlogit|:", delta.mean().item())

# also check probability shift on the original top token
orig_top = logits2[0, -1].argmax().item()
p_orig = F.softmax(logits2[0, -1], dim=-1)[orig_top].item()
p_ablt = F.softmax(logits_ablated[0, -1], dim=-1)[orig_top].item()
print("orig top token id:", orig_top, "p_orig:", p_orig, "p_ablt:", p_ablt)


print("------------------------")
