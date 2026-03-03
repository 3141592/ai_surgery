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

prompt = "What is the capital of France?"

print("------------------------")
print("Load model")
tokens = gpt2_small.to_str_tokens(prompt, prepend_bos=False)
logits, cache = gpt2_small.run_with_cache(prompt, prepend_bos=False)

print("cache[blocks.11.attn.hook_pattern].shape:", cache["blocks.11.attn.hook_pattern"].shape)

print("------------------------")
print("Scan for heads with peaky attention patterns")
pat = cache["blocks.11.attn.hook_pattern"][0]  # [12, 110, 110]

entropy = -(pat * pat.clamp(min=1e-9).log()).sum(dim=-1)  # [12, 110]
print(entropy[:, -1])  # entropy per head at last token

print("------------------------")
print("Print all heads in blocks.11.attn.hook_pattern")
pattern = cache["blocks.11.attn.hook_pattern"]
print(pattern.shape)  # [12, 110, 110]

print("------------------------")
print("Print information for each head")
qpos = -1  # last token

for h in range(pattern.shape[1]):               # 12 heads
    attn = pattern[0, h, qpos]                  # [110]

    vals, idx = attn.topk(10)
    print("------------------------")
    print("Top 10 attention values for head", h, "query position", qpos)
    print(list(zip(idx.tolist(), vals.tolist())))

print("------------------------")
