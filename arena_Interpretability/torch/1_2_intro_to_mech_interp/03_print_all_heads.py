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
    k = min(10, attn.numel())  # number of tokens

    vals, idx = attn.topk(k)
    top_positions = idx

    print("------------------------")
    print("top positions", top_positions)
    print("Top", k, " attention values for head", h, "query position", qpos)
    print(list(zip(idx.tolist(), vals.tolist())))
    for i in top_positions:
        print(i, repr(tokens[i]))

print("------------------------")
print("Run your prompt normally and inspect the top token.")
logits, cache = gpt2_small.run_with_cache(prompt)
next_logits = logits[0, -1]
vals, idx = next_logits.topk(20)

for v, i in zip(vals.tolist(), idx.tolist()):
    print(f"{v:8.3f}", repr(gpt2_small.to_string(i)))

tokens = gpt2_small.generate(
    prompt,
    max_new_tokens=5
)

print("tokens: ", tokens)
print("------------------------")
print("Force greedy decoding so generation matches the “top token” behavior:")
generated = gpt2_small.generate(
    prompt,
    max_new_tokens=20,
    temperature=0,      # greedy (no sampling)
)
print(generated)

print("------------------------")
