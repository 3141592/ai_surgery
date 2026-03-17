# 1_load_hooked_transformer.py
print("------------------------")
print("Load and run a HookedTransformer model pythia-70m")
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

pythia_70m: HookedTransformer = HookedTransformer.from_pretrained("pythia-70m", device=device)

model_description_text = """## Loading Models

HookedTransformer comes loaded with >40 open source GPT-style models. You can load any of them in with `HookedTransformer.from_pretrained(MODEL_NAME)`. Each model is loaded into the consistent HookedTransformer architecture, designed to be clean, consistent and interpretability-friendly.

For this demo notebook we'll look at pythia-70m, an 70M parameter model. To try the model out, let's find the loss on this paragraph!"""

repeated_text = "This This This This This This This This This This "

loss = pythia_70m(model_description_text, return_type="loss")
print("Model loss:", loss)

print("------------------------")
print("Find number of layers")
print("Number of layers:", pythia_70m.cfg.n_layers)

print("------------------------")
print("Use the model's tokenizer to convert text to tokens, and vice versa")
tokens = pythia_70m.to_tokens(model_description_text)
print("Tokens:", tokens)
print("Detokenized:", pythia_70m.to_string(tokens))

print("------------------------")
more_tokens = pythia_70m.to_tokens(repeated_text)
print("More tokens:", more_tokens)

print("Convert IDs to tokens")
for i in more_tokens[0].tolist():
    print(i, repr(pythia_70m.tokenizer.decode([i])))
    
print("Get the ID 0 token")
print(repr(pythia_70m.tokenizer.decode([0])))

print("------------------------")
print("Model summary:")
print(pythia_70m)

print("------------------------")
print("Know how to cache activations, and to access activations from the cache")
logits, cache = pythia_70m.run_with_cache(model_description_text)
cache_interesting_names = []
for key in cache:
    print(key, cache[key].shape)

print("------------------------")
print("Know how to cache activations, and to access activations from the cache")
pat = cache["blocks.5.attn.hook_pattern"]  # [1, 12, 110, 110]
row_sums = pat[0, 0, -1].sum()              # head 0, last query position, sum over keys
print("Sum of attention pattern for head 0, last query position:", row_sums)

print("------------------------")
pat = cache["blocks.5.attn.hook_pattern"]   # [1, 12, 110, 110]
print("Attention pattern:", pat)

head = 0
qpos = -1  # last token
attn = pat[0, head, qpos]                    # [110] over key positions

vals, idx = attn.topk(10)
print("------------------------")
print("Top 10 attention values for head", head, "query position", qpos)
print(list(zip(idx.tolist(), vals.tolist())))

print("------------------------")
print("Top 10 attention string tokens")
tokens = pythia_70m.to_str_tokens(model_description_text)

top_positions = [109, 73, 94, 72, 22, 47, 0, 108, 1, 4]
for i in top_positions:
    print(i, repr(tokens[i]))

print("------------------------")
print("pythia_70m.cfg.n_layers):", pythia_70m.cfg.n_layers)
print("pythia_70m.cfg.n_heads):", pythia_70m.cfg.n_heads)
print("pythia_70m.cfg.d_model):", pythia_70m.cfg.d_model)



