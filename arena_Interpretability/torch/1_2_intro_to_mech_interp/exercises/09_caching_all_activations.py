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
print("Caching all Activations")
print()

device = t.device("mps" if t.backends.mps.is_available() else "cuda" if t.cuda.is_available() else "cpu")

gpt2_small: HookedTransformer = HookedTransformer.from_pretrained("gpt2-small")

gpt2_text = "Natural language processing tasks, such as question answering, machine translation, reading comprehension, and summarization, are typically approached with supervised learning on task-specific datasets."

gpt_tokens = gpt2_small.to_tokens(gpt2_text)
print()
print("Tokens shape:", gpt_tokens.shape)
print("Tokens:", gpt_tokens)

print()
print("String for token 11: ", gpt2_small.to_string(11))
print("String for token 220: ", gpt2_small.to_string(220))

gpt2_logits, gpt2_cache = gpt2_small.run_with_cache(gpt_tokens, remove_batch_dim=True)
print()
print(type(gpt2_logits), type(gpt2_cache))

attn_patterns_from_shorthand = gpt2_cache["pattern", 0]
attn_patterns_from_full_name = gpt2_cache["blocks.0.attn.hook_pattern"]
print()
print("Attention patterns from shorthand key:", attn_patterns_from_shorthand.shape)
print()
print("Attention patterns from full name key:", attn_patterns_from_full_name.shape)

t.testing.assert_close(attn_patterns_from_shorthand, attn_patterns_from_full_name)

act_name = utils.get_act_name("pattern", 0)
print()
print("Activation name for pattern in block 0:", act_name)

