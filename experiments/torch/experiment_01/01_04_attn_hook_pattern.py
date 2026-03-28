# Developing a better understanding in order to choose a 2-3 week  project.
import torch

print("------------------------")
print("Load and run a HookedTransformer model")
import torch as t
from ai_shared_utilities import log_experiment

from transformer_lens import (
    ActivationCache,
    FactoredMatrix,
    HookedTransformer,
    HookedTransformerConfig,
    utils,
)
from transformer_lens.hook_points import HookPoint

device = t.device("mps" if t.backends.mps.is_available() else "cuda" if t.cuda.is_available() else "cpu")
model: HookedTransformer = HookedTransformer.from_pretrained("gpt2-small", device=device)

prompt_a = "The cat sat on the bed"
prompt_b = "The dog sat on the bed"
print("Prompt A:", repr(prompt_a))
print("Prompt B:", repr(prompt_b))

logits_a, cache_a = model.run_with_cache(prompt_a, prepend_bos=False)
logits_b, cache_b = model.run_with_cache(prompt_b, prepend_bos=False)

attn_pattern_a = cache_a.cache_dict['blocks.11.attn.hook_pattern']
attn_pattern_b = cache_b.cache_dict['blocks.11.attn.hook_pattern']

print()
print("Attention pattern for prompt A (shape):", attn_pattern_a.shape)
print("Attention pattern for prompt B (shape):", attn_pattern_b.shape)
#print("Attention pattern for prompt A (values):", attn_pattern_a)
#print("Attention pattern for prompt B (values):", attn_pattern_b)

print()
print("Language tokens for prompt A and hook_pattern:\n", model.to_str_tokens(prompt_a))
print(attn_pattern_a[0, 11, :, :])
print("Language tokens for prompt B and hook_pattern:\n", model.to_str_tokens(prompt_b))
print(attn_pattern_b[0, 11, :, :])


log_experiment(
    log_path="experiments/experiment_log.jsonl",
    script=__file__,
    question="What is in the block.11 attn.hook_pattern tensor? Add printing the language tokens?",
    model="gpt2-small",
    prompt_a=prompt_a,
    prompt_b=prompt_b,
    comparison_type="Last layer values",
    metric_summary={
        "layers_checked": [10],
    },
    result_summary="TBD.",
    notes="attn.hook_pattern.",
    artifacts=["NA"],
)