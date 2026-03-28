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

prompt_a = "The cat sat on the"
prompt_b = "The dog sat on the"
print("Prompt A:", repr(prompt_a))
print("Prompt B:", repr(prompt_b))

logits_a, cache_a = model.run_with_cache(prompt_a, prepend_bos=False)
logits_b, cache_b = model.run_with_cache(prompt_b, prepend_bos=False)

log_experiment(
    log_path="experiments/experiment_log.jsonl",
    script=__file__,
    question="What do values captured by tflen in the last layer give us?",
    model="gpt2-small",
    prompt_a=prompt_a,
    prompt_b=prompt_b,
    comparison_type="Last layer values",
    metric_summary={
        "layers_checked": [10],
    },
    result_summary="TBD.",
    notes="Try different prompts.",
    artifacts=["NA"],
)