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

print()
print("prompt_a tokens:", model.to_str_tokens(prompt_a))
print("prompt_b tokens:", model.to_str_tokens(prompt_b))

logits_a, cache_a = model.run_with_cache(prompt_a, prepend_bos=True)
logits_b, cache_b = model.run_with_cache(prompt_b, prepend_bos=True)

logits_c = model(model.to_tokens(prompt_a))
logits_d = model(model.to_tokens(prompt_b))

print()
print("logits_a shape:", logits_a.shape)
print("logits_b shape:", logits_b.shape)
print("logits_c shape:", logits_c.shape)
print("logits_d shape:", logits_d.shape)

print()
print("Measure \"bed\" for both prompts:")
bed_id = model.to_single_token(" bed")

print()
print("bed_id: ", bed_id)

last_a = logits_a[0, -1, :]
last_b = logits_b[0, -1, :]

print()
print("logit A for ' bed':", last_a[bed_id].item())
print("logit B for ' bed':", last_b[bed_id].item())

rank_a = (last_a > last_a[bed_id]).sum().item() + 1
rank_b = (last_b > last_b[bed_id]).sum().item() + 1

print()
print("rank A for ' bed':", rank_a)
print("rank B for ' bed':", rank_b)

topk_a = last_a.topk(10)
topk_b = last_b.topk(10)

print()
print("Top 10 A:")
for val, idx in zip(topk_a.values, topk_a.indices):
    print(val.item(), repr(model.to_string(idx)))

print()
print("\nTop 10 B:")
for val, idx in zip(topk_b.values, topk_b.indices):
    print(val.item(), repr(model.to_string(idx)))

print()
print("Measure \"ground\" for both prompts:")
ground_id = model.to_single_token(" ground")

print()
print("ground_id: ", ground_id)

print()
print("logit A for ' ground':", last_a[ground_id].item())
print("logit B for ' ground':", last_b[ground_id].item())

rank_a = (last_a > last_a[ground_id]).sum().item() + 1
rank_b = (last_b > last_b[ground_id]).sum().item() + 1

print()
print("rank A for ' ground':", rank_a)
print("rank B for ' ground':", rank_b)


print()
print("Measure \"mat\" for both prompts:")
mat_id = model.to_single_token(" mat")

print()
print("mat_id: ", mat_id)

print()
print("logit A for ' mat':", last_a[mat_id].item())
print("logit B for ' mat':", last_b[mat_id].item())

rank_a = (last_a > last_a[mat_id]).sum().item() + 1
rank_b = (last_b > last_b[mat_id]).sum().item() + 1

print()
print("rank A for ' mat':", rank_a)
print("rank B for ' mat':", rank_b)


log_experiment(
    log_path="experiments/experiment_log.jsonl",
    script=__file__,
    question="What is the logit value for ' bed'?",
    model="gpt2-small",
    prompt_a=prompt_a,
    prompt_b=prompt_b,
    comparison_type="Last layer values",
    metric_summary={
        "layers_checked": [10],
    },
    result_summary="TBD.",
    notes="Logits after final token.",
    artifacts=["NA"],
)