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

prompt_a = "Q: What is the capital of France?\nA:"
prompt_b = "Q: What is the capital of Germany?\nA:"

logits_a = model(prompt_a, prepend_bos=True)
logits_b = model(prompt_b, prepend_bos=True)

# Final-position logits = prediction for the next token
final_logits_a = logits_a[0, -1]
final_logits_b = logits_b[0, -1]

print()
print("final_logits_a shape:", final_logits_a.shape)
print("Final logits for prompt A:", final_logits_a)
print("final_logits_b shape:", final_logits_b.shape)
print("Final logits for prompt B:", final_logits_b)

# Target tokens
print()
paris_token = model.to_single_token(" Paris")
berlin_token = model.to_single_token(" Berlin")
print("Paris token id:", paris_token)
print("Berlin token id:", berlin_token)

# Rank helper
def token_rank(final_logits, token_id):
    sorted_ids = torch.argsort(final_logits, descending=True)
    return (sorted_ids == token_id).nonzero(as_tuple=True)[0].item()

paris_rank_a = token_rank(final_logits_a, paris_token)
berlin_rank_b = token_rank(final_logits_b, berlin_token)

print()
print("Rank of ' Paris' after France prompt:", paris_rank_a)
print("Rank of ' Berlin' after Germany prompt:", berlin_rank_b)

# Optional: top 10 predictions
topk_a = torch.topk(final_logits_a, 10)
topk_b = torch.topk(final_logits_b, 10)

print("\nTop 10 after France prompt:")
for logit, tok in zip(topk_a.values, topk_a.indices):
    print(float(logit.detach()), repr(model.to_string(tok)))

print("\nTop 10 after Germany prompt:")
for logit, tok in zip(topk_b.values, topk_b.indices):
    print(float(logit.detach()), repr(model.to_string(tok)))

log_experiment(
    log_path="experiments/experiment_log.jsonl",
    script=__file__,
    question="Choosing a better behaved prompt.",
    model="gpt2-small",
    prompt_a=prompt_a,
    prompt_b=prompt_b,
    comparison_type="prompt differences",
    metric_summary={
        "layers_checked": "NA",
    },
    result_summary="TBD.",
    notes="Replace the prompts with hopefully stronger ones.",
    artifacts=["NA"],
)