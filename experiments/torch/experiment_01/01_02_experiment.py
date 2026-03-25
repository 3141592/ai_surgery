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

logits_a = model(prompt_a, prepend_bos=True)
logits_b = model(prompt_b, prepend_bos=True)

# Final-position logits = prediction for the next token
final_logits_a = logits_a[0, -1]
final_logits_b = logits_b[0, -1]

# Target tokens
print()
token_a = " bed"
token_b = " bed"
prompt_a_token = model.to_single_token(token_a)
prompt_b_token = model.to_single_token(token_b)
print("Prompt A token id:", prompt_a_token)
print("Prompt B token id:", prompt_b_token)

# Rank helper
def token_rank(final_logits, token_id):
    sorted_ids = torch.argsort(final_logits, descending=True)
    return (sorted_ids == token_id).nonzero(as_tuple=True)[0].item()

prompt_a_rank = token_rank(final_logits_a, prompt_a_token)
prompt_b_rank = token_rank(final_logits_b, prompt_b_token)

print()
print(f"Rank of {token_a} after Prompt A:", prompt_a_rank)
print(f"Rank of {token_b} after Prompt B:", prompt_b_rank)

# Optional: top 10 predictions
topk_a = torch.topk(final_logits_a, 10)
topk_b = torch.topk(final_logits_b, 10)

print("\nTop 10 after Prompt A:")
for logit, tok in zip(topk_a.values, topk_a.indices):
    print(float(logit.detach()), repr(model.to_string(tok)))

print("\nTop 10 after Prompt B:")
for logit, tok in zip(topk_b.values, topk_b.indices):
    print(float(logit.detach()), repr(model.to_string(tok)))

log_experiment(
    log_path="experiments/experiment_log.jsonl",
    script=__file__,
    question="How will gpt2-small complete next token selection on non-quesrtion prompts?",
    model="gpt2-small",
    prompt_a=prompt_a,
    prompt_b=prompt_b,
    comparison_type="prompt differences",
    metric_summary={
        "layers_checked": "NA",
    },
    result_summary="TBD.",
    notes="Try different prompts.",
    artifacts=["NA"],
)