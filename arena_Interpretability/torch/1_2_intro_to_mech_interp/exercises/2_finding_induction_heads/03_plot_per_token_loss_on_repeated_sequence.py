import torch as t
import eindex
from torch import Tensor
from jaxtyping import Int, Float, Bool

from transformer_lens import (
    ActivationCache,
    FactoredMatrix,
    HookedTransformer,
    HookedTransformerConfig,
    utils,
)
from transformer_lens.hook_points import HookPoint

device = t.device("mps" if t.backends.mps.is_available() else "cuda" if t.cuda.is_available() else "cpu")

model: HookedTransformer = HookedTransformer.from_pretrained("gpt2-small")

print("------------------------")
print("Exercise - Plot per-token loss on repeated sequence")

def generate_repeated_tokens(
    model: HookedTransformer, seq_len: int, batch_size: int = 1
) -> Int[Tensor, "batch_size full_seq_len"]:
    """
    Generates a sequence of repeated random tokens

    Outputs are:
        rep_tokens: [batch_size, 1+2*seq_len]
    """
    t.manual_seed(0)  # for reproducibility
    prefix = (t.ones(batch_size, 1) * model.tokenizer.bos_token_id).long()
    print(f"Prefix tokens: {model.to_str_tokens(prefix)}")
    rep_tokens = t.randint(0, model.cfg.d_vocab, (batch_size, seq_len)).long()
    # Break it intentionally.
    #rep_tokens = t.randint(0, 150000, (batch_size, seq_len)).long()
    print("rep_tokens.shape:", rep_tokens.shape)
    print("rep_tokens:", rep_tokens)
    #print(f"Random tokens: {model.to_str_tokens(rep_tokens)}")
    full_seq = t.cat([prefix, rep_tokens, rep_tokens], dim=1)
    #print(f"Full sequence tokens: {model.to_str_tokens(full_seq)}")
    return full_seq


def run_and_cache_model_repeated_tokens(
    model: HookedTransformer, seq_len: int, batch_size: int = 1
) -> tuple[Tensor, Tensor, ActivationCache]:
    """
    Generates a sequence of repeated random tokens, and runs the model on it, returning (tokens,
    logits, cache). This function should use the `generate_repeated_tokens` function above.

    Outputs are:
        rep_tokens: [batch_size, 1+2*seq_len]
        rep_logits: [batch_size, 1+2*seq_len, d_vocab]
        rep_cache: The cache of the model run on rep_tokens
    """
    rep_tokens = generate_repeated_tokens(model, seq_len, batch_size)
    rep_logits, rep_cache = model.run_with_cache(rep_tokens)
    return rep_tokens, rep_logits, rep_cache


def get_log_probs(
    logits: Float[Tensor, "batch posn d_vocab"], tokens: Int[Tensor, "batch posn"]
) -> Float[Tensor, "batch posn-1"]:
    logprobs = logits.log_softmax(dim=-1)
    # We want to get logprobs[b, s, tokens[b, s+1]], in eindex syntax this looks like:
    # shift tokens to get the "next token"
    next_tokens = tokens[:, 1:]              # shape (batch, posn-1)

    # drop last position from logits to match
    logprobs = logprobs[:, :-1, :]           # shape (batch, posn-1, vocab)

    # gather the correct logprobs
    index = next_tokens.to(logprobs.device).unsqueeze(-1)
    correct_logprobs = logprobs.gather(
        dim=-1,
        index=index
    ).squeeze(-1)
    return correct_logprobs


seq_len = 50
batch_size = 1
(rep_tokens, rep_logits, rep_cache) = run_and_cache_model_repeated_tokens(model, seq_len, batch_size)
rep_cache.remove_batch_dim()
rep_str = model.to_str_tokens(rep_tokens)
model.reset_hooks()
log_probs = get_log_probs(rep_logits, rep_tokens).squeeze()

print(f"Performance on the first half: {log_probs[:seq_len].mean():.3f}")
print(f"Performance on the second half: {log_probs[seq_len:].mean():.3f}")

#plot_loss_difference(log_probs, rep_str, seq_len)
import matplotlib.pyplot as plt
import numpy as np

x = np.arange(len(log_probs))

plt.figure(figsize=(10, 4))

# line plot
plt.plot(x, log_probs.detach().cpu().numpy())

# shaded regions
plt.axvspan(0, seq_len, alpha=0.2)              # first half
plt.axvspan(seq_len, len(log_probs), alpha=0.2) # second half

# labels
plt.xlabel("Sequence position")
plt.ylabel("Log prob")
plt.title(f"Per token log prob on correct token, for sequence of length {seq_len}*2 (repeated twice)")

plt.show()
plt.savefig("03_log_probs_plot.png", dpi=150, bbox_inches="tight")
print("Saved plot to log_probs_plot.png")