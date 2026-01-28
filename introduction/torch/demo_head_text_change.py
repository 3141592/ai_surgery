# demo_head_text_changes.py
#
# Shows how the generated text changes when you ablate (zero) specific attention heads.
#
# Install:
#   pip install transformer-lens torch
#
# Run:
#   python demo_head_text_changes.py

import torch
from transformer_lens import HookedTransformer

torch.set_grad_enabled(False)

# ---- Config ----
MODEL_NAME = "gpt2-small"      # small, fast, widely available
DEVICE = "cpu"                # change to "cuda" if you want and have it
MAX_NEW_TOKENS = 40           # how much text to generate after the prompt
DO_SAMPLE = True             # False = deterministic (greedy). True = more "fun", less reproducible
TEMPERATURE = 0.8             # used only if DO_SAMPLE=True

# Prompt (use your wife's version; I’ll keep the typo to match your memory)
PROMPT = "Tell me about the capital of France."

# Pick which heads to show (layer, head). You can change this list.
# gpt2-small has 12 layers (0-11) and 12 heads (0-11).
HEADS_TO_TEST = [
    (0, 0),
    (0, 5),
    (3, 7),
    (5, 3),
    (8, 1),
    (11, 9),
]

# ---- Load model ----
model = HookedTransformer.from_pretrained(MODEL_NAME, device=DEVICE)

def generate_with_optional_hooks(prompt: str, fwd_hooks=None) -> str:
    """
    Generates a continuation from prompt, optionally with forward hooks applied.
    Returns the full prompt+completion as a single string.
    """
    input_tokens = model.to_tokens(prompt)

    if fwd_hooks is None:
        out_tokens = model.generate(
            input_tokens,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=DO_SAMPLE,
            temperature=TEMPERATURE,
        )
    else:
        # model.generate doesn't accept hooks directly, so we generate step-by-step with hooks:
        # We do greedy/sampling ourselves by repeatedly running the model with hooks.
        tokens = input_tokens.clone()

        for _ in range(MAX_NEW_TOKENS):
            logits = model.run_with_hooks(tokens, fwd_hooks=fwd_hooks)
            next_logits = logits[:, -1, :]

            if DO_SAMPLE:
                probs = torch.softmax(next_logits / TEMPERATURE, dim=-1)
                next_tok = torch.multinomial(probs, num_samples=1)
            else:
                next_tok = torch.argmax(next_logits, dim=-1, keepdim=True)

            tokens = torch.cat([tokens, next_tok], dim=-1)

        out_tokens = tokens

    return model.to_string(out_tokens[0])

def ablate_head_hook(layer: int, head: int):
    """
    Returns a hook for blocks.{layer}.attn.hook_result that zeros one head.
    hook_result shape: [batch, pos, n_heads, d_head]
    """
    def hook_fn(result, hook):
        result[:, :, head, :] = 0.0
        return result
    return (f"blocks.{layer}.attn.hook_result", hook_fn)

# ---- Baseline ----
print("=" * 90)
print("PROMPT:")
print(PROMPT)
print("=" * 90)

baseline = generate_with_optional_hooks(PROMPT)
print("\nBASELINE COMPLETION:\n")
print(baseline)

# ---- Head ablations ----
print("\n" + "=" * 90)
print("HEAD ABLATIONS (one head zeroed at a time):")
print("=" * 90)

for layer, head in HEADS_TO_TEST:
    hooks = [ablate_head_hook(layer, head)]
    edited = generate_with_optional_hooks(PROMPT, fwd_hooks=hooks)

    # Print just a compact view: first line or two
    print(f"\n--- Ablate (layer={layer}, head={head}) ---")
    print(edited)

print("\nDone.")

