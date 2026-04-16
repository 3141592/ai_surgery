# Finding induction heads
print("------------------------")
print("Finding induction heads")
import torch as t
import matplotlib.pyplot as plt

from transformer_lens import (
    ActivationCache,
    FactoredMatrix,
    HookedTransformer,
    HookedTransformerConfig,
    utils,
)
from transformer_lens.hook_points import HookPoint

device = t.device("mps" if t.backends.mps.is_available() else "cuda" if t.cuda.is_available() else "cpu")

print()
print("Define a model using a HookedTransformerConfig object")
cfg = HookedTransformerConfig(
    d_model=768,
    d_head=64,
    n_heads=12,
    n_layers=2,
    n_ctx=2048,
    d_vocab=50278,
    attention_dir="causal",
    attn_only=True,
    tokenizer_name="EleutherAI/gpt-neox-20b",
    seed=398,
    use_attn_result=True,
    normalization_type=None,
    positional_embedding_type="shortformer"
)

print()
print("Load weights from HuggingFace")
from huggingface_hub import hf_hub_download

REPO_ID = "callummcdougall/attn_only_2L_half"
FILENAME = "attn_only_2L_half.pth"
weights_path = hf_hub_download(repo_id=REPO_ID, filename=FILENAME)

def visualize_attention_pattern(attention_pattern):
    print()
    fig, axes = plt.subplots(3, 4, figsize=(16, 10))
    axes = axes.flatten()
    for i in range(cache.model.cfg.n_heads):
        head = attention_pattern[i].cpu()
        print(f"Layer 0 Head {i} Attention Pattern: {head}")
        ax = axes[i]
        im = ax.imshow(head, aspect="auto")
        fig.colorbar(im, ax=ax)
        ax.set_xlabel("Key position")
        ax.set_ylabel("Query position")
        ax.set_title(f"Layer 0 Head {i} Attention Pattern")

    #cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    #fig.colorbar(im, cax=cbar_ax)
    plt.tight_layout()
    plt.show()

print()
print("Create our model and load in the weights:")
model = HookedTransformer(cfg)
pretrained_weights = t.load(weights_path, map_location=device, weights_only=True)
model.load_state_dict(pretrained_weights)

print()
print("Visualize the attention patterns for both layers of the model:")
text = "We think that powerful, significantly superhuman machine intelligence is more likely than not to be created this century. If current machine learning techniques were scaled up to this level, we think they would by default produce systems that are deceptive or manipulative, and that no solid plans are known for how to avoid this."

logits, cache = model.run_with_cache(text, remove_batch_dim=True)

print() 
attention_pattern = cache["pattern", 0]
print("attention_pattern shape:", attention_pattern.shape)

visualize_attention_pattern(attention_pattern)

attention_pattern = cache["pattern", 1]
visualize_attention_pattern(attention_pattern)