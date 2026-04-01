import torch as t
import math
import circuitsvis as cv
import matplotlib.pyplot as plt
from torch import Tensor
from circuitsvis.attention import attention_heads
import einops

from transformer_lens import (
    ActivationCache,
    FactoredMatrix,
    HookedTransformer,
    HookedTransformerConfig,
    utils,
)
from transformer_lens.hook_points import HookPoint

print("------------------------")
print("Visualizing Attention Heads")

device = t.device("mps" if t.backends.mps.is_available() else "cuda" if t.cuda.is_available() else "cpu")

gpt2_small: HookedTransformer = HookedTransformer.from_pretrained("gpt2-small")

#gpt2_text = "Natural language processing tasks, such as question answering, machine translation, reading comprehension, and summarization, are typically approached with supervised learning on task-specific datasets."
gpt2_text = "The cat sat on the bed."

gpt_tokens = gpt2_small.to_tokens(gpt2_text)
gpt2_logits, gpt2_cache = gpt2_small.run_with_cache(gpt_tokens, remove_batch_dim=True)

print()
print("type(gpt2_cache):", type(gpt2_cache))

print() 
attention_pattern = gpt2_cache["pattern", 0]
print("attention_pattern shape:", attention_pattern.shape)

print()
gpt2_str_tokens = gpt2_small.to_str_tokens(gpt_tokens)
print("gpt2_str_tokens:", gpt2_str_tokens)

print()
print("Visualizing Neuron Activations")
import matplotlib.pyplot as plt

# Pick a layer to inspect
layer = 0

# Shape: (seq_pos, neurons)
neuron_activations_for_cat = t.stack([
    gpt2_cache["post", layer] for layer in range(gpt2_small.cfg.n_layers)
], dim=1)

gpt2_text_dog = "The dog sat on the bed."

gpt_tokens_dog = gpt2_small.to_tokens(gpt2_text_dog)
gpt2_logits_dog, gpt2_cache_dog = gpt2_small.run_with_cache(gpt_tokens_dog, remove_batch_dim=True)

neuron_activations_for_dog = t.stack([
    gpt2_cache_dog["post", layer] for layer in range(gpt2_small.cfg.n_layers)
], dim=1)

print()
print("neuron_activations_for_dog.shape: ", neuron_activations_for_dog.shape)

import matplotlib.pyplot as plt
import torch

n_layers = gpt2_small.cfg.n_layers
seq_len = len(gpt2_str_tokens)

fig, axes = plt.subplots(
    n_layers, 1,
    figsize=(max(8, seq_len * 0.8), 2 * n_layers),
    constrained_layout=True
)

if n_layers == 1:
    axes = [axes]

for layer in range(n_layers):
    activations = neuron_activations_for_cat[:, layer, :].cpu().detach()

    vmin = torch.quantile(activations, 0.01).item()
    vmax = torch.quantile(activations, 0.99).item()

    im = axes[layer].imshow(
        activations.T,
        aspect="auto",
        vmin=vmin,
        vmax=vmax,
        interpolation="nearest"
    )

    axes[layer].set_title(f"Layer {layer}")
    axes[layer].set_ylabel("Neuron")

    if layer == n_layers - 1:
        axes[layer].set_xlabel("Sequence Position")
        axes[layer].set_xticks(range(seq_len))
        axes[layer].set_xticklabels(gpt2_str_tokens, rotation=90)
    else:
        axes[layer].set_xticks([])

    fig.colorbar(im, ax=axes[layer], fraction=0.02, pad=0.01)

plt.show()

print()
print("Layer differences")
# difference between the two runs
diff = neuron_activations_for_cat - neuron_activations_for_dog  # same shape

# pick a layer to inspect
layer = 6

layer_diff = diff[:, layer, :].cpu().detach()

vmax = torch.quantile(layer_diff.abs(), 0.99).item()

plt.figure(figsize=(8, 4))
plt.imshow(
    layer_diff.T,
    aspect="auto",
    vmin=-vmax,
    vmax=vmax
)

plt.title(f"Difference (cat - dog) - Layer {layer}")
plt.xlabel("Sequence Position")
plt.ylabel("Neuron")

#plt.xticks(range(len(tokens)), tokens, rotation=90)

plt.colorbar()
plt.tight_layout()
plt.show()

print()
print("Targeted neurons")
# Find neurons most affected overall in this layer
importance = layer_diff.abs().mean(dim=0)  # (neurons,)

topk = torch.topk(importance, k=30).indices

focused = layer_diff[:, topk]  # (seq_pos, 30)

plt.figure(figsize=(8, 4))
plt.imshow(
    focused.T,
    aspect="auto",
    vmin=-vmax,
    vmax=vmax
)

plt.title(f"Top 30 Changing Neurons - Layer {layer}")
plt.xlabel("Sequence Position")
plt.ylabel("Neuron (top diff)")

#plt.xticks(range(len(tokens)), tokens, rotation=90)

plt.colorbar()
plt.tight_layout()
plt.show()

print()
print("Which neurons respond specifically at the token where the change occurs?")
# focus ONLY on the differing token position (index 2)
pos = 2

importance = layer_diff[pos].abs()  # (neurons,)

topk = torch.topk(importance, k=30).indices
focused = layer_diff[:, topk]

plt.figure(figsize=(8, 4))
plt.imshow(
    focused.T,
    aspect="auto",
    vmin=-vmax,
    vmax=vmax
)

plt.title(f"Top Neurons at Position {pos} - Layer {layer}")
plt.xlabel("Sequence Position")
plt.ylabel("Neuron (top @ pos 2)")

#plt.xticks(range(len(tokens)), tokens, rotation=90)

plt.colorbar()
plt.tight_layout()
plt.show()