import torch as torch
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
print("Investigating Neurons for 'bed'")

device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")

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

# Pick a layer to inspect
layer = 0

# Shape: (seq_pos, layer, neurons)
neuron_activations_for_cat = torch.stack([
    gpt2_cache["post", layer] for layer in range(gpt2_small.cfg.n_layers)
], dim=1)

print()
print("neuron_activations_for_cat.shape: ", neuron_activations_for_cat.shape)
n_layers = gpt2_small.cfg.n_layers

print()
activations_shape = neuron_activations_for_cat[:, layer, :].shape
activations = torch.empty((n_layers, *activations_shape))
print("activations.shape: ", activations.shape)

print()
activations_not_bed = torch.empty((n_layers, 3072))
activations_bed = torch.empty((n_layers, 3072))
print("activations_bed.shape: ", activations_bed.shape)
activations_differences = torch.empty((n_layers, 3072))

for layer in range(n_layers):
    for token in range(gpt_tokens.itemsize - 1):
        if token == 6:
            continue
        layer_activation = neuron_activations_for_cat[token, layer, :].cpu().detach()
        activations_not_bed[layer] = layer_activation

for layer in range(n_layers):
    bed_activation = neuron_activations_for_cat[6, layer, :].cpu().detach()
    activations_bed[layer] = bed_activation

print()
print("Get activation differences:")
activations_differences = activations_bed - activations_not_bed
    
print("activations_differences.shape: ", activations_differences.shape)

print()
print("Show differences between bed and ~bed per layer:")
for layer in range(n_layers):
    difference = torch.sum(activations_differences[layer], -1)
    topk = torch.topk(activations_differences[layer], k=5)
    print(f"Layer: {layer}, Difference: {topk}")


