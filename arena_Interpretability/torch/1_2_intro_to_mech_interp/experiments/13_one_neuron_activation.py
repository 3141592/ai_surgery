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

print()
layer_0 = neuron_activations_for_cat[:, 0, :]
print("layer_0.shape: ", layer_0.shape)

print()
print("Highest activation in each layer for 'bed':\n")
n_layers = gpt2_small.cfg.n_layers
seq_len = len(gpt2_str_tokens)

print()
activations_shape = neuron_activations_for_cat[6, layer, :].shape
print("activations_shape: ", activations_shape)

print()
activations = torch.empty((n_layers, *activations_shape))
print("activations_shape: ", activations_shape)

print()
for layer in range(n_layers):
    layer_activation = neuron_activations_for_cat[6, layer, :].cpu().detach()
    print(f"    Layer {layer} activation shape: {layer_activation.shape}")
    activations[layer] = layer_activation

print()
print("activations[0]:\n", activations[0])

print()
print("activations[0][0]:\n", activations[0][0])

print()
print("Get max activations:")
max_values = torch.empty((n_layers, 1, 1))
print("max_values.shape: ", max_values.shape)
print("Max Activations per Layer for 'bed':")
for layer in range(n_layers):
    max_value, max_index = torch.max(activations, 0)
    print(f"    Layer: {layer}, Neuron: {max_index[layer]}, Value: {max_value[layer]}")
