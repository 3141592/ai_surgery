import torch as torch
import math
import matplotlib.pyplot as plt
from torch import Tensor
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

gpt2_text = "The dog sat on the bed."

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
neuron_activations_for_bed = torch.stack([
    gpt2_cache["post", layer] for layer in range(gpt2_small.cfg.n_layers)
], dim=1)

print()
print("neuron_activations_for_bed.shape: ", neuron_activations_for_bed.shape)
n_layers = gpt2_small.cfg.n_layers

print()
activations_shape = neuron_activations_for_bed[:, layer, :].shape
activations = torch.empty((n_layers, *activations_shape))
print("activations.shape: ", activations.shape)

print()
activations_not_bed = torch.empty((n_layers, 3072))
activations_bed = torch.empty((n_layers, 3072))
print("activations_bed.shape: ", activations_bed.shape)
activations_differences = torch.empty((n_layers, 3072))

bed_token = 6

bed_token = 6

for layer in range(n_layers):
    bed_activation = neuron_activations_for_bed[bed_token, layer, :].cpu().detach()

    non_bed_activations = []
    for token in range(neuron_activations_for_bed.shape[0]):
        if token in [0, bed_token]:   # skip BOS and " bed"
            continue
        non_bed_activations.append(
            neuron_activations_for_bed[token, layer, :].cpu().detach()
        )

    mean_not_bed = torch.stack(non_bed_activations).mean(dim=0)
    diff = bed_activation - mean_not_bed

    topk = torch.topk(diff, k=5)
    print(f"Layer {layer}: {topk}")

for layer in range(n_layers):
    bed_activation = neuron_activations_for_bed[6, layer, :].cpu().detach()
    activations_bed[layer] = bed_activation

# Now let's compare to a sentence using grounds instead of bed
print("------------------------")
print("Investigating Neurons for 'ground'")
gpt2_text = "The dog sat on the ground."

gpt_tokens = gpt2_small.to_tokens(gpt2_text)
gpt2_logits, gpt2_cache = gpt2_small.run_with_cache(gpt_tokens, remove_batch_dim=True)

neuron_activations_for_ground = torch.stack([
    gpt2_cache["post", layer] for layer in range(gpt2_small.cfg.n_layers)
], dim=1)

print()
activations_shape = neuron_activations_for_ground[:, layer, :].shape
activations = torch.empty((n_layers, *activations_shape))
print("activations.shape: ", activations.shape)

print()
activations_not_ground = torch.empty((n_layers, 3072))
activations_ground = torch.empty((n_layers, 3072))
activations_differences = torch.empty((n_layers, 3072))

ground_token = 6

for layer in range(n_layers):
    ground_activation = neuron_activations_for_ground[ground_token, layer, :].cpu().detach()

    non_ground_activations = []
    for token in range(neuron_activations_for_ground.shape[0]):
        if token in [0, ground_token]:   # skip BOS and " ground"
            continue
        non_ground_activations.append(
            neuron_activations_for_ground[token, layer, :].cpu().detach()
        )

    mean_not_ground = torch.stack(non_ground_activations).mean(dim=0)
    diff = ground_activation - mean_not_ground 

    topk = torch.topk(diff, k=5)
    print(f"Layer {layer}: {topk}")

for layer in range(n_layers):
    ground_activation = neuron_activations_for_ground[6, layer, :].cpu().detach()
    activations_ground[layer] = ground_activation




