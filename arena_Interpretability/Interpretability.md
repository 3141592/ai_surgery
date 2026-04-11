# Interpretability

- This code is my study of Transformer Interpretability based on the ARENA Mechanistic Interpretability Tutorials.
- https://learn.arena.education/chapter0_fundamentals/01_ray_tracing/
- https://arena-chapter1-transformer-interp.streamlit.app/

## Chapter 0: Fundamentals

### 04-10-2026
#### [1.2] Intro to Mech Interp

- Finding induction heads
  - What are induction heads?
  - DONE Checking for the induction capability
    - DONE Exercise - plot per-token loss on repeadted seqeunce
  - NEXT Looking for Induction Attention Patterns

### 04-09-2026
#### [1.2] Intro to Mech Interp

- Finding induction heads
  - What are induction heads?
  - WIP Checking for the induction capability
    - WIP Exercise - plot per-token loss on repeadted seqeunce

### 04-01-2026
#### [1.2] Intro to Mech Interp

- Finding induction heads
  - Learning Objectives
    - Understand what induction heads are, and the algorithm they are implementing
    - Inspect activation patterns to identify basic attention head patterns, and write your own functions to detect attention heads for you
    - Identify induction heads by looking at the attention patterns produced from a repeating random sequence
  - Exercise - write your own detectors

### 03-31-2026
#### [1.2] Intro to Mech Interp

- TransformerLens: Introduction
  - Visualizing Neuron Activations
    - I need to go back and understand the data pulled for the neuron activations.
      - neuron_activations_for_dog.shape = [8, 12, 3072] == [tokens, layers, neurons]
      - found activations for the 'bed' token in each layer
  - NEXT Finding induction heads


### 03-26-2026
#### [1.2] Intro to Mech Interp

- TransformerLens: Introduction
  - Visualising Attention Heads
    - Create images for all heads in Layer 0
    - Heads are not static roles—they are reused and repurposed across layers.
  - Created attention head heatmaps for:
    - The cat sat on the bed
    - The dog sat on the bed
  - There were no noticeable differences.
  - WIP Visualizing Neuron Activations
    - Using ChatGPT changed the AGENDA code to matplotlib.
    - Generated charts showing differences between the 'cat' and 'dog' prompts.
    - I need to go back and understand the data pulled for the neuron activations.


### 03-19-2026
#### [1.2] Intro to Mech Interp

- TransformerLens: Introduction
  - WIP Visualising Attention Heads
    - Create images for all heads in Layer 0
    - Heads are not static roles—they are reused and repurposed across layers.

### 03-18-2026
#### [1.2] Intro to Mech Interp

- TransformerLens: Introduction
  - WIP Visualising Attention Heads

### 03-17-2026
#### [1.2] Intro to Mech Interp

- TransformerLens: Introduction
  - Caching all Activations
    - Excercise - verify activations
      - In QK^T:
        - Each row represents the attention score of a single token for every other token
  - NEXT Visualising Attention Heads

### 03-16-2026
#### [1.2] Intro to Mech Interp

- TransformerLens: Introduction
  - Running your model
    - return_type = "loss"
    - return_type = "logits"
  - Experiment with utils.test_prompt(prompt, answr, model)
    - This returns the next 10 predicted tokens.
    - It does include some indication of knowledge, but is driven more by simply next token.
  - Tokenization
    - Exercise - how many tokens does your model guess correctly?
  - WIP Caching all Activations
    - TBD Excercise - verify activations


### 03-07-2026
#### [1.2] Intro to Mech Interp

- TransformerLens: Introduction
  - Loading and Running Models
  - Exercise - inspect your model
    - Number of layers
    - Number of heads per layer
    - Maximum context window
  - Running your model
    - return_type = "loss"
    - return_type = "logits"
  - Transformer architecture
    - The weight matrices W_K, W_Q, W_V, W_O
      - The activations all have shape [batch, position, head_index, d_head].
      - W_K, W_Q, W_V have shape [head_index, d_model, d_head] and W_O has shape [head_index, d_head, d_model]

### 03-06-2026
#### [1.2] Intro to Mech Interp

- TransformerLens Introduction
  - Use `circuitsvis` to visualise attention heads
    - I had to rely on the Exploratory Analysis Demo for this, but it worked.
  - Start back at the Running your model section.


### 03-05-2026
#### [1.2] Intro to Mech Interp

- Learned about dumping information from heads.
- Learned that model.generate(prompt) outputs a prediction.

### 03-03-2026
#### [1.2] Intro to Mech Interp

- Intro to Mechanistic Interpretability video
- TransformerLens Introduction
  - Load and run a `HookedTransformer` model
  - Understand the basic architecture of these models
  - Use the model's tokenizer to convert text to tokens, and vice versa
  - Know how to cache activations, and to access activations from the cache
  - Investigated attention heads in the last block
  - TODO Use `circuitsvis` to visualise attention heads

### 03-02-2026
#### [1.2] Intro to Mech Interp

**Note**: It turned out it is to difficult (or time consuming) to translate all
of the ARENA notebooks to python files, so I am going to work through the
tutorial a little more loosely.

- Gain a better understanding of tf.tensor [:, p, :]
- Intro to Mechanistic Interpretability video
  - Stopped at 14:14 

### 03-01-2026
#### [0.1] - Ray Tracing
- Clone repo and move into ai_surgery repo.
- Installed
  - pip install jaxtyping einops
  - pip install plotly
  - pip install ipython
  - pip install ipywidgets
- Completed setup
- Completed Exercise make_rays_1d.py
  - Took me about an hour.
