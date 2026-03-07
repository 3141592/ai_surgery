# Interpretability

- This code is my study of Transformer Interpretability based on the ARENA Mechanistic Interpretability Tutorials.
- https://learn.arena.education/chapter0_fundamentals/01_ray_tracing/
- https://arena-chapter1-transformer-interp.streamlit.app/

## Chapter 0: Fundamentals

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
