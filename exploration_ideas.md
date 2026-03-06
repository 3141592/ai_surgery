# Exploration Ideas

## Mechanistic Interpretability

Understand how specific computations are implemented inside a model.

Instead of treating the model as a black box, researchers identify things like:

- an attention head that detects matching brackets
- a neuron that activates on negative sentiment
- a circuit that performs induction (copying earlier tokens)

## Sparse Autoencoders (SAEs)

Problem:

Hidden layers contain thousands of overlapping features.
Individual neurons do not represent clean concepts.

Solution:

Train a sparse autoencoder on the activations.

The autoencoder learns a new basis of features where each feature is:

- sparse
- interpretable
- closer to a concept

Researchers have discovered features like:

- “Python code”
- “quotation marks”
- “geographic locations”
- “first-person narration”

Projects working on this:

- Anthropic’s dictionary learning
- SAELens
- Neel Nanda’s interpretability work

## Model behavior experiments (the “biology style” approach)

This is the closest thing to neuroscience.

Instead of analyzing weights directly, researchers run controlled experiments:

- ablate heads
- remove neurons
- modify activations
- observe behavioral changes

Examples:

Remove certain heads → model loses ability to perform:

- long-range copying
- certain reasoning steps
- translation behaviors

This mirrors neuroscience experiments where scientists:

- lesion brain regions
- observe behavior changes
- Many interesting results here have come from curious individuals running systematic experiments.

## Open Source Options

Careful replication studies

- replicate
- document
- clarify

## Small systematic experiments

Example questions:

- What happens if you zero out specific MLP neurons?
- Which heads encode punctuation structure?
- Do induction heads exist in different models?

## Tools and educational resources

- reproducible notebooks
- experiment pipelines
- visualization tools

