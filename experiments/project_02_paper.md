# ABSTRACT

Large Language Models are freqently updated with new training data. 
An important safety practice is auditing changes that were caused by the additional training data.

This work demonstrates "model diffing" using a simple 2-layer Transformer architecture trained on controlled data.
A small model with controlled data avoids a large number of model changes to investigate.

After additional training, internal representations (activations) and output distributions (topk activations and token probabilities) are compared between the initially trained and updated models.


# METHODOLOGY

We create a 2-layer Transformer architecture using pytorch. The model is initially trained on simple sequence data ("AAA BBB").

The initial model is saved to disk, then is trained on a second simple sequence data ("123 256").

Following the second training activation values and token probablities from the output layer are captured for both models, then compared.

I am not yet writing the final methodology. I am determining the methodology.”