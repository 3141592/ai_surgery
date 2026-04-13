ABSTRACT: 

Large Language Models are freqently updated with new training data. 
An important safety practice is auditing changes that were caused by the additional training data.

This work demonstrates "model diffing" using a simple 2-layer Transformer architecture trained on controlled data.
After additional training, internal representations (activations) and output distributions (tokens rankings) are compared between the initially trained  and updated models.

This will provide a simple, reproducible method of comparing a simple 2-layer model which will show quantifiable differences between two versions of a Transformer-based model.

The inspiration for this paper is "model diffing" as described in the paper CROSS-ARCHITECTURE MODEL DIFFING WITH
CROSSCODERS: UNSUPERVISED DISCOVERY OF DIFFERENCES BETWEEN LLMS (https://arxiv.org/pdf/2602.11729).

