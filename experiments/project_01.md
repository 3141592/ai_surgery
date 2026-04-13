# Experiments

## 04/13/202V

- project 1: How does changing 'cat' to 'dog' in a short prompt change activations?
  - Renamed experimenty to project and removed tensorflow/torch directories.
  - Blindly see what I can discover about 'bed' vs 'ground' for dogs.
  - Computed 'bed' vs all other tokens and 'ground' vs all other, but need to compare 'bed' vs 'ground'.
  - NEXT Create script to compare neuron activations for 'bed' vs 'ground'.t a

## 04/12/202V

- project 1: How does changing 'cat' to 'dog' in a short prompt change activations?
  - Blindly see what I can discover about 'bed' vs 'ground' for dogs.
  - Computed 'bed' vs all other tokens and 'ground' vs all other, but need to compare 'bed' vs 'ground'.
  - NEXT Create script to compare neuron activations for 'bed' vs 'ground'.t a

## 04/01/2026

- project 1: How does changing 'cat' to 'dog' in a short prompt change activations?
  - Next, calculate the deltas for the top 10 ranked tokens.
  - Investigating Neuron Activations
    - WIP Which neurons fire more for ‘bed’ than for other tokens?”
      - Instead of max per layer, compute: bed_activations - average_activations
      - Which neurons increase specifically for "bed"?
  - AFTER Pick a single token (like ' bed') and trace:
    - which heads increased its logit
    - which FFN neurons contributed

## 03/31/2026

- project 1: How does changing 'cat' to 'dog' in a short prompt change activations?
  - Next, calculate the deltas for the top 10 ranked tokens.
  - Investigating Neuron Activations
    - NEXT Which neurons fire more for ‘bed’ than for other tokens?”
      - Instead of max per layer, compute: bed_activations - average_activations
      - Which neurons increase specifically for "bed"?
  - AFTER Pick a single token (like ' bed') and trace:
    - which heads increased its logit
    - which FFN neurons contributed

## 03/28/2026

- project 1: How does changing 'cat' to 'dog' in a short prompt change activations?
  - Got the logits and rank for 'bed' and for dog the higher ranked 'ground'.
  - Next, calculate the deltas for the top 10 ranked tokens.

## 03/24/2026

- Tested prompts for using in a small mechanical interpretability project.
  - experiment_01: Decided on comparing the where in the network replacing 'cat' with 'dog' shift from bed/couch to ground/floor.


# Experiment Logging

This project uses a lightweight JSONL-based logging system to track experiments.

The goal is simple:
- Capture what was tested
- Record what happened
- Avoid losing useful insights

No databases, no dashboards—just structured records.

---

## Overview

Each experiment is logged as a single JSON object appended to:

experiments/experiment_log.jsonl

Each line = one experiment.

This makes it:
- easy to append
- easy to inspect
- easy to load later with Python or pandas

---

## Logging an Experiment

Import the helper:

from ai_shared_utilities import log_experiment

Call it once per experiment run (usually at the end of your script):

log_experiment(
    log_path="experiments/experiment_log.jsonl",
    script=__file__,
    question="How does attention change when replacing France with Germany?",
    model="gpt2-small",
    prompt_a="The capital of France is",
    prompt_b="The capital of Germany is",
    comparison_type="prompt_diff_attention",
    metric_summary={
        "layers_checked": [0, 1, 2],
    },
    result_summary="Attention differs more in later layers for the final token.",
    notes="Initial pass, no baseline yet.",
    artifacts=["artifacts/exp_001/layer2.png"],
)

---

## Required Mindset

This is not about perfection. It is about closing the loop.

Every logged experiment should answer:

- What did I try?
- What changed?
- What did I observe?

---

## Field Descriptions

timestamp:
  Automatically added (UTC)

experiment_id:
  Unique ID (auto-generated unless provided)

script:
  Script that ran the experiment

question:
  The specific question being tested

model:
  Model used (e.g., gpt2-small)

prompt_a / prompt_b:
  Inputs being compared

comparison_type:
  Type of experiment (e.g., prompt_diff_attention)

metric_summary:
  Key structured observations (small, not raw data)

result_summary:
  Plain-English conclusion (required)

notes:
  Follow-ups, uncertainties, next ideas

artifacts:
  Paths to saved plots or outputs

---

## Example Log Entry

{
  "timestamp": "2026-03-24T22:05:00Z",
  "experiment_id": "exp_20260324_001",
  "script": "compare_attention_prompts.py",
  "question": "How does attention change when replacing France with Germany?",
  "model": "gpt2-small",
  "prompt_a": "The capital of France is",
  "prompt_b": "The capital of Germany is",
  "comparison_type": "prompt_diff_attention",
  "metric_summary": {
    "layers_checked": [0, 1, 2],
    "top_changed_heads": ["1.4", "2.7"]
  },
  "result_summary": "Later layers show stronger differences in attention patterns.",
  "notes": "Need baseline with unrelated word substitution.",
  "artifacts": [
    "artifacts/exp_001/layer2_diff.png"
  ]
}

---

## Guidelines

- Log every real experiment (even failed ones)
- Keep entries short and honest
- Always include a result_summary
- Do not log raw tensors or large data
- Save plots/artifacts separately and reference them

---

## What Not to Do

- Do not build a complex tracking system
- Do not over-structure fields prematurely
- Do not skip logging because the result is unclear

---

## Minimal Workflow

1. Run experiment
2. Observe result
3. Write 1–2 sentence summary
4. Call log_experiment()

That’s it.

---

## Why This Matters

Without logging:
- insights are lost
- experiments repeat unintentionally

With logging:
- patterns emerge
- questions improve
- progress becomes visible

