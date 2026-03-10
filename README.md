# "Interventions, probes, and controlled damage to language models"

## Shared Data Repository

This project depends on the shared asset repository:

https://github.com/3141592/ai_shared_data

Datasets and model assets are **not stored in this repository**.  
They are managed through `ai_shared_data`.

Example setup:

```bash
git clone https://github.com/<yourname>/ai_shared_data
pip install -e ai_shared_data
```

See that repository for dataset download and configuration instructions.

## Environments

This repo intentionally uses **two Python virtual environments**:

- **PyTorch / TransformerLens**
  - venv: `~/venvs/torch`
  - requirements: `requirements-torch.txt`
  - runner: `./run-torch`

- **TensorFlow / Keras**
  - venv: `~/venvs/tf`
  - requirements: `requirements-tf.txt`
  - runner: `./run-tf`

Scripts must be run using the appropriate runner.
Mixing frameworks in one venv is intentionally avoided.

## VS Code

To ensure the right venv is used for each folder, open the workspace file:

- `ai_surgery.code-workspace`

This applies the root `tf` interpreter by default and the `torch` interpreter in any `**/torch` subfolder.

Debugging from VS Code also auto-selects env by file location:
- files under `*/torch/*` use `~/venvs/torch`
- all other files use `~/venvs/tf`
