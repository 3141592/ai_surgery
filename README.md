# "Interventions, probes, and controlled damage to language models"

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
