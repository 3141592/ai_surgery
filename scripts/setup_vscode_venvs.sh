#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TF_PY="${HOME}/venvs/tf/bin/python"
TORCH_PY="${HOME}/venvs/torch/bin/python"

write_settings() {
  local dir="$1"
  local interp="$2"
  mkdir -p "$dir/.vscode"
  cat > "$dir/.vscode/settings.json" <<JSON
{
  "python.defaultInterpreterPath": "${interp}"
}
JSON
}

write_settings "$ROOT_DIR" "$TF_PY"

while IFS= read -r torch_dir; do
  write_settings "$torch_dir" "$TORCH_PY"
done < <(find "$ROOT_DIR" -type d -name torch)

echo "VS Code interpreter settings written." >&2
