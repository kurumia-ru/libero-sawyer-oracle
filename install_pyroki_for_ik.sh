#!/usr/bin/env bash
# Optional: install PyRoki + deps into openpi-xemb venv for ik_track --ik-backend pyroki
set -euo pipefail

export PATH="${HOME}/.local/bin:/usr/local/bin:${PATH}"
ROOT="${OPENPI_XEMB_ROOT:-$HOME/openpi_ws/openpi-xemb}"
cd "$ROOT"

echo "Installing PyRoki IK deps into $ROOT/.venv ..."
uv pip install yourdfpy robot_descriptions
uv pip install "jaxls @ git+https://github.com/brentyi/jaxls.git"
uv pip install "pyroki @ git+https://github.com/chungmin99/pyroki.git"

uv run python -c "import pyroki; import yourdfpy; print('pyroki OK', pyroki.__file__)"
echo "Done. Use: --action-mode ik_track --ik-backend pyroki"
