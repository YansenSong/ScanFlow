#!/usr/bin/env bash
set -euo pipefail

# ScanFlow Conda environment bootstrap.
#
# Usage:
#   bash setup_conda_env.sh              # auto: NVIDIA -> cu126, otherwise CPU
#   bash setup_conda_env.sh cpu
#   bash setup_conda_env.sh cu126
#   bash setup_conda_env.sh cu128
#
# Optional environment variables:
#   SCANFLOW_ENV_NAME=myenv bash setup_conda_env.sh
#   SCANFLOW_SKIP_SMOKE=1 bash setup_conda_env.sh

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="${ROOT_DIR}/environment.yml"
ENV_NAME="${SCANFLOW_ENV_NAME:-scanflow}"
TORCH_VERSION="${SCANFLOW_TORCH_VERSION:-2.7.0}"
BACKEND="${1:-auto}"

if ! command -v conda >/dev/null 2>&1; then
  echo "[ScanFlow] conda was not found in PATH." >&2
  echo "Install Miniconda/Anaconda first, then rerun this script." >&2
  exit 1
fi

if [[ ! -f "${ENV_FILE}" ]]; then
  echo "[ScanFlow] missing ${ENV_FILE}" >&2
  exit 1
fi

case "${BACKEND}" in
  auto)
    if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi >/dev/null 2>&1; then
      BACKEND="cu126"
    else
      BACKEND="cpu"
    fi
    ;;
  cpu|cu126|cu128)
    ;;
  *)
    echo "Usage: bash setup_conda_env.sh [auto|cpu|cu126|cu128]" >&2
    exit 2
    ;;
esac

echo "[ScanFlow] environment : ${ENV_NAME}"
echo "[ScanFlow] PyTorch     : ${TORCH_VERSION} (${BACKEND})"

# `conda env list --json` is stable across shells and avoids relying on activation.
if conda env list --json | "$(conda info --base)/bin/python" -c \
  'import json,sys; name=sys.argv[1]; d=json.load(sys.stdin); print(any(p.rstrip("/").endswith("/envs/"+name) or p.rstrip("/").endswith("\\\\envs\\\\"+name) for p in d["envs"]))' \
  "${ENV_NAME}" | grep -qx True; then
  echo "[ScanFlow] updating existing Conda environment..."
  conda env update --name "${ENV_NAME}" --file "${ENV_FILE}" --prune
else
  echo "[ScanFlow] creating Conda environment..."
  conda env create --name "${ENV_NAME}" --file "${ENV_FILE}"
fi

TORCH_INDEX="https://download.pytorch.org/whl/${BACKEND}"
echo "[ScanFlow] installing PyTorch from ${TORCH_INDEX}"
conda run --name "${ENV_NAME}" python -m pip install --upgrade \
  "torch==${TORCH_VERSION}" \
  --index-url "${TORCH_INDEX}"

echo "[ScanFlow] verifying imports..."
conda run --name "${ENV_NAME}" python - <<'PY'
import sys
import numpy as np
import torch
import casadi as ca
import matplotlib

print("python    :", sys.version.split()[0])
print("numpy     :", np.__version__)
print("torch     :", torch.__version__)
print("cuda avail:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("cuda      :", torch.version.cuda)
    print("gpu       :", torch.cuda.get_device_name(0))
print("casadi    :", ca.__version__)
print("matplotlib:", matplotlib.__version__)
PY

if [[ "${SCANFLOW_SKIP_SMOKE:-0}" != "1" ]]; then
  echo "[ScanFlow] running project smoke tests..."
  conda run --name "${ENV_NAME}" python "${ROOT_DIR}/model.py"
  conda run --name "${ENV_NAME}" python "${ROOT_DIR}/train.py" --smoke-test
  conda run --name "${ENV_NAME}" python "${ROOT_DIR}/generate_dataset.py" --smoke-test
  conda run --name "${ENV_NAME}" python "${ROOT_DIR}/planner.py"
fi

cat <<MSG

[ScanFlow] environment is ready.
Activate it with:
  conda activate ${ENV_NAME}

Useful first checks:
  python model.py
  python train.py --smoke-test
  python test/test_gt_motion_field_nmpc.py
MSG
