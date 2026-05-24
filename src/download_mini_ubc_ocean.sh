#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATASET_DIR="${ROOT_DIR}/dataset"
SELECTION_CSV="${DATASET_DIR}/mini_train_selection.csv"
OUT_DIR="${DATASET_DIR}/raw/UBC-OCEAN"
COMPETITION="UBC-OCEAN"
DOWNLOAD_TRAIN_IMAGES="${DOWNLOAD_TRAIN_IMAGES:-1}"
DOWNLOAD_THUMBNAILS="${DOWNLOAD_THUMBNAILS:-1}"

export XDG_CONFIG_HOME="${ROOT_DIR}/.config"
export KAGGLE_CONFIG_DIR="${ROOT_DIR}/.kaggle"
mkdir -p "${XDG_CONFIG_HOME}" "${KAGGLE_CONFIG_DIR}"

if [[ -n "${KAGGLE_BIN:-}" ]]; then
  KAGGLE_BIN="${KAGGLE_BIN}"
elif [[ "${KAGGLE_API_TOKEN:-}" == KGAT_* && -x "${ROOT_DIR}/.python_packages/bin/kaggle" ]]; then
  export PYTHONPATH="${ROOT_DIR}/.python_packages:${PYTHONPATH:-}"
  KAGGLE_BIN="${ROOT_DIR}/.python_packages/bin/kaggle"
elif command -v kaggle >/dev/null 2>&1; then
  KAGGLE_BIN="$(command -v kaggle)"
elif [[ -x "${ROOT_DIR}/.python_packages/bin/kaggle" ]]; then
  export PYTHONPATH="${ROOT_DIR}/.python_packages:${PYTHONPATH:-}"
  KAGGLE_BIN="${ROOT_DIR}/.python_packages/bin/kaggle"
else
  KAGGLE_BIN=""
fi

if [[ -z "${KAGGLE_API_TOKEN:-}" && ! -f "${KAGGLE_CONFIG_DIR}/kaggle.json" ]]; then
  echo "No Kaggle credential found." >&2
  echo "Set KAGGLE_API_TOKEN or place kaggle.json at: ${KAGGLE_CONFIG_DIR}/kaggle.json" >&2
  exit 1
fi

if [[ -z "${KAGGLE_BIN}" ]] || [[ ! -x "${KAGGLE_BIN}" ]]; then
  echo "The kaggle Python package is not available in this Python environment." >&2
  echo "Install it first, for example: python -m pip install kaggle" >&2
  exit 1
fi

if [[ ! -f "${SELECTION_CSV}" ]]; then
  echo "Missing selection file: ${SELECTION_CSV}" >&2
  exit 1
fi

mkdir -p "${OUT_DIR}/train_images" "${OUT_DIR}/train_thumbnails"

echo "Using Kaggle CLI: ${KAGGLE_BIN}"
echo "Checking Kaggle competition access..."
"${KAGGLE_BIN}" competitions files -c "${COMPETITION}" >/dev/null

echo "Downloading metadata..."
"${KAGGLE_BIN}" competitions download -c "${COMPETITION}" -f train.csv -p "${OUT_DIR}"
"${KAGGLE_BIN}" competitions download -c "${COMPETITION}" -f sample_submission.csv -p "${OUT_DIR}" || true

echo "Downloading selected files..."
tail -n +2 "${SELECTION_CSV}" | cut -d, -f1 | while read -r image_id; do
  [[ -z "${image_id}" ]] && continue
  echo "  image_id=${image_id}"
  if [[ "${DOWNLOAD_TRAIN_IMAGES}" == "1" ]]; then
    "${KAGGLE_BIN}" competitions download -c "${COMPETITION}" -f "train_images/${image_id}.png" -p "${OUT_DIR}/train_images"
  fi
  if [[ "${DOWNLOAD_THUMBNAILS}" == "1" ]]; then
    "${KAGGLE_BIN}" competitions download -c "${COMPETITION}" -f "train_thumbnails/${image_id}_thumbnail.png" -p "${OUT_DIR}/train_thumbnails" || true
  fi
done

echo "Done. Files are under: ${OUT_DIR}"
