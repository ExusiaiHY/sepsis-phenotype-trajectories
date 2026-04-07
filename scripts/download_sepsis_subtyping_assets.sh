#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
BASE_DIR="${1:-${PROJECT_ROOT}/external_assets/sepsis_subtyping}"
DATA_DIR="${BASE_DIR}/new_datasets"
TOOLS_DIR="${BASE_DIR}/tools"
INCLUDE_CHALLENGE="${INCLUDE_CHALLENGE:-0}"
INCLUDE_TOOLS="${INCLUDE_TOOLS:-0}"

mkdir -p "${DATA_DIR}" "${TOOLS_DIR}"

download() {
  local url="$1"
  local output="$2"
  if [[ -f "${output}" ]]; then
    echo "[skip] ${output}"
    return 0
  fi
  echo "[download] ${output}"
  curl -L --fail --retry 3 --retry-delay 2 "${url}" -o "${output}"
}

try_download() {
  local url="$1"
  local output="$2"
  if ! download "${url}" "${output}"; then
    echo "[warn] skipped after download failure: ${url}"
    rm -f "${output}"
  fi
}

mirror_index_tree() {
  local url="$1"
  local output_dir="$2"
  mkdir -p "${output_dir}"
  echo "[mirror] ${output_dir} <= ${url}"
  local index_file
  index_file="$(mktemp)"
  curl -L --fail --retry 3 --retry-delay 2 "${url}" -o "${index_file}"
  while IFS= read -r href; do
    if [[ -z "${href}" ]] || [[ "${href}" == "../" ]]; then
      continue
    fi
    if [[ "${href}" == */ ]]; then
      mirror_index_tree "${url%/}/${href}" "${output_dir}/${href%/}"
    else
      download "${url%/}/${href}" "${output_dir}/${href}"
    fi
  done < <(
    rg -o 'href="[^"]+"' "${index_file}" \
      | sed -E 's/^href="(.*)"$/\1/' \
      | rg -v '^(mailto:|#)'
  )
  rm -f "${index_file}"
}

try_mirror_index_tree() {
  local url="$1"
  local output_dir="$2"
  if ! mirror_index_tree "${url}" "${output_dir}"; then
    echo "[warn] skipped after mirror failure: ${url}"
  fi
}

# New datasets for subtype and treatment-response research.
if [[ "${INCLUDE_CHALLENGE}" == "1" ]]; then
  mirror_index_tree \
    "https://physionet.org/files/challenge-2019/1.0.0/training/" \
    "${DATA_DIR}/physionet_challenge2019_v1_0_0"
else
  echo "[skip] PhysioNet Challenge 2019 training tree (set INCLUDE_CHALLENGE=1 to mirror)"
fi
download \
  "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE65nnn/GSE65682/matrix/GSE65682_series_matrix.txt.gz" \
  "${DATA_DIR}/GSE65682_series_matrix.txt.gz"
download \
  "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE185nnn/GSE185263/suppl/GSE185263_raw_counts.csv.gz" \
  "${DATA_DIR}/GSE185263_raw_counts.csv.gz"
download \
  "https://download.cncb.ac.cn/OMIX/OMIX006238/OMIX006238-01.csv" \
  "${DATA_DIR}/OMIX006238_proteomics.csv"

# Public tools and reference implementations are opt-in.
if [[ "${INCLUDE_TOOLS}" == "1" ]]; then
  download \
    "https://physionet.org/files/challenge-2019/1.0.0/sources/AI4Sepsis.zip" \
    "${TOOLS_DIR}/AI4Sepsis_challenge2019.zip"
  download \
    "https://physionet.org/files/challenge-2019/1.0.0/sources/CanIgetyoursignature.zip" \
    "${TOOLS_DIR}/CanIgetyoursignature_challenge2019.zip"
else
  echo "[skip] Public tools bundle (set INCLUDE_TOOLS=1 to download)"
fi

echo
echo "Assets downloaded under: ${BASE_DIR}"
