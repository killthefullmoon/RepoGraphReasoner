#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <repo-name>"
  echo "Example: $0 flask"
  exit 1
fi

REPO_NAME="$1"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$ROOT_DIR/dataset/repos/$REPO_NAME"
OUT_DIR="$ROOT_DIR/processed_data/$REPO_NAME"
INDEX_DIR="$OUT_DIR/index"
GRAPH_PREFIX="$OUT_DIR/graph"
GRAPH_PKL="$GRAPH_PREFIX.pkl"
QA_OUTPUT="$OUT_DIR/multi_hop_qas_en_unique.jsonl"

if [[ ! -d "$REPO_DIR" ]]; then
  echo "Repository directory not found: $REPO_DIR" >&2
  exit 2
fi

mkdir -p "$INDEX_DIR"

echo "==> Indexing repository $REPO_NAME"
python "$ROOT_DIR/utils/index_repo.py" \
  --repo-root "$REPO_DIR" \
  --output-dir "$INDEX_DIR"

echo "==> Building knowledge graph"
python "$ROOT_DIR/utils/build_code_KG.py" \
  --imports "$INDEX_DIR/imports.jsonl" \
  --symbols "$INDEX_DIR/symbols.jsonl" \
  --out-prefix "$GRAPH_PREFIX"

echo "==> Generating QA pairs"
python "$ROOT_DIR/utils/gen_code_qa.py" \
  --graph "$GRAPH_PKL" \
  --output "$QA_OUTPUT"

echo "Pipeline complete. Outputs stored under $OUT_DIR"
