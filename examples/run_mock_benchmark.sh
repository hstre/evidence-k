#!/usr/bin/env bash
# Run the full Evidence-k pipeline end-to-end with the offline MockModel.
#
# Usage:  ./examples/run_mock_benchmark.sh
#
# Produces a run under runs/<run_id>/ and a k_profile.json in the repo root.
set -euo pipefail

cd "$(dirname "$0")/.."

CONFIG="configs/example.yaml"

if [[ ! -f "$CONFIG" ]]; then
  echo "Config not found; scaffolding with 'evidence-k init'..."
  evidence-k init
fi

echo "==> Running k-sweep (mock model, offline)"
evidence-k run --config "$CONFIG"

# Pick the most recent run directory.
RUN_DIR="$(ls -dt runs/*/ | head -1)"
echo
echo "==> Summarizing $RUN_DIR"
evidence-k summarize --run-dir "$RUN_DIR"

echo
echo "==> Exporting portable k_profile.json"
evidence-k export-profile --run-dir "$RUN_DIR" --out k_profile.json

echo
echo "Done. Profile written to ./k_profile.json"
