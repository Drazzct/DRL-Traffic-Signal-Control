#!/usr/bin/env bash
set -euo pipefail

python -m scenarios.generators.scenario_factory \
  --manifest "${MANIFEST_PATH:-configs/scenario_manifest.csv}" \
  --split "${SCENARIO_SPLIT:-TR}" \
  --output-root "${SCENARIO_OUTPUT_ROOT:-scenarios}"
