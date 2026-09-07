#!/usr/bin/env bash
set -euo pipefail

# Copy completed checkpoint bundles from runtime storage to persistent storage.
cp -r "${WORK_ROOT:?Set WORK_ROOT}/checkpoints/." "${DRIVE_ROOT:?Set DRIVE_ROOT}/checkpoints/"
