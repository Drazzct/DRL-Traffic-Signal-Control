#!/usr/bin/env bash
set -euo pipefail

# Phase 1 network compilation entry point.
netconvert --osm-files "${OSM_FILE:?Set OSM_FILE}" --output-file "${NET_FILE:?Set NET_FILE}"
