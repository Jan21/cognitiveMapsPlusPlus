#!/bin/bash

set -e

module --force purge
module load apptainer

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEF_FILE="${SCRIPT_DIR}/container.def"
SIF_FILE="${SCRIPT_DIR}/cognitiveMapsPlusPlus.sif"

[ -f "$SIF_FILE" ] && rm -f "$SIF_FILE"

export APPTAINER_TMPDIR="${TMPDIR:-/tmp}/apptainer_build_$$"
mkdir -p "$APPTAINER_TMPDIR"

apptainer build --fakeroot "$SIF_FILE" "$DEF_FILE"

rm -rf "$APPTAINER_TMPDIR"

echo "Build complete: $SIF_FILE"