#!/bin/bash
# Make the single script a job runs: an entry script of this folder with the parameters of one launch written in.
#
#   bash make_main.sh <calibrate|train_smoke|train> <env file> <output file>
#
# The env file holds `NAME=value` or `export NAME=value` lines (comments and blank lines are skipped); it stays outside
# the repository (it names the output location). The output is the entry script with those lines as `export`
# statements after its first line, so the job needs nothing but this one file and what REPO_SRC points to.
# Refused: an env file without RUN_ID, OUT_URI or REPO_SRC; MRGYM_STUB=1 (a simulation is not a job); a secret written
# as a value (MRGYM_SECRET, MRGYM_SALT): the node reads the key from a file named by SECRET_URI.
set -euo pipefail
[ $# = 3 ] || { sed -n '2,10p' "$0"; exit 2; }
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ENTRY=$HERE/$1.sh; ENVF=$2; OUT=$3
[ -f "$ENTRY" ] || { echo "no entry script $ENTRY"; exit 2; }
LINES=$(grep -E '^[[:space:]]*(export[[:space:]]+)?[A-Za-z_][A-Za-z0-9_]*=' "$ENVF" | sed -E 's/^[[:space:]]*(export[[:space:]]+)?/export /')
for need in RUN_ID OUT_URI REPO_SRC; do echo "$LINES" | grep -q "^export $need=" || { echo "$ENVF sets no $need"; exit 2; }; done
echo "$LINES" | grep -Eq '^export MRGYM_STUB=["'"'"']?1' && { echo "MRGYM_STUB=1 is for the local simulation, not for a job"; exit 2; }
echo "$LINES" | grep -Eq '^export (MRGYM_SECRET|MRGYM_SALT)=' && { echo "a secret must not be written into a job script; use SECRET_URI"; exit 2; }
{ head -n 1 "$ENTRY"; echo "# ---- parameters of this launch (written by make_main.sh from $(basename "$ENVF"))"; echo "$LINES"; tail -n +2 "$ENTRY"; } > "$OUT"
bash -n "$OUT"
echo "$OUT: $(wc -c < "$OUT") bytes, sha256 $(sha256sum "$OUT" | cut -c1-16), $(echo "$LINES" | wc -l) parameter(s)"
