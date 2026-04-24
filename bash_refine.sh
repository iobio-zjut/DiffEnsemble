#!/bin/bash
set -euo pipefail


LIST_FILE="/mydata/cuixinyue/Ensemble_server/example/6d7y_A/temp/list.txt"
TARGET_FOLDER="/mydata/cuixinyue/Ensemble_server/example/6d7y_A/temp"
PY_FILE="/mydata/cuixinyue/Ensemble_server/refine.py"
MAX_JOBS=10
CLEAN_JOBS=5

OUTPUT_DIR="${TARGET_FOLDER%/temp}/input"

mapfile -t LIST_NAMES < "$LIST_FILE"

process_job() {
    local name=$1
    local folder="${TARGET_FOLDER}/${name}"
    local pdb="${folder}/${name}.pdb"
    local fasta="${folder}/${name}.fasta"

    [[ ! -f "$pdb" || ! -f "$fasta" ]] && return 1

    python "$PY_FILE" "$pdb" "$fasta"
}

for name in "${LIST_NAMES[@]}"; do
    while [[ $(jobs -r | wc -l) -ge $MAX_JOBS ]]; do sleep 1; done
    (process_job "$name") &
done
wait

current=0
for name in "${LIST_NAMES[@]}"; do
    (
        rm -f "${TARGET_FOLDER}/${name}/${name}.ali"
        rm -f "${TARGET_FOLDER}/${name}/${name}-${name}.ali"
    ) &
    ((current++))
    ((current >= CLEAN_JOBS)) && { wait -n; ((current--)); }
done
wait

mkdir -p "$OUTPUT_DIR"

idx=1
for name in "${LIST_NAMES[@]}"; do
    src="${TARGET_FOLDER}/${name}/${name}.B99990001.pdb"
    [[ -f "$src" ]] && cp "$src" "${OUTPUT_DIR}/${name}_${idx}.pdb"
    ((idx++))
done

echo "ok"