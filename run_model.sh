#!/bin/bash

base="./example/6d7y_A"
input="$base/input"
output="$base/output"
temp="$base/temp"
fasta="$base/fasta/seq.fasta"

mkdir -p "$input" "$temp"

cd "$input"
ls . | grep -v 'list.txt' | sed 's/\.pdb$//' > list.txt
list="$input/list.txt"

python /mydata/cuixinyue/Ensemble_server/run_inference.py \
  --protein_path "$base" \
  --target_txt "$list" \
  --out_dir "$base" \
  --esm_embeddings_path "$base/esm" \
  --profile_features "$base/profile/6d7y_A.npz" \
  --model_dir "model.pt" \
  --inference_steps 10 \
  --batch_size 1 \
  --inference_num 1

while IFS= read -r line; do
  cp -f "$output/${line}1.pdb" "$input/" 2>/dev/null || true
done < "$list"

cd "$input"
ls . | grep -v 'list.txt' | sed 's/\.pdb$//' > list.txt
list="$input/list.txt"

i=1
rm -rf "$temp/"*
while IFS= read -r line; do
  dir="$temp/6d7y_A_$i"
  mkdir -p "$dir"
  cp "$fasta" "$dir/6d7y_A_$i.fasta"
  cp "$input/${line}.pdb" "$dir/6d7y_A_$i.pdb"
  ((i++))
done < "$list"

ls "$temp/" | grep -v 'list.txt' > "$temp/list.txt"
list="$temp/list.txt"
bash bash_refine.sh
rm -f "$input"/*.pdb

while IFS= read -r line; do
  cp "$temp/$line/$line.B99990001.pdb" "${input}/${line}_.pdb"
done < "$list"
