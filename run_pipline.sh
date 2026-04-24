#!/bin/bash
########## file_prepare#########
cd "./Ensemble_server"

mkdir "./6d7y_A/example"
python ./lm_embedding_preparation.py --pdb_path ./example/6d7y_A/input/6d7y_A_0.pdb --out_file ./example/fasta/seq.fasta

## get esm ##
target_folder="./6d7y_A"
base_folder=".."
name=$(basename "$target_folder")
mkdir -p "$target_folder/esm"

python ./esm/scripts/extract.py \
  esm2_t33_650M_UR50D \
  "$target_folder/fasta/seq.fasta" \
  "$target_folder/esm" \
  --repr_layers 33 \
  --include per_tok \
  --truncation_seq_length 10000 \
  --model_dir "$base_folder/esm_models" \
  --class_name "$name"

###########DiffEnsemble##########
target_folder="./example/6d7y_A"
for i in {1..5}; do
  bash ./run_model.sh
done

mkdir -p "${target_folder}/output_result"
mv "${target_folder}/input/"* "${target_folder}/output_result/"
ls "${target_folder}/output_result" | head -45 | xargs -I {} cp "${target_folder}/output_result/{}" "${target_folder}/input/"
bash /mydata/cuixinyue/Ensemble_server/run_model.sh

for file in "${target_folder}/input"/*.pdb; do
  filename=$(basename "$file")
  mv "$file" "${target_folder}/output_result/last_${filename}"
done
rm -rf "${target_folder}/output/"*
