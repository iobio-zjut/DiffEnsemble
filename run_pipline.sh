#!/bin/bash
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
