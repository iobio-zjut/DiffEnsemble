# DiffEnsemble: Exploring protein conformational ensembles using evolutionary conditional diffusion

**DiffEnsemble** is a protein conformational ensembles modeling method.  

Motivated by the hypothesis that static structural diversity can be mapped to a dynamic conformational spectrum, we propose DiffEnsemble, a diffusion-based framework designed for modeling protein conformational ensembles. DiffEnsemble learns latent dynamical representations from static protein structures in the Protein Data Bank, integrated with the structural profile derived from the AlphaFold Protein Structure Database as conditional guidance during the diffusion process. 

---
## 🧪 Method Overview

The method was evaluated on the ATLAS molecular dynamics simulation dataset using a comprehensive set of complementary metrics, including Pearson correlation coefficients of pairwise RMSD and RMSF, the distribution of Cα atoms in PCA conformational space, as well as assessments of physical and stereochemical plausibility.

---

## 🚀 Getting Started

### 🔧 Software Requirements

Create a new environment for inference. While in the project directory run:

conda create -n DiffEnsemble python=3.10 
conda activate DiffEnsemble
conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia
conda install -c conda-forge rdkit
conda install pyg  pyyaml  biopython -c pyg
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.0.0+cu117.html
pip install e3nn  fair-esm spyrmsd

## 🏃 Running the Pipeline

### 📌 Example: Run inference

```
python run_inference.py --protein_path ${target_path} --target_txt ${target_list} --out_dir ${output folder} --esm_embeddings_path ${esm embedding} --profile_features ${tructural profile} --model_dir ${ckpt.pt} --inference_steps ${inference_steps} --batch_size ${num}
```
