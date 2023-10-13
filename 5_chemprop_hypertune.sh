#!/bin/bash
#SBATCH --chdir=./
#SBATCH --job-name=Hyper_ChemProp
#SBATCH --output=%x.o%j
#SBATCH --error=%x.e%j
#SBATCH --partition=matador
#SBATCH --gpus-per-node=1
#SBATCH --nodes=1
#SBATCH --ntasks=20
#SBATCH --time=48:00:00

. $HOME/conda/etc/profile.d/conda.sh
conda activate chem

mapfile -t myArray < sampled_targets_for_chemprop_hypertune.txt

cd /home/ruibzhan/chemml
for (( i = 0 ; i < 1 ; i++))
do
  echo "Element [$i]: ${myArray[$i]}"
  python chemprop_hypertune.py\
    --data_path "${myArray[$i]}/data_cp.csv"\
    --smiles_column Smiles\
    --target_columns "pChEMBL Value"\
    --dataset_type regression\
    --num_iters 20\
    --config_save_path "${myArray[$i]}/hypertune/chemprop/best_params.json"


done
