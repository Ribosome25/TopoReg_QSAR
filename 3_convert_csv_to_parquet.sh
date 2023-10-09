#!/bin/bash
#SBATCH --chdir=./
#SBATCH --job-name=Convert_df_files
#SBATCH --output=%x.o%j
#SBATCH --error=%x.e%j
#SBATCH --partition nocona 

#SBATCH --nodes=1
#SBATCH --ntasks=16
#SBATCH --time=48:00:00
#SBATCH --mem-per-cpu=4G

. $HOME/conda/etc/profile.d/conda.sh
conda activate tf-gpu 

python scripts/ChEMBL/convert_data_format.py
