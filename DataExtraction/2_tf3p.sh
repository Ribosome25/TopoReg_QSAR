#!/bin/bash
#SBATCH --chdir=./
#SBATCH --job-name=TF3P_ChemProp
#SBATCH --output=%x_%a.o%j
#SBATCH --error=%x_%a.e%j
#SBATCH --partition=matador
#SBATCH --gpus-per-node=1
#SBATCH --nodes=1
#SBATCH --ntasks=20
#SBATCH --time=48:00:00

#SBATCH -a 1-28

. $HOME/conda/etc/profile.d/conda.sh
conda activate chem

# generate TF3P for all subfolders.

declare -i batch=$SLURM_ARRAY_TASK_ID

wd="/home/ruibzhan/TopoReg_QSAR/data/ChEMBL/batch$batch"
yourfilenames=`ls $wd`
cd /home/ruibzhan/TF3P
for eachfile in $yourfilenames
do
    echo "TF3P: $eachfile"
    python inference.py -wd $wd/$eachfile
done

