#!/bin/bash
#SBATCH --chdir=./
#SBATCH --job-name=ChEMBL_pipeline
#SBATCH --output=%x_%a.o%j
#SBATCH --error=%x_%a.e%j
#SBATCH --partition nocona 

#SBATCH --nodes=1
#SBATCH --ntasks=64
#SBATCH --time=48:00:00
#SBATCH --mem-per-cpu=4G

#SBATCH -a 62-64

. $HOME/conda/etc/profile.d/conda.sh
conda activate chem
# conda activate dl

# declare -i batch=2
declare -i batch=$SLURM_ARRAY_TASK_ID

wd="data/ChEMBL/batch$batch"
yourfilenames=`ls $wd`

for eachfile in $yourfilenames
do
    echo $eachfile
    # cd ~/Topo_Regression/topological_regression/
    python chembl_pipeline_cv --path $wd/$eachfile --schm ECFP4+TF3P
    python chembl_pipeline_cv_disjoint --path $wd/$eachfile --schm ECFP4+TF3P
    python chembl_pipeline_scaff --path $wd/$eachfile --schm ECFP4+TF3P
    python chembl_pipeline_scaff_disjoint --path $wd/$eachfile --schm ECFP4+TF3P

    python chembl_pipeline_mlkr.py --path $wd/$eachfile --schm ECFP4+TF3P

done

