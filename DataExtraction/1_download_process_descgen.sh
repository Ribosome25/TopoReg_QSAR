#!/bin/bash
#SBATCH --chdir=./
#SBATCH --job-name=Prepare_dataset
#SBATCH --output=%x_%a.o%j
#SBATCH --error=%x_%a.e%j
#SBATCH --partition nocona 

#SBATCH --nodes=1
#SBATCH --ntasks=32
#SBATCH --time=48:00:00
#SBATCH --mem-per-cpu=4G

#SBATCH -a 1-25

# download data + process, split + gen descs

. $HOME/conda/etc/profile.d/conda.sh
conda activate chem

# declare -i batch=7
declare -i batch=$SLURM_ARRAY_TASK_ID

python ~/TopoReg_QSAR/scripts/ChEMBL/batch_download_chembl_data.py $((20*($batch-1))) $((20*$batch)) "batch$batch"

wd="data/ChEMBL/batch$batch"
yourfilenames=`ls $wd`

for eachfile in $yourfilenames
do
    echo $eachfile

    cd ~/TopoReg_QSAR/
    python chembl_process_data_v2.py --path $wd/$eachfile

    cd ~/TopoReg_QSAR/chemml/

    # # For some targets that did not generate descs correctly: skip and reduce some time. 
    # if test -f ~/TopoReg_QSAR/$wd/$eachfile/data_RDKdesc.csv; then
    #	    continue
    # fi
    
    python generate_descriptors.py --descriptors RDKit Mordred ECFP4 --path ~/TopoReg_QSAR/$wd/$eachfile

done

echo "================ Success. E3FP starts ================="

cd ~/TopoReg_QSAR/chemml/
for eachfile in $yourfilenames
do
    echo $eachfile
    python generate_descriptors.py --descriptors E3FP --path ~/TopoReg_QSAR/$wd/$eachfile
done
