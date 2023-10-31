# Created on 10/30/2023 @Ruibo
# Example code to illustrate how to prepare the data from TR demos. 

wd="SampleDatasets/ChEMBL"
tgt="CHEMBL278"

python DataExtraction/scripts/single_download_chembl_data.py --target_ID $tgt --save_dir $wd

python DataExtraction/scripts/chembl_process_data_v2.py --path $wd/$tgt

python chemml/generate_descriptors.py --descriptors RDKit Mordred ECFP4 --path $wd/$tgt

python chemml/generate_descriptors.py --descriptors E3FP --path $wd/$tgt

python DataExtraction/scripts/convert_data_format.py --work_dir $wd

