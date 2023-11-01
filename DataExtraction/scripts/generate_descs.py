"""
Decriptors generation.

Note:
    * All functions except the general entry generate()
    support both csv and txt files. 

    * RDKit and RDKit 3D can be used for generating descs for single file;
    E3FP, Mordred, csv will be only used for the whole folder with txt files. 

    * each_file, file_list must be the full path to file(s).

"""
import argparse
import os
import numpy as np
import pandas as pd
from tqdm import tqdm


def check_output_path(input_file: str, output_path, suffix=''):
    """
    """
    folder, file_name = os.path.split(input_file)
    new_file_name = file_name.replace(".txt", "").replace(".csv", "") + suffix + ".csv"
    if output_path is None:
        output_path = os.path.join(folder, new_file_name)
    elif os.path.isdir(output_path):
        output_path = os.path.join(output_path, new_file_name)
    return output_path


def gen_morganfp(file_list: list, output_path: str):
    from rdkit import Chem
    from rdkit.Chem import MolFromSmiles
    from rdkit.Chem import AllChem

    for each_file in file_list:
        raw_data = pd.read_table(each_file, index_col=0)
        arr = np.zeros((len(raw_data), 2048))
        output_idx = []
        count_na = 0
        ii = 0

        for idx, each_row in tqdm(raw_data.iterrows()):
            smiles = each_row['Smiles']
            mol = MolFromSmiles(smiles)
            if mol is None:
                count_na += 1
                fp = None
                arr[ii] = np.empty(2048)
            else:
                fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)  # r=2, equalv to ECFP4
                Chem.DataStructs.ConvertToNumpyArray(fp, arr[ii])
            output_idx.append(idx)
            ii += 1

        if count_na > 0:
            print("{} / {} mols are NaNs (cannot be found in RDKit) in {}.".format(count_na, ii, each_file))
        output = pd.DataFrame(arr, index=output_idx, columns=["FP_{}".format(ii) for ii in range(2048)])
        new_output_path = check_output_path(each_file, output_path, "_ECFP4")
        output.to_parquet(new_output_path, engine='fastparquet', compression='gzip')



def generate(args):
    """
    """
    path = args.path
    out_path = args.output
    descs = args.descriptors

    if isinstance(path, str):
        # print(type(path))
        # input path must exist.
        if os.path.isdir(path):
            file_list = [os.path.join(path, x) for x in os.listdir(path) if x.endswith('.txt')]  # complete path 
            assert len(file_list) > 0, "Cannot find .txt files in {}".format(path)
            assert (out_path is None or os.path.isdir(path)), "Output path must be a dir if input path is a dir."
        elif os.path.exists(path):
            file_list = [path]
        else:
            raise ValueError("Cannot resolve the input data path.")
    elif isinstance(path, list):
        # Leave a opening. Change the arg type to List if one day need to gen 2 at the same time.
        file_list = path


    for each_desc in descs:
        if 'morgan' in each_desc.lower() or 'ecfp4' in each_desc.lower():
            gen_morganfp(file_list, out_path)
        else:
            raise ValueError("Unknown descriptor: {}".format(each_desc))

if __name__ == "__main__":
    """

    
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--descriptors", type=str, nargs='+')
    parser.add_argument("--path", type=str, default="SampleDatasets/ChEMBL")
    parser.add_argument("--output", type=str, default="SampleDatasets/ChEMBL")
    args = parser.parse_args()

    generate(args)
