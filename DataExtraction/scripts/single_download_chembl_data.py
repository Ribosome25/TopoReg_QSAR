# -*- coding: utf-8 -*-
"""
Created on 10/30/2023

@author: Ruibo
"""

import os
import pandas as pd
from chembl_webresource_client.new_client import new_client
from rdkit import Chem
from rdkit.Chem.EnumerateStereoisomers import EnumerateStereoisomers, StereoEnumerationOptions
from rdkit.Chem.Descriptors import ExactMolWt
from tqdm import tqdm
import numpy as np

import argparse

def download_single_target(args):
    target = new_client.target
    activity = new_client.activity

    res = activity.filter(target_chembl_id=args.target_ID, assay_type='B', pchembl_value__isnull=False)

    df_from = pd.DataFrame(list(res))
    # print(df_from.columns)
    df_to = pd.DataFrame()
    df_to["Standard Type"] = df_from["standard_type"]
    df_to["Standard Units"] = df_from["standard_units"]
    df_to["Smiles"] = df_from["canonical_smiles"]
    df_to["Standard Value"] = df_from["standard_value"]
    df_to["Standard Units"] = df_from["standard_units"]
    df_to["pChEMBL Value"] = df_from['pchembl_value']
    df_to['Assay ChEMBL ID'] = df_from["assay_chembl_id"]
    df_to['Standard Relation'] = df_from["standard_relation"]
    df_to['Target ChEMBL ID'] = df_from["target_chembl_id"]


    n_isomers = []
    mol_weights = []
    for idx, item in df_to.iterrows():
        sml = item["Smiles"]
        if sml is None:
            n_isomers.append(np.nan)
            mol_weights.append(np.nan)
            continue
        m = Chem.MolFromSmiles(sml)
        n_isomers.append(len(tuple(EnumerateStereoisomers(m))))
        mol_weights.append(ExactMolWt(m))
    df_to['n_isomers'] = n_isomers
    df_to["Molecular Weight"] = mol_weights
    df_to.index = df_from['molecule_chembl_id']
    df_to.index.name = "Compound_ID"

    os.makedirs(os.path.join(args.save_dir, args.target_ID))
    df_to.to_csv(os.path.join(args.save_dir, args.target_ID, "py_downloaded.tsv"), sep="\t")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--target_ID", type=str)
    parser.add_argument("--save_dir", type=str, default="SampleDatasets/ChEMBL")
    args = parser.parse_args()

    download_single_target(args)
