# -*- coding: utf-8 -*-
"""
Created on Sun Dec 19 00:11:34 2021

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
parser = argparse.ArgumentParser()
parser.add_argument('start', type=int)
parser.add_argument('end', type=int)
parser.add_argument("batch_tag", type=str)
args = parser.parse_args()



targets = pd.read_excel("~/TopoReg_QSAR/data/ChEMBL/Large_scale SM2.xlsx", sheet_name=1, index_col=0).index
targets = targets[args.start: args.end]
batch_tag = args.batch_tag
print("Downloading from to:", args.start, args.end)

for ID in tqdm(targets):
    print(ID)
    target = new_client.target
    activity = new_client.activity

    res = activity.filter(target_chembl_id=ID, assay_type='B', pchembl_value__isnull=False)

    len(res)
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
    for idx, item in df_to.iterrows():  # Calculate # of stereo isomers. 
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

    os.makedirs("data/ChEMBL/{}/{}/".format(batch_tag, ID))
    df_to.to_csv("~/TopoReg_QSAR/data/ChEMBL/{}/{}/py_downloaded.tsv".format(batch_tag, ID), sep="\t")


