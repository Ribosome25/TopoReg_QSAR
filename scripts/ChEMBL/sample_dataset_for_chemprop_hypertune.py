"""
Randomly pick 20 datasets for ChemProp hypertuning. 

"""

import os
from os.path import join as pj
from random import sample, seed

import pandas as pd

cb_dir = "data/ChEMBL/"

# walk into 1 lvl
dir_list_l1 = [x for x in os.listdir(cb_dir) if x.startswith('batch')]
dir_list = []
for batch in dir_list_l1:
    targ = [x for x in os.listdir(pj(cb_dir, batch))]
    dir_list.extend(
        [pj(cb_dir, batch, x) for x in targ]
    )
# test
print(len(dir_list))
dir_list = sorted(dir_list)
print(dir_list[0], dir_list[-1])
seed(2022)
sampled = sample(dir_list, 20)

full_path = []

textfile = open("sampled_targets_for_chemprop_hypertune.txt", "w")
for element in sampled:
    df = pd.read_table(pj(element, "data.txt"), index_col=0)
    new_f_path = pj("/home/ruibzhan/Topo_Regression/topological_regression/", element)
    print(new_f_path)
    df.to_csv(pj(new_f_path, "data_cp.csv"))
    textfile.write(new_f_path + "\n")
textfile.close()
