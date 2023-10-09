"""
Convert csv to parquet, save some disk usage
"""

import os
from timeit import timeit
import pandas as pd
from joblib import Parallel, delayed
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

def get_file_list():
    file_list = []
    for root, dirs, files in os.walk("data/ChEMBL/", topdown=False):
        for name in files:
            if name.endswith(".csv"):
                if name.endswith("_dist.csv"):
                    file_list.append(os.path.join(root, name))
                    continue
                if name.startswith("data_") and not name.startswith("data_cp"):
                    file_list.append(os.path.join(root, name))
                    continue
    return file_list

def convert(path: str):
    each_df = pd.read_csv(path, index_col=0)
    if "Mordred" in path:
        _mask = each_df.applymap(lambda x: isinstance(x, (int, float)))
        each_df = each_df.where(_mask)
        each_df.fillna(each_df.mean(), inplace=True)
        each_df.dropna(axis=1, how='all', inplace=True)
        each_df = each_df.astype(float)
    each_df = each_df.astype(float)
    each_df.index = each_df.index.astype(str)
    each_df.columns = each_df.columns.astype(str)
    each_df.to_parquet(
        path.replace(".csv", ".parquet"),
        engine='fastparquet',
        compression='gzip'
    )
#     print("saved parquet")
#     raise
    os.remove(path)
#     print(path, "Finished.")

file_list = get_file_list()
print(len(file_list))

Parallel(n_jobs=-1, verbose=1)(delayed(convert)(path) for path in file_list)
# from tqdm import tqdm

# for each_path in tqdm(file_list):
#     convert(each_path)


