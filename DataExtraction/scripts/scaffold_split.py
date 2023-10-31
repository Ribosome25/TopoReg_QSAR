"""

For ChEMBL dataset. 

Scaffolds split based on ChemProp. 

save index as json, convient for topo reg. 

also should save a copy of splited csv, for chemprop. 


"""

import logging
from random import Random
from typing import Tuple
import os
from os.path import join as pj
import json
import pandas as pd
from tqdm import tqdm
from chemai.args import GenDescArgs
from chemprop.data.data import MoleculeDataset
from chemprop.data.scaffold import log_scaffold_stats, scaffold_to_smiles
from chemprop.data.utils import get_data, scaffold_split


def get_file_list(root_dir):
    file_list = []
    for root, dirs, files in os.walk(root_dir, topdown=False):
        for name in files:
            if name.endswith("data.txt"):
                file_list.append(os.path.join(root, name))
    return file_list


def scaffold_split_cp(data: MoleculeDataset,
                   sizes: Tuple[float, float, float] = (0.8, 0.1, 0.1),
                   balanced: bool = False,
                   seed: int = 0,
                   logger: logging.Logger = None) -> Tuple[MoleculeDataset,
                                                           MoleculeDataset,
                                                           MoleculeDataset]:
    r"""
    Splits a :class:`~chemprop.data.MoleculeDataset` by scaffold so that no molecules sharing a scaffold are in different splits.

    :param data: A :class:`MoleculeDataset`.
    :param sizes: A length-3 tuple with the proportions of data in the train, validation, and test sets.
    :param balanced: Whether to balance the sizes of scaffolds in each set rather than putting the smallest in test set.
    :param seed: Random seed for shuffling when doing balanced splitting.
    :param logger: A logger for recording output.
    :return: A tuple of :class:`~chemprop.data.MoleculeDataset`\ s containing the train,
             validation, and test splits of the data.
    """
    assert sum(sizes) == 1

    if data.number_of_molecules > 1:
        raise ValueError('Cannot perform a scaffold split with more than one molecule per datapoint.')

    # Split
    train_size, val_size, test_size = sizes[0] * len(data), sizes[1] * len(data), sizes[2] * len(data)
    train, val, test = [], [], []
    train_scaffold_count, val_scaffold_count, test_scaffold_count = 0, 0, 0

    # Map from scaffold to index in the data
    scaffold_to_indices = scaffold_to_smiles(data.mols(flatten=True), use_indices=True)  # 注：dict, {scaffold smile：related data indice}

    # Seed randomness
    random = Random(seed)

    if balanced:  # Put stuff that's bigger than half the val/test size into train, rest just order randomly
        index_sets = list(scaffold_to_indices.values())
        big_index_sets = []
        small_index_sets = []
        for index_set in index_sets:
            # if len(index_set) > val_size / 2 or len(index_set) > test_size / 2:  # 原版
            if len(index_set) > test_size / 2:  # 如果没有val set，会全变成大set
                big_index_sets.append(index_set)
            else:
                small_index_sets.append(index_set)
        random.seed(seed)
        random.shuffle(big_index_sets)
        random.shuffle(small_index_sets)
        index_sets = big_index_sets + small_index_sets
    else:  # Sort from largest to smallest scaffold sets
        index_sets = sorted(list(scaffold_to_indices.values()),
                            key=lambda index_set: len(index_set),
                            reverse=True)

    for index_set in index_sets:
        if len(train) + len(index_set) <= train_size:
            train += index_set
            train_scaffold_count += 1
        elif len(val) + len(index_set) <= val_size:
            val += index_set
            val_scaffold_count += 1
        else:
            test += index_set
            test_scaffold_count += 1

    if logger is not None:
        logger.debug(f'Total scaffolds = {len(scaffold_to_indices):,} | '
                     f'train scaffolds = {train_scaffold_count:,} | '
                     f'val scaffolds = {val_scaffold_count:,} | '
                     f'test scaffolds = {test_scaffold_count:,}')

    if logger is not None:
        log_scaffold_stats(data, index_sets, logger=logger)

    # Map from indices to data
    # train = [data[i] for i in train]
    # val = [data[i] for i in val]
    # test = [data[i] for i in test]

    # 注：顺序是对的. 返回 i index
    return train, val, test
    return MoleculeDataset(train), MoleculeDataset(val), MoleculeDataset(test)


def split(path):
    data_df = pd.read_table(path, index_col=0)
    data_df.to_csv(path.replace("data.txt", "data_cp.csv"))

    data = get_data(
        path=path.replace("data.txt", "data_cp.csv"), 
        smiles_columns='Smiles',
        target_columns=['pChEMBL Value'],
    )

    train_ii, val_ii, test_ii = scaffold_split_cp(
        data=data,
        sizes=(0.8, 0, 0.2),
        balanced=True
    )


    idx = {
        "train_idx": data_df.index[train_ii].tolist(),
        "val_idx": data_df.index[val_ii].tolist(),
        "test_idx": data_df.index[test_ii].tolist()
    }

    # print(idx)

    with open(os.path.dirname(path) + '/scaffold_split_index.json', 'w') as f:
        json.dump(idx, f)

    csv_dir = pj(os.path.dirname(path), 'For_Chemprop_scaffold/')
    try:
        os.mkdir(csv_dir)
    except FileExistsError:
        pass
    data_df.loc[idx['train_idx']].to_csv(pj(csv_dir, "train.csv"))
    data_df.loc[idx['test_idx']].to_csv(pj(csv_dir, "test.csv"))

if __name__ == "__main__":

    root_dir = "/home/ruibzhan/Topo_Regression/topological_regression/data/"
    # root_dir = "G:/topological_regression/data/ChEMBL/"  # debug
    # args = GenDescArgs().parse_args()  # first try
    f_list = get_file_list(root_dir)
    f_list = sorted(f_list)
    # print(f_list)
    for each in tqdm(f_list):
        # print(each)
        split(each)
        pass
