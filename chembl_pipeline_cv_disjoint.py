"""
A script for chembl data.
Modif from v3. CV with non neg regression


"""
import os
from os.path import join as pj
from myToolbox.Metrics import octuple
import numpy as np
import pandas as pd
# from py import test
from topo_reg import reconstruct

from topo_reg.args import ChemblPipelineArgs
from topo_reg.calculate_distance import simple_y_train

# from rdkit import Chem
# from rdkit.Chem.EnumerateStereoisomers import EnumerateStereoisomers, StereoEnumerationOptions

from model_utils import load_dfs, load_dist_from_precomputed_parquet, calc_fp_dist_v2

#%%
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestRegressor as RFR
from sklearn.linear_model import LinearRegression as LR
from sklearn.linear_model import Ridge
from sklearn.gaussian_process import GaussianProcessRegressor as GPR
from sklearn.gaussian_process.kernels import RBF
from sklearn.neighbors import KNeighborsRegressor
import itertools

from distance_regressions import Log_LR, LogLog_LR, NNLS, Log0_LR, LogLog0_LR

def pipe_1(args: ChemblPipelineArgs):

    data, df_dict = load_dfs(args)
    size = len(data)
    target = data["pChEMBL Value"]
    k1, k2 = args.schm.split("+")

    # compute_distance
    precompute_dist = {}
    for each_key in df_dict:
        if each_key.endswith("fp") or each_key.endswith("fp4"):
            _metric = args.metric
        elif each_key == "tf3p":
            _metric = 'normcos'  # normalized cosine.
        else:
            _metric = 'euclidean'  # by default, numerical descs uses euclid distance

        each_tuple = df_dict[each_key] # as a tuple

        precompute_dist[each_key] = calc_fp_dist_v2(input=each_tuple,
            save_path=pj(args.path, "{}_{}_dist.parquet".format(each_key, _metric)),
            metric=_metric)
        # max Size: types 6 * 7700^2 * 64 / 1024/1024/1024 = 21 (G), 2000 items is 1.4 G
    if size >= 2000:  # TODO small num to debug
        del precompute_dist  # 删掉var 释放内存，后面有name error 再读文件

    Ks = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]

    # Ks = [0.2,]  # Debug
    rbf_gammas = [0.2, 0.5, 1, 2, 5]
    rbf_gammas = [0.5]  # Debug

    rbf_idx_names = ["RBF_{}".format(each) for each in rbf_gammas]
    Ks_str = ["-Ank{}".format(k) for k in Ks]
    # prepared output table
    topo_index_list = [
        "".join(x) for x in itertools.product(
            # ["rdk2d", "mordred", k1, k2, "concat"],
            [k1, k2, "concat"],
            Ks_str,
            ["-Log_LR-", "-LogLog_LR-", "-NNLS-", "-log0_LR-", "-loglog0_LR-"],
            rbf_idx_names
        )
    ]
    perf_index = topo_index_list


    perf_col = ["Spearman", "Pearson", "MSE", "MAE", "NRMSE", "NMAE", "RMSE", "R2"]
    performance = np.zeros((len(perf_index), 8))

    # split data
    kf = KFold(n_splits=args.cv_fold, shuffle=True, random_state=2021)
    i_fold = 1
    for train_i, test_i in kf.split(target):
        print("Fold {}/{}".format(i_fold, args.cv_fold))
        # split train / test
        train_idx = target.index[train_i]
        test_idx = target.index[test_i]

        # Common reg models
        i_table_row = 0

        # topo reg
        for dset_name in [k1.lower(), k2.lower(), "concatfp"]:  # skip RDKit and Mordred.
            print("\tTopo reg", dset_name)
            try:
                distance = precompute_dist[dset_name]
            except NameError:  # 如果已经删掉
                if dset_name.endswith("fp") or dset_name.endswith("fp4"):
                    _metric = args.metric
                elif dset_name == "tf3p":
                    _metric = 'normcos'  # normalized cosine.
                else:
                    _metric = 'euclidean'  # by default, numerical descs uses euclid distance
                distance = pd.read_parquet(pj(args.path, "{}_{}_dist.parquet".format(dset_name, _metric)), engine='fastparquet')

            """
            3/1/2023: the distance of RDKit and Mordred can go up to 1e8, this may induce further numerical issue. 
            A better way could be normalize the initial RDKit and Mordred values. 
            """

            # This cound be a hyper pramam: to avoid LR overfitting.
            for each_k in Ks:
                # if not(each_k == 0.5 and i_fold == 5):
                #     continue
                print(each_k)  # debug
                anchors_idx = distance.loc[train_idx].sample(frac=each_k, random_state=args.seed).index
                if len(anchors_idx) > 2000:  # if takes too long. 
                    anchors_idx = distance.loc[train_idx].sample(n=2000, random_state=args.seed).index
                if len(anchors_idx) < 8:
                    for _ in range(25):
                        performance[i_table_row] = np.full(8, fill_value=np.nan)
                        i_table_row += 1
                    continue

                # pick out the index values in A but not in B
                index_diff = train_idx.difference(anchors_idx)

                dist_x_train = distance.loc[index_diff, anchors_idx].values
                dist_y_train = simple_y_train(target, anchors_idx, "euclidean", train_idx=index_diff)  # TODO: Euclidean or correlation  m_anch x N
                dist_test = distance.loc[test_idx, anchors_idx]
                target_test = target.loc[test_idx]

                # modelling TODO
                for mdl in [Log_LR(), LogLog_LR(), NNLS(), Log0_LR(), LogLog0_LR()]:
                    try:
                        mdl.fit(dist_x_train, dist_y_train.T)
                    except MemoryError:
                        print("Mem Err")
                        dist_x_train = dist_x_train.astype(np.float16)
                        dist_y_train = dist_y_train.astype(np.float16)
                        mdl.fit(dist_x_train, dist_y_train.T)

                    # model prediction
                    dist_array_test = mdl.predict(dist_test.values).T

                    for each_gamma in rbf_gammas:
                        each_gamma = float(each_gamma)
                        try:
                            response_array_r_t = reconstruct.rbf(dist_array_test, target, anchors_idx, each_gamma, False, False)
                            fold_perform = octuple(target_test, response_array_r_t.ravel(), False)
                        except (np.linalg.LinAlgError, ValueError) as e:
                            fold_perform = np.full(8, fill_value=np.nan)

                        performance[i_table_row] += np.array(fold_perform) / args.cv_fold

                        # # For debug only. 
                        # debug_message = [len(train_idx), len(dist_x_train), len(anchors_idx), len(test_idx), each_gamma, 0, 0, 0]
                        # performance[i_table_row] += np.array(debug_message) / args.cv_fold

                        i_table_row += 1

        # Mean model # todo
        # mean_pred = (rf_pred + topo_pred.ravel()) / 2
        # performance[i_table_row] += np.array(octuple(target_test, mean_pred.ravel(), False)) / args.cv_fold
        # i_table_row += 1

        i_fold += 1

    # save result table.
    tb = pd.DataFrame(performance, index=perf_index, columns=perf_col)
    output_path = pj(args.path, "Results_nonneg_Ghosh.csv")
    tb.to_csv(output_path, mode='a', header=not os.path.exists(output_path))
    return tb

#%%
if __name__ == "__main__":
    args = ChemblPipelineArgs().parse_args()
    pq_files = [f for f in os.listdir(args.path) if f.endswith(".parquet")]
    assert len(pq_files) > 0
    print("Modeling:", args.path)

    pipe_1(args)

