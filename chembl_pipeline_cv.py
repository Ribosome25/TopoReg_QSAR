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

from distance_regressions import Log_LR, LogLog_LR, NNLS

def pipe(args: ChemblPipelineArgs):

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

    # prepared output table
    perf_index = [
        "".join(x) for x in itertools.product(
            ["RDKit desc", "Mordred desc", k1, k2, "Concat FP"],
            [" - RF"]
        )
    ]
    topo_index_list = [
        "".join(x) for x in itertools.product(
            ["rdk2d", "mordred", k1, k2, "concat"],
            ["-LR-", "-Log_LR-", "-LogLog_LR-", "-NNLS-"],
            ["RBF_0.2", "RBF_0.5", "RBF_1", "RBF_2", "RBF_5",]
        )
    ]
    perf_index.extend(topo_index_list)

    # add here:
    dual_topo_index_list = [
        "".join(x) for x in itertools.product(
            ["Dual-" + k1 + "+" + k2],
            ["-LR-", "-Log_LR-", "-LogLog_LR-", "-NNLS-"],
            ["RBF_0.2", "RBF_0.5", "RBF_1", "RBF_2", "RBF_5"] 
        )
    ]
    perf_index.extend(dual_topo_index_list)

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

        # save for chemprop
        if not os.path.exists(pj(args.path, "For_Chemprop/Fold_{}".format(i_fold)):
            chemprop_path = check_path_exists(pj(args.path, "For_Chemprop/Fold_{}".format(i_fold)))
            data.loc[train_idx].to_csv(pj(chemprop_path, "train.csv"))
            data.loc[test_idx].to_csv(pj(chemprop_path, "test.csv"))

        # Common reg models
        i_table_row = 0
        print("\tCommon reg models")
        for dset_name in ["rdk2d", "mordred", k1.lower(), k2.lower(), "concatfp"]:
            x_train = df_dict[dset_name][0][train_i].reshape(len(train_i), -1)  # keep the same for 2d
            x_test = df_dict[dset_name][0][test_i].reshape(len(test_i), -1)
            y_train = target.loc[train_idx]
            y_test = target.loc[test_idx]

            for mdl in [RFR(n_jobs=-1, random_state=args.seed)]:
                mdl.fit(x_train, y_train)
                pred = mdl.predict(x_test)
                if dset_name == "mordred" and isinstance(mdl, RFR):
                    rf_pred = np.copy(pred)
                fold_perform = octuple(y_test, pred, False)

                performance[i_table_row] += np.array(fold_perform) / args.cv_fold
                i_table_row += 1

        # topo reg
        print("\tTopo reg")
        for dset_name in ["rdk2d", "mordred", k1.lower(), k2.lower(), "concatfp"]:
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
            # anchors_idx = train_idx  # TODO using all of the anchor idx. OR:
            anchors_idx = distance.loc[train_idx].sample(frac=0.9, random_state=args.seed).index
            if len(anchors_idx) > 2000:  # if takes too long. 
                anchors_idx = distance.loc[train_idx].sample(n=2000, random_state=args.seed).index

            dist_x_train = distance.loc[train_idx, anchors_idx].values
            dist_y_train = simple_y_train(target, anchors_idx, "euclidean", train_idx=train_idx)  # TODO: Euclidean or correlation  m_anch x N
            dist_test = distance.loc[test_idx, anchors_idx]
            target_test = target.loc[test_idx]

            # modelling TODO
            for mdl in [LR(n_jobs=-1), Log_LR(), LogLog_LR(), NNLS()]:
                try:
                    mdl.fit(dist_x_train, dist_y_train.T)
                except MemoryError:
                    print("Mem Err")
                    dist_x_train = dist_x_train.astype(np.float16)
                    dist_y_train = dist_y_train.astype(np.float16)
                    mdl.fit(dist_x_train, dist_y_train.T)

                # model prediction
                dist_array_test = mdl.predict(dist_test.values).T

                for each_gamma in [0.2, 0.5, 1, 2, 5]:
                    each_gamma = float(each_gamma)
                    try:
                        response_array_r_t = reconstruct.rbf(dist_array_test, target, anchors_idx, each_gamma, False, False)
                        fold_perform = octuple(target_test, response_array_r_t.ravel(), False)
                    except np.linalg.LinAlgError:
                        fold_perform = np.full(8, fill_value=np.nan)

                    performance[i_table_row] += np.array(fold_perform) / args.cv_fold
                    i_table_row += 1


        # Dual topo reg
        print("\tDual Topo reg")
        for dual_dsets in [(k1.lower(), k2.lower())]:
            try:
                distance1 = precompute_dist[dual_dsets[0]]
                distance2 = precompute_dist[dual_dsets[1]]
            except NameError:  # 如果已经删掉
                distance1 = load_dist_from_precomputed_parquet(dual_dsets[0], args)
                distance2 = load_dist_from_precomputed_parquet(dual_dsets[1], args)

            # This frac cound be a hyper pramam: to avoid LR overfitting.
            # 用上一节的 distance 也行，反正是随机抽 index
            # anchors_idx = distance.loc[train_idx].sample(frac=1, random_state=args.seed).index
            anchors_idx = distance.loc[train_idx].sample(frac=0.45, random_state=args.seed).index
            if len(anchors_idx) > 2000:  # if takes too long. 
                anchors_idx = distance.loc[train_idx].sample(n=2000, random_state=args.seed).index

            dist_x_train = np.hstack((distance1.loc[train_idx, anchors_idx].values, distance2.loc[train_idx, anchors_idx].values))
            dist_y_train = simple_y_train(target, anchors_idx, "euclidean", train_idx=train_idx)  # TODO: Euclidean or correlation
            dist_test = np.hstack((distance1.loc[test_idx, anchors_idx].values, distance2.loc[test_idx, anchors_idx].values))
            target_test = target.loc[test_idx]

            for mdl in [LR(n_jobs=-1), Log_LR(), LogLog_LR(), NNLS()]:
                try:
                    mdl.fit(dist_x_train, dist_y_train.T)
                except MemoryError:
                    dist_x_train = dist_x_train.astype(np.float16)
                    dist_y_train = dist_y_train.astype(np.float16)
                    mdl.fit(dist_x_train, dist_y_train.T)

                # model prediction
                dist_array_test = mdl.predict(dist_test).T

                for each_gamma in [0.2, 0.5, 1, 2, 5]:
                    each_gamma = float(each_gamma)
                    try:
                        response_array_r_t = reconstruct.rbf(dist_array_test, target, anchors_idx, each_gamma, False, False)
                        fold_perform = octuple(target_test, response_array_r_t.ravel(), False)
                    except np.linalg.LinAlgError:
                        fold_perform = np.full(8, fill_value=np.nan)

                    performance[i_table_row] += np.array(fold_perform) / args.cv_fold
                    i_table_row += 1

        # Mean model # todo
        # mean_pred = (rf_pred + topo_pred.ravel()) / 2
        # performance[i_table_row] += np.array(octuple(target_test, mean_pred.ravel(), False)) / args.cv_fold
        # i_table_row += 1

        i_fold += 1

    # save result table.
    tb = pd.DataFrame(performance, index=perf_index, columns=perf_col)
    output_path = pj(args.path, "Results_nonneg_cv.csv")
    tb.to_csv(output_path, mode='a', header=not os.path.exists(output_path))
    return tb

#%%
if __name__ == "__main__":
    args = ChemblPipelineArgs().parse_args()
    pq_files = [f for f in os.listdir(args.path) if f.endswith(".parquet")]
    assert len(pq_files) > 0
    print("Modeling:", args.path)
    
    pipe(args)
    # result_csv = pd.read_csv(pj(args.path, "Results.csv"), index_col=0)
    # # Add a filter to fill some unfilled ones. 
    # if not 'ECFP4-LR-RBF_0.5' in result_csv.index:
    #     pipe(args)
    # print("Finished.")

