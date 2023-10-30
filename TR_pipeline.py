import json
import numpy as np
import pandas as pd
from utils.args import ChemblPipelineArgs
from utils.topoReg import simple_y_train, rbf
from sklearn.linear_model import LinearRegression as LR
from sklearn.metrics import pairwise_distances
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold
from warnings import warn


def TR_pipeline_scaffold(args: ChemblPipelineArgs):
    # Load data
    data = pd.read_csv(args.path+ "data_cp.csv", index_col=0)
    ecfp4 = pd.read_parquet(args.path+ "data_ECFP4.parquet", engine='fastparquet').astype('bool')
    target = data["pChEMBL Value"]
    # Calculate distances
    distance = pairwise_distances(ecfp4.values, metric="jaccard", n_jobs=-1)
    distance = pd.DataFrame(distance, index=ecfp4.index, columns=ecfp4.index)

    # load indicies for scaffold split
    with open(args.path+"scaffold_split_index.json", 'r') as f:
        index = json.load(f)  
    train_idx = index['train_idx']
    test_idx = index['test_idx']
    
    # Initialize model and sample anchor points
    mdl =LR(n_jobs=-1)
    anchors_idx = distance.loc[train_idx].sample(frac=args.anchor_percentage).index
    if len(anchors_idx) > 2000:  # if takes too long. 
        anchors_idx = distance.loc[train_idx].sample(n=2000).index
        
    # Sample training and testing distances
    dist_x_train = distance.loc[train_idx, anchors_idx]
    dist_y_train =simple_y_train(target, anchors_idx, "euclidean", train_idx=train_idx)
    dist_test = distance.loc[test_idx, anchors_idx]
    
    # Modelling 
    mdl.fit(dist_x_train, dist_y_train.T)
    
    # Prediction
    dist_array_test = abs(mdl.predict(dist_test)).T
    predictedResponse = rbf(dist_array_test, target, anchors_idx, args.rbf_gamma, False, False).ravel()
    
    # Metric calculation
    scorr, pvalue = spearmanr(target.loc[test_idx], predictedResponse)
    r2 = r2_score(target.loc[test_idx], predictedResponse)
    rmse = mean_squared_error(target.loc[test_idx], predictedResponse, squared=False)
    std_i = np.std(target.loc[test_idx])
    nrmse = rmse / std_i
    print('\nScaffold TR Performance')
    print('Spearman: '+str(scorr))
    print('R2: '+str(r2))
    print('RMSE: '+str(rmse))
    print('NRMSE: '+str(nrmse))
    return scorr,r2,rmse,nrmse,predictedResponse
    
def TR_pipeline_cv(args: ChemblPipelineArgs):
    # Load data
    data = pd.read_csv(args.path+ "data_cp.csv", index_col=0)
    ecfp4 = pd.read_parquet(args.path+ "data_ECFP4.parquet", engine='fastparquet').astype('bool')
    # Calculate distances
    target = data["pChEMBL Value"]
    distance = pairwise_distances(ecfp4.values, metric="jaccard", n_jobs=-1)
    distance = pd.DataFrame(distance, index=ecfp4.index, columns=ecfp4.index)
    
    # Split data and perform CV
    kf = KFold(n_splits=args.cv_fold, shuffle=True, random_state=args.seed)
    fold = 1
    fold_perform = []
    for train_i, test_i in kf.split(target):
        # split train / test
        train_idx = target.index[train_i]
        test_idx = target.index[test_i]
        
        # Initialize model and sample anchor points
        mdl =LR(n_jobs=-1)
        anchors_idx = distance.loc[train_idx].sample(frac=args.anchor_percentage).index
        if len(anchors_idx) > 2000:  # if takes too long. 
            anchors_idx = distance.loc[train_idx].sample(n=2000).index
            
        # Sample training and testing distances
        dist_x_train = distance.loc[train_idx, anchors_idx]
        dist_y_train =simple_y_train(target, anchors_idx, "euclidean", train_idx=train_idx)
        dist_test = distance.loc[test_idx, anchors_idx]
        
        # Modelling 
        mdl.fit(dist_x_train, dist_y_train.T)
        # Prediction
        dist_array_test = abs(mdl.predict(dist_test)).T
        predictedResponse = rbf(dist_array_test, target, anchors_idx, args.rbf_gamma, False, False).ravel()
        # Metric calculation
        scorr, pvalue = spearmanr(target.loc[test_idx], predictedResponse)
        r2 = r2_score(target.loc[test_idx], predictedResponse)
        rmse = mean_squared_error(target.loc[test_idx], predictedResponse, squared=False)
        std_i = np.std(target.loc[test_idx])
        nrmse = rmse / std_i
        print('\nTR Performance Fold'+str(fold))
        print('Spearman: '+str(scorr))
        print('R2: '+str(r2))
        print('RMSE: '+str(rmse))
        print('NRMSE: '+str(nrmse))
        fold+=1
        fold_perform.append([scorr,r2,rmse,nrmse])
    # Metric averaging across folds
    avgPerformance = np.array(fold_perform).mean(axis=0)
    print('\nAverage CV TR Performance')
    print('Spearman: '+str(avgPerformance[0]))
    print('R2: '+str(avgPerformance[1]))
    print('RMSE: '+str(avgPerformance[2]))
    print('NRMSE: '+str(avgPerformance[3]))
    return avgPerformance
        

if __name__ == "__main__":
    args = ChemblPipelineArgs()
    np.random.seed(args.seed)
    if args.split == 'scaffold':
        TR_pipeline_scaffold(args)
    elif args.split == 'cv':
        TR_pipeline_cv(args)
    else:
        warn("Input argument split must be set to either 'scaffold' (default) or 'cv'")