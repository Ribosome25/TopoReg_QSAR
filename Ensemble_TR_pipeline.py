import json
import numpy as np
import pandas as pd
from scripts.args import ChemblPipelineArgsEnsemble
from sklearn.linear_model import LinearRegression as LR
from sklearn.metrics import pairwise_distances
from scipy.stats import spearmanr
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold
from warnings import warn

def simple_y_train(response, anchors_idx, metric, train_idx=None):
    # A quick fix, considering the compatability. 
    anchors_response = response.loc[anchors_idx]
    if train_idx is not None:
        response = response.loc[train_idx]
    if response.ndim == 1:
        response = response.values.reshape(-1, 1)
        anchors_response = anchors_response.values.reshape(-1, 1)
    response_dist = pairwise_distances(anchors_response, response, metric=metric)
    return response_dist

def _rbf(x, s):
    return np.exp(-(x/s)**2)

def rbf(dist_array, response_train, anchors_idx, gamma=1, debug_plot=False, clip=True) -> np.array:
    """

    :param dist_array: distance array predicted. n_anchors x n_test
    :type dist_array: np.array
    :param response_train: DESCRIPTION
    :type response_train: pd.DataFrame
    :param anchors_idx: DESCRIPTION
    :type anchors_idx: TYPE
    :param gamma: If None, defaults to 1.0.
    :type gamma: TYPE, optional
    :return: DESCRIPTION
    :rtype: TYPE

    """
    #  Cut off duistance at 0, added 1/16:
    if clip:
        dist_array = np.clip(dist_array, 1e-3, None)

    n_a, n_t = dist_array.shape
    response_real_values = response_train.loc[anchors_idx]
    if response_real_values.ndim == 1:
        response_real_values = response_real_values.values.reshape(-1, 1)
    if gamma is None:
        # gamma = 1 / response_real_values.shape[1]
        # gamma = np.mean(response_real_values.values.ravel(), axis=None)
        gamma = np.mean(dist_array, axis=None)

    rbf_v = np.vectorize(_rbf)
    k = rbf_v(dist_array, gamma).T  # rbf of distance. n_t x n_a
    h = np.linalg.inv(np.diag(k.sum(axis=1)))  # normalize mat. n_test x n_test
    r = np.asarray(response_real_values)# .values  # real y. n_anchors x n_features.
    rt = h @ k @ r  # np.matmul. Does it work?
    if debug_plot:
        t = h@k
        import seaborn as sns
        sns.distplot(t[2, :], bins=50)
    return rt
        
def Ensemble_TR_pipeline_scaffold(args: ChemblPipelineArgsEnsemble):
    data = pd.read_csv(args.path+ "data_cp.csv", index_col=0)
    ecfp4 = pd.read_parquet(args.path+ "data_ECFP4.parquet", engine='fastparquet').astype('bool')

    target = data["pChEMBL Value"]
    distance = pairwise_distances(ecfp4.values, metric="jaccard", n_jobs=-1)
    distance = pd.DataFrame(distance, index=ecfp4.index, columns=ecfp4.index)
    
    Ks = np.random.normal(args.mean_k,args.std_k,args.num_TR_models)
    Ks[Ks<0.3]=0.300
    Ks[Ks>0.9]=0.900
    with open(args.path+"scaffold_split_index.json", 'r') as f:
        index = json.load(f)  
    train_idx = index['train_idx']
    test_idx = index['test_idx']
    
    
    
    
    preds = []
    for k in Ks:
        mdl =LR(n_jobs=-1)
        anchors_idx = distance.loc[train_idx].sample(frac=k).index
        if len(anchors_idx) > 2000:  # if takes too long. 
            anchors_idx = distance.loc[train_idx].sample(n=2000).index
            
            
        dist_x_train = distance.loc[train_idx, anchors_idx]
        dist_y_train =simple_y_train(target, anchors_idx, "euclidean", train_idx=train_idx)
        dist_test = distance.loc[test_idx, anchors_idx]
        
        # modelling 
        mdl.fit(dist_x_train, dist_y_train.T)
        dist_array_test = abs(mdl.predict(dist_test)).T
        predictedResponse = rbf(dist_array_test, target, anchors_idx, args.rbf_gamma, False, False).ravel()
        preds.append(predictedResponse)
        
    predictedResponse = np.array(preds).mean(axis=0)
    scorr, pvalue = spearmanr(target.loc[test_idx], predictedResponse)
    r2 = r2_score(target.loc[test_idx], predictedResponse)
    rmse = mean_squared_error(target.loc[test_idx], predictedResponse, squared=False)
    std_i = np.std(target.loc[test_idx])
    nrmse = rmse / std_i
    print('\nScaffold Ensemble TR Performance')
    print('Spearman: '+str(scorr))
    print('R2: '+str(r2))
    print('RMSE: '+str(rmse))
    print('NRMSE: '+str(nrmse))
    return scorr,r2,rmse,nrmse,predictedResponse
    
def Ensemble_TR_pipeline_cv(args: ChemblPipelineArgsEnsemble):
    data = pd.read_csv(args.path+ "data_cp.csv", index_col=0)
    ecfp4 = pd.read_parquet(args.path+ "data_ECFP4.parquet", engine='fastparquet').astype('bool')

    target = data["pChEMBL Value"]
    distance = pairwise_distances(ecfp4.values, metric="jaccard", n_jobs=-1)
    distance = pd.DataFrame(distance, index=ecfp4.index, columns=ecfp4.index)
    
    Ks = np.random.normal(args.mean_k,args.std_k,args.num_TR_models)
    Ks[Ks<0.3]=0.300
    Ks[Ks>0.9]=0.900
    
    kf = KFold(n_splits=args.cv_fold, shuffle=True, random_state=args.seed)
    fold = 1
    fold_perform = []
    for train_i, test_i in kf.split(target):
        # split train / test
        train_idx = target.index[train_i]
        test_idx = target.index[test_i]
        
        preds = []
        for k in Ks:
            mdl =LR(n_jobs=-1)
            anchors_idx = distance.loc[train_idx].sample(frac=k).index
            if len(anchors_idx) > 2000:  # if takes too long. 
                anchors_idx = distance.loc[train_idx].sample(n=2000).index
                
                
            dist_x_train = distance.loc[train_idx, anchors_idx]
            dist_y_train =simple_y_train(target, anchors_idx, "euclidean", train_idx=train_idx)
            dist_test = distance.loc[test_idx, anchors_idx]
            
            # modelling 
            mdl.fit(dist_x_train, dist_y_train.T)
            dist_array_test = (mdl.predict(dist_test)).T
            predictedResponse = rbf(dist_array_test, target, anchors_idx, args.rbf_gamma, False, False).ravel()
            preds.append(predictedResponse)
        predictedResponse = np.array(preds).mean(axis=0)
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
    avgPerformance = np.array(fold_perform).mean(axis=0)
    print('\nAverage CV Ensemble TR Performance')
    print('Spearman: '+str(avgPerformance[0]))
    print('R2: '+str(avgPerformance[1]))
    print('RMSE: '+str(avgPerformance[2]))
    print('NRMSE: '+str(avgPerformance[3]))
    return avgPerformance
        

if __name__ == "__main__":
    args = ChemblPipelineArgsEnsemble().parse_args()
    np.random.seed(args.seed)
    if args.split == 'scaffold':
        Ensemble_TR_pipeline_scaffold(args)
    elif args.split == 'cv':
        Ensemble_TR_pipeline_cv(args)
    else:
        warn("Input argument split must be set to either 'scaffold' (default) or 'cv'")