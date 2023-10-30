import json
import numpy as np
import pandas as pd
import random 
from utils.args import ChemblNNVisualizationArgs
from utils.topoReg import simple_y_train
from sklearn.linear_model import LinearRegression as LR
from sklearn.metrics import pairwise_distances
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from metric_learn import MLKR
import networkx as nx
import matplotlib.pyplot as plt
from warnings import warn


def generate_graph_from_edges(edges,distance,le,threshold):
    G=nx.Graph() #create the graph and add edges
    # For each sample, add its 5 NN edges based on threshold
    for e in edges:
        if (distance.loc[le.inverse_transform([e[0]]),le.inverse_transform([e[1]])].values <threshold)&(e[0]!=e[1]):
            G.add_edge(e[0],e[1])
        if (distance.loc[le.inverse_transform([e[0]]),le.inverse_transform([e[2]])].values <threshold)&(e[0]!=e[2]):
            G.add_edge(e[0],e[2])
        if (distance.loc[le.inverse_transform([e[0]]),le.inverse_transform([e[3]])].values <threshold)&(e[0]!=e[3]):
            G.add_edge(e[0],e[3])
        if (distance.loc[le.inverse_transform([e[0]]),le.inverse_transform([e[4]])].values <threshold)&(e[0]!=e[4]):
            G.add_edge(e[0],e[4])
        if (distance.loc[le.inverse_transform([e[0]]),le.inverse_transform([e[5]])].values <threshold)&(e[0]!=e[5]):
            G.add_edge(e[0],e[5])
    return G

def visualize_NN_test_predictions(args):
    #Set random seed
    np.random.seed(args.seed)
    random.seed(args.seed)
    top_k = args.k
    
    # load data
    data = pd.read_csv(args.path+ "data_cp.csv", index_col=0)
    ecfp4 = pd.read_parquet(args.path+ "data_ECFP4.parquet", engine='fastparquet').astype('bool')
    
    # Calcaulte distances and mean TC threshold 
    target = data["pChEMBL Value"]
    distance = pairwise_distances(ecfp4.values, metric="jaccard", n_jobs=-1)
    distance = pd.DataFrame(distance, index=ecfp4.index, columns=ecfp4.index)
    threshold = distance.mean().mean()

    # Get split indicies based on input argument
    if args.split == 'cv':
        cv_fold = args.cv_fold
        # split data
        kf = KFold(n_splits=cv_fold, shuffle=True, random_state=args.seed)
        for train_i, test_i in kf.split(target):
            # split train / test
            train_idx = target.index[train_i]
            test_idx = target.index[test_i]
            break
    elif args.split == 'scaffold':
        with open(args.path+ "scaffold_split_index.json", 'r') as f:
            index = json.load(f)   
        train_idx = index['train_idx']
        test_idx = index['test_idx']#+index['stack_idx']
    else:
        warn("Input argument split must be set to either 'scaffold' or 'cv' (default)")
    
    # sort test idx based on target
    target_test = target[test_idx]
    target_test = target_test.sort_values()
    test_idx = target_test.index
    
    # Define LR model
    mdl =LR(n_jobs=-1)
    
    #Sample anchor points
    anchors_idx = distance.loc[train_idx].sample(frac=args.anchor_percentage).index
    if len(anchors_idx) > 2000:  # if takes too long. 
        anchors_idx = distance.loc[train_idx].sample(n=2000).index
    
    # Sample training and testing distances
    dist_x_train = distance.loc[train_idx, anchors_idx]
    dist_y_train = simple_y_train(target, anchors_idx, "euclidean", train_idx=train_idx) 
    dist_test = distance.loc[test_idx, anchors_idx]
    
    # modelling 
    mdl.fit(dist_x_train, dist_y_train.T)
    # Prediction 
    dist_array_test = abs(mdl.predict(dist_test))

    # TR and KNN Test Predictions
    dist_test_knn = distance.loc[test_idx, train_idx]
    ranks_raw = np.argsort(dist_test_knn.values, axis=1)
    ranks_topo = np.argsort(dist_array_test, axis=1)
    idx_stacked_knn = np.vstack([train_idx] * len(test_idx))
    idx_stacked = np.vstack([anchors_idx] * len(test_idx))
    top_idx_raw = np.take_along_axis(idx_stacked_knn, ranks_raw, axis=1)[:, :args.k]
    top_idx_topo = np.take_along_axis(idx_stacked, ranks_topo, axis=1)[:, :args.k]
     
    # MLKR Modeling
    mlkr = MLKR(max_iter=200, verbose=False, tol=1e-9, init='identity', random_state=args.seed)
    mlkr.fit(ecfp4.loc[train_idx], target.loc[train_idx])
    x_tr_t = mlkr.transform(ecfp4.loc[train_idx])
    x_tst_t = mlkr.transform(ecfp4.loc[test_idx])
    x_mlkr = mlkr.transform(ecfp4)

    

    # MLKR Test Predictions
    distanceMLKR = pairwise_distances(x_mlkr, metric="euclidean", n_jobs=-1)
    distanceMLKR = pd.DataFrame(distanceMLKR, index=ecfp4.index, columns=ecfp4.index)
    dist_test_MLKR = distanceMLKR.loc[test_idx, train_idx]
    ranks_rawMLKR = np.argsort(dist_test_MLKR.values, axis=1)
    idx_stacked_knn = np.vstack([train_idx] * len(test_idx))
    top_idx_rawMLKR  = np.take_along_axis(idx_stacked_knn, ranks_rawMLKR, axis=1)[:, :args.k]
    
    # Color bar definitions
    vmin = min(target)
    vmax = max(target)
    cmap=plt.cm.viridis
    
    # TR Subpot
    fig1 = plt.figure(figsize=(5/6*30, 5/6*10))
    sub_i = 133
    ax = fig1.add_subplot(sub_i)
    
    # Label encoder to map between integer node names and chembl molecule names
    le=LabelEncoder()
    le.fit(data.index)
    # Find all edges of KNN graph using TR NN predictions
    l=pd.DataFrame(top_idx_topo,index=test_idx)
    l['index1'] = l.index
    edges=[x for idx,x in l[['index1']+[i for i in range(args.k)]].iterrows()]
    edges=[le.transform(np.array(x.values)) for x in edges]
    # Generate graph
    G=generate_graph_from_edges(edges,distance,le,threshold)
    # Plot graph
    nx.set_node_attributes(G, target[le.inverse_transform(G.nodes)])
    pos = nx.spring_layout(G,seed=args.seed) 
    nx.draw(G,pos=pos,node_color=target[le.inverse_transform(G.nodes)], vmin=vmin, vmax=vmax)
    components =  (G.subgraph(c) for c in nx.connected_components(G)) #analyze connected subgraphs
    comp_dict_trTest  = {idx: comp.nodes() for idx, comp in enumerate(components)}
    ax.set_title("TR: Intercluster Std: {:.3f}".format(np.mean([target[le.inverse_transform(comp_dict_trTest[i])].std() for i in range(len(comp_dict_trTest))])), fontsize=25)
    
    # KNN Subpot
    sub_i = 131
    ax = fig1.add_subplot(sub_i)
    # Find all edges of KNN graph using KNN predictions
    l=pd.DataFrame(top_idx_raw,index=test_idx)
    l['index1'] = l.index
    edges=[x for idx,x in l[['index1']+[i for i in range(args.k)]].iterrows()]
    edges=[le.transform(np.array(x.values)) for x in edges]
    G=generate_graph_from_edges(edges,distance,le,threshold)
    nx.set_node_attributes(G, target[le.inverse_transform(G.nodes)])
    pos = nx.spring_layout(G,seed=args.seed) 
    nx.draw(G,pos=pos,node_color=target[le.inverse_transform(G.nodes)], vmin=vmin, vmax=vmax)
    components =  (G.subgraph(c) for c in nx.connected_components(G)) #analyze connected subgraphs
    comp_dict_knnTest = {idx: comp.nodes() for idx, comp in enumerate(components)}
    ax.set_title("KNN: Intercluster Std: {:.3f}".format(np.mean([target[le.inverse_transform(comp_dict_knnTest[i])].std() for i in range(len(comp_dict_knnTest))])), fontsize=25)
    
    # MLKR Subpot
    sub_i = 132
    ax = fig1.add_subplot(sub_i)
    # Find all edges of KNN graph using MLKR NN predictions
    l=pd.DataFrame(top_idx_rawMLKR,index=test_idx)
    l['index1'] = l.index
    edges=[x for idx,x in l[['index1']+[i for i in range(args.k)]].iterrows()]
    edges=[le.transform(np.array(x.values)) for x in edges]
    G=generate_graph_from_edges(edges,distance,le,threshold)
    nx.set_node_attributes(G, target[le.inverse_transform(G.nodes)])
    pos = nx.spring_layout(G,seed=args.seed) 
    im1 = nx.draw(G,pos=pos,node_color=target[le.inverse_transform(G.nodes)], vmin=vmin, vmax=vmax)
    components =  (G.subgraph(c) for c in nx.connected_components(G)) #analyze connected subgraphs
    comp_dict_MLKRTest = {idx: comp.nodes() for idx, comp in enumerate(components)}
    ax.set_title("MLKR: Intercluster Std: {:.3f}".format(np.mean([target[le.inverse_transform(comp_dict_MLKRTest[i])].std() for i in range(len(comp_dict_MLKRTest))])), fontsize=25)

    # Format Plot
    fig1.suptitle('KNN Graph Clustering', fontsize=30)
    cbar_ax = fig1.add_axes([0.96, 0.05, 0.01, 0.90])
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin = vmin, vmax=vmax))
    cbar = fig1.colorbar(sm, cax=cbar_ax)
    cbar.vmin=vmin
    cbar.vmax=vmax
    fig1.tight_layout()
    fig1.subplots_adjust(left=0.00, bottom=0.00, top=0.88, right=0.96, wspace=0.1)
    fig1.savefig('visualize_NN_predictions_'+args.path.strip("/").split("/")[-1]+".png",dpi=300)



if __name__ == "__main__":
    args = ChemblNNVisualizationArgs().parse_args()
    np.random.seed(args.seed)
    visualize_NN_test_predictions(args)
    