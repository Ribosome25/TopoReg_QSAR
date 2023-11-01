"""
This code visualizes lead optimization pathways and training clusters with TR 
predictions on scaffold or CV split depending on input arguments, see 
scripts/args ChemblLeadOptVisualizationArgs for argument defaults and details

@author: Daniel Nolte
"""

import json
import numpy as np
import pandas as pd
import random 
from utils.args import ChemblLeadOptVisualizationArgs
from utils.topoReg import simple_y_train
from sklearn.linear_model import LinearRegression as LR
from sklearn.metrics import pairwise_distances
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
import networkx as nx
import matplotlib.pyplot as plt
from rdkit import Chem
from rdkit.Chem import Draw
from warnings import warn

def generate_graph_from_edges(edges,distance,le,threshold):
    """
    function to generate a graph from a list of edge connections based on 
    threshold value
    
    inputs: edges: list of knn predictions
            distance: pairwise distance between samples
            le: label encoder to transform from interger to chembl id
            threshold: mean threshold for distance cutoff
    return: G, the calcualted networkx graph
    """
    
    G=nx.Graph() #initialize the graph
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
def ax_disable_ticks(ax):
    """
    Function to turn off x and y axis ticks and spines for given ax
    
    input: ax: ax to modify
    """
    
    ax.get_xaxis().set_ticks([])
    ax.get_yaxis().set_ticks([])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
def visualize_TR_leadopt(args: ChemblLeadOptVisualizationArgs):
    """
    Function to visualize TR lead optimization pathways and training clusters
    in the form of kNN-grpahs
    
    input: args: ChemblLeadOptVisualizationArgs
    
    saves a figure with 3 subplots, one with the training clusters, one with 
    the minimum spanning tree of the most active cluster, and one with 5 
    molecules from the minimum spanning path between the most active and least 
    active molecules in the most active cluster
    """
    
    # set font size
    font_size=16
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
    # If more than 2000 anchors, limit to 2000 to speeds up compuation with 
    # little to no predictive performance cost
    if len(anchors_idx) > 2000:  # if takes too long. 
        anchors_idx = distance.loc[train_idx].sample(n=2000).index
    
    # Sample training and testing distances
    dist_x_train = distance.loc[train_idx, anchors_idx]
    dist_y_train = simple_y_train(target, anchors_idx, "euclidean", train_idx=train_idx) 
    
    # modelling 
    mdl.fit(dist_x_train, dist_y_train.T)
    
    # Prediction 
    dist_pred_train = abs(mdl.predict(dist_x_train))
    ranks_topo = np.argsort(dist_pred_train, axis=1)
    idx_stacked = np.vstack([anchors_idx] * len(train_idx))
    top_idx_topo_train = np.take_along_axis(idx_stacked, ranks_topo, axis=1)[:, :top_k]
    
    # Color bar definitions
    vmin = min(target)
    vmax = max(target)
    

    # setup figure and axes
    fig1 = plt.figure(figsize=(21,14))
    gs = fig1.add_gridspec(2,2, width_ratios=[2, 1],height_ratios=[2.25, 1])
    ax1 = fig1.add_subplot(gs[0, 0])
    ax2 = fig1.add_subplot(gs[0, 1])
    ax3 = fig1.add_subplot(gs[1, :])
    axes=[ax2,ax2,ax3]
    for ax in axes:
        ax_disable_ticks(ax)

    # Label encoder to map between integer node names and chembl molecule names
    le=LabelEncoder()
    le.fit(data.index)
    
    # Find all edges of KNN graph using TR NN predictions
    l=pd.DataFrame(top_idx_topo_train,index=train_idx)
    l['index1'] = l.index
    edges=[x for idx,x in l[['index1']+[i for i in range(top_k)]].iterrows()]
    edges=[le.transform(np.array(x.values)) for x in edges]
    # generate the graph based on NN/edge connections
    G=generate_graph_from_edges(edges,distance,le,threshold)
    nx.set_node_attributes(G, target[le.inverse_transform(G.nodes)])
    # Plot graph
    pos = nx.spring_layout(G,seed=args.seed) 
    nx.draw(G,ax=ax1,pos=pos,node_color=target[le.inverse_transform(G.nodes)], vmin=vmin, vmax=vmax)
    
    #analyze connected subgraphs
    components =  (G.subgraph(c) for c in nx.connected_components(G))
    comp_dict_knnTest = {idx: comp.nodes() for idx, comp in enumerate(components)}
    
    # Find highest active component
    highestActiveComp= np.argmax([target[le.inverse_transform(comp_dict_knnTest[i])].mean() for i in range(len(comp_dict_knnTest))])

    #analyze active subgraph
    Gactive = G.subgraph(comp_dict_knnTest[highestActiveComp])
    nodeList = {node: list(le.inverse_transform([node]))[0] for node in Gactive.nodes}
    edge_weight = {e:{'weight':distance.loc[le.inverse_transform([e[0]]),le.inverse_transform([e[1]])].values[0][0]} for e in Gactive.edges()}
    nx.set_edge_attributes(Gactive, edge_weight)
    
    # draw minimum spanning tree of most active cluster/component
    T = nx.minimum_spanning_tree(Gactive)
    nx.draw(T,ax=ax2,pos = pos,node_color=target[le.inverse_transform(Gactive.nodes)], vmin=vmin, vmax=vmax)
    
    # draw minimum spanning path between most active and least active sample
    Tpath = nx.shortest_path(T, list(Gactive.nodes)[target[le.inverse_transform(Gactive.nodes)].argmax()], list(Gactive.nodes)[target[le.inverse_transform(Gactive.nodes)].argmin()])
    path_edges = list(zip(Tpath,Tpath[1:]))
    nx.draw_networkx_edges(T,ax=ax2,pos=pos,edgelist=path_edges,edge_color='r',width=3)
    nx.draw_networkx_nodes(T,ax=ax2,pos = pos,nodelist = Tpath,node_color=target[le.inverse_transform(Tpath)], vmin=vmin, vmax=vmax,node_size=450)
    
    # Select up to 5 molecules along the path including the first and last
    maxChems = 5
    labels = ['C1','C2','C3','C4','C5','C6']
    if len(Tpath) > maxChems:
        idx = np.round(np.linspace(0, len(Tpath) - 1, maxChems)).astype(int)
        Tpath = [Tpath[i] for i in idx]
    chems = le.inverse_transform(Tpath)
    # plot molecules names on minimum spanning tree
    nx.draw_networkx_labels(T,ax=ax2,pos = pos,labels=dict((k,labels[idx]) for idx,k in enumerate(Tpath) if k in nodeList),font_size=font_size,font_color='r')

    # Draw the 5 molecules as images
    fig2 = plt.figure(figsize=(20,3))
    smiles = data['Smiles'].loc[le.inverse_transform(Tpath)]
    imgs = []
    for sm in smiles:
        mol = Chem.MolFromSmiles(sm)
        imgs.append(Draw.MolToImage(mol, size=(500, 300)))
    
   # plot all 5 images in one figure as subplots
    plotindex = 1
    for idx,img in enumerate(imgs):
        ax = fig2.add_subplot (1,len(imgs), plotindex)
        plt.imshow(img)
        plt.text(125,15, 'pChEMBL Value: '+str(target[chems[idx]]), size=font_size)
        ax_disable_ticks(ax)
        plt.tight_layout()
        plotindex+=1
    fig2.tight_layout()
    fig2.subplots_adjust(left=0.00, bottom=0.00, top=0.96, right=1)
    fig2.canvas.draw()
    
    # convert the figure to an image to plot with NN Graph subplots
    image_from_plot = np.frombuffer(fig2.canvas.tostring_rgb(), dtype=np.uint8)
    image_from_plot = image_from_plot.reshape(fig2.canvas.get_width_height()[::-1] + (3,))
    ax3.imshow(image_from_plot)
    
    # format and save the main figure
    fig1.tight_layout()
    fig1.savefig('visualizations/lead_opt_pathway_'+args.path.strip("/").split("/")[-1]+".png",dpi=300)


if __name__ == "__main__":
    args = ChemblLeadOptVisualizationArgs()
    np.random.seed(args.seed)
    visualize_TR_leadopt(args)
    