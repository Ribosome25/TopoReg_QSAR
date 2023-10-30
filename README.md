# TopoReg_QSAR

## Introduction
This code repository contains example code for the paper Topological Regression in Quantitative Structure-Activity Relationship Modeling. Quantitative structure-activity relationship (QSAR) modeling is a powerful tool used in drug discovery, yet the lack of interpretability of commonly used QSAR models hinders their application in molecular design. 
We proposed a similarity-based regression framework, topological regression (TR), that offers a statistically grounded, computationally fast, and interpretable technique. TR provides predictive performance that competes with deep-learning methods, such as Transformer-CNN, as well as informative visualizations, which can help guide lead optimization in drug discovery. 

In this package, we provide Python files for TR, Ensemble TR, k-Nearest-Neighbor Graph Visualization, and Lead optimization pathway visualization, as well as sample ChEMBL datasets and the code to extract all datasets used in the paper.

   
## Dependencies
The code has been tested in Windows 10 and CentOS 8.1 with Python 3.8 or greater and the following packages:
* NumPy >= 1.23.5
* SciPy >= 1.10.1
* pandas >= 1.5.0
* scikit-learn >= 1.2.2
* NetworkX >= 3.1
* metric-learn >= 0.6.2
* Matplotlib >= 3.6.1
* RDKit >= 2020.09.5

## TR Demo
To train and evaluate TR on a requested dataset, simply run the file 'TR_pipeline.py' with the path argument set to the path to the data folder
```
python TR_pipeline.py --path 'PATH TO DATA'
```
The script will output the model performance with metrics NRMSE and Spearman correlation. 
### Parameters (Defaults)
One can select the data splitting method, anchor point percentage, RBF gamma, number of CV folds, the distance metric, and the random seed with the following input arguments:
```
path: str = 'SampleDatasets/ChEMBL/CHEMBL278/'  # the working folder that contains the data files
metric: str='tanimoto'  # the default distance metric to use on FPs
split: str='scaffold'  # which data split to evaluate, either 'scaffold' (default) or 'cv'
seed: int=2021  # random seed 
cv_fold: int=5  # number of folds in cross-validation
anchor_percentage: float=0.5    # Anchor point percentage
rbf_gamma: float=0.5    # gamma for RBF reconstruction
```
## Ensemble TR Demo
Similarly to above, Ensemble TR can be trained and evaluated by running the 'Ensemble_TR_pipeline.py' file
### Parameters (Defaults)
One can select the data splitting method, number of TR models, average anchor point percentage, standard deviation of anchor point percentages, RBF gamma, number of CV folds, the distance metric, and the random seed with the following input arguments:
```
path: str = 'SampleDatasets/ChEMBL/CHEMBL278/'  # the working folder that contains the data files
metric: str='tanimoto'  # the default distance metric to use on FPs
split: str='scaffold'  # which data split to evaluate, either 'scaffold' (default) or 'cv'
seed: int=2021  # random seed 
cv_fold: int=5  # number of folds in cross-validation
mean_anchor_percentage: float=0.6   # mean anchor point percentage for TR ensemble
std_anchor_percentage: float=0.2     # std of anchor point percentage for TR ensemble
num_TR_models = 30   # Number of TR models included in the ensemble
rbf_gamma: float=0.5    # gamma for RBF reconstruction
```

## Visualizations
The visualizations folder contains scripts to regenerate the figures in the main manuscript of the TR paper. 
### Nearest-Neighbor Prediction Visualization
To visualize the predictions made by TR compared to KNN and MLKR, run the 'visualize_NN_test_predictions.py' file with the following default input arguments
```
# Defaults for regenerating figures from manuscript
path: str = '../SampleDatasets/ChEMBL/CHEMBL2734/'  # the working folder that contains the data files
metric: str='tanimoto'  # the default distance metric to use on FPs
split: str='cv'  # which data split to evaluate, either 'scaffold' (default) or 'cv'
seed: int=2021  # random seed 
cv_fold: int=5  # number of folds in cross-validation
anchor_percentage: float=0.8    # Anchor point percentage
k: int=5             # k for k-NN prediction visualization
```
For example, running the following:
```
visualize_NN_test_predictions.py
```
will generate and save the following image to the visualization folder:
![NN Predictions](https://github.com/Ribosome25/TopoReg_QSAR/blob/main/visualizations/visualize_NN_predictions_CHEMBL2734.png)
### TR Lead Optimization Visualization
To visualize lead optimization pathways in the training predictions made by TR, run the 'visualize_TR_leadopt_pathways.py' file with the following default input arguments
```
  # Defaults for regenerating figures from manuscript
  path: str = '../SampleDatasets/ChEMBL/CHEMBL278/'   # the working folder that contains the data files
  metric: str='tanimoto'  # the default distance metric to use on FPs (ends with FP or FP4)
  split: str='cv'  # which data split to evaluate, either 'scaffold' (default) or 'cv'
  seed: int=2021  # random seed 
  cv_fold: int=5  # number of folds in cross-validation
  anchor_percentage: float=0.9   # Anchor point percentage
  k: int=5             # k for k-NN prediction visualization
```
For example, running the following:
```
visualize_TR_leadopt_pathways.py
```
Will generate and save the following image to the visualization folder:
![NN Predictions](https://github.com/Ribosome25/TopoReg_QSAR/blob/main/visualizations/lead_opt_pathway_CHEMBL278.png)
## Citation

