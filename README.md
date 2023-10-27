# TopoReg_QSAR

## Introduction
This code repository contains example code for the paper Topological Regression in Quantitative Structure-Activity Relationship Modeling. Quantitative structure-activity relationship (QSAR) modeling is a powerful tool used in drug discovery, yet the lack of interpretability of commonly used QSAR models hinders their application in molecular design. 
We proposed a similarity-based regression framework, topological regression (TR), that offers a statistically grounded, computationally fast, and interpretable technique. TR provides predictive performance that competes with deep-learning methods, such as Transformer-CNN, as well as informative visualizations, which can help guide lead optimization in drug discovery. 

In this package, we provide Python files for TR, Ensemble TR, k-Nearest-Neighbor Graph Visualization, and Lead optimization pathway visualization, as well as sample ChEMBL datasets and the code to extract all datasets used in the paper.


## Installation
Requirements:
Installation:


## Demo


## Citation


## Notes on model metrics

Spearman correlation coef. and NRMSE are used in this paper. NRMSE is defined as 
$$
NRMSE = \frac{\sqrt{MSE}} {std(y_{target})}
$$
,which provides a quick evaluation of model errors across different subdatasets. When model predicts mean for all samples, NRMSE = 1. 

In some part of the experiments, R^2 is used as metric too. Here, the R2 is defined in the same way as in scikit-learn (also PyTorch, Keras) that is generalized to model prediction cases, 
$$
R^2 = 1 - \frac{MSE}{Var(y_{target})}
$$
And the R2 value can be negative if MSE is even greater than predicting the mean. In our definitions, R2 and NRMSE can be converted to each other by 
$$
R^2 = 1 - NRMSE^2
$$
