# TopoReg_QSAR
Code repository for paper Topological Regression in Quantitative Structure-Activity Relationship Modeling





## Experimental steps

### Download data from ChEMBL and generate descriptors

`1_download_process_descgen.sh`. 

### Generate TF3P 3D fingerprints

`2_tf3p.sh`

Code and explanation for TF3P are available at [Ribosome25/TF3P: Three-dimensional force fields fingerprints (github.com)](https://github.com/Ribosome25/TF3P)   forked from [CanisW/TF3P: Three-dimensional force fields fingerprints (github.com)](https://github.com/CanisW/TF3P) and modifications were made to generate 3D FPs. Pretrained model is provided by the original authors on request. 

### Data format conversion

During experiments, csv format files became a burden to storage space and IO speed. All data, intermediate files were converted to parquet format. Checkout `3_convert_csv_to_parquet` to see more conversion details. 

### TR modeling

`chembl_pipeline_cv` and `chembl_pipeline_cv_disjoint` for TR and TR* respectively. Scripts are run as descripted in `4_modeling_pipeline.sh`. 



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
