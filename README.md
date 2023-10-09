# TopoReg_QSAR
Code repository for paper Topological Regression in Quantitative Structure-Activity Relationship Modeling





## Experimental steps

### Download data from ChEMBL and generate descriptors

### Generate TF3P 3D fingerprints

`2_tf3p.sh`

Code and explanation available at [Ribosome25/TF3P: Three-dimensional force fields fingerprints (github.com)](https://github.com/Ribosome25/TF3P)   forked from [CanisW/TF3P: Three-dimensional force fields fingerprints (github.com)](https://github.com/CanisW/TF3P) and modifications are made to generate 3D FPs. Pretrained model is provided by the original authors on request. 

### Data format conversion

During experiments, csv format files became a burden to storage space and IO speed. All data, intermediate files were converted to parquet format. Checkout `3_convert_csv_to_parquet` to see more conversion details. 

