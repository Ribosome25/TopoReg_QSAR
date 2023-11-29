# Data Extraction

This works uses ChEMBL [1] data to illustrate the effectiveness of proposed method. After following the protocol proposed in [2] and filtering out the biotargets having less than 100 samples, we have 530 datasets for model benchmarking. A list of ChEMBL targets can be found in Large_scale SM2.xlsx from [2].

A demonstration of data preprocessing is provided in prepare_data_for_one_target.sh. Given working dir and target ID at the top, the data preprocessing goes through four steps:

1. Downloading. Data downloaded through `chembl_webresource_client` package, and saved in .tsv format. 
2. Preprocessing. Records without SMILES, without measurements, with duplicated indices, and with molecular weights > 1000 are dropped. Missing pChEMBL of inactive molecules are filled with minimum activity p target values. Data are saved as .txt format. 
3. Descriptor generation. In this demo, only ECFP4 is presented. ECFP4 are generated with RDKit's Morgan fingerprints. For TF3P, please refer to [Ribosome25/TF3P: Three-dimensional force fields fingerprints (github.com)](https://github.com/Ribosome25/TF3P) for the inference code, and the original authors for the pretrained model. 
4.  Converting data format. During the experiments, inconsistent data format and large csv files became a burden to file storage and computation. This script is for converting csv files to parquet files.

Besides the steps included in the .sh file, there are two functions included in this repo for user's reference: 

1) scaffold split. Scaffold split code is presented in `scripts/scaffold_split.py`. It relies on ChemProp, and can be copied to chemprop dir when using. 
2) batch download. Due to the high volunm of data, the experiments can be done in a paralleled way. `scripts/batch_download_chembl_data.py` provides an example on how the data are processed in batches. 




## Citation

[1] Mendez, D., Gaulton, A., Bento, A. P., Chambers, J., De Veij, M., Félix, E., Magariños, M. P., Mosquera, J. F., Mutowo, P., Nowotka, M., Gordillo-Marañón, M., Hunter, F., Junco, L., Mugumbate, G., Rodriguez-Lopez, M., Atkinson, F., Bosc, N., Radoux, C. J., Segura-Cabrera, A., Hersey, A., … Leach, A. R. (2019). ChEMBL: towards direct deposition of bioassay data. Nucleic acids research, 47(D1), D930–D940. https://doi.org/10.1093/nar/gky1075

[2] Lo, Yu-Chen, Silvia Senese, Chien-Ming Li, Qiyang Hu, Yong Huang, Robert Damoiseaux, and Jorge Z. Torres. "Large-scale chemical similarity networks for target profiling of compounds identified in cell-based chemical screens." *PLoS computational biology* 11, no. 3 (2015): e1004153.

