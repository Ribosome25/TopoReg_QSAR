# """
# TAP Arguments

# """
# from tap import Tap


# #%%  Pipeline Args
# class ChemblPipelineArgs(Tap):
#     path: str = 'SampleDatasets/ChEMBL/CHEMBL278/'  # the working folder that contains the target and descs files
#     metric: str='tanimoto'  # the default distance metric to use on FPs (ends with FP or FP4)
#     split: str='scaffold'  # which data split to evaluate, either 'scaffold' (default) or 'cv'
#     seed: int=2021  # random seed 
#     cv_fold: int=5  # number of folds in cross-validation
#     anchor_percentage: float=0.5    # Anchor point percentage
#     rbf_gamma: float=0.5    # gamma for RBF reconstruction
# class ChemblPipelineArgsEnsemble(Tap):
#     path: str = 'SampleDatasets/ChEMBL/CHEMBL278/'  # the working folder that contains the target and descs files
#     metric: str='tanimoto'  # the default distance metric to use on FPs (ends with FP or FP4)
#     split: str='scaffold'  # which data split to evaluate, either 'scaffold' (default) or 'cv'
#     seed: int=2021  # random seed 
#     cv_fold: int=5  # how many folds in cross-validation
#     mean_anchor_percentage: float=0.6   # mean anchor point percentage for TR ensemble
#     std_anchor_percentage: float=0.2     # std of anchor point percentage for TR ensemble
#     num_TR_models = 30   # Number of TR models included in ensemble
#     rbf_gamma: float=0.5    # gamma for RBF reconstruction
# class ChemblNNVisualizationArgs(Tap):
#     # Defaults for regenerating figures from manuscript
#     path: str = '../SampleDatasets/ChEMBL/CHEMBL2734/'  # the working folder that contains the target and descs files
#     metric: str='tanimoto'  # the default distance metric to use on FPs (ends with FP or FP4)
#     split: str='cv'  # which data split to evaluate, either 'scaffold' (default) or 'cv'
#     seed: int=2021  # random seed 
#     cv_fold: int=5  # number of folds in cross-validation
#     anchor_percentage: float=0.8    # Anchor point percentage
#     k: int=5             # k for k-NN prediction visualization
# class ChemblLeadOptVisualizationArgs(Tap):
#     # Defaults for regenerating figures from manuscript
#     path: str = '../SampleDatasets/ChEMBL/CHEMBL278/'  # the working folder that contains the target and descs files
#     metric: str='tanimoto'  # the default distance metric to use on FPs (ends with FP or FP4)
#     split: str='cv'  # which data split to evaluate, either 'scaffold' (default) or 'cv'
#     seed: int=2021  # random seed 
#     cv_fold: int=5  # number of folds in cross-validation
#     anchor_percentage: float=0.9   # Anchor point percentage
#     k: int=5 # k for k-NN prediction visualization

import argparse


#%%  Pipeline Args
class ChemblPipelineArgs():
    def parse_args():
        parser = argparse.ArgumentParser(description='TR_pipeline.py')
        parser.add_argument('path', type=str, default='SampleDatasets/ChEMBL/CHEMBL278/') # the working folder that contains the data files
        parser.add_argument('metric', type=str, default='tanimoto')  # the default distance metric to use on FPs
        parser.add_argument('split', type=str, default='scaffold')  # which data split to evaluate, either 'scaffold' or 'cv'
        parser.add_argument('seed', type=int, default=2021) # random seed 
        parser.add_argument('cv_fold', type=int, default=5) # number of folds in cross-validation
        parser.add_argument('anchor_percentage', type=float, default=0.5)   # Anchor point percentage
        parser.add_argument('rbf_gamma', type=float, default=0.5)   # gamma for RBF reconstruction
        return parser
class ChemblPipelineArgsEnsemble():
    def parse_args():
        parser = argparse.ArgumentParser(description='Ensemble_TR_pipeline.py')
        parser.add_argument('path', type=str, default='SampleDatasets/ChEMBL/CHEMBL278/') # the working folder that contains the data files
        parser.add_argument('metric', type=str, default='tanimoto')  # the default distance metric to use on FPs
        parser.add_argument('split', type=str, default='scaffold')  # which data split to evaluate, either 'scaffold' or 'cv'
        parser.add_argument('seed', type=int, default=2021) # random seed 
        parser.add_argument('cv_fold', type=int, default=5) # number of folds in cross-validation
        parser.add_argument('rbf_gamma', type=float, default=0.5)   # gamma for RBF reconstruction
        parser.add_argument('mean_anchor_percentage', type=float, default=0.6)   # mean anchor point percentage for TR ensemble
        parser.add_argument('std_anchor_percentage', type=float, default=0.2)    # std of anchor point percentage for TR ensemble
        parser.add_argument('num_TR_models', type=int, default= 30 )  # Number of TR models included in ensemble
        return parser
class ChemblNNVisualizationArgs():
    def parse_args():
        parser = argparse.ArgumentParser(description='visualize_NN_test_predictions.py')
        # Defaults for regenerating figures from manuscript
        parser.add_argument('path', type=str, default='../SampleDatasets/ChEMBL/CHEMBL2734/') # the working folder that contains the data files
        parser.add_argument('metric', type=str, default='tanimoto')  # the default distance metric to use on FPs
        parser.add_argument('split', type=str, default='cv')  # which data split to evaluate, either 'scaffold' or 'cv'
        parser.add_argument('seed', type=int, default=2021) # random seed 
        parser.add_argument('cv_fold', type=int, default=5) # number of folds in cross-validation
        parser.add_argument('anchor_percentage', type=float, default=0.8)   # gamma for RBF reconstruction
        parser.add_argument('k', type=int,default=5)             # k for k-NN prediction visualization
        return parser
class ChemblLeadOptVisualizationArgs():
    def parse_args():
        parser = argparse.ArgumentParser(description='TR_leadopt_pathways.py')
        # Defaults for regenerating figures from manuscript
        parser.add_argument('path', type=str, default='../SampleDatasets/ChEMBL/CHEMBL278/') # the working folder that contains the data files
        parser.add_argument('metric', type=str, default='tanimoto')  # the default distance metric to use on FPs
        parser.add_argument('split', type=str, default='cv')  # which data split to evaluate, either 'scaffold' or 'cv'
        parser.add_argument('seed', type=int, default=2021) # random seed 
        parser.add_argument('cv_fold', type=int, default=5) # number of folds in cross-validation
        parser.add_argument('anchor_percentage', type=float, default=0.9)   # gamma for RBF reconstruction
        parser.add_argument('k', type=int,default=5)             # k for k-NN prediction visualization
