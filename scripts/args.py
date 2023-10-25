"""
TAP Arguments

"""
from tap import Tap


#%%  Pipeline Args
class ChemblPipelineArgs(Tap):
    path: str = 'SampleDatasets/ChEMBL/CHEMBL278/'  # the working folder that contains the target and descs files
    metric: str='tanimoto'  # the default distance metric to use on FPs (ends with FP or FP4)
    split: str='scaffold'  # which data split to evaluate, either 'scaffold' (default) or 'cv'
    seed: int=2021  # random seed 
    cv_fold: int=5  # number of folds in cross-validation
    k: float=0.5    # Anchor point percentage
    rbf_gamma: float=0.5    # gamma for RBF reconstruction
class ChemblPipelineArgsEnsemble(Tap):
    path: str = 'SampleDatasets/ChEMBL/CHEMBL278/'  # the working folder that contains the target and descs files
    metric: str='tanimoto'  # the default distance metric to use on FPs (ends with FP or FP4)
    split: str='cv'  # which data split to evaluate, either 'scaffold' (default) or 'cv'
    seed: int=2021  # random seed 
    cv_fold: int=5  # how many folds in cross-validation
    mean_k: float=0.6   # mean anchor point percentage for TR ensemble
    std_k: float=0.2     # std of anchor point percentage for TR ensemble
    num_TR_models = 30   # Number of TR models included in ensemble
    rbf_gamma: float=0.5    # gamma for RBF reconstruction
