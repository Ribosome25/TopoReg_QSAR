import numpy as np
from sklearn.metrics import pairwise_distances

def simple_y_train(response, anchors_idx, metric, train_idx=None):
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
        