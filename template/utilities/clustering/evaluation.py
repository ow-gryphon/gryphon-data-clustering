import random

import numpy as np
from sklearn import metrics


# noinspection PyDefaultArgument
def score_all(frame, predicted, arguments=dict(), exhaustive=False):
    """
    Computes the cluster performance metrics on data without ground truths

    :param frame: A pandas Data Frame with x-values.
    :param predicted: Prediction series
    :param arguments: Dictionary containing the optional **kwargs for each test
    :param exhaustive:

    :return: Dictionary of performance metrics
    """

    x_values = frame
    y_pred = np.array(predicted, dtype='f')
    nan_values = y_pred.copy()

    # Check for missing indices
    nan_values[nan_values == -1] = np.nan
    nan_index = np.isnan(nan_values)

    if all(nan_index):
        return {}

    # if any(nan_index):
    #    x_values = x_values.loc[~nan_index, :]
    #    y_pred = y_pred[~nan_index]

    # Check that there are multiple clusters in the data. This is not used for the standard metrics below
    num_clusters = len(np.unique(y_pred))
    if num_clusters == 1 or num_clusters == len(y_pred):
        multiple_cluster = False
    else:
        multiple_cluster = True

    score = {}

    # Silhouette score
    if multiple_cluster:
        nobs = len(y_pred)
        size_threshold = 5000
        seed_multiplier = 42
        if hasattr(arguments, 'silhouette'):
            this_kwargs = arguments.silhouette
            try:
                if exhaustive or nobs <= size_threshold:
                    score['silhouette'] = metrics.silhouette_score(x_values, y_pred, **this_kwargs)
                    score['calinski_harabaz'] = metrics.calinski_harabasz_score(x_values, y_pred)

                else:
                    if nobs <= 10 * size_threshold:
                        num_loop = int(np.floor(nobs / size_threshold))
                    else:
                        num_loop = 10
                    sil_score = 0
                    cal_score = 0
                    # Note: Silhouette score has a built in sampling mechanism, but Calinski does not. Therefore
                    #       manual dataset sampling is used

                    for loop in range(num_loop):
                        use_seed = (1 + loop) * seed_multiplier  # Set the seed value
                        random.seed(use_seed)
                        sampled_index = random.sample(range(nobs), size_threshold)

                        sil_score += metrics.silhouette_score(x_values.iloc[sampled_index, :], y_pred[sampled_index],
                                                              **this_kwargs)
                        cal_score += metrics.calinski_harabasz_score(x_values.iloc[sampled_index, :],
                                                                     y_pred[sampled_index])

                    score['silhouette'] = sil_score / num_loop
                    score['calinski_harabaz'] = cal_score / num_loop

            except Exception as e:
                raise e
                score['silhouette'] = np.nan
                score['calinski_harabaz'] = np.nan

        else:
            try:
                if exhaustive or nobs <= size_threshold:
                    score['silhouette'] = metrics.silhouette_score(x_values, y_pred, metric='euclidean')
                    score['calinski_harabaz'] = metrics.calinski_harabasz_score(x_values, y_pred)
                else:
                    if nobs <= 10 * size_threshold:
                        num_loop = int(np.floor(nobs / size_threshold))
                    else:
                        num_loop = 10
                    sil_score = 0
                    cal_score = 0
                    for loop in range(num_loop):
                        use_seed = (1 + loop) * seed_multiplier  # Set the seed value
                        random.seed(use_seed)
                        sampled_index = random.sample(range(nobs), size_threshold)

                        sil_score += metrics.silhouette_score(x_values.iloc[sampled_index, :],
                                                              y_pred[sampled_index],
                                                              metric='euclidean')
                        cal_score += metrics.calinski_harabasz_score(
                            x_values.iloc[sampled_index, :],
                            y_pred[sampled_index]
                        )

                    score['silhouette'] = sil_score / num_loop
                    score['calinski_harabaz'] = cal_score / num_loop

            except Exception as e:
                raise e
                score['silhouette'] = np.nan
                score['calinski_harabaz'] = np.nan

    return score


# noinspection PyDefaultArgument
def score_all_labeled(frame, predicted, target, arguments={}):
    """
    Computes the cluster performance metrics on data with ground truths

    :param frame: A pandas Data Frame.
    :param predicted: Prediction series
    :param target: Ground truth series
    :param arguments: Dictionary containing the optional **kwargs for each test

    :return: Dictionary of performance metrics
    """

    y_pred = np.array(predicted, dtype='f')
    y_act = np.array(target, dtype='f')

    # Check for missing indices
    y_pred[y_pred == -1] = np.nan
    nan_index = np.isnan(y_pred) | np.isnan(y_act)
    if all(nan_index):
        return {}

    if any(nan_index):
        x_values = x_values.loc[~nan_index, :]
        y_pred = y_pred[~nan_index]
        y_act = y_act[~nan_index]

    # Check that there are multiple clusters in the data. This is not used for the standard metrics below
    if len(np.unique(y_pred)) == 1:
        multiple_cluster = False
    else:
        multiple_cluster = True

    score_label = dict()
    # Adjusted Rand Score - no optional arguments
    score_label['adj_rand'] = metrics.adjusted_rand_score(y_act, y_pred)

    # Adjusted Information Score - no optional arguments
    score_label['adj_info'] = metrics.adjusted_mutual_info_score(y_act, y_pred)

    # Homogeneity score
    score_label['homogeneity'] = metrics.homogeneity_score(y_act, y_pred)

    # Completeness measure
    score_label['completeness'] = metrics.completeness_score(y_act, y_pred)

    # V-measure
    score_label['v_measure'] = metrics.v_measure_score(y_act, y_pred)

    # Fowlkes-Mallows score
    score_label['fowlkes_mallows'] = metrics.fowlkes_mallows_score(y_act, y_pred)

    return score_label
