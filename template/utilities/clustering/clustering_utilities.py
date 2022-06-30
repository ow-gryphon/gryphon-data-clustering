from sklearn import cluster, metrics
import numpy as np


def standardize(frame, variables, type, suffix = None):
    '''
    Standardize numerical variables in dataset
    :param frame: pandas dataframe
    :param variables: list of numerical variables to standardize
    :param type: type of standardization: 'standardize', 'scale', 'demean', '[0,1]','[-1,1]
    :param suffix: suffix for new variables. If None, will use '_std','_scale', '_center', '_unit', '_unit2'
    :return: pandas dataframe with new columns
    '''

    new_columns = frame[variables]
    orig_names = new_columns.columns.values

    if type == "standardize":
        new_columns = (new_columns - new_columns.mean()) / new_columns.std()
        if suffix is None:
            proposed_names = ["{}_std".format(name) for name in new_columns.columns.values]
        else:
            proposed_names = ["{}{}".format(name, suffix) for name in new_columns.columns.values]
    elif type == "scale":
        new_columns = new_columns / new_columns.std()
        if suffix is None:
            proposed_names = ["{}_scale".format(name) for name in new_columns.columns.values]
        else:
            proposed_names = ["{}{}".format(name, suffix) for name in new_columns.columns.values]
    elif type == "demean":
        new_columns = new_columns - new_columns.mean()
        if suffix is None:
            proposed_names = ["{}_center".format(name) for name in new_columns.columns.values]
        else:
            proposed_names = ["{}{}".format(name, suffix) for name in new_columns.columns.values]
    elif type == "[0,1]":
        new_columns = (new_columns - new_columns.min()) / (new_columns.max() - new_columns.min())
        if suffix is None:
            proposed_names = ["{}_unit".format(name) for name in new_columns.columns.values]
        else:
            proposed_names = ["{}{}".format(name, suffix) for name in new_columns.columns.values]
    elif type == "[-1,1]":
        new_columns = 2*(new_columns - new_columns.min()) / (new_columns.max() - new_columns.min()) - 1
        if suffix is None:
            proposed_names = ["{}_unit2".format(name) for name in new_columns.columns.values]
        else:
            proposed_names = ["{}{}".format(name, suffix) for name in new_columns.columns.values]

    new_names = [generate_column_name(name, frame) for name in proposed_names]
    new_columns = new_columns.rename(columns=dict(zip(orig_names, new_names)))
    new_frame = pd.concat([frame, new_columns], axis='columns')

    return new_frame
    
    
    
def generate_column_name(proposed_name, frame):
    """
    Generate a variable name that is not already in the dataset

    :param proposed_name: string representing the proposed name of the variable
    :param frame: pandas dataframe for which the new variable name should be used
    :return: string with a non-duplicated and appropriate name for the dataframe
    """

    orig_name = proposed_name.strip().replace(' ', '_').replace('(', '').replace(')', '')
    proposed_name = orig_name
    num = 1
    while proposed_name in frame.columns.values:
        num = num + 1
        proposed_name = "{}_{}".format(orig_name, num)

    return proposed_name
    
    
    
def score_all(frame, predicted, arguments = {}, exhaustive = False):
    """
    Computes the cluster performance metrics on data without ground truths

    :param frame: A pandas Data Frame with x-values.
    :param predicted: Prediction series
    :param arguments: Dictionary containing the optional **kwargs for each test
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

    #if any(nan_index):
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
                    score['calinski_harabaz'] = metrics.calinski_harabaz_score(x_values, y_pred)
                else:
                    if nobs <= 10*size_threshold:
                        num_loop = np.floor(nobs / size_threshold)
                    else:
                        num_loop = 10
                    sil_score = 0
                    cal_score = 0
                    # Note: Silhouette score has a built in sampling mechanism, but Calinski does not. Therefore
                    #       manual dataset sampling is used

                    for loop in range(num_loop):
                        use_seed = (1+loop) * seed_multiplier # Set the seed value
                        random.seed(use_seed)
                        sampled_index = random.sample(range(nobs), size_threshold)

                        sil_score += metrics.silhouette_score(x_values.iloc[sampled_index,:], y_pred[sampled_index],
                                                              **this_kwargs)
                        cal_score += metrics.calinski_harabaz_score(x_values.iloc[sampled_index,:],
                                                                    y_pred[sampled_index])

                    score['silhouette'] = sil_score / num_loop
                    score['calinski_harabaz'] = cal_score / num_loop

            except:
                score['silhouette'] = "N/A"
                score['calinski_harabaz'] = "N/A"
        else:
            try:
                if exhaustive or nobs <= size_threshold:
                    score['silhouette'] = metrics.silhouette_score(x_values, y_pred, metric='euclidean')
                    score['calinski_harabaz'] = metrics.calinski_harabaz_score(x_values, y_pred)
                else:
                    if nobs <= 10 * size_threshold:
                        num_loop = np.floor(nobs / size_threshold)
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
                        cal_score += metrics.calinski_harabaz_score(x_values.iloc[sampled_index, :],
                                                                    y_pred[sampled_index])

                    score['silhouette'] = sil_score / num_loop
                    score['calinski_harabaz'] = cal_score / num_loop
            except:
                score['silhouette'] = "N/A"
                score['calinski_harabaz'] = "N/A"

    return score

def score_all_labeled(frame, predicted, target, arguments = {}):
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
        x_values = x_values.loc[~nan_index,:]
        y_pred = y_pred[~nan_index]
        y_act = y_act[~nan_index]

    # Check that there are multiple clusters in the data. This is not used for the standard metrics below
    if len(np.unique(y_pred)) == 1:
        multiple_cluster = False
    else:
        multiple_cluster = True

    score_label = {}
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