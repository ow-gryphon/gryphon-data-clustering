import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from pandas.plotting import parallel_coordinates
from sklearn.decomposition import PCA
from sklearn.manifold import MDS
import random
from sklearn import metrics


# PLOTS
def slice_1d(frame, x, cluster=None, max_points=1000, style="scatter"):
    """
    Slice multidimensional data along axes into 1 dimension for simple plotting purposes
    :param frame: A pandas DataFrame
    :param x: name of variable to be used for x-axis in the plot.  Y dimension set to 0
    :param cluster: name of variable with cluster assignments. If clusters are used, stratified sampling is provided
    :param max_points: integer representing the maximum number of observations to provide in the plot.
    If more observations exists, then apply the treatment defined by the large_style argument
    :param style: string representing the plot type of data. Values include:
    "scatter": Simple scatter plot
    "density": Summarize using joint density plot
    "hex": Summarize using hex plot
    "heatmap": Summarize plot as a heatmap
    :return: tuple with a plot, its axis, and the dataframe
    """

    # Check that variables exists in dataframe
    assert x in frame.columns, "{feat} not in data frame".format(feat=x)

    if cluster is not None:
        assert cluster in frame.columns, "{feat} not in data frame".format(feat=cluster)
        frame[cluster].fillna(-1, inplace=True)  # NA get their own cluster label: -1
        plot_data = frame[[x, cluster]].dropna(axis=0, how='any')
        if plot_data.shape[0] > max_points:
            plot_data = plot_data.sample(max_points)

    else:
        plot_data = frame[[x]].dropna(axis=0, how='any')
        cluster = 'cluster'  # Need to check the availability of the cluster name
        plot_data[cluster] = 1
        if plot_data.shape[0] > max_points:
            plot_data = plot_data.sample(max_points)  # To be updated to stratified sample

    if style == 'scatter':
        plot_data.insert(1, "One", (plot_data[cluster] * 0) + 1)
        fig, ax = plt.subplots(figsize=(10, 8))
        unique_clusters = plot_data[cluster].unique()

        frame_grouped_by_cluster = plot_data.groupby([cluster])

        for c_idx, cluster in enumerate(unique_clusters):
            c_cluster = frame_grouped_by_cluster.get_group(cluster)
            plt.scatter(c_cluster.iloc[:, 0], c_cluster["One"], label='Cluster {}'.format(c_idx))  # set y as 0

        ax.set_yticklabels([])
        plt.yticks([])

        x_axis_label = x
        plot_title = "1D Scatter"
        ax.set_xlabel(x_axis_label)
        ax.set_title(plot_title)

        return fig, ax, plot_data

    elif style == "density":
        plot = sns.jointplot(x=plot_data.reset_index().iloc[:, 1], y=plot_data.reset_index().iloc[:, 2],
                             kind='kde', stat_func=None)
        # Currently not able to extract data from jointplot, so using a sample dataset in return

    elif style == "hex":
        plot = sns.jointplot(x=plot_data.reset_index().iloc[:, 1], y=plot_data.reset_index().iloc[:, 2],
                             kind='hex', stat_func=None)
        # Currently not able to extract data from jointplot, so using a sample dataset in return
    else:
        raise ValueError(f"The style selected '{style}' for large scale plotting is not valid")

    return plot, plot_data


def slice_2d(frame, x, y, cluster=None, max_points=1000, style="scatter"):
    """
    Slice multidimensional data along axes into 2 dimensions for simple plotting purposes
    :param frame: A pandas DataFrame
    :param x: name of variable to be used for x-axis in the plot
    :param y: name of variable to be used for y-axis in the plot
    :param cluster: name of variable with cluster assignments. If clusters are used, stratified sampling is provided
    :param max_points: integer representing the maximum number of observations to provide in the plot.
    If more observations exists, then apply the treatment defined by the large_style argument
    :param style: string representing the plot type of data. Values include:
    "scatter": Simple scatter plot
    "density": Summarize using joint density plot
    "hex": Summarize using hex plot
    "heatmap": Summarize plot as a heatmap
    :return: tuple with a plot, its axis, and the dataframe
    """

    # Check that variables exists in dataframe
    assert x in frame.columns, "{feat} not in data frame".format(feat=x)
    assert y in frame.columns, "{feat} not in data frame".format(feat=y)

    if cluster is not None:
        assert cluster in frame.columns, "{feat} not in data frame".format(feat=cluster)
        frame[cluster].fillna(-1, inplace=True)  # NA get their own cluster label: -1
        plot_data = frame[[x, y, cluster]].dropna(axis=0, how='any')
        if plot_data.shape[0] > max_points:
            plot_data = plot_data.sample(max_points)

    else:
        plot_data = frame[[x, y]].dropna(axis=0, how='any')
        cluster = 'cluster'  # Need to check the availability of the cluster name
        plot_data[cluster] = 1
        if plot_data.shape[0] > max_points:
            plot_data = plot_data.sample(max_points)  # To be updated to stratified sample

    if style == 'scatter':
        # plot = sns.jointplot(x=plot_data.reset_index().iloc[:, 1], y=plot_data.reset_index().iloc[:, 2],
        #             kind='scatter', stat_func=None)

        fig, ax = plt.subplots(figsize=(10, 8))
        unique_clusters = plot_data[cluster].unique()

        frame_grouped_by_cluster = plot_data.groupby([cluster])

        for c_idx, cluster in enumerate(unique_clusters):
            c_cluster = frame_grouped_by_cluster.get_group(cluster)
            plt.scatter(c_cluster.iloc[:, 0], c_cluster.iloc[:, 1], label='Cluster {}'.format(c_idx))

        return fig, ax, plot_data

    elif style == "density":
        plot = sns.jointplot(x=plot_data.reset_index().iloc[:, 1], y=plot_data.reset_index().iloc[:, 2],
                             kind='kde', stat_func=None)
        # Currently not able to extract data from jointplot, so using a sample dataset in return
    elif style == "hex":
        plot = sns.jointplot(x=plot_data.reset_index().iloc[:, 1], y=plot_data.reset_index().iloc[:, 2],
                             kind='hex', stat_func=None)
        # Currently not able to extract data from jointplot, so using a sample dataset in return
    else:
        raise ValueError(f"The style selected '{style}' for large scale plotting is not valid")

    return plot, plot_data


def slice_3d(frame, x, y, z, cluster=None, max_points=1000, style="scatter"):
    """
    Slice multidimensional data along axes into 2 dimensions for simple plotting purposes
    :param frame: A pandas DataFrame
    :param x: name of variable to be used for x-axis in the plot
    :param y: name of variable to be used for y-axis in the plot
    :param z: name of variable to be used for z-axis in the plot
    :param cluster: name of variable containing the clusters, which should be positive integers
    :param max_points: integer representing the maximum number of observations to provide in the plot.
    If more observations exists, then apply the treatment defined by the large_style argument
    :param style: string representing the plot type of data. Values include:
    "scatter": Simple scatter plot
    "density": Summarize using joint density plot
    "hex": Summarize using hex plot
    "heatmap": Summarize plot as a heatmap
    :return: tuple with a plot, its axis, and the dataframe
    """

    # Check that variables exists in dataframe
    assert x in frame.columns, f"Column \"{x}\" not in data frame"
    assert y in frame.columns, f"Column \"{y}\" not in data frame"
    assert z in frame.columns, f"Column \"{z}\" not in data frame"

    if cluster is not None:
        assert cluster in frame.columns, f"Column \"{cluster}\" not in data frame"
        frame[cluster].fillna(-1, inplace=True)  # NA get their own cluster label: -1
        plot_data = frame[[x, y, z, cluster]].dropna(axis=0, how='any')
        if plot_data.shape[0] > max_points:
            plot_data = plot_data.sample(max_points)

    else:
        plot_data = frame[[x, y, z]].dropna(axis=0, how='any')
        cluster = 'cluster'  # Need to check the availability of the cluster name
        plot_data[cluster] = 1
        if plot_data.shape[0] > max_points:
            plot_data = plot_data.sample(max_points)  # To be updated to stratified sample

    fig = plt.figure(figsize=(15, 15))
    ax = None

    if style == "scatter":
        ax = fig.add_subplot(111, projection='3d')

        scatter = ax.scatter(plot_data.iloc[:, 0], plot_data.iloc[:, 1], plot_data.iloc[:, 2],
                             c=plot_data.iloc[:, 3], s=60, cmap="tab10")

        ax.legend(*scatter.legend_elements(prop="colors"), title="Cluster")

    elif style == "density":
        print("https://stackoverflow.com/questions/25286811/how-to-plot-a-3d-density-map-in-python-with-matplotlib")
    else:
        raise ValueError(f"The style selected '{style}' for large scale plotting is not valid")

    return fig, ax, plot_data


def plot_clusters_by_feature(frame, target_prediction, cols_to_exclude: list = None):
    """
    Creates a parallel plot using predicted cluster membership.

    :param frame: A Pandas DataFrame.
    :param target_prediction: Name of the predicted class membership column.
    :param cols_to_exclude: List or single value of columns to exclude.  Preferably one
        should exclude the original target column.
    :return: A matplotlib.pyplot object.
    """

    if cols_to_exclude is not None:
        trimmed_frame = frame[[col for col in frame.columns if col not in cols_to_exclude]]
    else:
        trimmed_frame = frame

    fig, ax = plt.subplots(figsize=(10, 8))
    parallel_coordinates(trimmed_frame, target_prediction)
    ax.set_ylabel('Feature Value')
    labels = ax.get_xticklabels()
    ax.set_xticklabels(labels, rotation=90)
    ax.set_title('Feature Value by Cluster and by Feature')

    return fig, ax, trimmed_frame


def clustering_scatter_plot(var_names, model_with_prediction, new_variable_name):
    """
        Returns temp_data_info, my_plot

        :return: dictionary of plot information for k-means, plot object
    """
    temp_data_info = dict()
    num_dim = len(var_names)

    # Call PCA plotting if more than 3 variables
    if num_dim > 3:
        my_plot, plot_data, x_axis_label, y_axis_label, z_axis_label, plot_title, full_data, res, total_var_explained \
            = plot_by_pc_by_cluster_top_three_dims(model_with_prediction, new_variable_name)
        temp_data_info["plot_type"] = "3D scatter"
        temp_data_info["res"] = res
        temp_data_info["total_var_explained"] = total_var_explained
        temp_data_info["axes_labels"] = {
            "x_axis_label": x_axis_label,
            "y_axis_label": y_axis_label,
            "z_axis_label": z_axis_label
        }

    # Call plotting based on number of variables
    else:
        if num_dim == 3:
            my_plot, ax, plot_data = slice_3d(model_with_prediction, var_names[0], var_names[1], var_names[2],
                                              cluster=new_variable_name, max_points=1000, style="scatter")

            temp_data_info["axes_labels"] = {
                "x_axis_label": var_names[0],
                "y_axis_label": var_names[1],
                "z_axis_label": var_names[2]
            }

        elif num_dim == 2:
            my_plot, ax, plot_data = slice_2d(model_with_prediction, var_names[0], var_names[1],
                                              cluster=new_variable_name, max_points=1000, style="scatter")

            temp_data_info["axes_labels"] = {
                "x_axis_label": var_names[0],
                "y_axis_label": var_names[1]
            }
        else:
            my_plot, ax, plot_data = slice_1d(model_with_prediction, var_names[0],
                                              cluster=new_variable_name, max_points=1000, style="scatter")
            temp_data_info["axes_labels"] = {"x_axis_label": var_names[0]}
        # Common attributes across 3 or less variables
        temp_data_info["plot_type"] = f"{num_dim}D scatter"

    return my_plot, temp_data_info


def factor_plot(data, new_variable_name, palette=None):
    """
    Returns factor plot

    :param data: data
    :param new_variable_name: name of column containing the predicted clusters

    :return: g, containing the factor_plot
    """
    df_long = pd.melt(data, new_variable_name, var_name="Variable Names", value_name="Value")

    g = sns.catplot(
        data=df_long, x="Variable Names", y="Value",
        kind="box", hue=new_variable_name,
        height=8, aspect=1.9, fliersize=1, sym='.',
        palette=palette, legend=False
    )
    plt.legend(title="Cluster number", loc='upper left')
    g.set_xticklabels(rotation=45, ha="right")

    return g


def elbow_plot(scores: pd.DataFrame):
    """
    Function that plots the results clustering scores versus the number of clusters used.
    Useful for determining the number of clusters for algorithms with a priori defined
    number of clusters (i.e. K-Means)

    :param scores: A pandas DataFrame where each of the columns represents some clustering score
        and each line is related to a certain number of clusters considered.
        The DataFrame should have also one mandatory column called n_clusters.

    :return: None
    """
    assert "n_clusters" in scores.columns, "Required column \"n_clusters\" is not present in the DataFrame"

    n_clusters = scores["n_clusters"]
    metrics = scores.drop(columns=["n_clusters"])

    fig, axis = plt.subplots(1, len(metrics.columns), figsize=(24, 7), sharex='all')
    fig.suptitle("Metrics vs Number of clusters")

    for m, ax in zip(metrics, axis):

        ax.plot(n_clusters, scores[m])
        ax.scatter(n_clusters, scores[m])

        ax.set_title(f"Metric: {m}")
        ax.set_ylabel("Metric score")
        ax.set_xlabel("Number of clusters")
        ax.set_xticks(n_clusters)
        ax.grid()

    return fig


# PCA PLOTS
def plot_by_pc_by_cluster_top_two_dims(frame, target_prediction, other_cols_to_exclude=None):
    """
    Performs PCA and graphs the predicted clusters over the top two dimensions.

    :param frame: A pandas DataFrame.  Should include all known classes and predictions.
    :param target_prediction:  The predicted class value.
    :param other_cols_to_exclude: Columns to exclude from the principal component calculation.
    :return: A Matplotlib.pyplot object.
    """

    n_components = 2

    if other_cols_to_exclude is not None:
        trimmed_frame = frame[[col for col in frame.columns if col not in other_cols_to_exclude]]
    else:
        trimmed_frame = frame

    if target_prediction is not None:
        pca_to_fit_df = trimmed_frame.drop(target_prediction, 1)
    else:
        pca_to_fit_df = trimmed_frame
    # if trimmed_frame.shape[1] < 2 (throw error (when porting to 3)) - TODO

    pca_to_fit = PCA(n_components=n_components)
    
    pca_frame = pd.DataFrame(pca_to_fit.fit_transform(pca_to_fit_df),
                             columns=['pc_dim_{}'.format(ix+1) for ix in range(n_components)])
    frame_with_pc = pd.concat([pca_frame, frame], axis=1)

    fig, ax, plot_data = slice_2d(frame_with_pc, "pc_dim_1", "pc_dim_2", cluster=target_prediction, style="scatter")

    x_axis_label = "PC1"
    y_axis_label = "PC2"
    plot_title = "First Two Principal Components"

    ax.set_xlabel(x_axis_label)
    ax.set_ylabel(y_axis_label)
    ax.set_title(plot_title)

    if target_prediction is None:
        full_output_data = frame_with_pc[["pc_dim_1", "pc_dim_2"]]
    else:
        full_output_data = frame_with_pc[["pc_dim_1", "pc_dim_2", target_prediction]]

    return fig, plot_data, x_axis_label, y_axis_label, plot_title, full_output_data


def plot_by_pc_by_cluster_top_three_dims(frame, target_prediction, other_cols_to_exclude: list = None):
    """
    Performs PCA and graphs the predicted clusters over the top three dimensions.

    :param frame: A pandas DataFrame.  Should include all known classes and predictions.
    :param target_prediction:  The predicted class value.
    :param other_cols_to_exclude: Columns to exclude from the principal component calculation.
    :return: A Matplotlib.pyplot object, along with several features
    """

    n_components = 3

    if other_cols_to_exclude is not None:
        trimmed_frame = frame[[col for col in frame.columns if col not in other_cols_to_exclude]]
    else:
        trimmed_frame = frame

    if target_prediction is not None:
        pca_to_fit_df = trimmed_frame.drop(columns=target_prediction)
    else:
        pca_to_fit_df = trimmed_frame

    if trimmed_frame.shape[1] <= 2:
        pass  # TODO (throw error)

    pca_to_fit = PCA(n_components=n_components)
    
    fitted_pca = pca_to_fit.fit(pca_to_fit_df)
    
    pca_frame = pd.DataFrame(
        fitted_pca.transform(pca_to_fit_df),
        columns=[
            f'pc_dim_{ix + 1}'
            for ix in range(n_components)
        ]
    )

    var_names = list(pca_to_fit_df)
    horiz_dim = len(var_names)
    vert_dim = horiz_dim
    # res = pd.DataFrame(pca_to_fit.transform(np.eye(horiz_dim, vert_dim)), index=var_names)
    res = pca_to_fit.transform(pd.DataFrame(np.eye(horiz_dim, vert_dim), columns=var_names))
    explained_var = pca_to_fit.explained_variance_ratio_
    total_var_explained = explained_var.sum()

    frame_with_pc = pd.concat([pca_frame, frame], axis=1)

    fig, ax, plot_data = slice_3d(frame_with_pc, "pc_dim_1", "pc_dim_2", "pc_dim_3", cluster=target_prediction,
                                  max_points=1000, style="scatter")

    x_axis_label = "PC1"
    y_axis_label = "PC2"
    z_axis_label = "PC3"
    plot_title = "First Three Principal Components"

    ax.set_xlabel(x_axis_label)
    ax.set_ylabel(y_axis_label)
    ax.set_zlabel(z_axis_label)
    ax.set_title(plot_title)

    if target_prediction is None:
        full_output_data = frame_with_pc[["pc_dim_1", "pc_dim_2", "pc_dim_3"]]
    else:
        full_output_data = frame_with_pc[["pc_dim_1", "pc_dim_2", "pc_dim_3", target_prediction]]

    return (fig, plot_data, x_axis_label, y_axis_label, z_axis_label, plot_title,
            full_output_data, res, total_var_explained)


def generate_mds(frame, features, n_components=2, dissimilarity="euclidean", verbose=1, **kwargs):
    """
    :param frame: A Pandas DataFrame.
    :param features: List of features names to use in the Multi-Dimensional Scaling algorithm
    :param n_components: Number of dimensions in which to immerse the dissimilarities
    :param dissimilarity:  ‘euclidean’ | ‘precomputed’
    :param verbose: Level of verbosity
    :param kwargs: other arguments to feed the MDS function

    :return: MDS_coordinates
    """

    mds_model = MDS(n_components=n_components, verbose=verbose, dissimilarity=dissimilarity, **kwargs)
    mds_coordinates = mds_model.fit_transform(frame[features].dropna())

    return mds_coordinates


def num_clusters_df(frame, cluster_var_name):
    """
    :param frame: Pandas dataframe with cluster assignments
    :param cluster_var_name: variable with clusters

    :return: 1D array with cluster numbers
    """
    df = frame[cluster_var_name].value_counts().reset_index()
    df.columns = [cluster_var_name, 'Count']

    return df




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
                    score['calinski_harabasz'] = metrics.calinski_harabasz_score(x_values, y_pred)

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
                    score['calinski_harabasz'] = cal_score / num_loop

            except Exception as e:
                raise e
                score['silhouette'] = np.nan
                score['calinski_harabasz'] = np.nan

        else:
            try:
                if exhaustive or nobs <= size_threshold:
                    score['silhouette'] = metrics.silhouette_score(x_values, y_pred, metric='euclidean')
                    score['calinski_harabasz'] = metrics.calinski_harabasz_score(x_values, y_pred)
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
                    score['calinski_harabasz'] = cal_score / num_loop

            except Exception as e:
                raise e
                score['silhouette'] = np.nan
                score['calinski_harabasz'] = np.nan

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