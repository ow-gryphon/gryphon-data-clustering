# -*- coding: utf-8 -*-
from typing import List

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from . import visualize_clusters
from .clustering_utilities import generate_column_name, score_all


def execute_k_means(dataset, variables, num_clusters, standardize_vars=False, generate_charts=True, **kwargs):
    """
    Create a K-means model, fit it with data, and predict clusters on the data

    :param dataset: dataset to operate on
    :param variables: variables to operate on
    :param num_clusters: number of clusters for KMeans algorithm, default to 2
    :param standardize_vars: yes/no depending on standardization param
    :param generate_charts: Whether to generate plots or not

    :return: tuple of (fitted k-means-extension model with summary information saved, warning_info)
    """

    if standardize_vars:
        scalar = StandardScaler().fit(dataset[variables])
        use_data = pd.DataFrame(scalar.transform(dataset[variables].dropna()), columns=variables)

    else:
        use_data = dataset[variables].dropna()

    # Create a model based on loaded_dataset (need col names to be string)
    k_means_model = KMeans(n_clusters=num_clusters, **kwargs)

    # Fit the model
    fit_model = k_means_model.fit(use_data)

    # Predict inline
    new_variable_name = generate_column_name("Cluster_assigned", use_data)

    prediction = fit_model.predict(use_data.reset_index(drop=True))
    use_data[new_variable_name] = prediction

    pd_centroids = pd.DataFrame(fit_model.cluster_centers_)
    pd_centroids.columns = variables
    pd_centroids["Cluster Number"] = pd_centroids.index + 1

    # Set number of observations in each cluster
    cluster_n = visualize_clusters.num_clusters_df(use_data, new_variable_name)

    # Set train metrics
    scores = score_all(use_data.drop(columns=new_variable_name), use_data[new_variable_name])

    output = {
        "model": fit_model,
        "data": use_data,
        "centroids": pd_centroids,
        "cluster_n": cluster_n,
        "scores": scores
    }

    if generate_charts:
        # Sending model prediction to get plot information
        cluster_plot, plot_info = visualize_clusters.tool_plot(variables, use_data, new_variable_name)

        # create factor plot
        g = visualize_clusters.get_factor_plot(use_data, new_variable_name)

        output['cluster_plot'] = cluster_plot
        output['cluster_plot_info'] = plot_info
        output['factor_plot'] = g

    return output


def k_means_range(dataset: pd.DataFrame, variables: List[str],
                  min_clusters=2, max_clusters=5, standardize_vars=False, generate_charts=True, **kwargs):
    """
    Create a K-means model, fit it with data, and predict clusters on the data

    :param dataset: dataset to operate on
    :param variables: variables to operate on
    :param min_clusters: minimum number of clusters for KMeans algorithm, default to 2
    :param max_clusters: maximum number of clusters for KMeans algorithm, default to 5
    :param standardize_vars: yes/no depending on standardization param
    :param generate_charts: Whether to generate plots or not

    :return: Warning_info for any errors in the run, and also saves a .json with summary information
    """

    all_models = {}
    
    for num_clusters in range(min_clusters, max_clusters + 1):
        
        print(f"Working on {num_clusters} clusters")
    
        k_means_model_outputs = execute_k_means(
            dataset, variables, num_clusters,
            standardize_vars=standardize_vars,
            generate_charts=generate_charts,
            **kwargs
        )
        all_models[num_clusters] = k_means_model_outputs
        
    return all_models
