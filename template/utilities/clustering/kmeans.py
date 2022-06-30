# -*- coding: utf-8 -*-
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn import preprocessing
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from .clustering_utilities import standardize, generate_column_name, score_all
from . import visualize_clusters


def execute_KMeans(dataset, variables, num_clusters, standardize_vars = False, generate_charts = True, **kwargs):
    """
    Create a Kmeans model, fit it with data, and predict clusters on the data

    :param work_dir: working directory path
    :param data_dir: data directory path
    :param dataset_name: name of dataset to operate on
    :param var_names: variables to operate on
    :param num_clusters: number of clusters for KMeans algorithm, default to 2
    :param timestamp: date and time stamp for the json file to be saved. When using the tool, the Node.js server will
    generate this timestamp and will expect the output json file to be named accordingly
    :param run_name name of run
    :param data_file_type: file type of the underlying data
    :param standardize_vars: yes/no depending on standardization param
    :param save_xlsx: yes/no depending on if you want to save the xlsx
    :param status_file: saves status of run
    :param warning_info: warnings that occur in run
    :param save_png: flag to turn off if desired during unit testing

    :return: tuple of (fitted kmeans-extension model with summary information saved, warning_info)
    """

    if standardize_vars:
        scalar = StandardScaler().fit(dataset[variables])
        use_data = pd.DataFrame(scalar.transform(dataset[variables].dropna()), columns=variables)
    else: 
        use_data = dataset[variables].dropna()
        
    # Create a model based on loaded_dataset (need col names to be string)
    kmeans_model = KMeans(n_clusters=num_clusters, **kwargs)

    # Fit the model
    fit_model = kmeans_model.fit(use_data)

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
        g = visualize_clusters.get_factorplot(use_data, new_variable_name)
        
        output['cluster_plot'] = cluster_plot
        output['cluster_plot_info'] = plot_info
        output['factor_plot'] = g
    
    return output




def KMeans_range(dataset, variables,  min_clusters = 2, max_clusters = 5, standardize_vars = False, generate_charts = True, **kwargs):
    """
    Create a Kmeans model, fit it with data, and predict clusters on the data

    :param work_dir: working directory path
    :param data_dir: data directory path
    :param dataset_name: name of dataset to operate on
    :param var_names: variables to operate on
    :param min_clusters: minimum number of clusters for KMeans algorithm, default to 2
    :param max_clusters: maximum number of clusters for KMeans algorithm, default to 5
    :param timestamp: date and time stamp for the json file to be saved. When using the tool, the Node.js server will
    generate this timestamp and will expect the output json file to be named accordingly
    :param run_name: Name of the run
    :param data_file_type: file type of the underlying data
    :param standardize_vars: yes/no depending on standardization param
    :param save_xlsx: yes/no depending on if you want to save the xlsx
    :param status_file: saves status of run
    :param warning_info: warnings that occur in run
    :param only_return: Only return model object and not export into JSON
    :return: Warning_info for any errors in the run, and also saves a .json with summary information
    """

    all_models = {}
    
    for num_clusters in range(min_clusters, max_clusters + 1):
        
        print("Working on {} clusters".format(num_clusters))
    
        kmeans_model_outputs = execute_KMeans(dataset, variables, num_clusters, standardize_vars = False, generate_charts = True, **kwargs)
        all_models[num_clusters] = kmeans_model_outputs
        
    return all_models