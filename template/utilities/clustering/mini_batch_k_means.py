# -*- coding: utf-8 -*-
import logging
from pathlib import Path

import pandas as pd
from sklearn.cluster import MiniBatchKMeans

from . import pre_processing
from . import evaluation
from . import clustering_utilities
from . import visualize_clusters

logger = logging.getLogger(__name__)


class MiniBatchKMeansClustering:

    @staticmethod
    def execute_mini_batch_k_means(
            data, variables, num_clusters=2,
            output_path=Path.cwd() / "outputs",
            standardize_vars=False, generate_charts=True,
            save_results_to_excel=False, export_charts=False,
            **kwargs):
        """
        Create a K means model using mini batches for training, fit it with data, and predict clusters on the data

        :param data: dataset to operate on
        :param variables: variables to operate on
        :param num_clusters: number of clusters for KMeans algorithm, default to 2
        :param standardize_vars: yes/no depending on standardization param
        :param generate_charts: Whether to generate plots or not
        :param save_results_to_excel: Whether to save the clustering information into an excel file
        :param output_path: Path used to save the outputs of the function
        :param export_charts: Whether to save plots to disc or not

        :return: tuple of (fitted k means-extension model with summary information saved, warning_info)
        """

        if standardize_vars:
            dataset = pre_processing.standardize_variables(data[variables])
        else:
            dataset = data[variables]

        k_means_model = MiniBatchKMeans(n_clusters=num_clusters, **kwargs)
        # TODO: random_state=get_random_state()

        # Fit the model
        prediction = k_means_model.fit_predict(dataset)
        new_variable_name = clustering_utilities.generate_column_name("Cluster_assigned", dataset)
        dataset[new_variable_name] = prediction

        output = {
            "model": k_means_model,
            "data": dataset,  # .reset_index(drop=True),
            "raw_data": dataset[["Cluster_assigned"]].join(data, how="outer")
        }

        if generate_charts:
            # Sending model prediction to get plot information
            cluster_plot, plot_info = visualize_clusters.clustering_scatter_plot(variables, dataset, new_variable_name)

            # create factor plot
            factor_plot = visualize_clusters.factor_plot(dataset, new_variable_name)

            output['cluster_plot'] = cluster_plot
            output['cluster_plot_info'] = plot_info
            output['factor_plot'] = factor_plot

            if export_charts:
                clustering_utilities.export_plot(cluster_plot, prefix="clusters", output_path=output_path)
                clustering_utilities.export_plot(factor_plot, prefix="factors", output_path=output_path)

            # close out plt to save memory
            # plt.close('all')

        # Set centroids
        pd_centroids = pd.DataFrame(k_means_model.cluster_centers_, columns=variables)
        pd_centroids["Cluster Number"] = pd_centroids.index

        cluster_n = visualize_clusters.num_clusters_df(dataset, new_variable_name)
        scores = evaluation.score_all(dataset.drop(columns=[new_variable_name]), prediction)

        output.update({
            'centroids': pd_centroids,
            'cluster_n': cluster_n,
            'scores': scores
        })

        if save_results_to_excel:
            clustering_utilities.to_excel(output, output_path=output_path)

        return output

    @classmethod
    def mini_batch_k_means_range(
            cls, data, variables, min_clusters=2, max_clusters=5,
            output_path=Path.cwd() / "outputs",
            standardize_vars=False, generate_charts=True, save_results_to_excel=False, export_charts=False,
            **kwargs):
        """
        Create a MiniBatchKMeans model, fit it with data, and predict clusters on the data

        :param data: dataset to operate on
        :param variables: variables to operate on
        :param min_clusters: minimum number of clusters for MiniBatchKMeans algorithm, default to 2
        :param max_clusters: maximum number of clusters for MiniBatchKMeans algorithm, default to 5
        :param standardize_vars: yes/no depending on standardization param
        :param generate_charts: Whether to generate plots or not
        :param save_results_to_excel: Whether to save the clustering information into an excel file
        :param output_path: Path used to save the outputs of the function
        :param export_charts: Whether to save plots to disc or not

        :return: Warning_info for any errors in the run, and also saves a .json with summary information
        """

        output_models = dict()

        for num_clusters in range(min_clusters, max_clusters + 1):
            print(f"Working on {num_clusters} clusters")

            k_means_model = cls.execute_mini_batch_k_means(
                data, variables, num_clusters,
                output_path=output_path,
                standardize_vars=standardize_vars,
                generate_charts=generate_charts,
                save_results_to_excel=False,
                export_charts=False,
                **kwargs
            )

            if k_means_model is not None:
                output_models[num_clusters] = k_means_model

        if export_charts:
            pass

        if save_results_to_excel:
            clustering_utilities.to_excel_range(output_models, output_path=output_path)

        return output_models
