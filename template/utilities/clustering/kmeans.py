# -*- coding: utf-8 -*-
from pathlib import Path
from typing import List

import pandas as pd
from sklearn.cluster import KMeans

from .utilities import evaluation
from .utilities import visualize_clusters
from .utilities import clustering_utilities
from .utilities import file_utilities
from .utilities import pre_processing


class KMeansClustering:

    @classmethod
    def execute_k_means(cls, data: pd.DataFrame, variables: List[str], num_clusters: int,
                        output_path=Path.cwd() / "outputs",
                        standardize_vars=False, generate_charts=True,
                        save_results_to_excel=False, export_charts=False,
                        save_training_data=False,
                        **kwargs):
        """
        Create a K-means model, fit it with data, and predict clusters on the data

        :param data: dataset to operate on
        :param variables: variables to operate on
        :param num_clusters: number of clusters for KMeans algorithm, default to 2
        :param standardize_vars: yes/no depending on standardization param
        :param generate_charts: Whether to generate plots or not
        :param save_results_to_excel: Whether to save the clustering information into an excel file
        :param output_path: Path used to save the outputs of the function
        :param save_training_data: Save training data into the excel. Only applicable when
        `save_results_to_excel` is set to True
        :param export_charts: Whether to save plots to disc or not

        :return: tuple of (fitted k-means-extension model with summary information saved, warning_info)
        """

        if standardize_vars:
            use_data = pre_processing.standardize_variables(data[variables])
        else:
            use_data = data[variables].dropna()

        # Create a model based on loaded_dataset (need col names to be string)
        k_means_model = KMeans(n_clusters=num_clusters, **kwargs)
        prediction = k_means_model.fit_predict(use_data.reset_index(drop=True))

        new_variable_name = "Cluster_assigned"
        use_data[new_variable_name] = prediction

        pd_centroids = pd.DataFrame(k_means_model.cluster_centers_)
        pd_centroids.columns = variables
        pd_centroids["Cluster Number"] = pd_centroids.index + 1

        # Set number of observations in each cluster
        cluster_n = visualize_clusters.num_clusters_df(use_data, new_variable_name)

        # Set train metrics
        scores = evaluation.score_all(use_data.drop(columns=new_variable_name), use_data[new_variable_name])

        output = {
            "model": k_means_model,
            "data": use_data,  # .reset_index(drop=True),
            "raw_data": use_data[["Cluster_assigned"]].join(data, how="outer"),
            "centroids": pd_centroids,
            "cluster_n": cluster_n,
            "scores": scores
        }

        if generate_charts:
            # Sending model prediction to get plot information
            cluster_plot, plot_info = visualize_clusters.clustering_scatter_plot(
                variables,
                use_data.reset_index(drop=True),
                new_variable_name
            )

            # create factor plot
            g = visualize_clusters.factor_plot(use_data, new_variable_name)

            output['cluster_plot'] = cluster_plot
            output['cluster_plot_info'] = plot_info
            output['factor_plot'] = g

            if save_results_to_excel:
                clustering_utilities.to_excel(output, output_path=output_path, save_training_data=save_training_data)

            if export_charts:
                file_utilities.export_plot(output['cluster_plot'], prefix="clusters", output_path=output_path)
                file_utilities.export_plot(output['factor_plot'], prefix="factors", output_path=output_path)

        return output

    @classmethod
    def k_means_range(cls, dataset: pd.DataFrame, variables: List[str],
                      min_clusters=2, max_clusters=5, output_path: Path = Path.cwd() / "outputs",
                      standardize_vars=False, generate_charts=True,
                      save_results_to_excel=False, export_charts=False,
                      save_training_data=False,
                      **kwargs):
        """
        Create a K-means model, fit it with data, and predict clusters on the data

        :param dataset: dataset to operate on
        :param variables: variables to operate on
        :param min_clusters: minimum number of clusters for KMeans algorithm, default to 2
        :param max_clusters: maximum number of clusters for KMeans algorithm, default to 5
        :param standardize_vars: yes/no depending on standardization param
        :param generate_charts: Whether to generate plots or not
        :param save_results_to_excel: Whether to save the clustering information into an excel file
        :param output_path: Path used to save the outputs of the function
        :param save_training_data: Save training data into the excel. Only applicable when
        `save_results_to_excel` is set to True
        :param export_charts: Whether to save plots to disc or not

        :return: Warning_info for any errors in the run, and also saves a .json with summary information
        """

        all_models = {}

        for num_clusters in range(min_clusters, max_clusters + 1):

            print(f"Working on {num_clusters} clusters")

            k_means_model_outputs = cls.execute_k_means(
                dataset, variables, num_clusters,
                standardize_vars=standardize_vars,
                generate_charts=generate_charts,
                output_path=output_path,
                save_results_to_excel=False,
                export_charts=False,
                **kwargs
            )
            all_models[num_clusters] = k_means_model_outputs

        if save_results_to_excel:
            clustering_utilities.to_excel_range(
                all_models,
                output_path=output_path,
                save_training_data=save_training_data
            )

        if generate_charts:
            all_scores = []

            for cluster_num, result in all_models.items():
                scores = result["scores"]
                all_scores.append(scores)

            elbow_plot = visualize_clusters.elbow_plot(scores=pd.DataFrame(all_scores))

            if export_charts:

                clustering_utilities.export_plot(elbow_plot, prefix="metrics", output_path=output_path)
                for n_clusters, output in all_models.items():

                    clustering_utilities.export_plot(
                        output['cluster_plot'],
                        prefix=f"clusters_{n_clusters}",
                        output_path=output_path
                    )

                    clustering_utilities.export_plot(
                        output['factor_plot'],
                        prefix=f"factors_{n_clusters}",
                        output_path=output_path
                    )

        return all_models
