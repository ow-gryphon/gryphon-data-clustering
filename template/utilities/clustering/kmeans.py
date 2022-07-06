# -*- coding: utf-8 -*-
import time
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from . import visualize_clusters
from .clustering_utilities import generate_column_name, score_all


class KMeansClustering:

    @classmethod
    def execute_k_means(cls, dataset, variables, num_clusters,
                        standardize_vars=False, generate_charts=True, save_results_to_excel=False,
                        output_path=Path.cwd() / "outputs", export_charts=False,
                        **kwargs):
        """
        Create a K-means model, fit it with data, and predict clusters on the data

        :param dataset: dataset to operate on
        :param variables: variables to operate on
        :param num_clusters: number of clusters for KMeans algorithm, default to 2
        :param standardize_vars: yes/no depending on standardization param
        :param generate_charts: Whether to generate plots or not
        :param save_results_to_excel: Whether to save the clustering information into an excel file
        :param output_path: Path used to save the outputs of the function

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

            if save_results_to_excel:
                cls.to_excel(output, output_path=output_path)

            if export_charts:
                cls.export_plot(output['cluster_plot'], prefix="clusters", output_path=output_path)
                cls.export_plot(output['factor_plot'], prefix="factors", output_path=output_path)

        return output

    @classmethod
    def k_means_range(cls, dataset: pd.DataFrame, variables: List[str],
                      min_clusters=2, max_clusters=5, standardize_vars=False, generate_charts=True,
                      output_path: Path = Path.cwd() / "outputs",
                      save_results_to_excel=False, export_charts=False, **kwargs):
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
            cls.to_excel_range(all_models)

        if generate_charts:
            all_scores = []

            for cluster_num, result in all_models.items():
                scores = result["scores"]
                all_scores.append(scores)

            elbow_plot = visualize_clusters.get_elbow_plot(scores=pd.DataFrame(all_scores))

            if export_charts:
                cls.export_plot(elbow_plot, prefix="metrics", output_path=output_path)
                for n_clusters, output in all_models.items():
                    cls.export_plot(output['cluster_plot'], prefix=f"clusters_{n_clusters}", output_path=output_path)
                    cls.export_plot(output['factor_plot'], prefix=f"factors_{n_clusters}", output_path=output_path)

        return all_models

    @staticmethod
    def export_plot(figure: plt.figure, prefix="figure", output_path=Path.cwd(), **kwargs):
        timestamp = time.strftime("%Y%m%d_%Hh%Mm%Ss", time.localtime())
        file_name = f"{prefix}_{timestamp}.png"

        figure.savefig(
            output_path / file_name,
            **kwargs
        )

    @staticmethod
    def to_excel(result_dict: dict, file_name: str = None, save_training_data=False, output_path=Path.cwd()):
        timestamp = time.strftime("%Y%m%d_%Hh%Mm%Ss", time.localtime())

        if file_name is None:
            file_name = f"clustering_result_{timestamp}.xlsx"

        data = result_dict["data"]
        centroids = result_dict["centroids"]
        scores = pd.DataFrame([result_dict["scores"]])

        with pd.ExcelWriter(output_path / file_name) as writer:
            if save_training_data:
                data.to_excel(writer, sheet_name='clustered_data', index=False)

            centroids.to_excel(writer, sheet_name='centroids', index=False)
            scores.to_excel(writer, sheet_name='scores', index=False)

    @classmethod
    def to_excel_range(cls, results: dict, save_training_data=False, output_path=Path.cwd()):

        timestamp = time.strftime("%Y%m%d_%Hh%Mm%Ss", time.localtime())
        file_name = f"multiple_clustering_result_{timestamp}.xlsx"

        with pd.ExcelWriter(output_path / file_name) as writer:

            all_scores = []
            for cluster_num, result in results.items():
                data = result["data"]
                centroids = result["centroids"]
                scores = result["scores"]
                scores["n_clusters"] = cluster_num

                all_scores.append(scores)

                if save_training_data:
                    data.to_excel(writer, sheet_name=f'clustered_data_{cluster_num}_clusters', index=False)

                centroids.to_excel(writer, sheet_name=f'centroids_{cluster_num}_clusters', index=False)

            pd.DataFrame(all_scores).to_excel(writer, sheet_name='scores', index=False)
