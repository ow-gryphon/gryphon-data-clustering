import time
from pathlib import Path

import pandas as pd


# EXPORT DATA
def to_excel(result_dict: dict, file_name: str = None, output_path=Path.cwd(), save_training_data=False):
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


def to_excel_range(results: dict, output_path=Path.cwd(), save_training_data=False):
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
