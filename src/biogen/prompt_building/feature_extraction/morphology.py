import logging
from typing import Any, Dict, List, Union

import numpy as np
import pandas as pd
from natsort import natsorted
from s_dbw import SD, S_Dbw
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

SEED = 18021999
logger = logging.getLogger(__name__)


def cluster_labels(
    n_cluster: int,
    df_clean: pd.DataFrame,
    full_token_data_clean: np.ndarray,
    batch_size: Union[int, None] = None,
) -> pd.DataFrame:
    """
    Performs SINGLE KMeans clustering on the input data and storing the resulting
    labels in a new column in the dataframe.

    Args:
        n_cluster: An integer specifying the number of clusters to use in
            the KMeans clustering.
        df_clean: A pandas DataFrame containing the cleaned data with added features.
        full_token_data_clean: A numpy array containing the cleaned token data.

    Returns:
        The input dataframe with added column for cluster labels from KMeans clustering.

    """
    if batch_size is None:
        logger.info("Using all cpus: n_jobs=-1: Kmeans")
        kmean_clustering_model_token = KMeans(
            n_clusters=n_cluster,
            random_state=SEED,
            verbose=1,
            n_init=10,
        )
    else:
        logger.info(f"MiniBatch K means - Batch size: {batch_size}")
        kmean_clustering_model_token = MiniBatchKMeans(
            n_clusters=n_cluster,
            batch_size=batch_size,
            random_state=SEED,
            verbose=1,
            n_init=10,
            max_no_improvement=None,
            tol=1e-4,
            max_iter=300,
        )
    labels = kmean_clustering_model_token.fit_predict(full_token_data_clean)

    df_clean[f"cluster_lbl_n{n_cluster}"] = labels

    return df_clean, kmean_clustering_model_token


def get_cluster_metrics(data: np.ndarray, labels: List[int]) -> Dict[str, Any]:
    """
    Calculate the S_Dbw and S_D clustering validation metrics for a given set of cluster labels.

    Args:
        data (np.ndarray): A 2D array containing the data to cluster.
        labels (List[int]): A list of cluster labels for the given data.
        verbose (bool, optional): If True, print the calculated metrics. Defaults to False.

    Returns:
        Dict[str, Any]: A dictionary containing the calculated clustering metrics.
    """
    sdbw_score = S_Dbw(
        data,
        labels,
        centers_id=None,
        method="Tong",
        alg_noise="bind",
        centr="mean",
        nearest_centr=True,
        metric="euclidean",
    )
    sd_score = SD(
        data,
        labels,
        centers_id=None,
        alg_noise="bind",
        centr="mean",
        nearest_centr=True,
        metric="euclidean",
    )
    metric_dict = {
        "S_Dbw": sdbw_score,
        "S_D": sd_score,
    }
    return metric_dict


def get_internal_cluster_validation(
    df_clean_morpho: pd.DataFrame, full_token_data_clean: np.ndarray
) -> pd.DataFrame:
    """
    Calculates the internal validation metrics for each clustering in `df_clean_morpho`.

    Parameters:
        df_clean_morpho (pd.DataFrame): Dataframe with multiple clustering labels.
        full_token_data_clean (np.ndarray): Cleaned array of tokens to use for clustering.

    Returns:
        pd.DataFrame: Dataframe with internal validation metrics for each clustering.
    """
    df_internal_cluster_validation = pd.DataFrame(
        columns=["S_Dbw", "S_D", "clustering_method"]
    )

    key_list = natsorted([x for x in df_clean_morpho.keys() if "cluster_lbl_n" in x])

    for key in tqdm(key_list):
        metric_dict = get_cluster_metrics(
            data=full_token_data_clean,
            labels=df_clean_morpho[key].tolist(),
        )
        metric_dict["clustering_method"] = [key]

        df_row = pd.DataFrame.from_dict(metric_dict)
        df_internal_cluster_validation = pd.concat(
            [df_internal_cluster_validation, df_row], ignore_index=True
        )
    df_internal_cluster_validation["n"] = df_internal_cluster_validation[
        "clustering_method"
    ].map(
        lambda x: int(x.split("lbl_n")[-1]),
    )
    df_internal_cluster_validation.set_index("n", inplace=True)

    return df_internal_cluster_validation


def get_internal_clustering_validation_metrics(
    df_clean_morpho,
    cleaned_token_data,
):
    """Take df with labels for different number of clusters and compute metrics."""
    df_internal_cluster_validation = get_internal_cluster_validation(
        df_clean_morpho=df_clean_morpho,
        full_token_data_clean=cleaned_token_data,
    )
    logger.info(
        f"after get_internal_cluster_validation {df_internal_cluster_validation.shape}"
    )

    optimal_n_c_S_Dbw, optimal_n_c_S_D = (
        df_internal_cluster_validation["S_Dbw"].idxmin(),
        df_internal_cluster_validation["S_D"].idxmin(),
    )
    logger.info(f" best n_clusters (S_Dbw) {optimal_n_c_S_Dbw.shape}")
    logger.info(f" best n_clusters (S_D) {optimal_n_c_S_D.shape}")

    return df_internal_cluster_validation, optimal_n_c_S_Dbw, optimal_n_c_S_D


def generate_multiple_cluster_labels(
    n_cluster: int, df_clean: pd.DataFrame, full_token_data_clean: np.ndarray
) -> pd.DataFrame:
    """
    Performs multiple KMeans clustering on the input data, each time incrementing \
    the number of clusters by 1 and storing the resulting labels in a new column in the dataframe.

    Args:
        n_cluster: An integer specifying the maximum number of clusters to use in
            the KMeans clustering.
        df_clean: A pandas DataFrame containing the cleaned data with added features.
        full_token_data_clean: A numpy array containing the cleaned token data.

    Returns:
        The input dataframe with added columns for cluster labels from KMeans clustering.

    """
    for k in range(n_cluster - 3):
        scaler = StandardScaler()
        tokens_scaled = scaler.fit_transform(full_token_data_clean)
        kmean_clustering_model_token = KMeans(n_clusters=3 + k, random_state=SEED)
        labels = kmean_clustering_model_token.fit_predict(tokens_scaled)
        df_clean[f"cluster_lbl_n{3+k}"] = labels

    return df_clean
