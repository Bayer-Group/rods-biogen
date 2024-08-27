"""This module provides functionality for calculating embeddings on image latents."""
import logging
from typing import Any, Dict, List, Optional, Tuple

import einops
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import umap
from datasets import ClassLabel, DatasetDict, Image
from sklearn.decomposition import PCA

logger = logging.getLogger(__name__)


def softmax(x: np.ndarray) -> np.ndarray:
    """Calculate the softmax of the provided numpy array."""
    return np.exp(x) / np.exp(x).sum(1).reshape(-1, 1)


def _dataset_to_df(dataset: DatasetDict) -> pd.DataFrame:
    scalar_cols = {
        "label",
        "lpips_loss",
        "mae",
        "mse",
        "cross_entropy",
        "hinge_loss",
    }
    data_dict = {k: dataset[k] for k in dataset.features.keys() if k in scalar_cols}
    return pd.DataFrame(data_dict)


def processed_dataset_to_df(dataset: DatasetDict) -> pd.DataFrame:
    """Return a :class:`DataFrame` containing data from provided dataset."""
    df_list_concat = []
    for split in [
        x for x in dataset.keys()
    ]:  # iterate over existing splits and not hardcoded tr/te/val
        split_df = _dataset_to_df(dataset[split])
        df_list_concat.append(split_df)
    return pd.concat(df_list_concat)


def create_embedding_df(emb_data: Dict[str, Any]) -> pd.DataFrame:
    """
    Return a :class:`DataFrame` containing the embedding coordinates and relevant data provided.
    """  # noqa: D200
    cols = ["targets", "target_names", "split_names", "split_index"]
    data = {k: v for k, v in emb_data.items() if k in cols}
    for col in ["umap_embedding", "pca_embedding"]:
        name = col.split("_embedding")[0]
        for i, coord in enumerate(["x", "y"]):
            data[f"{name}_{coord}"] = emb_data[col][:, i]
    for col in ["umap_3d_embedding", "pca_3d_embedding"]:
        name = col.split("_embedding")[0]
        for i, coord in enumerate(["x", "y", "z"]):
            data[f"{name}_{coord}"] = emb_data[col][:, i]
    return pd.DataFrame(data)


def get_image_paths(dataset: DatasetDict, col: str = "image") -> pd.DataFrame:
    """
    Return a :class:`DataFrame` containing the absolute paths of the images in the \
    :class:`DatasetDict`.
    """  # noqa: D200
    ds_splits = tuple(dataset.keys())
    image_example = dataset[ds_splits[0]][0][col]
    if not isinstance(image_example, dict):  # Images already decoded as PIL images.
        dataset = dataset.cast_column(col, Image(decode=False))
    paths = [img["path"] for split in ds_splits for img in dataset[split][col]]
    return pd.Series(paths, name="image").to_frame()


def create_analysis_df(
    dataset: DatasetDict,
    latents_dataset: DatasetDict,
    emb_data: Dict[str, Any],
) -> pd.DataFrame:
    """
    Combine a dataset, its latent representations and its embedding analysis into a dataframe.
    """  # noqa: D200
    df_imgs = get_image_paths(dataset)
    df_emb = create_embedding_df(emb_data)
    df_proc = processed_dataset_to_df(latents_dataset)
    df = pd.concat([df_emb, df_proc.reset_index(), df_imgs], axis=1)
    return df


def prepare_dataset_split(
    dataset: DatasetDict,
    split: str = "train",
    data_key: str = "data",
    label_key: str = "label",
    path_key: str = None,
) -> Tuple[np.ndarray, List[str], List[str]]:
    """
    Return the data, targets, and target names contained in a :class:`DatasetDict` split \
    as numpy arrays.
    """  # noqa: D200
    dataset = (dataset[split] if split is not None else dataset).with_format("np")
    np_data = dataset[data_key]
    np_data = einops.rearrange(np_data, "b ... -> b (...)")
    if isinstance(dataset.features[label_key], ClassLabel):
        label_names = dataset.features[label_key].names
        logger.info(
            f"Fetaure {label_key} is a ClassLabel with unique values: {label_names}"
        )
    else:
        # then could be either a number or a string with the lable name already
        unique_labels = np.unique(dataset[label_key]).tolist()
        logger.info(
            f"Fetaure {label_key} is NOT a ClassLabel and has unique values: {label_names}"
        )
        if unique_labels[0].isnumeric():
            label_names = [str(x) for x in unique_labels]
    targets = dataset[label_key]
    target_names = [label_names[int(i)].capitalize() for i in targets]
    img_paths = dataset["path"] if path_key is not None else []
    return np_data, targets, target_names, img_paths


def prepare_dataset(
    dataset: DatasetDict,
    splits: Optional[str] = None,
    data_key: str = "data",
    label_key: str = "label",
    path_key: str = None,
    name: str = "",
) -> Tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray
]:
    """
    Concatenate the different splits of a :class:`DatasetDict` and return their \
    relevant data as numpy arrays.
    """  # noqa: D200
    if splits is None:
        splits = list(dataset.keys())
    else:
        splits = [splits] if isinstance(splits, str) else splits
    data, targets, target_names, split_names, indexes, names, paths = (
        [],
        [],
        [],
        [],
        [],
        [],
        [],
    )
    for split in splits:
        data_, targets_, target_names_, paths_ = prepare_dataset_split(
            dataset,
            split=split,
            data_key=data_key,
            label_key=label_key,
            path_key=path_key,
        )
        ds_len = len(targets_)
        split_names_ = [split] * ds_len
        names_ = [str(name)] * ds_len
        index_ = list(range(ds_len))
        data.append(data_)
        targets.append(targets_)
        target_names.extend(target_names_)
        split_names.extend(split_names_)
        indexes.extend(index_)
        names.extend(names_)
        paths.extend(paths_)
    data, targets = np.concatenate(data, 0), np.concatenate(targets, 0)
    return data, targets, target_names, split_names, indexes, names, paths


def vectorize_datasets(
    splits: Optional[str] = None,
    data_key: str = "data",
    label_key: str = "label",
    path_key: str = None,
    **kwargs: Dict[str, DatasetDict],
) -> Dict[str, np.ndarray]:
    """Transform one or more datasets into a vector-like format for data analysis."""
    data, targets, target_names, split_names, indexes, ds_names, paths = (
        [],
        [],
        [],
        [],
        [],
        [],
        [],
    )
    for ds_name, dataset in kwargs.items():
        (
            data_,
            targets_,
            target_names_,
            split_names_,
            indexes_,
            names_,
            paths_,
        ) = prepare_dataset(
            dataset=dataset,
            splits=splits,
            data_key=data_key,
            label_key=label_key,
            path_key=path_key,
            name=ds_name,
        )
        target_names.extend(target_names_)
        split_names.extend(split_names_)
        indexes.extend(indexes_)
        ds_names.extend(names_)
        paths.extend(paths_)
        data.append(data_)
        targets.append(targets_)

    data, targets = np.concatenate(data, 0), np.concatenate(targets, 0)
    data_dict = dict(
        data=data,
        target=targets,
        target_name=np.array(target_names),
        split_name=np.array(split_names),
        index=np.array(indexes),
        dataset_name=np.array(ds_names),
    )
    if path_key is not None:
        data_dict[path_key] = paths
    return data_dict


def calculate_umap_embedding(
    data, n_components: int = 2, random_state=160290, **kwargs
):
    """Create umap embeddings of the provided data."""
    umap_kwargs_2d = kwargs.copy()
    umap_kwargs_2d["n_epochs"] = umap_kwargs_2d.get("n_epochs", 100)
    umap_kwargs_2d["n_components"] = n_components
    umap_kwargs_2d["random_state"] = random_state
    umap_mapper = umap.UMAP(**umap_kwargs_2d).fit(data)
    umap_embedding = umap_mapper.transform(data)
    return umap_embedding, umap_mapper


def calculate_pca_embedding(data, n_components: int = 2, random_state=160290, **kwargs):
    """Create pca embeddings of the provided data."""
    kwargs = kwargs.copy()
    kwargs["n_components"] = n_components
    kwargs["random_state"] = random_state
    pca_mapper = PCA(**kwargs).fit(data)
    pca_embedding = pca_mapper.transform(data)
    return pca_embedding, pca_mapper


def run_embedding_analysis(
    splits: List[str] = None,
    data_key: str = "data",
    label_key: str = "label",
    path_key: str = None,
    n_components: int = 3,
    pca_kwargs: Optional[Dict[str, Any]] = None,
    umap_kwargs: Optional[Dict[str, Any]] = None,
    random_state: int = 160290,
    **datasets: Dict[str, DatasetDict],
) -> Dict[str, Any]:
    """Calculate the dbpac, umap and PCA embeddings of the target dataset of latents."""
    if not datasets:
        raise ValueError("You need to specify at least one dataset.")
    dataset_dict = vectorize_datasets(
        splits=splits,
        data_key=data_key,
        label_key=label_key,
        path_key=path_key,
        **datasets,
    )
    data = dataset_dict["data"]
    pca_kwargs = dict() if pca_kwargs is None else pca_kwargs
    pca_embedding, pca_mapper = calculate_pca_embedding(
        data,
        n_components=2,
        random_state=random_state,
        **pca_kwargs,
    )
    pca_3d_embedding, pca_3d_mapper = calculate_pca_embedding(
        data,
        n_components=n_components,
        random_state=random_state,
        **pca_kwargs,
    )

    umap_kwargs = dict() if umap_kwargs is None else umap_kwargs
    umap_embedding, umap_mapper = calculate_umap_embedding(
        data,
        n_components=2,
        random_state=random_state,
        **umap_kwargs,
    )
    umap_3d_embedding, umap_3d_mapper = calculate_umap_embedding(
        data,
        n_components=n_components,
        random_state=random_state,
        **umap_kwargs,
    )

    embedding_dict = dict(
        umap_mapper=umap_mapper,
        umap_embedding=umap_embedding,
        umap_3d_mapper=umap_3d_mapper,
        umap_3d_embedding=umap_3d_embedding,
        pca_mapper=pca_mapper,
        pca_embedding=pca_embedding,
        pca_3d_mapper=pca_3d_mapper,
        pca_3d_embedding=pca_3d_embedding,
    )
    data_dict = {**dataset_dict, **embedding_dict}
    return data_dict


def plot_embeddings(
    x_list: List[np.ndarray],
    y_list: List[np.ndarray],
    targets_arr_ls: List[int],
    targets_names_ls: List[str],
    title_str: str,
) -> None:
    """Plot a set of embeddings with corresponding labels and specified figure title.

    Args:
        x_list (List[np.ndarray]): List of the embeddings x coordinates
        y_list (List[np.ndarray]): List of the embeddings y coordinates
        targets_arr_ls (List[int]): List of the embeddings labels (int)
        targets_names_ls (List[str]): List of the embeddings labels (str)
        title_str (str): Title to be given to the plot
    """
    plt.figure()
    for x, y, targets_arr, targets_names in zip(
        x_list, y_list, targets_arr_ls, targets_names_ls
    ):
        label_list = list(dict.fromkeys(targets_arr))
        label_name_list = list(dict.fromkeys(targets_names))
        label_mapping = dict(zip(label_list, label_name_list))
        # e.g.{"0": "healthy", "1":"cancer"}
        for label in label_list:
            label_l_idx = targets_arr == label
            plt.scatter(
                x[label_l_idx], y[label_l_idx], label=label_mapping[label], alpha=0.5
            )
    plt.gca().set_aspect("equal", "datalim")
    plt.title(title_str, fontsize=24)
    plt.legend(loc="best", shadow=False, scatterpoints=1)


def plot_by_mapper(
    embd_type: str,
    mapper_dict: Dict[str, Any],
    list_dict: List[Dict[str, Any]],
    title_str: str,
) -> None:
    """Plot UMAP or PCA embeddings of data according to a specified mapper and embedding type.

    Args:
        embd_type (str): Embedding type. Only accepts PCA or UMAP.
        mapper_dict (Dict[str, Any]): Dictionary from run_embedding_analysis
        function holding the data from which the UMAP mapping is computed. Holds
        the mapper object that will be used to embed the other provided on the
        next argument.
        list_dict (List[Dict[str, Any]]): List of run_embedding_analysis
        dictionaries of the datasets that will be embedded according to
        specified mapper.
        title_str (str): Title to be given to the final plot.
    """
    # Check if input is correct
    if embd_type != "pca" and embd_type != "umap":
        raise ValueError("Invalid embedding type. Choose pca or umap.")
    embd_str = f"{embd_type}_embedding"
    embd_mapper_str = f"{embd_type}_mapper"
    # Get mapper and embeddings from base dictionary
    data_dict_0 = mapper_dict
    x_0 = data_dict_0[embd_str][:, 0]
    y_0 = data_dict_0[embd_str][:, 1]
    umap_mapper_0 = data_dict_0[embd_mapper_str]
    x_list = [x_0]
    y_list = [y_0]
    # Transform every other data into embeddings with base mapper
    for dict_i in list_dict:
        data_i = dict_i["data"]
        umap_embedding_of_synth_on_gt = umap_mapper_0.transform(data_i)
        x_list.append(umap_embedding_of_synth_on_gt[:, 0])
        y_list.append(umap_embedding_of_synth_on_gt[:, 1])
    targets_arr_list = [dicct["targets"] for dicct in [mapper_dict] + list_dict]
    targets_names_list = []
    for i, dicct in enumerate([mapper_dict] + list_dict):
        targets_names_i = dicct["target_names"]
        targets_names_i = [f"{x}_{i}" for x in targets_names_i]
        targets_names_list.append(targets_names_i)
    plot_embeddings(
        x_list, y_list, targets_arr_list, targets_names_list, title_str=title_str
    )


def umap_plot(data_df: pd.DataFrame) -> None:
    """Plot UMAP embeddings from data df computed with the create_analysis_df function."""
    x = data_df["umap_x"]
    y = data_df["umap_y"]

    targets_arr = data_df["targets"]  # 0, 0, 1, 1, 0, (...)
    targets_names = data_df["target_names"]  # healthy, healthy, cancer, (...)
    plot_embeddings(
        [x],
        [y],
        [targets_arr],
        [targets_names],
        title_str="UMAP projection of dataset latents",
    )


def pca_plot(data_df: pd.DataFrame) -> None:
    """Plot UMAP embeddings from data df computed with the create_analysis_df function."""
    x = data_df["pca_x"]
    y = data_df["pca_y"]

    targets_arr = data_df["targets"]  # 0, 0, 1, 1, 0, (...)
    targets_names = data_df["target_names"]  # healthy, healthy, cancer, (...)
    plot_embeddings(
        [x],
        [y],
        [targets_arr],
        [targets_names],
        title_str="PCA projection of dataset latents",
    )
