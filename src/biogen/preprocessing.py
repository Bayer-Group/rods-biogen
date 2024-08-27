"""This module contains preprocessing logic for evaluating synthetic data."""
import os
from pathlib import Path
from typing import Any, Dict, Union

import numpy as np
import pandas as pd
from datasets import ClassLabel, Dataset, DatasetDict
from datasets import Image as ImageFeature
from datasets import Value

from biogen.dataset_utils import cast_image, has_decoded_image


def get_image_paths(
    dataset: Union[DatasetDict, Dataset], col: str = "image"
) -> pd.DataFrame:
    """
    Return a :class:`DataFrame` containing the absolute paths of the images in the \
    :class:`DatasetDict`.
    """  # noqa: D200
    if has_decoded_image(dataset):
        dataset = cast_image(dataset, decode=False, name=col)
    if isinstance(dataset, DatasetDict):
        ds_splits = tuple(dataset.keys())
        data = [
            (img["path"], split) for split in ds_splits for img in dataset[split][col]
        ]
    else:
        data = [
            (img["path"], "") for img in dataset[col]
        ]  # No split represented by empty string.
    paths, splits = list(zip(*data))
    return pd.DataFrame(data=np.stack([paths, splits], 1), columns=["path", "split"])


def add_img_tooltip(df: pd.DataFrame, col: str = "image") -> pd.DataFrame:
    """
    Add a column named "img_tooltip" containing the appropriate path for displaying the original \
     images in the embedding plot tooltips.

    Calling this function allows to create the paths referenced to the current working directory.
    This is specially useful when working within a notebook, given that the tooltip image paths
    need to be referenced with respect to the current working directory.

    Args:
        df (pd.DataFrame): Target plot dataframe containing the absolute path of the images.
        col (str, optional): Column where the absolute path of the images is stored.
            Defaults to "image".

    Returns:
        pd.DataFrame: The provided dataset with an extra "img_tooltip" column that contains \
        the path required to display the images in the plot tooltip.
    """
    cwd = os.getcwd()
    df["img_tooltip"] = df[col].map(lambda x: os.path.relpath(x, start=cwd))
    return df


def split_to_df(
    dataset, split=None, name=None, only_values: bool = False
) -> pd.DataFrame:
    """Transform one split of a dataset into a DataFrame for data analysis."""
    data = {k: dataset[k] for k, v in dataset.features.items() if isinstance(v, Value)}
    df = pd.DataFrame(data).reset_index()
    for k, v in dataset.features.items():
        if isinstance(v, ClassLabel) and not only_values:
            df[k] = dataset[k]
            df[f"{k}_name"] = df[k].map(v.int2str)
        elif isinstance(v, ImageFeature):
            if has_decoded_image(dataset, name=k):
                dataset = cast_image(dataset, decode=False, name=k)
            df["path"] = [x["path"] for x in dataset[k]]
            df["image"] = df["path"].map(lambda x: Path(x).name)

    if only_values:
        return df.drop(columns=["index"])
    if split is not None:
        df["split"] = split
    if name is not None:
        df["dataset"] = name
    return df


def dataset_to_df(
    dataset: Union[Dataset, DatasetDict],
    name: str = None,
    only_values: bool = False,
) -> pd.DataFrame:
    """Transform the provided dataset into a DataFrame for performing data analysis."""
    if isinstance(dataset, Dataset):
        dataframes = [split_to_df(dataset, None, name, only_values)]
    else:
        dataframes = [
            split_to_df(dataset[s], s, name, only_values) for s in dataset.keys()
        ]
    return pd.concat(dataframes, axis=0).reset_index(drop=True)


def datasets_to_df(only_values: bool = False, **kwargs) -> pd.DataFrame:
    """Transform the provided datasets into a DatafRame for performing data analysis."""
    dataframes = pd.DataFrame()
    for name, dataset in kwargs.items():
        df = dataset_to_df(dataset, name=name, only_values=only_values)
        dataframes = pd.concat([dataframes, df], axis=0)
    return dataframes.reset_index(drop=True)


def scalar_dict_to_df(data_dict: Dict[str, Any]) -> pd.DataFrame:
    """
    Transform the scalar values found in the provided dictionary into a `class:pandas.DataFrame`.

    Args:
        data_dict (Dict[str, Any]): Dictionary containing the data that will be \
            transformed into a DataFrame.

    Returns:
        pd.DataFrame: DataFrame containing the scalar data contained in data_dict.
    """

    def _is_scalar_array(x):
        arr_types = (np.ndarray, list, tuple)
        return isinstance(x, arr_types) and len(x) and not isinstance(x[0], arr_types)

    return pd.DataFrame.from_dict(
        {k: v for k, v in data_dict.items() if _is_scalar_array(v)}
    )


def embedding_dict_to_df(
    data_dict: Dict[str, Any], suffix: str = "_embedding"
) -> pd.DataFrame:
    """
    Transform the embedding values found in the provided dictionary into a `class:pd.DataFrame`.

    Assumes the columns that will be transformed into the DataFrame are represented as keys
    finished with `_embedding`. For example: `pca_3d_embedding`.

    Args:
        data_dict (Dict[str, Any]): Dictionary containing the data that will be \
            transformed into a DataFrame.
        suffix (str): suffix indicating that a given key represents an embedding.

    Returns:
        pd.DataFrame: DataFrame containing the scalar data contained in data_dict.
    """

    def _is_vector_array(x):
        arr_types = (np.ndarray, list, tuple)
        return isinstance(x, arr_types) and len(x) and isinstance(x[0], arr_types)

    dims = ["x", "y", "z"]
    df = pd.DataFrame()
    for k, v in data_dict.items():
        if not k.endswith(suffix) or not _is_vector_array(v):
            continue  # Ignore keys that do not represent embeddings.
        emb_name = k.replace(suffix, "")
        emb_dim = v.shape[
            1
        ]  # Assumes embedding array with shape (n_samples, embedding_dim).
        # Add one column for embedding coordinate up to 3d. Columns are named:
        # emb-name_x, emb-name_y, emb-name_z.
        for i, dim_name in enumerate(dims[:emb_dim]):
            df[f"{emb_name}_{dim_name}"] = v[:, i]
    return df


def data_dict_to_df(data_dict: Dict[str, Any]) -> pd.DataFrame:
    """Transform a dictionary of scalar values and embeddings into a DaraFrame."""
    scalar_df = scalar_dict_to_df(data_dict)
    embedding_df = embedding_dict_to_df(data_dict)
    return pd.concat([scalar_df, embedding_df], axis=1).reset_index(drop=True)
