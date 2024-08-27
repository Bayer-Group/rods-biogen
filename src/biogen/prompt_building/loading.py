"""This moduel contains the functions to load and handle the datasets and data needed for
prompt_building"""

import logging
import os
from pathlib import Path
from typing import Union

import numpy as np
from datasets import ClassLabel, load_from_disk
from datasets.dataset_dict import DatasetDict

from biogen.embeddings import vectorize_datasets
from biogen.preprocessing import data_dict_to_df

logger = logging.getLogger(__name__)


def convert_to_ds_to_df(token_data_dict):
    """Convert HF dataset into a dataframe with some extra columns."""

    def correct_path_to_local(x):
        path = x
        start = os.getcwd()
        return os.path.relpath(path, start)

    # Convert HF dataset into a df
    token_dada_df = data_dict_to_df(data_dict=token_data_dict)
    # Correct paths with relative paths
    token_dada_df["filename"] = token_dada_df["path"].map(lambda x: x.split("/")[-1])
    token_dada_df["img_tooltip"] = token_dada_df["path"].map(correct_path_to_local)
    # Add column named path for later code to function - path coming from image dataset
    token_dada_df["dataset_target"] = (
        token_dada_df["dataset_name"] + "_" + token_dada_df["target_name"]
    )
    return token_dada_df


def add_dummy_label_column(ds_dict, label_col="label"):
    ds_list, split_list = [], []
    for split in ds_dict:
        ds = ds_dict[split]
        new_column = ["undefined"] * len(ds)
        ds = ds.add_column(label_col, new_column)
        ds_list.append(ds)
        split_list.append(split)
    ds_dict_new = DatasetDict({split: ds for split, ds in zip(split_list, ds_list)})
    return ds_dict_new


# TODO Probably should be functionality in codebase
def convert_label_to_ClassLabel(ds_dict, label_col="label"):
    ds_list, split_list = [], []
    for split in ds_dict:
        ds = ds_dict[split]
        # NOTE: Does not work for splits with only one label, but unlikely to be useful
        names = np.sort(np.unique(ds[label_col])).tolist()
        ds = ds.cast_column(label_col, ClassLabel(num_classes=len(names), names=names))
        ds_list.append(ds)
        split_list.append(split)
    ds_dict_new = DatasetDict({split: ds for split, ds in zip(split_list, ds_list)})
    return ds_dict_new


def load_data(
    ds_path: Union[Path, str],
    data_key: str = "class_token",
    label_key: Union[str, None] = "label",
    path_key: str = "path",
    class_names_dict=None,
):
    """
    Load the DiNO embedding dataset and convert it to a dataframe with relevant data.
    """
    # Load dataset
    logger.info("Loading the data ... ")
    token_ds_full = DatasetDict({"train": load_from_disk(ds_path)})
    logger.info(token_ds_full)

    # NOTE add dummy label if label not there
    # TODO in the future considering refactoring prompt building pipeline
    if label_key is None:
        token_ds_full = add_dummy_label_column(token_ds_full)
        label_key = "label"

    # Converting class label to ClassLabel format because vectorise_datasets expects that
    logger.info(token_ds_full["train"].features)
    if not isinstance(token_ds_full["train"].features[label_key], ClassLabel):
        token_ds_full = convert_label_to_ClassLabel(token_ds_full)
    logger.info(token_ds_full["train"].features)

    # Vectorising teh dataset so that we can get the raw token data
    dict_full = vectorize_datasets(
        data_key=data_key,
        label_key=label_key,
        path_key=path_key,
        original=token_ds_full,
    )
    # Convert this dictionary into a df with all importnat metadata and paths
    logger.info(f"after vectorising dataset {dict_full['data'].shape}")
    df_plot_full_ds = convert_to_ds_to_df(dict_full)
    logger.info(f"after df_plot_full_ds {df_plot_full_ds.shape}")
    logger.info(
        f"unique labels df_plot_full_ds {df_plot_full_ds['target_name'].unique()}"
    )
    # Rename target_names if class_names_dict is provided
    if class_names_dict is not None:
        df_plot_full_ds = update_label_names(df_plot_full_ds, class_names_dict)
    logger.info(
        f"unique labels df_plot_full_ds "
        f"after applying class dict {df_plot_full_ds['target_name'].unique()}"
    )
    logger.info(f"path: {df_plot_full_ds['path'][0]}")
    return df_plot_full_ds, dict_full


# TODO this is a temporary function to have proper names as target_names. needs a classname dict
# to match from boolean to actual names
def update_label_names(df_plot_full_ds, class_names_dict):
    df_plot_full_ds["target_name"] = df_plot_full_ds["target_name"].map(
        lambda x: class_names_dict[str(x)] if str(x) in class_names_dict else x
    )
    return df_plot_full_ds
