"""This moduel contains the functions to process the loaded data, particularly
removing outliers."""
import logging
from typing import Dict, Tuple

import numpy as np
import pandas as pd

from biogen.prompt_building.feature_extraction.extract_features import (
    extract_hsv_features,
)

logger = logging.getLogger(__name__)


def set_remove_label(
    df,
    th_V_min=-np.inf,
    th_V_max=100,
    th_S_min=-np.inf,
    th_S_max=100,
    th_std_V=-np.inf,
    th_std_S=-np.inf,
    th_std_H=-np.inf,
    th_var_L=-np.inf,
    th_n_contour=-np.inf,
    th_v_s=[-np.inf, -np.inf],
):
    """Sets keep label to true to every sample whose mean V and S values are within the interval"""

    # TODO (guillemdb): Separate in two functions. One does the filtering and the
    # other creates the remove column for plotting purposes
    def filter_col(df, feat_str, th):
        # new col with new condidtioning
        df[f"remove_{feat_str}"] = df[feat_str].map(
            lambda x: "remove" if x < th else "keep"
        )
        # update remove column with new condition
        df["remove"] = df.apply(
            lambda x: "keep"
            if (x["remove"] == "keep" and x[f"remove_{feat_str}"] == "keep")
            else "remove",
            axis=1,
        )
        return df

    df["remove_V"] = df["mean_V"].map(
        lambda x: "remove" if x > th_V_min and x < th_V_max else "keep"
    )
    df["remove_S"] = df["mean_S"].map(
        lambda x: "remove" if x > th_S_min and x < th_S_max else "keep"
    )
    df["remove"] = df.apply(
        lambda x: "keep"
        if (x["remove_V"] == "keep" and x["remove_S"] == "keep")
        else "remove",
        axis=1,
    )
    df["remove_std_V"] = df["std_V"].map(lambda x: "remove" if x < th_std_V else "keep")
    df["remove_std_S"] = df["std_S"].map(lambda x: "remove" if x < th_std_S else "keep")
    df["remove_std"] = df.apply(
        lambda x: "keep"
        if (x["remove_std_V"] == "keep" and x["remove_std_S"] == "keep")
        else "remove",
        axis=1,
    )
    df["remove"] = df.apply(
        lambda x: "keep"
        if (x["remove"] == "keep" and x["remove_std"] == "keep")
        else "remove",
        axis=1,
    )
    df = filter_col(df, "std_H", th=th_std_H)
    df = filter_col(df, "var_L", th=th_var_L)
    df = filter_col(df, "n_contour", th=th_n_contour)
    df["remove_VandS"] = df.apply(
        lambda x: "remove"
        if (x["mean_V"] < th_v_s[0] and x["mean_S"] < th_v_s[1])
        else "keep",
        axis=1,
    )
    df["remove"] = df.apply(
        lambda x: "keep"
        if (x["remove"] == "keep" and x["remove_VandS"] == "keep")
        else "remove",
        axis=1,
    )

    return df


def preprocess_data(
    df_plot_full_ds: pd.DataFrame,
    dict_full: Dict,
    th_V_max=70,
    th_S_max=3,
    th_std_V=4,
    th_std_S=4,
    th_std_H=1.5,
    th_var_L=100,
    th_n_contour=10,
):
    """Extracts features from the image sin teh dataset and filters it them based on them.

    Args:
        df_plot_full_ds (pd.DataFrame): _description_
        dict_full (Dict): _description_
        th_V_max (int, optional): _description_. Defaults to 70.
        th_S_max (int, optional): _description_. Defaults to 3.
        th_std_V (int, optional): _description_. Defaults to 4.
        th_std_S (int, optional): _description_. Defaults to 4.
        th_std_H (float, optional): _description_. Defaults to 1.5.
        th_var_L (int, optional): _description_. Defaults to 100.
        th_n_contour (int, optional): _description_. Defaults to 10.
        logger (_type_, optional): _description_. Defaults to None.

    Returns:
        _type_: _description_
    """
    logger.info(" ... .... .... Reomving outliers from dataset ... .... ....")
    # Add features to the dataframe
    df_all = extract_hsv_features(df_plot_full_ds)
    logger.info(f"after extract_hsv_features {df_all.shape}")
    # Set remove label in extra column of dataset (in dataframe format)
    df_plt = set_remove_label(
        df=df_all,
        th_V_max=th_V_max,
        th_S_max=th_S_max,
        th_std_V=th_std_V,
        th_std_S=th_std_S,
        th_std_H=th_std_H,
        th_var_L=th_var_L,
        th_n_contour=th_n_contour,
    )
    logger.info(f"after set_remove_label {df_plt.shape}")
    # Create prompts
    # Get clean df and corresponding token set
    df_clean, ix_clean = prepare_dataset_for_classification(df_plt)
    cleaned_token_data = clean_token_data(dict_full, ix_remove=ix_clean)
    logger.info(f"after prepare_dataset_for_classification {df_clean.shape}")
    logger.info(f"remove {len(ix_clean)} samples")
    logger.info(f"after cleaned_token_data {cleaned_token_data.shape}")
    return cleaned_token_data, df_clean


def prepare_dataset_for_classification(
    df: pd.DataFrame,
    split_column: str = "split_name",
    remove_label_column: str = "remove",
    remove_label_value: str = "remove",
) -> Tuple[pd.DataFrame, pd.Index]:
    """
    Prepare a dataset for classification by filtering out outlier rows based on
    a specified remove label and sorting the remaining rows by index.

    Args:
        df (pd.DataFrame): The input dataframe.
        split_column (str, optional): The name of the column containing the split names.
            Defaults to "split_name".
        split_values (Tuple[str, str, str], optional): The split values in the order "train",
            "test", "validation". Defaults to ("train", "test", "validation").
        remove_label_column (str, optional): The name of the column containing the remove
            labels. Defaults to "remove".
        remove_label_value (str, optional): The value of the remove label for rows to be
            filtered out. Defaults to "remove".

    Returns:
        Tuple[pd.DataFrame, pd.Index]: A tuple containing the cleaned dataframe and \
        the indices of the removed rows.
    """
    logger.info(" .... .... prepare_dataset_for_classification .....")
    # Sorting samples within each split
    df_split_list = []
    split_values = df[split_column].unique()
    for split_name in split_values:
        df_split = df[df[split_column] == split_name].sort_values(by="index")
        df_split_list.append(df_split)
    df_split_sorted = pd.concat(df_split_list, ignore_index=True).reset_index(drop=True)
    # Filter out outlier rows based on remove label
    remove_idx = df_split_sorted[
        df_split_sorted[remove_label_column] == remove_label_value
    ].index
    df_clean = df_split_sorted[
        df_split_sorted[remove_label_column] != remove_label_value
    ].reset_index(drop=True)
    return df_clean, remove_idx


def clean_token_data(dict_full, ix_remove) -> np.ndarray:
    """
    Loads the token dataset and runs outlier removal on it using the set_remove_label function.
    Saves the cleaned token data as a numpy array.

    Args:
    - dataset_dir (str): The path to the directory containing the dataset.

    Returns:
    - np.ndarray: The cleaned token data as a numpy array.
    """
    full_token_data_clean = np.delete(dict_full["data"], ix_remove, axis=0)
    return full_token_data_clean
