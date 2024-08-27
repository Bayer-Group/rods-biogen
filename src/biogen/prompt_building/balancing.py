"""This moduel contains the functions to subset a prompted dataset to ensure a uniform distribution
 of samples per prompt and also splits teh final dataset."""

import logging
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from biogen.prompt_building.feature_extraction.extract_features import (
    get_feature_counts,
)

logger = logging.getLogger(__name__)
SEED = 18021999


def get_prompt_counts_dfs(
    df_prompts: pd.DataFrame,
    inv_text_2_promptidx: dict,
    morpho_lbl_key: str,
    cluste_lbl: str = "prompt_idx",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Get prompt counts DataFrames for cancer and healthy prompts seperately."""
    lbl_names = [str(x) for x in np.unique(df_prompts["target_name"])]
    df = get_feature_counts(df_prompts, cluste_lbl=cluste_lbl)  # df_prompt_counts
    df["perc"] = df["#samples"] * 100 / np.sum(df["#samples"])

    df["target_name"] = df["prompt_idx"].map(
        lambda x: (
            lbl_names[0]
            if f" {lbl_names[0].lower()} " in inv_text_2_promptidx[x].lower()
            else lbl_names[1]
        ),
    )
    df[morpho_lbl_key] = df["prompt_idx"].map(
        lambda x: inv_text_2_promptidx[x].split("gy type ")[-1].split(",")[0],
    )

    logger.info(f"lbl_names = {lbl_names}")
    logger.info(f"df['target_name'].unique() = {df['target_name'].unique()}")
    prompt_counts_df_c = df[df["target_name"] == lbl_names[0]].reset_index(drop=True)
    prompt_counts_df_h = df[df["target_name"] == lbl_names[1]].reset_index(drop=True)
    return prompt_counts_df_c, prompt_counts_df_h


def check_max_n_prompts_to_be_used(
    df,
    n_classes: int = 2,
    n_samples_train: int = 50_000,
    n_samples_test: int = 5000,
    n_samples_val: int = 1000,
):
    """
    Iterates over prompts from the ones with highest sample count to the ones with the least.
    For each n, from 0 to (n_prompts-1), checks whether selecting the n first most
    populated prompts (from this class) would allow creating a prompt balanced subset
    of only these prompts for a user specified dataset size (n_samples_train + n_samples_test
    + n_samples_val). If from the set of n selected prompts, the prompt with minimum
    number of samples has less samples than the minimum required for a balanced set of user
    specified size, then we cannot use this set of prompts. Note in column "select_n_maj_prompts?"
    whether the n first most populated prompts match criteria.

    Args:
        df (pandas.DataFrame): A DataFrame with columns "#samples" and "perc" containing the
            sample counts per prompt and their percentage of the total, respectively.

        n_classes (int, optional): The number of classes in the dataset. Defaults to 2.

        n_samples_train (int, optional): The number of training samples desired. Defaults to 50000.

        n_samples_test (int, optional): The number of testing samples desired. Defaults to 5000.

        n_samples_val (int, optional): The number of validation samples desired. Defaults to 1000.

    Returns:
        pandas.DataFrame: The input DataFrame with two extra columns:
        - "n_needed_per_prmpt": The number of samples needed in each prompt at minimum to fill all
        splits in a balanced way.
        - "select_n_maj_prompts?": A boolean indicating whether or not to select the n most populated
            prompts for the desired dataset size.

    """
    # df is prompt_counts_df
    df["perc"] = df["#samples"] / np.sum(df["#samples"])
    df = df.sort_values(by="#samples", ascending=False).reset_index(drop=True)
    for n_first in range(len(df)):
        # Compute total number of smaple sneeded in each prompt at minimum to fill all splits
        n_needed_per_prmpt = (
            (n_samples_train * (1 / n_classes) / (n_first + 1))
            + (n_samples_test * (1 / n_classes) / (n_first + 1))
            + (n_samples_val * (1 / n_classes) / (n_first + 1))
        )
        # Get number of sample sin the minority prompt
        n_min = int(df.iloc[n_first]["#samples"])
        df.at[n_first, "n_needed_per_prmpt"] = n_needed_per_prmpt
        df.at[n_first, "n_min"] = n_min
    df["select_n_maj_prompts?"] = df.apply(
        lambda x: True if x["n_needed_per_prmpt"] < x["n_min"] else False,
        axis=1,
    )
    return df


def get_last_common_consider_index(df1, df2, n_minimum_of_prompts_per_class=5):
    """Find the index of the last row where the "select_n_maj_prompts?" column value is
    True in both input dataframes. Outputs the number of majority prompts
    we should (i.e. prompts with largest sample counts) that we should select
    for our final balanced subset.

    Args:
        df1 (pandas.DataFrame): The first input dataframe.
        df2 (pandas.DataFrame): The second input dataframe.

    Returns:
        int: The index of the last row where the "select_n_maj_prompts?" column value is
        True in both input dataframes.

    Raises:
        ValueError: If there is no common index with column value True.
    """
    # Get subset of df1 with only True rows
    consider_rows1 = df1[df1["select_n_maj_prompts?"] == True]

    # Get subset of df2 with only True rows
    consider_rows2 = df2[df2["select_n_maj_prompts?"] == True]

    # Get a list of the indexes of the True rows in both dataframes
    indexes1 = consider_rows1.index.tolist()
    indexes2 = consider_rows2.index.tolist()

    # Find the largest common index in the list of indexes of each dataframe subset
    common_indexes = list(set(indexes1) & set(indexes2))
    if common_indexes:
        n = max(common_indexes) + 1
        if n <= n_minimum_of_prompts_per_class:
            logger.info(
                f"There are commono indexes but their value is too low, i.e. < {n_minimum_of_prompts_per_class}"
            )
            raise ValueError(
                f" Largest common index is too small ( < {n_minimum_of_prompts_per_class})."
                f" Would result in too lottle number of prompts in final dataset."
            )
        else:
            return n
    else:
        raise ValueError(" There is no common index with column value True.")


def get_subset_based_on_prompt_indexes(
    n_needed_per_prmpt,
    prompt_ix_c,
    prompt_ix_h,
    df_prompts_c,
    df_prompts_h,
):
    """
    This function takes the subset of the full dataset that we want to use for training and
    returns a subset of that subset that is balanced in terms of cancer and healthy samples.

    Args:
        n_needed_per_prmpt (_type_): _description_
        prompt_idx_to_consider_c (_type_): _description_
        prompt_idx_to_consider_h (_type_): _description_
        df_prompts_5k_subset_to_consider_c (_type_): _description_
        df_prompts_5k_subset_to_consider_h (_type_): _description_

    Returns:
        _type_: _description_
    """
    df_subset_c_list = []
    for prompt_idx in prompt_ix_c:
        # get df with only one cancer prompt. df_prompts_5k_subset_to_consider_c_prompt_idx_i
        df_prompts_c_prompt_idx = df_prompts_c[df_prompts_c["prompt_idx"] == prompt_idx]
        df_prompts_c_prompt_idx = df_prompts_c_prompt_idx.reset_index(drop=True)
        # sample n_needed_per_prmpt sample from prompt
        if n_needed_per_prmpt <= len(df_prompts_c_prompt_idx):
            df_subset_prompt_idx_lbl = df_prompts_c_prompt_idx.sample(
                n=n_needed_per_prmpt,
                random_state=18021999,
            )
        else:
            logger.info(
                "Number of samples for this prompt "
                + f"idx {prompt_idx} is {len(df_prompts_c_prompt_idx)}"
                + f"\n Number of samples we want to sample from it is {n_needed_per_prmpt}"
            )
            df_subset_prompt_idx_lbl = df_prompts_c_prompt_idx.copy()
        df_subset_c_list.append(df_subset_prompt_idx_lbl)
    df_subset_h_list = []
    for prompt_idx in prompt_ix_h:
        # sample n_needed_per_prmpt sample per prompt.
        # df_prompts_5k_subset_to_consider_h_prompt_idx_i
        df_prompts_h_idx = df_prompts_h[df_prompts_h["prompt_idx"] == prompt_idx]
        df_prompts_h_idx = df_prompts_h_idx.reset_index(drop=True)
        if n_needed_per_prmpt <= len(df_prompts_h_idx):
            df_subset_prompt_idx_lbl = df_prompts_h_idx.sample(
                n=n_needed_per_prmpt,
                random_state=18021999,
            )
        else:
            df_subset_prompt_idx_lbl = df_prompts_h_idx.copy()

        df_subset_h_list.append(df_subset_prompt_idx_lbl)
    df_subset_h = pd.concat(df_subset_h_list, axis=0).reset_index(drop=True)
    df_subset_c = pd.concat(df_subset_c_list, axis=0).reset_index(drop=True)
    df_subset = pd.concat([df_subset_h, df_subset_c], axis=0).reset_index(drop=True)
    return df_subset


def get_subset_and_counts_dfs(
    df_prompts: pd.DataFrame,
    prompt_counts_df_c: pd.DataFrame,
    prompt_counts_df_h: pd.DataFrame,
    n_prompts: int,  # n_selected_prompts_per_class
    morpho_lbl_key: str,
) -> pd.DataFrame:
    """Get a subset of prompts and plot the distributions."""
    # prompt_ix_to_consider_c
    prompt_ix_c = prompt_counts_df_c["prompt_idx"].to_list()[:n_prompts]
    prompt_idx_to_consider_h = prompt_counts_df_h["prompt_idx"].to_list()[:n_prompts]
    # df_prompts_5k_subset_to_consider_c
    df_prompts_c = df_prompts[df_prompts["prompt_idx"].isin(prompt_ix_c)]
    # df_prompts_5k_subset_to_consider_h
    df_prompts_h = df_prompts[df_prompts["prompt_idx"].isin(prompt_idx_to_consider_h)]
    n_needed_per_prmpt = int(
        np.ceil(prompt_counts_df_c.at[n_prompts - 1, "n_needed_per_prmpt"])
    )
    df_subset = get_subset_based_on_prompt_indexes(
        n_needed_per_prmpt,
        prompt_ix_c,
        prompt_idx_to_consider_h,
        df_prompts_c,
        df_prompts_h,
    )
    morpho_counts_df = get_feature_counts(df_subset, cluste_lbl=morpho_lbl_key)
    morpho_counts_df["perc"] = morpho_counts_df["#samples"] / np.sum(
        morpho_counts_df["#samples"]
    )
    prompt_counts_df = get_feature_counts(df_subset, cluste_lbl="prompt_idx")
    return df_subset, prompt_counts_df


def balance_dataset(
    df_prompts: pd.DataFrame,
    text_2_promptidx: Dict,
    n_clusters_prompts: int,
    n_samples_train: int = 30,
    n_samples_test: int = 20,
    n_samples_val: int = 10,
    n_minimum_of_prompts_per_class: int = 10,
):
    """Generate a subste of the full dataset that is balanced on prompts and labels,
    given the size of the final subset.
    """
    logger.info(" ... .... .... Balancing prompted dataset ... .... ....")
    inv_text_2_promptidx = {v: k for k, v in text_2_promptidx.items()}
    morpho_lbl_key = f"cluster_lbl_n{n_clusters_prompts}"
    prompt_counts_df_c, prompt_counts_df_h = get_prompt_counts_dfs(
        df_prompts=df_prompts,
        cluste_lbl="prompt_idx",
        inv_text_2_promptidx=inv_text_2_promptidx,
        morpho_lbl_key=morpho_lbl_key,
    )
    logger.info(
        f"after get_prompt_counts_dfs {prompt_counts_df_c.shape}, {prompt_counts_df_h.shape}"
    )

    # Get largest balanced set possible:
    # Iterate over various total sizes and get the largets possible
    sizes = [
        int(len(df_prompts) * 1000 * x / len(df_prompts))
        for x in np.arange(50, 1000, 5)
    ]
    train_r = 50 / 56
    test_r = 5 / 56
    val_r = 1 / 56
    for idx, N in enumerate(sizes):
        logger.info(f" SUBSET SIZE = {N}")
        if N > len(df_prompts):
            break

        n_samples_train = int(N * train_r)
        n_samples_test = int(N * test_r)
        n_samples_val = int(N * val_r)
        prompt_counts_df_c = check_max_n_prompts_to_be_used(
            prompt_counts_df_c,
            n_classes=2,
            n_samples_train=n_samples_train,
            n_samples_test=n_samples_test,
            n_samples_val=n_samples_val,
        )
        prompt_counts_df_h = check_max_n_prompts_to_be_used(
            prompt_counts_df_h,
            n_classes=2,
            n_samples_train=n_samples_train,
            n_samples_test=n_samples_test,
            n_samples_val=n_samples_val,
        )
        # Compare the 2 dfs that we have for each class and get the number of majority prompts
        # (i.e. prompts with largest sample counts) that we should select for our balanced subset.
        try:
            n_selected_prompts_per_class = get_last_common_consider_index(
                prompt_counts_df_h,
                prompt_counts_df_c,
                n_minimum_of_prompts_per_class=n_minimum_of_prompts_per_class,
            )
        except ValueError:
            if idx == 0:
                raise ValueError(
                    "No size matches the constraints for balancing prompt building."
                )
            logger.info(
                f"Largest prompt balanced dataset we can get with this prompt set is {sizes[idx-1]}"
            )
            break

    # Creating prompt balanced subsets of the full dataset
    # Largest size possible
    N = sizes[idx - 1]
    n_samples_train = int(N * train_r)
    n_samples_test = int(N * test_r)
    n_samples_val = int(N * val_r)
    prompt_counts_df_c = check_max_n_prompts_to_be_used(
        prompt_counts_df_c,
        n_classes=2,
        n_samples_train=n_samples_train,
        n_samples_test=n_samples_test,
        n_samples_val=n_samples_val,
    )
    prompt_counts_df_h = check_max_n_prompts_to_be_used(
        prompt_counts_df_h,
        n_classes=2,
        n_samples_train=n_samples_train,
        n_samples_test=n_samples_test,
        n_samples_val=n_samples_val,
    )
    n_selected_prompts_per_class = get_last_common_consider_index(
        prompt_counts_df_h,
        prompt_counts_df_c,
    )
    logger.info(
        f"Using {n_selected_prompts_per_class} most populated prompts from each class "
        + "to build final balanced ds"
    )

    df_subset, prompt_counts_df = get_subset_and_counts_dfs(
        df_prompts=df_prompts,
        prompt_counts_df_c=prompt_counts_df_c,
        prompt_counts_df_h=prompt_counts_df_h,
        n_prompts=n_selected_prompts_per_class,
        morpho_lbl_key=morpho_lbl_key,
    )
    logger.info(f" df_subset shape {df_subset.shape}")
    logger.info(f" prompt_counts_df shape {prompt_counts_df.shape}")
    logger.info(prompt_counts_df)
    # Get only teh prompt sused in the final subset
    text_2_promptidx_valid_susbet = {
        k: v for k, v in text_2_promptidx.items() if k in np.unique(df_subset["prompt"])
    }
    return df_subset, text_2_promptidx_valid_susbet


def process_final_df(df_subset, morpho_lbl_key):
    df_subset_simple = df_subset.rename(columns={"target": "label", "prompt": "text"})
    df_subset_simple["image"] = df_subset_simple["path"].map(lambda x: x.split("/")[-1])
    df_subset_simple = df_subset_simple[
        ["path", "image", "label", "target_name", morpho_lbl_key, "text", "prompt_idx"]
    ]

    return df_subset_simple.reset_index(drop=True)


def create_classification_dataset(
    df_subset,
    n_clusters_prompts: int,
    n_samples_train: int = 30,
    n_samples_test: int = 20,
    n_samples_val: int = 10,
    seed=SEED,
    single_split=False,
):
    """Process and split dataset"""
    morpho_lbl_key = f"cluster_lbl_n{n_clusters_prompts}"
    if not single_split:
        N = n_samples_train + n_samples_test + n_samples_val
        test_frac = 1 - n_samples_train / N
        X_train, X_test = train_test_split(
            df_subset,
            test_size=test_frac,
            random_state=seed,
            stratify=df_subset["prompt"],
        )
        test_frac = n_samples_val / (N - n_samples_train)
        X_test, X_val = train_test_split(
            X_test, test_size=test_frac, random_state=seed, stratify=X_test["prompt"]
        )
        X_train = process_final_df(X_train, morpho_lbl_key)
        X_test = process_final_df(X_test, morpho_lbl_key)
        X_val = process_final_df(X_val, morpho_lbl_key)
    else:
        X_train = process_final_df(df_subset, morpho_lbl_key)
        X_test = []
        X_val = []

    return X_train, X_val, X_test
