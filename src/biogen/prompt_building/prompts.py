"""Functions to build the prompts."""

import logging

import numpy as np
import pandas as pd
import inflect

from biogen.prompt_building.feature_extraction.extract_features import (
    get_feature_counts,
)
from biogen.prompt_building.feature_extraction.morphology import cluster_labels

logger = logging.getLogger(__name__)
n2w_util = inflect.engine()

def edit_prompt(target_name: str, c_morpho: int, text_template: str):
    if "[LABEL]" in text_template:
        text_template = text_template.replace("[LABEL]", target_name.lower())
    text = text_template.replace("[MEP-CLUSTER-INDEX]", n2w_util.number_to_words(c_morpho))
    return text


def create_prompts(df, morpho_lbl_key, text_template):
    df_copy = df.copy()
    df_copy["prompt"] = df.apply(
        lambda x: edit_prompt(
            target_name=x["target_name"],
            c_morpho=x[morpho_lbl_key],
            text_template=text_template,
        ),
        axis=1,
    )
    return df_copy


def prompt_df_builder(df, morpho_lbl_key, text_template):
    df_prompts = create_prompts(
        df,
        morpho_lbl_key=morpho_lbl_key,
        text_template=text_template,
    )
    text_2_promptidx = dict(
        zip(
            np.unique(df_prompts["prompt"]), range(len(np.unique(df_prompts["prompt"])))
        )
    )
    df_prompts["prompt_idx"] = df_prompts["prompt"].map(lambda x: text_2_promptidx[x])

    morpho_counts_df = get_feature_counts(df_prompts, cluste_lbl=morpho_lbl_key)
    morpho_counts_df["perc"] = morpho_counts_df["#samples"] / np.sum(
        morpho_counts_df["#samples"]
    )
    prompt_counts_df = get_feature_counts(df_prompts, cluste_lbl="prompt_idx")
    return df_prompts, prompt_counts_df, morpho_counts_df, text_2_promptidx


# MAIN PROMPT BUILDING FUNCTION
def get_prompted_dataset(
    cleaned_token_data: np.ndarray,
    df_clean: pd.DataFrame,
    text_template: str,
    n_clusters_prompts: int = 3,
    kmeans_batch_size=None,
):
    """
    Main prompt building function. Clusters tokens into morphological groups and creates
    prompts with that information.
    """
    logger.info("Clustering DiNO class tokens to build prompts ...")
    # TODO """Try faiss for clustering, supposed to be muuuuuch faster.
    # Careful w/ package versions"""
    (
        df_clean_morpho,
        clustering_model,
    ) = cluster_labels(
        n_clusters_prompts,
        df_clean,
        cleaned_token_data,
        kmeans_batch_size,
    )

    n_col = len(df_clean.columns)
    logger.info(f" new columns: {df_clean_morpho.columns.tolist()[(n_col-1):]}")

    logger.info(" ... .... .... Use features to build prompts ... .... ....")
    morpho_lbl_key = f"cluster_lbl_n{n_clusters_prompts}"
    (
        df_prompts,
        prompt_counts_df,
        _,
        text_2_promptidx,
    ) = prompt_df_builder(
        df_clean_morpho,
        morpho_lbl_key=morpho_lbl_key,
        text_template=text_template,
    )
    logger.info(f"after promptr_df_builder (n={n_clusters_prompts}) {df_prompts.shape}")
    logger.info(f" Number of prompts = {prompt_counts_df.shape[0]}")
    logger.info(
        f"df_prompts {df_prompts.columns}"
        f" \n {df_prompts['prompt'].unique()}"
        f" \n {df_prompts['target_name'].unique()}"
    )
    return (
        df_prompts,
        text_2_promptidx,
        clustering_model,
    )
