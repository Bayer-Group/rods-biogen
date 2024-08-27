"""Mian fucntion that takes all input args and performs the full prompt buliding pipeline.

File structure in use:

DATA_PATH / "experiment_name" / "dataset_balanced"
- For keeping the catual file structure with the balanced dataset
DATA_PATH / "experiment_name" / "metadata"
- For keeping all df´s related to this prompt building approach. df´s and prompt dicts.
DATA_PATH / "experiment_name" / "synthetic"
- For keeping teh synthtic datagenerated after fine tuning on this balanced subset
DATA_PATH / "experiment_name" / "metrics"
- For keeping output of analysis script of teh synthtic dataset in compariosn with the one used
for training the SD that generated it.
DATA_PATH / "experiment_name" / "config.yaml"
- For keeping all the input params used to build this prompt set and final balanced subsets.

"""

import logging
import os
import json
from pathlib import Path
from typing import Union

from biogen.prompt_building.balancing import (
    balance_dataset,
    create_classification_dataset,
)
from biogen.prompt_building.loading import load_data
from biogen.prompt_building.outlier_removal import preprocess_data
from biogen.prompt_building.prompts import get_prompted_dataset

logger = logging.getLogger(__name__)


# MAIN FUNCTION
def preprocess_and_build_balanced_prompt_dataset(
    ds_path: Union[Path, str],
    text_template: str,
    save_path: Union[Path, str],
    data_key: str = "class_token",
    label_key: str = "label",
    path_key: str = "path",
    experiment_name: str = "subset_balanced_prompt",
    outlier_removal: bool = True,
    th_V_max=70,
    th_S_max=3,
    th_std_V=4,
    th_std_S=4,
    th_std_H=1.5,
    th_var_L=100,
    th_n_contour=10,
    class_names_dict=None,
    n_minimum_of_prompts_per_class=10,
    balance=True,
    single_label_dataset_label=None,
    n_clusters_prompts: int = 3,
    kmeans_batch_size=None,
    single_split=None,
    n_samples_train=30,
    n_samples_test=20,
    n_samples_val=10,
):
    logger.info("Starting pipeline")

    # Load data
    df_plot_full_ds, dict_full = load_data(
        ds_path=ds_path,
        data_key=data_key,
        label_key=label_key,
        path_key=path_key,
        class_names_dict=class_names_dict,
    )

    # Outlier removal, if specified
    if outlier_removal:
        cleaned_token_data, df_clean = preprocess_data(
            df_plot_full_ds=df_plot_full_ds,
            dict_full=dict_full,
            th_V_max=th_V_max,
            th_S_max=th_S_max,
            th_std_V=th_std_V,
            th_std_S=th_std_S,
            th_std_H=th_std_H,
            th_var_L=th_var_L,
            th_n_contour=th_n_contour,
        )
    else:
        cleaned_token_data, df_clean = dict_full["data"], df_plot_full_ds

    # Get prompted dataset
    (
        df_prompts,
        text_2_promptidx,
        clustering_model,
    ) = get_prompted_dataset(
        cleaned_token_data,
        df_clean,
        text_template=text_template,
        n_clusters_prompts=n_clusters_prompts,
        kmeans_batch_size=kmeans_batch_size,
    )

    # NOTE only run if there is sufficient data. Small datasets will not have enough
    # data for the balancing computations.
    if balance:
        # ... ... Balance prompted dataset. Ensure same number of images per prompt.
        df_subset, text_2_promptidx_valid_susbet = balance_dataset(
            df_prompts,
            text_2_promptidx,
            n_minimum_of_prompts_per_class=n_minimum_of_prompts_per_class,
            n_clusters_prompts=n_clusters_prompts,
            n_samples_train=n_samples_train,
            n_samples_test=n_samples_test,
            n_samples_val=n_samples_val,
        )
    else:
        df_subset, text_2_promptidx_valid_susbet = (
            df_prompts.copy(),
            text_2_promptidx.copy(),
        )
    # Process final subset and create the splits.
    # Process the df_subset in order to be as expected by the classifier code
    # and also to be able to be used for creating actual file structure with
    # those images via create_dataset_v3.py
    df_train, df_val, df_test = create_classification_dataset(
        df_subset,
        n_clusters_prompts,
        n_samples_train=n_samples_train,
        n_samples_test=n_samples_test,
        n_samples_val=n_samples_val,
        single_split=single_split,
    )
    logger.info(
        f" Sizes: \n df_train - {len(df_train)}\n"
        f" df_val - {len(df_val)}\n"
        f" df_test - {len(df_test)}"
    )
    logger.info(text_2_promptidx_valid_susbet)
    # Save everything relevant related to the dataset just created
    save_prompt_engineered_balanced_dataset_data(
        df_plot_full_ds=df_plot_full_ds,
        df_clean=df_clean,
        df_prompts=df_prompts,
        text_2_promptidx=text_2_promptidx,
        df_subset=df_subset,
        text_2_promptidx_valid_susbet=text_2_promptidx_valid_susbet,
        df_train=df_train,
        df_val=df_val,
        df_test=df_test,
        experiment_name=experiment_name,
        single_label_dataset_label=single_label_dataset_label,
        save_path=save_path,
    )


def save_prompt_engineered_balanced_dataset_data(
    save_path: Union[str, Path],
    df_plot_full_ds,
    df_clean,
    df_prompts,
    text_2_promptidx,
    df_subset,
    text_2_promptidx_valid_susbet,
    df_train,
    df_val,
    df_test,
    experiment_name,
    single_label_dataset_label=None,
):
    """Save ervy input as json if dict and as .csv if pd.Dataframe.

    Args:
        experiment_name (_type_): _description_
        save_path (Union[str, Path], optional): _description_. Defaults to DATA_PATH.
    """

    if single_label_dataset_label is not None:
        # TODO move asssertion to input params handler function
        assert single_label_dataset_label in [
            "tumor",
            "normal",
            "braf",
            "non-braf",
            "cancer",
            "healthy",
        ]
        df_train = df_train[
            df_train["target_name"] == single_label_dataset_label
        ].reset_index(drop=True)
        text_2_promptidx = {
            k: v for k, v in text_2_promptidx.items() if single_label_dataset_label in k
        }
        text_2_promptidx_valid_susbet = {
            k: v
            for k, v in text_2_promptidx_valid_susbet.items()
            if single_label_dataset_label in k
        }

    metadata_dir = Path(save_path) / experiment_name / "metadata"
    os.makedirs(metadata_dir, exist_ok=True)
    logger.info(f" Saving data in: \n {metadata_dir}")
    # Save all df as .csv
    file_path = metadata_dir / "df_plot_full_ds.csv"
    df_plot_full_ds.to_csv(file_path, index=False)

    file_path = metadata_dir / "df_clean.csv"
    df_clean.to_csv(file_path, index=False)

    file_path = metadata_dir / "df_prompts.csv"
    df_prompts.to_csv(file_path, index=False)

    file_path = metadata_dir / "df_subset.csv"
    df_subset.to_csv(file_path, index=False)

    file_path = metadata_dir / "df_train.csv"
    df_train.to_csv(file_path, index=False)

    if df_val != []:
        file_path = metadata_dir / "df_val.csv"
        df_val.to_csv(file_path, index=False)

    if df_test != []:
        file_path = metadata_dir / "df_test.csv"
        df_test.to_csv(file_path, index=False)

    # Save prompt dicts as json
    file_path = metadata_dir / "text_2_promptidx.json"
    with open(file_path, "w") as f:
        json.dump(text_2_promptidx, f)

    file_path = metadata_dir / "text_2_promptidx_valid_susbet.json"
    with open(file_path, "w") as f:
        json.dump(text_2_promptidx_valid_susbet, f)
