import logging
import os
from pathlib import Path
from typing import Optional, Union

import pandas as pd
from datasets.dataset_dict import DatasetDict

from biogen.dataset_transforms.dino_vit_tokens import normalise_token_dataset
from biogen.dataset_transforms.inceptionv3_latents import (
    convert_dataset_to_inceptionv3_latents,
)
from biogen.dataset_utils import load_tensor_dataset
from biogen.metrics.feature_metrics import FEATURE_METRICS

logger = logging.getLogger(__name__)


def calculate_FID_and_coverage(
    target_synth,
    reference_real,
    target_ds_name: str = "synthetic",
    reference_ds_name: str = "real",
    features_col: str = "inceptionv3_latent",
    token_feature_metrics=FEATURE_METRICS,
    n_max_real: int = 50_000,
    n_max_synth: int = 50_000,
    straify_subset_col="prompt_idx",
) -> pd.DataFrame:
    """
    Compute the FID (Fréchet Inception Distance) and PR (Precision-Recall) metrics for comparing two different data distributions
    on the basis of their compressed representations.

    Parameters:
    - target_synth: The synthetic dataset.
    - reference_real: The real dataset.
    - target_ds_name: The name of the synthetic dataset (default: "synthetic").
    - reference_ds_name: The name of the real dataset (default: "real").
    - features_col: The column name of the compressed representations (default: "inceptionv3_latent").
    - token_feature_metrics: The dictionary of token feature metrics (default: FEATURE_METRICS).
    - n_max_real: The maximum number of samples to consider from the real dataset (default: 50,000).
    - n_max_synth: The maximum number of samples to consider from the synthetic dataset (default: 50,000).
    - straify_subset_col: The column name to stratify the subset (default: "prompt_idx").

    Returns:
    - df: A pandas DataFrame containing the computed metrics.

    Note:
    - The function subsets the real and synthetic datasets if the number of samples exceeds the specified maximums.
    - The function computes the FID and PR metrics over all data and adds the results to the DataFrame.
    """
    # Process columns in order for them to match
    logger.info("Updating label and label_name columns based on prompt ...")
    target_synth = DatasetDict({"train": target_synth["train"]})
    reference_real = DatasetDict({"train": reference_real["train"]})
    n_real = len(reference_real["train"])
    n_synth = len(target_synth["train"])
    # Get only n_max_real or less
    if n_real > n_max_real:
        logger.info("Subsetting the real dataset")
        reference_real = reference_real.class_encode_column(straify_subset_col)
        ratio_real = n_real / len(reference_real["train"])
        ratio_real = 0.999 if ratio_real >= 1 else ratio_real
        reference_real = DatasetDict(
            {
                "train": reference_real["train"].train_test_split(
                    test_size=ratio_real,
                    stratify_by_column=straify_subset_col,
                )["test"]
            }
        )
    logger.info(f"reference_real: {reference_real}")
    # get only n_max_synth or less
    if n_synth > n_max_synth:
        logger.info("Subsetting the synthetic dataset")
        target_synth = target_synth.class_encode_column(straify_subset_col)
        ratio_synth = n_synth / len(target_synth["train"])
        ratio_synth = 0.999 if ratio_synth >= 1 else ratio_synth
        target_synth = DatasetDict(
            {
                "train": target_synth["train"].train_test_split(
                    test_size=ratio_synth,
                    stratify_by_column=straify_subset_col,
                )["test"]
            }
        )
    logger.info(f"target_synth: {target_synth}")
    # Cast real and synthetic datasets into the right format
    reference_real = reference_real.with_format("torch")["train"]
    target_synth = target_synth.with_format("torch")["train"]
    x = reference_real[features_col].reshape((reference_real.shape[0], -1))
    y = target_synth[features_col].reshape((target_synth.shape[0], -1))
    # Get fid computing function
    FIDLatentFunction = token_feature_metrics["fid_vit"]
    # Get PR computing function
    PRLatentFunction = token_feature_metrics["precision_recall"]
    # Initialise results df
    df = pd.DataFrame(
        columns=[
            "metric",
            "reference_ds",
            "target_ds",
            "split",
            "value",
        ]
    )
    # First over all data
    logger.info("Computing FID and PR over all data ...")
    # FID
    fid_bs_set = FIDLatentFunction(x, y)
    logger.info(f"FID: {fid_bs_set}")
    row = pd.DataFrame.from_records(
        [
            {
                "metric": "fid_based_on_latents",
                "reference_ds": reference_ds_name,
                "target_ds": target_ds_name,
                "split": "train",
                "value": float(fid_bs_set),
            }
        ]
    )
    df = pd.concat([df, row], ignore_index=True)
    # PR
    value = PRLatentFunction(x, y)
    precision, recall = float(value[0]), float(value[1])
    logger.info(f"Precision: {precision} and Recall {recall}")
    row = pd.DataFrame.from_records(
        [
            {
                "metric": "precision",
                "reference_ds": reference_ds_name,
                "target_ds": target_ds_name,
                "split": "train",
                "value": precision,
            }
        ]
    )
    df = pd.concat([df, row], ignore_index=True)
    row = pd.DataFrame.from_records(
        [
            {
                "metric": "recall",
                "reference_ds": reference_ds_name,
                "target_ds": target_ds_name,
                "split": "train",
                "value": recall,
            }
        ]
    )
    df = pd.concat([df, row], ignore_index=True)
    return df


def run_synthetic_dataset_analysis(
    real_ds_path: Union[Path, str],
    synthetic_ds_path: Union[Path, str],
    metric_ds_path: Optional[str] = None,
    batch_size: int = 128,
    token_type: str = "token",
    target_ds_name: str = "synthetic",
    reference_ds_name: str = "real",
):
    """
    Run image quality analysis on synthetic dataset via FID and PR metrics. Save the
    results in a csv file.

    Args:
        real_ds_path (Union[Path, str]): Path to the real dataset.
        synthetic_ds_path (Union[Path, str]): Path to the synthetic dataset.
        metric_ds_path (Optional[str], optional): Path to save the metrics. Defaults to None.
        batch_size (int, optional): Batch size for processing the datasets. Defaults to 128.
        token_type (str, optional): Type of token for analysis. Defaults to "token".
        target_ds_name (str, optional): Name of the synthetic dataset. Defaults to "synthetic".
        reference_ds_name (str, optional): Name of the real dataset. Defaults to "real".

    Returns:
        int: Returns 1.
    """
    # Handle paths
    real_ds_path = Path(real_ds_path)
    synthetic_ds_path = Path(synthetic_ds_path)
    if metric_ds_path is None:
        metric_ds_path = synthetic_ds_path.parent / "metrics"
    else:
        metric_ds_path = Path(metric_ds_path)

    # Get token col name based on token type chosen for analysis
    token_feat_name = (
        "inceptionv3_latent" if token_type == "inceptionv3_tokens" else "class_token"
    )

    # Create output dir if needed
    os.makedirs(metric_ds_path, exist_ok=True)
    logger.info(" ################### Parameters ################# ")
    logger.info(f"Synthetic DS path = {synthetic_ds_path}")
    logger.info(f"Real DS path  = {real_ds_path}")
    logger.info(f"Metric saving path  = {metric_ds_path}")
    logger.info(f"Batch size = {batch_size}")
    logger.info(" ############################################ ")

    # Load datasets
    logger.info("Loading feature datasets ... ")
    token_ds_real = load_tensor_dataset(real_ds_path.parent / token_type)
    logger.info(f" ======================= token_ds_real \n {token_ds_real}")
    token_ds_synth = load_tensor_dataset(synthetic_ds_path.parent / token_type)
    logger.info(f" ======================= token_ds_synth \n {token_ds_synth}")

    # Process token datasets: Normalise
    if token_type == "token":
        logger.info(" ... .... ... Normalising dino vit tokens ... ")
        token_ds_real = normalise_token_dataset(
            token_ds_real["train"],
            token_feat_name,
        )
        token_ds_real = DatasetDict({"train": token_ds_real})

        dataset_synth = normalise_token_dataset(
            token_ds_synth["train"],
            token_feat_name,
        )
        token_ds_synth = DatasetDict({"train": dataset_synth})

    # ... Compute
    logger.info("Computing FID and coverage with Precision and Recall")
    prompt_analysis_df = calculate_FID_and_coverage(
        target_synth=token_ds_synth,
        reference_real=token_ds_real,
        target_ds_name=target_ds_name,
        reference_ds_name=reference_ds_name,
        features_col=token_feat_name,
        token_feature_metrics=FEATURE_METRICS,
        straify_subset_col="text",
        n_max_real=50_000,
        n_max_synth=50_000,
    )
    prompt_analysis_df.to_csv(
        metric_ds_path / f"arxiv_analysis_fid_{token_type}_df.csv", index=False
    )

    return 1


def do_inceptionv3_based_image_quality_analysis(
    real_image_data_path: Union[Path, str],
    synthetic_image_data_path: Union[Path, str],
    token_type: str,
    batch_size: int = 92,
    batch_size_conversion: int = 128,
) -> None:
    """
    Perform image quality analysis using the InceptionV3 model.

    This function takes in the paths to real and synthetic image datasets, along with other parameters,
    and performs image quality analysis using the InceptionV3 model. It generates InceptionV3 tokens for
    the datasets if they don't already exist in the expected locations. It then computes teh FID
    and Precision&Recall metrics based on the generated tokens.

    Parameters:
        real_image_data_path (Union[Path, str]): Path to the real image dataset.
        synthetic_image_data_path (Union[Path, str]): Path to the synthetic image dataset.
        token_type (str): Type of tokens to be generated.
        batch_size (int, optional): Batch size for processing the datasets. Defaults to 92.
        batch_size_conversion (int, optional): Batch size for converting datasets to InceptionV3 tokens.
            Defaults to 128.

    Returns:
        None

    Raises:
        None
    """
    # Create real inception tokens if needed
    real_image_data_path = Path(real_image_data_path)
    real_ds_path_token = real_image_data_path.parent / token_type
    if not real_ds_path_token.is_dir() or len(os.listdir(real_ds_path_token)) == 0:
        logger.info(f"Creating inception v3 tokens of {real_ds_path_token.parents[0]}")
        _ = convert_dataset_to_inceptionv3_latents(
            dataset_path=real_image_data_path,
            batch_size=batch_size_conversion,
        )

    # Convert synthetic dataset to InceptionV3 tokens if needed
    synthetic_image_data_path = Path(synthetic_image_data_path)
    synthetic_ds_path_token = synthetic_image_data_path.parent / token_type
    if (
        not synthetic_ds_path_token.is_dir()
        or len(os.listdir(synthetic_ds_path_token)) == 0
    ):
        logger.info(
            f"Creating inception v3 tokens for synthetic dataset: {synthetic_ds_path_token}"
        )
        _ = convert_dataset_to_inceptionv3_latents(
            dataset_path=synthetic_image_data_path,
            batch_size=batch_size_conversion,
        )

    logger.info("Starting metric comutation ...")
    _ = run_synthetic_dataset_analysis(
        real_ds_path=real_image_data_path,
        synthetic_ds_path=synthetic_ds_path_token,
        batch_size=batch_size,
        token_type=token_type,
    )
