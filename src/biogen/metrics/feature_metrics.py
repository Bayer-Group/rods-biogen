"""This module contains code for computing metrics for comparing two datasets."""

from pathlib import Path
from typing import Tuple, Union

import numpy as np
import pandas as pd
import torch
from piq import FID, PR, BRISQUELoss, TVLoss

from biogen.dataset_utils import load_dataset
from biogen.paths import DATASET_MODES

NO_REFERENCE_METRICS = {
    "total_variation_l2": TVLoss(reduction="none", norm_type="l2"),
    "total_variation_l1": TVLoss(reduction="none", norm_type="l1"),
    "brisque": BRISQUELoss(reduction="None"),
}

FEATURE_METRICS = {
    "fid_vit": FID(),
    "precision_recall": PR(),
}


def load_token_datasets(
    target_ds: str,
    reference_ds: str,
    reference_path: Union[str, Path],
    target_path: Union[str, Path],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Load the target datasets and return the tensors needed to compute feature metrics."""
    if isinstance(reference_ds, str):
        reference_ds = load_dataset(
            name=reference_ds,
            mode=DATASET_MODES.token,
            path=reference_path,
        )
    if isinstance(target_ds, str):
        target_ds = load_dataset(
            name=target_ds, mode=DATASET_MODES.token, path=target_path
        )
    x = reference_ds.with_format("torch")["train"]
    y = target_ds.with_format("torch")["train"]
    return x, y


def append_metric(
    x, y, metric, name, df, label, reference_ds, target_ds, split="train"
):
    """Compute the target metric and append it to the provided DataFrame."""
    if name != "fid_inceptionv3":
        value = metric(x, y)
        if name == "precision_recall":
            precision, recall = float(value[0]), float(value[1])
            row = pd.DataFrame.from_records(
                [
                    {
                        "metric": "precision",
                        "reference_ds": reference_ds,
                        "target_ds": target_ds,
                        "label": label,
                        "split": split,
                        "value": precision,
                    }
                ]
            )
            df = pd.concat([df, row], ignore_index=True)
            row = pd.DataFrame.from_records(
                [
                    {
                        "metric": "recall",
                        "reference_ds": reference_ds,
                        "target_ds": target_ds,
                        "label": label,
                        "split": split,
                        "value": recall,
                    }
                ]
            )
            df = pd.concat([df, row], ignore_index=True)
        else:
            row = pd.DataFrame.from_records(
                [
                    {
                        "metric": name,
                        "reference_ds": reference_ds,
                        "target_ds": target_ds,
                        "label": label,
                        "split": split,
                        "value": float(value),
                    }
                ]
            )
            df = pd.concat([df, row], ignore_index=True)
    else:
        # x and y must be absolute paths to dris with images
        value = metric(x, y)
        row = pd.DataFrame.from_records(
            [
                {
                    "metric": "fid_inceptionv3",
                    "reference_ds": reference_ds,
                    "target_ds": target_ds,
                    "label": label,
                    "split": split,
                    "value": float(value),
                }
            ]
        )
        df = pd.concat([df, row], ignore_index=True)
    return df


def calculate_feature_metrics(
    target_ds: str,
    reference_ds: str,
    reference_path: Union[str, Path],
    target_path: Union[str, Path],
    features_col: str = "class_token",
    label_col: str = "label",
    metrics=FEATURE_METRICS,
) -> pd.DataFrame:
    """Compute the metrics for comparing two different data distributions."""
    reference, target = load_token_datasets(
        target_ds=target_ds,
        reference_ds=reference_ds,
        reference_path=reference_path,
        target_path=target_path,
    )
    if not isinstance(reference_ds, str):
        reference_ds = "real"
    if not isinstance(target_ds, str):
        target_ds = "synthetic"
    df = pd.DataFrame(
        columns=["metric", "reference_ds", "target_ds", "label", "split", "value"]
    )
    x = reference[features_col].reshape((reference.shape[0], -1))
    y = target[features_col].reshape((target.shape[0], -1))
    for name, metric in metrics.items():
        df = append_metric(
            x, y, metric, name, df, "all", reference_ds, target_ds, split="train"
        )

    if label_col is None:
        return df
    labels_x, labels_y = reference[label_col], target[label_col]
    unique_labels = set(np.unique(labels_x.cpu().numpy()).tolist())
    for name, metric in metrics.items():
        for label in unique_labels:
            x_, y_ = x[labels_x == label], y[labels_y == label]
            df = append_metric(
                x_,
                y_,
                metric,
                name,
                df,
                label,
                reference_ds,
                target_ds,
                split="train",
            )
    return df
