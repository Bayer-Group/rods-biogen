import logging

import numpy as np
import pandas as pd
from tqdm import tqdm

logger = logging.getLogger(__name__)


def calculate_prompt_compactness(
    reference,
    features_col: str = "class_token",
    prompt_col: str = "label_name",
    save_path=None,
) -> pd.DataFrame:
    # Get data in tensor format
    reference = reference.with_format("torch")["train"]
    x = reference[features_col].reshape((reference.shape[0], -1))
    df = pd.DataFrame(
        columns=[
            "prompt_idx",
            "mean_var_through_N",
            "max_sqrd_dist_to_center",
        ]
    )
    # Iterate over prompts, compute compactenss measure
    logger.info(f"Computing metrics per {prompt_col} ...")
    prompts_x = reference[prompt_col]
    if not isinstance(prompts_x, list):
        unique_idx = set(np.unique(prompts_x.cpu().numpy()).tolist())
    else:
        unique_idx = set(np.unique(prompts_x).tolist())
    logger.debug(f" unique prompts : {unique_idx}")
    for prompt_ix in tqdm(unique_idx):
        idx_prompts = np.where(np.array(prompts_x) == prompt_ix)[0]
        x_ = np.array(x[idx_prompts])
        logger.debug(np.shape(x_))
        # Mean variance across dimensions: Larger -> Less compact
        mean_var_through_N = np.mean(np.var(x_, axis=0))
        mean_token = np.mean(x_, axis=0)
        max_sqrd_dist_to_center = np.max(
            np.square(
                np.linalg.norm(
                    x_ - mean_token,
                    axis=1,
                )
            )
        )
        row = pd.DataFrame.from_records(
            [
                {
                    "prompt_idx": prompt_ix,
                    "mean_var_through_N": mean_var_through_N,
                    "max_sqrd_dist_to_center": max_sqrd_dist_to_center,
                }
            ]
        )
        df = pd.concat([df, row], ignore_index=True)
    if save_path is not None and str(save_path).endswith(".csv"):
        df.to_csv(save_path, index=False)
    return df
