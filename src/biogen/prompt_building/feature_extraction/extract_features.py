"""This module includes function to extract features from the images of the dataset in the form of
a pandas df"""

import functools
from multiprocessing import Pool, cpu_count
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

from biogen.paths import PROJECT_ROOT_PATH


def get_feature_counts(df: pd.DataFrame, cluste_lbl: str) -> pd.DataFrame:
    """
    Given a DataFrame with clustering data and a clustering label, groups the data
    by the clustering label, counts the number of occurrences for each group, and
    returns the resulting DataFrame sorted by count in ascending order.

    Args:
        df: A DataFrame with clustering data.
        cluste_lbl: The label of the column containing the clustering data.

    Returns:
        A pandas DataFrame with columns "cluster_lbl" and "#samples". The DataFrame
        is sorted by "#samples" in ascending order.

    """
    col_df = df.groupby([cluste_lbl]).size().reset_index()
    col_df_bg = col_df.copy()
    col_df_bg.rename(columns={0: "#samples"}, inplace=True)
    col_df_bg = col_df_bg.sort_values(by="#samples")
    return col_df_bg.reset_index(drop=True)


def variance_of_laplacian(image):
    """Calculate the variance of the Laplacian for a given image"""
    return cv2.Laplacian(image, cv2.CV_64F).var()


def process_image(row, path_col="path"):
    im_path = row[path_col]
    if not Path(im_path).exists():
        im_path = str(PROJECT_ROOT_PATH / im_path)
    img_rgb = Image.open(im_path)
    img_hsv = img_rgb.convert("HSV")
    img_hsv_arr = np.array(img_hsv)
    img_gray = img_rgb.convert("L")
    img_gray_arr = np.array(img_gray)

    _, thresh = cv2.threshold(img_gray_arr, 150, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    return (
        np.mean(img_hsv_arr[:, :, 2]),
        np.median(img_hsv_arr[:, :, 2]),
        np.std(img_hsv_arr[:, :, 2]),
        np.mean(img_hsv_arr[:, :, 1]),
        np.median(img_hsv_arr[:, :, 1]),
        np.std(img_hsv_arr[:, :, 1]),
        np.std(img_hsv_arr[:, :, 0]),
        variance_of_laplacian(img_gray_arr),
        len(contours),
    )


def extract_hsv_features(df: pd.DataFrame, path_col: str = "path") -> pd.DataFrame:
    """
    Extracts features from each image in the dataset and appends them as columns to the dataframe.

    Args:
        df (pd.DataFrame): The dataframe to extract features from.
        path_col (str, optional): The name of the column in the dataframe that
            contains the image paths. Defaults to "path".

    Returns:
        pd.DataFrame: The resulting dataframe with the appended feature columns.
    """

    with Pool(cpu_count()) as pool:
        process_image_partial = functools.partial(process_image, path_col=path_col)
        features_list = list(
            tqdm(
                pool.imap(
                    process_image_partial, (row._asdict() for row in df.itertuples())
                ),
                total=len(df),
            )
        )

    features = pd.DataFrame(
        features_list,
        columns=[
            "mean_V",
            "median_V",
            "std_V",
            "mean_S",
            "median_S",
            "std_S",
            "std_H",
            "var_L",
            "n_contour",
        ],
    )
    return pd.concat([df.reset_index(drop=True), features], axis=1)


def get_median_hsv_from_path(row, path_col="path"):
    im_path = row[path_col]
    img_rgb = Image.open(im_path)

    # NOTE Much faster with pillow but nomrmalisation is needed
    img_hsv = img_rgb.convert("HSV")
    img_hsv_arr = np.array(img_hsv)
    h = np.mean(img_hsv_arr[:, :, 0]) / 255
    s = np.mean(img_hsv_arr[:, :, 1]) / 255
    v = np.mean(img_hsv_arr[:, :, 2]) / 255
    return h, s, v


def get_median_hsv(df: pd.DataFrame, path_col: str = "path") -> pd.DataFrame:
    """
    Computes median hue, saturation, and value (HSV) values for each image in a DataFrame.

    Args:
        df (pd.DataFrame): DataFrame with a 'path' column representing file paths.

    Returns:
        pd.DataFrame: Input DataFrame with 'medianH', 'medianS', and 'medianV' columns added.
    """

    with Pool(cpu_count()) as pool:
        process_image_partial = functools.partial(
            get_median_hsv_from_path, path_col=path_col
        )
        features_list = list(
            tqdm(
                pool.imap(
                    process_image_partial, (row._asdict() for row in df.itertuples())
                ),
                total=len(df),
            )
        )

    features = pd.DataFrame(
        features_list,
        columns=[
            "medianH",
            "medianS",
            "medianV",
        ],
    )
    return pd.concat([df.reset_index(drop=True), features], axis=1)
