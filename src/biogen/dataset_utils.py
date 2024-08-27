"""
This module contains all the functions needed to load and preprocess image datasets.

The datasets are located in the `data` folder at the root of the project, a directory that is
accessible through the `PROJECT_DATA_DIR` constant. Each dataset is located in the data folder
into a folder with the dataset's name.

The dataset structure is the following:

* In the dataset folder there are three folders corresponding with the dataset splits named\
 `train`, `validation` and `test`.
* Inside each split folder there is one folder per class label named after each class.
* The images corresponding to each class label and split are located into the class label folder.
"""

import json
import logging
import os
import shutil
from pathlib import Path
from typing import List, Union

# NOTE: new imports to fix pyarrow issue stemming from these datasets having been saved with
# an older version of pyarrow
import pyarrow

pyarrow.PyExtensionType.set_auto_load(True)
import pyarrow_hotfix

pyarrow_hotfix.uninstall()

import numpy as np
import pandas as pd
from datasets import ClassLabel, Dataset, DatasetDict, Features
from datasets import Image as ImageFeature
from datasets import Value, concatenate_datasets
from datasets import load_dataset as load_hf_dataset
from datasets import load_from_disk
from tqdm import tqdm

from biogen.paths import DATASET_MODES, METADATA_FILE_NAME

logger = logging.getLogger(__name__)
SEED = 18021999


def load_metadata(path: Union[str, Path], file_name=METADATA_FILE_NAME) -> pd.DataFrame:
    """Load the target metadata file."""
    return pd.read_json(Path(path) / file_name, lines=True)


def load_dataset_metadata(
    path: Union[str, Path],
    file_name=METADATA_FILE_NAME,
) -> pd.DataFrame:
    """Load the metadata files for the different splits of a Dataset."""
    ds_path = Path(path)
    splits = os.listdir(ds_path)
    metadata = pd.DataFrame()
    for split in splits:
        df = load_metadata(ds_path / split, file_name=file_name)
        df["split"] = split
        metadata = pd.concat([metadata, df], axis=0)
    return metadata


# TODO probably duplicate of load_dataset_metadata FIXME
def get_metadata_df_from_dataset_path(ds_path):
    metadata_split_list = []
    split_list = os.listdir(ds_path)
    assert "train" in split_list, "No train split found."
    logger.info(f"Splits found: {split_list}")
    for split in split_list:
        split_path = Path(ds_path) / split
        metadata_split = pd.read_json(
            split_path / "metadata.jsonl", lines=True, orient="records"
        )
        metadata_split["split"] = [split for x in range(len(metadata_split))]
        metadata_split["split_index"] = [x for x in range(len(metadata_split))]
        metadata_split_list.append(metadata_split)
    full_metadata_df_synth = pd.concat(
        metadata_split_list, ignore_index=True
    ).reset_index(drop=True)
    return full_metadata_df_synth


def get_image_ds_features(
    path: Union[str, Path],
    file_name=METADATA_FILE_NAME,
    label_col: str = "label",
    image_col: str = "image",
) -> Features:
    """Compute the features for loading an image Dataset with the appropriate data types."""
    df = load_dataset_metadata(path=path, file_name=file_name)
    labels = ClassLabel(names=list(sorted(df[label_col].unique().tolist())))
    feats = {label_col: labels, image_col: ImageFeature(decode=False)}
    ignore_cols = {"split", "file_name", label_col}
    for col in df.columns:
        if col in ignore_cols:
            continue
        # This will only work for string and numpy numeric data types
        type_str = type(df[col].values[0]).__name__.replace("str", "string")
        feats[col] = Value(dtype=type_str)

    return Features(feats)


def load_image_dataset(path: Union[str, Path], **kwargs) -> DatasetDict:
    """
    Load a local dataset and return it as a :class:`DatasetDict`.

    Expects dataset directory to be structured train/im_class1 and
    test/im_class2, (...). Each split must include metadata file.
    """
    return load_hf_dataset("imagefolder", data_dir=path, **kwargs)


def load_tensor_dataset(path: Union[Path, str], **kwargs) -> DatasetDict:
    """
    Load the :class:`DatasetDict` corresponding to the specified latents dataset.

    Args:
        path (Union[Path, str], optional): Path where the target dataset folder is located.
        kwargs (dict): Passed to datasets.load_from_disk.

    Returns:
        DatasetDict: Dataset of the encoded latents.
    """
    # datasets==2.20.0
    # pyarrow>=14.0.0
    return load_from_disk(str(path), **kwargs)


def load_dataset(
    name: str,
    mode: str,
    path: Union[Path, str],
    **kwargs,
) -> DatasetDict:
    """
    Load a dataset from the shared NAS storage.

    Args:
        name (str): Name of the dataset.
        mode (str): One of the available DatasetModes ("image", "latent", "token").
        path (Union[Path, str], optional): Path where the dataset is stored.
            Defaults to DATASETS_DIR (/central_data/biogen/datasets).
        kwargs (dict): Passed to datasets.load_dataset or datasets.load_from_disk.

    Returns:
        DatasetDict: The requested dataset.
    """
    path = Path(path) / name / mode
    is_image = mode == DATASET_MODES.image
    return (
        load_image_dataset(path, **kwargs)
        if is_image
        else load_tensor_dataset(path, **kwargs)
    )


def load_balanced_subset(
    name: str,
    mode: str,
    path: Union[Path, str],
    n_samples: int,
    balancing_col: str = "label",
    seed: int = SEED,
    **kwargs,
) -> DatasetDict:
    """Load full dataset will try to get a balanced set, but only guarantees balanced set if N/2
    is smaller or equal to teh amount of sample sin teh minority class. Breaks if N/2 is larger
    than len(ds)/2. Could have done it with test_train_split on each singel label subset but it
    would be slow. # NOTE only works for binary classes
    """
    # Load train split into same dataset
    dataset = load_dataset(
        name=name,
        mode=mode,
        path=path,
        **kwargs,
    )
    _ds = dataset["train"]
    ds_len = len(_ds)
    assert (
        n_samples < ds_len
    ), f"Size of subset ({n_samples}) cannot be larger than actual size of ds ({ds_len})."
    # Shuffle and sort dataset
    _ds = _ds.shuffle(seed=seed).sort(balancing_col)
    # Get first and last N/2 samples
    ds_lbl_a = _ds.select(
        list(range(ds_len - 1, ds_len - 1 - int(np.floor(n_samples / 2)), -1))
    )
    ds_lbl_b = _ds.select(list(range(np.ceil(n_samples / 2).astype(int))))
    balanced_binary_ds = concatenate_datasets([ds_lbl_a, ds_lbl_b]).shuffle(seed=seed)
    # NOTE this would work for multiple labels but slow
    # unique_labels = np.unique(dataset_all_splits["label"])
    # lbl_ds_s = {}
    # for lbl in unique_labels:
    #     lbl_ds_s[lbl]=dataset_all_splits.filter(lambda row: row["label"] == lbl)
    # lbl_ds_s
    return DatasetDict({"train": balanced_binary_ds})


def has_decoded_image(
    dataset: Union[Dataset, DatasetDict], name: str = "image"
) -> bool:
    """Return True if the provided column of the dataset corresponds to decoded images."""
    dataset = (
        dataset if isinstance(dataset, Dataset) else dataset[list(dataset.keys())[0]]
    )
    example = dataset[name][0]
    return not (isinstance(example, dict) and "path" in example)


def cast_image(
    dataset, decode: bool, name: str = "image"
) -> Union[DatasetDict, Dataset]:
    """Set the decode parameter of the target image feature of the provided dataset."""
    return dataset.cast_column(name, ImageFeature(decode=decode))


def get_dataset_names(path: Union[str, Path]) -> List[str]:
    """List the available datasets in share storage."""
    path = Path(path)
    return os.listdir(path)


def create_dataset_dirs(
    path: Union[str, Path],
    splits=("train", "test", "validation"),
    mode="image",
) -> None:
    """Create train, test, validation dirs at the saving_dir."""
    for split in splits:
        os.makedirs(Path(path) / mode / split, exist_ok=True)


def create_json_save_img_from_df(
    subset_df: pd.DataFrame,
    sav_split_path: Union[Path, str],
) -> None:
    """Create metadata.jsonl and save image files from the subset dataframe."""
    list_row = []
    for _, row in tqdm(subset_df.iterrows()):
        img_src_path = Path(row.path)
        img_sav_path = Path(sav_split_path) / row.image
        # Copy image from original dataset to subset
        shutil.copy(img_src_path, img_sav_path)
        jsonl_row = {
            "file_name": row.image,
            "path": str(img_sav_path),
            "label": row.label,
            "label_name": row.target_name,
            "text": row.text,
            "prompt_idx": row.prompt_idx,
        }
        list_row.append(jsonl_row)

    with open(sav_split_path / "metadata.jsonl", "w") as f:
        for item in list_row:
            f.write(json.dumps(item) + "\n")


def save_img_from_df_unconditional(
    subset_df: pd.DataFrame,
    sav_split_path: Union[Path, str],
    label_column: str = "label_name",
) -> None:
    """Create metadata.jsonl and save image files from the subset dataframe."""
    for _, row in tqdm(subset_df.iterrows()):
        img_src_path = Path(row.path)
        img_sav_path = Path(sav_split_path) / row[label_column]
        os.makedirs(img_sav_path, exist_ok=True)
        # Copy image from original dataset to subset
        shutil.copy(img_src_path, img_sav_path / row.image)


def get_susbet_clean_balanced_splits(path):
    """Get subsets with splits."""
    df_train = pd.read_csv(Path(path) / "df_train.csv")
    df_test = (
        pd.read_csv(Path(path) / "df_test.csv") if "df_test.csv" in os.listdir() else []
    )
    df_val = (
        pd.read_csv(Path(path) / "df_val.csv") if "df_val.csv" in os.listdir() else []
    )
    return df_train, df_test, df_val


def create_subset_dataset(
    SAVE_DATA_DIR: Union[Path, str],
    DF_DATA_DIR: Union[Path, str],
    seperate_folder_per_label: bool = False,
    label_col: str = "target_name",
) -> None:
    """Create a subset from the whole dataset."""
    # Get dfs with new dataset from which to create teh splits from
    df_train, df_test, df_val = get_susbet_clean_balanced_splits(DF_DATA_DIR)

    # Create directory in central data, copy ims into respective folders
    if df_test == [] or df_val == []:
        splits = ["train"]
    else:
        splits = ["train", "test", "validation"]
    logger.info(f"Splits found:  {splits}")

    if not seperate_folder_per_label:
        create_dataset_dirs(SAVE_DATA_DIR, splits)
        split_dfs = [df_train, df_test, df_val]
        for split, split_df in zip(splits, split_dfs):
            sav_split_path = Path(SAVE_DATA_DIR) / "image" / split
            logger.info(f"Saving images onto: {sav_split_path}")
            create_json_save_img_from_df(split_df, sav_split_path)
    else:
        # one file structure for each label in teh dataset
        split_dfs = [df_train, df_test, df_val]
        split_dfs = [x for x in split_dfs if isinstance(x, pd.DataFrame)]
        single_split_df = pd.concat(split_dfs, axis=0)
        list_of_labels = single_split_df[label_col].unique()
        logger.info(f"Labels in dataset: {list_of_labels}")
        for split, split_df in zip(splits, split_dfs):
            for label in list_of_labels:
                SAVE_DATA_DIR_lbl = Path(f"{str(SAVE_DATA_DIR)}_{label}")
                sav_split_path = Path(SAVE_DATA_DIR_lbl) / "image" / split
                os.makedirs(sav_split_path, exist_ok=True)
                split_df_lbl = split_df[split_df[label_col] == label].reset_index(
                    drop=True
                )
                logger.info(f"Saving images onto: {sav_split_path}")
                create_json_save_img_from_df(split_df_lbl, sav_split_path)


def create_subset_UNconditional_dataset(
    SAVE_DATA_DIR: Union[Path, str],
    DF_DATA_DIR: Union[Path, str],
    label_column: str = "label_name",
) -> None:
    """Create a subset from the whole dataset."""
    # Get dfs with new dataset from which to create teh splits from
    df_train, df_test, df_val = get_susbet_clean_balanced_splits(DF_DATA_DIR)

    # Create directory in central data, copy ims into respective folders
    if df_test == [] or df_val == []:
        splits = ["train"]
    else:
        splits = ["train", "test", "validation"]
    logger.info("Splits found:  ", splits)
    create_dataset_dirs(SAVE_DATA_DIR, splits)
    split_dfs = [df_train, df_test, df_val]
    for split, split_df in zip(splits, split_dfs):
        sav_split_path = Path(SAVE_DATA_DIR) / "image" / split
        logger.info(f"Saving images onto: {sav_split_path}")
        save_img_from_df_unconditional(split_df, sav_split_path, label_column)


def create_file_structure(
    experiment_name,
    data_path: Union[str, Path],
    df_data_dir=None,
    save_files_dir=None,
    unconditional_dataset: bool = False,
    seperate_folder_per_label: bool = False,
    label_column: str = "label_name",
):
    """Create file structure to be read by fine tuning scripts base don teh
    datfarames output by the pormpt building pipeline"""

    DF_DATA_DIR = (
        Path(data_path) / experiment_name / "metadata"
        if df_data_dir is None
        else Path(df_data_dir)
    )

    if not unconditional_dataset:
        # create conditional dataset

        SAVE_FILES_DIR = (
            Path(data_path) / experiment_name / "dataset_balanced"
            if save_files_dir is None
            else Path(save_files_dir)
        )

        logger.info(f"DATA_DIR: {DF_DATA_DIR}")
        logger.info(f"SAVE_DATA_DIR : {SAVE_FILES_DIR}")

        create_subset_dataset(
            SAVE_FILES_DIR,
            DF_DATA_DIR,
            seperate_folder_per_label=seperate_folder_per_label,
            label_col=label_column,
        )
    else:
        # create UNconditional dataset
        SAVE_FILES_DIR = (
            Path(data_path) / experiment_name / "dataset_balanced_unconditional"
            if save_files_dir is None
            else Path(save_files_dir)
        )

        logger.info(f"DATA_DIR: {DF_DATA_DIR}")
        logger.info(f"SAVE_DATA_DIR : {SAVE_FILES_DIR}")

        create_subset_UNconditional_dataset(
            SAVE_FILES_DIR, DF_DATA_DIR, label_column=label_column
        )

    return DF_DATA_DIR
