"""This module allows to generate captions for an image dataset and store them as metadata."""

import json
import os
from pathlib import Path
from typing import Dict, List, Union

from biogen.paths import IMAGE_MODE

METADATA_FILE_NAME = "metadata.jsonl"


def create_jsonl_row(
    file_name: Union[Path, str],
    metadata: Dict[str, str],
    **kwargs,
) -> Dict[str, str]:
    """Create a dictionary containing the metadata asociated with an image."""
    label = metadata[file_name]["label"]
    jsonl_row = {
        "file_name": file_name,
        "label": label,
        "text": f"Histology image of {label} tissue",
    }
    return jsonl_row


def create_jsonl_contents(
    folder_path: Union[Path, str],
    metadata: Dict[str, str],
    create_jsonl_row,
    **kwargs,
):
    """Create a json file representing the metadata of a dataset split."""
    jsonl_rows = []
    for file_name in os.listdir(folder_path):
        if file_name.endswith(".jsonl"):
            continue
        jsonl_row = create_jsonl_row(file_name=file_name, metadata=metadata, **kwargs)
        jsonl_rows.append(jsonl_row)
    jsonl_string = (
        json.dumps(jsonl_rows).replace("[", "").replace("]", "").replace("}, ", "}\n")
    )
    return jsonl_string


def load_jsonl(
    path: Union[Path, str], file_name: str = METADATA_FILE_NAME
) -> List[Dict]:
    """Load a metadata file in jsonl format as a list of dictionaries."""
    with open(path / file_name, "r") as f:
        jsonl_str = f.read()
        return [json.loads(line) for line in jsonl_str.splitlines()]


def load_metadata(
    path: Union[Path, str], file_name: str = METADATA_FILE_NAME
) -> Dict[str, str]:
    """Load the target metadata file and return it as a dictionary indexes by image file name."""
    jsonl_metadata = load_jsonl(path=path, file_name=file_name)
    data = {}
    for example in jsonl_metadata:
        img_name = example["file_name"]
        others = {k: v for k, v in example.items() if k != "file_name"}
        data[img_name] = others
    return data


def caption_dataset(
    dataset_name: str,
    base_dataset_dir: Union[str, Path],
    splits=("train", "test", "validation"),
    create_jsonl_row=create_jsonl_row,
    metadata_filename: str = METADATA_FILE_NAME,
    overwrite: bool = False,
    **kwargs,
):
    """Add text captions to a dataset and store them as metadata for each split in jsonl format."""
    dataset_path = Path(base_dataset_dir) / dataset_name / IMAGE_MODE
    metadata_files = []
    for split in splits:
        split_path = dataset_path / split
        metadata = load_metadata(path=split_path, file_name=metadata_filename)
        jsonl_string = create_jsonl_contents(
            split_path,
            metadata=metadata,
            create_jsonl_row=create_jsonl_row,
            **kwargs,
        )
        metadata_files.append(jsonl_string)
        if overwrite:
            with open(split_path / metadata_filename, "w") as f:
                f.write(jsonl_string)
    return metadata_files
