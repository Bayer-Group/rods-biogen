"""Contains :class:`Path` representing to the relevant directories and files of the project."""

from collections import namedtuple
from pathlib import Path

METADATA_FILE_NAME = "metadata.jsonl"
# Dataset modes
IMAGE_MODE = "image"
LATENT_MODE = "latent"
TOKEN_MODE = "token"
_DatasetModes = namedtuple("DatasetModes", ["image", "latent", "token"])
DATASET_MODES = _DatasetModes(IMAGE_MODE, LATENT_MODE, TOKEN_MODE)

# Project directories
PROJECT_ROOT_PATH = Path(__file__).parent.parent.parent
NOTEBOOKS_DIR = PROJECT_ROOT_PATH / "notebooks"
SCRIPTS_DIR = PROJECT_ROOT_PATH / "scripts"
CONFIG_DIR = PROJECT_ROOT_PATH / "config"
