import json
import logging
import os
import sys
from pathlib import Path
from typing import Optional, Sequence, Union
from unittest.mock import patch

import hydra
import hydra._internal.utils as hydra_utils
from omegaconf import DictConfig, OmegaConf

_logger = logging.getLogger(__name__)


def get_from_env(
    identifier: str = "CONFIG:",
    argv: Optional[Sequence[str]] = None,
):
    """Retrieves envvars from EC2 instances as set in AWS Batch or AWS SageMaker"""
    # Firstly, add any envvars starting with identifier
    overrides = [f"{k}={v}" for k, v in os.environ.items() if k.startswith(identifier)]

    # Secondly, add cli arguments
    # 1. Case SageMaker: assume "--key value" arguments & discard Hydra's argparse
    if "SM_HPS" in os.environ:
        # Assume no cli arguments in Hydra format are passed; scan from SM_HPS
        # Convert SageMaker to Hydra variables. It's a string containing a JSON
        sagemaker_env = json.loads(os.environ.get("SM_HPS", "{}"))
        overrides.extend([f"{k}={v}" for k, v in sagemaker_env.items()])
    # 2. Case not SageMaker: assume "foo=bar" arguments without option decorator --
    else:
        overrides.extend(hydra_utils.get_args(argv).overrides)  # Use Hydra's argparse

    # Finally, replace any string identifier or json-converted "None"s from
    # overrides
    for i, ovr in enumerate(overrides):
        assert (
            len(ovr.split("=")) == 2
        ), f"override {ovr} has more than a single '=' sign"
        k, v = ovr.split("=")
        if k.startswith(identifier):
            k = k.replace(identifier, "")
        if v.replace(" ", "").lower() == "none":
            v = "null"

        # Remove value if starts with ~ (see hydra documentation)
        if k.startswith("~"):
            overrides[i] = f"{k}".replace(" ", "")
        else:
            overrides[i] = f"{k}={v}".replace(" ", "")

    return tuple(overrides)


def load_config(config_file: Union[Path, str], hydra_args: str) -> DictConfig:
    """
    Load the necessary configuration for running mloq from a mloq.yaml file.

    If no path to mloq.yaml is provided, it returns a template to be filled in
    using the interactive mode.

    Args:
        config_file: Path to the target mloq.yaml file.
        hydra_args: Arguments passed to hydra for composing the project configuration.

    Returns:
        DictConfig containing the project configuration.
    """
    # Get arguments from environment variables, useful only for AWS Batch and Sagemaker
    hydra_args = get_from_env(identifier="CONFIG:", argv=hydra_args)

    # Continue loading config file
    config_file = Path(config_file)
    if config_file.exists() and config_file.is_file():
        _logger.info(f"Loading config file from {config_file}")
        hydra_args = ["--config-dir", str(config_file.parent)] + list(hydra_args)
        config = DictConfig({})

        @hydra.main(config_path=".", config_name=config_file.name)
        def load_config(loaded_config: DictConfig):
            nonlocal config
            config = loaded_config

        with patch("sys.argv", [sys.argv[0]] + list(hydra_args)):
            load_config()
    else:
        _logger.error(f"Invalid config file path provided: {config_file}")
        raise ValueError(f"Invalid config file: {config_file}")
    return config


def unpack_config(conf, keys):
    OmegaConf.resolve(conf)
    subconf = OmegaConf.masked_copy(conf, keys)
    params = {}  # TODO: maybe use omegaconf.merge or other native methods
    for v in subconf.values():
        params.update(v)
    return params


def unpack_config_nested(conf):
    re = {}
    for k, v in conf.items():
        print(k, type(v))
        if isinstance(v, dict):
            re.update(unpack_config_nested(v))
        else:
            re.update({k: v})
    return re
