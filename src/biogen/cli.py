import datetime
import logging
import sys
from pathlib import Path

import click
from omegaconf import OmegaConf

from biogen.config import load_config, unpack_config_nested
from biogen.paths import CONFIG_DIR

_logger = logging.getLogger(__name__)
hydra_args = click.argument("hydra_args", nargs=-1, type=click.UNPROCESSED)

file_option = click.option(
    "--file",
    "-f",
    "file",
    help="Path to the configuration file",
    default="",
    type=click.Path(
        exists=False,
        file_okay=True,
        dir_okay=False,
        readable=True,
        resolve_path=False,
        allow_dash=False,
        path_type=None,
    ),
)


@click.group()
def cli():
    pass


@cli.command()
@hydra_args
@file_option
def dataset_to_tokens(file, hydra_args):
    """Convert image dataset into tokens dataset."""
    from biogen.dataset_transforms.dino_vit_tokens import (
        convert_dataset_to_tokens as conv_ds_2_tokens,
    )

    if not file:
        file = CONFIG_DIR / "config-dataset2tokens.yaml"
    config = load_config(file, hydra_args)

    output_path = conv_ds_2_tokens(**config.dataset2tokens)
    ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    config_path = output_path.parent / f"cli-config-d2t-{ts}.yaml"
    OmegaConf.save(config, str(config_path))
    return 1


@cli.command()
@hydra_args
@file_option
def build_prompts(file, hydra_args):
    """Build prompts for a dataset."""
    import biogen.prompt_building.pipeline as pipeline

    ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

    if not file:
        file = CONFIG_DIR / "config-pb.yaml"
    config = load_config(file, hydra_args)
    print(OmegaConf.to_yaml(config))
    click.echo(config)

    # flatten DictConfig to match expected input format of pipeline
    OmegaConf.resolve(config)
    parms = OmegaConf.to_container(config)
    parms = unpack_config_nested(parms)
    pipeline.preprocess_and_build_balanced_prompt_dataset(**parms)

    # Save config
    config_path = (
        Path(parms["save_path"]) / parms["experiment_name"] / f"cli-config-pb-{ts}.yaml"
    )
    OmegaConf.save(config, str(config_path))
    return 1


@cli.command()
@hydra_args
@file_option
def create_file_structure(file, hydra_args):
    """Create dataset structure for fine-tuning diffusion models."""
    from biogen.dataset_utils import create_file_structure as _create_file_structure

    if not file:
        file = CONFIG_DIR / "config-create-file-structure.yaml"
    config = load_config(file, hydra_args)
    df_data_dir = _create_file_structure(**config.create_fs)
    ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    OmegaConf.save(config, str(df_data_dir / f"cli-config-cfs-{ts}.yaml"))
    return 1


@cli.command()
@hydra_args
@file_option
def generate_image_sd(file, hydra_args):
    """Prompt-wise image generation with stable diffusion model."""
    from biogen.image_synthesis.image_gen_pipeline import generate_synthetic_dataset

    if not file:
        file = CONFIG_DIR / "config-image-gen.yaml"
    config = load_config(file, hydra_args)

    _ = generate_synthetic_dataset(**config.image_gen)
    ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    OmegaConf.save(
        config,
        str(
            Path(config.image_gen.save_im_path).parent / f"cli-config-img-gen-{ts}.yaml"
        ),
    )
    return 1


@cli.command()
@hydra_args
@file_option
def evaluate_synthetic_dataset(file, hydra_args):
    """Evaluate synthetic dataset against real dataset with FID score."""
    from biogen.metrics.analysis_pipeline import (
        do_inceptionv3_based_image_quality_analysis,
    )

    if not file:
        file = CONFIG_DIR / "config-eval.yaml"
    config = load_config(file, hydra_args)

    do_inceptionv3_based_image_quality_analysis(**config.evaluation)
    save_path = Path(config.evaluation.synthetic_image_data_path).parent / "metrics"
    OmegaConf.save(config, str(save_path / f"cli-config-eval-{config.now}.yaml"))
    return 1


@cli.command()
@hydra_args
@file_option
def test_config(file, hydra_args):
    """Create dataset structure for fine-tuning diffusion models."""

    if not file:
        file = CONFIG_DIR / "config-pb.yaml"
    config = load_config(file, hydra_args)
    click.echo(config)
    return 1


if __name__ == "__main__":
    sys.exit(cli())
