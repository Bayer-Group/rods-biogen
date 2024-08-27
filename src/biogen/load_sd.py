"""This modules allows to load a StableDiffusionPipeline from local checkpoints & the model hub."""

import logging
import os
from pathlib import Path
from typing import List, Optional, Union

import torch
from accelerate import load_checkpoint_and_dispatch
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    StableDiffusionPipeline,
    UNet2DConditionModel,
)
from diffusers.utils.import_utils import is_xformers_available
from transformers import CLIPTextModel, CLIPTokenizer

logger = logging.getLogger(__name__)


STABLE_DIFFUSION_MODELS = (
    "CompVis/stable-diffusion-v1-4",
    "runwayml/stable-diffusion-v1-5",
    "stabilityai/stable-diffusion-2-1-base",
)


def get_finetuned_model_names(path: Union[Path, str]) -> List[str]:
    """Return the names of the available fine-tuned models."""
    return os.listdir(path)


def get_available_checkpoints(name: str, path: Union[Path, str]) -> List[int]:
    """Return a list containing the training steps of the available checkpoints for a model."""
    path = Path(path)
    fine_tuned_models = get_finetuned_model_names(path)
    assert (
        name in fine_tuned_models
    ), f"Invalid model name. Available names are {fine_tuned_models}"
    return [
        int(s.replace("checkpoint-", ""))
        for s in os.listdir(path / name)
        if "checkpoint" in s
    ]


def init_modules(
    pretrained_model_name_or_path: Union[Path, str],
    revision: str = None,
    noise_scheduler=None,
    tokenizer: CLIPTokenizer = None,
    text_encoder: CLIPTextModel = None,
    vae: AutoencoderKL = None,
    unet: UNet2DConditionModel = None,
):
    """Initialize the stable diffusion modules."""
    # Load scheduler, tokenizer and models.
    pretrained_model_name_or_path = Path(pretrained_model_name_or_path)
    if noise_scheduler is None:
        noise_scheduler = DDPMScheduler.from_pretrained(
            str(pretrained_model_name_or_path),
            subfolder="scheduler",
        )
    if tokenizer is None:
        tokenizer = CLIPTokenizer.from_pretrained(
            str(pretrained_model_name_or_path),
            subfolder="tokenizer",
            revision=revision,
        )
    if text_encoder is None:
        if Path(pretrained_model_name_or_path).exists():
            text_encoder = CLIPTextModel.from_pretrained(
                str(pretrained_model_name_or_path / "text_encoder"),
                revision=revision,
            )
        else:
            text_encoder = CLIPTextModel.from_pretrained(
                str(pretrained_model_name_or_path),
                revision=revision,
                subfolder="text_encoder",
            )
        text_encoder.requires_grad_(False)
    if vae is None:
        vae = AutoencoderKL.from_pretrained(
            str(pretrained_model_name_or_path),
            subfolder="vae",
            revision=revision,
        )
        vae.requires_grad_(False)
    if unet is None:
        unet = UNet2DConditionModel.from_pretrained(
            str(pretrained_model_name_or_path),
            subfolder="unet",
            revision=revision,
        )
    return unet, vae, text_encoder, tokenizer, noise_scheduler


def load_unet_checkpoint(model, model_folder, checkpoint: int, **kwargs):
    """Load the model weights from a checkpoint."""
    kwargs["offload_state_dict"] = kwargs.get("offload_state_dict", True)
    path = (
        Path(model_folder)
        / f"checkpoint-{checkpoint}"
        / "unet"
        / "diffusion_pytorch_model.bin"
    )
    model = load_checkpoint_and_dispatch(model, str(path), **kwargs)
    return model


def load_sd_pipeline(
    pretrained_model_name_or_path: str,
    checkpoint: Optional[int] = None,
    revision: str = None,
    enable_xformers_memory_efficient_attention: bool = True,
    gradient_checkpointing: bool = True,
    allow_tf32: bool = True,
    noise_scheduler=None,
    tokenizer: CLIPTokenizer = None,
    text_encoder: CLIPTextModel = None,
    vae: AutoencoderKL = None,
    unet: UNet2DConditionModel = None,
):
    """Load a stable diffusion pipeline."""
    if unet is not None and checkpoint is not None:
        raise ValueError("checkpoint and unet cannot be both different than None.")

    unet, vae, text_encoder, tokenizer, noise_scheduler = init_modules(
        pretrained_model_name_or_path=pretrained_model_name_or_path,
        revision=revision,
        noise_scheduler=noise_scheduler,
        tokenizer=tokenizer,
        text_encoder=text_encoder,
        vae=vae,
        unet=unet,
    )
    if checkpoint is not None:
        unet = load_unet_checkpoint(unet, pretrained_model_name_or_path, checkpoint)
    if enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            unet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError(
                "xformers is not available. Make sure it is installed correctly"
            )
    if gradient_checkpointing:
        unet.enable_gradient_checkpointing()

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
    pipeline = StableDiffusionPipeline.from_pretrained(
        pretrained_model_name_or_path,
        text_encoder=text_encoder,
        vae=vae,
        unet=unet,
        revision=revision,
        tokenizer=tokenizer,
        scheduler=noise_scheduler,
        torch_dtype=torch.float16,
    )
    return pipeline
