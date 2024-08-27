import logging
import json
from itertools import islice

import torch
from diffusers import DPMSolverMultistepScheduler, StableDiffusionPipeline

from biogen.load_sd import load_sd_pipeline

logger = logging.getLogger(__name__)


def load_pipeline_LoRA_im_gen(model_base, model_path, ckpt, enable_xformers=True):
    logger.info(" ####################################### ")
    logger.info(f"Loading pretrained model: {model_path}")
    logger.info(" ####################################### ")
    pipe_h = StableDiffusionPipeline.from_pretrained(
        pretrained_model_name_or_path=str(model_base),
        torch_dtype=torch.float16,
    )
    pipe_h.scheduler = DPMSolverMultistepScheduler.from_config(pipe_h.scheduler.config)
    if isinstance(ckpt, int):
        logger.info(f"Loading LORA chcckpoint-{ckpt}")
        pipe_h.unet.load_attn_procs(
            model_path,
            subfolder=f"checkpoint-{ckpt}",
            weight_name="pytorch_model.bin",
        )
    else:
        logger.info("Loading last step")
        pipe_h.unet.load_attn_procs(model_path)
    if enable_xformers:
        pipe_h.enable_xformers_memory_efficient_attention()
    pipe_h.safety_checker = None
    pipe_h.to("cuda")
    return pipe_h


def load_pipeline_im_gen(model_path, ckpt, enable_xformers):
    logger.info(" ####################################### ")
    logger.info(f"Loading pretrained model: {model_path}")
    if ckpt is not None:
        logger.info(f"Checkpoint selected: {ckpt}")
        pipe_h = load_sd_pipeline(
            pretrained_model_name_or_path=str(model_path),
            checkpoint=ckpt,
            enable_xformers_memory_efficient_attention=enable_xformers,
            gradient_checkpointing=False,
            allow_tf32=True,
        )
    else:
        logger.info("Checkpoint selected: Last")
        pipe_h = StableDiffusionPipeline.from_pretrained(
            pretrained_model_name_or_path=str(model_path),
            torch_dtype=torch.float16,
        )
    if enable_xformers:
        pipe_h.enable_xformers_memory_efficient_attention()
    logger.info(pipe_h.scheduler)
    logger.info(" ####################################### ")
    pipe_h.safety_checker = None
    pipe_h.to("cuda")
    return pipe_h


def get_prompt_dict(full_path):
    """Load valid prompt set."""

    with open(full_path, "r") as fp:
        text_2_promptidx_valid_susbet = json.load(fp)

    for key, value in text_2_promptidx_valid_susbet.items():
        assert isinstance(key, str)
        assert isinstance(int(value), int)
        break

    return text_2_promptidx_valid_susbet


def chunks(data_dict, size=100):
    """Defining an dict chunk iterator"""
    it = iter(data_dict)
    for _ in range(0, len(data_dict), size):
        yield {k: data_dict[k] for k in islice(it, size)}
