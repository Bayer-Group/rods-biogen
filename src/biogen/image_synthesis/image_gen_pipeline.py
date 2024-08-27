import json
import logging
import os
from pathlib import Path, PurePath
from typing import Optional, Union

import numpy as np
import ray
import torch

from biogen.image_synthesis.generate_and_save import (
    generate_images,
    reformat_tmp_metadata_files_into_hf,
    save_images_and_metadata,
)
from biogen.image_synthesis.load_sd_pipeline import (
    chunks,
    get_prompt_dict,
    load_pipeline_im_gen,
    load_pipeline_LoRA_im_gen,
)

logger = logging.getLogger(__name__)


def generate_synthetic_dataset(
    model_path: Union[Path, str],
    LORA_base_model: Union[Path, str, None],
    ckpt: Optional[int],
    prompt_dict_path: Union[Path, str],
    n_batch: int,
    num_images_batch: int = 10,
    guidance_scale: float = 7.5,
    enable_xformers: bool = True,
    num_inference_steps: int = 50,
    testing_flag: bool = False,
    lora_intensity: float = 1,
    n_most_common: Optional[int] = None,
    save_im_dir: Optional[Union[Path, str]] = None,
    save_im_path: Optional[Union[str, Path]] = None,
):
    """Generates a synthetic dataset whielleveraging parallel computing along the
    prompt direction, i.e. Each GPUs generates images from a subset of the prompts.

    If LORA then make sure to give as input: cross_attention_kwargs={"scale":
    lora_intensity}, will be stored in LoRa_kwargs
    """
    model_type = "LoRa" if LORA_base_model is not None else "regular"
    logger.info(f"Creating synthetic dataset with {model_type} finetuned model")

    # Define relevant paths
    model_path = Path(model_path)
    model_name = PurePath(model_path).name
    if save_im_path is None:
        if save_im_dir is None:
            save_im_path = (
                model_path.parents[1] / "synthetic" / model_name / "image" / "train"
            )
            save_im_dir = model_name
        else:
            save_im_dir = save_im_dir
            save_im_path = (
                model_path.parents[1] / "synthetic" / save_im_dir / "image" / "train"
            )
    else:
        assert str(save_im_path).endswith("image/train")
    os.makedirs(save_im_path, exist_ok=True)
    logger.info(f"save_im_path : {save_im_path}")
    save_im_path = Path(save_im_path)
    # Load model pipeline # FIXME Loading from ckpt leads to inference 2x slowrr than when loading normally
    if model_type == "LoRa":
        pipe_h = load_pipeline_LoRA_im_gen(
            LORA_base_model,
            model_path=model_path,
            ckpt=ckpt,
            enable_xformers=enable_xformers,
        )
        LoRa_kwargs = {"cross_attention_kwargs": {"scale": lora_intensity}}
    else:
        pipe_h = load_pipeline_im_gen(
            model_path=model_path,
            ckpt=ckpt,
            enable_xformers=enable_xformers,
        )
        LoRa_kwargs = {}
    # Load adequate prompt dictionary
    prompt_dictionary = get_prompt_dict(full_path=prompt_dict_path)
    if testing_flag:
        prompt_dictionary = {k: v for k, v in prompt_dictionary.items() if int(v) < 64}
    n_prompts = len(prompt_dictionary)

    # Final dataset size
    n_batch = n_batch
    num_images_batch = num_images_batch
    n_chunks = torch.cuda.device_count()
    prompt_dict_chunk_len = int(
        np.ceil(n_prompts / n_chunks)
    )  # last chunk will have a bit less than teh others
    logger.info(f"n_chunks: {n_chunks}")
    logger.info(f"prompt_dict_chunk_len: {prompt_dict_chunk_len}")

    # Set recurrent input objects to local object store
    logger.info("Set recurrent input objects to local object store ...")
    pipe_h_id = ray.put(pipe_h)
    guidance_scale = ray.put(guidance_scale)

    # Generation loop
    logger.info("Starting generation loop:")
    try:
        process_ids = []
        for chunk_idx, prompt_dict_chunk in enumerate(
            chunks(prompt_dictionary, prompt_dict_chunk_len)
        ):
            logger.info(f" prompt_dict_chunk = {chunk_idx + 1} / {n_chunks}")
            process_id = generate_and_save_images_for_chunk_multibatch.remote(
                pipe_h=pipe_h_id,
                num_images_batch=num_images_batch,
                guidance_scale=guidance_scale,
                num_inference_steps=num_inference_steps,
                prompt_dict_chunk=prompt_dict_chunk,
                n_batch=n_batch,
                save_im_path=save_im_path,
                metadata_filename=f"metadata_{chunk_idx}_tmp.jsonl",
                **LoRa_kwargs,
            )
            process_ids.append(process_id)
        # Run ray.get to allows all processed to finish and ignore unsused outputs
        _ = ray.get(process_ids)
        # if generation reaches an end or keyboard interrupt concatenate all metadata.jsonl files and
        # delete other temp metadata files
        logger.info(" Combining all temporary chunked metadata ... ")
        reformat_tmp_metadata_files_into_hf(save_im_path)
    except KeyboardInterrupt:
        logger.info(
            " ::::::::::::::::::::::: :::::::::::::::::::::::: ::::::::::::::::::::::: "
        )
        logger.info(
            " ... Keyboard interrupt exception was caught during image generation ... "
        )
        logger.info(
            " ::::::::::::::::::::::: :::::::::::::::::::::::: ::::::::::::::::::::::: "
        )
        logger.info(" Combining all temporary chunked metadata ... ")
        reformat_tmp_metadata_files_into_hf(save_im_path)

    return 1


@ray.remote(num_gpus=1)
def generate_and_save_images_for_chunk_multibatch(
    pipe_h,
    num_images_batch,
    guidance_scale,
    num_inference_steps,
    prompt_dict_chunk,
    n_batch,
    save_im_path,
    metadata_filename: str = "metadata.jsonl",
    **kwargs,
):
    """Generate all batches of images for a particular (chunk) prompt dictionary and save all
    metadata in a chunk specific metadata.jsonl file."""
    n_prompt_chunk = len(prompt_dict_chunk)
    for batch_idx in range(n_batch):
        logger.info(f"Generating Batch = {batch_idx + 1} / {n_batch}")
        for iter, (text_prompt, prompt_idx) in enumerate(prompt_dict_chunk.items()):
            # TODO Get label name some other way
            # We should be getting the raw values isntead fo the prompt to avoid string parsing
            # As the prompts may change a lot from one experiment to the other.
            # We should create a dictionary containing the information specific to a given prompt
            # And a template string to fill it in.
            possible_labels = ["tumor", "normal", "cancer", "healthy"]
            if any(element in text_prompt for element in possible_labels):
                label = text_prompt.split("tissue,")[0].split("of")[1].replace(" ", "")
            else:
                label = "unknown"
            logger.info(f" ... Prompt idx = {iter + 1} / {n_prompt_chunk} ... ")
            logger.info(f"text_prompt: {text_prompt}")
            logger.info(f"label:  {label}")
            images = generate_images(
                pipe_h,
                text_prompt,
                num_images_batch,
                guidance_scale,
                num_inference_steps,
                **kwargs,
            )
            # Save each image in batch and accumulate metadata
            list_row = save_images_and_metadata(
                images,
                label,
                prompt_idx,
                text_prompt,
                batch_idx,
                save_im_path,
            )
            # Store append new metadata to metadata.jsonl
            with open(Path(save_im_path) / metadata_filename, "a+") as f:
                for item in list_row:
                    f.write(json.dumps(item) + "\n")
    return 1


def generate_synthetic_dataset_batch_parallel(
    model_path: Union[Path, str],
    LORA_base_model: Union[Path, str, None],
    ckpt: Optional[int],
    prompt_dict_path: Union[Path, str],
    n_batch: int,
    num_images_batch: int = 10,
    guidance_scale: float = 7.5,
    enable_xformers: bool = True,
    num_inference_steps: int = 50,
    testing_flag: bool = False,
    lora_intensity: float = 1,
    n_most_common: Optional[int] = None,
    save_im_dir: Optional[Union[Path, str]] = None,
    save_im_path: Optional[Union[str, Path]] = None,
):
    """Generates a synthetic dataset while leveraging parallel computing along the
    batch diretion, i.e. Each GPUs generates images from all prompts but only a fraction
     of all the batches needed.

    Useful for synthetic sets with low number of prompts.

    If LORA then make sure to give as input: cross_attention_kwargs={"scale":
    lora_intensity}, will be stored in LoRa_kwargs
    """
    model_type = "LoRa" if LORA_base_model is not None else "regular"
    logger.info(f"Creating synthetic dataset with {model_type} finetuned model")

    # Define relevant paths
    model_path = Path(model_path)
    model_name = PurePath(model_path).name
    if save_im_path is None:
        if save_im_dir is None:
            save_im_path = (
                model_path.parents[1] / "synthetic" / model_name / "image" / "train"
            )
            save_im_dir = model_name
        else:
            save_im_dir = save_im_dir
            save_im_path = (
                model_path.parents[1] / "synthetic" / save_im_dir / "image" / "train"
            )
    else:
        assert str(save_im_path).endswith("image/train")
    os.makedirs(save_im_path, exist_ok=True)
    logger.info(f"save_im_path : {save_im_path}")
    # Load model pipeline # FIXME Loading from ckpt leads to inference 2x slowrr than when loading normally
    if model_type == "LoRa":
        pipe_h = load_pipeline_LoRA_im_gen(
            LORA_base_model,
            model_path=model_path,
            ckpt=ckpt,
            enable_xformers=enable_xformers,
        )
        LoRa_kwargs = {"cross_attention_kwargs": {"scale": lora_intensity}}
    else:
        pipe_h = load_pipeline_im_gen(
            model_path=model_path,
            ckpt=ckpt,
            enable_xformers=enable_xformers,
        )
        LoRa_kwargs = {}
    # Load adequate prompt dictionary
    prompt_dictionary = get_prompt_dict(full_path=prompt_dict_path)
    if testing_flag:
        prompt_dictionary = {k: v for k, v in prompt_dictionary.items() if int(v) < 64}

    # Final dataset size
    # n_batch per prompt
    n_batch = n_batch
    num_images_batch = num_images_batch
    n_chunks = torch.cuda.device_count()
    n_batches_per_gpu = int(
        np.ceil(n_batch / n_chunks)
    )  # last chunk will have a bit less than teh others
    logger.info(f"n_chunks or n_gpus: {n_chunks}")
    logger.info(f"n_batches_per_gpu: {n_batches_per_gpu}")

    # Set recurrent input objects to local object store
    logger.info("Set recurrent input objects to local object store ...")
    pipe_h_id = ray.put(pipe_h)
    guidance_scale = ray.put(guidance_scale)

    # Generation loop
    logger.info("Starting generation loop:")
    try:
        process_ids = []
        for batch_set_idx in range(n_chunks):
            logger.info(f" prompt_dict_chunk = {batch_set_idx + 1} / {n_chunks}")
            process_id = generate_and_save_images_for_set_of_batches.remote(
                pipe_h=pipe_h_id,
                num_images_batch=num_images_batch,
                guidance_scale=guidance_scale,
                num_inference_steps=num_inference_steps,
                prompt_dict=prompt_dictionary,
                n_batch_per_gpu=n_batches_per_gpu,
                save_im_path=save_im_path,
                metadata_filename=f"metadata_{batch_set_idx}_tmp.jsonl",
                batch_set_idx=batch_set_idx,
                **LoRa_kwargs,
            )
            process_ids.append(process_id)
        # Run ray.get to allows all processed to finish and ignore unsused outputs
        _ = ray.get(process_ids)
        # if generation reaches an end or keyboard interrupt concatenate all metadata.jsonl files and
        # delete other temp metadata files
        logger.info(" Combining all temporary chunked metadata ... ")
        reformat_tmp_metadata_files_into_hf(save_im_path)
    except KeyboardInterrupt:
        logger.info(
            " ::::::::::::::::::::::: :::::::::::::::::::::::: ::::::::::::::::::::::: "
        )
        logger.info(
            " ... Keyboard interrupt exception was caught during image generation ... "
        )
        logger.info(
            " ::::::::::::::::::::::: :::::::::::::::::::::::: ::::::::::::::::::::::: "
        )
        logger.info(" Combining all temporary chunked metadata ... ")
        reformat_tmp_metadata_files_into_hf(save_im_path)

    return 1


def generate_synthetic_dataset_NORAY(
    model_path: Union[Path, str],
    LORA_base_model: Union[Path, str, None],
    ckpt: Optional[int],
    prompt_dict_path: Union[Path, str],
    n_batch: int,
    num_images_batch: int = 10,
    guidance_scale: float = 7.5,
    enable_xformers: bool = True,
    num_inference_steps: int = 50,
    testing_flag: bool = False,
    lora_intensity: float = 1,
    n_most_common: Optional[int] = None,
    save_im_dir: Optional[Union[Path, str]] = None,
    save_im_path: Optional[Union[str, Path]] = None,
):
    """Generates a synthetic dataset while leveraging parallel computing along the
    batch diretion, i.e. Each GPUs generates images from all prompts but only a fraction
     of all the batches needed.

    Useful for synthetic sets with low number of prompts.

    If LORA then make sure to give as input: cross_attention_kwargs={"scale":
    lora_intensity}, will be stored in LoRa_kwargs
    """
    model_type = "LoRa" if LORA_base_model is not None else "regular"
    logger.info(f"Creating synthetic dataset with {model_type} finetuned model")

    # Define relevant paths
    model_path = Path(model_path)
    model_name = PurePath(model_path).name
    if save_im_path is None:
        if save_im_dir is None:
            save_im_path = (
                model_path.parents[1] / "synthetic" / model_name / "image" / "train"
            )
            save_im_dir = model_name
        else:
            save_im_dir = save_im_dir
            save_im_path = (
                model_path.parents[1] / "synthetic" / save_im_dir / "image" / "train"
            )
    else:
        assert str(save_im_path).endswith("image/train")
    os.makedirs(save_im_path, exist_ok=True)
    logger.info(f"save_im_path : {save_im_path}")
    # Load model pipeline # FIXME Loading from ckpt leads to inference 2x slowrr than when loading normally
    if model_type == "LoRa":
        pipe_h = load_pipeline_LoRA_im_gen(
            LORA_base_model,
            model_path=model_path,
            ckpt=ckpt,
            enable_xformers=enable_xformers,
        )
        LoRa_kwargs = {"cross_attention_kwargs": {"scale": lora_intensity}}
    else:
        pipe_h = load_pipeline_im_gen(
            model_path=model_path,
            ckpt=ckpt,
            enable_xformers=enable_xformers,
        )
        LoRa_kwargs = {}
    # Load adequate prompt dictionary
    prompt_dictionary = get_prompt_dict(full_path=prompt_dict_path)
    if testing_flag:
        prompt_dictionary = {k: v for k, v in prompt_dictionary.items() if int(v) < 64}

    # Final dataset size
    # n_batch per prompt
    n_batch = n_batch
    num_images_batch = num_images_batch
    n_chunks = torch.cuda.device_count()
    n_batches_per_gpu = int(
        np.ceil(n_batch / n_chunks)
    )  # last chunk will have a bit less than teh others
    logger.info(f"n_chunks or n_gpus: {n_chunks}")
    logger.info(f"n_batches_per_gpu: {n_batches_per_gpu}")

    # Generation loop
    logger.info("Starting generation loop:")
    try:
        lits_of_outputs = []
        for batch_set_idx in range(n_chunks):
            logger.info(f" prompt_dict_chunk = {batch_set_idx + 1} / {n_chunks}")
            process_id = generate_and_save_images_for_set_of_batches_NORAY(
                pipe_h=pipe_h,
                num_images_batch=num_images_batch,
                guidance_scale=guidance_scale,
                num_inference_steps=num_inference_steps,
                prompt_dict=prompt_dictionary,
                n_batch_per_gpu=n_batches_per_gpu,
                save_im_path=save_im_path,
                metadata_filename=f"metadata_{batch_set_idx}_tmp.jsonl",
                batch_set_idx=batch_set_idx,
                **LoRa_kwargs,
            )
            lits_of_outputs.append(process_id)
        logger.info(" Combining all temporary chunked metadata ... ")
        reformat_tmp_metadata_files_into_hf(save_im_path)
    except KeyboardInterrupt:
        logger.info(
            " ::::::::::::::::::::::: :::::::::::::::::::::::: ::::::::::::::::::::::: "
        )
        logger.info(
            " ... Keyboard interrupt exception was caught during image generation ... "
        )
        logger.info(
            " ::::::::::::::::::::::: :::::::::::::::::::::::: ::::::::::::::::::::::: "
        )
        logger.info(" Combining all temporary chunked metadata ... ")
        reformat_tmp_metadata_files_into_hf(save_im_path)

    return 1


@ray.remote(num_gpus=1)
def generate_and_save_images_for_set_of_batches(
    pipe_h,
    num_images_batch,
    guidance_scale,
    num_inference_steps,
    prompt_dict,
    n_batch_per_gpu,
    save_im_path,
    batch_set_idx,
    metadata_filename: str = "metadata.jsonl",
    **kwargs,
):
    """Generate given number of batches for all prompts and save all
    metadata in a batch_set specific metadata.jsonl file."""
    n_prompt_chunk = len(prompt_dict)
    for batch_idx in range(n_batch_per_gpu):
        logger.info(f"Generating Batch = {batch_idx + 1} / {n_batch_per_gpu}")
        for iter, (text_prompt, prompt_idx) in enumerate(prompt_dict.items()):
            # TODO Get label name some other way
            # We should be getting the raw values isntead fo the prompt to avoid string parsing
            # As the prompts may change a lot from one experiment to the other.
            # We should create a dictionary containing the information specific to a given prompt
            # And a template string to fill it in.
            possible_labels = ["tumor", "normal", "cancer", "healthy"]
            if any(element in text_prompt for element in possible_labels):
                label = text_prompt.split("tissue,")[0].split("of")[1].replace(" ", "")
            else:
                label = "unknown"
            logger.info(f" ... Prompt idx = {iter + 1} / {n_prompt_chunk} ... ")
            logger.info(f"text_prompt: {text_prompt}")
            logger.info(f"label:  {label}")
            images = generate_images(
                pipe_h,
                text_prompt,
                num_images_batch,
                guidance_scale,
                num_inference_steps,
                **kwargs,
            )
            # Save each image in batch and accumulate metadata
            list_row = save_images_and_metadata(
                images,
                label,
                prompt_idx,
                text_prompt,
                (
                    batch_idx + batch_set_idx * 10000
                ),  # NOTE for filenmae uniqueness and dependance on batch set, i.e. gpu
                save_im_path,
            )
            # Store append new metadata to metadata.jsonl
            with open(save_im_path / metadata_filename, "a+") as f:
                for item in list_row:
                    f.write(json.dumps(item) + "\n")
    return 1


def generate_and_save_images_for_set_of_batches_NORAY(
    pipe_h,
    num_images_batch,
    guidance_scale,
    num_inference_steps,
    prompt_dict,
    n_batch_per_gpu,
    save_im_path,
    batch_set_idx,
    metadata_filename: str = "metadata.jsonl",
    **kwargs,
):
    """Generate given number of batches for all prompts and save all
    metadata in a batch_set specific metadata.jsonl file."""
    n_prompt_chunk = len(prompt_dict)
    for batch_idx in range(n_batch_per_gpu):
        logger.info(f"Generating Batch = {batch_idx + 1} / {n_batch_per_gpu}")
        for iter, (text_prompt, prompt_idx) in enumerate(prompt_dict.items()):
            # TODO Get label name some other way
            # We should be getting the raw values isntead fo the prompt to avoid string parsing
            # As the prompts may change a lot from one experiment to the other.
            # We should create a dictionary containing the information specific to a given prompt
            # And a template string to fill it in.
            possible_labels = ["tumor", "normal", "cancer", "healthy"]
            if any(element in text_prompt for element in possible_labels):
                label = text_prompt.split("tissue,")[0].split("of")[1].replace(" ", "")
            else:
                label = "unknown"
            logger.info(f" ... Prompt idx = {iter + 1} / {n_prompt_chunk} ... ")
            logger.info(f"text_prompt: {text_prompt}")
            logger.info(f"label:  {label}")
            images = generate_images(
                pipe_h,
                text_prompt,
                num_images_batch,
                guidance_scale,
                num_inference_steps,
                **kwargs,
            )
            # Save each image in batch and accumulate metadata
            list_row = save_images_and_metadata(
                images,
                label,
                prompt_idx,
                text_prompt,
                (
                    batch_idx + batch_set_idx * 10000
                ),  # NOTE for filenmae uniqueness and dependance on batch set, i.e. gpu
                save_im_path,
            )
            # Store append new metadata to metadata.jsonl
            with open(save_im_path / metadata_filename, "a+") as f:
                for item in list_row:
                    f.write(json.dumps(item) + "\n")
    return 1
