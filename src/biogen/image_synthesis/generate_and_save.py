import json
import logging
import os
import random
import string
from pathlib import Path

import torch

logger = logging.getLogger(__name__)


def generate_images(
    pipe_h,
    text_prompt,
    num_images_batch,
    guidance_scale,
    num_inference_steps,
    **kwargs,
):
    with torch.inference_mode():
        images = pipe_h(
            [text_prompt] * num_images_batch,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            **kwargs,
        ).images
    return images


def generate_random_string(length):
    random.seed()  # set random seed
    return "".join(random.choices(string.ascii_letters + string.digits, k=length))


def save_images_and_metadata(
    images,
    label,
    prompt_idx,
    text_prompt,
    batch_idx,
    save_im_path,
    tag_len=25,
):
    """Takes list of pil images output by teh model during one inference batch
    and saves tehm along wit thei metadata"""
    list_row = []
    for j, im in enumerate(images):
        tag = generate_random_string(tag_len)
        filename = f"{label}_prompt_{prompt_idx}_batch{batch_idx}_im{j}_{tag}.jpg"
        # Save image
        save_path = Path(save_im_path) / filename
        im.save(save_path)
        # Create metadata jsonl row
        jsonl_row = create_jsonl_row(
            file_name=filename,
            path=str(save_path),
            text_prompt=text_prompt,
            prompt_idx=prompt_idx,
            label=label,
        )
        list_row.append(jsonl_row)
    return list_row


def create_jsonl_row(file_name, path, text_prompt, prompt_idx, label):
    jsonl_row = {
        "file_name": file_name,
        "path": path,
        "label": label,
        "label_name": label,
        "text": text_prompt,
        "prompt_idx": prompt_idx,
    }
    return jsonl_row


def read_jsonl_as_list_of_dict(path_to_file):
    list_row = []
    with open(path_to_file) as f:
        for line in f:
            list_row.append(json.loads(line))
    return list_row


def reformat_tmp_metadata_files_into_hf(save_im_path):
    metadata_temp_files = [
        x for x in os.listdir(save_im_path) if x.startswith("metadata_")
    ]
    logging.info(f" Temporary metadata files found: \n {metadata_temp_files}")
    for filename in metadata_temp_files:
        # loading tmp metadata file
        tmp_file_path = Path(save_im_path) / filename
        list_row = read_jsonl_as_list_of_dict(tmp_file_path)
        # append contents to final metadata file
        with open(Path(save_im_path) / "metadata.jsonl", "a+") as f:
            for item in list_row:
                f.write(json.dumps(item) + "\n")
        # delete temp file
        os.remove(tmp_file_path)
    logging.info(
        f" Final metadata len: {len(read_jsonl_as_list_of_dict(Path(save_im_path) / 'metadata.jsonl'))}"
    )
    logging.info(f" Number of images generated: {len(os.listdir(save_im_path))-1}")
