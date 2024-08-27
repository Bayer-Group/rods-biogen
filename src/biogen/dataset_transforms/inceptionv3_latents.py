import logging
import os
from timeit import default_timer as timer

import numpy as np
import torch
from datasets import Array2D, Dataset, DatasetDict, Features, concatenate_datasets
from PIL import Image
from piq import FID
from piq.feature_extractors import InceptionV3

from biogen.dataset_utils import load_image_dataset

logger = logging.getLogger(__name__)
DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print("DEVICE:", DEVICE)
torch.cuda.empty_cache()


def tf(examples):
    images = []
    for p in examples.pop("image"):
        if isinstance(p, dict):
            p = Image.open(p["path"]).convert("RGB")
        if isinstance(p, Image.Image):
            p = np.array(p.convert("RGB"))
        images.append(p)
    imgs = np.array(images)
    if np.max(imgs) > 1:
        imgs = imgs / 255
    examples["images"] = np.moveaxis(imgs, [-3, -2], [-2, -1])
    return examples


def convert_dataset_to_inceptionv3_latents(
    dataset_path,
    image_key: str = "image",
    drop_images: bool = True,
    batch_size: int = 512,
    save_ds: bool = True,
):
    """Ensure loaded dataset has at least on train split"""
    # Get image dataset path and output path form input image dir path
    token_dir_name = "inceptionv3_tokens"
    if not dataset_path.name == "image":
        raise ValueError("Image dataset path must end with '/image'")
    output_path = dataset_path.parent / token_dir_name
    # Create output dir if needed
    os.makedirs(output_path, exist_ok=True)
    logger.info(" ################### Parameters ################# ")
    logger.info(f"Image DS input directory  = {dataset_path}")
    logger.info(f"Token DS saving directory = {output_path}")
    logger.info(" ############################################ ")
    # Load dataset to be converted to tokens
    ds = load_image_dataset(dataset_path).with_transform(tf)
    if isinstance(ds, DatasetDict):
        splits = list(ds.keys())
    else:
        raise ValueError("Dataset needs at least one split")
    logger.info(ds)
    # Instantiate metric
    metric = FID()
    model = InceptionV3()
    # Compute all InceptionV3 scores
    logger.info(f"n cpus available: {os.cpu_count()}")
    inception_ds_dict = {}
    for split in splits:
        logger.info(f"split = {split}")
        _ds = ds[split]
        logger.info(_ds.features)
        dataloader = torch.utils.data.DataLoader(
            _ds,
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=os.cpu_count(),
        )
        logger.info("Computing the latents ... ")
        start = timer()
        inception_tokens = metric.compute_feats(dataloader, model, device=DEVICE)
        end = timer()
        logger.info(
            f"Generated {len(ds['train'])} latents in {(end-start) // 60} minutes"
        )
        logger.info(np.shape(inception_tokens))
        # Adding new columns to dataset and drop image columns if flagged
        features = Features(
            {"inceptionv3_latent": Array2D(shape=(1, 2048), dtype="float32")}
        )
        inception_tokens_ds = Dataset.from_dict(
            {"inceptionv3_latent": inception_tokens[:, None, :]}, features=features
        )
        inception_tokens_ds_full = concatenate_datasets(
            [inception_tokens_ds, _ds], axis=1
        )
        if drop_images:
            inception_tokens_ds_full = inception_tokens_ds_full.remove_columns(
                [image_key]
            )
        # Add split ds to dataste dict
        inception_ds_dict[split] = inception_tokens_ds_full
    inception_ds = DatasetDict(inception_ds_dict)
    if output_path is not None and save_ds:
        inception_ds.save_to_disk(str(output_path))
    return inception_tokens_ds_full
