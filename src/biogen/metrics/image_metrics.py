"""This module contains all the logic to transform a dataset of images into image metrics."""

from functools import partial
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import einops
import numpy as np
import torch
from datasets.dataset_dict import DatasetDict
from PIL import Image
from piq import BRISQUELoss, TVLoss

from biogen.dataset_utils import cast_image
from biogen.images import reformat_img

NO_REFERENCE_METRICS = {
    # "total_variation_l2": TVLoss(reduction="none", norm_type="l2"),
    "total_variation_l1": TVLoss(reduction="none", norm_type="l1"),
    "brisque": BRISQUELoss(reduction="none"),
}


def image_to_metrics(
    examples: Dict[str, Any],
    metrics,
    size: tuple = (512, 512),
    image_key: str = "image",
) -> Dict[str, np.ndarray]:
    """
    Transform a batch of examples into their latent representation.

    This function is mean to be used as a batch transformation to preprocess a
    Huggingface :class:`Dataset` or :class:`DatasetDict`.

    Args:
        examples (Dict[str, Any]): Dictionary containing a batch of dataset samples.
        metrics (Dict[str, Any]): Dictionary containing the metrics to be calculated.
        size (tuple, optional): Size of the input image fed to the encoder. Defaults to (512, 512).
        image_key (str, optional): Name of the key containing the dataset images.
            Defaults to "image".

    Returns:
        Dict[str, np.ndarray]: Dictionary containing the tensor representing the embeddings \
        in its "latent" key.
    """
    img_paths = [x["path"] for x in examples[image_key]]
    output = {"path": img_paths}
    input_images = np.array(
        [
            reformat_img(
                Image.open(x),
                out_fmt="norm",
                backend="np",
                size=size,
                channels_first=True,
            )
            for x in img_paths
        ],
    )
    input_images = torch.from_numpy(input_images)
    for name, metric in metrics.items():
        values = metric(input_images)
        output[name] = einops.asnumpy(values).flatten().astype(np.float32)
    return output


def create_metrics_dataset(
    dataset: DatasetDict,
    metrics: Dict[str, Any] = NO_REFERENCE_METRICS,
    size: Tuple[int, int] = (512, 512),
    image_key: str = "image",
    output_dir: Optional[Union[Path, str]] = None,
    **kwargs,
) -> DatasetDict:
    """
    Transform the images of a dataset into their latent representation.

    This functions allow to preprocess an image dataset and transform it into latent
    representations using an autoencoder. It will create a dataset with a "latent" key
    containg an array representing the latents of each image.

    Args:
        dataset (DatasetDict): :class:`Dataset` or :class:`DatasetDict` representing the dataset
            of images to be encoded as latents.
        metrics (Dict[str, Any]): Dictionary containing the image metrics.
        size (Tuple[int, int], optional): Size of the input image fed to the encoder.
            Defaults to (256, 256).
        image_key (str, optional): Name of the key containing the dataset images.
            Defaults to "image".
        output_dir (Optional[Union[Path, str]], optional): Directory where the latents
            dataset will be stored. Defaults to None.
        kwargs: Passed to `dataset.map`. It cannot contain the arguments `batched`,
            `features`, nor `remove_columns`.

    Returns:
        DatasetDict: :class:`Dataset` containing the image latents in its "latents" key.
    """
    # if has_decoded_image(dataset):
    dataset = cast_image(dataset, decode=False, name=image_key)
    _ds = dataset["train"] if isinstance(dataset, DatasetDict) else dataset
    feats = _ds.features.copy()
    partial_kwargs = dict(size=size, image_key=image_key, metrics=metrics)
    transform = partial(image_to_metrics, **partial_kwargs)
    ds = dataset.map(
        transform,
        batched=True,
        remove_columns=list(feats.keys()),
        **kwargs,
    )
    if output_dir is not None:
        ds.save_to_disk(str(output_dir))
    return ds
