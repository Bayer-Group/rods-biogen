"""This module contains all the logic to transform a dataset of images into SD-VAE latents."""
from functools import partial
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import einops
import numpy as np
from datasets import Array3D, ClassLabel, Value
from datasets.dataset_dict import DatasetDict

from biogen.autoencoder import SDVae
from biogen.latents import pil_to_input_img


def image_to_tensor(
    examples: Dict[str, Any],
    vae: SDVae,
    size: tuple = (512, 512),
    image_key: str = "image",
    img_array_key: Optional[str] = None,
) -> Dict[str, np.ndarray]:
    """
    Transform a batch of examples into their latent representation.

    This function is mean to be used as a batch transformation to preprocess a
    Huggingface :class:`Dataset` or :class:`DatasetDict`.

    Args:
        examples (Dict[str, Any]): Dictionary containing a batch of dataset samples.
        vae (SDVae): Autoencoder used to calculte the latents.
        size (tuple, optional): Size of the input image fed to the encoder. Defaults to (512, 512).
        image_key (str, optional): Name of the key containing the dataset images.
            Defaults to "image".
        img_array_key (Optional[str], optional): Key that will be used to store the
            numpy array representing the image fed to the autoencoder in RGB (b, h, w, c)
            format. If None the image tensors will not be saved. Defaults to None.

    Returns:
        Dict[str, np.ndarray]: Dictionary containing the tensor representing the embeddings \
        in its "latent" key.
    """
    input_images = np.array([pil_to_input_img(x, size) for x in examples[image_key]])
    if img_array_key is not None:
        examples[img_array_key] = input_images
    latents = vae.encode(input_images)
    examples["latent"] = einops.asnumpy(latents).astype(np.float32)
    if isinstance(examples[image_key][0], dict):
        examples["path"] = [x["path"] for x in examples[image_key]]
    return examples


def create_latents_dataset(
    dataset: DatasetDict,
    vae: SDVae,
    size: Tuple[int, int] = (512, 512),
    image_key: str = "image",
    img_array_key: Optional[str] = None,
    n_channels: int = 3,
    drop_images: bool = True,
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
        vae (SDVae): Autoencoder used to calculte the latents.
        size (Tuple[int, int], optional): Size of the input image fed to the encoder.
            Defaults to (256, 256).
        image_key (str, optional): Name of the key containing the dataset images.
            Defaults to "image".
        img_array_key (Optional[str], optional): Key that will be used to store the
            numpy array representing the image fed to the autoencoder in RGB (b, h, w, c)
            format. If None the image tensors will not be saved. Defaults to None.
        n_channels (int, optional): Number of channels of the input images. Currently
            supports only 3 channels. Defaults to 3.
        drop_images (bool, optional): Remove the PIL images from the processed dataset.
            Defaults to False.
        output_dir (Optional[Union[Path, str]], optional): Directory where the latents
            dataset will be stored. Defaults to None.
        kwargs: Passed to `dataset.map`. It cannot contain the arguments `batched`,
            `features`, nor `remove_columns`.

    Returns:
        DatasetDict: :class:`Dataset` containing the image latents in its "latents" key.
    """
    _ds = dataset["train"] if isinstance(dataset, DatasetDict) else dataset
    feats = _ds.features.copy()
    remove_columns = image_key if drop_images else None
    if drop_images:
        del feats[image_key]
    feats["latent"] = Array3D(shape=(size[0] // 8, size[1] // 8, 4), dtype="float32")
    if img_array_key is not None:
        feats[img_array_key] = Array3D(
            shape=(size[0], size[1], n_channels), dtype="float32"
        )
    if isinstance(_ds[image_key][0], dict):  # Images not yet decoded
        feats["path"] = Value(dtype="string", id=None)
    feats["label"] = ClassLabel(num_classes=2, names=["healthy", "cancer"])
    partial_kwargs = dict(
        size=size, image_key=image_key, vae=vae, img_array_key=img_array_key
    )
    transform = partial(image_to_tensor, **partial_kwargs)
    ds = dataset.map(
        transform,
        batched=True,
        features=feats,
        remove_columns=remove_columns,
        **kwargs,
    )
    if output_dir is not None:
        ds.save_to_disk(str(output_dir))
    return ds
