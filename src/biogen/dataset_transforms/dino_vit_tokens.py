"""This module contains all the logic to transform a dataset of images into DINO-ViT tokens."""

import logging
import os
from functools import partial
from pathlib import Path
from typing import Any, Dict, Optional, Union

import cv2
import einops
import numpy as np
import torch
from datasets import Array2D, ClassLabel
from datasets import Image as DatasetImage
from datasets import Value
from datasets.dataset_dict import DatasetDict
from PIL import Image
from sklearn.preprocessing import MinMaxScaler
from torchvision.transforms.functional import to_tensor
from transformers import AutoFeatureExtractor, AutoModel, TensorType

from biogen.dataset_utils import cast_image, has_decoded_image, load_dataset

logger = logging.getLogger(__name__)
DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print("DEVICE:", DEVICE)
torch.cuda.empty_cache()


def PIL_to_opencv(im_pil):
    im_arr = np.array(im_pil)
    open_cv_image = im_arr[:, :, ::-1].copy()
    return open_cv_image


def im_to_canny_edges(im, low_threshold, high_threshold, L2gradient=True):
    """Convert rgb image in PIL format to canny filtered edge detector images with same grayscale on all 3 channels."""
    im_arr = np.array(im)
    open_cv_image = PIL_to_opencv(im)

    # Get Canny edges on the 2 apertures:
    canny_image_arr_3 = cv2.Canny(
        im_arr, low_threshold, high_threshold, apertureSize=3, L2gradient=L2gradient
    )
    canny_image_arr_7 = cv2.Canny(
        im_arr, low_threshold, high_threshold, apertureSize=7, L2gradient=L2gradient
    )

    # Get otsu BW image
    gray = cv2.cvtColor(open_cv_image, cv2.COLOR_BGR2GRAY)
    _, im_bw = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU)
    im_bw = np.array(im_bw)

    # Join all 3 maps into single RGB image
    rgb_arr = np.stack([im_bw, canny_image_arr_3, canny_image_arr_7], axis=-1)

    # Convert the NumPy array back to PIL image
    rgb_canny_image_pil = Image.fromarray(np.uint8(rgb_arr))

    return rgb_canny_image_pil


def normalise_token_dataset(dataset, feature_key):
    """Normalise token dataset."""
    # ... Fit scaler on both sets features
    # Extract the feature vectors from the datasets
    feature_vectors = [example[feature_key] for example in dataset]
    # Reshape to rmeove extra dimension
    feature_vectors = np.reshape(feature_vectors, (np.shape(feature_vectors)[0], -1))
    # Create a MinMaxScaler object
    scaler = MinMaxScaler()
    # Fit the scaler to the feature vectors
    scaler_fitted = scaler.fit(feature_vectors)

    # Define function to be used in map
    def scale_feature(example, feature_key=feature_key, scaler=scaler_fitted):
        # Reshape for min max scaler
        reshaped_vecs = np.reshape(
            example[feature_key], (np.shape(example[feature_key])[0], -1)
        )
        reshaped_vecs = scaler.transform(reshaped_vecs)
        shape__ = np.shape(reshaped_vecs)
        # Reshape back to riginal dataset shape
        reshaped_vecs = np.reshape(reshaped_vecs, (shape__[0], 1, shape__[-1]))
        # Allocate to dataset
        example[feature_key] = reshaped_vecs
        return example

    # Process dataset
    dataset = dataset.map(scale_feature, batched=True)
    return dataset


def image_to_token(
    examples: Dict[str, Any],
    feature_extractor: Any,
    dino_vit_model: Any,
    image_key: str = "image",
    canny_edge_based: bool = False,
    low_threshold: int = 5,
    high_threshold: int = 200,
    device: torch.DeviceObjType = DEVICE,
) -> Dict[str, np.ndarray]:
    """Transform a batch of examples into their token representation.

    Args:
        examples (Dict[str, Any]): Dictionary containing a batch of dataset samples.
        feature_extractor (Any): Feature extractor object for preprocessing
            images before feeding them into the vision transformer model.
        dino_vit_model (Any): Vision transfromer model object to convert images
            into tokens.
        image_key (str, optional): Name of the key containing the dataset iamges.
            Defaults to "image".
        save_patch_tokens (bool, optional): Whether to also save the patch tokens.

    Returns:
        Dict[str, np.ndarray]: Dictionary containing the tensor representing the embeddings \
        in its class token key and each patch token key.
    """
    # Feature extraction. Needs PIL images so convert image colum to PIL
    input_images = [Image.open(x["path"]).convert("RGB") for x in examples[image_key]]

    # convert rgb images to canny edge filtering image with same grayscale image at each rgb channel
    if canny_edge_based:
        input_images = [
            im_to_canny_edges(x, low_threshold, high_threshold) for x in input_images
        ]

    input_images = [to_tensor(img).to(device) for img in input_images]
    input_features = feature_extractor(
        images=input_images, return_tensors=TensorType.PYTORCH
    )
    input_features = input_features.to(device)
    # Run model
    with torch.no_grad():
        outputs = dino_vit_model(**input_features)
    # Get class token from output ensuring shape matches initialised Array2D shape
    class_token_batch = outputs.last_hidden_state[:, 0, :][:, None, :]
    class_token_batch = einops.asnumpy(class_token_batch).astype(
        np.float32
    )  # bs, 1, 768
    examples["class_token"] = class_token_batch
    if isinstance(examples[image_key][0], dict):  # Images not yet decoded into
        examples["path"] = [x["path"] for x in examples[image_key]]
    return examples


def create_token_dataset(
    dataset: DatasetDict,
    dino_vit_name: str = "dino-vitb8",
    image_key: str = "image",
    drop_images: bool = True,
    output_dir: Optional[Union[Path, str]] = None,
    label_column: str = None,
    canny_edge_based=False,
    low_threshold: int = 5,
    high_threshold: int = 200,
    **kwargs,
) -> DatasetDict:
    """Transform a batch of examples into their dino-vit token representation.

    This function is mean to be used as a batch transformation to preprocess a
    Huggingface :class:`Dataset` or :class:`DatasetDict`.

    Args:
        dataset (DatasetDict): :class:`Dataset` or :class:`DatasetDict` representing the dataset
            of images to be encoded as latents.
        dino_vit_name (str): Name of the dino-vit model to use.
        image_key (str, optional): Name of the key containing the dataset images.
            Defaults to "image".
        drop_images (bool, optional): Remove the PIL images from the processed dataset.
            Defaults to False.
        output_dir (Optional[Union[Path, str]], optional): Directory where the token
            dataset will be stored. Defaults to None.
        save_patch_tokens (bool, optional): Whether to also save the patch tokens.
        kwargs: Passed to `dataset.map`. It cannot contain the arguments `batched`,
            `features`, nor `remove_columns`.

    Returns:
        DatasetDict: :class:`Dataset` containing the image tokens in its
            "class_token" and "patches_token" key.
    """
    if has_decoded_image(dataset):
        dataset = cast_image(dataset, decode=False, name=image_key)
    _ds = dataset["train"] if isinstance(dataset, DatasetDict) else dataset
    feats = _ds.features.copy()
    # Get number of classes and their string id
    if label_column is not None:
        if isinstance(_ds.features[label_column], ClassLabel):
            class_list = _ds.features[label_column].names
        else:
            class_list = np.unique(_ds[label_column]).tolist()
        num_classes = len(class_list)
        feats[label_column] = ClassLabel(num_classes=num_classes, names=class_list)

    remove_columns = image_key if drop_images else None
    if drop_images:
        del feats[image_key]
    # TODO(guillemdb): Adapt sizes to small dino models
    feats["class_token"] = Array2D(shape=(1, 768), dtype="float32")
    if isinstance(_ds[image_key][0], dict):  # Images not yet decoded
        feats["path"] = Value(dtype="string", id=None)

    # Initialise feature extractor object
    feature_extractor = AutoFeatureExtractor.from_pretrained(
        f"facebook/{dino_vit_name}", device=DEVICE
    )

    # Intialise model
    model = AutoModel.from_pretrained(f"facebook/{dino_vit_name}").eval()
    model.to(DEVICE)
    partial_kwargs = dict(
        image_key=image_key,
        feature_extractor=feature_extractor,
        dino_vit_model=model,
        canny_edge_based=canny_edge_based,
        low_threshold=low_threshold,
        high_threshold=high_threshold,
    )
    transform = partial(image_to_token, **partial_kwargs)
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


def convert_dataset_to_tokens(
    dataset_name,
    path: Union[str, Path],
    model_id: str = "dino-vitb8",
    label_key: Union[str, None] = "label",
    image_key: str = "image",
    canny_edge_based: bool = True,
    low_threshold: int = 100,
    high_threshold: int = 200,
    batch_size: int = 92,
):
    # Get image dataset path and output path form input image dir path
    token_dir_name = "token_edge_based" if canny_edge_based else "token"
    dataset_path = Path(path) / dataset_name / "image"
    output_path = dataset_path.parent / token_dir_name
    # Create output dir if needed
    os.makedirs(output_path, exist_ok=True)
    logger.info(" ################### Parameters ################# ")
    logger.info(f"DINO-ViT model name = {model_id}")
    logger.info(f"Image DS input directory  = {dataset_path}")
    logger.info(f"Token DS saving directory = {output_path}")
    logger.info(f"Batch size = {batch_size}")
    logger.info(f"Label column = {label_key}")
    logger.info(" ############################################ ")
    # Load dataset to be converted to tokens
    ds = load_dataset(
        name=dataset_name,
        mode="image",
        path=path,
    ).cast_column(image_key, DatasetImage(decode=False))
    logger.info(ds)
    # Create and save tokens
    logger.info(ds["train"].features)
    create_token_dataset(
        dataset=ds,
        dino_vit_name=model_id,
        output_dir=output_path,
        batch_size=batch_size,
        label_column=label_key,
        canny_edge_based=canny_edge_based,
        low_threshold=low_threshold,
        high_threshold=high_threshold,
    )
    return output_path
