"""
This module contains the functionality for making perturbations in the latent space.

It contains three different types of functions:

* Functions for dealing with type conversions from Images into latent representations.
* Functions for running a linear interpolation between latents.
* Functions for computing a random walk in latent space.

"""
from typing import Any, Dict, List, Optional, Tuple, Union

import einops
import numpy as np
import pandas as pd
import torch
import umap
from PIL import Image

from biogen.autoencoder import SDVae


def pil_to_input_img(x: Union[Any, Dict[str, str]], size: Tuple[int]) -> np.ndarray:
    """
    Transform a :class:`PIL.Image` into a numpy array in RGB format with its values in\
    the [-1, 1] range.

    Args:
        x (Image): Image to be processed.
        size (Tuple[int]): Size of the image tensor that will be returned.

    Returns:
        np.ndarray: Array encoding the provided image as an RGB tensor of floats in the\
        [-1, 1] range and shape (h, w, c).
    """
    if isinstance(x, dict):
        x = Image.open(x["path"])
    x = np.array(x.resize(size, resample=Image.Resampling.LANCZOS).convert("RGB"))
    x = ((x / 255) - 0.5) * 2
    return x


def pil_to_latent(
    image: Image, vae: SDVae, size: Tuple[int, int] = (256, 256)
) -> torch.Tensor:
    """
    Calculate the latent representation of an image in PIL format.

    Args:
        image (Image): Target image.
        vae (SDVae): Encoder to be used to calculate the latent representations.
        size (Tuple[int, int], optional): Size of the input image. Defaults to (256, 256).

    Returns:
        torch.Tensor: Tensor with shape (c, h, w) containing the latent representation of the \
        target image.
    """
    image_np = pil_to_input_img(image, size)[None]
    return vae.encode(image_np)


def latents_to_embedding_input(latents: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
    """
    Reorder the latent representations into shape (b, val) so they can be passed to an embedding \
    algorithm.

    Args:
        latents (Union[np.ndarray, torch.Tensor]): Latent vectors to be embedded.

    Returns:
        np.ndarray: Array represeting the latents as a two dimensional matrix.
    """
    latents = einops.asnumpy(latents).astype(dtype="float32")
    latents_vec = einops.rearrange(latents, "b ... -> b (...)")
    return latents_vec


def embed_latents(
    latents: Union[np.ndarray, torch.Tensor],
    mapper,
    dataset_latents: Optional[np.ndarray] = None,
    fit=True,
) -> Union[Tuple[np.ndarray], Tuple[np.ndarray, np.ndarray]]:
    """
    Calculate the embeddings of the target latent representations.

    Args:
        latents (Union[np.ndarray, torch.Tensor]): Tensor containing a batch of latent
            representations of images.
        mapper (_type_): Model for calculating the embeddings.
        dataset_latents (Optional[np.ndarray], optional): To be deprecated. Defaults to None.
        fit (bool, optional):If True fit the mapper to the provided data. Otherwise
            transform the data assuming that mapper is a pretrained model. Defaults to True.

    Returns:
        Union[Tuple[np.ndarray], Tuple[np.ndarray, np.ndarray]]: If dataset_latents is None \
        return a tuple containing an array of the embeddings of the provided \
        latent representations. If dataset_latents is not None return a two element \
        tuple containing the embeddings of the latents and the embeddings of the dataset_latents.
    """
    latents_vec = latents_to_embedding_input(latents)
    if dataset_latents is not None:
        latents_vec = np.vstack([dataset_latents, latents_vec])
    embeddings = (
        mapper.fit_transform(latents_vec) if fit else mapper.transform(latents_vec)
    )
    if dataset_latents is not None:
        latents_size = latents.shape[0]
        latents_emb = embeddings[-latents_size:]
        dataset_emb = embeddings[:-latents_size]
        return latents_emb, dataset_emb
    return (embeddings,)


def latents_to_images(
    x: Union[np.ndarray, torch.Tensor],
    vae: SDVae,
    batch_size: int = 2,
) -> List[np.ndarray]:
    """
    Reconstruct an array of latent representations into images.

    Args:
        x (Union[np.ndarray, torch.Tensor]): Array of latent representations.
        vae (SDVae): Autoencoder used to reconstruct the latents.
        batch_size (int, optional): How many images to reconstruct at once. Defaults to 2.

    Returns:
        List[np.ndarray]: List containing the reconstructed images as RGB uint8 \
        numpy arrays with dimensions (h, w, c).
    """
    all_images = []
    for i in range(x.shape[0] // batch_size):
        start = batch_size * i
        end = start + batch_size
        images = vae.tensor_as_image(x[start:end], decode=True)
        all_images.extend(images)
    return all_images


def index_to_image(
    df: pd.DataFrame, ix: int, size: Tuple[int, int] = (512, 512)
) -> Image:
    """
    Return a PIL image representing the original image indexed in the target DataFrame.

    Args:
        df (pd.DataFrame): DataFrame containing the post-processing data of a given dataset.
        ix (int): Index of the target image.
        size (Tuple[int, int], optional): Size of the returned PIL Image. Defaults to (512, 512).

    Returns:
        Image: PIL Image corresponding to the image data in the target column of the DataFrame.
    """
    return Image.open(df.iloc[ix]["image"]).resize(size)


def interpolate_latents(
    start_latent: Union[np.ndarray, torch.Tensor],
    end_latent: Union[np.ndarray, torch.Tensor],
    steps=50,
) -> torch.Tensor:
    """
    Calculate the linear interpolation between the provided latent vectors.

    Args:
        start_latent (Union[np.ndarray, torch.Tensor]): Latent vector containing the starting
            image in the interpolation process.
        end_latent (Union[np.ndarray, torch.Tensor]): Latent vector containing the final
            image in the interpolation process.
        steps (int, optional): Number of interpolation steps between the images. Defaults to 50.

    Returns:
        torch.Tensor: tensor of shape (steps, c, h, w) containing the interpolations. The first \
        element in the batch dimension corresponds to start_latent, and the last element \
        corresponds to end_latent.
    """
    # Transform latent vector to torch with dims (b, c, h, w)
    if isinstance(start_latent, np.ndarray):
        start_latent = torch.from_numpy(start_latent)
    if isinstance(end_latent, np.ndarray):
        end_latent = torch.from_numpy(end_latent)
    x_start = start_latent[None] if len(start_latent.shape) == 3 else start_latent
    x_end = end_latent[None] if len(end_latent.shape) == 3 else end_latent
    # Repeat latent vectors along column
    lat_1 = torch.cat([x_start for _ in range(steps)], 0)
    lat_2 = torch.cat([x_end for _ in range(steps)], 0)
    # Linear interpolation with varying weights
    w = torch.from_numpy(np.linspace(0, 1, steps))
    w = w.reshape(-1, 1, 1, 1).type_as(lat_1)
    interp = (1 - w) * lat_1 + w * lat_2
    return interp


def interpolate_images(
    start_image: Image,
    end_image: Image,
    vae: SDVae,
    steps: int = 50,
    size: Tuple[int, int] = (512, 512),
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Run a latent linear interpolation between the provided PIL Images.

    This function encodes the provided images into their respective latent representations
    and calculates the linear interpolation between them.

    Args:
        start_image (Image): Initial image to compute the linear interpolation.
        end_image (Image): Final image of the interpolation process.
        vae (SDVae): Autoencoder used for encoding the images into latents.
        steps (int, optional): Number of interpolation steps between the images. Defaults to 50.
        size (Tuple[int, int], optional): Size of the input images fed to the encoder.
            Defaults to (512, 512).

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: A tuple of three numpy arrays containing the \
        latent interpolations, the latent representation of start_image, and the latent \
        representation of the end_image.
    """
    start_np = pil_to_input_img(start_image, size)[None]
    end_np = pil_to_input_img(end_image, size)[None]
    vae_input = np.vstack([start_np, end_np])
    latents = vae.encode(vae_input)
    start_latent = latents[0]
    end_latent = latents[1]
    interp = interpolate_latents(start_latent, end_latent, steps=steps)
    return interp, start_latent, end_latent


def run_interpolation(
    start_img: Image,
    end_img: Image,
    vae: SDVae,
    mapper=None,
    dataset_latents: Optional[np.ndarray] = None,
    fit=False,
    steps=50,
    size: Tuple[int, int] = (256, 256),
) -> Union[Tuple[np.ndarray, np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
    """
    Compute the latent interpolation between two images.

    Optionally, compute the embeddings of the interpolation.

    Args:
        start_img (Image): Initial image to compute the linear interpolation.
        end_img (Image): Final image of the interpolation process.
        vae (SDVae): Autoencoder used for encoding the images into latents.
        mapper (Any, optional): Model to transform latents into embeddings. Defaults to None.
        dataset_latents (Optional[np.ndarray], optional): Vector of latents used to fit the
            embedding model in addition to the random walk latents. Defaults to None.
        fit (bool, optional):If True fit the provided mapper, otherwise only use it to
            transform the latents. Defaults to False.
        steps (int, optional): Number of interpolation steps between the images. Defaults to 50.
        size (Tuple[int, int], optional): Size of the input images fed to the encoder.
            Defaults to (256, 256).

    Returns:
        Union[Tuple[np.ndarray, np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:  If \
        dataset_latents is None return a tuple containing (interpolation latents, \
        interpolation embeddings). If dataset_latents is not None return a tuple containing \
        (interpolation latents, interpolation embeddings, dataset_latents embeddings).
    """
    if mapper is None:
        mapper = umap.UMAP(random_state=160290, n_epochs=500).fit(dataset_latents)
    interp, *_ = interpolate_images(start_img, end_img, vae, steps=steps, size=size)
    interp_emb, *dataset_emb = embed_latents(
        latents=interp,
        dataset_latents=dataset_latents,
        mapper=mapper,
        fit=fit,
    )
    return (interp, interp_emb, dataset_emb[0]) if dataset_emb else (interp, interp_emb)


def interpolate_indices(
    start_ix: int,
    end_ix: int,
    df: pd.DataFrame,
    vae: SDVae,
    mapper: Optional[Any] = None,
    dataset_latents: Optional[np.ndarray] = None,
    steps: int = 250,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Run a linear interpolation on the images located in the provided indices of the plot dataframe.

    Args:
        start_ix (int): Index of the row that contains the starting image.
        end_ix (int): Index of the row that contains the starting image.
        df (pd.DataFrame): Plot dataframe containing the dataset information.
        vae (SDVae): Autoencoder used for encoding the images into latents.
        mapper (Optional[Any], optional): Model to transform latents into embeddings.
            Defaults to None.
        dataset_latents (Optional[np.ndarray], optional): Vector of latents used to fit the
            embedding model in addition to the random walk latents. Defaults to None.
        steps (int, optional): Number of interpolation steps between the images. Defaults to 50.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Tuple containing numpy arrays of (interpolations, \
        interpolation embeddings).
    """
    start_img = index_to_image(df, start_ix)
    end_img = index_to_image(df, end_ix)
    interp, interp_emb, *_ = run_interpolation(
        start_img=start_img,
        end_img=end_img,
        dataset_latents=dataset_latents,
        mapper=mapper,
        steps=steps,
        vae=vae,
    )
    return interp, interp_emb


def interpolate_sequence(
    indexes: List[int],
    df: pd.DataFrame,
    vae: SDVae,
    mapper: Optional[Any] = None,
    dataset_latents: Optional[np.ndarray] = None,
    steps: int = 250,
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    Interpolate a sequence of images given a sequence of plot dataframe indexes.

    The linear interpolations are calculated for each pair of consecutive indexes.

    Args:
        indexes (List[int]): List of indexes of the rows containing the images to be interpolated.
        df (pd.DataFrame): Plot dataframe containing the dataset information.
        vae (SDVae): Autoencoder used for encoding the images into latents.
        mapper (Optional[Any], optional): Model to transform latents into embeddings.
            Defaults to None.
        dataset_latents (Optional[np.ndarray], optional): Vector of latents used to fit the
            embedding model in addition to the random walk latents. Defaults to None.
        steps (int, optional): Number of interpolation steps between the images. Defaults to 250.

    Returns:
        Tuple[List[np.ndarray], List[np.ndarray]]: Tuple of two lists that contain the latent \
        values of the interpolation and their respective embeddings.
    """
    interpolations = []
    embeddings = []
    for i in range(len(indexes)):
        start_i, end_i = i, (0 if i == len(indexes) - 1 else i + 1)
        start_ix, end_ix = indexes[start_i], indexes[end_i]
        interp, embedding = interpolate_indices(
            start_ix,
            end_ix,
            df,
            vae,
            mapper,
            dataset_latents,
            steps=steps,
        )
        interpolations.append(interp)
        embeddings.append(embedding)
    return interpolations, embeddings


def compute_random_walk(
    latent: np.ndarray, sigma: float = 0.01, steps: int = 100
) -> np.ndarray:
    """
    Compute a random walk in latent space using gaussian perturbations.

    Args:
        latent (np.ndarray): Latent vector used as a starting point of the random walk.
        sigma (float, optional): Standard deviation of each walk step. Defaults to 0.01.
        steps (int, optional): Number of steps of the generated random walk. Defaults to 100.

    Returns:
        np.ndarray: Array containing the generated random walk. Its first element corresponds \
        to the latent vector used as the walk starting position.
    """
    # Assumes image shape (c, w, h)
    latent = (
        latent if len(latent.shape) == 3 else latent[0]
    )  # Remove batch dim if present
    start_val = latent[None]
    noise = np.random.normal(loc=0, scale=sigma, size=(steps,) + latent.shape)
    walk = start_val + noise.cumsum(0)
    return np.vstack([start_val, walk])


def run_random_walk(
    image: Image,
    vae: SDVae,
    mapper=None,
    dataset_latents: Optional[np.ndarray] = None,
    sigma: float = 0.1,
    steps: int = 250,
    fit: bool = False,
    size: Tuple[int, int] = (256, 256),
) -> Union[
    Tuple[np.ndarray, np.ndarray, np.ndarray],
    Tuple[np.ndarray, np.ndarray],
    Tuple[np.ndarray],
]:
    """
    Compute a random walk starting at the latent representation of the provided image.

    Optionally, it allows to compute the embeddings of the random walk if a mapper is provided.

    Args:
        image (Image): Image which latent will be used as the starting point of the random walk.
        vae (SDVae): Autoencoder used for encoding the images into latents.
        mapper (_type_, optional): Model to transform latents into embeddings. Defaults to None.
        dataset_latents (Optional[np.ndarray], optional): Vector of latents used to fit the
            embedding model in addition to the random walk latents. Defaults to None.
        sigma (float, optional): Standard deviation of each walk step. Defaults to 0.1.
        steps (int, optional): Number of steps of the generated random walk. Defaults to 250.
        fit (bool, optional): If True fit the provided mapper, otherwise only use it to
            transform the latents. Defaults to False.
        size (Tuple[int, int], optional): Input size of the image passed to the encoder.
            Defaults to (256, 256).

    Returns:
        Union[Tuple[np.ndarray, np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray],
        Tuple[np.ndarray]]: If mapper is None return a single element tuple containing \
        the latents of the random walk. If mapper is not None return also the \
        embeddings of the latents. If mapper is not None and dataset_latents is not None return \
        a tuple containing (walk_latents, walk_embeddings, dataset_latents embeddings).
    """
    latent = einops.asnumpy(pil_to_latent(image, vae, size=size))[0]
    walk = compute_random_walk(latent, steps=steps, sigma=sigma)
    if mapper is None:
        return (walk,)
    walk_emb, *dataset_emb = embed_latents(
        walk,
        dataset_latents=dataset_latents,
        mapper=mapper,
        fit=fit,
    )
    return (walk, walk_emb, dataset_emb[0]) if dataset_emb else (walk, walk_emb)
