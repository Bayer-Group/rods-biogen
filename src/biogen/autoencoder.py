"""This module contains the interfacing code to load and run the Stable Diffusion's autoencoder."""

import warnings
from typing import Dict, Optional, Tuple, Union

import einops
import numpy as np
import pytorch_lightning as pl
import torch
from diffusers.models.autoencoder_kl import AutoencoderKL, DiagonalGaussianDistribution

STABLE_DIFFUSION_PIPELINE_NAME = "CompVis/stable-diffusion-v1-4"


def load_sd_vae(
    name: str = STABLE_DIFFUSION_PIPELINE_NAME,
    half_precission=False,
    device: str = "cpu",
) -> AutoencoderKL:
    """
    Initialize the variational autoencoder from Stable Diffusion.

    Args:
        name (str, optional): String corresponding to the model name in the Huggingface registry.
            Defaults to STABLE_DIFFUSION_PIPELINE_NAME.
        half_precission (bool, optional): Use half precission revision when device is not CPU.
            Defaults to False.
        device (str, optional): Device of the initialized model. Defaults to "cpu".

    Returns:
        AutoencoderKL: Stable diffusion autoencoder.
    """
    params = dict(subfolder="vae", use_auth_token=True)
    if half_precission and device != "cpu":
        params["revision"] = "fp16"
        params["torch_dtype"] = torch.float16
    elif half_precission:
        warnings.warn(
            "Half precision only works in cuda devices. Loading model using float32."
        )

    return AutoencoderKL.from_pretrained(name, **params).to(device)


class SDVae(pl.LightningModule):
    """
    The :class:`SDVae` exposes the Stable Diffusion autoencoder's functionality.

    It also provides convenience methods for formatting data into the format that the
    model expects, and converting the model reconstructions back into RGB arrays.
    """

    def __init__(
        self,
        name: str = STABLE_DIFFUSION_PIPELINE_NAME,
        half_precission: bool = False,
        device: Union[str, torch.DeviceObjType] = "cpu",
        lpips_device: Optional[str] = None,
        model: AutoencoderKL = None,
        **kwargs,
    ) -> None:
        """
        Initialize a :class:`SDVae`.

        Args:
            name (str, optional): String corresponding to the model name in the
                Huggingface registry. Defaults to STABLE_DIFFUSION_PIPELINE_NAME.
            half_precission (bool, optional): Use half precission revision when device is not CPU.
            Defaults to False.
            device (device, optional): Device where the autoencoder will be placed.
                Defaults to "cpu".
            lpips_device (Optional[str], optional):  Device where the LPIPS loss will be placed.
                Defaults to "cpu".
            kargs: Passed to init_autoencoder.
            model (AutoencoderKL, optional): Pytorch module that will be used as the
                autoencoder instead of initializing the model using the HuggingFace registry.
                Defaults to None.
        """
        super().__init__()
        self.model = (
            self.init_autoencoder(
                name=name,
                half_precission=half_precission,
                device=device,
                **kwargs,
            )
            if model is None
            else model
        )

    def init_autoencoder(
        self,
        name: str = STABLE_DIFFUSION_PIPELINE_NAME,
        half_precission: bool = False,
        device: Union[str, torch.DeviceObjType] = "cpu",
        **kwargs,
    ) -> AutoencoderKL:
        """
        Initialize the variational autoencoder from Stable Diffusion.

        Args:
            name (str, optional): String corresponding to the model name in the Huggingface
                registry. Defaults to STABLE_DIFFUSION_PIPELINE_NAME.
            half_precission (bool, optional): Use half precission revision when device is not CPU.
                Defaults to False.
            device (str, optional): Device of the initialized model. Defaults to "cpu".
            kwargs: Passed to load_sd_vae.

        Returns:
            AutoencoderKL: Stable diffusion autoencoder.
        """
        return load_sd_vae(
            name=name, half_precission=half_precission, device=device, **kwargs
        )

    def encode(
        self,
        x: Union[torch.Tensor, np.ndarray],
        sample_posterior: bool = False,
        return_posterior: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, DiagonalGaussianDistribution]]:
        """
        Encode the provided images into latent representations.

        Args:
            x (Union[torch.Tensor, np.ndarray]): Tensor containing a batch of images with shape
                (b, c, h, w) or shape (b, h, w, c).
            sample_posterior (bool, optional): If True decode a sample from the
                encoder distribution, if False sample the mode of the distribution.
                Defaults to False.
            return_posterior (bool, optional): If True include the posterior
                distribution in the returned tuple. Defaults to False.

        Returns:
            Union[torch.Tensor, Tuple[torch.Tensor, DiagonalGaussianDistribution]]: Tuple \
            containing (latents, posterior) if sample_posterior is True. Otherwise \
            return a single element tuple containing the latents.
        """
        with torch.no_grad():
            x = self.tensor_to_input(x)
            posterior = self.model.encode(x).latent_dist
            if sample_posterior:
                latent = posterior.sample()
            else:
                latent = posterior.mode()
        return (latent, posterior) if return_posterior else latent

    def decode(self, latents: Union[torch.Tensor, np.ndarray]) -> torch.Tensor:
        """
        Decode the provided latents to return image reconstructions.

        Args:
            latents (Union[torch.Tensor, np.ndarray]): Vector with shape (b, c, h, w) containing
                the latent representations to be reconstructed.

        Returns:
            torch.Tensor: Reconstructed images with shape (b, c, h, w).
        """
        with torch.no_grad():
            latents = self.tensor_to_input(latents)
            dec = self.model.decode(latents).sample
        return dec

    def forward(
        self,
        x: Union[torch.Tensor, np.ndarray],
        sample_posterior: bool = False,
        return_posterior: bool = False,
        return_latents: bool = False,
    ) -> Union[
        Tuple[torch.Tensor],
        Tuple[torch.Tensor, torch.Tensor],
        Tuple[torch.Tensor, torch.Tensor, DiagonalGaussianDistribution],
    ]:
        """
        Run the autoencoder pipeline from input images and return reconstructions.

        Args:
            x (Union[torch.Tensor, np.ndarray]): Input tensor corresponding to a
                batch of images in the format (b, c, h, w).
            sample_posterior (bool, optional): If True decode a sample from the
                encoder distribution, if False sample the mode of the distribution.
                Defaults to False.
            return_posterior (bool, optional): If True include the posterior
                distribution in the returned tuple. Defaults to False.
            return_latents (bool, optional): If True include the latent vectors
                in the returned tuple. Defaults to False.

        Returns:
            Union[ Tuple[torch.Tensor], Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, \
            torch.Tensor, DiagonalGaussianDistribution], ]: Return a tuple containing \
            the relevant tensors calculated during the autoencoding process. By default \
            it returns a single element tuple containing the image reconstructions \
            in the same format as the input images. If return_latents is True, the returned \
            tuple is (reconstructions, latents). If return_posterior is True the returned \
            tuple is (reconstructions, latents), and if both return_posterior and \
            return_latents are True the returned tuple is (reconstructions, latents, posterior).

        """
        latents, posterior = self.encode(
            x,
            sample_posterior=sample_posterior,
            return_posterior=True,
        )
        reconstructions = self.decode(latents)
        return_data = (
            [reconstructions, latents] if return_latents else [reconstructions]
        )
        if return_posterior:
            return_data.append(posterior)
        return tuple(return_data)

    def tensor_to_input(
        self,
        x: Union[torch.Tensor, np.ndarray],
        dtype: torch.dtype = None,
        device: Union[str, torch.DeviceObjType] = None,
    ) -> torch.Tensor:
        """
        Convert a batch of images into a tensor with the appropriate shape placed \
        on the autoencoder's device.

        Args:
            x (Union[torch.Tensor, np.ndarray]): Input batch containing images or latents.
            dtype (torch.dtype, optional): Tensor data type after conversion. Defaults to None.
            device (Union[str, torch.DeviceObjType], optional): Target device. Defaults to None.

        Returns:
            torch.Tensor: Tensor with the appropriate shape and deviced to be passed to the \
            autoencoder.
        """
        # VAE expects shapes as (b, c, h, w). Images have 3 channels and latents have 4 channels
        channels_first = x.shape[1] == 3 or x.shape[1] == 4
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x)
        if not channels_first:
            x = einops.rearrange(x, "b h w c -> b c h w")
        device = self.model.device if device is None else device
        dtype = self.model.dtype if dtype is None else dtype
        return x.to(device=device, dtype=dtype)

    def tensor_as_image(
        self,
        x: Union[torch.Tensor, np.ndarray],
        decode: bool = False,
    ) -> np.ndarray:
        """
        Transform x into a numpy array representing RGB images with shape (b, h, w, c).

        Args:
            x (Union[torch.Tensor, np.ndarray]): Input array with shape (b, c, h, w).
            decode (bool, optional): If decode is False, the input array is treated
                as image reconstructions. If decode is False, the input array is
                treated as latent encodings and subsequently decoded. Defaults to False.

        Returns:
            np.ndarray: Batch of images with shape (b, h, w, c)
        """
        images = self.decode(x) if decode else x
        images = (images / 2 + 0.5).clamp(0, 1)  # from [-1, 1] range to [0, 1]
        images = einops.rearrange(images, "b c h w -> b h w c")
        images = einops.asnumpy(images * 255).round().astype("uint8")
        return images

    def evaluate_reconstructions(
        self,
        reconstructions: Union[torch.Tensor, np.ndarray],
        originals: Union[torch.Tensor, np.ndarray],
    ) -> Dict[str, np.ndarray]:
        """
        Calculate reconstruction metrics for a batch of images.

        It calculates the following metrics: LPIPS, RMSE, MAE.

        Args:
            reconstructions (Union[torch.Tensor, np.ndarray]): Batch of reconstructions
                generated by the decoder.
            originals (Union[torch.Tensor, np.ndarray]): Original images to compare
                with the reconstructions.

        Returns:
            Dict[str, np.ndarray]: Dictionary containing the name of the metrics \
            and their corresponding value.
        """
        vectorize = "b ... -> b (...)"
        reconstructions = self.tensor_to_input(reconstructions)
        originals = self.tensor_to_input(originals)
        recs_vec = einops.rearrange(reconstructions, vectorize)
        origs_rec = einops.rearrange(originals, vectorize)
        vec_diff = recs_vec - origs_rec
        rmse = torch.sqrt((vec_diff**2).mean(1).flatten())
        mae = torch.abs(vec_diff).mean(1).flatten()
        data = {
            "rmse": einops.asnumpy(rmse),
            "mae": einops.asnumpy(mae),
        }
        return data
