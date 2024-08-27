"""
This module contains utilities for transforming images into the different formarts required to \
analyze them.

We define 4 different image formats:
- **pil**: An instance of :class:`PIL.Image`. Used for storage and visualization.
- **int**: RGB array of integers in the range [0, 255]. Used for exporting and visualization.
- **norm**: Array of floats in the range [0, 1]. Used for plotting and calculating metrics.
- **dl**: Array of floats in the range [-1, 1]. Mean to be used as input for the SD VAE.

"""
from typing import Any, Dict, List, Optional, Tuple, Union

import einops
import numpy as np
import torch
from PIL import Image

TensorImage = Union[np.ndarray, torch.Tensor]
PilImage = Union[Image.Image, Any]
AnyImage = Union[np.ndarray, torch.Tensor, PilImage]
DEFAULT_SIZE: Tuple[int, int] = (256, 256)
FORMAT_RANGE: Dict[str, Tuple[int, int]] = {
    "int": (0, 255),
    "norm": (0, 1),
    "dl": (-1, 1),
}


def clip_vals(img: TensorImage, fmt: str) -> TensorImage:
    """Clip the values of an image tensor to their predefined range of valid values."""
    min_, max_ = FORMAT_RANGE[fmt]
    return (
        np.clip(img, min_, max_)
        if isinstance(img, np.ndarray)
        else img.clamp(min_, max_)
    )


def is_int_image(img: TensorImage) -> bool:
    """Return True if the provided tensor represents images encoded as integers."""
    if not isinstance(img, (np.ndarray, torch.Tensor)):
        return False
    is_numpy_uint8 = isinstance(img, np.ndarray) and img.dtype in (np.uint8, np.uint64)
    is_torch_long = isinstance(img, torch.Tensor) and img.dtype in (
        torch.int64,
        torch.uint8,
    )
    return is_numpy_uint8 or is_torch_long


def is_batch(img: TensorImage) -> bool:
    """Return True if the provided tensor represents a batch of images."""
    return isinstance(img, (np.ndarray, torch.Tensor)) and len(img.shape) == 4


def is_channel_fist(img: TensorImage, max_channels: int = 10) -> bool:
    """Return True if the provided tensor's first dimension represents image channels."""
    return (img.shape[1] if is_batch(img) else img.shape[0]) <= max_channels


def to_channels_first(img: TensorImage, max_channels: int = 10) -> TensorImage:
    """Return the provided tensor with channels as first image dimension."""
    if is_channel_fist(img, max_channels=max_channels):
        return img
    reshape = "b h w c -> b c h w" if is_batch(img) else "h w c -> c h w"
    return einops.rearrange(img, reshape)


def to_channels_last(img: TensorImage, max_channels: int = 10) -> TensorImage:
    """Return the provided tensor with as the last image dimension."""
    if not is_channel_fist(img, max_channels=max_channels):
        return img
    reshape = "b c h w -> b h w c" if is_batch(img) else "c h w -> h w c"
    return einops.rearrange(img, reshape)


def to_backend(
    img: TensorImage,
    backend: Optional[str] = None,
    device: Optional[torch.DeviceObjType] = None,
) -> TensorImage:
    """Cast image to the appropriate backend, either "numpy" (alias "np") or pytorch."""
    if backend is None:
        return img
    elif backend in ("numpy", "np"):
        return einops.asnumpy(img)
    elif isinstance(img, np.ndarray):
        x = torch.from_numpy(img)
        return x if device is None else x.to(device)
    return img if device is None else img.to(device)


def to_channel(img: TensorImage, channels_first: Optional[bool] = True) -> TensorImage:
    """Reformat image to be either channels first or channels last."""
    if channels_first is None:
        return img
    return to_channels_first(img) if channels_first else to_channels_last(img)


def _pil_to_int_img(img: PilImage, size: Tuple[int, int] = DEFAULT_SIZE) -> np.ndarray:
    """Transform a :class:`PIL.Image` into an RGB uint8 numpy array with dimension (h, w, c)."""
    if size is not None:
        img = img.resize(size, resample=Image.Resampling.LANCZOS)
    return np.array(img.convert("RGB"))


def _int_to_pil_img(
    img: TensorImage, size: Optional[Tuple[int, int]] = None
) -> PilImage:
    """Transform a an RGB uint8 numpy array with dimension (h, w, c) a into :class:`PIL.Image`."""
    img = to_channels_last(einops.asnumpy(img))
    pil_img = Image.fromarray(img)
    if size is not None:
        pil_img = pil_img.resize(size, resample=Image.Resampling.LANCZOS)
    return pil_img


def resize_pil_img(
    img: Union[List[PilImage], PilImage],
    size: Optional[Tuple[int, int]] = None,
) -> Union[List[PilImage], PilImage]:
    """Resize one or more PIL images into the target size."""

    def _resize_one_img(_im):
        if size is not None:
            _im = _im.resize(size, resample=Image.Resampling.LANCZOS)
        return _im

    return (
        [_resize_one_img(x) for x in img]
        if isinstance(img, list)
        else _resize_one_img(img)
    )


def pil_to_int_img(
    img: Union[List[PilImage], PilImage],
    size: Tuple[int, int] = DEFAULT_SIZE,
) -> np.ndarray:
    """Transform a :class:`PIL.Image` into an RGB uint8 numpy array with dimension (h, w, c)."""
    if isinstance(img, list):
        return np.vstack([_pil_to_int_img(im, size=size)[None] for im in img])
    else:
        return _pil_to_int_img(img, size=size)


def int_to_pil_img(
    img: TensorImage,
    size: Optional[Tuple[int, int]] = None,
) -> Union[PilImage, List[PilImage]]:
    """Transform a an RGB uint8 numpy array with dimension (h, w, c) a into :class:`PIL.Image`."""
    if not is_batch(img):
        return _int_to_pil_img(img, size=size)
    return [_int_to_pil_img(im, size=size) for im in img]


def pil_to_norm_img(
    img: Union[PilImage, List[PilImage]],
    size: Tuple[int, int] = DEFAULT_SIZE,
) -> np.ndarray:
    """Transform a :class:`PIL.Image` into an RGB [0, 1] numpy array with dimension (h, w, c)."""
    img = pil_to_int_img(img, size=size)
    return int_to_norm_img(img)


def pil_to_dl_img(
    img: Union[PilImage, List[PilImage]],
    size: Tuple[int, int] = DEFAULT_SIZE,
) -> np.ndarray:
    """Transform a :class:`PIL.Image` into an RGB [-1, 1] numpy array with dimension (h, w, c)."""
    img = pil_to_int_img(img, size=size)
    return int_to_dl_img(img)


def int_to_norm_img(img: TensorImage, *args, **kwargs) -> TensorImage:
    """Transform an image from range [0, 255] to range [0, 1]."""
    return img / 255.0


def int_to_dl_img(img: TensorImage, *args, **kwargs) -> TensorImage:
    """Transform an image from range [0, 255] to range [-1, 1]."""
    return (img / 255.0 - 0.5) * 2


def norm_to_int_img(img: TensorImage, *args, **kwargs) -> TensorImage:
    """Transform an image from range [0, 1] to range [0, 255]."""
    x = (img * 255).round()
    return x.astype(np.uint8) if isinstance(x, np.ndarray) else x.to(torch.uint8)


def norm_to_pil_img(
    img: TensorImage,
    size: Optional[Tuple[int, int]] = None,
) -> Union[PilImage, List[PilImage]]:
    """Transform an image from range [0, 1] into a :class:`PIL.Image`."""
    x = norm_to_int_img(img)
    return int_to_pil_img(x, size=size)


def norm_to_dl_img(img: TensorImage, *args, **kwargs) -> TensorImage:
    """Transform an image from range [0, 1] to range [-1, 1]."""
    return (img - 0.5) * 2


def dl_to_norm_img(img: TensorImage, *args, **kwargs) -> TensorImage:
    """Transform an image from range [-1, 1] to range [0, 1]."""
    return img / 2 + 0.5


def dl_to_int_img(img: TensorImage, *args, **kwargs) -> TensorImage:
    """Transform an image from range [-1, 1] to range [0, 255]."""
    x = img / 2 + 0.5
    return norm_to_int_img(x)


def dl_to_pil_img(
    img: TensorImage,
    size: Optional[Tuple[int, int]] = None,
) -> Union[PilImage, List[PilImage]]:
    """Transform an image from range [-1, 1] into a :class:`PIL.Image`."""
    x = dl_to_int_img(img)
    return int_to_pil_img(x, size=size)


FORMAT_DICT = {
    ("pil", "int"): pil_to_int_img,
    ("pil", "norm"): pil_to_norm_img,
    ("pil", "dl"): pil_to_dl_img,
    ("int", "pil"): int_to_pil_img,
    ("int", "norm"): int_to_norm_img,
    ("int", "dl"): int_to_dl_img,
    ("norm", "pil"): norm_to_pil_img,
    ("norm", "int"): norm_to_int_img,
    ("norm", "dl"): norm_to_dl_img,
    ("dl", "pil"): dl_to_pil_img,
    ("dl", "int"): dl_to_int_img,
    ("dl", "norm"): dl_to_norm_img,
}


def to_format(
    img: AnyImage,
    in_fmt: str,
    out_fmt: str,
    size: Optional[Tuple[int, int]] = None,
) -> AnyImage:
    """Transform an image between different formats."""
    return FORMAT_DICT[(in_fmt, out_fmt)](img, size=size)


def infer_format(img, hint: Optional[str] = None) -> str:
    """Guess the format of the provided image using hint as guidance."""
    if isinstance(img, Image.Image) or (
        isinstance(img, list) and img and isinstance(img[0], Image.Image)
    ):
        return "pil"
    elif hint is not None:
        return hint
    elif img.dtype in (np.int64, np.uint8, torch.int64, torch.uint8):
        return "int"
    elif bool(img.min() < 0):
        return "dl"
    return "norm"


def reformat_img(
    img: Union[AnyImage, List[PilImage]],
    in_fmt: str = None,
    out_fmt: str = None,
    size: Optional[Tuple[int, int]] = None,
    backend: Optional[str] = None,
    channels_first: bool = None,
    device: Optional[torch.DeviceObjType] = None,
    clip: bool = True,
) -> Union[AnyImage, List[PilImage]]:
    """Reformat an image between different formats, devices and channel orders."""
    # TODO: Document this properly.
    in_fmt = infer_format(img, in_fmt)
    out_fmt = out_fmt if out_fmt is not None else in_fmt
    x = clip_vals(img, in_fmt) if clip and in_fmt != "pil" else img
    if in_fmt != out_fmt:
        x = to_format(x, in_fmt=in_fmt, out_fmt=out_fmt, size=size)
    elif in_fmt == "pil" and size is not None:
        x = resize_pil_img(x, size=size)
    if out_fmt == "pil":
        return x
    x = to_backend(x, backend=backend, device=device)
    x = to_channel(x, channels_first=channels_first)
    return x
