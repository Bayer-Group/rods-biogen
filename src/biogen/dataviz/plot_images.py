import logging
import os
import re
from pathlib import Path
from typing import List, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import torchvision
from PIL import Image
from tqdm import tqdm

from biogen.dataset_utils import get_metadata_df_from_dataset_path

logger = logging.getLogger(__name__)


def plot_torch_grid(
    images: List,
    nperrow: int,
    savedir: Optional[Union[str, Path]] = None,
    title_text="",
    size=96,
) -> None:
    """Plot each image in images on a grid of width nperrow and of infered height."""
    transform_to_tensor = torchvision.transforms.Compose(
        [
            torchvision.transforms.Resize((size, size)),
            torchvision.transforms.ToTensor(),
        ],
    )

    grid_img = torchvision.utils.make_grid(
        [transform_to_tensor(x) for x in images],
        nrow=nperrow,
        padding=10,
    )
    plt.figure(figsize=(15, 15))
    plt.imshow(grid_img.permute(1, 2, 0))
    plt.title(title_text)
    plt.axis("off")
    if savedir is not None:
        plt.savefig(savedir)
    plt.show()
    plt.close()


def plot_images_from_df(
    df_subset_100, n=100, nperrow=10, title_text="", savedir=None, seed=None, **kwargs
):
    """Plot images from df."""
    # Shuffle
    df_subset_100 = df_subset_100.sample(frac=1, random_state=seed)
    image_list = []
    for i, im_path in enumerate(df_subset_100["path"].tolist()):
        image_list.append(Image.open(im_path))
        if i > n - 2:
            break

    plot_torch_grid(
        images=image_list,
        nperrow=nperrow,
        title_text=title_text,
        savedir=savedir,
        **kwargs,
    )


def plot_corresponding_im_grids(
    im_ds_real_path,
    im_ds_synth_path,
    save_path,
    text_2_promptidx,
    prompt_col="prompt_idx",
    label_col="label_name",
):
    """Plot image grids."""
    # Load directly from metadata.jsonl files
    df_real = get_metadata_df_from_dataset_path(im_ds_real_path)
    df_synth = get_metadata_df_from_dataset_path(im_ds_synth_path)
    label_names = np.unique(df_real[label_col])
    logger.info(f"label_names = {label_names}")
    inv_text_2_promptidx = {v: k for k, v in text_2_promptidx.items()}
    logger.info("Looping over training prompts ... ")
    for prompt_idx in tqdm(np.unique(df_synth[prompt_col])):
        df_r = df_real[df_real[prompt_col] == prompt_idx]
        df_s = df_synth[df_synth[prompt_col] == prompt_idx]
        title_text = inv_text_2_promptidx[prompt_idx]
        label = (
            label_names[0]
            if f" {label_names[0]}".lower() in title_text.lower()
            else label_names[1]
        )
        p_lbl = save_path / label / str(prompt_idx)
        os.makedirs(p_lbl, exist_ok=True)
        plot_images_from_df(
            df_r,
            n=100,
            nperrow=10,
            title_text=title_text,
            savedir=p_lbl / f"prmpt_{prompt_idx}_real.png",
        )
        plot_images_from_df(
            df_s,
            n=100,
            nperrow=10,
            title_text=title_text,
            savedir=p_lbl / f"prmpt_{prompt_idx}_synth.png",
        )
        # Join the two generated grids for comparison
        im_path_list = [
            p_lbl / f"prmpt_{prompt_idx}_real.png",
            p_lbl / f"prmpt_{prompt_idx}_synth.png",
        ]
        combine_images(
            columns=2,
            space=10,
            images=im_path_list,
            path=p_lbl,
            factor=0.85,
            grid_name="grids_real_vs_synth.png",
        )


def combine_images(columns, space, images, path, grid_name, factor=0.5):
    """Combine image grids."""
    rows = len(images) // columns
    if len(images) % columns:
        rows += 1

    im_list = [Image.open(image) for image in images]
    crop_size = 100
    im_list = [
        image.crop(
            (crop_size, crop_size, image.width - crop_size, image.height - crop_size)
        )
        for image in im_list
    ]
    im_list = [
        image.resize((int(image.width * factor), int(image.height * factor)))
        for image in im_list
    ]

    width_max = max([image.width for image in im_list])
    height_max = max([image.height for image in im_list])

    background_width = width_max * columns + (space * columns) - space
    background_height = height_max * rows + (space * rows) - space
    background = Image.new(
        "RGBA", (background_width, background_height), (255, 255, 255, 255)
    )
    x = 0
    y = 0
    idx_order = []
    for i, img in enumerate(im_list):
        grid_prompt_idx = int(re.findall(r"\d+", os.path.basename(images[i]))[0])
        idx_order.append(grid_prompt_idx)

        x_offset = int((width_max - img.width) / 2)
        y_offset = int((height_max - img.height) / 2)
        background.paste(img, (x + x_offset, y + y_offset))
        x += width_max + space
        if (i + 1) % columns == 0:
            y += height_max + space
            x = 0
    background.save(Path(path) / grid_name)
