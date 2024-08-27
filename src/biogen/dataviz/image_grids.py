import holoviews as hv
import numpy as np
import panel as pn
from PIL import Image

hv.extension("bokeh")
pn.extension()


def plot_image(x, **kwargs):
    return hv.RGB(x).opts(xaxis=None, yaxis=None, **kwargs)


def plot_grid(images, n_cols=5, **kwargs):
    return hv.Layout([plot_image(im, **kwargs) for im in images]).cols(n_cols)


def load_images(df, prompt_idx, n_images, cond_col="prompt_idx"):
    subset = df.loc[df[cond_col] == prompt_idx, "path"]
    return [
        np.array(Image.open(path).convert("RGB")) for path in subset.values[:n_images]
    ]


def load_img_grid(df, prompt_idx, n_images, cond_col="prompt_idx", n_cols=5, **kwargs):
    imgs = load_images(df, prompt_idx, n_images, cond_col=cond_col)
    return plot_grid(imgs, n_cols=n_cols, **kwargs)


def compare_grids(
    df_real,
    df_synth,
    prompt_idx,
    n_images,
    cond_col="prompt_idx",
    title_1="Real images",
    title_2="Synthetic images",
    n_cols=5,
    **kwargs,
):
    grid_real = load_img_grid(
        df_real, prompt_idx, n_images, cond_col=cond_col, n_cols=n_cols, **kwargs
    )
    grid_real = grid_real.opts(title=title_1)
    grid_synth = load_img_grid(
        df_synth, prompt_idx, n_images, cond_col=cond_col, n_cols=n_cols, **kwargs
    )
    grid_synth = grid_synth.opts(title=title_2)
    return pn.Row(grid_real, grid_synth)
