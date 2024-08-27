"""This module contains functions for generating holoviews plots for visualizing embeddings."""
from pathlib import Path
from typing import List, Union

import holoviews as hv
import numpy as np
import pandas as pd
from bokeh.models import HoverTool

EMBEDDING_TOOLTIP = """
    <div>
        <div>
            <img
                src="@img_tooltip" height="128" alt="@image" width="128"
                style="float: left; margin: 0px 0px 0px 0px;"
                border="2"
            ></img>
        </div>
        <div>
            <span style="font-size: 17px; font-weight: bold;">@target_names</span>
            <span style="font-size: 15px; color: #966;">[$index]</span>
        </div>
        <div>
            <span>@split_names{safe}</span>
        </div>
        <div>
            <span style="font-size: 15px;">Location</span>
            <span style="font-size: 10px; color: #696;">($x, $y)</span>
        </div>
    </div>
"""

hover_tool = HoverTool(tooltips=EMBEDDING_TOOLTIP)


def imshow(x: np.ndarray) -> hv.RGB:
    return hv.RGB(x).opts(xaxis=None, yaxis=None)


def export_as_gif(
    filename: Union[Path, str],
    images: List[np.ndarray],
    frames_per_second: int = 10,
    rubber_band: bool = False,
) -> None:
    if rubber_band:
        images += images[2:-1][::-1]
    images[0].save(
        filename,
        save_all=True,
        append_images=images[1:],
        duration=1000 // frames_per_second,
        loop=0,
    )


def split_to_alphas(split_names: List[str], splits: List[str]) -> List[float]:
    splits = set(split_names) if splits is None else set(splits)
    return [0.7 if x in splits else 0.0 for x in split_names]


def split_to_marker(splits: List[Union[int, str]]) -> List[str]:
    mapping = {
        "train": "circle",
        "test": "square",
        "validation": "triangle",
        0: "circle",
        1: "square",
    }
    return [mapping.get(x, "cross") for x in splits]


def plot_eigenvalues(eigenvalues: np.ndarray) -> hv.Overlay:
    plot = hv.Curve((np.arange(len(eigenvalues)), eigenvalues))
    plot = plot.opts(
        width=800,
        ylabel="Eigenvalue",
        xlabel="component",
        title="Embeddings eigenvalues",
        tools=["hover"],
    )
    plot = plot * hv.HLine(0.0).opts(line_color="black", line_width=0.5)
    return plot


def plot_embeddings_3d(
    db_umap_emb_3d: np.ndarray,
    target: np.ndarray,
    splits: np.ndarray,
) -> hv.Scatter3D:
    umap_emb_data = {
        "x": db_umap_emb_3d[:, 0],
        "y": db_umap_emb_3d[:, 1],
        "z": db_umap_emb_3d[:, 2],
        "label": target,
        "marker": split_to_marker(splits),
    }

    plot = hv.Scatter3D(umap_emb_data, kdims=["x", "y", "z"], vdims=["label", "marker"])
    plot = plot.opts(
        color="label",
        width=800,
        height=800,
        size=4,
        alpha=0.9,
        cmap="spectral",
        ylabel="Dim 2",
        xlabel="Dim 1",
        zlabel="Dim 3",
        title="Umap embedding of the diffusion graph representing the latent space",
        marker="marker",
    )
    return plot


def plot_embeddings(
    data: pd.DataFrame,
    marker: str = "split_names",
    color: str = "label",
    layout: str = "dbpac",
) -> hv.Points:
    data["marker"] = split_to_marker(data[marker])
    vdims = [c for c in data.columns if "umap" not in c and "dbpac" not in c]
    kdims = ["dbpac_x", "dbpac_y"] if layout == "dbpac" else ["umap_x", "umap_y"]
    plot = hv.Points(data, kdims=kdims, vdims=vdims)
    plot = plot.opts(
        color=color,
        marker="marker",
        width=800,
        height=800,
        size=5,
        cmap="spectral",
        alpha=0.7,
        ylabel="Dim 2",
        xlabel="Dim 1",
        tools=[hover_tool, "tap"],
        title="Embedding of the diffusion graph representing the latent space",
        colorbar=True,
        show_legend=True,
    )
    return plot


def plot_path_ends(
    interp_emb: np.ndarray,
    size: int = 10,
    color_start: str = "blue",
    color_end: str = "blue",
) -> hv.Overlay:
    plot_start = hv.Scatter(interp_emb[0][None]).opts(size=size, color=color_start)
    plot_end = hv.Scatter(interp_emb[-1][None]).opts(size=size, color=color_end)
    return plot_start * plot_end


def plot_path(
    interp_emb: np.ndarray,
    size: int = 5,
    color: str = "step",
    cmap: str = "RdYlGn",
    colorbar: bool = True,
    show_legend: bool = True,
    color_start: str = "blue",
    color_end: str = "blue",
    alpha_curve: float = 0.3,
    size_end: int = 10,
    **kwargs,
) -> hv.Overlay:
    tooltips = [("Step", "$index"), ("x", "@x"), ("y", "@y")]
    path_df = pd.DataFrame(data=interp_emb, columns=["x", "y"])
    path_df["step"] = np.arange(len(path_df))
    hover_path = HoverTool(tooltips=tooltips)
    plot = (
        hv.Points(path_df, kdims=["x", "y"], vdims=["step"]).opts(
            size=size,
            color=color,
            tools=[hover_path],
            cmap=cmap,
            colorbar=colorbar,
            show_legend=show_legend,
            **kwargs,
        )
        * hv.Curve(interp_emb).opts(color="black", alpha=alpha_curve)
        * plot_path_ends(
            interp_emb, color_end=color_end, color_start=color_start, size=size_end
        )
    )
    return plot


def plot_embeddings_for_path(df: pd.DataFrame, dataset_emb: np.ndarray) -> hv.Points:
    df2 = df.copy()
    df2["umap_x"] = dataset_emb[:, 0]
    df2["umap_y"] = dataset_emb[:, 1]
    plot = (
        plot_embeddings(df2, layout="umap")
        .opts(tools=[hover_tool], alpha=0.1)
        .opts(cmap="BrBG", colorbar=False)
    )
    return plot
