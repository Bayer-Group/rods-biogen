from typing import List

import holoviews as hv
import numpy as np
import panel as pn
import param
from panel.interact import fixed, interact

from biogen.dataviz.plot_embd import imshow, plot_embeddings


def _play_transitions(images: List[np.ndarray], i: int) -> hv.RGB:
    return imshow(images[i])


def play_transitions(images: List[np.ndarray], interval=100, **kwargs) -> pn.Pane:
    player = pn.widgets.Player(
        name="Player",
        start=0,
        end=len(images),
        value=0,
        loop_policy="loop",
        interval=interval,
        **kwargs,
    )
    return interact(_play_transitions, images=fixed(images), i=player)


def evaluate_interpolation(images: List[np.ndarray], **kwargs) -> pn.Pane:
    player, play_plot = play_transitions(images, **kwargs)
    refs = (
        imshow(images[0]).opts(title="Start image")
        + imshow(images[-1]).opts(title="End image")
    ).cols(1)
    return pn.Row(pn.pane.HoloViews(refs), pn.Column(play_plot, player))


def _play_transitions_walk(images: List[np.ndarray], i: int) -> hv.Layout:
    return (
        imshow(images[0]).opts(title="Start image")
        + imshow(images[i]).opts(title=f"Image step {i}")
        + imshow(images[i] - images[0]).opts(title="Noise added")
    )


def play_transitions_walk(images: List[np.ndarray], **kwargs) -> pn.Pane:
    player = pn.widgets.Player(
        name="Player",
        start=0,
        end=len(images),
        value=0,
        loop_policy="loop",
        **kwargs,
    )
    return interact(_play_transitions_walk, images=fixed(images), i=player)


def evaluate_walk(images: List[np.ndarray], **kwargs) -> pn.Pane:
    player, play_plot = play_transitions_walk(images, **kwargs)
    return pn.Column(play_plot, player)


class EmbeddingDashboard(param.Parameterized):
    splits = pn.widgets.MultiSelect(
        options=["train", "test", "validation"],
        value=["train", "test", "validation"],
        width=200,
        name="Dataset split",
    )
    color = param.Selector(
        [
            "targets",
            "split_name",
            # "discrim_loss",
            # "lpips_loss",
            # "correct",
            # "predicted",
            # "mae",
            # "mse",
            # "logits_0",
            # "logits_1",
            # "prob_0",
            # "prob_1",
        ],
    )
    marker = param.Selector(["split_name", "targets"])
    layout = param.Selector(["pca", "umap"])
    data = param.DataFrame(precedence=-1)
    filtered_data = param.DataFrame(precedence=-1)

    @param.depends("splits.value")
    def filter_dataset(self) -> None:
        self.filtered_data = self.data[self.data["split_name"].isin(self.splits.value)]

    @param.depends("marker", "color", "layout", "filtered_data")
    def plot_embeddings(self) -> pn.pane.HoloViews:
        plot = plot_embeddings(
            self.filtered_data,
            marker=self.marker,
            color=self.color,
            layout=self.layout,
        )
        return pn.pane.HoloViews(plot)

    def panel(self) -> pn.Pane:
        return pn.panel(
            pn.Column(
                pn.Row(self.param, self.splits),
                self.filter_dataset,
                self.plot_embeddings,
            ),
        )
