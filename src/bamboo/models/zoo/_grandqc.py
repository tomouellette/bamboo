# Copyright (c) 2025, Tom Ouellette
# Licensed under the Apache 2.0 License

import numpy as np
import torch

from enum import Enum
from numpy.typing import NDArray

from bamboo.models import CACHE_DIR
from bamboo.tile import RGBTile


class GrandQCClasses(Enum):
    """Class labels for GrandQC semantic segmentation."""

    TISSUE = 1
    FOLDS = 2
    SPOTS = 3
    MARKINGS = 4
    ARTIFACTS = 5
    OUT_OF_FOCUS = 6
    BACKGROUND = 7


_MAX_CLASS_VALUE = max(c.value for c in GrandQCClasses)


class GrandQC:
    """A semantic segmentation model for detecting histology image artifacts.

    Parameters
    ----------
    device : str | torch.device
        A valid torch.device (e.g. "cpu", torch.device("cuda:0"), etc).

    References
    ----------
    .. [1] Zhilong Weng et al. "GrandQC: A comprehensive solution to quality
       control problem in digital pathology". Nature Communications. 2024.

    License
    -------
    Creative Commons Attribution Non Commercial Share Alike 4.0 International
    """

    # Weights defines the path to the torchscript model
    weights: str = f"{CACHE_DIR}/model.grandqc.pt"

    # Normalization/scaling coefficients
    mean: list[float, ...] = [0.485, 0.456, 0.406]
    std: list[float, ...] = [0.229, 0.224, 0.225]

    # Tile specifications
    tile_width: int = 512
    tile_height: int = 512
    mpp: float = 1.5
    dtype: np.dtype = np.uint8

    # Note: Storing our loaded model as a class-level attribute allows
    # us to load the model only once per process instead of each time.
    _model: torch.nn.Module = None

    def __init__(self, device: str | torch.device = "cpu"):
        self.device = device

    def _preprocess(self, tile: RGBTile) -> torch.Tensor:
        x = (tile[:] / 255.0) - np.array(self.mean)
        x /= np.array(self.std)
        x = torch.from_numpy(x).permute(2, 0, 1)
        return x.unsqueeze(0).float().to(self.device)

    def _run(self, tile: RGBTile) -> NDArray[np.uint8]:
        if GrandQC._model is None:
            GrandQC._model = torch.jit.load(self.weights).to(self.device)
            GrandQC._model.eval()

        return (
            GrandQC._model(self._preprocess(tile))
            .detach()
            .cpu()
            .numpy()[0]
            .argmax(axis=0)
        )

    __call__ = _run


def GrandQCStats(tile: NDArray[np.uint8]) -> dict[str, float]:
    """Compute GrandQC class stats from input segmentation mask.

    Parameters
    ----------
    tile : np.ndarray
        2D array (H, W) with integer labels corresponding to GrandQCClasses.

    Returns
    -------
    dict[str, float]
        Mapping from class name (string) to frequency.
    """
    if tile.ndim != 2:
        raise ValueError(f"Expected 2D segmentation mask, got shape {tile.shape}")

    counts = np.bincount(tile.ravel(), minlength=_MAX_CLASS_VALUE + 1)
    stats = {cls.name: counts[cls.value] / tile.size for cls in GrandQCClasses}
    stats["ARTIFACT"] = 1 - stats["TISSUE"] - stats["BACKGROUND"]
    return stats
