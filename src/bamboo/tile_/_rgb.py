# Copyright (c) 2025, Tom Ouellette
# Licensed under the GNU GPLv3 License

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from dataclasses import dataclass
from pathlib import Path
from PIL import Image

from bamboo.error import ShapeError
from bamboo.tile._base import Tile

RGBTILE_PIL_FORMATS: set[str] = {
    "avif",
    "blp",
    "bmp",
    "dds",
    "dib",
    "eps",
    "gif",
    "icns",
    "ico",
    "im",
    "jpeg",
    "mpo",
    "msp",
    "pcx",
    "pfm",
    "png",
    "ppm",
    "qoi",
    "sgi",
    "spider",
    "tga",
    "tiff",
    "webp",
    "xbm",
}

RGBTILE_WRITE_FORMATS: set[str] = RGBTILE_PIL_FORMATS | {"npy"}


@dataclass(frozen=True)
class RGBTile(Tile):
    """In-memory two-dimensional RGB tile defining an image subregion.

    Attributes
    ----------
    buffer : np.ndarray
        Two-dimensional 8-bit RGB pixel container (channel in last index).
    dtype : np.dtype
        Data type of buffer elements (subpixels).
    shape : Tuple[int, int, int]
        Height, width, and channels for the tile.
    h : int
        Height of the tile.
    w : int
        Width of the tile.
    c : int
        Number of channels in the tile.
    """

    def __post_init__(self):
        """Buffer validation and attribute initialization."""
        if not isinstance(self.buffer, np.ndarray):
            raise TypeError("[RGBTile] 'buffer' must be of type np.ndarray.")

        dtype = self.buffer.dtype
        if not np.isdtype(dtype, np.uint8):
            raise TypeError("[RGBTile] 'buffer' must be uint8 type.")

        object.__setattr__(self, "dtype", dtype)

        if self.buffer.ndim != 3:
            raise ShapeError("[RGBTile] 'buffer' must be have 3 dimensions.")

        if self.buffer.shape[-1] != 3:
            raise ShapeError("[RGBTile] 'buffer' must have 3 channels.")

        shape = self.buffer.shape
        object.__setattr__(self, "shape", shape)
        object.__setattr__(self, "h", shape[0])
        object.__setattr__(self, "w", shape[1])
        object.__setattr__(self, "c", 3)

    def show(
        self,
        ax: None | plt.Axes = None,
        figsize: None | tuple[int, int] = None,
        hide_axes: bool = True,
    ) -> plt.Axes:
        """Visualize the tile.

        Parameters
        ----------
        ax : None | plt.Axes
            Optional matplot axes object.
        figsize : None | tuple[int, int]
            Option figure size.
        hide_axes : bool
            If True, then all x-axis, y-axis, and spines are removed.

        Returns
        -------
        plt.Axes
            Matplotlib axes object of tile.
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)

        ax.imshow(self.buffer)

        if hide_axes:
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            for pos in ["left", "right", "top", "bottom"]:
                ax.spines[pos].set_visible(False)

        return ax

    def resize(
        self,
        w: None | int = None,
        h: None | int = None,
        resample: str | None = "bilinear",
    ) -> RGBTile:
        """Resize the tile.

        Parameters
        ----------
        w : None | int
            Resize width.
        h : None | int
            Resize height.
        resample : str | None
            Resampling filter ('bilinear', 'bicubic', 'lanczos', 'nearest')
        """
        match resample:
            case "bilinear":
                resample = Image.Resampling.BILINEAR
            case "bicubic":
                resample = Image.Resampling.BICUBIC
            case "lanczos":
                resample = Image.Resampling.LANCZOS
            case "nearest":
                resample = Image.Resampling.NEAREST
            case _:
                raise ValueError(
                    "'resample' must be one of: "
                    + "'bilinear', 'bicubic', 'lanczos', 'nearest'."
                )

        if h is None and w is None:
            raise ValueError("One of 'height' or 'width' must be specified.")

        if h is None:
            h = int(self.h * w / self.w)

        if w is None:
            w = int(self.w * h / self.h)

        # Note: PIL Image resize is not particularly fast - so it may be useful
        # to write an interface to the fast_image_resize rust crate
        return RGBTile(np.array(Image.fromarray(self.buffer).resize((w, h), resample)))

    def write(
        self,
        path: str | Path,
    ) -> None:
        """Write the RGBTile to disk.

        Parameters
        ----------
        path : str | Path
            A path specifying file with valid image extension.

        Returns
        -------
        None
            RGBTile is written to disk.
        """
        if isinstance(path, str):
            path = Path(path)

        if path.suffix[1:] in RGBTILE_PIL_FORMATS:
            Image.fromarray(self.buffer).save(path)
        elif path.suffix == ".npy":
            np.save(path, self.buffer)
        else:
            raise ValueError(
                "'RGBTile' write failed. 'path' must specify a supported image format: "
                f"{RGBTILE_WRITE_FORMATS}"
            )
