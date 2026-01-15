# Copyright (c) 2025, Tom Ouellette
# Licensed under the GNU GPLv3 License

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import torch

from bamboo.error import ShapeError
from dataclasses import dataclass
from numpy.typing import NDArray
from pathlib import Path
from PIL import Image
from typing import Tuple

from bamboo.types import IndexKey, Numeric

PIL_FORMATS: set[str] = {
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


@dataclass(frozen=True)
class Tile:
    """A base class for a two-dimensional in-memory tile of image subregion.

    Attributes
    ----------
    buffer : np.ndarray
        Two-dimensional pixel container with optional last index channel.
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

    buffer: np.ndarray
    dtype: Numeric = None
    shape: Tuple[int, int, int] = None
    h: int = None
    w: int = None
    c: int = None

    def __getitem__(self, key: IndexKey) -> Numeric | NDArray[Numeric]:
        """Direct indexing of the underlying buffer.

        Parameters
        ----------
        key : IndexKey
            A slice, ellipsis, or any valid index into numpy ndarray.

        Returns
        -------
        Numeric | NDArray[Numeric]
            Slice
        """
        return self.buffer[key]

    def to_numpy(self) -> NDArray[Numeric]:
        """Return tile as a numpy array.

        Returns
        -------
        NDArray[Numeric]
            Pixels in numpy array format.
        """
        return self.buffer[:]

    def to_tensor(self) -> torch.Tensor:
        """Return tile as a torch tensor.

        Returns
        -------
        torch.Tensor
            Pixels in torch tensor format.
        """
        return torch.from_numpy(self.buffer[:])

    def min(self) -> Numeric:
        """Get minimum subpixel value in tile.

        Returns
        -------
        Numeric
            Minimum subpixel value.
        """
        return self.buffer.min()

    def max(self) -> Numeric:
        """Get maximum subpixel value in tile.

        Returns
        -------
        Numeric
            Maximum subpixel value.
        """
        return self.buffer.max()

    def row_min(self) -> NDArray[Numeric]:
        """Get minimum subpixel values in each row of tile.

        Returns
        -------
        NDArray[Numeric]
            Minimum subpixel values per row.
        """
        return self.buffer.min(axis=(1, 2))

    def row_max(self) -> NDArray[Numeric]:
        """Get maximum subpixel values in each row of tile.

        Returns
        -------
        NDArray[Numeric]
            Maximum subpixel values per row.
        """
        return self.buffer.max(axis=(1, 2))

    def col_min(self) -> NDArray[Numeric]:
        """Get minimum subpixel values in each column of tile.

        Returns
        -------
        NDArray[Numeric]
            Minimum subpixel values per column.
        """
        return self.buffer.min(axis=(0, 2))

    def col_max(self) -> NDArray[Numeric]:
        """Get maximum subpixel values in each column of tile.

        Returns
        -------
        NDArray[Numeric]
            Maximum subpixel values per column.
        """
        return self.buffer.max(axis=(0, 2))

    def channel_min(self) -> NDArray[Numeric]:
        """Get minimum subpixel values in each channel of tile.

        Returns
        -------
        NDArray[Numeric]
            Minimum subpixel values per channel.
        """
        return self.buffer.min(axis=(0, 1))

    def channel_max(self) -> NDArray[Numeric]:
        """Get maximum subpixel values in each channel of tile.

        Returns
        -------
        NDArray[Numeric]
            Maximum subpixel values per channel.
        """
        return self.buffer.max(axis=(0, 1))


RGBTILE_WRITE_FORMATS: set[str] = PIL_FORMATS | {"npy"}


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

        if path.suffix[1:] in PIL_FORMATS:
            Image.fromarray(self.buffer).save(path)
        elif path.suffix == ".npy":
            np.save(path, self.buffer)
        else:
            raise ValueError(
                "'RGBTile' write failed. 'path' must specify a supported image format: "
                f"{RGBTILE_WRITE_FORMATS}"
            )
