# Copyright (c) 2025, Tom Ouellette
# Licensed under the GNU GPLv3 License

import numpy as np
import torch

from dataclasses import dataclass
from numpy.typing import NDArray
from typing import Tuple

from bamboo.types import IndexKey, Numeric


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
