# Copyright (c) 2025, Tom Ouellette
# Licensed under the GNU GPLv3 License

import numpy as np

from dataclasses import dataclass

from bamboo.error import ShapeError
from bamboo.tile._base import Tile


@dataclass(frozen=True)
class GenericTile(Tile):
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

    def __post_init__(self):
        """Buffer validation and attribute initialization."""
        if not isinstance(self.buffer, np.ndarray):
            raise TypeError("[GenericTile] 'buffer' must be of type np.ndarray.")

        dtype = self.buffer.dtype
        if not np.any(
            [np.issubdtype(dtype, np.integer), np.issubdtype(dtype, np.floating)]
        ):
            raise TypeError("[GenericTile] 'buffer' must be integer or float type.")

        object.__setattr__(self, "dtype", dtype)

        if len(self.buffer.shape) == 2:
            object.__setattr__(self, "buffer", self.buffer[:, :, None])

        if self.buffer.ndim != 3:
            raise ShapeError("[GenericTile] 'buffer' must be 2 or 3 dimensions.")

        shape = self.buffer.shape
        object.__setattr__(self, "shape", shape)
        object.__setattr__(self, "h", shape[0])
        object.__setattr__(self, "w", shape[1])
        object.__setattr__(self, "c", shape[2])
