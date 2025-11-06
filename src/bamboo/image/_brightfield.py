# Copyright (c) 2025, Tom Ouellette
# Licensed under the GNU GPLv3 License

from pathlib import Path

from numpy.typing import NDArray
from bamboo.types import IndexKey, Numeric
from bamboo.tile import RGBTile

from bamboo.backend._brightfield import (
    BrightfieldTiffBackend,
    BrightfieldOpenslideBackend,
    BrightfieldMemoryBackend,
    BRIGHTFIELD_TIFF_CONFIG,
    BRIGHTFIELD_OPENSLIDE_CONFIG,
    BRIGHTFIELD_MEMORY_CONFIG,
)


class Brightfield:
    """A class for interacting with brightfield images.

    This class provides a simple interface for interacting with single-level
    in-memory or multi-level memory-mapped/on-disk brightfield RGB images.

    Parameters
    ----------
    path : str or Path
        Path to the input image file.
    mpp : None | float
        User-provided microns per pixel.
    level : None | int | str
        User-provided level to read.
    backend : None | str
        Manually set the backend ('memory', 'tifffile', 'openslide'). The
        backend will automatically selected if not provided.

    Attributes
    ----------
    backend : BrightfieldBackend
        A backend providing access to pixel-level data.
    """

    def __init__(
        self,
        path: str | Path,
        mpp: None | float = None,
        level: None | str | int = None,
        backend: None | str = None,
    ):
        if not isinstance(path, (str, Path)):
            raise ValueError("'path' must be a str or pathlib.Path.")

        if not (path := Path(path)).exists():
            raise FileNotFoundError("'path' does not exist.")

        if path.is_dir():
            raise OSError("'path' must be a file not a directory.")

        if backend is None:
            if BRIGHTFIELD_TIFF_CONFIG.supports_format(path):
                backend = "tifffile"
            elif BRIGHTFIELD_OPENSLIDE_CONFIG.supports_format(path):
                backend = "openslide"
            elif BRIGHTFIELD_MEMORY_CONFIG.supports_format(path):
                backend = "memory"
            else:
                raise ValueError(
                    f"The provided file format ({path.suffix}) is not "
                    + "currently supported."
                )

        match backend:
            case "tifffile":
                self.backend = BrightfieldTiffBackend(path, mpp, level)
            case "openslide":
                self.backend = BrightfieldOpenslideBackend(path, mpp, level)
            case "memory":
                if level:
                    raise ValueError("'level' is not supported for 'memory' backend.")

                self.backend = BrightfieldMemoryBackend(path, mpp)
            case _:
                raise ValueError(
                    "If 'backend' is provided, must be one of: "
                    + "'tifffile', 'openslide', 'memory'."
                )

    def __repr__(self):
        if isinstance((mpp := self.mpp), float):
            mpp = round(self.mpp, 4)

        return f"Brightfield(l={self.l}, h={self.h}, w={self.w}, c={self.c}, mpp={mpp})"

    def __getstate__(self):
        """Handle state for multiprocess pickling."""
        return {"path": self.backend.path}

    def __setstate__(self, state):
        """Recover state for multiprocess pickling."""
        self.__init__(state["path"])

    @property
    def levels(self):
        """Store number of avaiable levels."""
        return self.backend.levels

    @property
    def levels_keys(self):
        """Store keys for each available level."""
        return self.backend.levels_keys

    @property
    def levels_shape(self):
        """Store shape of each available level."""
        return self.backend.levels_shape

    @property
    def l(self):
        """Set current slide level."""
        return self.backend.l

    @property
    def h(self):
        """Set image height."""
        return self.backend.h

    @property
    def w(self):
        """Set image height."""
        return self.backend.w

    @property
    def c(self):
        """Set image channels."""
        return self.backend.c

    @property
    def shape(self):
        """Set image shape"""
        return (self.h, self.w, self.c)

    @property
    def mpp_x(self):
        """Set microns per pixel over x dimension."""
        return self.backend.mpp_x

    @property
    def mpp_y(self):
        """Set microns per pixel over x dimension."""
        return self.backend.mpp_x

    @property
    def mpp(self):
        """Set average microns per pixel."""
        return self.backend.mpp

    def __getitem__(self, key: IndexKey) -> Numeric | NDArray[Numeric]:
        """Direct indexing of the underlying pixel data.

        Parameters
        ----------
        key : IndexKey
            A slice, ellipsis, or any valid index into numpy ndarray.

        Returns
        -------
        Numeric | NDArray[Numeric]
            Subregion of brightfield image.
        """
        return self.backend[key]

    def tile(self, x: int, y: int, w: int, h: int) -> RGBTile:
        """Get an RGB tile specifying an image subregion.

        Parameters
        ----------
        x : int
            Initial x-coordinate of tile.
        y : int
            Initial y-coordinate of tile.
        w : int
            Width of tile
        h : int
            Height of tile.

        Returns
        -------
        RGBTile
            An RGB tile of image subregion.
        """
        if x > self.w or x < 0:
            raise ValueError("x must be greater than 0 and smaller than image width.")

        if y > self.h or y < 0:
            raise ValueError("y must be greater than 0 and smaller than image height.")

        if h == 1 or w < 1:
            raise ValueError("h and w must be greater than 0.")

        return RGBTile(self.backend[y : y + h, x : x + w])

    def thumbnail(
        self,
        t: int = 5000,
        force: bool = False,
        height: int = None,
        width: int = None,
        resample: str | None = "bilinear",
    ) -> RGBTile:
        """Get a lower resolution thumbnail.

        Parameters
        ----------
        t: int
            If pyramidal, find level with height + width closest to this value.
        force: bool
            If True and image is not pyramidal, then the thumbnail will be
            generated at the height and width of the full resolution image.
        height: None | int
            Optionally resize the thumbnail to a user-provided height.
        width: None | int
            Optionally resize the thumbnail to a user-provided width.
        resample: str | None
            Resampling filter ('bilinear', 'bicubic', 'lanczos', 'nearest').
        """
        if force and not self.levels_shape:
            tile = RGBTile(self[:])
        else:
            if not self.levels_shape and (self.w > t or self.h > t):
                return ValueError(
                    "Image only has a single level and current width and "
                    + "height are greater than 't'. This suggests the lowest "
                    + "level may be large. If you still want to generate the "
                    + "thumbnail, re-run `.thumbnail()` with `force=True`."
                )

            closest_key = None
            min_difference = float("inf")

            for key, (h, w, _) in self.levels_shape.items():
                difference = abs(h - t) + abs(w - t)
                if difference < min_difference:
                    min_difference = difference
                    closest_key = key

            tile = RGBTile(Brightfield(self.backend.path, level=closest_key)[:])

        if height or width:
            tile = tile.resize(height, width, resample)

        return tile
