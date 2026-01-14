# Copyright (c) 2025, Tom Ouellette
# Licensed under the GNU GPLv3 License

from pathlib import Path
from typing import Self

from numpy.typing import NDArray
from bamboo.types import IndexKey, Numeric
from bamboo.tile import RGBTile

from bamboo.backend._brightfield import (
    BrightfieldTiffBackend,
    BrightfieldOpenslideBackend,
    BrightfieldMemoryBackend,
    BRIGHTFIELD_BACKENDS,
)


_BACKEND_PRIORITY = ["tifffile", "openslide", "memory"]


class Brightfield:
    """A unified interface for brightfield images across multiple backends.

    Parameters
    ----------
    path : str | Path
        Path to the input image file.
    mpp : float | None
        Optional microns per pixel.
    level : int | str | None
        Optional pyramid level to read.
    backend : str | None
        Optional backend override ('memory', 'tifffile', 'openslide').
        Automatically inferred if not provided.
    """

    def __init__(
        self,
        path: str | Path,
        mpp: float | None = None,
        level: int | str | None = None,
        backend: str | None = None,
    ) -> None:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Path does not exist: {path}")
        if path.is_dir():
            raise IsADirectoryError(f"Path must be a file, not a directory: {path}")

        # Determine backend automatically if not provided
        if backend is None:
            backend = None
            for b_name in _BACKEND_PRIORITY:
                cfg = BRIGHTFIELD_BACKENDS[b_name]
                if cfg.supports_format(path):
                    backend = b_name
                    break
            if backend is None:
                all_formats = ", ".join(
                    sorted(
                        ext
                        for cfg in BRIGHTFIELD_BACKENDS.values()
                        for ext in cfg.formats
                    )
                )
                raise ValueError(
                    f"Unsupported file format '{path.suffix}'. Supported: {all_formats}"
                )

        # Instantiate backend
        self.backend = self._init_backend(backend, path, mpp, level)

    def _init_backend(
        self, backend: str, path: Path, mpp: float | None, level: int | str | None
    ):
        match backend:
            case "memory":
                if level is not None:
                    raise ValueError("'level' is not supported for memory backend.")
                return BrightfieldMemoryBackend(path, mpp)
            case "tifffile":
                return BrightfieldTiffBackend(path, mpp, level)
            case "openslide":
                return BrightfieldOpenslideBackend(path, mpp, level)
            case _:
                raise ValueError(
                    f"Unknown backend '{backend}'. Must be one of {list(BRIGHTFIELD_BACKENDS.keys())}."
                )

    @property
    def levels(self):
        return self.backend.levels

    @property
    def levels_keys(self):
        return self.backend.levels_keys

    @property
    def levels_shape(self):
        return self.backend.levels_shape

    @property
    def l(self):
        return self.backend.l

    @property
    def h(self):
        return self.backend.h

    @property
    def w(self):
        return self.backend.w

    @property
    def c(self):
        return self.backend.c

    @property
    def shape(self):
        return (self.h, self.w, self.c)

    @property
    def mpp_x(self):
        return self.backend.mpp_x

    @property
    def mpp_y(self):
        return self.backend.mpp_y

    @property
    def mpp(self):
        return self.backend.mpp

    def __getitem__(self, key: IndexKey) -> Numeric | NDArray[Numeric]:
        """Index into the image data.

        Parameters
        ----------
        key : IndexKey
            Numpy-style indexing.

        Returns
        -------
        Numeric | NDArray[Numeric]
            Pixel value or array of pixel values.
        """
        return self.backend[key]

    def tile(self, x: int, y: int, w: int, h: int) -> RGBTile:
        """Extract a tile from the image.

        Parameters
        ----------
        x : int
            X-coordinate of the top-left corner of the tile.
        y : int
            Y-coordinate of the top-left corner of the tile.
        w : int
            Width of the tile.
        h : int
            Height of the tile.

        Returns
        -------
        RGBTile
            RGBTile object.
        """
        if not (0 <= x < self.w) or not (0 <= y < self.h):
            raise ValueError(
                f"x/y coordinates ({x}, {y}) must be within image dimensions ({self.w}, {self.h})"
            )
        if w < 1 or h < 1:
            raise ValueError("Tile width and height must be greater than 0.")
        return RGBTile(self.backend[y : y + h, x : x + w])

    def thumbnail(
        self,
        t: int = 5000,
        force: bool = False,
        height: int | None = None,
        width: int | None = None,
        resample: str | None = "bilinear",
    ) -> RGBTile:
        """Generate a thumbnail of the image.

        Parameters
        ----------
        t : int
            Target size (maximum dimension) of the thumbnail.
        force : bool
            If True, generate thumbnail even if image exceeds target size
            or if image has no pyramid levels.
        height : int | None
            Optional height to resize the thumbnail to.
        width : int | None
            Optional width to resize the thumbnail to.
        resample : str | None
            Resampling method for resizing. One of 'nearest', 'bilinear',
            'bicubic', or 'lanczos'.

        Returns
        -------
        RGBTile
            Thumbnail as an RGBTile object.
        """
        if not self.levels_shape:
            if self.w > t or self.h > t:
                if force:
                    tile = RGBTile(self[:])
                else:
                    raise ValueError(
                        "Image has no pyramid levels and exceeds target size. "
                        "Use force=True to generate thumbnail anyway."
                    )
            else:
                tile = RGBTile(self[:])
        else:
            closest_key = min(
                self.levels_shape,
                key=lambda k: abs(self.levels_shape[k][0] - t)
                + abs(self.levels_shape[k][1] - t),
            )
            tile = RGBTile(Brightfield(self.backend.path, level=closest_key)[:])

        if height or width:
            tile = tile.resize(height, width, resample)

        return tile

    def __repr__(self):
        mpp_val = round(self.mpp, 4) if isinstance(self.mpp, float) else None
        return f"Brightfield(l={self.l}, h={self.h}, w={self.w}, c={self.c}, mpp={mpp_val})"

    def __getstate__(self):
        return {"path": self.backend.path}

    def __setstate__(self, state):
        self.__init__(state["path"])
