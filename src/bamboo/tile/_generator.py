# Copyright (c) 2025, Tom Ouellette
# Licensed under the GNU GPLv3 License

from typing import Any, Iterator


class Tiles(Iterator):
    """A tile generator yielding tile index, coordinates, and pixel data.

    Parameters
    ----------
    image : Any
        An Image object.
    tile_w : int
        Tile width in pixels.
    tile_h : int
        Tile height in pixels.
    mpp : float | None
        Target microns per pixel for resizing.
    resample : str | None
        Resampling filter if resizing is performed.
    overlap_x, overlap_y : int
        Tile overlap in pixels.
    _scale : float | None
        Optional scale override (for testing/debugging).
    """

    def __init__(
        self,
        image: Any,
        tile_w: int,
        tile_h: int,
        mpp: None | float = None,
        resample: str | None = "bilinear",
        overlap_x: int = 0,
        overlap_y: int = 0,
        mode: str = "default",
        _scale: None | float = None,
    ):
        self.image = image

        # Handle scaling and mpp
        if _scale:
            self.scale = _scale
        else:
            self.scale = None
            if mpp is not None:
                if not hasattr(image, "mpp"):
                    raise ValueError("Image must have an 'mpp' attribute to use 'mpp'.")
                self.scale = mpp / image.mpp

        self.image_w = image.w
        self.image_h = image.h

        self.crop_w = int(round(tile_w * (self.scale if self.scale else 1.0)))
        self.crop_h = int(round(tile_h * (self.scale if self.scale else 1.0)))

        if self.crop_w <= 0 or self.crop_h <= 0:
            raise ValueError("Invalid tile size after rescaling.")

        self.overlap_x = overlap_x
        self.overlap_y = overlap_y
        self.tile_w = tile_w
        self.tile_h = tile_h
        self.resize = bool(self.scale)
        self.resample = resample

        # Compute coordinates
        x_min, y_min = 0, 0
        x_max, y_max = image.w, image.h
        step_x = self.crop_w - self.overlap_x
        step_y = self.crop_h - self.overlap_y

        if step_x <= 0 or step_y <= 0:
            raise ValueError("Overlap must be less than tile dimension.")

        self.coordinates = []
        for y in range(y_min, y_max, step_y):
            for x in range(x_min, x_max, step_x):
                orig_x, orig_y = x, y
                if x + self.crop_w > image.w:
                    x = image.w - self.crop_w
                if y + self.crop_h > image.h:
                    y = image.h - self.crop_h
                overlap_w = max(0, orig_x - x)
                overlap_h = max(0, orig_y - y)
                self.coordinates.append(
                    (x, y, self.crop_w, self.crop_h, overlap_w, overlap_h)
                )

        self.n_tiles = len(self.coordinates)
        self._index = 0

    @property
    def hw(self):
        return (self.tile_h, self.tile_w) if self.resize else (self.crop_h, self.crop_w)

    def exclude(self, coords: list[tuple[int, int, int, int]]) -> None:
        """Remove tiles that overlap with any in the given exclusion coordinates.

        Parameters
        ----------
        coords : list of (x, y, w, h)
            Tile coordinates to exclude (e.g., from QC results).
        """

        def overlaps(a, b):
            ax, ay, aw, ah, *_ = a
            bx, by, bw, bh, *_ = b
            return ax < bx + bw and ax + aw > bx and ay < by + bh and ay + ah > by

        filtered = [
            coord
            for coord in self.coordinates
            if not any(overlaps(coord, ex) for ex in coords)
        ]

        self.coordinates = filtered
        self.n_tiles = len(filtered)
        self._index = 0

    def __len__(self):
        return len(self.coordinates)

    def __iter__(self):
        self._index = 0
        return self

    def __next__(self):
        if self._index >= self.n_tiles:
            raise StopIteration

        coord = self.coordinates[self._index]
        x, y, w, h, overlap_w, overlap_h = coord
        tile = self.image.tile(x, y, w, h)
        if self.resize:
            tile = tile.resize(self.tile_w, self.tile_h, self.resample)

        idx = self._index
        self._index += 1

        return idx, coord, tile
