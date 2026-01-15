# Copyright (c) 2025, Tom Ouellette
# Licensed under the GNU GPLv3 License

import numpy as np

from typing import Callable

from bamboo.tile import Tiles, Tile
from bamboo.tools._resize import resize_nearest
from bamboo.types import Array
from bamboo.map import apply_map
from bamboo.writer import ArrayWriter, NumpyWriter, ZarrWriter, inject_writer


class Semantic:
    """Perform semantic segmentation across tiled images.

    Parameters
    ----------
    func : Callable[Tile, np.ndarray]
        A semantic segmentation model that takes as input a tile and
        returns as output a numpy ndarray with shape (h, w).
    tiles : Tiles
        An iterable of tiles that are scaled to a size compatible with
        the provided semantic segmentation model.
    writer : ArrayWriter
        Array writer backend. If `str`, then one of 'numpy', 'shared', or
        'memmap'. Otherwise, a valid ArrayWriter derived class.
    n_jobs : None | int
        Number of parallel jobs.
    prefer : None | str
        Either None, 'processes, or 'threads'. If n_jobs and prefer are both
        None, then no parallelism will be used (n_jobs == 1). If n_jobs > 1 and
        prefer is None, then 'processes' will be used. If prefer is 'processes'
        and n_jobs is None, all available processes will be used.

    Example
    -------
    This example demonstrates the general usage of the `Semantic` runner class.
    A `Brightfield` image is loaded and then a `Semantic` segmentation runner is
    initialized using a `GrandQC` model, a `Tiles` tile generator at a specific
    tile size and microns per pixel resolution, and with the results are written
    to a memory mapped array using `MemmapWriter`.

    >>> from bamboo.image import Brightfield
    >>> from bamboo.tile import Tiles
    >>> from bamboo.writer import MemmapWriter
    >>> from bamboo.models import Semantic
    >>> from bamboo.models.zoo import GrandQC

    >>> image = Brightfield("slide.svs")

    >>> runner = Semantic(
    ...     GrandQC(device="cpu"),
    ...     Tiles(image, tile_width=512, tile_height=512, mpp=1.5),
    ...     writer=MemmapWriter(dtype=np.uint8, path="array.npy"),
    ...     n_jobs=8,
    ...     prefer="processes",
    ... )

    >>> output = runner.run()
    """

    def __init__(
        self,
        func: Callable[[Tile], np.ndarray],
        tiles: Tiles,
        writer: ArrayWriter,
        n_jobs: int | None = None,
        prefer: str | None = None,
    ):
        if prefer == "processes" and isinstance(writer, NumpyWriter):
            raise ValueError(
                "If prefer is 'processes', then writer must one of: "
                "`SharedMemoryWriter` or `MemmapWriter`."
            )

        self.func = func
        self.tiles = tiles
        self.n_jobs = n_jobs
        self.prefer = prefer
        self.writer = writer
        self.dtype = writer.dtype

        if isinstance(writer, ZarrWriter):
            writer.chunks = (tiles.crop_h, tiles.crop_w)

        if tiles.scale is None and writer.shape is None:
            raise ValueError(
                "If 'tiles' has no scale or microns per pixel information,"
                "then `writer` must have shape set prior to running `Semantic`."
            )

        if writer.shape is None:
            writer.shape = (tiles.image_h, tiles.image_w)

    def run(self, silent: bool = False) -> Array:
        self.writer.create()

        @inject_writer(self.writer)
        def _worker(output: Array, data: tuple[tuple[int, ...], Tile]) -> np.ndarray:
            _, (x, y, w, h, overlap_w, overlap_h), tile = data

            processed = self.func(tile).astype(output.dtype)
            if processed.shape[:2] != (h, w):
                processed = resize_nearest(processed, (h, w))

            output[y + overlap_h : y + h, x + overlap_w : x + w] = processed[
                overlap_h:, overlap_w:
            ]

        apply_map(
            _worker,
            self.tiles,
            self.n_jobs,
            self.prefer,
            silent,
        )

        return self.writer.output


class SemanticStats:
    """Run a semantic segmentation model and compute statistics on each tile mask.

    Parameters
    ----------
    func : Callable[[Tile], np.ndarray]
        A semantic segmentation function that takes a tile and returns a mask (H, W).
    counter : Callable[np.ndarray, dict]
        A function that computes statistics using the generated mask.
    tiles : Tiles
        Iterable of tiles compatible with `func`.
    n_jobs : None | int, optional
        Number of parallel jobs for processing tiles.
    prefer : None | str, optional
        Parallel backend preference ('processes' or 'threads').

    Returns
    -------
    list[dict]
        Each entry corresponds to a tile and includes its coordinates and statistics.
    """

    def __init__(
        self,
        func: Callable[[Tile], np.ndarray],
        counter: Callable[np.ndarray, dict],
        tiles: Tiles,
        n_jobs: int | None = None,
        prefer: str | None = None,
    ):
        self.func = func
        self.counter = counter
        self.tiles = tiles
        self.n_jobs = n_jobs
        self.prefer = prefer

    def run(self, silent: bool = False):
        """Run the segmentation and compute statistics for each tile."""
        results = []

        def _worker(data: tuple[tuple[int, ...], Tile]):
            _, (x, y, w, h, overlap_w, overlap_h), tile = data
            mask = self.func(tile)
            stats = self.counter(mask)
            return {"stats": stats, "coords": [x, y, w, h, overlap_w, overlap_h]}

        results = apply_map(
            _worker,
            self.tiles,
            self.n_jobs,
            self.prefer,
            silent,
        )

        return results
