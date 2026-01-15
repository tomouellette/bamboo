# Copyright (c) 2025, Tom Ouellette
# Licensed under the GNU GPLv3 License

import numpy as np

from anci import Arg, cmd, base
from pathlib import Path

from bamboo.types import Array
from bamboo.map import apply_map

from bamboo.writer import (
    NumpyWriter,
    SharedMemoryWriter,
    MemmapWriter,
    ZarrWriter,
    inject_writer,
)

SUPPORTED_MODELS: set[str] = {"grandqc"}


@base("brightfield", "tile")
def brightfield_nn():
    """Methods for processing, filtering, and writing brightfield tiles,."""
    pass


@cmd("brightfield", "tile", "extract")
def brightfield_tile_extract(
    path: Arg[Path, "Path to brightfield image."],
    output: Arg[Path, "Output path."],
    tile_width: Arg[int, "Tile width."],
    tile_height: Arg[int, "Tile height."],
    mpp: Arg[float, "Collect tiles at the specified microns per pixel."] = None,
    overlap_x: Arg[int, "Horizontal overlap when saving tiles."] = 0,
    overlap_y: Arg[int, "Vertical overlap when saving tiles."] = 0,
    mmap: Arg[bool, "If True, use a memory map when writing .npy outputs."] = True,
    n_jobs: Arg[int, "Number of parallel jobs."] = 1,
    prefer: Arg[str, "Parallel backend hint."] = None,
    resample: Arg[str, "Resampling filter to use when resizing tiles."] = "bilinear",
    save_fig: Arg[Path, "Save image thumbnail with tiles overlaid."] = None,
    silent: Arg[bool, "Suppress printed messages."] = False,
):
    """Extract and write all tiles from a brightfield gigapixel image."""

    from bamboo.image import Brightfield
    from bamboo.tile import Tiles, Tile

    image = Brightfield(path)

    if prefer not in ["processes", "threads", None]:
        raise ValueError(
            f"''--prefer' {prefer} is not valid. "
            "Must be one of 'processes', 'threads', or None."
        )

    match output.suffix:
        case ".npy":
            if mmap:
                writer = MemmapWriter(dtype=np.uint8, path=output)
            else:
                if n_jobs == 1:
                    writer = NumpyWriter(dtype=np.uint8)
                else:
                    writer = SharedMemoryWriter(dtype=np.uint8)
        case ".zarr":
            writer = ZarrWriter(dtype=np.uint8, path=output)
        case _:
            raise ValueError(f"The --output {output} must end with .npy or .zarr.")

    mpp = image.mpp if mpp is None else mpp

    tiles = Tiles(
        image,
        tile_width,
        tile_height,
        mpp,
        resample=resample,
        overlap_x=overlap_x,
        overlap_y=overlap_y,
        mode="enumerate",
    )

    if isinstance(writer, ZarrWriter):
        writer.chunks = (tile_height, tile_width, 3)

    writer.shape = (tiles.n_tiles, tile_height, tile_width, 3)

    @inject_writer(writer)
    def _worker(output: Array, data: tuple[tuple[int, ...], Tile]) -> np.ndarray:
        i, tile = data
        output[i, :tile_height, :tile_width, :3] = tile.buffer

    writer.create()

    try:
        if save_fig:
            tiles.show(save_fig=save_fig)

        apply_map(
            _worker,
            tiles,
            n_jobs,
            prefer,
            silent,
        )

        if output.suffix == ".npy" and not mmap:
            np.save(output, writer.output)
            writer.cleanup()
    except Exception as e:
        writer.cleanup()
        raise RuntimeError("Failed to write tiles to disk") from e


@cmd("brightfield", "tile", "filter")
def brightfield_tile_filter(
    path: Arg[Path, "Path to brightfield image."],
    model: Arg[str, "Filtering model (one of: 'grandqc')."],
    output: Arg[Path, "Output path."],
    tile_width: Arg[int, "Tile width."],
    tile_height: Arg[int, "Tile height."],
    min_tissue_fraction: Arg[float, "Minimum tissue fraction to keep a tile."] = 0.5,
    max_artifact_fraction: Arg[float, "Max artifact fraction to keep a tile."] = 0.0,
    max_marker_fraction: Arg[float, "Max marker fraction to keep a tile."] = 0.0,
    mpp: Arg[float, "Collect tiles at the specified microns per pixel."] = None,
    overlap_x: Arg[int, "Horizontal overlap when saving tiles."] = 0,
    overlap_y: Arg[int, "Vertical overlap when saving tiles."] = 0,
    mmap: Arg[bool, "If True, use a memory map when writing .npy outputs."] = True,
    n_jobs: Arg[int, "Number of parallel jobs."] = 1,
    prefer: Arg[str, "Parallel backend hint."] = None,
    resample: Arg[str, "Resampling filter to use when resizing tiles."] = "bilinear",
    device: Arg[str, "Device to run model on."] = "cpu",
    silent: Arg[bool, "Suppress printed messages."] = False,
    save_fig: Arg[Path, "Save image thumbnail with tiles overlaid."] = None,
):
    """Filter and write all tiles from a brightfield gigapixel image."""

    from bamboo.image import Brightfield
    from bamboo.tile import Tiles, Tile
    from bamboo.models import SemanticStats

    image = Brightfield(path)

    if prefer not in ["processes", "threads", None]:
        raise ValueError(
            f"''--prefer' {prefer} is not valid. "
            "Must be one of 'processes', 'threads', or None."
        )

    if not any(
        [device.startswith("cpu"), device.startswith("cuda"), device.startswith("mps")]
    ):
        raise ValueError(
            f"The '--device' {device} is not valid. "
            "Currently supported devices include: 'cpu', 'cuda', 'mps'"
        )

    if model not in SUPPORTED_MODELS:
        raise ValueError(
            f"The '--model' {model} is not valid. "
            f"Currently supported models include: {SUPPORTED_MODELS}"
        )

    match model:
        case "grandqc":
            from bamboo.models.zoo import GrandQC, GrandQCStats

            model = GrandQC(device=device)
            stats = GrandQCStats

    match output.suffix:
        case ".npy":
            if mmap:
                writer = MemmapWriter(dtype=np.uint8, path=output)
            else:
                if n_jobs == 1:
                    writer = NumpyWriter(dtype=np.uint8)
                else:
                    writer = SharedMemoryWriter(dtype=np.uint8)
        case ".zarr":
            writer = ZarrWriter(dtype=np.uint8, path=output)
        case _:
            raise ValueError(f"The --output {output} must end with .npy or .zarr.")

    mpp = image.mpp if mpp is None else mpp

    # Note: To filter out "bad" tiles we first have to apply a model to each tile
    # compute the relevant statistics and then re-iterate through the retained set
    # of coordinates once more for writing. As such the writer shouldn't be created
    # until the filtering step has been performed.

    tiles = Tiles(
        image,
        model.tile_width,
        model.tile_height,
        model.mpp,
        resample=resample,
        overlap_x=0,
        overlap_y=0,
        mode="coords",
    )

    try:
        engine = SemanticStats(model, stats, tiles, n_jobs, prefer)
        results = engine.run(silent)
    except Exception as e:
        raise RuntimeError(f"Filtering failed using {model} model.") from e

    exclude_coords = []
    for r in results:
        check_tissue = r["stats"]["TISSUE"] < min_tissue_fraction
        check_artifact = r["stats"]["ARTIFACT"] > max_artifact_fraction
        check_marker = False
        if "MARKINGS" in r["stats"]:
            check_marker = r["stats"]["MARKINGS"] > max_marker_fraction

        if any([check_tissue, check_artifact, check_marker]):
            exclude_coords.append(r["coords"])

    # Note: We construct new tiles at the requested size and resolution, then
    # drop the tiles that not pass the quality control metrics. This seems a
    # bit redundant generating tiles twice. An alternative would be to add a
    # method to `Tiles` to regenerate coordinates at a new size given a set of
    # coordinates to exclude.
    tiles = Tiles(
        image,
        tile_width,
        tile_height,
        mpp,
        resample=resample,
        overlap_x=overlap_x,
        overlap_y=overlap_y,
        mode="coords",
    )

    n_tiles = tiles.n_tiles
    tiles.exclude(exclude_coords)

    if not silent:
        print(
            f"Retained {n_tiles} and dropped {n_tiles - tiles.n_tiles} after filtering."
        )

    # Note: We switch back to enumeration here since we don't need coordinate
    # information to write to tiles. With that said, users may want coordinate
    # information to map tiles back to original images after downstream analysis
    # so maybe come back to here when that arises. If we simplified the API to
    # just work with zarrs then we could store coordinate information in the meta
    # data.
    tiles.mode = "enumerate"

    if isinstance(writer, ZarrWriter):
        writer.chunks = (1, tile_height, tile_width, 3)

    writer.shape = (tiles.n_tiles, tile_height, tile_width, 3)

    writer.create()

    @inject_writer(writer)
    def _worker(output: Array, data: tuple[tuple[int, ...], Tile]) -> np.ndarray:
        i, tile = data
        output[i, :tile_height, :tile_width, :3] = tile.buffer

    try:
        if save_fig:
            tiles.show(save_fig=save_fig)

        apply_map(
            _worker,
            tiles,
            n_jobs,
            prefer,
            silent,
        )

        if output.suffix == ".npy" and not mmap:
            np.save(output, writer.output)
            writer.cleanup()
    except Exception as e:
        writer.cleanup()
        raise RuntimeError("Failed to write tiles to disk") from e
