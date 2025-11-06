# Copyright (c) 2025, Tom Ouellette
# Licensed under the GNU GPLv3 License

import os
import shutil
import numpy as np
import tempfile

from anci import Arg, cmd, base
from pathlib import Path

from bamboo.tools._resize import resize_mask_memmap, resize_mask_zarr
from bamboo.writer import NumpyWriter, SharedMemoryWriter, MemmapWriter, ZarrWriter

SUPPORTED_MODELS: set[str] = {"grandqc"}


@base("brightfield", "nn")
def brightfield_nn():
    """Neural networks for brightfield imaging."""
    pass


@cmd("brightfield", "nn", "semantic")
def brightfield_nn_semantic(
    path: Arg[Path, "Path to brightfield image."],
    model: Arg[str, "Model name."],
    output: Arg[Path, "Output path."],
    mmap: Arg[bool, "If True, use a memory map when writing .npy outputs."] = True,
    n_jobs: Arg[int, "Number of parallel jobs."] = 1,
    prefer: Arg[str, "Parallel backend hint."] = None,
    resample: Arg[str, "Resampling filter to use when resizing tiles."] = "bilinear",
    resize: Arg[str, "Resize segmentation mask to original image size."] = False,
    device: Arg[str, "Device to run model on."] = "cpu",
    silent: Arg[bool, "Suppress printed messages."] = False,
):
    """Generate a semantic segmentation mask from an input brightfield image."""

    from bamboo.image import Brightfield
    from bamboo.models import Semantic
    from bamboo.tile import Tiles

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
            from bamboo.models.zoo import GrandQC

            model = GrandQC(device=device)

    match output.suffix:
        case ".npy":
            if mmap:
                writer = MemmapWriter(dtype=model.dtype, path=output)
            else:
                if n_jobs == 1:
                    writer = NumpyWriter(dtype=model.dtype)
                else:
                    writer = SharedMemoryWriter(dtype=model.dtype)
        case ".zarr":
            writer = ZarrWriter(dtype=model.dtype, path=output)
        case _:
            raise ValueError(f"The --output {output} must end with .npy or .zarr.")

    tiles = Tiles(
        image,
        model.tile_width,
        model.tile_height,
        model.mpp,
        resample=resample,
        overlap_x=0,
        overlap_y=0,
    )

    engine = Semantic(
        model,
        tiles,
        writer=writer,
        n_jobs=n_jobs,
        prefer=prefer,
    )

    results = engine.run(silent=silent)

    if output.suffix == ".npy" and not mmap:
        np.save(output, results)
        writer.cleanup()
