# Copyright (c) 2025, Tom Ouellette
# Licensed under the GNU GPLv3 License

import numpy as np
import zarr


def resize_nearest(img: np.ndarray, new_shape: tuple[int, int]) -> np.ndarray:
    """Resize array using nearest-neighbor interpolation.

    Parameters
    ----------
    img : np.ndarray
        Input image of shape (H, W) or (H, W, C)
    new_shape : (new_H, new_W)
        Desired output shape.

    Returns
    -------
    np.ndarray
        Resized image of shape (new_H, new_W[, C])
    """
    h, w = img.shape[:2]
    new_h, new_w = new_shape

    # Compute scale factors
    row_scale = h / new_h
    col_scale = w / new_w

    # Compute source indices
    row_idx = (np.arange(new_h) * row_scale).astype(int)
    col_idx = (np.arange(new_w) * col_scale).astype(int)

    # Clamp just in case (avoid out-of-bounds on last pixel)
    row_idx = np.clip(row_idx, 0, h - 1)
    col_idx = np.clip(col_idx, 0, w - 1)

    # Use advanced indexing for fast vectorized sampling
    if img.ndim == 3:
        return img[row_idx[:, None], col_idx[None, :], :]
    else:
        return img[row_idx[:, None], col_idx[None, :]]


def resize_mask_memmap(
    input_path: str,
    output_path: str,
    target_shape: tuple[int, ...],
    row_block: int = 1000,
    col_block: int = 1000,
) -> np.memmap:
    """Resize memory-mapped segmentation mask using block-wise nearest-neighbor indexing.

    Parameters
    ----------
    input_path : str
        Path to original .npy mask.
    output_path : str
        Path to save resized mask.
    target_shape : tuple[int, int]
        Resize to specified height and width.
    row_block : int
        Number of output rows per block.
    col_block : int
        Number of output columns per block.
    """
    # Load original mask as memmap
    orig_mask = np.load(input_path, mmap_mode="r")
    old_H, old_W = orig_mask.shape
    new_H, new_W = target_shape

    # Create output memmap
    resized_mask = np.lib.format.open_memmap(
        output_path, dtype=orig_mask.dtype, mode="w+", shape=(new_H, new_W)
    )

    # Precompute nearest-neighbor indices
    row_idx = (np.linspace(0, old_H - 1, new_H)).astype(int)
    col_idx = (np.linspace(0, old_W - 1, new_W)).astype(int)

    # Process in row + column blocks
    for row_start in range(0, new_H, row_block):
        row_end = min(row_start + row_block, new_H)
        rows = row_idx[row_start:row_end]

        for col_start in range(0, new_W, col_block):
            col_end = min(col_start + col_block, new_W)
            cols = col_idx[col_start:col_end]

            # Slice original mask and write to output memmap
            resized_mask[row_start:row_end, col_start:col_end] = orig_mask[
                rows[:, None], cols[None, :]
            ]

    # Flush changes to disk
    resized_mask.flush()
    return resized_mask


def resize_mask_zarr(
    input_path: str,
    output_path: str,
    target_shape: tuple[int, int],
    row_block: int = 1000,
    col_block: int = 1000,
) -> zarr.Array:
    """
    Resize a Zarr segmentation mask array using block-wise nearest-neighbor interpolation.

    Parameters
    ----------
    input_path : str
        Path to source .zarr directory.
    output_path : str
        Path to destination .zarr directory.
    target_shape : tuple[int, int]
        Desired (height, width) of the output mask.
    row_block : int
        Number of output rows per block.
    col_block : int
        Number of output columns per block.

    Returns
    -------
    zarr.Array
        Resized Zarr array on disk.
    """
    # Open input Zarr in read-only mode
    src = zarr.open(input_path, mode="r")
    old_H, old_W = src.shape
    dtype = src.dtype

    new_H, new_W = target_shape

    # Create destination Zarr array
    dst = zarr.open(
        output_path,
        mode="w",
        shape=(new_H, new_W),
        dtype=dtype,
        chunks=(row_block, col_block),
    )

    # Precompute nearest-neighbor index maps
    row_idx = np.linspace(0, old_H - 1, new_H).astype(int)
    col_idx = np.linspace(0, old_W - 1, new_W).astype(int)

    # Iterate through output blocks
    for row_start in range(0, new_H, row_block):
        row_end = min(row_start + row_block, new_H)
        rows = row_idx[row_start:row_end]

        for col_start in range(0, new_W, col_block):
            col_end = min(col_start + col_block, new_W)
            cols = col_idx[col_start:col_end]

            # Read block from source and write to destination
            dst[row_start:row_end, col_start:col_end] = src[
                rows[:, None], cols[None, :]
            ]

    return dst
