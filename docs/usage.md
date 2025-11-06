# Usage

- [Interactive](#interactive)
  - [Basic usage](#basic-usage)
  - [Tiles](#tiles)
- [Command-line](#command-line)
  - [`brightfield`](#brightfield)
    - [`brightfield nn`](#brightfield-nn)
    - [`brightfield tile`](#brightfield-tile)
- [Development](#development)

# Interactive

## Basic usage

At the most basic level `bamboo` provides a simple interface to interact with gigapixel images in an array-like or object-oriented manner.

```python
from bamboo.image import Brightfield

# Load an RGB formatted slide
image = Brightfield("histology.svs")

# Extract arrays via numpy-like indexing and slicing
array = image[0:100, 0:100]

# Grab a tile with a variety of useful methods 
tile = image.tile(0, 0, 100, 100)

# View the tile
tile.show()

# Resize the tile
tile = tile.resize(10, 10)

# Write the tile
tile.write("tile.png")

# Generate a thumbnail of the full image
thumbnail = image.thumbnail()
```

## Tiles

Processing large gigapixel images (e.g. histology) generally requires iteration across smaller patches or tiles. `bamboo` supports this by allowing generation of tiles at a user-specified size or resolution.

```python
from bamboo.tile import Tiles

# Instantiate a tile generator 
tiles = Tiles(
    image,
    tile_w=256,
    tile_h=256,
    mpp=0.25,
    resample=resample,
    overlap_x=0,
    overlap_y=0,
)

# View the tiles drawn on the image prior to iteration
tiles.show()

# Iterate through the tiles (parallel-friendly)
for tile in tiles:
    sum = tile.buffer.sum()

# A useful argument for `Tiles` is mode = "coords"
for (x, y, w, h, overlap_x, overlap_y), tile in tiles:
    sum = tile.buffer.sum()

# Or, mode = "enumerate"
for i, tile in tiles:
    sum = tile.buffer.sum()

# Or, mode = "enumerate+coords"
for i, (x, y, w, h, overlap_x, overlap_y), tile in tiles:
    sum = tile.buffer.sum()
```

Note that `Tiles` is somewhat smart. It automatically computes an appropriate crop size internally so that the final requested tile dimensions and microns per pixel is achieved. This will fail, however, if the combination isn’t possible given the image’s resolution.

# Command-line

A variety of reproducible processing and analysis tasks are available via a command-line interface.

## `brightfield` 

### `brightfield nn` 

#### `brightfield nn semantic`

Generate a semantic segmentation mask from an input brightfield image.

```bash
bamboo brightfield nn semantic \
    --path PATH \          # Path to brightfield image.
    --model MODEL \        # Model name (e.g. grandqc)
    --output OUTPUT \      # Output path.
    --mmap MMAP \          # Use a memory map when writing .npy outputs (default: True).
    --n_jobs N_JOBS \      # Number of parallel jobs
    --prefer PREFER \      # Parallel backend hint (e.g. None, processes, threads).
    --resample RESAMPLE \  # Resampling filter to use when resizing tiles (e.g. bilinear).
    --device DEVICE \      # Device to run model on (e.g. cpu, cuda, mps).
    --silent SILENT        # Suppress printed messages (default: False).
```

### `brightfield tile`

#### `brightfield tile extract`

Extract and write all tiles from a brightfield gigapixel image.

```bash
bamboo brightfield tile extract \
    --path PATH \                # Path to brightfield image.
    --output OUTPUT \            # Output path (e.g. tiles.zarr)
    --tile_width TILE_WIDTH \    # Tile width.
    --tile_height TILE_HEIGHT \  # Tile height
    --mpp MPP \                  # Collect tiles at the specified microns per pixel (default: None).
    --overlap_x OVERLAP_X \      # Horizontal overlap when saving tiles (default: 0).
    --overlap_y OVERLAP_Y \      # Vertical overlap when saving tiles (default: 0).
    --mmap MMAP \                # If True, use a memory map when writing .npy outputs (default: True).
    --n_jobs N_JOBS \            # Number of parallel jobs (default: 1).
    --prefer PREFER \            # Parallel backend hint (default: None).
    --resample RESAMPLE \        # Resampling filter to use when resizing tiles (default: bilinear).
    --silent SILENT              # Suppress printed messages (default: False).
```

#### `brightfield tile filter`

Filter and write all tiles from a brightfield gigapixel image.

```bash
bamboo brightfield tile filter \
    --path PATH \                                    # Path to brightfield image.
    --model MODEL \                                  # Filtering model (one of: 'grandqc').
    --output OUTPUT \                                # Output path.
    --tile_width TILE_WIDTH \                        # Tile width.
    --tile_height TILE_HEIGHT \                      # Tile height.
    --min_tissue_fraction MIN_TISSUE_FRACTION \      # Minimum tissue fraction to keep a tile (default: 0.5).
    --max_artifact_fraction MAX_ARTIFACT_FRACTION \  # Max artifact fraction to keep a tile (default: 0.0).
    --mpp MPP \                                      # Collect tiles at the specified microns per pixel (default: None).
    --overlap_x OVERLAP_X \                          # Horizontal overlap when saving tiles (default: 0).
    --overlap_y OVERLAP_Y \                          # Vertical overlap when saving tiles (default: 0).
    --mmap MMAP \                                    # If True, use a memory map when writing .npy outputs (default: True).
    --n_jobs N_JOBS \                                # Number of parallel jobs (default: 1).
    --prefer PREFER \                                # Parallel backend hint (default: None).
    --resample RESAMPLE \                            # Resampling filter to use when resizing tiles (default: bilinear).
    --device DEVICE \                                # Device to run model on (default: cpu).
    --silent SILENT                                  # Suppress printed messages (default: False).
```

# Development

`bamboo` is currently built around a variety of open source image readers (`tifffile`, `PIL`, and, optionally, `openslide`). Additionally, there is support for writing segmentation masks, tiles, and self-supervised features to `zarr`, `numpy.ndarray`, and `numpy.memmap`. Continued support for `numpy` based outputs remains tentative as transitioning completely to `zarr` for writing data would greatly simplify the API.
