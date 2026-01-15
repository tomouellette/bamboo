import pytest
import numpy as np

from bamboo.error import ShapeError
from bamboo.image import Brightfield
from bamboo.tile import RGBTile, Tiles
from tests._fixtures.slides import SlideSvs


def test_rgb_tile_init():
    tile = RGBTile(np.random.randint(0, 255, (10, 10, 3), dtype=np.uint8))
    assert tile.shape == (10, 10, 3)
    assert tile.dtype == np.uint8
    assert tile.h == 10
    assert tile.w == 10
    assert tile.c == 3


def test_rgb_tile_type_error():
    with pytest.raises(TypeError):
        _ = RGBTile(np.random.rand(10, 10, 3))


def test_rgb_tile_dim_error():
    with pytest.raises(ShapeError):
        _ = RGBTile(np.random.rand(10, 10).astype(np.uint8))


def test_rgb_tile_channel_error():
    with pytest.raises(ShapeError):
        _ = RGBTile(np.random.randint(0, 255, (10, 10, 4), dtype=np.uint8))


def test_rgb_tile_resize():
    tile = RGBTile(np.random.randint(0, 255, (10, 10, 3), dtype=np.uint8))
    resized = tile.resize(5, 10)
    assert resized.shape == (10, 5, 3)


def test_tiles_basic_iteration():
    image = Brightfield(SlideSvs.path, level=2)
    for *_, tile in Tiles(image, 128, 128):
        assert tile.shape == (128, 128, 3)


def test_tiles_overlap():
    image = Brightfield(SlideSvs.path, level=2)
    tiles = Tiles(image, tile_w=128, tile_h=128, overlap_x=64, overlap_y=64)

    *_, tile_one = next(tiles)
    *_, tile_two = next(tiles)

    # Check that overlapping region matches
    assert np.all(tile_one[:, 64:] == tile_two[:, :64])


class DummyImage:
    """Mock image with minimal API to support Tiles."""

    def __init__(self, w, h):
        self.w = w
        self.h = h
        self.mpp = 0.5  # microns per pixel

    def tile(self, x, y, w, h):
        # Return a dummy placeholder to simulate a Tile object
        return f"tile@({x},{y},{w},{h})"

    def resize(self, w, h, resample=None):
        return self

    def thumbnail(self, **kwargs):
        return self

    def to_numpy(self):
        import numpy as np

        return np.zeros((10, 10, 3), dtype=np.uint8)

    @property
    def shape(self):
        return (self.h, self.w, 3)


def test_exclude_overlaps_removes_expected_tiles():
    """Ensure that overlapping tiles are correctly excluded."""
    img = DummyImage(2048, 2048)
    tiles = Tiles(image=img, tile_w=512, tile_h=512)

    # Expect a 4x4 grid of tiles for a 2048x2048 image
    assert len(tiles) == 16

    # Exclude region overlapping roughly the top-left 2x2 tiles
    exclude = [(0, 0, 600, 600)]

    # Apply exclusion
    tiles.exclude(exclude)

    # Should remove 4 tiles: (0,0), (512,0), (0,512), (512,512)
    remaining = len(tiles)
    assert remaining == 12

    # Ensure no remaining tiles overlap with the exclusion area
    for coord in tiles.coordinates:
        x, y, w, h, *_ = coord
        # Simple overlap check
        overlap = x < 600 and x + w > 0 and y < 600 and y + h > 0
        assert not overlap, f"Tile at ({x},{y}) should have been excluded"


def test_exclude_overlaps_no_overlap_keeps_all():
    """Tiles with no overlap should remain unaffected."""
    img = DummyImage(1024, 1024)
    tiles = Tiles(image=img, tile_w=256, tile_h=256)
    n_before = len(tiles)

    # Exclude region far outside the image
    exclude = [(2000, 2000, 256, 256)]
    tiles.exclude(exclude)

    # Should keep all tiles
    assert len(tiles) == n_before
