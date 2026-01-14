import pytest
import numpy as np
from pathlib import Path
from bamboo.image import Brightfield
from bamboo.tile import RGBTile

from tests._fixtures.slides import (
    SlideBif,
    SlideNdpi,
    SlidePhilips,
    SlideLeica,
    SlideSvs,
    SlideTiff,
    SlideTrestle,
    SlideOmeTiff,
    SlideError,
)

from tests._fixtures.images import (
    RgbBmp,
    RgbJpeg,
    RgbNpy,
    RgbPbm,
    RgbPng,
    RgbTga,
    RgbWebp,
)


def test_init_str():
    image = Brightfield(SlideSvs.path)
    for attr in [
        "backend",
        "levels",
        "levels_keys",
        "levels_shape",
        "l",
        "h",
        "w",
        "c",
        "mpp",
        "mpp_x",
        "mpp_y",
    ]:
        assert hasattr(image, attr)


def test_init_pathlib():
    image = Brightfield(Path(SlideSvs.path))
    for attr in [
        "backend",
        "levels",
        "levels_keys",
        "levels_shape",
        "l",
        "h",
        "w",
        "c",
        "mpp",
        "mpp_x",
        "mpp_y",
    ]:
        assert hasattr(image, attr)


def test_init_error_path():
    with pytest.raises(ValueError):
        _ = Brightfield(SlideError.path)


def test_slides_auto():
    for slide in [
        SlideBif,
        SlideNdpi,
        SlidePhilips,
        SlideLeica,
        SlideSvs,
        SlideTiff,
        SlideOmeTiff,
    ]:
        _ = Brightfield(slide.path)

    with pytest.raises(ValueError):
        _ = Brightfield(SlideTrestle.path)


def test_images_auto():
    for image in [RgbBmp, RgbJpeg, RgbNpy, RgbPbm, RgbPng, RgbTga, RgbWebp]:
        _ = Brightfield(image.path, mpp=0.5)


def test_backend_tifffile():
    _ = Brightfield(SlideSvs.path, backend="tifffile")


def test_backend_openslide():
    _ = Brightfield(SlideSvs.path, backend="openslide")


def test_backend_memory():
    _ = Brightfield(RgbBmp.path, backend="memory", mpp=0.5)


def test_indexing():
    image = Brightfield(SlideSvs.path, level=2)
    h, w, c = image.backend.h, image.backend.w, 3

    assert isinstance(image[0, 0, 0], (int, np.integer))
    assert image[0:10, 0, 0].shape == (10,)
    assert image[0, 0:10, 0].shape == (10,)
    assert image[0, 0, 0:c].shape == (c,)
    assert image[0:5, 0:5, 0].shape == (5, 5)
    assert image[0:5, 0, 0:c].shape == (5, c)
    assert image[0, 0:5, 0:c].shape == (5, c)
    assert image[0:2, 0:2, 0:c].shape == (2, 2, c)
    assert image[...].shape == (h, w, c)
    assert image[..., 0].shape == (h, w)
    assert image[0, ..., 0].shape == (w,)
    assert image[0].shape == (w, c)
    assert image[0:10].shape[0] == 10


def test_tile():
    image = Brightfield(SlideSvs.path)
    tile = image.tile(0, 0, 512, 256)
    assert tile.w == 512
    assert tile.h == 256


def test_thumbnail_default():
    image = Brightfield(SlideSvs.path)
    h, w, c = list(image.levels_shape.values())[-1]
    thumbnail = image.thumbnail(t=0)
    assert isinstance(thumbnail, RGBTile)
    assert thumbnail.h == h
    assert thumbnail.w == w
    assert thumbnail.c == c


def test_thumbnail_resize():
    image = Brightfield(SlideSvs.path)
    thumbnail = image.thumbnail(height=100, width=100)
    assert thumbnail.h == 100
    assert thumbnail.w == 100
    assert thumbnail.c == 3


def test_thumbnail_resize_height():
    image = Brightfield(SlideSvs.path)
    h, w = image.h, image.w
    thumbnail = image.thumbnail(height=1000)
    tolerance = 1 / min(thumbnail.w, thumbnail.h)
    assert abs(thumbnail.w / thumbnail.h - w / h) < tolerance


def test_thumbnail_resize_width():
    image = Brightfield(SlideSvs.path)
    h, w = image.h, image.w
    thumbnail = image.thumbnail(width=1000)
    tolerance = 1 / min(thumbnail.w, thumbnail.h)
    assert abs(thumbnail.w / thumbnail.h - w / h) < tolerance
