import pytest
import numpy as np
import openslide
from pathlib import Path
from zarr.core.group import Group as ZarrGroup
from zarr.core.group import Array as ZarrArray

from bamboo.backend._brightfield import (
    BrightfieldTiffBackend,
    BrightfieldOpenslideBackend,
    BrightfieldMemoryBackend,
)

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

FORMATS_TIFF_BACKEND = [
    SlideNdpi,
    SlidePhilips,
    SlideLeica,
    SlideSvs,
    SlideTiff,
    SlideOmeTiff,
]

FORMATS_OPENSLIDE_BACKEND = [
    SlideBif,
    SlideNdpi,
    SlidePhilips,
    SlideSvs,
    SlideTiff,
    SlideTrestle,
]

FORMATS_MEMORY_BACKEND = [
    RgbBmp,
    RgbJpeg,
    RgbNpy,
    RgbPbm,
    RgbPng,
    RgbTga,
    RgbWebp,
]


success_tiff = SlideSvs()
failure_tiff = SlideError()


def test_tiff_init_str():
    backend = BrightfieldTiffBackend(success_tiff.path)
    for attr in [
        "path",
        "backend",
        "pyramidal",
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
        assert hasattr(backend, attr)


def test_tiff_init_pathlib():
    backend = BrightfieldTiffBackend(Path(success_tiff.path))
    for attr in [
        "path",
        "backend",
        "pyramidal",
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
        assert hasattr(backend, attr)


def test_tiff_validate_failure():
    with pytest.raises(ValueError):
        _ = BrightfieldTiffBackend(failure_tiff.path)


def test_tiff_read():
    backend = BrightfieldTiffBackend(success_tiff.path)
    assert hasattr(backend, "data")
    assert isinstance(backend.data, ZarrGroup)


def test_tiff_formats():
    for fmt in FORMATS_TIFF_BACKEND:
        path = Path(fmt.path)
        backend = BrightfieldTiffBackend(path)
        assert hasattr(backend, "data")
        assert isinstance(backend.data, (ZarrGroup, ZarrArray))
        assert backend.mpp_x == fmt.mpp_x
        assert backend.mpp_y == fmt.mpp_y
        if fmt.mpp_x is not None and fmt.mpp_y is not None:
            assert backend.mpp == (fmt.mpp_x + fmt.mpp_y) / 2.0
        if backend.levels_keys:
            assert backend.pyramidal
            assert backend.l == "0"
        else:
            assert not backend.pyramidal
            assert backend.l is None
        assert backend.w == fmt.width
        assert backend.h == fmt.height
        assert backend.c == fmt.channels
        assert backend[:100, :100].shape == (100, 100, fmt.channels)


def test_tiff_indexing():
    backend = BrightfieldTiffBackend(success_tiff.path, level="2")
    h, w, c = backend.h, backend.w, 3
    # Simplified check of shapes
    assert backend[0:5, 0:5, 0].shape == (5, 5)
    assert backend[0:2, 0:2, 0:c].shape == (2, 2, c)
    assert backend[...].shape == (h, w, c)
    assert backend[..., 0].shape == (h, w)
    assert backend[0, ..., 0].shape == (w,)
    assert backend[0].shape == (w, c)
    assert backend[0:10].shape[0] == 10


success_os = SlideSvs()
failure_os = SlideError()


def test_openslide_init_str():
    backend = BrightfieldOpenslideBackend(success_os.path)
    for attr in [
        "path",
        "backend",
        "pyramidal",
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
        assert hasattr(backend, attr)


def test_openslide_init_pathlib():
    backend = BrightfieldOpenslideBackend(Path(success_os.path))
    for attr in [
        "path",
        "backend",
        "pyramidal",
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
        assert hasattr(backend, attr)


def test_openslide_validate_failure():
    with pytest.raises(ValueError):
        _ = BrightfieldOpenslideBackend(failure_os.path)


def test_openslide_read():
    backend = BrightfieldOpenslideBackend(success_os.path)
    assert hasattr(backend, "data")
    assert isinstance(backend.data, openslide.OpenSlide)


def test_openslide_formats():
    for fmt in FORMATS_OPENSLIDE_BACKEND:
        path = Path(fmt.path)
        backend = BrightfieldOpenslideBackend(path)
        assert backend.mpp_x == fmt.mpp_x
        assert backend.mpp_y == fmt.mpp_y
        if fmt.mpp_x and fmt.mpp_y:
            assert backend.mpp == (fmt.mpp_x + fmt.mpp_y) / 2.0
        if backend.levels_keys:
            assert backend.pyramidal
            assert backend.l == 0
        else:
            assert not backend.pyramidal
            assert backend.l is None
        assert backend.w == fmt.width
        assert backend.h == fmt.height
        assert backend.c == fmt.channels
        assert backend[:100, :100].shape == (100, 100, fmt.channels)


def test_openslide_indexing():
    backend = BrightfieldOpenslideBackend(success_os.path, level=2)
    h, w, c = backend.h, backend.w, 3
    assert backend[0:5, 0:5, 0].shape == (5, 5)
    assert backend[0:2, 0:2, 0:c].shape == (2, 2, c)
    assert backend[...].shape == (h, w, c)
    assert backend[..., 0].shape == (h, w)
    assert backend[0, ..., 0].shape == (w,)
    assert backend[0].shape == (w, c)
    assert backend[0:10].shape[0] == 10


success_mem = RgbBmp()
failure_mem = SlideError()


def test_memory_init_str():
    backend = BrightfieldMemoryBackend(success_mem.path)
    for attr in [
        "path",
        "backend",
        "pyramidal",
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
        assert hasattr(backend, attr)


def test_memory_init_pathlib():
    backend = BrightfieldMemoryBackend(Path(success_mem.path))
    for attr in [
        "path",
        "backend",
        "pyramidal",
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
        assert hasattr(backend, attr)


def test_memory_validate_failure():
    with pytest.raises(ValueError):
        _ = BrightfieldMemoryBackend(failure_mem.path)


def test_memory_validate_level():
    with pytest.raises(ValueError):
        _ = BrightfieldMemoryBackend(success_mem.path, level=2)


def test_memory_read():
    backend = BrightfieldMemoryBackend(success_mem.path)
    assert hasattr(backend, "data")
    assert isinstance(backend.data, np.ndarray)


def test_memory_formats():
    for fmt in FORMATS_MEMORY_BACKEND:
        path = Path(fmt.path)
        backend = BrightfieldMemoryBackend(path)
        assert backend.mpp_x == fmt.mpp_x
        assert backend.mpp_y == fmt.mpp_y
        if fmt.mpp_x and fmt.mpp_y:
            assert backend.mpp == (fmt.mpp_x + fmt.mpp_y) / 2.0
        assert not backend.pyramidal
        assert backend.l is None
        assert backend.w == fmt.width
        assert backend.h == fmt.height
        assert backend.c == fmt.channels
        assert backend[:100, :100].shape == (100, 100, fmt.channels)


def test_memory_indexing():
    backend = BrightfieldMemoryBackend(success_mem.path)
    h, w, c = backend.h, backend.w, 3
    assert backend[0:5, 0:5, 0].shape == (5, 5)
    assert backend[0:2, 0:2, 0:c].shape == (2, 2, c)
    assert backend[...].shape == (h, w, c)
    assert backend[..., 0].shape == (h, w)
    assert backend[0, ..., 0].shape == (w,)
    assert backend[0].shape == (w, c)
    assert backend[0:10].shape[0] == 10
