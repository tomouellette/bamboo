# Copyright (c) 2025, Tom Ouellette
# Licensed under the GNU GPLv3 License

import tifffile
import importlib.util
import numpy as np
import re
import xml.etree.ElementTree as ET
import warnings
import zarr

from abc import ABC, abstractmethod
from bamboo.types import IndexKey, Numeric
from dataclasses import dataclass, field
from functools import cached_property
from numpy.typing import NDArray
from pathlib import Path
from PIL import Image
from tifffile import TiffFile
from typing import Self, Final, Set


@dataclass(frozen=True)
class BrightfieldBackendConfig:
    """Configuration for a Brightfield image backend.

    Attributes
    ----------
    name : str
        Backend name, e.g., 'memory', 'tifffile', 'openslide'.
    formats : Set[str]
        Set of file extensions (including leading dot) supported by this backend.
    """

    name: str
    formats: Set[str] = field(default_factory=set)

    def supports_format(self, path: Path | str) -> bool:
        """Check if this backend supports the file at the given path.

        Parameters
        ----------
        path : Path | str
            Path to the file, or file suffix string (e.g., '.tiff').

        Returns
        -------
        bool
            True if backend supports this format, False otherwise.
        """
        suffix = path.suffix.lower() if isinstance(path, Path) else path.lower()
        return suffix in self.formats

    @property
    def supported_extensions(self) -> str:
        """Return a comma-separated string of supported file extensions."""
        return ", ".join(sorted(self.formats))


BRIGHTFIELD_MEMORY_CONFIG: Final = BrightfieldBackendConfig(
    name="memory",
    formats=set(
        sorted(
            [
                ".apng",
                ".blp",
                ".bmp",
                ".bufr",
                ".bw",
                ".cur",
                ".dcx",
                ".dds",
                ".dib",
                ".emf",
                ".eps",
                ".fit",
                ".fits",
                ".flc",
                ".fli",
                ".ftc",
                ".ftu",
                ".gbr",
                ".gif",
                ".grib",
                ".h5",
                ".hdf",
                ".icb",
                ".icns",
                ".ico",
                ".iim",
                ".im",
                ".j2c",
                ".j2k",
                ".jfif",
                ".jp2",
                ".jpc",
                ".jpe",
                ".jpeg",
                ".jpf",
                ".jpg",
                ".jpx",
                ".mpeg",
                ".mpg",
                ".msp",
                ".pbm",
                ".pcd",
                ".pcx",
                ".pfm",
                ".pgm",
                ".png",
                ".pnm",
                ".ppm",
                ".ps",
                ".psd",
                ".pxr",
                ".qoi",
                ".ras",
                ".rgb",
                ".rgba",
                ".sgi",
                ".tga",
                ".tif",
                ".tiff",
                ".vda",
                ".vst",
                ".webp",
                ".wmf",
                ".xbm",
                ".xpm",
                ".npy",
            ]
        )
    ),
)

BRIGHTFIELD_TIFF_CONFIG: Final = BrightfieldBackendConfig(
    name="tifffile",
    formats=set(sorted([".ndpi", ".scn", ".svs", ".tif", ".tiff", ".ome.tiff"])),
)

BRIGHTFIELD_OPENSLIDE_CONFIG: Final = BrightfieldBackendConfig(
    name="openslide",
    formats=set(
        sorted(
            [
                ".bif",
                ".czi",
                ".mrxs",
                ".ndpi",
                ".scn",
                ".svs",
                ".svslide",
                ".tif",
                ".tiff",
                ".vms",
                ".vmu",
            ]
        )
    ),
)

BRIGHTFIELD_BACKENDS: Final[dict[str, BrightfieldBackendConfig]] = {
    "memory": BRIGHTFIELD_MEMORY_CONFIG,
    "tifffile": BRIGHTFIELD_TIFF_CONFIG,
    "openslide": BRIGHTFIELD_OPENSLIDE_CONFIG,
}


class BrightfieldBackend(ABC):
    """Base class for various histology image backends.

    Parameters
    ----------
    path : str or Path
        Path to the input image file.
    mpp : None | float
        User-provided microns per pixel.
    level : None | int | str
        User-provided level to read.

    Raises
    ------
    FileNotFoundError
        If the specified path does not exist.
    IsADirectoryError
        If the path is a directory instead of a file.

    Attributes
    ----------
    path : Path
        Path to the input image file.
    backend : str
        Name of the backend used for reading.
    pyramidal : bool | None
        Whether the image is in pyramidal format.
    levels : int | None
        Number of image pyramid levels.
    levels_keys : list[str | int]
        If pyramidal, key/integer for indexing a specific level.
    levels_shape : dict[str, tuple[int, int, int]]
        If pyramidal, height, width, and channels for each level.
    mpp : float | None
        Micron per pixel information.
    mpp_x : float | None
        Micron per pixel information across x-axis.
    mpp_y : float | None
        Micron per pixel information across y-axis.
    l : int | None
        Current level.
    h : int | None
        Image height at the current level.
    w : int | None
        Image width at the current level.
    c : int | None
        Number of channels in the image.
    """

    formats: Final[list[str]] = []
    """Backend-specific supported image formats."""

    def __init__(
        self, path: str | Path, mpp: None | float = None, level: None | int | str = None
    ) -> Self:
        self.path = Path(path)

        if not self.path.exists():
            raise FileNotFoundError(f"Path does not exist: {self.path}")

        if self.path.is_dir():
            raise IsADirectoryError(
                f"Path must be a file, not a directory: {self.path}"
            )

        self.backend: str = None
        self.pyramidal: bool | None = None
        self.dtype: Numeric | None = None

        self.levels: int | None = None
        self.levels_keys: list[str | int] = None
        self.levels_shape: dict[str | int, tuple[int, int, int]] = None

        self.mpp_x: float | None = mpp
        self.mpp_y: float | None = mpp

        self.l: str | int | None = level
        self.h: int | None = None
        self.w: int | None = None
        self.c: int | None = None

        self.data = None
        self._closed = False

        try:
            self._validate()
            if mpp is None:
                self._resolution()
            self._read()
        except Exception:
            self.close()
            raise

    @cached_property
    def mpp(self) -> float | None:
        """Average microns per pixel."""
        if self.mpp_x is not None and self.mpp_y is not None:
            return (self.mpp_x + self.mpp_y) / 2.0
        return None

    @property
    def shape(self) -> tuple[int, int, int]:
        """Shape of current level as (height, width, channels)."""
        return (self.h, self.w, self.c)

    @property
    def closed(self) -> bool:
        """Whether the backend has been closed."""
        return self._closed

    @abstractmethod
    def _validate(self) -> None:
        """Validate input image format and data type."""
        pass

    @abstractmethod
    def _resolution(self) -> None:
        """Extract image resolution metadata."""
        pass

    @abstractmethod
    def _read(self) -> None:
        """Read pixel data from input image."""
        pass

    @abstractmethod
    def __getitem__(self, key: IndexKey) -> Numeric | NDArray[Numeric]:
        """Index into the image data.

        Parameters
        ----------
        key : IndexKey
            Numpy-style index.

        Returns
        -------
        Numeric | NDArray[Numeric]
            Pixel value or array slice.
        """
        pass

    def close(self) -> None:
        """Release resources held by the backend."""
        if self._closed:
            return
        if self.data is not None and hasattr(self.data, "close"):
            try:
                self.data.close()
            except Exception:
                pass
        self._closed = True

    def __enter__(self) -> Self:
        """Enter context manager."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Exit context manager."""
        self.close()

    def __del__(self) -> None:
        """Cleanup on deletion."""
        try:
            self.close()
        except Exception:
            pass

    def __repr__(self) -> str:
        """String representation."""
        status = "closed" if self._closed else "open"
        if self.h is not None:
            shape_str = f"{self.h}×{self.w}×{self.c}"
        else:
            shape_str = "uninitialized"
        return (
            f"{self.__class__.__name__}("
            f"path={self.path.name!r}, "
            f"shape={shape_str}, "
            f"mpp={self.mpp}, "
            f"{status})"
        )


class BrightfieldMemoryBackend(BrightfieldBackend):
    """Brightfield image backend using in-memory NumPy or Pillow loading.

    Notes
    -----
    - No pyramidal support
    - Always exposes an (H, W, 3) RGB array
    - Indexing semantics exactly match NumPy ndarray behavior
    - Returned arrays may be views; data is read-only
    """

    config: BrightfieldBackendConfig = BRIGHTFIELD_MEMORY_CONFIG

    def _validate(self) -> None:
        if self.l is not None:
            raise ValueError(
                "Level selection is not supported for BrightfieldMemoryBackend."
            )

        if not self.config.supports_format(self.path):
            supported = ", ".join(sorted(self.config.formats))
            raise ValueError(
                f"Unsupported format '{self.path.suffix}'. Supported formats: {supported}"
            )

    def _resolution(self) -> None:
        if self.mpp_x is None:
            warnings.warn(
                f"No resolution metadata found in {self.path.suffix}. "
                "Microns per pixel (mpp) should be provided explicitly.",
                UserWarning,
                stacklevel=2,
            )

    def _read(self) -> None:
        self.backend = "memory"
        self.pyramidal = False
        self.l = None

        if self.path.suffix == ".npy":
            data = np.load(self.path, mmap_mode="r")
            self.dtype = data.dtype
        else:
            with Image.open(self.path) as img:
                img = img.convert("RGB")
                data = np.asarray(img)
            self.dtype = data.dtype

        if data.ndim != 3 or data.shape[2] != 3:
            raise ValueError(
                f"Expected RGB image with shape (H, W, 3); got {data.shape}"
            )

        self.h, self.w, self.c = data.shape

        # Enforce immutability to match read-only backend semantics
        data.setflags(write=False)
        self.data = data

    def __getitem__(self, key: IndexKey) -> Numeric | NDArray[Numeric]:
        if self._closed:
            raise ValueError("Cannot index a closed backend")

        result = self.data[key]
        return result.item() if np.ndim(result) == 0 else result

    def close(self) -> None:
        if self._closed:
            return
        self.data = None
        self._closed = True


class BrightfieldTiffBackend(BrightfieldBackend):
    """Brightfield image backend using `tifffile` with zarr-backed access."""

    config: BrightfieldBackendConfig = BRIGHTFIELD_TIFF_CONFIG

    def _validate(self) -> None:
        self.backend = "tifffile"

        if not self.config.supports_format(self.path):
            raise ValueError(
                f"Unsupported format '{self.path.suffix}'. "
                f"Supported formats: {', '.join(sorted(self.config.formats))}"
            )

        with tifffile.TiffFile(self.path) as tiff:
            if copyright_tag := tiff.pages[0].tags.get(33432):
                if "trestle" in copyright_tag._str().lower():
                    raise ValueError(
                        "Trestle TIFFs are not supported (overlapping tiles)."
                    )

            if tiff.pages[0].photometric not in (
                tifffile.PHOTOMETRIC.RGB,
                tifffile.PHOTOMETRIC.YCBCR,
            ):
                raise ValueError("BrightfieldTiffBackend expects RGB or YCBCR images.")

    def _resolution(self) -> None:
        with tifffile.TiffFile(self.path) as tiff:
            self.dtype = tiff.pages[0].dtype

            suffix = self.path.suffix.lower()
            if suffix == ".ndpi" and tiff.is_ndpi:
                self._resolution_tags(tiff)
            elif suffix in {".tif", ".tiff"}:
                if tiff.is_ome:
                    self._resolution_ome_tiff(tiff)
                elif tiff.is_philips:
                    self._resolution_philips(tiff)
            elif suffix == ".scn" and tiff.is_scn:
                self._resolution_tags(tiff)
            elif suffix == ".svs":
                self._resolution_svs(tiff)

    def _resolution_ome_tiff(self, tiff: TiffFile) -> None:
        xml_str = tiff.pages[0].description
        root = ET.fromstring(xml_str)
        pixels = root.find(".//{*}Pixels")
        if pixels is not None:
            x = float(pixels.get("PhysicalSizeX", 0))
            y = float(pixels.get("PhysicalSizeY", 0))
            x_unit = pixels.get("PhysicalSizeXUnit", "")
            y_unit = pixels.get("PhysicalSizeYUnit", "")

            if x_unit in {"µm", "cm"} and y_unit in {"µm", "cm"}:
                if x_unit == "cm":
                    x = 1e4 / x
                if y_unit == "cm":
                    y = 1e4 / y
                self.mpp_x = x
                self.mpp_y = y

    def _resolution_philips(self, tiff: TiffFile) -> None:
        xml_str = tiff.pages[0].description
        root = ET.fromstring(xml_str)
        mmpp = root.find(".//Attribute[@Name='DICOM_PIXEL_SPACING']").text
        if mmpp:
            mmpp_y, mmpp_x = map(float, mmpp.strip('"').split('" "'))
            self.mpp_x = 1000 * mmpp_x
            self.mpp_y = 1000 * mmpp_y

    def _resolution_svs(self, tiff: TiffFile) -> None:
        meta = tiff.pages[0].description
        if match := re.search(r"MPP\s*=\s*([0-9.]+)", meta):
            self.mpp_x = self.mpp_y = float(match.group(1))

    def _resolution_tags(self, tiff: TiffFile) -> None:
        page = max(tiff.pages, key=lambda p: p.shape)
        x = page.tags.get("XResolution")
        y = page.tags.get("YResolution")

        self.mpp_x = 1e4 / x.value[0] if x else None
        self.mpp_y = 1e4 / y.value[0] if y else None

    def _read(self) -> None:
        if self.path.suffix.lower() == ".scn":
            store = tifffile.imread(self.path, mode="r", aszarr=True, series=1)
        else:
            store = tifffile.imread(self.path, mode="r", aszarr=True)

        self.data = zarr.open(store, mode="r")
        self.pyramidal = store.is_multiscales

        if self.pyramidal:
            # Normalize levels to integers
            self._level_map: dict[int, str] = {}
            self.levels_shape = {}

            for i, key in enumerate(sorted(self.data.keys(), key=int)):
                self._level_map[i] = key
                self.levels_shape[i] = self.data[key].shape

            self.levels = len(self._level_map)
            self.levels_keys = list(self._level_map.keys())

            if self.l is None:
                self.l = 0

            if self.l not in self._level_map:
                raise ValueError(
                    f"Invalid level {self.l}. Available levels: {self.levels_keys}"
                )

            self.h, self.w, self.c = self.levels_shape[self.l]

        else:
            self.l = None
            self.h, self.w, self.c = self.data.shape

    def __getitem__(self, key: IndexKey) -> Numeric | NDArray[Numeric]:
        if self._closed:
            raise ValueError("Cannot index a closed backend")

        if self.pyramidal:
            arr = self.data[self._level_map[self.l]][key]
        else:
            arr = self.data[key]

        return arr.item() if np.ndim(arr) == 0 else np.asarray(arr)


class BrightfieldOpenslideBackend(BrightfieldBackend):
    """Brightfield image backend using `openslide`.

    Notes
    -----
    - Indexing supports integers, slices, and ellipsis
    - Fancy indexing and boolean masks are NOT supported
    - All reads are eager and materialized into NumPy arrays
    """

    config: BrightfieldBackendConfig = BRIGHTFIELD_OPENSLIDE_CONFIG

    def _validate(self) -> None:
        self.backend = "openslide"

        if importlib.util.find_spec("openslide") is None:
            raise ModuleNotFoundError("'openslide' is required for this backend.")

        if not self.config.supports_format(self.path):
            raise ValueError(
                f"Unsupported format '{self.path.suffix}'. "
                f"Supported formats: {', '.join(sorted(self.config.formats))}"
            )

    def _resolution(self) -> None:
        import openslide

        with openslide.OpenSlide(self.path) as slide:
            self.mpp_x = float(slide.properties.get("openslide.mpp-x", 0)) or None
            self.mpp_y = float(slide.properties.get("openslide.mpp-y", 0)) or None

            if self.mpp_x and self.mpp_x > 10:
                self.mpp_x = None
            if self.mpp_y and self.mpp_y > 10:
                self.mpp_y = None

    def _read(self) -> None:
        import openslide

        self.data = openslide.OpenSlide(self.path)
        self.dtype = np.uint8
        self.pyramidal = True
        self.c = 3

        self.levels = self.data.level_count
        self.levels_keys = list(range(self.levels))
        self.levels_shape = {
            i: (h, w, 3) for i, (w, h) in enumerate(self.data.level_dimensions)
        }

        if self.l is None:
            self.l = 0

        if self.l not in self.levels_keys:
            raise ValueError(
                f"Invalid level {self.l}. Available levels: {self.levels_keys}"
            )

        self.h, self.w, _ = self.levels_shape[self.l]

    def __getitem__(self, key: IndexKey) -> Numeric | NDArray[Numeric]:
        if self._closed:
            raise ValueError("Cannot index a closed backend")

        key = self._normalize_key(key)
        yk, xk, ck = key

        y_slice = self._to_slice(yk, self.h)
        x_slice = self._to_slice(xk, self.w)

        y0, y1 = y_slice.start, y_slice.stop
        x0, x1 = x_slice.start, x_slice.stop

        if y1 <= y0 or x1 <= x0:
            raise IndexError("Empty slice requested")

        region = self.data.read_region(
            (x0, y0),
            self.l,
            (x1 - x0, y1 - y0),
        ).convert("RGB")

        arr = np.asarray(region)
        arr = arr[:: y_slice.step, :: x_slice.step]

        # Channel indexing
        if ck is not None:
            arr = arr[..., ck]

        # Drop spatial axes indexed by integers (numpy semantics)
        if isinstance(yk, int):
            arr = np.squeeze(arr, axis=0)

        if isinstance(xk, int):
            arr = np.squeeze(arr, axis=0)

        return arr.item() if arr.ndim == 0 else arr

    @staticmethod
    def _normalize_key(key: IndexKey) -> tuple:
        if not isinstance(key, tuple):
            key = (key,)

        if Ellipsis in key:
            idx = key.index(Ellipsis)
            key = key[:idx] + (slice(None),) * (3 - len(key) + 1) + key[idx + 1 :]

        if len(key) > 3:
            raise IndexError("Too many indices for image")

        return key + (None,) * (3 - len(key))

    @staticmethod
    def _to_slice(k, size: int) -> slice:
        if k is None:
            return slice(0, size, 1)

        if isinstance(k, int):
            if k < 0:
                k += size
            if not 0 <= k < size:
                raise IndexError(f"Index {k} out of bounds for size {size}")
            return slice(k, k + 1, 1)

        if isinstance(k, slice):
            return slice(*k.indices(size))

        raise TypeError("Only int and slice indexing is supported")
