# Copyright (c) 2025, Tom Ouellette
# Licensed under the GNU GPLv3 License

import tifffile
import importlib.util
import numpy as np
import re
import xml.etree.ElementTree as ET
import zarr

from abc import ABC, abstractmethod
from bamboo.types import IndexKey, Numeric
from dataclasses import dataclass, field
from numpy.typing import NDArray
from pathlib import Path
from PIL import Image
from tifffile import TiffFile
from typing import Self, Final, Set


@dataclass(frozen=True)
class BrightfieldBackendConfig:
    name: str
    formats: Set[str] = field(default_factory=set)

    def supports_format(self, path: Path) -> bool:
        return path.suffix.lower() in self.formats


BRIGHTFIELD_MEMORY_CONFIG = BrightfieldBackendConfig(
    name="memory",
    formats={
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
    },
)

BRIGHTFIELD_TIFF_CONFIG = BrightfieldBackendConfig(
    name="tifffile", formats={".ndpi", ".scn", ".svs", ".tif", ".tiff", ".ome.tiff"}
)

BRIGHTFIELD_OPENSLIDE_CONFIG = BrightfieldBackendConfig(
    name="openslide",
    formats={
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
    },
)


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
    ValueError
        If `path` is not a string or pathlib.Path.
    FileNotFoundError
        If the specified path does not exist.
    OSError
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
    mpp_x : float | None
        Micron per pixel information across x-orientation.
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
        if not isinstance(path, (str, Path)):
            raise ValueError("'path' must be a str or pathlib.Path.")

        if not (path := Path(path)).exists():
            raise FileNotFoundError("'path' does not exist.")

        if path.is_dir():
            raise OSError("'path' must be a file not a directory.")

        self.path = path
        self.backend: str = None
        self.pyramidal: bool | None = None
        self.dtype: Numeric | None = None

        self.levels: int | None = None
        self.levels_keys: list[str | int] = None
        self.levels_shape: dict[str | int, tuple[int, int, int]] = None

        self.mpp_x: float | None = mpp
        self.mpp_y: float | None = mpp

        self.l: str | None = level
        self.h: int | None = None
        self.w: int | None = None
        self.c: int | None = None

        self._validate()
        if mpp is None:
            self._resolution()

        self._read()

    @property
    def mpp(self) -> float | None:
        """Set average microns per pixel."""
        if self.mpp_x is not None and self.mpp_y is not None:
            return (self.mpp_x + self.mpp_y) / 2.0
        return None

    @abstractmethod
    def _resolution(self):
        """Extract image resolution metadata."""
        pass

    @abstractmethod
    def _validate(self):
        """Validate input image format and data type."""
        pass

    @abstractmethod
    def _read(self):
        """Read pixel data from input image."""
        pass


class BrightfieldMemoryBackend(BrightfieldBackend):
    """Brightfield image backend using in-memory `Pillow` reader.

    Parameters
    ----------
    path : str or Path
        Path to the input image file.
    mpp : None | float
        User-provided microns per pixel.
    """

    config: BrightfieldBackendConfig = BRIGHTFIELD_MEMORY_CONFIG

    def _resolution(self) -> None:
        self.mpp_x = self.mpp
        self.mpp_y = self.mpp

    def _validate(self) -> None:
        """Validate input image format.

        Raises
        ------
        ValueError
            If the file does not have a supported format extension.
        """
        if self.l is not None:
            raise ValueError("Level cannot be set when using BrightfieldMemoryBackend.")

        if not self.config.supports_format(self.path):
            raise ValueError(
                "Invalid image format. " + f"Must be one of {', '.join(self.formats)}"
            )

    def _read(self) -> None:
        """Read pixels from validated input image data with `PIL`."""
        self.pyramidal = False

        if self.path.suffix == ".npy":
            data = np.load(self.path)
        else:
            data = np.array(Image.open(self.path).convert("RGB"))

        if data.ndim != 3:
            raise ValueError(
                "BrightfieldMemoryBackend only supports 3-dimensional inputs."
            )

        self.l = None
        self.h, self.w, self.c = data.shape

        if self.c != 3:
            raise ValueError(
                "BrightfieldMemoryBackend is meant for 3-channel RGB images."
            )

        self.data = data

    def __getitem__(self, key: IndexKey) -> Numeric | NDArray[Numeric]:
        """Direct indexing of the underlying pixel data.

        Parameters
        ----------
        key : IndexKey
            A slice, ellipsis, or any valid index into numpy ndarray.

        Returns
        -------
        Numeric | NDArray[Numeric]
            Subregion of brightfield image.
        """
        return self.data[key]


class BrightfieldTiffBackend(BrightfieldBackend):
    """Brightfield image backend using `tifffile` for zarr-backed TIFF formats.

    Parameters
    ----------
    path : str or Path
        Path to the input image file.
    mpp : None | float
        User-provided microns per pixel.
    level : None | int | str
        User-provided level to read.
    """

    config: BrightfieldBackendConfig = BRIGHTFIELD_TIFF_CONFIG

    def _validate(self) -> None:
        """Validate input image format.

        Raises
        ------
        ValueError
            If the file does not have a supported format extension.
        ValueError
            If the file specifies a trestle TIFF.
        ValueError
            If the file species a non-RGB or non-YCBCR photometric image.
        """
        self.backend = "tifffile"
        if not self.config.supports_format(self.path):
            raise ValueError(
                "Invalid image format. " + f"Must be one of {', '.join(self.formats)}"
            )

        with tifffile.TiffFile(self.path) as tiff:
            if copyright_tag := tiff.pages[0].tags.get(33432):
                if "trestle" in copyright_tag._str().lower():
                    # Note: We do not support trestle `tif` files in the
                    # BrightfieldTiffBackend as tiles overlap/aren't stitched
                    # together properly.
                    raise ValueError(
                        "'BrightfieldTiffBackend' does not support trestle "
                        + "`.tif` images."
                    )

            if tiff.pages[0].photometric not in [
                tifffile.PHOTOMETRIC.RGB,
                tifffile.PHOTOMETRIC.YCBCR,
            ]:
                raise ValueError(
                    "'BrightfieldTiffBackend' expects RGB and YCBCR images."
                )

    def _resolution(self) -> None:
        """Extract resolution info from `tifffile` supported formats."""
        with tifffile.TiffFile(self.path) as tiff:
            self.dtype = tiff.pages[0].dtype

            # TO-DO: It may be beneficial to add warnings or errors if an
            # extension is present but the extension is not validated in the
            # parsed header metadata.
            match self.path.suffix.lower():
                case ".ndpi":
                    if tiff.is_ndpi:
                        self._resolution_tags(tiff)
                case ".tiff" | ".tif":
                    if tiff.is_ome:
                        self._resolution_ome_tiff(tiff)
                    if tiff.is_philips:
                        self._resolution_philips(tiff)
                case ".scn":
                    if tiff.is_scn:
                        self._resolution_tags(tiff)
                case ".svs":
                    self._resolution_svs(tiff)

    def _resolution_ome_tiff(self, tiff: TiffFile) -> None:
        """Extract resolution info from `ome.tiff` images."""
        xml_str = tiff.pages[0].description
        root = ET.fromstring(xml_str)
        pixels = root.find(".//{*}Pixels")
        if pixels is not None:
            x = float(pixels.get("PhysicalSizeX", 0))
            y = float(pixels.get("PhysicalSizeY", 0))
            x_unit = pixels.get("PhysicalSizeXUnit", "")
            y_unit = pixels.get("PhysicalSizeYUnit", "")

            if x_unit in ["µm", "cm"] and y_unit in ["µm", "cm"]:
                if x_unit == "cm":
                    x = 1e4 / x

                if y_unit == "cm":
                    y = 1e4 / y

                self.mpp_x = x
                self.mpp_y = y

    def _resolution_philips(self, tiff: TiffFile) -> None:
        """Extract resolution info from philips `.tiff` images."""
        xml_str = tiff.pages[0].description
        root = ET.fromstring(xml_str)
        mmpp = root.find(".//Attribute[@Name='DICOM_PIXEL_SPACING']").text
        if mmpp is not None:
            mmpp_y, mmpp_x = map(float, mmpp.strip('"').split('" "'))
            self.mpp_x = 1000 * mmpp_x
            self.mpp_y = 1000 * mmpp_y

    def _resolution_svs(self, tiff: TiffFile) -> None:
        """Extract resolution metadata from `.svs` images."""
        meta_tag = tiff.pages[0].description
        if match := re.search(r"MPP\s*=\s*([0-9.]+)", meta_tag):
            self.mpp_x = self.mpp_y = float(match.group(1))

    def _resolution_tags(self, tiff: TiffFile) -> None:
        """Extract resolution metadata from `.scn` and `.ndpi` images."""
        page = max(tiff.pages, key=lambda p: p.shape)
        x = page.tags.get("XResolution")
        y = page.tags.get("YResolution")

        self.mpp_x = 1e4 / x.value[0] if x else None
        self.mpp_y = 1e4 / y.value[0] if y else None

    def _read(self) -> None:
        """Read pixels from validated input image data with `tifffile`."""
        if self.path.suffix == ".scn":
            # Note: `.scn` images store a macro image in the first series so we
            # manually set the series here. We may want to build all the format
            # specific readers as methods for clarity.
            store = tifffile.imread(self.path, mode="r", aszarr=True, series=1)
        else:
            store = tifffile.imread(self.path, mode="r", aszarr=True)

        self.pyramidal = store.is_multiscales
        self.data = zarr.open(store, mode="r")

        if self.pyramidal:
            self.levels_keys = []
            self.levels_shape = {}
            for key in sorted(self.data.keys()):
                self.levels_keys.append(key)
                self.levels_shape[key] = self.data[key].shape

            if self.l is None:
                self.l = "0"

            if self.l not in self.levels_keys:
                raise ValueError(
                    f"The provided level ({self.l}) was not found. "
                    + f"Available levels include: {self.levels_keys}"
                )

            self.h, self.w, self.c = self.data[self.l].shape
        else:
            self.h, self.w, self.c = self.data.shape

    def __getitem__(self, key: IndexKey) -> Numeric | NDArray[Numeric]:
        """Direct indexing of the underlying pixel data.

        Parameters
        ----------
        key : IndexKey
            A slice, ellipsis, or any valid index into numpy ndarray.

        Returns
        -------
        Numeric | NDArray[Numeric]
            Subregion of brightfield image.
        """
        if self.pyramidal:
            result = self.data[self.l][key]
        else:
            result = self.data[key]

        if result.size == 1:
            # Note: Single values indexed from `zarr` arrays are returned
            # as arrays. Therefore, we manually check the final size and
            # return scalar values to maintain numpy-like indexing.
            return result.item()

        return np.array(result)


class BrightfieldOpenslideBackend(BrightfieldBackend):
    """Brightfield image backend using `openslide-python` for certain formats.

    Parameters
    ----------
    path : str or Path
        Path to the input image file.
    mpp : None | float
        User-provided microns per pixel.
    """

    config: BrightfieldBackendConfig = BRIGHTFIELD_OPENSLIDE_CONFIG

    def _validate(self) -> None:
        """Validate input image format and `openslide` installation.

        Raises
        ------
        ModuleNotFoundError
            If `openslide` installation is not found.
        ValueError
            If the file does not have a supported format extension.
        """
        self.backend = "openslide"
        if importlib.util.find_spec(self.backend) is None:
            raise ModuleNotFoundError(
                f"'{self.backend}' was not found. Please install to work "
                + f"with {', '.join(self.formats)} formats."
            )

        if not self.config.supports_format(self.path):
            raise ValueError(
                "Invalid image format. " + f"Must be one of {', '.join(self.formats)}"
            )

    def _resolution(self) -> None:
        """Extract resolution info from `openslide` supported formats."""
        import openslide

        with openslide.OpenSlide(self.path) as openslide:
            if "openslide.mpp-x" in openslide.properties:
                self.mpp_x = float(openslide.properties["openslide.mpp-x"])

            if "openslide.mpp-y" in openslide.properties:
                self.mpp_y = float(openslide.properties["openslide.mpp-y"])

            # Note: OpenSlide will fill mpp metadata with large values in some
            # cases, like generic tiffs. For now, we just check for nonsensical
            # values and convert mpp back to the empty placeholder. Most slides
            # are read at 20x/40x magnification so things above 10 are highly
            # unlikely since most if not all slides are imaged at mpp < 1.
            if self.mpp_x > 10.0:
                self.mpp_x = None

            if self.mpp_y > 10.0:
                self.mpp_y = None

    def _read(self) -> None:
        """Read pixels from validated input image data with `openslide`."""
        import openslide

        self.data = openslide.OpenSlide(self.path)

        # Note: Openslide only supports 8-bit RGB/A slides.
        self.dtype = np.uint8
        self.pyramidal = True
        self.c = 3

        self.levels = self.data.level_count
        self.levels_keys = list(range(self.levels))

        # Note: Openslide returns (width, height) so we swap the dimensions
        # here to maintain consistency across different brightfield backends.
        self.levels_shape = {
            k: (v[1], v[0], 3)
            for k, v in zip(self.levels_keys, self.data.level_dimensions)
        }

        if self.l is None:
            self.l = 0

        if self.l not in self.levels_keys:
            raise ValueError(
                f"The provided level ({self.l}) was not found. "
                + f"Available levels include: {self.levels_keys}"
            )

        self.h, self.w, _ = self.levels_shape[self.l]

    def __getitem__(self, key: IndexKey) -> Numeric | NDArray[Numeric]:
        """Direct indexing of the underlying pixel data.

        Parameters
        ----------
        key : IndexKey
            A slice, ellipsis, or any valid index into numpy ndarray.

        Returns
        -------
        Numeric | NDArray[Numeric]
            Subregion of brightfield image.
        """

        def _expand_ellipsis(key, ndim=3):
            if not isinstance(key, tuple):
                key = (key,)
            if Ellipsis not in key:
                return key + (slice(None),) * (ndim - len(key))
            n_ellipsis = key.count(Ellipsis)
            if n_ellipsis > 1:
                raise IndexError("An index can only have a single ellipsis ('...')")
            idx = key.index(Ellipsis)
            num_missing = ndim - (len(key) - 1)
            if num_missing < 0:
                raise IndexError("Too many indices for image dimensions")
            return key[:idx] + (slice(None),) * num_missing + key[idx + 1 :]

        def _normalize_index(k, max_val):
            if isinstance(k, int):
                if k < 0:
                    k += max_val
                if not (0 <= k < max_val):
                    raise IndexError(
                        f"Index {k} out of bounds for dimension size {max_val}"
                    )
                return slice(k, k + 1), 1
            elif isinstance(k, slice):
                start, stop, step = k.indices(max_val)
                if step == 0:
                    raise ValueError("Slice step cannot be zero")
                length = max(0, (stop - start + (step - 1)) // step)
                return slice(start, stop, step), length
            else:
                raise TypeError("Index must be int or slice")

        key = _expand_ellipsis(key, ndim=3)

        if len(key) > 3:
            raise IndexError(
                "Indexing must be in the form [y], [y, x], or [y, x, channel]"
            )

        while len(key) < 3:
            key = key + (None,)

        y_key, x_key, channel = key

        y_slice, y_len = _normalize_index(y_key, self.h)
        x_slice, x_len = _normalize_index(x_key, self.w)

        y_start, y_stop, y_step = y_slice.start, y_slice.stop, y_slice.step
        x_start, x_stop, x_step = x_slice.start, x_slice.stop, x_slice.step

        region_width = x_stop - x_start
        region_height = y_stop - y_start

        if region_width <= 0 or region_height <= 0:
            raise IndexError("Empty or invalid slice requested")

        region = self.data.read_region(
            (x_start, y_start), self.l, (region_width, region_height)
        )
        region = region.convert("RGB")
        region = np.array(region)

        region = region[::y_step, ::x_step]
        if channel is not None:
            region = region[..., channel]

        return region.item() if region.size == 1 else np.squeeze(region)
