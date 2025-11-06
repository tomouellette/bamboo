from dataclasses import dataclass

DATA: str = "src/tests/data/"


@dataclass(frozen=True)
class RgbBmp:
    path: str = DATA + "bmp/rgb.bmp"
    mpp_x: None | float = None
    mpp_y: None | float = None
    width: int = 621
    height: int = 621
    channels: int = 3


@dataclass(frozen=True)
class RgbJpeg:
    path: str = DATA + "jpeg/rgb.jpeg"
    mpp_x: None | float = None
    mpp_y: None | float = None
    width: int = 621
    height: int = 621
    channels: int = 3


@dataclass(frozen=True)
class RgbNpy:
    path: str = DATA + "npy/rgb.npy"
    mpp_x: None | float = None
    mpp_y: None | float = None
    width: int = 621
    height: int = 621
    channels: int = 3


@dataclass(frozen=True)
class RgbPbm:
    path: str = DATA + "pbm/rgb.pbm"
    mpp_x: None | float = None
    mpp_y: None | float = None
    width: int = 621
    height: int = 621
    channels: int = 3


@dataclass(frozen=True)
class RgbPng:
    path: str = DATA + "png/rgb.png"
    mpp_x: None | float = None
    mpp_y: None | float = None
    width: int = 621
    height: int = 621
    channels: int = 3


@dataclass(frozen=True)
class RgbTga:
    path: str = DATA + "tga/rgb.tga"
    mpp_x: None | float = None
    mpp_y: None | float = None
    width: int = 621
    height: int = 621
    channels: int = 3


@dataclass(frozen=True)
class RgbWebp:
    path: str = DATA + "webp/rgb.webp"
    mpp_x: None | float = None
    mpp_y: None | float = None
    width: int = 621
    height: int = 621
    channels: int = 3
