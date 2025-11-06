from dataclasses import dataclass

DATA: str = "src/tests/data/"


@dataclass(frozen=True)
class SlideBif:
    path: str = DATA + "bif/OS-1.bif"
    mpp_x: float = 0.23250000000000001
    mpp_y: float = 0.23250000000000001
    width: int = 105813
    height: int = 93951
    channels: int = 3


@dataclass(frozen=True)
class SlideNdpi:
    path: str = DATA + "ndpi/CMU-1.ndpi"
    mpp_x: float = 0.45641259698767683
    mpp_y: float = 0.45506257110352671
    width: int = 51200
    height: int = 38144
    channels: int = 3


@dataclass(frozen=True)
class SlidePhilips:
    path: str = DATA + "philips/Philips-1.tiff"
    mpp_x: float = 0.226907
    mpp_y: float = 0.22689100000000001
    width: int = 45056
    height: int = 35840
    channels: int = 3


@dataclass(frozen=True)
class SlideLeica:
    path: str = DATA + "scn/Leica-1.scn"
    mpp_x: float = 0.5
    mpp_y: float = 0.5
    width: int = 36832
    height: int = 38432
    channels: int = 3


@dataclass(frozen=True)
class SlideSvs:
    path: str = DATA + "svs/CMU-1.svs"
    mpp_x: float = 0.499
    mpp_y: float = 0.499
    width: int = 46000
    height: int = 32914
    channels: int = 3


@dataclass(frozen=True)
class SlideTiff:
    path: str = DATA + "tiff/CMU-1.tiff"
    mpp_x: None | float = None
    mpp_y: None | float = None
    width: int = 46000
    height: int = 32914
    channels: int = 3


@dataclass(frozen=True)
class SlideTrestle:
    path: str = DATA + "trestle/CMU-1.tif"
    mpp_x: float = 0.57469189167022705
    mpp_y: float = 0.57506245374679565
    width: int = 40000
    height: int = 27712
    channels: int = 3


@dataclass(frozen=True)
class SlideOmeTiff:
    path: str = DATA + "xenium/image.ome.tif"
    mpp_x: float = 0.2737653984781153
    mpp_y: float = 0.2737746952056361
    width: int = 14896
    height: int = 27502
    channels: int = 3


@dataclass(frozen=True)
class SlideError:
    path: str = DATA + "README.md"
    mpp_x: float = None
    mpp_y: float = None
    width: int = 0
    height: int = 0
    channels: int = 0
