# Copyright (c) 2025, Tom Ouellette
# Licensed under the GNU GPLv3 License

from anci import base


@base("brightfield")
def brightfield():
    """Processing and analysis of gigapixel brightfield images."""
    pass


from .nn import brightfield_nn_semantic
from .tile import brightfield_tile_extract

__all__ = ["brightfield_semantic"]
