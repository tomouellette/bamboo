# Copyright (c) 2025, Tom Ouellette
# Licensed under the GNU GPLv3 License

import numpy as np
import zarr

from typing import Tuple, Any
from types import EllipsisType

# A general numeric type for integer and floating point types
Numeric = int | float | np.integer[Any] | np.floating[Any]

# Type alias for a single element of an indexing key
IndexElement = int | slice | EllipsisType | None

# Type alias for single indices, slices, Ellipsis, and associated tuples.
IndexKey = IndexElement | Tuple[IndexElement, ...]

# Valid output array types
Array = np.ndarray | np.memmap | zarr.Array
