# Copyright (c) 2025, Tom Ouellette
# Licensed under the GNU GPLv3 License

import os
import numpy as np
import shutil
import zarr

from abc import abstractmethod, ABC
from functools import wraps
from multiprocessing import shared_memory
from pathlib import Path
from typing import Callable


class ArrayWriter(ABC):
    """A base class for array writers.

    Any array writer may have shape and data type set at initialization or
    set dynamically after initialization. The post-initialization setters
    are used in cases where the shape and data type of the output array is
    hidden from the user. An optional context manager is also available.

    Parameters
    ----------
    shape : None | tuple[int, ...]
        Output array shape e.g. (h, w, c).
    dtype : None | np.dtype
        Output array data type e.g. np.uint8.

    Attributes
    ----------
    shape : tuple of int
        Output array shape.
    dtype : data-type
        Output array data type.
    output : nd.ndarray or None
        Created array or `None` if `create` has not been called yet.
    """

    def __init__(
        self, shape: None | tuple[int, ...] = None, dtype: None | np.dtype = None
    ):
        self._shape = shape
        self._dtype = dtype
        self.output = None

    @property
    def shape(self) -> tuple[int, ...]:
        return self._shape

    @shape.setter
    def shape(self, shape: tuple[int, ...]):
        if not isinstance(shape, tuple) or not all(isinstance(s, int) for s in shape):
            raise TypeError("'shape' must be a tuple of integers.")
        self._shape = shape

    @property
    def dtype(self) -> np.dtype:
        return self._dtype

    @dtype.setter
    def dtype(self, dtype: np.dtype):
        self._dtype = np.dtype(dtype)

    @abstractmethod
    def create(self):
        """Allocate the array."""
        pass

    @abstractmethod
    def cleanup(self):
        """Release resources."""
        pass

    def __enter__(self):
        if self.output is None:
            self.create()
        return self.output

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()


class NumpyWriter(ArrayWriter):
    """A basic in-memory numpy array writer."""

    def create(self):
        """
        Create a zero-initialized NumPy array.

        Returns
        -------
        ndarray
            A NumPy array initialized with zeros.
        """
        if self._shape is None or self._dtype is None:
            raise ValueError("'shape' and 'dtype' must be set before array creation.")

        self.output = np.zeros(self._shape, dtype=self._dtype)

    def cleanup(self):
        """No cleanup required for in-memory numpy arrays."""
        self.output = None


class SharedMemoryWriter(ArrayWriter):
    """An array writer backed by shared memory.

    Attributes
    ----------
    shm : shared_memory.SharedMemory
        Shared memory block used to store the array.
    """

    def __init__(
        self, shape: None | tuple[int, ...] = None, dtype: None | np.dtype = None
    ):
        super().__init__(shape, dtype)
        self.shm: shared_memory.SharedMemory = None

    def create(self):
        """Creates a shared memory-backed array."""
        if self._shape is None or self._dtype is None:
            raise ValueError("'shape' and 'dtype' must be set before array creation.")

        size = np.prod(self._shape) * np.dtype(self._dtype).itemsize
        self.shm = shared_memory.SharedMemory(create=True, size=size)
        self.output = np.ndarray(self._shape, dtype=self._dtype, buffer=self.shm.buf)
        self.output.setflags(write=True)

    def cleanup(self):
        """Clean up shared memory resources safely."""
        if self.output is not None:
            self.output = None
        if self.shm is not None:
            try:
                self.shm.close()
            except Exception:
                pass
            try:
                self.shm.unlink()
            except Exception:
                pass
            finally:
                self.shm = None


class MemmapWriter(ArrayWriter):
    """A memory-mapped array writer.

    Parameters
    ----------
    shape : tuple[int, ...]
        Array shape.
    dtype : np.dtype
        Array data type.
    path : str | Path
        File path where the memory-mapped file will be stored.

    Attributes
    ----------
    path : str
        File path of the memory-mapped file.
    """

    def __init__(
        self,
        path: str | Path,
        shape: None | tuple[int, ...] = None,
        dtype: None | np.dtype = None,
    ):
        super().__init__(shape, dtype)
        if not str(path).endswith(".npy"):
            raise ValueError("'path' must be specify a .npy file.")

        self.path = path

    def create(self):
        """Create a memory-mapped array on disk."""
        if self._shape is None or self._dtype is None:
            raise ValueError("'shape' and 'dtype' must be set before array creation.")

        self.output = np.lib.format.open_memmap(
            self.path, dtype=self._dtype, shape=self._shape, mode="w+"
        )

    def cleanup(self):
        """Clean up memory-mapped file resources by deleting the file."""
        if self.output is not None:
            if hasattr(self.output, "flush"):
                self.output.flush()
            del self.output
            self.output = None
        if os.path.exists(self.path):
            try:
                os.remove(self.path)
            except OSError:
                pass


class ZarrWriter(ArrayWriter):
    """A zarr array writer.

    Parameters
    ----------
    path : str
        File path where the .zarr file will be stored.
    chunks : tuple[int, ...]
        Chunks shape for the zarr array.
    shape : tuple[int, ...]
        Array shape.
    dtype : np.dtype
        Array data type.

    Attributes
    ----------
    path : str | Path
        File path of the zarr array.
    chunks : tuple[int, ...]
        Chunk shape.
    output : zarr.Array | None
        Initialized Zarr array once `create` is called.
    """

    def __init__(
        self,
        path: str | Path,
        chunks: None | tuple[int, ...] = None,
        shape: None | tuple[int, ...] = None,
        dtype: None | np.dtype = None,
    ):
        super().__init__(shape, dtype)
        if not str(path).endswith(".zarr"):
            raise ValueError("'path' must be specify a .zarr file.")

        self.path = path
        self.chunks = chunks

    def create(self):
        """Create a zarr array on disk."""
        if self._shape is None or self._dtype is None or self.chunks is None:
            raise ValueError(
                "'shape', 'dtype', and 'chunks' must be set before zarr array creation."
            )

        self.output = zarr.open(
            store=self.path,
            mode="w",
            shape=self._shape,
            chunks=self.chunks,
            dtype=self._dtype,
        )

    def cleanup(self):
        """Clean up .zarr file by deleting the file."""
        if os.path.exists(self.path):
            shutil.rmtree(self.path)


def inject_writer(writer: ArrayWriter) -> Callable:
    """Wraps a worker function to inject writer-specific context.

    This decorator should be used on any worker function that may
    use any of the possible derived ArrayWriter classes. For certain
    writers, additional data must be passed to each worker to ensure
    arrays can be written to in parallel. For any writer, the worker
    function must specify the output array in the first position.

    Parameters
    ----------
    writer : ArrayWriter
        A valid ArrayWriter.

    Returns
    -------
    Callable
        Decorated function with injected writer-specific data.
    """

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            if isinstance(writer, SharedMemoryWriter):
                shm = shared_memory.SharedMemory(name=writer.shm.name)
                arr = np.ndarray(writer._shape, dtype=writer._dtype, buffer=shm.buf)
                try:
                    return func(arr, *args, **kwargs)
                finally:
                    shm.close()
            elif isinstance(writer, ZarrWriter):
                arr = zarr.open(writer.path, mode="r+")
                return func(arr, *args, **kwargs)
            else:
                return func(writer.output, *args, **kwargs)

        return wrapper

    return decorator


def get_array_writer(
    shape: tuple[int, ...],
    backend: str,
    dtype: np.dtype,
    path: None | str = None,
    chunks: None | tuple[int, ...] = None,
) -> ArrayWriter:
    """Initialize an array writer backend.

    Parameters
    ----------
    shape : tuple[int, ...]
        Shape of array being written to.
    backend : str
        Array writer backend. One of 'numpy', 'shared', 'memmap'.
    path : None | str
        If backend is 'memmap' or 'zarr', then path to output file.
    chunks : tuple[int, ...]
        If backend is 'zarr', then Chunks shape for the zarr array.

    Returns
    -------
    ArrayWriter
        Array writer with specified backend.
    """
    match backend:
        case "numpy":
            return NumpyWriter(shape, dtype)
        case "shared":
            return SharedMemoryWriter(shape, dtype)
        case "memmap":
            if path is None:
                raise ValueError("If backend is 'memmap', 'path' must be specified.")
            return MemmapWriter(path, shape, dtype)
        case "zarr":
            if path is None:
                raise ValueError("If backend is 'zarr', 'path' must be specified.")
            if chunks is None:
                raise ValueError("If backend is 'zarr', 'chunks' must be specified.")
            return ZarrWriter(path, chunks, shape, dtype)
        case _:
            raise ValueError(
                "'backend' must be one of: 'numpy', 'shared', 'memmap', 'zarr'."
            )
