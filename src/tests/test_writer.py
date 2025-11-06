import numpy as np
from bamboo.utils import Map, ProcessMap, ThreadMap
from bamboo.writer import (
    SharedMemoryWriter,
    MemmapWriter,
    NumpyWriter,
    ZarrWriter,
    inject_writer,
)


def test_writer_shared_process():
    writer = SharedMemoryWriter((8, 8), np.uint8)
    writer.create()
    indices = range(8)

    @inject_writer(writer)
    def _worker(output, i):
        output[i, i] = i

    _ = ProcessMap(_worker, indices, n_jobs=2, silent=True)

    data = writer.output.copy()
    writer.cleanup()
    assert np.all(np.diag(data) == np.arange(8))


def test_writer_shared_threads():
    writer = SharedMemoryWriter((8, 8), np.uint8)
    writer.create()
    indices = range(8)

    @inject_writer(writer)
    def _worker(output, i):
        output[i, i] = i

    _ = ThreadMap(_worker, indices, n_jobs=2, silent=True)

    data = writer.output.copy()
    writer.cleanup()
    assert np.all(np.diag(data) == np.arange(8))


def test_writer_memmap_process():
    writer = MemmapWriter("memmap_a.npy", (8, 8), np.uint8)
    writer.create()
    indices = range(8)

    @inject_writer(writer)
    def _worker(output, i):
        output[i, i] = i

    _ = ProcessMap(_worker, indices, n_jobs=2, silent=True)

    data = writer.output.copy()
    writer.cleanup()
    assert np.all(np.diag(data) == np.arange(8))


def test_writer_memmap_threads():
    writer = MemmapWriter("memmap_b.npy", (8, 8), np.uint8)
    writer.create()
    indices = range(8)

    @inject_writer(writer)
    def _worker(output, i):
        output[i, i] = i

    _ = ThreadMap(_worker, indices, n_jobs=2, silent=True)

    data = writer.output.copy()
    writer.cleanup()
    assert np.all(np.diag(data) == np.arange(8))


def test_writer_zarr_process():
    writer = ZarrWriter("array_a.zarr", (2, 2), (8, 8), np.uint8)
    writer.create()
    indices = np.arange(8)

    @inject_writer(writer)
    def _worker(output, i):
        output[i, i] = i

    _ = ProcessMap(_worker, np.split(indices, 8 // 2), n_jobs=2, silent=True)

    data = writer.output[:]
    writer.cleanup()
    assert np.all(np.diag(data) == np.arange(8))


def test_writer_zarr_threads():
    writer = ZarrWriter("array_b.zarr", (1, 1), (8, 8), np.uint8)
    writer.create()
    indices = np.arange(8)

    @inject_writer(writer)
    def _worker(output, i):
        output[i, i] = i

    _ = ThreadMap(_worker, np.split(indices, 8 // 2), n_jobs=2, silent=True)

    data = writer.output[:]
    writer.cleanup()
    assert np.all(np.diag(data) == np.arange(8))


def test_writer_numpy_map():
    writer = NumpyWriter((8, 8), np.uint8)
    writer.create()
    indices = range(8)

    @inject_writer(writer)
    def _worker(output, i):
        output[i, i] = i

    _ = Map(_worker, indices, silent=True)

    data = writer.output.copy()
    writer.cleanup()
    assert np.all(np.diag(data) == np.arange(8))
