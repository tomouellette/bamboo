import pytest
from bamboo.image import Brightfield
from bamboo.tile import Tiles
from bamboo.utils import Map, ProcessMap, ThreadMap, apply_map
from tests._fixtures.slides import SlideSvs


def test_map_basic():
    image = Brightfield(SlideSvs.path, level=2)
    results = Map(lambda tile: tile.shape, Tiles(image, 128, 128))
    for shape in results:
        assert shape == (128, 128, 3)


def test_parallel_map_threads():
    image = Brightfield(SlideSvs.path, level=2)
    results_a = Map(lambda tile: tile.shape, Tiles(image, 128, 128))
    results_b = ThreadMap(lambda tile: tile.shape, Tiles(image, 128, 128), n_jobs=2)
    for a, b in zip(results_a, results_b):
        assert a == b


def test_parallel_map_processes():
    image = Brightfield(SlideSvs.path, level=2)
    results_a = Map(lambda tile: tile.shape, Tiles(image, 128, 128))
    results_b = ProcessMap(lambda tile: tile.shape, Tiles(image, 128, 128), n_jobs=2)
    for a, b in zip(results_a, results_b):
        assert a == b


def test_apply_map_functionality():
    def _func(i):
        return i * 2

    single_map = apply_map(_func, range(10), n_jobs=1, silent=True)
    process_map = apply_map(_func, range(10), n_jobs=2, silent=True)
    thread_map = apply_map(_func, range(10), n_jobs=2, prefer="threads", silent=True)

    for result in [single_map, process_map, thread_map]:
        for i, val in enumerate(result):
            assert val == i * 2
