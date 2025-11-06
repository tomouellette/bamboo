import numpy as np
from bamboo.tile import Tiles
from bamboo.image import Brightfield
from bamboo.models.zoo import GrandQC, GrandQCStats
from bamboo.models._semantic import Semantic, SemanticStats
from bamboo.writer import NumpyWriter, SharedMemoryWriter, MemmapWriter, ZarrWriter
from tests._fixtures.images import RgbBmp


def test_grandqc_semantic():
    """Test GrandQC semantic inference with multiple writer types and parallelization."""
    settings = [
        (NumpyWriter, None, 1),
        (SharedMemoryWriter, "processes", 2),
        (SharedMemoryWriter, "threads", 2),
        (MemmapWriter, "processes", 2),
        (MemmapWriter, "threads", 2),
        (ZarrWriter, "processes", 2),
        (ZarrWriter, "threads", 2),
    ]

    model = GrandQC(device="cpu")
    for writer_cls, prefer, jobs in settings:
        image = Brightfield(RgbBmp.path, mpp=model.mpp)

        if writer_cls == MemmapWriter:
            writer = writer_cls(dtype=model.dtype, path="temp.npy")
        elif writer_cls == ZarrWriter:
            writer = writer_cls(dtype=model.dtype, path="temp.zarr")
        else:
            writer = writer_cls(dtype=model.dtype)

        engine = Semantic(
            model,
            Tiles(image, model.tile_width, model.tile_height, mpp=model.mpp),
            writer=writer,
            n_jobs=jobs,
            prefer=prefer,
        )

        msg = f"Failed: {writer.__class__.__name__}, {prefer}"
        output = engine.run(silent=True)
        assert output.shape == image.shape[:2], msg
        assert np.sum(output) > 1, msg

        writer.cleanup()
        del engine


def test_grandqc_semantic_stats():
    """Test GrandQC semantic stats extraction with SemanticStats engine."""
    model = GrandQC(device="cpu")
    image = Brightfield(RgbBmp.path, mpp=model.mpp)

    tiles = Tiles(image, model.tile_width, model.tile_height, mpp=model.mpp)

    engine = SemanticStats(
        model,
        GrandQCStats,
        tiles,
        n_jobs=1,
        prefer=None,
    )

    results = engine.run(silent=True)
    assert len(results) == tiles.n_tiles
    assert "stats" in results[0]

    for category in [
        "TISSUE",
        "FOLDS",
        "SPOTS",
        "MARKINGS",
        "ARTIFACTS",
        "OUT_OF_FOCUS",
        "BACKGROUND",
        "ARTIFACT",
    ]:
        assert category in results[0]["stats"]

    assert "coords" in results[0]
    assert len(results[0]["coords"]) == 6

    del engine
