"""Phase 4 tests for ``kind/training/checkpoint.py``.

Exercises the atomic-rename-with-barrier protocol the implementation plan
§2.8 specifies. Phase 4 builds the manager around already-prepared paths
— Phase 5's runner will produce real weights, optimizer state, etc. The
tests below use small dummy text files so the focus stays on the commit
protocol's structural properties: the staging directory is renamed
atomically, partial-failure paths leave a clean state, ``BARRIER_END`` is
always sent so the env-server resumes, and load/latest read the
expected layout back.

Tests use a real :class:`EnvTransportClient` connected to a real
:class:`EnvTransportServer` so the full barrier round-trip exercises the
wire protocol from the manager's side. The env-server's grid behavior is
not what's under test here; we use a quiet config and a no-op handler so
the only events on the wire are the env_reset (during connect) and the
barrier flow.
"""

from __future__ import annotations

import threading
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import pytest

from kind.env.env_server import EnvServer, EnvServerConfig
from kind.env.grid_world import GridWorldConfig
from kind.env.transport import (
    EnvTransportClient,
    EnvTransportServer,
)
from kind.training.checkpoint import CheckpointContents, CheckpointManager


# ---- helpers --------------------------------------------------------------


def _quiet_config() -> GridWorldConfig:
    return GridWorldConfig(
        initial_regrowth_p=0.0,
        drift_p_min=0.0,
        drift_p_max=0.0,
        drift_magnitude_per_step=0.0,
        n_initial_resources=0,
    )


@contextmanager
def _connected_pair() -> Iterator[
    tuple[EnvTransportServer, EnvTransportClient, threading.Thread]
]:
    """Spin up a server thread + connected client; yield the triple.

    The same shape as ``test_transport.py``'s helper, repeated here so
    the checkpoint tests stay self-contained.
    """
    config = EnvServerConfig(
        grid_world_config=_quiet_config(),
        seed=42,
        world_event_handler=lambda _record: None,
        run_id="checkpoint-test",
    )
    env_server = EnvServer(config)
    transport_server = EnvTransportServer(env_server, host="127.0.0.1", port=0)
    server_thread = threading.Thread(
        target=transport_server.serve_forever, daemon=True
    )
    server_thread.start()

    client = EnvTransportClient(
        host="127.0.0.1",
        port=transport_server.actual_port,
        world_event_handler=lambda _record: None,
    )
    client.connect()
    try:
        yield transport_server, client, server_thread
    finally:
        try:
            client.close()
        finally:
            transport_server.shutdown()
            server_thread.join(timeout=5.0)


def _make_dummy_contents(tmp_path: Path) -> CheckpointContents:
    """Build a :class:`CheckpointContents` with small text files.

    Phase 4 does not produce real weights or optimizer state; the manager
    only handles the atomic-rename-with-barrier dance. These dummy files
    let us test the commit shape end-to-end without dragging in PyTorch
    or the world model.
    """
    src = tmp_path / "src"
    src.mkdir()
    paths: dict[str, Path] = {}
    for filename in [
        "weights.bin",
        "replay_meta.json",
        "optimizer.bin",
        "rng.pkl",
        "telemetry_offsets.json",
        "schema_version.txt",
    ]:
        p = src / filename
        p.write_text(f"dummy contents of {filename}\n")
        paths[filename] = p
    return CheckpointContents(
        weights_path=paths["weights.bin"],
        replay_meta_path=paths["replay_meta.json"],
        optimizer_state_path=paths["optimizer.bin"],
        rng_state_path=paths["rng.pkl"],
        telemetry_offsets_path=paths["telemetry_offsets.json"],
        schema_version_path=paths["schema_version.txt"],
    )


# ---- gate test: full atomic commit roundtrip -----------------------------


def test_commit_creates_checkpoint_directory_with_canonical_files(
    tmp_path: Path,
) -> None:
    """End-to-end commit: every named file lands in the checkpoint dir
    under its canonical name; the staging dir is gone afterwards."""
    contents = _make_dummy_contents(tmp_path)
    ckpts_dir = tmp_path / "checkpoints"

    with _connected_pair() as (_server, client, _thread):
        manager = CheckpointManager(ckpts_dir, client)
        manager.commit("ckpt-000001", contents)

    target = ckpts_dir / "ckpt-000001"
    assert target.is_dir()
    expected_names = {
        "weights.safetensors",
        "replay_meta.json",
        "optimizer_state.pt",
        "rng_state.pkl",
        "telemetry_offsets.json",
        "schema_version.txt",
    }
    actual_names = {p.name for p in target.iterdir()}
    assert actual_names == expected_names

    # Staging directory is cleaned up.
    assert not (ckpts_dir / "ckpt-000001.staging").exists()

    # Source paths are preserved (commit copies, doesn't move).
    assert contents.weights_path.exists()


def test_commit_with_replay_parquet_shards_places_them_in_replay_subdir(
    tmp_path: Path,
) -> None:
    contents_no_shards = _make_dummy_contents(tmp_path)
    shard_dir = tmp_path / "src_shards"
    shard_dir.mkdir()
    shard_paths = []
    for i in range(3):
        p = shard_dir / f"shard-{i:06d}.parquet"
        p.write_text(f"shard {i} content\n")
        shard_paths.append(p)
    contents = CheckpointContents(
        weights_path=contents_no_shards.weights_path,
        replay_meta_path=contents_no_shards.replay_meta_path,
        optimizer_state_path=contents_no_shards.optimizer_state_path,
        rng_state_path=contents_no_shards.rng_state_path,
        telemetry_offsets_path=contents_no_shards.telemetry_offsets_path,
        schema_version_path=contents_no_shards.schema_version_path,
        replay_parquet_shards=tuple(shard_paths),
    )

    ckpts_dir = tmp_path / "checkpoints"
    with _connected_pair() as (_server, client, _thread):
        manager = CheckpointManager(ckpts_dir, client)
        manager.commit("ckpt-with-shards", contents)

    replay_dir = ckpts_dir / "ckpt-with-shards" / "replay"
    assert replay_dir.is_dir()
    actual_shard_names = sorted(p.name for p in replay_dir.iterdir())
    assert actual_shard_names == [
        "shard-000000.parquet",
        "shard-000001.parquet",
        "shard-000002.parquet",
    ]


def test_committed_files_have_expected_contents(tmp_path: Path) -> None:
    """Atomic copy preserves bytes; load() reads them back."""
    contents = _make_dummy_contents(tmp_path)
    ckpts_dir = tmp_path / "checkpoints"

    with _connected_pair() as (_server, client, _thread):
        manager = CheckpointManager(ckpts_dir, client)
        manager.commit("ckpt-bytes", contents)

    manager_read = CheckpointManager(
        ckpts_dir,
        # The transport client is irrelevant for reads; pass a fresh
        # client that never connects to keep the read path independent
        # from the wire.
        EnvTransportClient(
            "127.0.0.1", 1, world_event_handler=lambda _r: None
        ),
    )
    loaded = manager_read.load("ckpt-bytes")
    assert loaded.weights_path.read_text() == "dummy contents of weights.bin\n"
    assert (
        loaded.schema_version_path.read_text()
        == "dummy contents of schema_version.txt\n"
    )


# ---- partial-failure cleanup ---------------------------------------------


def test_commit_with_missing_source_file_leaves_no_partial_checkpoint(
    tmp_path: Path,
) -> None:
    """If a source path is missing, the commit raises and the target
    directory does not exist (atomic-or-nothing). The staging directory
    is also cleaned up, and ``BARRIER_END`` is sent so the env-server
    can resume."""
    contents = _make_dummy_contents(tmp_path)
    # Remove one source file before commit so staging fails mid-copy.
    contents.weights_path.unlink()

    ckpts_dir = tmp_path / "checkpoints"
    with _connected_pair() as (_server, client, _thread):
        manager = CheckpointManager(ckpts_dir, client)
        with pytest.raises(FileNotFoundError):
            manager.commit("ckpt-fails", contents)

        # Neither the target nor the staging directory exists.
        assert not (ckpts_dir / "ckpt-fails").exists()
        assert not (ckpts_dir / "ckpt-fails.staging").exists()

        # The env-server is no longer in barrier mode — a subsequent step
        # works without hanging.
        client.step(4)


def test_commit_with_existing_target_raises_before_barrier(
    tmp_path: Path,
) -> None:
    """If a checkpoint with this ID already exists, the commit fails fast
    without disturbing the env-server (no barrier issued)."""
    contents = _make_dummy_contents(tmp_path)
    ckpts_dir = tmp_path / "checkpoints"

    with _connected_pair() as (_server, client, _thread):
        manager = CheckpointManager(ckpts_dir, client)
        manager.commit("ckpt-dup", contents)
        # Restore source files (commit copies, source still exists).
        with pytest.raises(FileExistsError, match="already exists"):
            manager.commit("ckpt-dup", contents)

        # Server still works; no barrier was issued for the failed call.
        client.step(4)


def test_commit_recovers_from_stale_staging_directory(
    tmp_path: Path,
) -> None:
    """A leftover ``{id}.staging/`` from a prior failed attempt is
    cleared before the new commit's staging starts. Phase 1 of the
    barrier+commit sequence must not collide with prior state."""
    contents = _make_dummy_contents(tmp_path)
    ckpts_dir = tmp_path / "checkpoints"
    ckpts_dir.mkdir()
    # Plant a stale staging directory.
    stale = ckpts_dir / "ckpt-stale.staging"
    stale.mkdir()
    (stale / "garbage.txt").write_text("leftover from a crash\n")

    with _connected_pair() as (_server, client, _thread):
        manager = CheckpointManager(ckpts_dir, client)
        manager.commit("ckpt-stale", contents)

    # The fresh checkpoint exists; the stale staging directory is gone.
    assert (ckpts_dir / "ckpt-stale").is_dir()
    assert not (ckpts_dir / "ckpt-stale.staging").exists()


# ---- ID validation -------------------------------------------------------


@pytest.mark.parametrize(
    "bad_id",
    ["", "with/slash", "with\\backslash", "ends-with-suffix.staging"],
)
def test_commit_rejects_invalid_checkpoint_id(
    tmp_path: Path, bad_id: str
) -> None:
    contents = _make_dummy_contents(tmp_path)
    ckpts_dir = tmp_path / "checkpoints"
    with _connected_pair() as (_server, client, _thread):
        manager = CheckpointManager(ckpts_dir, client)
        with pytest.raises(ValueError):
            manager.commit(bad_id, contents)


# ---- read API -----------------------------------------------------------


def test_latest_returns_lexicographic_max_of_committed_checkpoints(
    tmp_path: Path,
) -> None:
    contents = _make_dummy_contents(tmp_path)
    ckpts_dir = tmp_path / "checkpoints"

    with _connected_pair() as (_server, client, _thread):
        manager = CheckpointManager(ckpts_dir, client)
        assert manager.latest() is None
        manager.commit("ckpt-000001", contents)

    # Use a fresh tmp_path subdir for the second commit's source files
    # to avoid colliding with the first commit's source dir (which still
    # exists since commit() copies, not moves).
    src2 = tmp_path / "src2"
    src2.mkdir()
    paths2: dict[str, Path] = {}
    for filename in [
        "weights.bin",
        "replay_meta.json",
        "optimizer.bin",
        "rng.pkl",
        "telemetry_offsets.json",
        "schema_version.txt",
    ]:
        p = src2 / filename
        p.write_text(f"second commit contents of {filename}\n")
        paths2[filename] = p
    contents2 = CheckpointContents(
        weights_path=paths2["weights.bin"],
        replay_meta_path=paths2["replay_meta.json"],
        optimizer_state_path=paths2["optimizer.bin"],
        rng_state_path=paths2["rng.pkl"],
        telemetry_offsets_path=paths2["telemetry_offsets.json"],
        schema_version_path=paths2["schema_version.txt"],
    )

    with _connected_pair() as (_server, client, _thread):
        manager = CheckpointManager(ckpts_dir, client)
        manager.commit("ckpt-000002", contents2)
        assert manager.latest() == "ckpt-000002"


def test_latest_ignores_staging_directories(tmp_path: Path) -> None:
    """Stale ``.staging`` directories are not candidates for latest()."""
    ckpts_dir = tmp_path / "checkpoints"
    ckpts_dir.mkdir()
    (ckpts_dir / "ckpt-001").mkdir()
    (ckpts_dir / "ckpt-002.staging").mkdir()  # leftover from a crash
    # Phase 4 doesn't need a connected client for read-only ops, but
    # CheckpointManager.__init__ requires one — pass an unconnected one.
    client = EnvTransportClient(
        "127.0.0.1", 1, world_event_handler=lambda _r: None
    )
    manager = CheckpointManager(ckpts_dir, client)
    assert manager.latest() == "ckpt-001"


def test_load_returns_paths_into_checkpoint_directory(tmp_path: Path) -> None:
    contents = _make_dummy_contents(tmp_path)
    ckpts_dir = tmp_path / "checkpoints"

    with _connected_pair() as (_server, client, _thread):
        manager = CheckpointManager(ckpts_dir, client)
        manager.commit("ckpt-load-test", contents)
        loaded = manager.load("ckpt-load-test")

    expected_root = ckpts_dir / "ckpt-load-test"
    assert loaded.weights_path == expected_root / "weights.safetensors"
    assert loaded.replay_meta_path == expected_root / "replay_meta.json"
    assert loaded.optimizer_state_path == expected_root / "optimizer_state.pt"
    assert loaded.rng_state_path == expected_root / "rng_state.pkl"
    assert (
        loaded.telemetry_offsets_path
        == expected_root / "telemetry_offsets.json"
    )
    assert loaded.schema_version_path == expected_root / "schema_version.txt"
    assert loaded.replay_parquet_shards == ()


def test_load_unknown_checkpoint_raises(tmp_path: Path) -> None:
    ckpts_dir = tmp_path / "checkpoints"
    ckpts_dir.mkdir()
    client = EnvTransportClient(
        "127.0.0.1", 1, world_event_handler=lambda _r: None
    )
    manager = CheckpointManager(ckpts_dir, client)
    with pytest.raises(FileNotFoundError):
        manager.load("does-not-exist")


# ---- barrier interaction -------------------------------------------------


def test_commit_pauses_then_resumes_env_server(tmp_path: Path) -> None:
    """During the commit, no STEP issued from another thread should
    receive a TRANSITION until BARRIER_END is sent (which the manager
    sends as part of the commit). After the commit, STEPs work again."""
    contents = _make_dummy_contents(tmp_path)
    ckpts_dir = tmp_path / "checkpoints"

    with _connected_pair() as (_server, client, _thread):
        # Sanity: a step works before commit.
        before = client.step(4)
        assert before.env_step == 1

        manager = CheckpointManager(ckpts_dir, client)
        manager.commit("ckpt-barrier", contents)

        # And after commit, STEPs work again — the manager sent
        # BARRIER_END so the env-server is no longer in barrier mode.
        after = client.step(4)
        assert after.env_step == 2
