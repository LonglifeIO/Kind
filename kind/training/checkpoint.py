"""Phase 4 atomic checkpoint manager.

The commit-side of the checkpoint barrier protocol the implementation
plan §2.8 specifies. The wire-side of the same protocol — sending
``BARRIER_BEGIN`` / ``BARRIER_END`` and waiting for the server's
``BARRIER_BEGIN_ACK`` — lives in :mod:`kind.env.transport`.

**Five-step commit.** :meth:`CheckpointManager.commit` runs:

1. Send ``BARRIER_BEGIN`` over the transport client.
2. Wait for ``BARRIER_BEGIN_ACK`` (handled inside
   :meth:`~kind.env.transport.EnvTransportClient.barrier_begin`).
3. Stage all named files plus the replay parquet shards into a
   ``{checkpoint_id}.staging/`` directory next to the target.
4. ``fsync`` every staged file, ``fsync`` the staging directory itself,
   ``os.rename`` it to ``{checkpoint_id}/``, then ``fsync`` the parent
   directory so the rename is durable across crashes.
5. Send ``BARRIER_END`` over the transport client.

**Atomicity.** ``os.rename`` of a same-filesystem same-parent directory
is atomic on POSIX systems; the checkpoint either exists fully under
``{checkpoint_id}/`` or it does not exist at all. If staging or the
rename raises, the staging directory is removed (best-effort) so a
subsequent commit attempt does not collide with a partial result.

**What this does not do.** Phase 4 does not build the logic that
produces the weights, optimizer state, replay shards, RNG state, or
telemetry offsets — that is Phase 5's runner. The manager takes
:class:`CheckpointContents` carrying paths to files that already exist
and atomically commits them as a single unit. The dummy-content tests in
``tests/test_checkpoint.py`` exercise the manager end-to-end with small
text files standing in for the real artifacts.
"""

from __future__ import annotations

import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from kind.env.transport import EnvTransportClient

__all__ = [
    "CheckpointContents",
    "CheckpointManager",
]


# ---- canonical layout ----------------------------------------------------


_CANONICAL_FILE_NAMES: Final[dict[str, str]] = {
    "weights_path": "weights.safetensors",
    "replay_meta_path": "replay_meta.json",
    "optimizer_state_path": "optimizer_state.pt",
    "rng_state_path": "rng_state.pkl",
    "telemetry_offsets_path": "telemetry_offsets.json",
    "schema_version_path": "schema_version.txt",
}

_REPLAY_SUBDIR: Final[str] = "replay"
_STAGING_SUFFIX: Final[str] = ".staging"
#: Phase 8c — the optional offline-period state file (metabolic bucket +
#: StateController resume state). Present only on offline checkpoints; waking
#: checkpoints resume from defaults (waking + full bucket).
_OFFLINE_STATE_FILE: Final[str] = "offline_state.json"


# ---- contents ------------------------------------------------------------


@dataclass(frozen=True)
class CheckpointContents:
    """Source paths for one atomic commit.

    Each named field points to a pre-existing file. The manager copies
    the file into the checkpoint under a canonical name (see
    :data:`_CANONICAL_FILE_NAMES`); the source basename is not preserved.
    Replay parquet shards (zero or more) are copied under a
    ``replay/`` subdirectory by basename — those names are preserved
    because there can be multiple shards per checkpoint.
    """

    weights_path: Path
    replay_meta_path: Path
    optimizer_state_path: Path
    rng_state_path: Path
    telemetry_offsets_path: Path
    schema_version_path: Path
    replay_parquet_shards: tuple[Path, ...] = ()
    #: Phase 8c — additive, optional. The offline-period state (metabolic bucket +
    #: controller resume state); set on offline checkpoints, ``None`` on waking.
    offline_state_path: Path | None = None


# ---- manager -------------------------------------------------------------


class CheckpointManager:
    """Coordinates the network barrier and the on-disk atomic rename.

    Constructed once per run with the checkpoints directory and the
    :class:`~kind.env.transport.EnvTransportClient` connecting to the
    env-server. :meth:`commit` is the single-call interface for the
    runner; :meth:`latest` and :meth:`load` are read-side helpers Phase 5
    will use to resume a run from disk.
    """

    def __init__(
        self,
        checkpoints_dir: Path,
        transport_client: EnvTransportClient,
    ) -> None:
        self._checkpoints_dir = checkpoints_dir
        self._transport_client = transport_client
        self._checkpoints_dir.mkdir(parents=True, exist_ok=True)

    # ---- public API ------------------------------------------------------

    def commit(
        self, checkpoint_id: str, contents: CheckpointContents
    ) -> None:
        """Run the five-step barrier-and-commit protocol (waking checkpoint).

        Raises :class:`FileExistsError` if a checkpoint with this ID
        already exists (checked before the barrier so the env-server is
        not paused for a doomed commit). Raises :class:`FileNotFoundError`
        if any source file is missing. On any other staging failure, the
        staging directory is best-effort removed and a ``BARRIER_END`` is
        sent so the env-server resumes; the original exception then
        propagates to the caller.
        """
        self._commit_impl(checkpoint_id, contents, use_barrier=True)

    def commit_offline(
        self, checkpoint_id: str, contents: CheckpointContents
    ) -> None:
        """Commit during the *offline* period (dreaming / dormant) — Phase 8c.

        **No env-server barrier.** The desktop/env is off during the offline
        period, so there is no env state to coordinate and no transport to
        barrier with — only the canonical *mind* state to save. Same atomic
        temp-then-rename as :meth:`commit` (a crash mid-write can't corrupt the
        committed checkpoint), which keeps it split-ready. (The full Mac/desktop
        two-machine atomic sync is real-deployment infrastructure, out of scope.)
        """
        self._commit_impl(checkpoint_id, contents, use_barrier=False)

    def _commit_impl(
        self, checkpoint_id: str, contents: CheckpointContents, *, use_barrier: bool
    ) -> None:
        self._validate_id(checkpoint_id)
        target_dir = self._checkpoints_dir / checkpoint_id
        if target_dir.exists():
            raise FileExistsError(
                f"checkpoint already exists: {checkpoint_id!r}"
            )
        staging_dir = self._checkpoints_dir / f"{checkpoint_id}{_STAGING_SUFFIX}"
        if staging_dir.exists():
            # Leftover from a prior failed attempt; remove so the new
            # mkdir below succeeds.
            shutil.rmtree(staging_dir)

        if use_barrier:
            self._transport_client.barrier_begin(checkpoint_id)
        try:
            self._stage_and_rename(staging_dir, target_dir, contents)
        except BaseException:
            # Best-effort cleanup of the staging directory so the next
            # attempt does not collide. Then send BARRIER_END so the
            # env-server resumes; suppress any error from BARRIER_END so
            # the original commit error is what the caller sees.
            if staging_dir.exists():
                shutil.rmtree(staging_dir, ignore_errors=True)
            if use_barrier:
                try:
                    self._transport_client.barrier_end(checkpoint_id)
                except Exception:
                    pass
            raise
        if use_barrier:
            self._transport_client.barrier_end(checkpoint_id)

    def latest(self) -> str | None:
        """Return the lexicographically greatest committed checkpoint ID.

        Phase 5 will format IDs as zero-padded counters (e.g.
        ``ckpt-000123``) so lexicographic order matches commit order.
        Returns ``None`` if no checkpoint has been committed.
        """
        candidates = [
            entry.name
            for entry in self._checkpoints_dir.iterdir()
            if entry.is_dir() and not entry.name.endswith(_STAGING_SUFFIX)
        ]
        if not candidates:
            return None
        return max(candidates)

    def load(self, checkpoint_id: str) -> CheckpointContents:
        """Return :class:`CheckpointContents` pointing into ``checkpoint_id/``.

        Does not read any file; only constructs paths. Raises
        :class:`FileNotFoundError` if the checkpoint directory is missing
        — replay parquet shards are discovered by listing
        ``checkpoint_id/replay/`` if it exists, sorted by name for
        deterministic ordering.
        """
        target_dir = self._checkpoints_dir / checkpoint_id
        if not target_dir.is_dir():
            raise FileNotFoundError(
                f"checkpoint not found: {checkpoint_id!r}"
            )
        replay_dir = target_dir / _REPLAY_SUBDIR
        if replay_dir.is_dir():
            shards = tuple(sorted(replay_dir.iterdir()))
        else:
            shards = ()
        offline_state = target_dir / _OFFLINE_STATE_FILE
        return CheckpointContents(
            weights_path=target_dir / _CANONICAL_FILE_NAMES["weights_path"],
            replay_meta_path=target_dir
            / _CANONICAL_FILE_NAMES["replay_meta_path"],
            optimizer_state_path=target_dir
            / _CANONICAL_FILE_NAMES["optimizer_state_path"],
            rng_state_path=target_dir
            / _CANONICAL_FILE_NAMES["rng_state_path"],
            telemetry_offsets_path=target_dir
            / _CANONICAL_FILE_NAMES["telemetry_offsets_path"],
            schema_version_path=target_dir
            / _CANONICAL_FILE_NAMES["schema_version_path"],
            replay_parquet_shards=shards,
            offline_state_path=offline_state if offline_state.is_file() else None,
        )

    # ---- internals -------------------------------------------------------

    @staticmethod
    def _validate_id(checkpoint_id: str) -> None:
        if not checkpoint_id:
            raise ValueError("checkpoint_id must be non-empty")
        if "/" in checkpoint_id or "\\" in checkpoint_id:
            raise ValueError(
                f"checkpoint_id must not contain path separators: "
                f"{checkpoint_id!r}"
            )
        if checkpoint_id.endswith(_STAGING_SUFFIX):
            raise ValueError(
                f"checkpoint_id must not end with {_STAGING_SUFFIX!r}: "
                f"{checkpoint_id!r}"
            )

    def _stage_and_rename(
        self,
        staging_dir: Path,
        target_dir: Path,
        contents: CheckpointContents,
    ) -> None:
        """Copy → fsync → atomic rename. See module docstring for the order."""
        staging_dir.mkdir(parents=True)

        # Copy the named files under canonical names.
        for attr_name, dest_name in _CANONICAL_FILE_NAMES.items():
            source = getattr(contents, attr_name)
            assert isinstance(source, Path)  # narrows for mypy
            if not source.exists():
                raise FileNotFoundError(
                    f"source path missing for {attr_name}: {source}"
                )
            dest = staging_dir / dest_name
            shutil.copy2(source, dest)
            self._fsync_file(dest)

        # Copy parquet shards under the replay/ subdir, basename preserved.
        if contents.replay_parquet_shards:
            replay_dir = staging_dir / _REPLAY_SUBDIR
            replay_dir.mkdir()
            for shard in contents.replay_parquet_shards:
                if not shard.exists():
                    raise FileNotFoundError(
                        f"replay shard missing: {shard}"
                    )
                dest = replay_dir / shard.name
                shutil.copy2(shard, dest)
                self._fsync_file(dest)
            self._fsync_dir(replay_dir)

        # Phase 8c — the optional offline-period state file (additive).
        if contents.offline_state_path is not None:
            if not contents.offline_state_path.exists():
                raise FileNotFoundError(
                    f"offline_state source missing: {contents.offline_state_path}"
                )
            dest = staging_dir / _OFFLINE_STATE_FILE
            shutil.copy2(contents.offline_state_path, dest)
            self._fsync_file(dest)

        # fsync the staging directory itself so the directory entries are
        # durable, then atomic rename, then fsync the parent so the
        # rename is durable across crashes.
        self._fsync_dir(staging_dir)
        os.rename(staging_dir, target_dir)
        self._fsync_dir(self._checkpoints_dir)

    @staticmethod
    def _fsync_file(path: Path) -> None:
        with open(path, "rb") as fh:
            os.fsync(fh.fileno())

    @staticmethod
    def _fsync_dir(path: Path) -> None:
        # On POSIX, opening a directory for read and fsync-ing it makes
        # the directory entries durable. On platforms where this is not
        # supported, the operation is a best-effort no-op.
        try:
            fd = os.open(path, os.O_RDONLY)
        except OSError:
            return
        try:
            os.fsync(fd)
        except OSError:
            # macOS and Linux support fsync on a directory; if a future
            # platform doesn't, a non-fsync rename is still atomic in the
            # in-memory filesystem and recovers correctly on a graceful
            # shutdown. Probe 1 runs on macOS, where fsync on a directory
            # works.
            pass
        finally:
            os.close(fd)
