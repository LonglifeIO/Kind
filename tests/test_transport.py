"""Phase 4 tests for ``kind/env/transport.py``.

Exercises the wire protocol the implementation plan §2.8 specifies and
the user spec for Phase 4 elaborates: STEP/TRANSITION roundtrip,
MUTATE/MUTATE_ACK with per-mutator argument coercion, the WORLD_EVENT
side channel for both episode-boundary events and builder perturbations,
the BARRIER_BEGIN/BARRIER_BEGIN_ACK/BARRIER_END flow with STEPs queued
during the barrier, determinism over the wire vs direct calls, wallclock
preservation, length-prefix robustness on partial messages, and the
ephemeral-port pattern (``port=0`` exposes the OS-assigned port via
``actual_port``).

Each test spins up an :class:`EnvTransportServer` on a thread, connects
a client, runs the protocol exercise, then shuts down. The
``_make_transport_pair`` context manager handles the lifecycle so each
test reads as a single piece of protocol logic rather than as an
exercise in thread management.
"""

from __future__ import annotations

import socket
import struct
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pytest

from kind.env.env_server import EnvServer, EnvServerConfig
from kind.env.grid_world import CellType, GridWorldConfig, NUM_ACTIONS
from kind.env.transport import (
    ConnectionLostError,
    EnvTransportClient,
    EnvTransportServer,
    MessageType,
    MutateError,
    TransportError,
    _encode_mutate_args,
)
from kind.observer.schemas import WorldEvent


# ---- helpers --------------------------------------------------------------


def _make_quiet_config() -> GridWorldConfig:
    """Grid config with regrowth and drift disabled — keeps the world
    still so observation comparisons across the wire are deterministic
    independent of stochastic processes."""
    return GridWorldConfig(
        initial_regrowth_p=0.0,
        drift_p_min=0.0,
        drift_p_max=0.0,
        drift_magnitude_per_step=0.0,
        n_initial_resources=0,
    )


@contextmanager
def _make_transport_pair(
    *,
    grid_world_config: GridWorldConfig | None = None,
    seed: int = 42,
    run_id: str = "test-run",
) -> Iterator[
    tuple[
        EnvTransportClient,
        EnvTransportServer,
        list[WorldEvent],
    ]
]:
    """Spin up a server thread and a connected client.

    Yields ``(client, server, recorded_world_events)``. ``recorded`` is the
    list the client-side ``world_event_handler`` appends to as messages
    arrive. The server's :class:`EnvServer` is constructed with a no-op
    handler at config time; the transport server replaces that handler
    with its own wire-shipping callable upon connection.
    """
    config = EnvServerConfig(
        grid_world_config=grid_world_config or GridWorldConfig(),
        seed=seed,
        world_event_handler=lambda _record: None,
        run_id=run_id,
    )
    env_server = EnvServer(config)
    transport_server = EnvTransportServer(env_server, host="127.0.0.1", port=0)
    port = transport_server.actual_port

    server_thread = threading.Thread(
        target=transport_server.serve_forever,
        name="EnvTransportServerThread",
        daemon=True,
    )
    server_thread.start()

    recorded: list[WorldEvent] = []
    client = EnvTransportClient(
        host="127.0.0.1", port=port, world_event_handler=recorded.append
    )
    try:
        yield client, transport_server, recorded
    finally:
        try:
            client.close()
        finally:
            transport_server.shutdown()
            server_thread.join(timeout=5.0)


# ---- ephemeral-port test --------------------------------------------------


def test_ephemeral_port_zero_returns_actual_port() -> None:
    """``port=0`` binds the OS-assigned port and exposes it on
    ``actual_port`` before ``serve_forever``."""
    config = EnvServerConfig(
        grid_world_config=_make_quiet_config(),
        seed=42,
        world_event_handler=lambda _record: None,
        run_id="ephemeral-test",
    )
    env_server = EnvServer(config)
    transport_server = EnvTransportServer(env_server, host="127.0.0.1", port=0)
    try:
        port = transport_server.actual_port
        assert port > 0
        assert port < 65536
    finally:
        transport_server.shutdown()


# ---- handshake / connect --------------------------------------------------


def test_connect_returns_initial_env_step_and_emits_env_reset_world_event() -> None:
    with _make_transport_pair() as (client, _server, recorded):
        first = client.connect()
    assert first.env_step == 0
    assert first.episode_id == 0
    assert first.step_in_episode == 0
    assert first.observation.shape == (32, 32)
    assert first.observation.dtype == np.uint8

    # The env_reset emitted during start() arrived as a WORLD_EVENT
    # before the client's connect() returned.
    assert len(recorded) == 1
    assert recorded[0].event_type == "env_reset"
    assert recorded[0].source == "environment"


# ---- STEP / TRANSITION roundtrip ------------------------------------------


def test_step_returns_env_step_with_matching_request_id_handling() -> None:
    """A single STEP yields a TRANSITION; the client's ``step`` returns
    the decoded EnvStep with all fields populated and types preserved.
    """
    with _make_transport_pair() as (client, _server, _recorded):
        client.connect()
        result = client.step(4)
    assert result.env_step == 1
    assert result.episode_id == 0
    assert result.step_in_episode == 1
    assert result.observation.shape == (32, 32)
    assert result.observation.dtype == np.uint8


def test_observations_over_wire_equal_direct_call_observations() -> None:
    """Determinism gate: a sequence of STEPs over the wire produces the
    exact same observations as the same actions called against a direct
    :class:`EnvServer`. No re-rendering, no resampling, no precision loss
    in the base64-encoded NumPy serialization."""
    actions = [0, 1, 2, 3, 4, 0, 1, 2, 3, 4]

    direct_config = EnvServerConfig(
        grid_world_config=GridWorldConfig(),
        seed=42,
        world_event_handler=lambda _record: None,
        run_id="direct-run",
    )
    direct_server = EnvServer(direct_config)
    direct_first = direct_server.start()
    direct_observations = [direct_first.observation.copy()]
    for action in actions:
        direct_observations.append(direct_server.step(action).observation.copy())
    direct_server.close()

    with _make_transport_pair(seed=42, run_id="wire-run") as (
        client,
        _server,
        _recorded,
    ):
        wire_first = client.connect()
        wire_observations = [wire_first.observation]
        for action in actions:
            wire_observations.append(client.step(action).observation)

    assert len(wire_observations) == len(direct_observations)
    for i, (w, d) in enumerate(zip(wire_observations, direct_observations)):
        assert np.array_equal(w, d), f"observation mismatch at index {i}"


def test_wallclock_in_transition_is_monotonically_non_decreasing() -> None:
    """Wallclock comes from the server-side env_server's monotonic clock
    (set when ``EnvServer.step`` ran). The client decodes the value as-is
    — no re-stamping. Verify monotonicity across a STEP sequence; this is
    the property the plan §2.8 wallclock-preservation requirement names.
    """
    with _make_transport_pair(grid_world_config=_make_quiet_config()) as (
        client,
        _server,
        _recorded,
    ):
        first = client.connect()
        wallclocks = [first.wallclock_ms]
        for _ in range(50):
            wallclocks.append(client.step(4).wallclock_ms)
    for prev, curr in zip(wallclocks, wallclocks[1:]):
        assert curr >= prev, f"wallclock went backwards: {prev} → {curr}"


def test_wire_observation_is_uint8_2d_and_read_only() -> None:
    """The decode pipeline (base64 → tobytes → frombuffer → reshape)
    yields a read-only ``uint8`` array matching the rendered shape."""
    with _make_transport_pair() as (client, _server, _recorded):
        first = client.connect()
    obs = first.observation
    assert obs.dtype == np.uint8
    assert obs.shape == (32, 32)
    assert not obs.flags.writeable


# ---- MUTATE / MUTATE_ACK roundtrip ----------------------------------------


def test_mutate_add_resource_changes_grid_and_emits_world_event() -> None:
    with _make_transport_pair(grid_world_config=_make_quiet_config()) as (
        client,
        server,
        recorded,
    ):
        client.connect()
        # Clear any env_reset from the recorded list so we can isolate
        # the builder-perturbation event below.
        recorded.clear()
        client.mutate("add_resource", cell=(2, 5))
        # The server's underlying grid should have been mutated. We must
        # read this *inside* the with block — the transport-pair fixture
        # closes the env-server on exit.
        assert (
            server._env_server.grid_world_state.grid[2, 5]  # type: ignore[attr-defined]
            == CellType.RESOURCE.value
        )

    # The WORLD_EVENT side-channel delivered the builder_perturbation.
    builder_events = [
        r for r in recorded if r.event_type == "builder_perturbation"
    ]
    assert len(builder_events) == 1
    payload = builder_events[0].payload
    assert payload["mutator"] == "add_resource"
    assert payload["cell"] == [2, 5]
    assert payload["pre_state"] == "empty"
    assert payload["post_state"] == "resource"


def test_mutate_remove_object_with_cell_type_arg() -> None:
    """``CellType`` arguments serialize as their lowercase name string and
    decode back to the enum on the server side."""
    with _make_transport_pair(grid_world_config=_make_quiet_config()) as (
        client,
        server,
        _recorded,
    ):
        client.connect()
        # Plant a resource directly on the server-side grid so the remove
        # has something to remove.
        server._env_server._grid_world._grid[1, 2] = (  # type: ignore[union-attr]
            CellType.RESOURCE.value
        )
        client.mutate(
            "remove_object", cell=(1, 2), object_type=CellType.RESOURCE
        )
        assert (
            server._env_server.grid_world_state.grid[1, 2]  # type: ignore[attr-defined]
            == CellType.EMPTY.value
        )


def test_mutate_set_cell_state_with_cell_type_arg() -> None:
    with _make_transport_pair(grid_world_config=_make_quiet_config()) as (
        client,
        server,
        _recorded,
    ):
        client.connect()
        client.mutate(
            "set_cell_state", cell=(4, 4), state=CellType.WALL
        )
        assert (
            server._env_server.grid_world_state.grid[4, 4]  # type: ignore[attr-defined]
            == CellType.WALL.value
        )


def test_mutate_move_object_with_two_tuple_args() -> None:
    with _make_transport_pair(grid_world_config=_make_quiet_config()) as (
        client,
        server,
        _recorded,
    ):
        client.connect()
        server._env_server._grid_world._grid[1, 1] = (  # type: ignore[union-attr]
            CellType.RESOURCE.value
        )
        client.mutate(
            "move_object", cell_from=(1, 1), cell_to=(2, 2)
        )
        assert (
            server._env_server.grid_world_state.grid[1, 1]  # type: ignore[attr-defined]
            == CellType.EMPTY.value
        )
        assert (
            server._env_server.grid_world_state.grid[2, 2]  # type: ignore[attr-defined]
            == CellType.RESOURCE.value
        )


def test_mutate_with_invalid_args_raises_mutate_error_via_ack() -> None:
    """A server-side ``ValueError`` (e.g. out-of-bounds cell) returns a
    ``MUTATE_ACK`` with ``ok=false`` and a populated ``error`` field; the
    client raises :class:`MutateError`."""
    with _make_transport_pair() as (client, _server, _recorded):
        client.connect()
        with pytest.raises(MutateError, match="out of grid bounds"):
            client.mutate("add_resource", cell=(99, 99))


def test_mutate_with_unknown_mutator_raises_mutate_error() -> None:
    with _make_transport_pair() as (client, _server, _recorded):
        client.connect()
        with pytest.raises(MutateError, match="unknown mutator"):
            client.mutate("not_a_real_mutator", cell=(0, 0))


def test_mutate_after_failure_does_not_break_subsequent_steps() -> None:
    """A failed mutate (server-side error) returns MUTATE_ACK; the
    connection stays open and subsequent STEPs work."""
    with _make_transport_pair() as (client, _server, _recorded):
        client.connect()
        with pytest.raises(MutateError):
            client.mutate("add_resource", cell=(99, 99))
        # Subsequent step should still work.
        result = client.step(4)
        assert result.env_step == 1


# ---- WORLD_EVENT side channel --------------------------------------------


def test_world_event_side_channel_delivers_episode_boundary_events() -> None:
    """Episode boundaries emit ``internal_stochasticity_aggregate`` and
    ``env_reset`` over the wire; the client's handler receives both."""
    config = GridWorldConfig(
        episode_length=5,
        initial_regrowth_p=0.0,
        drift_p_min=0.0,
        drift_p_max=0.0,
        drift_magnitude_per_step=0.0,
        n_initial_resources=0,
    )
    with _make_transport_pair(grid_world_config=config) as (
        client,
        _server,
        recorded,
    ):
        client.connect()
        # 5 steps → episode boundary
        for _ in range(5):
            client.step(4)
        # Tiny synchronization barrier so the last WORLD_EVENT (env_reset
        # for the new episode) has time to be delivered to the handler
        # before we shutdown. The server emits events synchronously
        # before sending the TRANSITION response, but the handler is
        # invoked on the client's reader thread and the test thread
        # might race ahead. Issue a no-op step to ensure the reader has
        # processed all the boundary events.
        client.step(4)

    types = [r.event_type for r in recorded]
    # Initial env_reset, then aggregate + env_reset at the boundary.
    assert "env_reset" in types
    assert "internal_stochasticity_aggregate" in types
    aggs = [r for r in recorded if r.event_type == "internal_stochasticity_aggregate"]
    assert len(aggs) == 1
    assert aggs[0].payload["episode_id"] == 0
    resets = [r for r in recorded if r.event_type == "env_reset"]
    # First reset: episode 0 (initial). Second reset: episode 1 (boundary).
    assert resets[0].payload["episode_id"] == 0
    assert resets[1].payload["episode_id"] == 1


def test_world_event_side_channel_delivers_mutator_event_before_step_response() -> None:
    """A WORLD_EVENT for a mutator is delivered to the client-side
    handler before the next STEP's TRANSITION arrives — the server emits
    it synchronously during the mutator's execution, before sending the
    MUTATE_ACK."""
    with _make_transport_pair(grid_world_config=_make_quiet_config()) as (
        client,
        _server,
        recorded,
    ):
        client.connect()
        recorded.clear()
        client.mutate("add_resource", cell=(2, 2))
        # By the time mutate() returns (i.e. MUTATE_ACK received), the
        # WORLD_EVENT has already been delivered.
        builder_events = [
            r for r in recorded if r.event_type == "builder_perturbation"
        ]
        assert len(builder_events) == 1


# ---- Barrier protocol ----------------------------------------------------


def test_barrier_begin_blocks_until_ack_received() -> None:
    with _make_transport_pair() as (client, _server, _recorded):
        client.connect()
        # If the ACK doesn't arrive, this would block forever. The
        # client's await uses the default timeout so the test would
        # raise TransportError instead of hanging.
        client.barrier_begin("ckpt-test-001")
        client.barrier_end("ckpt-test-001")


def test_barrier_queues_steps_until_barrier_end() -> None:
    """During a barrier, STEP messages are queued on the server. After
    BARRIER_END, the queue is drained and TRANSITIONs are sent."""
    with _make_transport_pair() as (client, _server, _recorded):
        client.connect()
        client.barrier_begin("ckpt-test-002")

        # Spawn a thread that calls client.step. The STEP message goes
        # over the wire; the server queues it; no TRANSITION comes back.
        # The thread blocks until BARRIER_END is sent.
        result_holder: list[object] = []

        def do_step() -> None:
            try:
                step = client.step(0)
                result_holder.append(step)
            except Exception as e:  # pragma: no cover - exercised on failure
                result_holder.append(e)

        thread = threading.Thread(target=do_step, daemon=True)
        thread.start()

        # Wait briefly to ensure the STEP has reached the server and
        # would have been processed if the barrier weren't in effect.
        time.sleep(0.2)
        assert not result_holder, (
            "STEP should not have produced a TRANSITION while the "
            "server is in barrier mode"
        )

        # End the barrier; the queued STEP is now processed.
        client.barrier_end("ckpt-test-002")
        thread.join(timeout=5.0)

        assert len(result_holder) == 1
        assert not isinstance(result_holder[0], Exception), (
            f"step thread raised: {result_holder[0]!r}"
        )


def test_barrier_queues_mutates_and_drains_in_order() -> None:
    """A MUTATE and a STEP sent during a barrier are queued server-side
    (not processed) and drain in arrival order on BARRIER_END.

    Deterministic rewrite of a quarantined flake (Probe 3 Phase 4,
    quarantined 2026-06-01; marker lifted 2026-06-11). The old version
    issued both requests from concurrent threads, which races on the
    client's shared response queue: two outstanding requests violate the
    client's single-outstanding-request contract (``_await_response`` pops
    the *next* message and requires a request_id match), so whichever
    thread woke first could pop the other's response — the ~1/3 flake. The
    queued/drains-in-order property is *server-side*, so this version sends
    the raw wire messages without awaiting (single thread, no sleeps as
    assertions), waits on the server's queue state directly, and asserts
    the drain *order* on the wire after BARRIER_END — strictly stronger
    than the original, which raced and could not assert order at all.
    """
    with _make_transport_pair(grid_world_config=_make_quiet_config()) as (
        client,
        server,
        _recorded,
    ):
        client.connect()
        client.barrier_begin("ckpt-test-003")

        # Send (without awaiting) a MUTATE then a STEP — the same wire
        # payloads client.mutate()/client.step() build.
        mutate_id = client._issue_request_id()
        client._send(
            {
                "type": MessageType.MUTATE.value,
                "mutator": "add_resource",
                "args": _encode_mutate_args({"cell": (1, 1)}),
                "request_id": mutate_id,
            }
        )
        step_id = client._issue_request_id()
        client._send(
            {
                "type": MessageType.STEP.value,
                "action": 4,
                "request_id": step_id,
            }
        )

        # Wait (bounded) for both to be *queued* server-side — polling for
        # the actual condition, not asserting after a fixed sleep.
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            if len(server._pending_msgs) == 2:
                break
            time.sleep(0.005)
        queued_types = [m.get("type") for m in server._pending_msgs]
        assert queued_types == [
            MessageType.MUTATE.value,
            MessageType.STEP.value,
        ], "both messages must be queued, in arrival order, during the barrier"
        # The server is sequential and queued both without processing them,
        # so no response can have been produced during the barrier.
        assert client._response_queue.empty(), (
            "no responses should have been received during the barrier"
        )

        client.barrier_end("ckpt-test-003")

        # Drain order on the wire: the MUTATE (queued first) is processed
        # first, then the STEP. Sequential pops from this single thread; a
        # wrong order surfaces as a request_id mismatch (TransportError).
        first = client._await_response(mutate_id)
        assert first.get("type") == MessageType.MUTATE_ACK.value
        assert first.get("ok") is True
        second = client._await_response(step_id)
        assert second.get("type") == MessageType.TRANSITION.value


def test_concurrent_request_raises_immediately_positive_control() -> None:
    """The one-outstanding-request contract is *enforced*: a second request
    issued while one is outstanding raises TransportError immediately
    (non-blocking), naming the contract — it does not block, and it does
    not silently mispair responses off the shared queue.

    Positive control for the request lock: a barrier holds a ``step()``
    outstanding deterministically (its TRANSITION cannot arrive until
    BARRIER_END), so the contended state is real, not timing-dependent.
    """
    with _make_transport_pair(grid_world_config=_make_quiet_config()) as (
        client,
        server,
        _recorded,
    ):
        client.connect()
        client.barrier_begin("ckpt-lock-001")

        step_error: list[Exception] = []

        def do_step() -> None:
            try:
                client.step(4)
            except Exception as e:  # pragma: no cover
                step_error.append(e)

        t = threading.Thread(target=do_step, daemon=True)
        t.start()
        # Wait until the STEP is provably queued server-side: the step()
        # call is now outstanding (holding the request lock, awaiting a
        # TRANSITION that cannot arrive during the barrier).
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            if len(server._pending_msgs) == 1:
                break
            time.sleep(0.005)
        assert len(server._pending_msgs) == 1

        # Contention: each user-facing request method refuses immediately.
        with pytest.raises(TransportError, match="one outstanding request"):
            client.mutate("add_resource", cell=(1, 1))
        with pytest.raises(TransportError, match="one outstanding request"):
            client.step(4)
        with pytest.raises(TransportError, match="one outstanding request"):
            client.barrier_begin("ckpt-lock-002")
        # The refused calls consumed nothing: the outstanding step is still
        # queued and unanswered.
        assert len(server._pending_msgs) == 1

        # Release the barrier; the outstanding step completes cleanly and
        # the lock cycles — a subsequent sequential request succeeds.
        client.barrier_end("ckpt-lock-001")
        t.join(timeout=5.0)
        assert not t.is_alive()
        assert not step_error
        client.step(4)


# ---- length-prefix robustness --------------------------------------------


def test_partial_message_does_not_deadlock_server() -> None:
    """Closing the connection mid-send leaves the server's reader loop
    cleanly aware of EOF. The server thread joins within a short timeout."""
    config = EnvServerConfig(
        grid_world_config=_make_quiet_config(),
        seed=42,
        world_event_handler=lambda _record: None,
        run_id="partial-test",
    )
    env_server = EnvServer(config)
    transport_server = EnvTransportServer(env_server, host="127.0.0.1", port=0)
    port = transport_server.actual_port
    server_thread = threading.Thread(
        target=transport_server.serve_forever, daemon=True
    )
    server_thread.start()

    raw_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        raw_sock.connect(("127.0.0.1", port))
        # Read the initial TRANSITION the server sends on connect, so we
        # know the server is in the reader loop expecting our messages.
        prefix = raw_sock.recv(4)
        assert len(prefix) == 4
        (length,) = struct.unpack(">I", prefix)
        body = b""
        while len(body) < length:
            chunk = raw_sock.recv(length - len(body))
            assert chunk
            body += chunk

        # Now send only 2 bytes of a 4-byte length prefix and immediately
        # close. The server's _recv_exactly loop returns None on the
        # short read (because close → EOF), and the reader loop exits.
        raw_sock.sendall(b"\x00\x00")
    finally:
        raw_sock.close()

    server_thread.join(timeout=5.0)
    assert not server_thread.is_alive(), (
        "server thread did not exit after partial-message close"
    )
    transport_server.shutdown()


def test_client_close_terminates_server_cleanly() -> None:
    """Normal client close: server's reader loop reads EOF and exits."""
    config = EnvServerConfig(
        grid_world_config=_make_quiet_config(),
        seed=42,
        world_event_handler=lambda _record: None,
        run_id="close-test",
    )
    env_server = EnvServer(config)
    transport_server = EnvTransportServer(env_server, host="127.0.0.1", port=0)
    port = transport_server.actual_port
    server_thread = threading.Thread(
        target=transport_server.serve_forever, daemon=True
    )
    server_thread.start()

    client = EnvTransportClient(
        host="127.0.0.1",
        port=port,
        world_event_handler=lambda _record: None,
    )
    client.connect()
    client.step(4)
    client.close()

    server_thread.join(timeout=5.0)
    assert not server_thread.is_alive()
    transport_server.shutdown()


# ---- stress / sequence ---------------------------------------------------


def test_many_steps_all_succeed_with_correct_request_ids() -> None:
    """Long sequence of STEPs: every TRANSITION has a request_id that
    matches its STEP, observations are in expected env_step order."""
    with _make_transport_pair(grid_world_config=_make_quiet_config()) as (
        client,
        _server,
        _recorded,
    ):
        client.connect()
        for i in range(100):
            step = client.step(i % NUM_ACTIONS)
            assert step.env_step == i + 1


def test_close_idempotent() -> None:
    with _make_transport_pair() as (client, _server, _recorded):
        client.connect()
        client.close()
        client.close()  # second call is a no-op


def test_step_after_close_raises() -> None:
    config = EnvServerConfig(
        grid_world_config=_make_quiet_config(),
        seed=42,
        world_event_handler=lambda _record: None,
        run_id="step-after-close",
    )
    env_server = EnvServer(config)
    transport_server = EnvTransportServer(env_server, host="127.0.0.1", port=0)
    port = transport_server.actual_port
    server_thread = threading.Thread(
        target=transport_server.serve_forever, daemon=True
    )
    server_thread.start()
    try:
        client = EnvTransportClient(
            host="127.0.0.1",
            port=port,
            world_event_handler=lambda _record: None,
        )
        client.connect()
        client.close()
        with pytest.raises(ConnectionLostError):
            client.step(0)
    finally:
        transport_server.shutdown()
        server_thread.join(timeout=5.0)


def test_message_type_enum_values_are_their_string_names() -> None:
    """Sanity check the wire-format constants stay aligned with the spec."""
    assert MessageType.STEP.value == "STEP"
    assert MessageType.TRANSITION.value == "TRANSITION"
    assert MessageType.MUTATE.value == "MUTATE"
    assert MessageType.MUTATE_ACK.value == "MUTATE_ACK"
    assert MessageType.BARRIER_BEGIN.value == "BARRIER_BEGIN"
    assert MessageType.BARRIER_BEGIN_ACK.value == "BARRIER_BEGIN_ACK"
    assert MessageType.BARRIER_END.value == "BARRIER_END"
    assert MessageType.WORLD_EVENT.value == "WORLD_EVENT"


# ---- file-system independence -------------------------------------------


def test_no_files_written_by_transport_layer(tmp_path: Path) -> None:
    """The transport layer is filesystem-free: WorldEvents only flow over
    the wire and into the client-side handler. Verify no files are
    written under ``tmp_path`` during a roundtrip."""
    initial_contents = sorted(tmp_path.iterdir())
    with _make_transport_pair() as (client, _server, _recorded):
        client.connect()
        client.step(4)
        client.mutate("add_resource", cell=(2, 2))
    assert sorted(tmp_path.iterdir()) == initial_contents
