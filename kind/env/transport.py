"""Phase 4 TCP transport for the env-server.

Wraps Phase 3a's :class:`~kind.env.env_server.EnvServer` with the network
protocol the implementation plan §2.8 and §8 specify: length-prefixed
framed JSON messages over TCP, a small message-type vocabulary, a
WorldEvent side channel multiplexed onto the same socket, and the
network half of the checkpoint barrier protocol. The atomic-rename half
of the barrier lives in :mod:`kind.training.checkpoint`.

**Wire format.** Every message is a 4-byte big-endian unsigned length
prefix followed by N bytes of UTF-8 JSON. The payload's ``type`` field
names the message; remaining fields are payload-specific. Per the plan,
JSON with base64-encoded NumPy serialization for the observation tensor
is sufficient at Probe 1 sizes (32×32 grayscale); msgpack with NumPy
support is a deferred optimization documented in the plan §8 and is
**not** wired here.

**Message types.**

* ``STEP`` (Mac → desktop): ``action: int``, ``request_id: int``.
* ``TRANSITION`` (desktop → Mac): an :class:`EnvStep` dict plus the
  matching ``request_id``. The initial ``TRANSITION`` produced when the
  server starts the env-server uses ``request_id=0``.
* ``MUTATE`` (Mac → desktop): ``mutator``, ``args``, ``request_id``.
* ``MUTATE_ACK`` (desktop → Mac): ``ok: bool``, ``request_id``, plus
  ``error: str`` when ``ok=False``.
* ``BARRIER_BEGIN`` (Mac → desktop): ``checkpoint_id``.
* ``BARRIER_BEGIN_ACK`` (desktop → Mac): ``checkpoint_id``. Added per
  the user spec so the client knows the server has stopped processing
  ``STEP`` and ``MUTATE`` and the trainer can safely commit the
  checkpoint.
* ``BARRIER_END`` (Mac → desktop): ``checkpoint_id``. No ACK; the server
  resumes immediately.
* ``WORLD_EVENT`` (desktop → Mac): ``record`` (a :class:`WorldEvent`
  dict). Multiplexed on the same socket; the client routes these to its
  configured ``world_event_handler``.

**Threading.** Each side runs a single reader thread on the connected
socket. The server's reader thread also writes (responses and side-
channel ``WORLD_EVENT`` messages); the client's reader thread routes
incoming messages either to a synchronous response queue (TRANSITION,
MUTATE_ACK, BARRIER_BEGIN_ACK) or to the configured
``world_event_handler`` (WORLD_EVENT). Sends from user code on the client
side are guarded by a lock; sends from the server's reader thread are
inherently serialized because the reader is the only sender. Probe 1 has
one trainer connection per env-server, no concurrent senders, no
backpressure beyond TCP's built-in windowing.

**Barrier flow control.** When the server reads ``BARRIER_BEGIN`` it sets
an in-barrier flag and ACKs immediately (the server is sequential, so
"drain in-flight" is a no-op — the previous ``STEP`` finished before the
next message was read). While the flag is set, ``STEP`` and ``MUTATE``
messages are queued in arrival order. ``BARRIER_END`` clears the flag and
drains the queue. The client's ``barrier_begin`` blocks on the
``BARRIER_BEGIN_ACK`` so the trainer can commit the checkpoint with the
guarantee that no further ``TRANSITION`` will arrive until ``barrier_end``
is sent.

**No reconnection, no SSL, no auth.** Probe 1 runs both ends on the Mac
via loopback; the smoke is short-lived. If the connection drops, the test
fails and the human investigates. Production reliability is a Phase 8+
concern.
"""

from __future__ import annotations

import base64
import json
import socket
import struct
import threading
from collections import deque
from collections.abc import Callable
from enum import Enum
from queue import Empty, Queue
from types import TracebackType
from typing import Any, Final, Self

import numpy as np
from numpy.typing import NDArray

from kind.env.env_server import EnvServer
from kind.env.grid_world import CellType, EnvStep
from kind.observer.schemas import WorldEvent

__all__ = [
    "ConnectionLostError",
    "EnvTransportClient",
    "EnvTransportServer",
    "MessageType",
    "MutateError",
    "TransportError",
    "WorldEventHandler",
]


# ---- module-level constants ----------------------------------------------


_LENGTH_PREFIX_FORMAT: Final[str] = ">I"
_LENGTH_PREFIX_SIZE: Final[int] = 4
_INITIAL_REQUEST_ID: Final[int] = 0  # sentinel for the initial TRANSITION
_DEFAULT_RESPONSE_TIMEOUT_SEC: Final[float] = 30.0
_VALID_MUTATORS: Final[frozenset[str]] = frozenset(
    {"add_resource", "remove_object", "set_cell_state", "move_object"}
)
_EOF_SENTINEL: Final[dict[str, Any]] = {"type": "__EOF__"}


WorldEventHandler = Callable[[WorldEvent], None]


# ---- exceptions ----------------------------------------------------------


class TransportError(Exception):
    """Base class for transport-layer errors."""


class ConnectionLostError(TransportError):
    """Raised when the connection is closed while an operation is in flight."""


class MutateError(TransportError):
    """Raised on the client side when a server-side mutator returned an error."""


# ---- message type --------------------------------------------------------


class MessageType(str, Enum):
    """Wire-protocol message type. Subclasses :class:`str` so values are
    JSON-serializable directly via ``msg["type"] = MessageType.STEP.value``
    and comparable to the strings on the receive side without conversion.
    """

    STEP = "STEP"
    TRANSITION = "TRANSITION"
    MUTATE = "MUTATE"
    MUTATE_ACK = "MUTATE_ACK"
    BARRIER_BEGIN = "BARRIER_BEGIN"
    BARRIER_BEGIN_ACK = "BARRIER_BEGIN_ACK"
    BARRIER_END = "BARRIER_END"
    WORLD_EVENT = "WORLD_EVENT"


# ---- framing helpers -----------------------------------------------------


def _recv_exactly(sock: socket.socket, n: int) -> bytes | None:
    """Read exactly ``n`` bytes from ``sock``; return None on EOF/error.

    Wraps :meth:`socket.socket.recv` in a loop because TCP may deliver
    fewer bytes than requested. Returns ``None`` if the peer closes the
    connection before ``n`` bytes have been read or if an OS-level error
    interrupts the read.
    """
    buf = bytearray()
    while len(buf) < n:
        try:
            chunk = sock.recv(n - len(buf))
        except OSError:
            return None
        if not chunk:
            return None
        buf.extend(chunk)
    return bytes(buf)


def _recv_message(sock: socket.socket) -> dict[str, Any] | None:
    """Read one length-prefixed JSON message; return None on clean EOF."""
    prefix = _recv_exactly(sock, _LENGTH_PREFIX_SIZE)
    if prefix is None:
        return None
    (length,) = struct.unpack(_LENGTH_PREFIX_FORMAT, prefix)
    body = _recv_exactly(sock, length)
    if body is None:
        return None
    decoded = json.loads(body.decode("utf-8"))
    if not isinstance(decoded, dict):
        raise TransportError(
            f"expected a JSON object as message payload, got {type(decoded).__name__}"
        )
    return decoded


def _send_message(sock: socket.socket, msg: dict[str, Any]) -> None:
    """Serialize ``msg`` to JSON, prefix with length, send on ``sock``."""
    body = json.dumps(msg, separators=(",", ":")).encode("utf-8")
    prefix = struct.pack(_LENGTH_PREFIX_FORMAT, len(body))
    sock.sendall(prefix + body)


# ---- observation encoding ------------------------------------------------


def _encode_observation(obs: NDArray[np.uint8]) -> dict[str, Any]:
    """Pack a NumPy observation tensor into a JSON-friendly dict.

    ``shape`` is a list of ints, ``dtype`` is a NumPy dtype string, and
    ``data`` is base64-encoded raw bytes (``ndarray.tobytes()``). This is
    the per-message overhead the plan §8 accepts as adequate at 32×32
    grayscale sizes; msgpack's binary path is the documented optimization.
    """
    return {
        "shape": list(obs.shape),
        "dtype": str(obs.dtype),
        "data": base64.b64encode(obs.tobytes()).decode("ascii"),
    }


def _decode_observation(d: dict[str, Any]) -> NDArray[np.uint8]:
    """Reconstruct a read-only observation tensor from its encoded dict.

    :func:`np.frombuffer` returns a read-only array; ``reshape`` preserves
    that flag. Matching the contract in :class:`~kind.env.grid_world.GridWorld`,
    where the server-side observation is rendered with ``write=False``.
    """
    raw = base64.b64decode(d["data"])
    dtype = np.dtype(d["dtype"])
    arr: NDArray[Any] = np.frombuffer(raw, dtype=dtype).reshape(d["shape"])
    return arr


# ---- env step encoding ---------------------------------------------------


def _encode_env_step(step: EnvStep) -> dict[str, Any]:
    return {
        "observation": _encode_observation(step.observation),
        "env_step": step.env_step,
        "episode_id": step.episode_id,
        "step_in_episode": step.step_in_episode,
        "wallclock_ms": step.wallclock_ms,
        "sensed_energy": step.sensed_energy,
    }


def _decode_env_step(d: dict[str, Any]) -> EnvStep:
    return EnvStep(
        observation=_decode_observation(d["observation"]),
        env_step=int(d["env_step"]),
        episode_id=int(d["episode_id"]),
        step_in_episode=int(d["step_in_episode"]),
        wallclock_ms=int(d["wallclock_ms"]),
        # Probe 3.5: backward-tolerant — pre-3.5 wire messages have no
        # ``sensed_energy`` field; default to 0.0 so old shards/replays decode.
        sensed_energy=float(d.get("sensed_energy", 0.0)),
    )


# ---- mutator argument encoding/decoding ----------------------------------


def _name_to_cell_type(name: str) -> CellType:
    return CellType[name.upper()]


def _encode_mutate_args(kwargs: dict[str, Any]) -> dict[str, Any]:
    """Convert Python kwargs into a JSON-friendly dict.

    Tuples become lists; :class:`CellType` instances become their lowercase
    name strings; everything else passes through. The mutator-specific
    decoder on the server side reverses this per known mutator name.
    """
    out: dict[str, Any] = {}
    for k, v in kwargs.items():
        if isinstance(v, tuple):
            out[k] = list(v)
        elif isinstance(v, CellType):
            out[k] = v.name.lower()
        else:
            out[k] = v
    return out


def _decode_mutate_args(mutator: str, args: dict[str, Any]) -> dict[str, Any]:
    """Reverse :func:`_encode_mutate_args` per known mutator.

    Validation of out-of-bounds cells, invalid object types, etc. is
    deferred to :mod:`kind.env.mutators`; this decoder only restores the
    Python-side types (tuples, :class:`CellType`) that the harness's
    mutator methods expect.
    """
    if mutator == "add_resource":
        return {"cell": tuple(args["cell"])}
    if mutator == "remove_object":
        return {
            "cell": tuple(args["cell"]),
            "object_type": _name_to_cell_type(args["object_type"]),
        }
    if mutator == "set_cell_state":
        return {
            "cell": tuple(args["cell"]),
            "state": _name_to_cell_type(args["state"]),
        }
    if mutator == "move_object":
        return {
            "cell_from": tuple(args["cell_from"]),
            "cell_to": tuple(args["cell_to"]),
        }
    raise ValueError(f"unknown mutator: {mutator!r}")


# ---- world event encoding ------------------------------------------------


def _encode_world_event(record: WorldEvent) -> dict[str, Any]:
    return record.model_dump(mode="json")


def _decode_world_event(payload: dict[str, Any]) -> WorldEvent:
    return WorldEvent.model_validate(payload)


# ---- server side ---------------------------------------------------------


class EnvTransportServer:
    """TCP server that wraps an :class:`EnvServer` and exposes it over the wire.

    Single-connection-per-instance. ``__init__`` binds the listen socket
    so :attr:`actual_port` is queryable before :meth:`serve_forever`;
    :meth:`serve_forever` accepts one connection, takes over the
    env-server's WorldEvent handler, calls
    :meth:`~kind.env.env_server.EnvServer.start`, sends the initial
    ``TRANSITION`` (with ``request_id=0``), then runs the reader loop
    until the client disconnects or :meth:`shutdown` is called.

    The server-side reader thread is also the sender thread; sends are
    therefore serialized without explicit locking. The lock that exists
    is a defensive measure for the case where the WorldEvent handler is
    invoked from a thread other than the reader (which Probe 1 does not
    do, but the cost is negligible).
    """

    def __init__(
        self,
        env_server: EnvServer,
        host: str = "127.0.0.1",
        port: int = 0,
    ) -> None:
        self._env_server = env_server
        self._host = host
        listen_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        listen_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        listen_sock.bind((host, port))
        listen_sock.listen(1)
        self._listen_sock: socket.socket | None = listen_sock
        self._actual_port: int = listen_sock.getsockname()[1]

        self._client_sock: socket.socket | None = None
        self._send_lock = threading.Lock()
        self._shutdown_event = threading.Event()

        self._in_barrier: bool = False
        self._pending_msgs: deque[dict[str, Any]] = deque()

    @property
    def actual_port(self) -> int:
        """Port the OS bound (useful when ``port=0`` was requested)."""
        return self._actual_port

    # ---- lifecycle -------------------------------------------------------

    def serve_forever(self) -> None:
        """Accept one client connection and process messages until disconnect.

        Closes the listen socket as soon as the single connection is
        accepted. On disconnect, closes the env-server and the client
        socket. Returns to the caller cleanly so a thread joined on this
        method terminates without raising.
        """
        listen_sock = self._listen_sock
        if listen_sock is None:
            return
        try:
            client_sock, _addr = listen_sock.accept()
        except OSError:
            # Listen socket closed during accept (shutdown) or other error.
            self._close_listen_socket()
            return
        # Single connection only — close the listen socket immediately.
        self._close_listen_socket()

        self._client_sock = client_sock
        try:
            self._serve_connection(client_sock)
        finally:
            try:
                client_sock.close()
            except OSError:
                pass
            self._client_sock = None
            try:
                self._env_server.close()
            except Exception:
                # The env-server's close() is idempotent and pure-state at
                # Phase 4; an exception here is unexpected. Swallow so the
                # server thread exits cleanly regardless.
                pass

    def shutdown(self) -> None:
        """Signal shutdown: close sockets so blocked I/O wakes up.

        Safe to call from any thread, including before
        :meth:`serve_forever` has returned. Idempotent.
        """
        self._shutdown_event.set()
        self._close_listen_socket()
        if self._client_sock is not None:
            try:
                self._client_sock.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass

    def _close_listen_socket(self) -> None:
        listen_sock = self._listen_sock
        if listen_sock is None:
            return
        try:
            listen_sock.close()
        except OSError:
            pass
        self._listen_sock = None

    # ---- per-connection handling ----------------------------------------

    def _serve_connection(self, sock: socket.socket) -> None:
        # Wire the env-server's WorldEvent handler to send over the wire.
        # The harness's start() will fire env_reset which goes out as a
        # WORLD_EVENT message *before* the initial TRANSITION; the client
        # side's reader thread receives WORLD_EVENT first and routes it
        # to its own handler, then receives TRANSITION and unblocks
        # connect().
        self._env_server.set_world_event_handler(self._send_world_event)
        try:
            initial_step = self._env_server.start()
        except RuntimeError:
            # If start() was already called (the user hand-started the
            # env-server), we cannot proceed — there is no initial EnvStep
            # to send. Close the connection and let the client see EOF.
            return
        self._send_transition(initial_step, request_id=_INITIAL_REQUEST_ID)
        self._reader_loop(sock)

    def _reader_loop(self, sock: socket.socket) -> None:
        while not self._shutdown_event.is_set():
            msg = _recv_message(sock)
            if msg is None:
                return
            try:
                self._dispatch(msg)
            except (TransportError, ValueError, TypeError):
                # A malformed message or a dispatch-side error closes the
                # connection. The client will see EOF on its next read.
                return

    def _dispatch(self, msg: dict[str, Any]) -> None:
        msg_type = msg.get("type")
        # Queue STEP and MUTATE during a barrier; everything else is
        # processed inline (BARRIER_END is the only message that drains
        # the queue).
        if (
            self._in_barrier
            and msg_type
            in (MessageType.STEP.value, MessageType.MUTATE.value)
        ):
            self._pending_msgs.append(msg)
            return
        if msg_type == MessageType.STEP.value:
            self._process_step(msg)
        elif msg_type == MessageType.MUTATE.value:
            self._process_mutate(msg)
        elif msg_type == MessageType.BARRIER_BEGIN.value:
            self._process_barrier_begin(msg)
        elif msg_type == MessageType.BARRIER_END.value:
            self._process_barrier_end(msg)
        else:
            raise TransportError(f"unexpected message type: {msg_type!r}")

    # ---- per-message handlers -------------------------------------------

    def _process_step(self, msg: dict[str, Any]) -> None:
        action = int(msg["action"])
        request_id = int(msg["request_id"])
        env_step = self._env_server.step(action)
        self._send_transition(env_step, request_id=request_id)

    def _process_mutate(self, msg: dict[str, Any]) -> None:
        request_id = int(msg["request_id"])
        mutator = str(msg["mutator"])
        try:
            if mutator not in _VALID_MUTATORS:
                raise ValueError(f"unknown mutator: {mutator!r}")
            decoded_args = _decode_mutate_args(mutator, msg.get("args") or {})
            method = getattr(self._env_server, mutator)
            method(**decoded_args)
        except (ValueError, TypeError, KeyError) as exc:
            self._send(
                {
                    "type": MessageType.MUTATE_ACK.value,
                    "ok": False,
                    "error": str(exc),
                    "request_id": request_id,
                }
            )
            return
        self._send(
            {
                "type": MessageType.MUTATE_ACK.value,
                "ok": True,
                "request_id": request_id,
            }
        )

    def _process_barrier_begin(self, msg: dict[str, Any]) -> None:
        checkpoint_id = str(msg["checkpoint_id"])
        # Probe 1's env-server is sequential — no in-flight STEP exists at
        # this point because the previous reader-loop iteration finished
        # the previous STEP before reading the next message. The flag plus
        # ACK are still both required by the protocol.
        self._in_barrier = True
        self._send(
            {
                "type": MessageType.BARRIER_BEGIN_ACK.value,
                "checkpoint_id": checkpoint_id,
            }
        )

    def _process_barrier_end(self, msg: dict[str, Any]) -> None:
        self._in_barrier = False
        # Drain queued messages in arrival order. STEP and MUTATE share
        # one deque so a queued MUTATE that arrived before a STEP is
        # processed before the STEP — preserving the trainer's ordering.
        while self._pending_msgs:
            queued = self._pending_msgs.popleft()
            qtype = queued.get("type")
            if qtype == MessageType.STEP.value:
                self._process_step(queued)
            elif qtype == MessageType.MUTATE.value:
                self._process_mutate(queued)

    # ---- send helpers ---------------------------------------------------

    def _send_transition(self, env_step: EnvStep, *, request_id: int) -> None:
        self._send(
            {
                "type": MessageType.TRANSITION.value,
                "request_id": request_id,
                **_encode_env_step(env_step),
            }
        )

    def _send_world_event(self, record: WorldEvent) -> None:
        self._send(
            {
                "type": MessageType.WORLD_EVENT.value,
                "record": _encode_world_event(record),
            }
        )

    def _send(self, msg: dict[str, Any]) -> None:
        sock = self._client_sock
        if sock is None:
            return
        with self._send_lock:
            try:
                _send_message(sock, msg)
            except OSError:
                # Connection lost mid-send. The reader loop will see EOF
                # on its next read and exit; nothing to do here.
                pass


# ---- client side ---------------------------------------------------------


class EnvTransportClient:
    """TCP client to an :class:`EnvTransportServer`.

    :meth:`connect` opens the socket, starts the reader thread, and
    returns the initial :class:`EnvStep` that the server sends as the
    handshake. :meth:`step`, :meth:`mutate`, :meth:`barrier_begin`, and
    :meth:`barrier_end` each send a single message and (where applicable)
    wait for its response. The reader thread routes incoming messages:
    ``WORLD_EVENT`` to the configured ``world_event_handler``, everything
    else to the response queue consumed by the synchronous user-facing
    methods.

    The client is single-threaded from the user's point of view: **only one
    request may be outstanding at a time**, and this contract is *enforced* —
    :meth:`step`, :meth:`mutate`, and :meth:`barrier_begin` hold a
    non-blocking request lock across their send+await, and a second caller
    arriving while a request is outstanding gets an immediate
    :class:`TransportError` naming the contract (rather than silently
    mispairing responses off the shared queue, which is what concurrent
    awaiters would otherwise do). Issue MUTATEs from the same loop that
    issues STEPs, or serialize externally.
    """

    def __init__(
        self,
        host: str,
        port: int,
        world_event_handler: WorldEventHandler,
    ) -> None:
        self._host = host
        self._port = port
        self._world_event_handler = world_event_handler
        self._sock: socket.socket | None = None
        self._reader_thread: threading.Thread | None = None
        self._response_queue: Queue[dict[str, Any]] = Queue()
        self._next_request_id: int = _INITIAL_REQUEST_ID + 1
        self._connected: bool = False
        self._closed: bool = False
        self._send_lock = threading.Lock()
        # Enforces the one-outstanding-request contract (class docstring):
        # held non-blockingly across send+await by step/mutate/barrier_begin.
        self._request_lock = threading.Lock()

    # ---- lifecycle -------------------------------------------------------

    def connect(self) -> EnvStep:
        """Open the connection, await the handshake, return the initial step.

        The server side calls ``EnvServer.start()`` upon accepting the
        connection; this emits the ``env_reset`` WorldEvent (delivered to
        the configured ``world_event_handler`` synchronously from the
        client's reader thread) followed by the initial ``TRANSITION``
        (the message this method waits for and returns).
        """
        if self._connected:
            raise RuntimeError("EnvTransportClient.connect() already called")
        if self._closed:
            raise RuntimeError("EnvTransportClient is closed")
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.connect((self._host, self._port))
        self._sock = sock
        self._reader_thread = threading.Thread(
            target=self._reader_loop,
            name="EnvTransportClientReader",
            daemon=True,
        )
        self._reader_thread.start()
        self._connected = True
        msg = self._await_response(_INITIAL_REQUEST_ID)
        if msg.get("type") != MessageType.TRANSITION.value:
            raise TransportError(
                f"expected initial TRANSITION, got {msg.get('type')!r}"
            )
        return _decode_env_step(msg)

    def _acquire_request_lock_or_raise(self, method: str) -> None:
        """Acquire the request lock without blocking, or raise immediately.

        Enforces the one-outstanding-request contract: a concurrent caller
        fails loudly here rather than racing the first caller on the shared
        response queue and mispairing responses (a silent-corruption mode —
        see the barrier-queue test's history).
        """
        if not self._request_lock.acquire(blocking=False):
            raise TransportError(
                f"concurrent {method}(): EnvTransportClient supports one "
                "outstanding request at a time (send+await is exclusive) — "
                "issue MUTATEs from the same loop that issues STEPs, or "
                "serialize externally"
            )

    def step(self, action: int) -> EnvStep:
        """Send STEP, wait for matching TRANSITION, return the EnvStep."""
        self._acquire_request_lock_or_raise("step")
        try:
            request_id = self._issue_request_id()
            self._send(
                {
                    "type": MessageType.STEP.value,
                    "action": int(action),
                    "request_id": request_id,
                }
            )
            msg = self._await_response(request_id)
        finally:
            self._request_lock.release()
        if msg.get("type") != MessageType.TRANSITION.value:
            raise TransportError(
                f"expected TRANSITION for request {request_id}, got "
                f"{msg.get('type')!r}"
            )
        return _decode_env_step(msg)

    def mutate(self, mutator: str, **kwargs: Any) -> None:
        """Send MUTATE, wait for MUTATE_ACK; raise on ``ok=False``.

        Args are encoded by :func:`_encode_mutate_args` and decoded on the
        server side per known mutator. Validation errors raised by the
        underlying :mod:`kind.env.mutators` functions propagate as
        :class:`MutateError` with the original error message.
        """
        self._acquire_request_lock_or_raise("mutate")
        try:
            request_id = self._issue_request_id()
            self._send(
                {
                    "type": MessageType.MUTATE.value,
                    "mutator": mutator,
                    "args": _encode_mutate_args(kwargs),
                    "request_id": request_id,
                }
            )
            msg = self._await_response(request_id)
        finally:
            self._request_lock.release()
        if msg.get("type") != MessageType.MUTATE_ACK.value:
            raise TransportError(
                f"expected MUTATE_ACK for request {request_id}, got "
                f"{msg.get('type')!r}"
            )
        if not msg.get("ok"):
            raise MutateError(str(msg.get("error", "unknown mutate error")))

    def barrier_begin(self, checkpoint_id: str) -> None:
        """Send BARRIER_BEGIN; block until BARRIER_BEGIN_ACK arrives.

        After this method returns, the server has stopped processing
        ``STEP`` and ``MUTATE`` messages. The trainer can commit the
        checkpoint, then call :meth:`barrier_end` to resume the server.
        """
        self._acquire_request_lock_or_raise("barrier_begin")
        try:
            self._send(
                {
                    "type": MessageType.BARRIER_BEGIN.value,
                    "checkpoint_id": checkpoint_id,
                }
            )
            msg = self._await_response(request_id=None)
        finally:
            self._request_lock.release()
        if msg.get("type") != MessageType.BARRIER_BEGIN_ACK.value:
            raise TransportError(
                f"expected BARRIER_BEGIN_ACK, got {msg.get('type')!r}"
            )
        if msg.get("checkpoint_id") != checkpoint_id:
            raise TransportError(
                f"BARRIER_BEGIN_ACK checkpoint_id mismatch: expected "
                f"{checkpoint_id!r}, got {msg.get('checkpoint_id')!r}"
            )

    def barrier_end(self, checkpoint_id: str) -> None:
        """Send BARRIER_END; no acknowledgement is awaited.

        The server resumes processing immediately on receipt and drains
        any STEPs/MUTATEs queued during the barrier, sending their
        responses as they complete.
        """
        self._send(
            {
                "type": MessageType.BARRIER_END.value,
                "checkpoint_id": checkpoint_id,
            }
        )

    def close(self) -> None:
        """Close the connection. Idempotent.

        Shuts down the socket, which causes the reader thread to see EOF
        and exit. Joins the reader thread with a timeout so a broken peer
        does not block the caller indefinitely.
        """
        if self._closed:
            return
        self._closed = True
        sock = self._sock
        if sock is not None:
            try:
                sock.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            try:
                sock.close()
            except OSError:
                pass
            self._sock = None
        thread = self._reader_thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=2.0)

    def set_world_event_handler(self, handler: WorldEventHandler) -> None:
        """Replace the WorldEvent handler.

        Symmetric with :meth:`~kind.env.env_server.EnvServer.set_world_event_handler`.
        The handler is read by the reader thread on each ``WORLD_EVENT``
        message; rebinding via a single attribute assignment is
        thread-safe under CPython. Phase 5's runner uses this to redirect
        WorldEvent records to its own ``JsonlSink`` after both the
        client and the runner have been constructed (the client's
        constructor takes a handler but the runner's sink only exists
        after :meth:`Runner.__init__`).
        """
        self._world_event_handler = handler

    # ---- context manager -------------------------------------------------

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()

    # ---- internals -------------------------------------------------------

    def _reader_loop(self) -> None:
        sock = self._sock
        if sock is None:
            return
        try:
            while True:
                msg = _recv_message(sock)
                if msg is None:
                    break
                msg_type = msg.get("type")
                if msg_type == MessageType.WORLD_EVENT.value:
                    record = _decode_world_event(msg.get("record") or {})
                    self._world_event_handler(record)
                else:
                    self._response_queue.put(msg)
        finally:
            # Wake any pending await with an EOF sentinel so callers raise
            # ConnectionLostError rather than hang on queue.get().
            self._response_queue.put(_EOF_SENTINEL)

    def _issue_request_id(self) -> int:
        rid = self._next_request_id
        self._next_request_id += 1
        return rid

    def _send(self, msg: dict[str, Any]) -> None:
        sock = self._sock
        if sock is None:
            raise ConnectionLostError("client is not connected")
        with self._send_lock:
            try:
                _send_message(sock, msg)
            except OSError as e:
                raise ConnectionLostError(str(e)) from e

    def _await_response(
        self, request_id: int | None
    ) -> dict[str, Any]:
        """Pop the next response off the queue and verify request_id matches.

        ``request_id=None`` accepts any response (used for the
        ``BARRIER_BEGIN_ACK`` which has no request_id field). Otherwise the
        response's ``request_id`` must equal the argument; a mismatch
        raises :class:`TransportError`. The EOF sentinel raises
        :class:`ConnectionLostError`.
        """
        try:
            msg = self._response_queue.get(timeout=_DEFAULT_RESPONSE_TIMEOUT_SEC)
        except Empty as e:
            raise TransportError(
                f"timeout waiting for response (request_id={request_id})"
            ) from e
        if msg is _EOF_SENTINEL or msg.get("type") == "__EOF__":
            raise ConnectionLostError("connection lost while awaiting response")
        if request_id is not None and msg.get("request_id") != request_id:
            raise TransportError(
                f"response request_id mismatch: expected {request_id}, "
                f"got {msg.get('request_id')!r}"
            )
        return msg
