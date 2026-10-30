"""A small MCP (Model Context Protocol) client: stdio and streamable HTTP (2026-10-07, capabilities).

The transport half of the "MCP servers" switch. :class:`McpServer` connects to ONE configured
server, lists its tools and calls them; ``exocortex.capabilities`` owns the policy around it (the
ask box only, ``mcp:<server>:<tool>`` naming, sanitising every description and result, the
per-answer budget) and imports this module lazily. Qt-free and model-free: stdlib + ``httpx``.

A server spec is one entry of ``[capabilities].mcp``::

    {name, transport: "stdio" | "http", command, args, url, env, enabled}

``name`` must match ``[A-Za-z0-9_.-]{1,64}`` (the runner splits ``mcp:<server>:<tool>`` on the first
two colons). A bad spec raises :class:`McpError` from the constructor, so a caller builds servers
inside its ``try``.

**stdio.** ``command`` is resolved with ``shutil.which`` (so ``npx`` finds ``npx.cmd`` on Windows)
and started WITHOUT a shell, with a MINIMAL environment: only ``PATH SystemRoot TEMP TMP
USERPROFILE HOME APPDATA LOCALAPPDATA PATHEXT COMSPEC HOMEDRIVE HOMEPATH`` from the parent, plus
the spec's own ``env`` -- an API key in ExoCortex's environment never reaches a third-party server
by accident. Messages are newline-delimited JSON-RPC 2.0 (UTF-8); a daemon reader thread caps one
line at 1 MiB (a longer one marks the connection ``error: oversized message`` and kills it) and
hands each reply to the request waiting on its integer id. ``close()`` ends the server's stdin,
waits 2 s, then kills the whole process TREE (``taskkill /T /F`` on Windows, ``killpg`` elsewhere):
an ``npx`` launcher's real server is a grandchild.

**http (streamable HTTP).** Each JSON-RPC message is a POST to ``url`` with ``Accept:
application/json, text/event-stream``; the reply is a JSON body or an SSE stream read until the
message whose ``id`` matches. The ``Mcp-Session-Id`` the server hands out on ``initialize`` is
echoed on every later request and released with a best-effort ``DELETE`` on ``close()``. The POST
runs on a short-lived worker thread so the caller can honour ``timeout_s`` and poll ``cancel``
every 100 ms like the stdio path does.

Server-initiated requests (sampling, roots, elicitation) are answered with a JSON-RPC "method not
supported" error -- this client offers no client capabilities -- and server notifications are
ignored. Every failure is an :class:`McpError` with a readable message; ``state`` is ``"idle"``,
``"connecting"``, ``"connected"`` or ``"error: <why>"``. A server's replies are untrusted: the
caller sanitises them.

**Liveness.** ``state`` is kept truthful without anyone calling: a stdio server whose process exits
(a watcher thread waits on it) or whose output reaches end-of-file becomes ``error: exited (code
N)`` at once, and an http server that answers HTTP 5xx twice in a row becomes ``error: HTTP 503
twice in a row``. From then on ``tools()`` and ``call()`` raise :class:`McpError` (they need
``state == "connected"``) and ``connect()`` starts over. Reading ``state`` is an attribute read.
"""
from __future__ import annotations

import itertools
import json
import logging
import os
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import httpx

from exocortex import __version__

log = logging.getLogger(__name__)

#: The protocol revision this client speaks (sent in ``initialize``).
PROTOCOL_VERSION = "2025-06-18"
#: One JSON-RPC message (a stdio line, an HTTP body, one SSE event) may be at most this long.
MAX_MESSAGE_BYTES = 1024 * 1024
#: A tool's description is cut to this many characters.
DESCRIPTION_CHARS = 160

_NAME_RE = re.compile(r"[A-Za-z0-9_.-]{1,64}")
_ENV_KEEP = frozenset(k.upper() for k in (
    "PATH", "SystemRoot", "SYSTEMROOT", "TEMP", "TMP", "USERPROFILE", "HOME", "APPDATA",
    "LOCALAPPDATA", "PATHEXT", "COMSPEC", "HOMEDRIVE", "HOMEPATH"))
_POLL_S = 0.1                 # how often a wait looks at ``cancel``
_CLOSE_GRACE_S = 2.0          # close(): wait this long after ending stdin before killing the tree
_MAX_LIST_PAGES = 20
_MAX_TOOLS = 500
_METHOD_NOT_FOUND = -32601
_WINDOWS = sys.platform == "win32"
_clock = time.monotonic


class McpError(Exception):
    """A readable failure of an MCP connection or call."""


@dataclass(frozen=True)
class McpTool:
    """One tool a server offers: its (validated) ``name``, a ``description`` cut to 160 characters
    and its JSON-Schema ``input_schema`` exactly as the server sent it."""

    name: str
    description: str
    input_schema: dict[str, Any] = field(default_factory=dict)


class _Slot:
    """The place a reply is parked until the thread waiting for it picks it up."""

    __slots__ = ("event", "message", "error")

    def __init__(self) -> None:
        self.event = threading.Event()
        self.message: dict[str, Any] | None = None
        self.error: str | None = None


def _valid_name(name: object) -> bool:
    return isinstance(name, str) and _NAME_RE.fullmatch(name) is not None


def _result_text(result: Any) -> tuple[str, bool]:
    """A ``tools/call`` result -> (text, is_error). Text items are joined by newlines; other kinds
    are noted as ``[image]`` / ``[resource: uri]`` (an embedded text resource keeps its text)."""
    if not isinstance(result, dict):
        return "", False
    parts: list[str] = []
    content = result.get("content")
    for item in content if isinstance(content, list) else []:
        if not isinstance(item, dict):
            continue
        kind = item.get("type")
        if kind == "text":
            parts.append(str(item.get("text", "")))
        elif kind == "resource":
            res = item.get("resource") if isinstance(item.get("resource"), dict) else {}
            parts.append(f"[resource: {res.get('uri', '')}]")
            if isinstance(res.get("text"), str):
                parts.append(res["text"])
        elif kind == "resource_link":
            parts.append(f"[resource: {item.get('uri', '')}]")
        else:
            parts.append(f"[{kind if isinstance(kind, str) and kind else 'content'}]")
    if not parts and result.get("structuredContent") is not None:
        parts.append(json.dumps(result["structuredContent"], ensure_ascii=False))
    return "\n".join(parts), result.get("isError") is True


class McpServer:
    """One MCP server: ``connect()``, ``tools()``, ``call()``, ``close()`` (module docstring).

    ``timeout_s`` bounds every wait (the handshake, a listing, a call); ``env_base`` is the parent
    environment the minimal stdio environment is filtered from (default ``os.environ``);
    ``transport`` lets tests substitute ``httpx.MockTransport`` for the http kind. Thread-safe:
    several calls may be in flight at once.
    """

    def __init__(self, spec: dict[str, Any], *, timeout_s: float, env_base: dict[str, str] | None = None,
                 transport: httpx.BaseTransport | None = None) -> None:
        if not isinstance(spec, dict):
            raise McpError("an MCP server entry must be a table")
        name = spec.get("name")
        if not _valid_name(name):
            raise McpError(f"invalid MCP server name {str(name)[:40]!r}: use 1-64 of A-Z a-z 0-9 _ . -")
        self.name: str = name  # type: ignore[assignment]  # narrowed by _valid_name
        kind = str(spec.get("transport") or ("http" if spec.get("url") and not spec.get("command") else "stdio"))
        kind = kind.strip().lower()
        if kind not in ("stdio", "http"):
            raise McpError(f"MCP server {name}: unknown transport {kind!r} (use stdio or http)")
        self._kind = kind
        self._command = ""
        self._args: list[str] = []
        self._url = ""
        if kind == "stdio":
            command = spec.get("command")
            if not isinstance(command, str) or not command.strip():
                raise McpError(f"MCP server {name}: a stdio server needs a command")
            args = spec.get("args") or []
            if not isinstance(args, (list, tuple)):
                raise McpError(f"MCP server {name}: args must be a list")
            self._command = command.strip()
            self._args = [str(a) for a in args]
        else:
            url = spec.get("url")
            try:
                parsed = httpx.URL(str(url)) if url else None
            except (httpx.InvalidURL, ValueError) as exc:
                raise McpError(f"MCP server {name}: invalid url ({exc})") from exc
            if parsed is None or parsed.scheme not in ("http", "https") or not parsed.host:
                raise McpError(f"MCP server {name}: an http server needs an http(s) url")
            self._url = str(parsed)
        env = spec.get("env") or {}
        if not isinstance(env, dict):
            raise McpError(f"MCP server {name}: env must be a table of KEY = value")
        self._env = {str(k): str(v) for k, v in env.items()}
        self._timeout_s = max(0.1, float(timeout_s))
        self._env_base = env_base
        self._transport = transport
        self._lock = threading.RLock()
        self._write_lock = threading.Lock()
        self._state = "idle"
        self._closing = False
        self._ids = itertools.count(1)
        self._pending: dict[Any, _Slot] = {}
        self._proc: subprocess.Popen[bytes] | None = None
        self._reader: threading.Thread | None = None
        self._client: httpx.Client | None = None
        self._session_id = ""
        self._protocol = ""
        self._http_5xx = 0                   # consecutive HTTP 5xx answers (http transport)

    # ------------------------------------------------------------------------------ public API
    @property
    def state(self) -> str:
        """``"idle"`` | ``"connecting"`` | ``"connected"`` | ``"error: <why>"``."""
        return self._state

    def connect(self) -> None:
        """Start (or reach) the server and run the ``initialize`` handshake. Blocking. Raises
        :class:`McpError`; the state then reads ``error: <why>``. Calling it again after a
        failure starts over."""
        with self._lock:
            if self._state == "connected":
                return
            if self._state == "connecting":
                raise McpError(f"{self.name} is already connecting")
        self._teardown()
        with self._lock:
            self._state = "connecting"
            self._closing = False
            self._protocol = ""
            self._http_5xx = 0
        try:
            if self._kind == "stdio":
                self._spawn()
            else:
                self._open_http()
            init = self._request("initialize", {
                "protocolVersion": PROTOCOL_VERSION,
                "capabilities": {},
                "clientInfo": {"name": "exocortex", "version": __version__},
            })
            # The revision the server settled on is the one later HTTP requests announce.
            agreed = init.get("protocolVersion") if isinstance(init, dict) else None
            self._protocol = agreed if isinstance(agreed, str) and agreed else PROTOCOL_VERSION
            self._notify("notifications/initialized")
        except McpError as exc:
            self._connect_failed(str(exc))
            raise
        except Exception as exc:  # noqa: BLE001 - a connect must end as an McpError, never a traceback
            reason = f"{type(exc).__name__}: {exc}"
            self._connect_failed(reason)
            raise McpError(reason) from exc
        with self._lock:
            if self._closing:
                raise McpError("closed while connecting")
            if self._kind == "stdio" and self._proc is None:
                # The process died right after the handshake; _lost() has recorded why.
                why = self._state[len("error: "):] if self._state.startswith("error: ") else "not connected"
                raise McpError(why)
            self._state = "connected"
        log.info("MCP server %s connected (%s)", self.name, self._kind)

    def tools(self) -> list[McpTool]:
        """The server's tools (every ``tools/list`` page). Names outside ``[A-Za-z0-9_.-]{1,64}`` are
        dropped with a warning; descriptions are cut to 160 characters."""
        self._require_connected()
        found: list[McpTool] = []
        seen: set[str] = set()
        cursor = ""
        for _ in range(_MAX_LIST_PAGES):
            result = self._request("tools/list", {"cursor": cursor} if cursor else {})
            items = result.get("tools") if isinstance(result, dict) else None
            for item in items if isinstance(items, list) else []:
                if not isinstance(item, dict):
                    continue
                name = item.get("name")
                if not _valid_name(name):
                    log.warning("MCP server %s: dropping a tool with an invalid name %r", self.name, str(name)[:60])
                    continue
                if name in seen or len(found) >= _MAX_TOOLS:
                    continue
                seen.add(name)
                desc = item.get("description")
                schema = item.get("inputSchema")
                found.append(McpTool(
                    name=name,
                    description=" ".join(desc.split())[:DESCRIPTION_CHARS] if isinstance(desc, str) else "",
                    input_schema=schema if isinstance(schema, dict) else {}))
            nxt = result.get("nextCursor") if isinstance(result, dict) else None
            if not isinstance(nxt, str) or not nxt or nxt == cursor:
                break
            cursor = nxt
        return found

    def call(self, name: str, args: dict[str, Any] | None = None, *, cancel: threading.Event | None = None) -> str:
        """Call tool ``name`` and return its text (see :func:`_result_text`). A result flagged
        ``isError`` or a JSON-RPC error raises :class:`McpError` with the server's message; so do
        the timeout and a set ``cancel`` (the connection stays usable)."""
        self._require_connected()
        if args is None:
            args = {}
        if not isinstance(args, dict):
            raise McpError("tool arguments must be an object")
        result = self._request("tools/call", {"name": name, "arguments": args}, cancel=cancel)
        text, is_error = _result_text(result)
        if is_error:
            raise McpError(text or "the tool reported an error")
        return text

    def close(self) -> None:
        """End the connection: stdio -> close stdin, wait 2 s, kill the process tree; http ->
        ``DELETE`` the session (best effort). Idempotent; never raises; the state becomes ``idle``."""
        with self._lock:
            self._closing = True
            proc, self._proc = self._proc, None
            client, self._client = self._client, None
            session, self._session_id = self._session_id, ""
            protocol = self._protocol
            pending = list(self._pending.values())
            self._pending.clear()
            self._state = "idle"
        for slot in pending:
            slot.error = "closed"
            slot.event.set()
        if proc is not None:
            self._stop_process(proc, graceful=True)
            reader = self._reader
            if reader is not None and reader is not threading.current_thread():
                reader.join(1.0)
            self._reader = None
        if client is not None:
            if session:
                headers = {"Mcp-Session-Id": session}
                if protocol:
                    headers["MCP-Protocol-Version"] = protocol
                try:
                    client.delete(self._url, headers=headers, timeout=2.0)
                except Exception:  # noqa: BLE001 - the server may not support DELETE; best effort
                    log.debug("MCP server %s: session DELETE failed", self.name, exc_info=True)
            try:
                client.close()
            except Exception:  # noqa: BLE001
                log.debug("MCP server %s: closing the http client failed", self.name, exc_info=True)

    # -------------------------------------------------------------------------- state plumbing
    def _require_connected(self) -> None:
        if self._state != "connected":
            raise McpError(f"MCP server {self.name} is not connected ({self._state})")

    def _connect_failed(self, reason: str) -> None:
        self._teardown()
        with self._lock:
            self._state = "idle" if self._closing else f"error: {reason}"
        log.warning("MCP server %s could not connect: %s", self.name, reason)

    def _teardown(self) -> None:
        """Drop whatever connection exists, at once (no grace period)."""
        with self._lock:
            proc, self._proc = self._proc, None
            client, self._client = self._client, None
            self._session_id = ""
            pending = list(self._pending.values())
            self._pending.clear()
        for slot in pending:
            slot.error = "connection dropped"
            slot.event.set()
        if proc is not None:
            self._stop_process(proc, graceful=False)
        if client is not None:
            try:
                client.close()
            except Exception:  # noqa: BLE001
                log.debug("MCP server %s: closing the http client failed", self.name, exc_info=True)

    # ------------------------------------------------------------------------- JSON-RPC layer
    def _request(self, method: str, params: dict[str, Any] | None, *,
                 cancel: threading.Event | None = None) -> Any:
        """Send one request and wait for its reply -> the ``result``; a JSON-RPC ``error`` raises."""
        if cancel is not None and cancel.is_set():
            raise McpError("cancelled")
        rid = next(self._ids)
        message: dict[str, Any] = {"jsonrpc": "2.0", "id": rid, "method": method}
        if params is not None:
            message["params"] = params
        slot = _Slot()
        if self._kind == "stdio":
            with self._lock:
                proc = self._proc
                if proc is None:
                    raise McpError("not connected")
                self._pending[rid] = slot
            try:
                self._send(proc, message)
                reply = self._wait(slot, cancel, lambda why: self._send_cancelled(proc, rid, why))
            finally:
                with self._lock:
                    self._pending.pop(rid, None)
        else:
            abort = threading.Event()
            with self._lock:
                self._pending[rid] = slot             # so close() can fail a call that is still in flight
            threading.Thread(target=self._http_worker, args=(message, slot, abort), daemon=True,
                             name=f"mcp-http-{self.name}").start()
            try:
                reply = self._wait(slot, cancel, lambda why: abort.set())
            finally:
                with self._lock:
                    self._pending.pop(rid, None)
        error = reply.get("error")
        if error is not None:
            text = error.get("message") if isinstance(error, dict) else None
            raise McpError(str(text or error)[:500])
        result = reply.get("result")
        return result if result is not None else {}

    def _wait(self, slot: _Slot, cancel: threading.Event | None, on_abort: Callable[[str], None]) -> dict[str, Any]:
        """Block until ``slot`` is filled, polling ``cancel`` every 100 ms and honouring ``timeout_s``."""
        deadline = _clock() + self._timeout_s
        while not slot.event.wait(_POLL_S):
            if cancel is not None and cancel.is_set():
                on_abort("cancelled")
                raise McpError("cancelled")
            if _clock() >= deadline:
                on_abort("timed out")
                raise McpError(f"timed out after {self._timeout_s:g} s")
        if slot.error is not None:
            raise McpError(slot.error)
        return slot.message or {}

    def _notify(self, method: str, params: dict[str, Any] | None = None) -> None:
        message: dict[str, Any] = {"jsonrpc": "2.0", "method": method}
        if params is not None:
            message["params"] = params
        if self._kind == "stdio":
            proc = self._proc
            if proc is None:
                raise McpError("not connected")
            self._send(proc, message)
        else:
            client = self._client
            if client is None:
                raise McpError("not connected")
            try:
                resp = client.post(self._url, content=json.dumps(message, separators=(",", ":")).encode("utf-8"),
                                   headers=self._http_headers())
            except httpx.TimeoutException as exc:
                raise McpError(f"timed out after {self._timeout_s:g} s") from exc
            except httpx.HTTPError as exc:
                raise McpError(f"request failed: {str(exc) or type(exc).__name__}") from exc
            self._note_http_status(resp.status_code)
            if resp.status_code >= 300:
                raise McpError(f"HTTP {resp.status_code} on {method}")

    # ------------------------------------------------------------------------------------ stdio
    def _spawn(self) -> None:
        """Start the server process with the minimal environment and its reader thread."""
        base = os.environ if self._env_base is None else self._env_base
        env = {str(k): str(v) for k, v in base.items() if str(k).upper() in _ENV_KEEP}
        env.update(self._env)
        path = next((v for k, v in env.items() if k.upper() == "PATH"), None)
        exe = shutil.which(self._command, path=path)
        if exe is None:
            raise McpError(f"command not found: {self._command}")
        kwargs: dict[str, Any] = {}
        if _WINDOWS:
            kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW | subprocess.CREATE_NEW_PROCESS_GROUP
        else:
            kwargs["start_new_session"] = True      # its own process group: close() can killpg it
        try:
            proc = subprocess.Popen([exe, *self._args], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                    stderr=subprocess.DEVNULL, env=env, text=False, **kwargs)
        except OSError as exc:
            raise McpError(f"could not start {self._command}: {exc}") from exc
        with self._lock:
            self._proc = proc
        reader = threading.Thread(target=self._read_loop, args=(proc,), daemon=True, name=f"mcp-{self.name}-reader")
        self._reader = reader
        reader.start()
        # A launcher can exit while a child it started still holds the output pipe open, so the
        # reader never sees end-of-file: this thread notices the exit itself.
        threading.Thread(target=self._watch_exit, args=(proc,), daemon=True, name=f"mcp-{self.name}-exit").start()

    def _send(self, proc: subprocess.Popen[bytes], message: dict[str, Any]) -> None:
        data = json.dumps(message, separators=(",", ":")).encode("utf-8") + b"\n"
        with self._write_lock:
            try:
                assert proc.stdin is not None
                proc.stdin.write(data)
                proc.stdin.flush()
            except (OSError, ValueError) as exc:
                raise McpError("could not write to the server (has it exited?)") from exc

    def _send_cancelled(self, proc: subprocess.Popen[bytes], rid: int, why: str) -> None:
        """Tell the server we gave up on request ``rid`` (best effort)."""
        try:
            self._send(proc, {"jsonrpc": "2.0", "method": "notifications/cancelled",
                              "params": {"requestId": rid, "reason": why}})
        except McpError:
            log.debug("MCP server %s: could not send notifications/cancelled", self.name)

    def _read_loop(self, proc: subprocess.Popen[bytes]) -> None:
        """Reader thread: one JSON-RPC message per line, replies handed to their waiting request."""
        stream = proc.stdout
        assert stream is not None
        try:
            while True:
                try:
                    line = stream.readline(MAX_MESSAGE_BYTES + 2)
                except (OSError, ValueError):
                    line = b""
                if not line:
                    self._lost(proc, self._exit_reason(proc))
                    return
                content = line.rstrip(b"\r\n")
                if len(content) > MAX_MESSAGE_BYTES or (len(line) >= MAX_MESSAGE_BYTES + 2 and not line.endswith(b"\n")):
                    self._lost(proc, "oversized message")
                    return
                if not content.strip():
                    continue
                try:
                    message = json.loads(content)
                except ValueError:
                    log.debug("MCP server %s: ignoring a non-JSON line on stdout", self.name)
                    continue
                self._dispatch(proc, message)
        finally:
            try:
                stream.close()
            except Exception:  # noqa: BLE001
                pass

    def _dispatch(self, proc: subprocess.Popen[bytes], message: Any) -> None:
        if isinstance(message, list):
            for item in message:
                self._dispatch(proc, item)
            return
        if not isinstance(message, dict):
            return
        if "method" in message:
            if "id" in message:                       # a request FROM the server
                try:
                    self._send(proc, self._refusal(message))
                except McpError:
                    log.debug("MCP server %s: could not answer a server request", self.name)
            return                                    # a notification: ignored
        mid = message.get("id")
        if isinstance(mid, (int, str)):
            with self._lock:
                slot = self._pending.get(mid)
            if slot is not None:
                slot.message = message
                slot.event.set()

    @staticmethod
    def _refusal(request: dict[str, Any]) -> dict[str, Any]:
        """The answer to a server-initiated request: ``ping`` is fine, everything else unsupported."""
        if request.get("method") == "ping":
            return {"jsonrpc": "2.0", "id": request["id"], "result": {}}
        return {"jsonrpc": "2.0", "id": request["id"],
                "error": {"code": _METHOD_NOT_FOUND, "message": "method not supported"}}

    @staticmethod
    def _exit_reason(proc: subprocess.Popen[bytes]) -> str:
        """Why a server whose output ended is gone: ``exited (code N)`` once the process has an exit
        code (it gets half a second to produce one), else ``exited (output closed)``."""
        try:
            return f"exited (code {proc.wait(timeout=0.5)})"
        except subprocess.TimeoutExpired:
            return "exited (output closed)"

    def _watch_exit(self, proc: subprocess.Popen[bytes]) -> None:
        """Watcher thread: the moment ``proc`` exits, the connection is lost (a no-op after a close)."""
        try:
            code = proc.wait()
        except Exception:  # noqa: BLE001 - a watcher must never raise
            log.debug("MCP server %s: waiting for the process failed", self.name, exc_info=True)
            return
        self._lost(proc, f"exited (code {code})")

    def _lost(self, proc: subprocess.Popen[bytes], reason: str) -> None:
        """The connection to ``proc`` is gone (EOF, exit) or unusable: fail every waiter, record the
        state, kill it. Only the first report counts (the reader and the watcher both make one)."""
        with self._lock:
            if self._proc is not proc:
                return                                # already lost, replaced or closed on purpose
            self._proc = None
            self._state = "idle" if self._closing else f"error: {reason}"
            pending = list(self._pending.values())
            self._pending.clear()
        for slot in pending:
            slot.error = reason
            slot.event.set()
        log.warning("MCP server %s: %s", self.name, reason)
        self._stop_process(proc, graceful=False)

    @staticmethod
    def _stop_process(proc: subprocess.Popen[bytes], *, graceful: bool) -> None:
        """Stop ``proc`` and everything it started. ``graceful``: end its stdin and give it 2 s first."""
        if graceful:
            try:
                if proc.stdin is not None:
                    proc.stdin.close()
            except (OSError, ValueError):
                pass
            try:
                proc.wait(timeout=_CLOSE_GRACE_S)
            except subprocess.TimeoutExpired:
                pass
        try:
            if _WINDOWS:
                if proc.poll() is None:
                    system32 = os.path.join(os.environ.get("SystemRoot", r"C:\Windows"), "System32", "taskkill.exe")
                    subprocess.run([system32 if os.path.exists(system32) else "taskkill", "/T", "/F", "/PID", str(proc.pid)],
                                   stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                                   timeout=5, creationflags=subprocess.CREATE_NO_WINDOW, check=False)
            else:
                os.killpg(proc.pid, signal.SIGKILL)
        except (OSError, subprocess.SubprocessError):
            log.debug("killing the process tree of pid %s failed", proc.pid, exc_info=True)
        try:
            proc.kill()
        except OSError:
            pass
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            log.warning("MCP server process %s did not exit after being killed", proc.pid)
        try:
            if proc.stdin is not None:
                proc.stdin.close()                    # the reader thread closes stdout when it sees EOF
        except (OSError, ValueError):
            pass

    # ------------------------------------------------------------------------------------ http
    def _open_http(self) -> None:
        with self._lock:
            if self._client is None:
                self._client = httpx.Client(timeout=self._timeout_s, transport=self._transport)
            self._session_id = ""

    def _note_http_status(self, status: int) -> None:
        """Track consecutive 5xx answers: the second in a row marks a connected server as down (a
        single one is a failed call, not a dead server)."""
        with self._lock:
            if status < 500:
                self._http_5xx = 0
                return
            self._http_5xx += 1
            if self._http_5xx >= 2 and self._state == "connected" and not self._closing:
                self._state = f"error: HTTP {status} twice in a row"
                log.warning("MCP server %s: %s", self.name, self._state[len("error: "):])

    def _http_headers(self) -> dict[str, str]:
        headers = {"Accept": "application/json, text/event-stream", "Content-Type": "application/json"}
        if self._session_id:
            headers["Mcp-Session-Id"] = self._session_id
        if self._protocol:
            headers["MCP-Protocol-Version"] = self._protocol
        return headers

    def _http_worker(self, message: dict[str, Any], slot: _Slot, abort: threading.Event) -> None:
        """Run one POST exchange on its own thread and park the reply (or the failure) in ``slot``.
        A reason ``close()`` already gave the slot ("closed") is kept: it is the true one."""
        why: str | None = None
        try:
            slot.message = self._http_exchange(message, abort)
        except McpError as exc:
            why = str(exc)
        except httpx.TimeoutException:
            why = f"timed out after {self._timeout_s:g} s"
        except httpx.HTTPError as exc:
            why = f"request failed: {str(exc) or type(exc).__name__}"
        except Exception as exc:  # noqa: BLE001 - never leak a traceback; the waiter is blocked on this slot
            log.debug("MCP server %s: http worker failed", self.name, exc_info=True)
            why = f"{type(exc).__name__}: {exc}"
        finally:
            if why is not None and slot.error is None:
                slot.error = why
            slot.event.set()

    def _http_exchange(self, message: dict[str, Any], abort: threading.Event) -> dict[str, Any]:
        client = self._client
        if client is None:
            raise McpError("not connected")
        rid = message["id"]
        body = json.dumps(message, separators=(",", ":")).encode("utf-8")
        with client.stream("POST", self._url, content=body, headers=self._http_headers()) as resp:
            sid = resp.headers.get("mcp-session-id")
            if sid and not self._session_id:
                self._session_id = sid
            status = resp.status_code
            self._note_http_status(status)
            if status >= 400:
                detail = " ".join(self._read_body(resp, abort, limit=400).decode("utf-8", "replace").split())[:200]
                raise McpError(f"HTTP {status}" + (f": {detail}" if detail else ""))
            if status >= 300:
                raise McpError(f"unexpected HTTP {status} (redirects are not followed)")
            media = resp.headers.get("content-type", "").partition(";")[0].strip().lower()
            if media == "text/event-stream":
                return self._read_sse(client, resp, rid, abort)
            raw = self._read_body(resp, abort, limit=MAX_MESSAGE_BYTES)
        if not raw.strip():
            raise McpError(f"the server sent no reply (HTTP {status})")
        try:
            data = json.loads(raw)
        except ValueError as exc:
            raise McpError("the server's reply is not JSON") from exc
        found = self._find_reply(data, rid)
        if found is None:
            raise McpError("the server's reply does not answer the request")
        return found

    @staticmethod
    def _read_body(resp: httpx.Response, abort: threading.Event, *, limit: int) -> bytes:
        body = bytearray()
        for chunk in resp.iter_bytes():
            if abort.is_set():
                raise McpError("cancelled")
            body += chunk
            if len(body) > limit:
                if limit < MAX_MESSAGE_BYTES:
                    return bytes(body[:limit])        # an error page: the start is enough
                raise McpError("oversized message")
        return bytes(body)

    @staticmethod
    def _find_reply(data: Any, rid: Any) -> dict[str, Any] | None:
        for item in data if isinstance(data, list) else [data]:
            if isinstance(item, dict) and item.get("id") == rid and ("result" in item or "error" in item):
                return item
        return None

    def _read_sse(self, client: httpx.Client, resp: httpx.Response, rid: Any, abort: threading.Event) -> dict[str, Any]:
        """Read ``text/event-stream`` events until the one that answers request ``rid``."""
        data: list[str] = []
        size = 0
        for line in resp.iter_lines():
            if abort.is_set():
                raise McpError("cancelled")
            if line == "":
                reply = self._sse_event(client, data, rid)
                data, size = [], 0
                if reply is not None:
                    return reply
                continue
            if line.startswith(":"):
                continue
            name, _, value = line.partition(":")
            if name == "data":
                size += len(value)
                if size > MAX_MESSAGE_BYTES:
                    raise McpError("oversized message")
                data.append(value[1:] if value.startswith(" ") else value)
        reply = self._sse_event(client, data, rid)          # the stream may end without a closing blank line
        if reply is not None:
            return reply
        raise McpError("the server ended its reply without answering")

    def _sse_event(self, client: httpx.Client, data: list[str], rid: Any) -> dict[str, Any] | None:
        if not data:
            return None
        try:
            payload = json.loads("\n".join(data))
        except ValueError:
            log.debug("MCP server %s: ignoring a non-JSON SSE event", self.name)
            return None
        reply = self._find_reply(payload, rid)
        if reply is not None:
            return reply
        for item in payload if isinstance(payload, list) else [payload]:
            if isinstance(item, dict) and "method" in item and "id" in item:     # a request from the server
                try:
                    client.post(self._url, content=json.dumps(self._refusal(item), separators=(",", ":")).encode("utf-8"),
                                headers=self._http_headers())
                except httpx.HTTPError:
                    log.debug("MCP server %s: could not answer a server request", self.name)
        return None
