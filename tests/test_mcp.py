"""Unit tests for exocortex.mcp (2026-10-07, capabilities), all offline.

stdio is exercised against a tiny MCP server this file writes into ``tmp_path`` and launches with
``sys.executable`` (real pipes, a real process, a real process tree); streamable HTTP against
``httpx.MockTransport``. They pin the handshake, the listing, calls, the error paths, cancel and
timeout, the minimal environment, the 1 MiB line cap, that ``state`` follows a server that dies or
keeps failing (without anyone calling) and that ``close()`` leaves nothing running."""
from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import threading
import time
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import httpx
import pytest

import exocortex
from exocortex.mcp import McpError, McpServer, McpTool

SERVER_SCRIPT = r'''
import json, os, subprocess, sys, time

SEEN = []
INIT = {}
PAGE1 = [
    {"name": "echo", "description": "Echo the arguments back.\n   Second   line.",
     "inputSchema": {"type": "object", "properties": {"text": {"type": "string"}}}},
    {"name": "fail", "description": "Always reports an error.", "inputSchema": {"type": "object"}},
    {"name": "slow", "description": "Sleeps.", "inputSchema": {"type": "object"}},
    {"name": "big", "description": "Writes a 2 MiB line.", "inputSchema": {"type": "object"}},
]
PAGE2 = [
    {"name": "bad name!", "description": "dropped: the name is not allowed"},
    {"name": "long_desc", "description": "x" * 300},
    {"name": "no_schema"},
]


def send(obj):
    sys.stdout.write(json.dumps(obj) + "\n")
    sys.stdout.flush()


def main():
    while True:
        line = sys.stdin.readline()
        if not line:
            break
        msg = json.loads(line)
        method, mid, params = msg.get("method"), msg.get("id"), msg.get("params") or {}
        SEEN.append(method)
        if method == "initialize":
            INIT.update(params)
            send({"jsonrpc": "2.0", "id": mid, "result": {"protocolVersion": "2025-06-18",
                  "capabilities": {"tools": {}}, "serverInfo": {"name": "fake", "version": "1"}}})
        elif method == "tools/list":
            if params.get("cursor") == "p2":
                send({"jsonrpc": "2.0", "id": mid, "result": {"tools": PAGE2}})
            else:
                send({"jsonrpc": "2.0", "id": mid, "result": {"tools": PAGE1, "nextCursor": "p2"}})
        elif method == "tools/call":
            call(mid, params.get("name"), params.get("arguments") or {})
        elif mid is not None:
            send({"jsonrpc": "2.0", "id": mid, "error": {"code": -32601, "message": "no such method"}})
    if os.environ.get("EXO_HANG_ON_EOF"):
        time.sleep(120)


def call(mid, name, args):
    def text(t, error=False):
        send({"jsonrpc": "2.0", "id": mid, "result": {"content": [{"type": "text", "text": t}], "isError": error}})

    if name == "echo":
        text(json.dumps(args, sort_keys=True))
    elif name == "fail":
        text("boom: " + str(args.get("why", "")), True)
    elif name == "slow":
        time.sleep(float(args.get("seconds", 30)))
        text("done")
    elif name == "big":
        sys.stdout.write("x" * (2 * 1024 * 1024) + "\n")
        sys.stdout.flush()
    elif name == "env":
        text(json.dumps({"keys": sorted(os.environ), "marker": os.environ.get("EXO_MARKER")}))
    elif name == "spawn":
        p = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(120)"],
                             stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        text(str(p.pid))
    elif name == "sampling":
        send({"jsonrpc": "2.0", "id": "srv-1", "method": "sampling/createMessage", "params": {}})
        text(sys.stdin.readline().strip())
    elif name == "ping":
        send({"jsonrpc": "2.0", "id": "srv-2", "method": "ping"})
        text(sys.stdin.readline().strip())
    elif name == "mixed":
        send({"jsonrpc": "2.0", "id": mid, "result": {"content": [
            {"type": "text", "text": "line one"},
            {"type": "image", "data": "AAAA", "mimeType": "image/png"},
            {"type": "resource", "resource": {"uri": "file:///notes.txt", "text": "embedded text"}},
            {"type": "resource_link", "uri": "file:///linked.txt"},
            {"type": "text", "text": "line two"},
            {"type": "audio", "data": "AAAA", "mimeType": "audio/wav"},
        ]}})
    elif name == "rpc_error":
        send({"jsonrpc": "2.0", "id": mid, "error": {"code": -32602, "message": "Unknown tool: nope"}})
    elif name == "inspect":
        text(json.dumps({"seen": SEEN, "init": INIT}))
    elif name == "noise":
        sys.stdout.write("this is a log line, not JSON\n")
        sys.stdout.flush()
        text("after noise")
    elif name == "exit":
        os._exit(3)
    elif name == "die_soon":
        import threading
        threading.Timer(0.3, lambda: os._exit(4)).start()
        text("bye")
    else:
        send({"jsonrpc": "2.0", "id": mid, "error": {"code": -32602, "message": "Unknown tool: " + str(name)}})


main()
'''

# Answers the handshake, then exits (code 5) while a child it started still holds the output pipe
# open: the reader never sees end-of-file, only the exit of the process says the server is gone.
LAUNCHER_SCRIPT = r'''
import json, os, subprocess, sys

while True:
    line = sys.stdin.readline()
    if not line:
        break
    msg = json.loads(line)
    if msg.get("method") == "initialize":
        sys.stdout.write(json.dumps({"jsonrpc": "2.0", "id": msg["id"], "result": {
            "protocolVersion": "2025-06-18", "capabilities": {"tools": {}}}}) + "\n")
        sys.stdout.flush()
    elif msg.get("method") == "notifications/initialized":
        child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(6)"],
                                 stdin=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        with open(sys.argv[1], "w") as fh:
            fh.write(str(child.pid))
        os._exit(5)
'''


# ---------------------------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------------------------
@pytest.fixture
def server_script(tmp_path: Path) -> Path:
    path = tmp_path / "fake_mcp_server.py"
    path.write_text(SERVER_SCRIPT, encoding="utf-8")
    return path


def stdio_spec(script: Path, **extra: Any) -> dict[str, Any]:
    spec: dict[str, Any] = {"name": "fake", "transport": "stdio", "command": sys.executable, "args": [str(script)]}
    spec.update(extra)
    return spec


@pytest.fixture
def make_server(server_script: Path) -> Iterator[Callable[..., McpServer]]:
    made: list[McpServer] = []

    def make(*, connect: bool = True, timeout_s: float = 20.0, env_base: dict[str, str] | None = None,
             **spec_extra: Any) -> McpServer:
        srv = McpServer(stdio_spec(server_script, **spec_extra), timeout_s=timeout_s, env_base=env_base)
        made.append(srv)
        if connect:
            srv.connect()
        return srv

    yield make
    for srv in made:
        srv.close()


def wait_until(predicate: Callable[[], bool], timeout: float = 8.0) -> bool:
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if predicate():
            return True
        time.sleep(0.05)
    return predicate()


def pid_alive(pid: int) -> bool:
    if sys.platform == "win32":
        out = subprocess.run(["tasklist", "/FI", f"PID eq {pid}", "/NH", "/FO", "CSV"],
                             capture_output=True, text=True, check=False).stdout
        return f'"{pid}"' in out
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


# ---------------------------------------------------------------------------------------------
# spec validation
# ---------------------------------------------------------------------------------------------
@pytest.mark.parametrize("name", ["", "bad name", "a" * 65, "x/y", "a:b", "tool!", "café", "new\nline", None, 7])
def test_an_invalid_server_name_is_refused(name: Any) -> None:
    with pytest.raises(McpError, match="invalid MCP server name"):
        McpServer({"name": name, "transport": "stdio", "command": "python"}, timeout_s=5)


@pytest.mark.parametrize("name", ["fs", "my-server", "my_server.v2", "A" * 64, "0"])
def test_a_valid_server_name_is_accepted(name: str) -> None:
    srv = McpServer({"name": name, "transport": "stdio", "command": "python"}, timeout_s=5)
    assert srv.name == name and srv.state == "idle"


@pytest.mark.parametrize("spec, message", [
    ({"name": "a", "transport": "carrier-pigeon", "command": "x"}, "unknown transport"),
    ({"name": "a", "transport": "stdio"}, "needs a command"),
    ({"name": "a", "transport": "stdio", "command": "  "}, "needs a command"),
    ({"name": "a", "transport": "stdio", "command": "x", "args": "-y pkg"}, "args must be a list"),
    ({"name": "a", "transport": "stdio", "command": "x", "env": ["A=1"]}, "env must be a table"),
    ({"name": "a", "transport": "http"}, "needs an http"),
    ({"name": "a", "transport": "http", "url": "ftp://x.org/mcp"}, "needs an http"),
    ({"name": "a", "transport": "http", "url": "http://"}, "needs an http"),
])
def test_a_broken_spec_is_refused(spec: dict[str, Any], message: str) -> None:
    with pytest.raises(McpError, match=message):
        McpServer(spec, timeout_s=5)


def test_a_non_table_spec_is_refused() -> None:
    with pytest.raises(McpError):
        McpServer("fs", timeout_s=5)  # type: ignore[arg-type]


def test_the_transport_is_inferred_when_missing() -> None:
    assert McpServer({"name": "a", "url": "http://localhost:9/mcp"}, timeout_s=5)._kind == "http"
    assert McpServer({"name": "a", "command": "x"}, timeout_s=5)._kind == "stdio"


def test_call_and_tools_need_a_connection() -> None:
    srv = McpServer({"name": "a", "transport": "stdio", "command": "python"}, timeout_s=5)
    with pytest.raises(McpError, match="not connected"):
        srv.tools()
    with pytest.raises(McpError, match="not connected"):
        srv.call("echo", {})


# ---------------------------------------------------------------------------------------------
# stdio: handshake, listing, calls
# ---------------------------------------------------------------------------------------------
def test_stdio_handshake(make_server: Callable[..., McpServer]) -> None:
    srv = make_server(connect=False)
    assert srv.state == "idle"
    srv.connect()
    assert srv.state == "connected"
    seen = json.loads(srv.call("inspect", {}))
    assert seen["seen"][:2] == ["initialize", "notifications/initialized"]
    assert seen["init"]["protocolVersion"] == "2025-06-18"
    assert seen["init"]["capabilities"] == {}
    assert seen["init"]["clientInfo"] == {"name": "exocortex", "version": exocortex.__version__}
    srv.connect()                                    # already connected: a no-op
    assert srv.state == "connected"


def test_stdio_tools_follow_pages_validate_names_and_cut_descriptions(make_server: Callable[..., McpServer]) -> None:
    tools = make_server().tools()
    assert [t.name for t in tools] == ["echo", "fail", "slow", "big", "long_desc", "no_schema"]   # "bad name!" dropped
    by_name = {t.name: t for t in tools}
    assert isinstance(by_name["echo"], McpTool)
    assert by_name["echo"].description == "Echo the arguments back. Second line."
    assert by_name["echo"].input_schema == {"type": "object", "properties": {"text": {"type": "string"}}}
    assert by_name["long_desc"].description == "x" * 160
    assert by_name["no_schema"].description == "" and by_name["no_schema"].input_schema == {}
    with pytest.raises(Exception):
        by_name["echo"].name = "other"  # type: ignore[misc]   # frozen


def test_stdio_call_returns_the_text(make_server: Callable[..., McpServer]) -> None:
    srv = make_server()
    assert json.loads(srv.call("echo", {"text": "héllo ☃", "n": 3})) == {"n": 3, "text": "héllo ☃"}
    assert json.loads(srv.call("echo")) == {}


def test_stdio_call_notes_the_other_content_kinds(make_server: Callable[..., McpServer]) -> None:
    text = make_server().call("mixed", {})
    assert text.split("\n") == ["line one", "[image]", "[resource: file:///notes.txt]", "embedded text",
                                "[resource: file:///linked.txt]", "line two", "[audio]"]


def test_stdio_is_error_result_raises_with_the_text(make_server: Callable[..., McpServer]) -> None:
    srv = make_server()
    with pytest.raises(McpError, match="^boom: because$"):
        srv.call("fail", {"why": "because"})
    assert srv.state == "connected"
    assert json.loads(srv.call("echo", {"a": 1})) == {"a": 1}      # the connection is still good


def test_stdio_jsonrpc_error_raises_with_the_message(make_server: Callable[..., McpServer]) -> None:
    srv = make_server()
    with pytest.raises(McpError, match="^Unknown tool: nope$"):
        srv.call("rpc_error", {})
    with pytest.raises(McpError, match="Unknown tool: missing"):
        srv.call("missing", {})
    with pytest.raises(McpError, match="arguments must be an object"):
        srv.call("echo", ["not", "a", "dict"])        # type: ignore[arg-type]


def test_stdio_non_json_output_lines_are_ignored(make_server: Callable[..., McpServer]) -> None:
    assert make_server().call("noise", {}) == "after noise"


def test_stdio_server_requests_are_refused_and_pings_answered(make_server: Callable[..., McpServer]) -> None:
    srv = make_server()
    sampling = json.loads(srv.call("sampling", {}))
    assert sampling["id"] == "srv-1"
    assert sampling["error"]["code"] == -32601 and "not supported" in sampling["error"]["message"]
    ping = json.loads(srv.call("ping", {}))
    assert ping == {"jsonrpc": "2.0", "id": "srv-2", "result": {}}


def test_stdio_calls_from_several_threads_do_not_cross(make_server: Callable[..., McpServer]) -> None:
    srv = make_server()
    results: dict[int, str] = {}

    def worker(i: int) -> None:
        results[i] = srv.call("echo", {"i": i})

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(20)
    assert {i: json.loads(text) for i, text in results.items()} == {i: {"i": i} for i in range(8)}


# ---------------------------------------------------------------------------------------------
# stdio: cancel, timeout, close
# ---------------------------------------------------------------------------------------------
def test_stdio_cancel_during_a_slow_tool_returns_promptly(make_server: Callable[..., McpServer]) -> None:
    srv = make_server()
    cancel = threading.Event()
    threading.Timer(0.3, cancel.set).start()
    started = time.monotonic()
    with pytest.raises(McpError, match="^cancelled$"):
        srv.call("slow", {"seconds": 30}, cancel=cancel)
    assert time.monotonic() - started < 1.5
    assert srv.state == "connected"                  # a cancelled call does not kill the connection


def test_stdio_cancel_already_set_never_sends(make_server: Callable[..., McpServer]) -> None:
    srv = make_server()
    cancel = threading.Event()
    cancel.set()
    with pytest.raises(McpError, match="^cancelled$"):
        srv.call("echo", {"a": 1}, cancel=cancel)
    assert "echo" not in json.loads(srv.call("inspect", {}))["seen"]


def test_stdio_timeout(make_server: Callable[..., McpServer]) -> None:
    srv = make_server()
    srv._timeout_s = 0.5                             # after the handshake: Python start-up is not what we time
    started = time.monotonic()
    with pytest.raises(McpError, match=r"^timed out after 0\.5 s$"):
        srv.call("slow", {"seconds": 30})
    assert time.monotonic() - started < 2.0
    assert srv.state == "connected"


def test_stdio_close_leaves_no_process_behind(make_server: Callable[..., McpServer]) -> None:
    srv = make_server()
    proc = srv._proc
    reader = srv._reader
    assert proc is not None and proc.poll() is None
    srv.close()
    assert wait_until(lambda: proc.poll() is not None)
    assert srv.state == "idle"
    assert reader is not None and wait_until(lambda: not reader.is_alive())
    srv.close()                                      # idempotent
    with pytest.raises(McpError, match="not connected"):
        srv.call("echo", {})


def test_stdio_close_kills_a_process_that_ignores_eof_and_its_children(make_server: Callable[..., McpServer]) -> None:
    srv = make_server(env={"EXO_HANG_ON_EOF": "1"})
    proc = srv._proc
    assert proc is not None
    grandchild = int(srv.call("spawn", {}))
    assert pid_alive(grandchild)
    started = time.monotonic()
    srv.close()
    assert time.monotonic() - started < 10
    assert wait_until(lambda: proc.poll() is not None)
    assert wait_until(lambda: not pid_alive(grandchild)), "the server's child survived close()"


def test_stdio_close_of_a_server_that_never_connected_is_harmless(make_server: Callable[..., McpServer]) -> None:
    srv = make_server(connect=False)
    srv.close()
    assert srv.state == "idle"


def test_stdio_reconnect_after_close(make_server: Callable[..., McpServer]) -> None:
    srv = make_server()
    first = srv._proc
    srv.close()
    srv.connect()
    assert srv.state == "connected" and srv._proc is not first
    assert json.loads(srv.call("echo", {"again": True})) == {"again": True}


# ---------------------------------------------------------------------------------------------
# stdio: failures
# ---------------------------------------------------------------------------------------------
def test_a_missing_command_is_reported(tmp_path: Path) -> None:
    srv = McpServer({"name": "ghost", "transport": "stdio", "command": "definitely-not-a-command-xyz",
                     "args": ["--flag"]}, timeout_s=5)
    with pytest.raises(McpError, match="^command not found: definitely-not-a-command-xyz$"):
        srv.connect()
    assert srv.state == "error: command not found: definitely-not-a-command-xyz"


def test_a_command_that_is_not_an_mcp_server_fails_the_handshake(tmp_path: Path) -> None:
    script = tmp_path / "quiet.py"
    script.write_text("import sys\nsys.exit(0)\n", encoding="utf-8")
    srv = McpServer({"name": "quiet", "transport": "stdio", "command": sys.executable, "args": [str(script)]}, timeout_s=10)
    with pytest.raises(McpError, match=r"^exited \(code 0\)$"):
        srv.connect()
    assert srv.state == "error: exited (code 0)"


def test_the_handshake_times_out_on_a_silent_server(tmp_path: Path) -> None:
    script = tmp_path / "silent.py"
    script.write_text("import sys, time\nsys.stdin.readline()\ntime.sleep(60)\n", encoding="utf-8")
    srv = McpServer({"name": "silent", "transport": "stdio", "command": sys.executable, "args": [str(script)]}, timeout_s=0.8)
    started = time.monotonic()
    with pytest.raises(McpError, match="timed out"):
        srv.connect()
    assert time.monotonic() - started < 6
    assert srv.state.startswith("error: timed out")
    assert srv._proc is None                          # the half-started process was torn down


def test_an_oversized_line_marks_the_connection_broken_and_kills_it(make_server: Callable[..., McpServer]) -> None:
    srv = make_server()
    proc = srv._proc
    assert proc is not None
    with pytest.raises(McpError, match="^oversized message$"):
        srv.call("big", {})
    assert srv.state == "error: oversized message"
    assert wait_until(lambda: proc.poll() is not None)
    with pytest.raises(McpError, match="not connected"):
        srv.call("echo", {})
    srv.connect()                                    # a fresh start works
    assert srv.state == "connected"
    assert json.loads(srv.call("echo", {"ok": 1})) == {"ok": 1}


def test_a_server_that_dies_mid_call_fails_the_call(make_server: Callable[..., McpServer]) -> None:
    srv = make_server()
    with pytest.raises(McpError, match=r"^exited \(code 3\)$"):
        srv.call("exit", {})
    assert srv.state == "error: exited (code 3)"
    with pytest.raises(McpError, match=r"not connected \(error: exited \(code 3\)\)"):
        srv.tools()


def test_a_server_that_dies_while_idle_is_marked_dead_without_anyone_calling(make_server: Callable[..., McpServer]) -> None:
    srv = make_server()
    assert srv.call("die_soon", {}) == "bye"
    assert wait_until(lambda: srv.state == "error: exited (code 4)", timeout=5)
    with pytest.raises(McpError, match=r"not connected \(error: exited \(code 4\)\)"):
        srv.tools()
    with pytest.raises(McpError, match="not connected"):
        srv.call("echo", {})
    srv.connect()                                    # starting over works
    assert srv.state == "connected"
    assert json.loads(srv.call("echo", {"back": 1})) == {"back": 1}


def test_a_launcher_that_exits_while_its_child_holds_the_pipe_is_marked_dead(tmp_path: Path) -> None:
    script = tmp_path / "launcher.py"
    script.write_text(LAUNCHER_SCRIPT, encoding="utf-8")
    pidfile = tmp_path / "child.pid"
    srv = McpServer({"name": "launcher", "transport": "stdio", "command": sys.executable,
                     "args": [str(script), str(pidfile)]}, timeout_s=10)
    try:
        try:
            srv.connect()
        except McpError:
            pass                                     # the exit may land before connect() has returned
        assert wait_until(lambda: srv.state == "error: exited (code 5)", timeout=5)
        with pytest.raises(McpError, match="not connected"):
            srv.call("anything", {})
    finally:
        srv.close()
        try:
            os.kill(int(pidfile.read_text()), signal.SIGTERM)       # the child that outlived its launcher
        except (OSError, ValueError):
            pass


# ---------------------------------------------------------------------------------------------
# stdio: the minimal environment
# ---------------------------------------------------------------------------------------------
def test_the_server_gets_a_minimal_environment_plus_the_spec_env(make_server: Callable[..., McpServer]) -> None:
    base = dict(os.environ)
    base.update({"EXO_SECRET_TOKEN": "hunter2", "OPENAI_API_KEY": "sk-not-for-you", "GITHUB_TOKEN": "ghp_x"})
    srv = make_server(env_base=base, env={"EXO_MARKER": "from-spec"})
    reply = json.loads(srv.call("env", {}))
    keys = {k.upper() for k in reply["keys"]}
    assert reply["marker"] == "from-spec"
    for secret in ("EXO_SECRET_TOKEN", "OPENAI_API_KEY", "GITHUB_TOKEN"):
        assert secret not in keys
    kept = {"PATH", "SYSTEMROOT", "TEMP", "TMP", "USERPROFILE", "HOME", "APPDATA", "LOCALAPPDATA", "PATHEXT",
            "COMSPEC", "HOMEDRIVE", "HOMEPATH"}
    leaked = {k.upper() for k in base} & keys - kept - {"EXO_MARKER"}
    assert not leaked, f"variables that should not have reached the server: {sorted(leaked)}"
    assert "PATH" in keys


def test_a_custom_env_base_is_the_only_source(make_server: Callable[..., McpServer], monkeypatch: pytest.MonkeyPatch) -> None:
    # USERPROFILE / HOME are kept by the filter, so only env_base itself can keep them out: the child
    # must not see them although the REAL environment has them.
    monkeypatch.setenv("USERPROFILE", r"C:\exo-sentinel")
    monkeypatch.setenv("HOME", "/exo-sentinel")
    base = {k: v for k, v in os.environ.items() if k.upper() in {"PATH", "SYSTEMROOT", "TEMP", "TMP"}}
    keys = {k.upper() for k in json.loads(make_server(env_base=base).call("env", {}))["keys"]}
    assert "USERPROFILE" not in keys and "HOME" not in keys
    assert "PATH" in keys                            # and the base itself did reach the child


def test_the_default_env_base_is_the_filtered_real_environment(make_server: Callable[..., McpServer],
                                                               monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-not-for-you")
    monkeypatch.setenv("EXO_SENTINEL_DROPPED", "x")
    monkeypatch.setenv("HOME", "/exo-sentinel")                  # on the keep list: it must get through
    keys = {k.upper() for k in json.loads(make_server().call("env", {}))["keys"]}
    assert "OPENAI_API_KEY" not in keys and "EXO_SENTINEL_DROPPED" not in keys
    assert "HOME" in keys and "PATH" in keys


# ---------------------------------------------------------------------------------------------
# http (streamable): a fake server on httpx.MockTransport
# ---------------------------------------------------------------------------------------------
TOOL_PAGE1 = [{"name": "echo", "description": "Echo.", "inputSchema": {"type": "object"}},
              {"name": "weird name", "description": "dropped"}]
TOOL_PAGE2 = [{"name": "second", "description": "On page two."}]


class FakeMcpHttp:
    """A streamable-HTTP MCP server. ``sse=True`` answers every request as an event stream (with a
    notification and a server request ahead of the reply on ``tools/call``)."""

    def __init__(self, *, sse: bool = False, session: str = "sess-1", hold: threading.Event | None = None) -> None:
        self.sse, self.session, self.hold = sse, session, hold
        self.log: list[tuple[str, dict[str, str], dict[str, Any] | None]] = []
        self.call_status = 200

    def bodies(self, method: str) -> list[dict[str, Any]]:
        return [b for _m, _h, b in self.log if b and b.get("method") == method]

    def reply(self, rid: Any, result: dict[str, Any], *, headers: dict[str, str] | None = None,
              pre: list[str] | None = None, split: bool = False) -> httpx.Response:
        message = {"jsonrpc": "2.0", "id": rid, "result": result}
        if not self.sse:
            return httpx.Response(200, json=message, headers=headers)
        events = list(pre or [])
        if split:                                     # one event whose JSON spans several data: lines
            pretty = json.dumps(message, indent=1).split("\n")
            events.append("event: message\n" + "\n".join("data: " + line for line in pretty))
        else:
            events.append("event: message\ndata: " + json.dumps(message))
        body = ": keep-alive\n\n" + "\n\n".join(events) + "\n\n"
        return httpx.Response(200, headers={"content-type": "text/event-stream", **(headers or {})},
                              content=body.encode("utf-8"))

    def __call__(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content) if request.content else None
        self.log.append((request.method, dict(request.headers), body))
        if request.method == "DELETE":
            return httpx.Response(204)
        assert body is not None
        method = body.get("method")
        if "id" not in body:
            return httpx.Response(202)
        if method == "initialize":
            headers = {"Mcp-Session-Id": self.session} if self.session else {}
            return self.reply(body["id"], {"protocolVersion": "2025-06-18", "capabilities": {"tools": {}},
                                           "serverInfo": {"name": "fake-http", "version": "1"}}, headers=headers)
        if method == "tools/list":
            if (body.get("params") or {}).get("cursor") == "c2":
                return self.reply(body["id"], {"tools": TOOL_PAGE2})
            return self.reply(body["id"], {"tools": TOOL_PAGE1, "nextCursor": "c2"})
        if method == "tools/call":
            if self.hold is not None:
                self.hold.wait(10)
            if self.call_status != 200:
                return httpx.Response(self.call_status, content=b"upstream exploded")
            name = body["params"]["name"]
            args = body["params"].get("arguments")
            if name == "error":
                return self.reply(body["id"], {"content": [{"type": "text", "text": "it broke"}], "isError": True})
            if name == "rpc_error":
                return httpx.Response(200, json={"jsonrpc": "2.0", "id": body["id"],
                                                 "error": {"code": -32602, "message": "bad params"}})
            pre = None
            if self.sse:
                pre = ['event: message\ndata: {"jsonrpc":"2.0","method":"notifications/progress","params":{"progress":1}}',
                       'event: message\ndata: {"jsonrpc":"2.0","id":"srv-9","method":"sampling/createMessage","params":{}}']
            return self.reply(body["id"], {"content": [{"type": "text", "text": json.dumps(args, sort_keys=True)}]},
                              pre=pre, split=self.sse)
        return httpx.Response(200, json={"jsonrpc": "2.0", "id": body["id"],
                                         "error": {"code": -32601, "message": "no such method"}})


def http_server(fake: Callable[[httpx.Request], httpx.Response], **kwargs: Any) -> McpServer:
    kwargs.setdefault("timeout_s", 10.0)
    return McpServer({"name": "remote", "transport": "http", "url": "https://mcp.example.test/mcp"},
                     transport=httpx.MockTransport(fake), **kwargs)


@pytest.mark.parametrize("sse", [False, True], ids=["json", "sse"])
def test_http_handshake_listing_call_and_close(sse: bool) -> None:
    fake = FakeMcpHttp(sse=sse)
    srv = http_server(fake)
    srv.connect()
    assert srv.state == "connected"
    # the initialize request carries no session; everything after it does
    method0, headers0, init = fake.log[0]
    assert method0 == "POST" and init is not None and init["method"] == "initialize"
    assert init["params"]["protocolVersion"] == "2025-06-18"
    assert init["params"]["clientInfo"] == {"name": "exocortex", "version": exocortex.__version__}
    assert "mcp-session-id" not in headers0
    assert headers0["accept"] == "application/json, text/event-stream"
    assert headers0["content-type"] == "application/json"
    # notifications/initialized was POSTed (the 202 is fine)
    assert fake.bodies("notifications/initialized")
    assert fake.log[1][1]["mcp-session-id"] == "sess-1"
    assert fake.log[1][1]["mcp-protocol-version"] == "2025-06-18"

    tools = srv.tools()
    assert [t.name for t in tools] == ["echo", "second"]                 # 2 pages, the bad name dropped
    assert tools[1].description == "On page two."
    assert [b["params"].get("cursor") for b in fake.bodies("tools/list")] == [None, "c2"]

    assert json.loads(srv.call("echo", {"q": "café"})) == {"q": "café"}
    for _m, headers, body in fake.log[1:]:
        if body is not None:
            assert headers["mcp-session-id"] == "sess-1"

    srv.close()
    assert srv.state == "idle"
    method, headers, _body = fake.log[-1]
    assert method == "DELETE" and headers["mcp-session-id"] == "sess-1"


def test_http_sse_server_requests_are_rejected_in_flight() -> None:
    fake = FakeMcpHttp(sse=True)
    srv = http_server(fake)
    srv.connect()
    srv.call("echo", {"a": 1})
    rejections = [b for _m, _h, b in fake.log if b and b.get("id") == "srv-9"]
    assert len(rejections) == 1
    assert rejections[0]["error"]["code"] == -32601
    srv.close()


def test_http_without_a_session_id_sends_none_and_deletes_nothing() -> None:
    fake = FakeMcpHttp(session="")
    srv = http_server(fake)
    srv.connect()
    srv.call("echo", {})
    srv.close()
    assert all("mcp-session-id" not in headers for _m, headers, _b in fake.log)
    assert all(m != "DELETE" for m, _h, _b in fake.log)


def test_http_is_error_and_jsonrpc_error_raise() -> None:
    srv = http_server(FakeMcpHttp())
    srv.connect()
    with pytest.raises(McpError, match="^it broke$"):
        srv.call("error", {})
    with pytest.raises(McpError, match="^bad params$"):
        srv.call("rpc_error", {})
    assert srv.state == "connected"
    srv.close()


def test_http_error_status_raises_with_the_status() -> None:
    fake = FakeMcpHttp()
    srv = http_server(fake)
    srv.connect()
    fake.call_status = 500
    with pytest.raises(McpError, match="^HTTP 500: upstream exploded$"):
        srv.call("echo", {})
    fake.call_status = 404
    with pytest.raises(McpError, match="^HTTP 404"):
        srv.call("echo", {})
    assert srv.state == "connected"                  # a failed call is not a dead connection
    srv.close()


def test_http_error_status_on_initialize_fails_the_connect() -> None:
    srv = http_server(lambda request: httpx.Response(401, content=b"unauthorised"))
    with pytest.raises(McpError, match="^HTTP 401: unauthorised$"):
        srv.connect()
    assert srv.state == "error: HTTP 401: unauthorised"


def test_http_unreachable_server_is_a_readable_error() -> None:
    def refuse(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    srv = http_server(refuse)
    with pytest.raises(McpError, match="request failed: connection refused"):
        srv.connect()
    assert srv.state.startswith("error: request failed")


def test_http_redirects_are_not_followed() -> None:
    srv = http_server(lambda request: httpx.Response(307, headers={"location": "http://127.0.0.1/steal"}))
    with pytest.raises(McpError, match="unexpected HTTP 307"):
        srv.connect()


def test_http_a_reply_that_is_not_json_or_not_for_us_is_an_error() -> None:
    srv = http_server(lambda request: httpx.Response(200, headers={"content-type": "text/html"}, content=b"<html>hi</html>"))
    with pytest.raises(McpError, match="not JSON"):
        srv.connect()
    wrong = http_server(lambda request: httpx.Response(200, json={"jsonrpc": "2.0", "id": 999, "result": {}}))
    with pytest.raises(McpError, match="does not answer"):
        wrong.connect()
    empty = http_server(lambda request: httpx.Response(202))
    with pytest.raises(McpError, match="no reply"):
        empty.connect()


def test_http_a_stream_that_ends_without_the_reply_is_an_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, headers={"content-type": "text/event-stream"},
                              content=b'data: {"jsonrpc":"2.0","method":"notifications/message"}\n\n')

    with pytest.raises(McpError, match="ended its reply without answering"):
        http_server(handler).connect()


def test_http_an_oversized_reply_is_refused() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, headers={"content-type": "application/json"}, content=b" " * (2 * 1024 * 1024))

    with pytest.raises(McpError, match="oversized message"):
        http_server(handler).connect()


def test_http_cancel_and_timeout_while_the_server_is_slow() -> None:
    hold = threading.Event()
    fake = FakeMcpHttp(hold=hold)
    srv = http_server(fake)
    srv.connect()
    try:
        cancel = threading.Event()
        threading.Timer(0.3, cancel.set).start()
        started = time.monotonic()
        with pytest.raises(McpError, match="^cancelled$"):
            srv.call("echo", {}, cancel=cancel)
        assert time.monotonic() - started < 1.5
        srv._timeout_s = 0.4
        started = time.monotonic()
        with pytest.raises(McpError, match=r"^timed out after 0\.4 s$"):
            srv.call("echo", {})
        assert time.monotonic() - started < 1.5
    finally:
        hold.set()
        srv.close()


def test_http_close_fails_a_call_that_is_still_in_flight() -> None:
    hold = threading.Event()
    fake = FakeMcpHttp(hold=hold)
    srv = http_server(fake)
    srv.connect()
    outcome: list[str] = []

    def worker() -> None:
        try:
            srv.call("echo", {})
        except McpError as exc:
            outcome.append(str(exc))

    thread = threading.Thread(target=worker)
    thread.start()
    started = time.monotonic()
    try:
        # the worker's request is registered and its POST is inside the handler (holding the reply)
        assert wait_until(lambda: bool(fake.bodies("tools/call")))
        srv.close()
        thread.join(3)
        assert outcome == ["closed"] and time.monotonic() - started < 2.0
    finally:
        hold.set()


class IdleSse(httpx.SyncByteStream):
    """An event stream that sent a keep-alive comment and then says nothing until ``hold`` is set."""

    def __init__(self, hold: threading.Event) -> None:
        self.hold = hold

    def __iter__(self) -> Iterator[bytes]:
        yield b": keep-alive\n\n"
        self.hold.wait(15)


def test_http_cancel_while_an_sse_stream_idles() -> None:
    hold = threading.Event()
    fake = FakeMcpHttp()

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content) if request.content else None
        if body and body.get("method") == "tools/call":
            return httpx.Response(200, headers={"content-type": "text/event-stream"}, stream=IdleSse(hold))
        return fake(request)

    srv = http_server(handler)
    srv.connect()
    try:
        cancel = threading.Event()
        threading.Timer(0.3, cancel.set).start()
        started = time.monotonic()
        with pytest.raises(McpError, match="^cancelled$"):
            srv.call("echo", {}, cancel=cancel)
        assert time.monotonic() - started < 1.5
        assert srv.state == "connected"
    finally:
        hold.set()
        srv.close()


def test_http_two_5xx_answers_in_a_row_mark_the_server_down() -> None:
    fake = FakeMcpHttp()
    srv = http_server(fake)
    srv.connect()
    fake.call_status = 503
    with pytest.raises(McpError, match="^HTTP 503"):
        srv.call("echo", {})
    assert srv.state == "connected"                  # one 5xx is a failed call, not a dead server
    fake.call_status = 200
    assert srv.call("echo", {"n": 1}) == '{"n": 1}'  # an answer in between starts the count over
    fake.call_status = 502
    with pytest.raises(McpError, match="^HTTP 502"):
        srv.call("echo", {})
    assert srv.state == "connected"
    fake.call_status = 503
    with pytest.raises(McpError, match="^HTTP 503"):
        srv.call("echo", {})
    assert srv.state == "error: HTTP 503 twice in a row"
    with pytest.raises(McpError, match=r"not connected \(error: HTTP 503 twice in a row\)"):
        srv.tools()
    with pytest.raises(McpError, match="not connected"):
        srv.call("echo", {})
    fake.call_status = 200
    srv.connect()                                    # starting over brings it back
    assert srv.state == "connected"
    assert srv.call("echo", {"n": 2}) == '{"n": 2}'
    srv.close()


def test_http_reconnect_after_close_starts_a_new_session() -> None:
    fake = FakeMcpHttp()
    srv = http_server(fake)
    srv.connect()
    srv.close()
    fake.session = "sess-2"
    srv.connect()
    assert srv.call("echo", {"n": 1}) == '{"n": 1}'
    last_call = fake.bodies("tools/call")[-1]
    assert last_call["params"]["name"] == "echo"
    assert fake.log[-1][1]["mcp-session-id"] == "sess-2"
    srv.close()
