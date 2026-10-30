"""Unit tests for exocortex.webfetch (2026-10-07, capabilities), all offline.

Every request goes through ``httpx.MockTransport`` and every host through a fake resolver, so
nothing here touches the network or the system resolver. The tests pin the fetch policy (scheme,
domain lists, the refused address classes, redirects re-checked hop by hop), the body limits, the
search-result parser, that a cancel or the wall-clock limit ends a fetch against a server that
drips or says nothing (a raw-socket server on 127.0.0.1, ``web_allow_local``) and that a hostile
page cannot make the HTML conversion run for minutes."""
from __future__ import annotations

import re
import socket
import subprocess
import sys
import textwrap
import threading
import time
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import httpx
import pytest

from exocortex.config import Config
from exocortex.context import html_to_text
from exocortex.webfetch import (
    MAX_BODY_BYTES,
    FetchResult,
    SearchHit,
    WebError,
    WebFetcher,
)

Handler = Callable[[httpx.Request], httpx.Response]

PUBLIC_IP = "93.184.216.34"


def resolver_for(table: dict[str, str | list[str]] | None = None, default: str = PUBLIC_IP) -> Callable[..., list[Any]]:
    """A ``getaddrinfo`` stand-in: ``table`` maps a host to one address or a list; others get ``default``."""

    def resolve(host: str, port: int, *_a: Any, **_k: Any) -> list[Any]:
        value = (table or {}).get(host, default)
        addrs = [value] if isinstance(value, str) else value
        return [(socket.AF_INET6 if ":" in a else socket.AF_INET, socket.SOCK_STREAM, 6, "", (a, port)) for a in addrs]

    return resolve


def no_resolution(host: str, port: int, *_a: Any, **_k: Any) -> list[Any]:
    raise AssertionError(f"the resolver must not be reached for {host}")


def make(handler: Handler, *, resolver: Callable[..., Any] | None = None, **caps: Any) -> WebFetcher:
    cfg = Config()
    for key, value in caps.items():
        setattr(cfg.capabilities, key, value)
    return WebFetcher(cfg, transport=httpx.MockTransport(handler), resolver=resolver or resolver_for())


def html_page(body: str, status: int = 200, ctype: str = "text/html; charset=utf-8") -> httpx.Response:
    return httpx.Response(status, headers={"content-type": ctype}, content=body.encode("utf-8"))


class ChunkStream(httpx.SyncByteStream):
    """A response body of ``count`` chunks of ``size`` bytes that records how many were pulled."""

    def __init__(self, count: int, size: int = 16 * 1024, on_chunk: Callable[[int], None] | None = None) -> None:
        self.count, self.size, self.on_chunk = count, size, on_chunk
        self.pulled = 0

    def __iter__(self):
        for i in range(self.count):
            self.pulled += 1
            if self.on_chunk is not None:
                self.on_chunk(i)
            yield b"a" * self.size


PAGE = """<!doctype html><html><head><title>Example Page</title><style>p{color:red}</style></head>
<body><script>var secret = 1;</script><h1>Welcome</h1><p>First paragraph.</p>
<ul><li>one</li><li>two</li></ul></body></html>"""


# ---------------------------------------------------------------------------------------------
# fetching a page
# ---------------------------------------------------------------------------------------------
def test_fetch_converts_html_to_text_and_finds_the_title() -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return html_page(PAGE)

    result = make(handler).fetch("https://docs.example.com/a/b?x=1")
    assert isinstance(result, FetchResult)
    assert result.title == "Example Page"
    assert result.status == 200
    assert result.final_url == "https://docs.example.com/a/b?x=1"
    assert "# Welcome" in result.text and "First paragraph." in result.text and "- one" in result.text
    assert "secret" not in result.text and "color:red" not in result.text
    assert "\n\n\n" not in result.text
    assert seen[0].headers["accept-encoding"] == "identity"
    assert "ExoCortex" in seen[0].headers["user-agent"]


def test_fetch_title_falls_back_to_the_first_heading() -> None:
    result = make(lambda r: html_page("<html><body><h1>Only A Heading</h1><p>x</p></body></html>")).fetch("https://a.org/")
    assert result.title == "Only A Heading"


def test_fetch_keeps_plain_text_and_json_as_they_are() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/j":
            return httpx.Response(200, headers={"content-type": "application/json"}, content=b'{"a": [1, 2]}')
        if request.url.path == "/ld":
            return httpx.Response(200, headers={"content-type": "application/ld+json"}, content=b'{"@id": 1}')
        return httpx.Response(200, headers={"content-type": "text/plain; charset=iso-8859-1"},
                              content="caf\xe9\r\nline two\r\n".encode("latin-1"))

    fetcher = make(handler)
    assert fetcher.fetch("https://a.org/j").text == '{"a": [1, 2]}'
    assert fetcher.fetch("https://a.org/ld").text == '{"@id": 1}'
    plain = fetcher.fetch("https://a.org/t")
    assert plain.text == "café\nline two" and plain.title == ""


def test_fetch_sniffs_the_meta_charset_of_an_html_page() -> None:
    body = '<html><head><meta charset="windows-1252"><title>Caf\xe9</title></head><body>na\xefve</body></html>'
    result = make(lambda r: httpx.Response(200, headers={"content-type": "text/html"}, content=body.encode("cp1252"))
                  ).fetch("https://a.org/")
    assert result.title == "Café" and "naïve" in result.text


def test_fetch_http_errors_and_network_failures_are_readable() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/gone":
            return httpx.Response(404, content=b"nope")
        if request.url.path == "/slow":
            raise httpx.ReadTimeout("slow", request=request)
        raise httpx.ConnectError("refused", request=request)

    fetcher = make(handler)
    with pytest.raises(WebError, match="HTTP 404"):
        fetcher.fetch("https://a.org/gone")
    with pytest.raises(WebError, match="timed out"):
        fetcher.fetch("https://a.org/slow")
    with pytest.raises(WebError, match="request failed: refused"):
        fetcher.fetch("https://a.org/down")


def test_a_host_that_does_not_resolve_is_a_plain_error() -> None:
    def broken(host: str, port: int, *a: Any, **k: Any) -> list[Any]:
        raise socket.gaierror(11001, "getaddrinfo failed")

    with pytest.raises(WebError, match="could not resolve nowhere.invalid"):
        make(lambda r: html_page(PAGE), resolver=broken).fetch("https://nowhere.invalid/")


# ---------------------------------------------------------------------------------------------
# scheme and URL shape
# ---------------------------------------------------------------------------------------------
@pytest.mark.parametrize("url", [
    "file:///C:/Windows/win.ini",
    "data:text/html,<h1>hi</h1>",
    "ftp://example.com/x",
    "javascript:alert(1)",
    "",
    "   ",
    "http://",
    "https://user:secret@example.com/",
    "not a url at all",
])
def test_fetch_refuses_non_http_urls(url: str) -> None:
    calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(str(request.url))
        return html_page(PAGE)

    with pytest.raises(WebError, match="^denied: "):
        make(handler).fetch(url)
    assert calls == []


# ---------------------------------------------------------------------------------------------
# domain lists
# ---------------------------------------------------------------------------------------------
def test_blocked_domains_cover_the_domain_and_its_subdomains_only() -> None:
    fetcher = make(lambda r: html_page(PAGE), web_blocked_domains=["bad.com"])
    for host in ("bad.com", "www.bad.com", "a.b.BAD.com"):
        with pytest.raises(WebError, match="denied: .* blocked domain list"):
            fetcher.fetch(f"https://{host}/")
    assert fetcher.fetch("https://notbad.com/").status == 200
    assert fetcher.fetch("https://bad.com.example.org/").status == 200


def test_allowed_domains_restrict_to_the_list_by_whole_labels() -> None:
    fetcher = make(lambda r: html_page(PAGE), web_allowed_domains=["example.com", "*.docs.rs"])
    assert fetcher.fetch("https://example.com/").status == 200
    assert fetcher.fetch("https://docs.example.com/x").status == 200
    assert fetcher.fetch("https://docs.rs/y").status == 200
    for url in ("https://notexample.com/", "https://example.com.evil.net/", "https://other.org/"):
        with pytest.raises(WebError, match="denied: .* not on the allowed domain list"):
            fetcher.fetch(url)


def test_empty_allowed_list_means_any_public_host_and_blocked_wins_over_allowed() -> None:
    assert make(lambda r: html_page(PAGE)).fetch("https://anything.org/").status == 200
    fetcher = make(lambda r: html_page(PAGE), web_allowed_domains=["example.com"],
                   web_blocked_domains=["private.example.com"])
    assert fetcher.fetch("https://www.example.com/").status == 200
    with pytest.raises(WebError, match="blocked domain list"):
        fetcher.fetch("https://private.example.com/")


def test_the_lists_are_read_live() -> None:
    fetcher = make(lambda r: html_page(PAGE))
    assert fetcher.fetch("https://a.org/").status == 200
    fetcher._cfg.capabilities.web_blocked_domains = ["a.org"]
    with pytest.raises(WebError, match="denied"):
        fetcher.fetch("https://a.org/")


# ---------------------------------------------------------------------------------------------
# refused addresses
# ---------------------------------------------------------------------------------------------
@pytest.mark.parametrize("address", [
    "127.0.0.1", "10.0.0.1", "172.16.5.4", "192.168.1.10", "169.254.1.1", "100.64.0.1", "100.127.255.254",
    "0.0.0.0", "224.0.0.1", "240.0.0.1", "::1", "::", "fe80::1", "fc00::1", "ff02::1",
    "::ffff:127.0.0.1", "::ffff:8.8.8.8", "::7f00:1", "64:ff9b::7f00:1", "64:ff9b::a00:1",
])
def test_every_non_public_address_class_is_refused(address: str) -> None:
    calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(str(request.url))
        return html_page(PAGE)

    fetcher = make(handler, resolver=resolver_for({"intranet.example.org": address}))
    with pytest.raises(WebError, match=r"^denied: intranet\.example\.org resolves to"):
        fetcher.fetch("https://intranet.example.org/admin")
    assert calls == []


@pytest.mark.parametrize("address", [PUBLIC_IP, "8.8.8.8", "2606:2800:220:1:248:1893:25c8:1946", "64:ff9b::808:808"])
def test_public_addresses_pass(address: str) -> None:
    assert make(lambda r: html_page(PAGE), resolver=resolver_for({"ok.org": address})).fetch("https://ok.org/").status == 200


def test_one_private_address_among_public_ones_refuses_the_host() -> None:
    fetcher = make(lambda r: html_page(PAGE), resolver=resolver_for({"mixed.org": [PUBLIC_IP, "10.1.2.3"]}))
    with pytest.raises(WebError, match="^denied: mixed.org resolves to a private address"):
        fetcher.fetch("https://mixed.org/")


@pytest.mark.parametrize("url", [
    "http://127.0.0.1/",
    "http://127.0.0.1:8080/x",
    "http://[::1]/",
    "http://[::ffff:127.0.0.1]/",
    "http://169.254.169.254/latest/meta-data",
    "http://100.64.0.1/",
    "http://10.0.0.5/",
    "http://0.0.0.0/",
    "http://[fe80::1]/",
    "http://0x7f000001/",
    "http://2130706433/",
    "http://127.1/",
    "http://0177.0.0.1/",
    "http://0x7f.0.0.1/",
    "http://1.2.3.4.5/",
])
def test_literal_and_numeric_hosts_are_refused_without_resolving(url: str) -> None:
    fetcher = make(lambda r: pytest.fail("must not be requested"), resolver=no_resolution)
    with pytest.raises(WebError, match="^denied: "):
        fetcher.fetch(url)


def test_a_literal_public_ip_is_fine() -> None:
    result = make(lambda r: html_page(PAGE), resolver=no_resolution).fetch("http://93.184.216.34/")
    assert result.status == 200


def test_web_allow_local_lifts_the_address_check() -> None:
    fetcher = make(lambda r: html_page(PAGE), resolver=resolver_for({"nas.local": "192.168.1.5"}), web_allow_local=True)
    assert fetcher.fetch("http://nas.local/").status == 200
    assert fetcher.fetch("http://127.0.0.1:8080/").status == 200
    assert fetcher.fetch("http://[::1]/").status == 200
    # ... but only the address check: the domain lists still apply
    fetcher._cfg.capabilities.web_blocked_domains = ["nas.local"]
    with pytest.raises(WebError, match="blocked domain list"):
        fetcher.fetch("http://nas.local/")


# ---------------------------------------------------------------------------------------------
# redirects
# ---------------------------------------------------------------------------------------------
def redirect(location: str, status: int = 302) -> httpx.Response:
    return httpx.Response(status, headers={"location": location}, content=b"")


def test_a_redirect_is_followed_and_reported_in_final_url() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/old":
            return redirect("/new?x=1", 301)
        return html_page(PAGE)

    result = make(handler).fetch("https://a.org/old")
    assert result.final_url == "https://a.org/new?x=1" and result.status == 200


def test_a_redirect_to_a_blocked_host_is_refused_on_the_hop() -> None:
    hosts: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        hosts.append(request.url.host)
        return redirect("https://evil.net/landing") if request.url.host == "ok.org" else html_page(PAGE)

    fetcher = make(handler, web_blocked_domains=["evil.net"])
    with pytest.raises(WebError, match="^denied: evil.net is on the blocked domain list"):
        fetcher.fetch("https://ok.org/")
    assert hosts == ["ok.org"]            # evil.net was never contacted


def test_a_redirect_outside_the_allowed_list_is_refused() -> None:
    fetcher = make(lambda r: redirect("https://elsewhere.org/"), web_allowed_domains=["ok.org"])
    with pytest.raises(WebError, match="not on the allowed domain list"):
        fetcher.fetch("https://ok.org/")


@pytest.mark.parametrize("target", [
    "http://127.0.0.1/admin",
    "http://[::1]:9/",
    "http://169.254.169.254/latest/meta-data",
    "http://0x7f000001/",
    "http://rebind.example.org/",      # resolves to a private address
    "file:///C:/Windows/win.ini",
    "data:text/plain,hi",
    "ftp://example.com/x",
])
def test_a_redirect_to_a_local_or_odd_target_is_refused(target: str) -> None:
    hosts: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        hosts.append(request.url.host)
        return redirect(target) if request.url.host == "start.org" else html_page(PAGE)

    fetcher = make(handler, resolver=resolver_for({"rebind.example.org": "10.9.9.9"}))
    with pytest.raises(WebError, match="^denied: "):
        fetcher.fetch("https://start.org/")
    assert hosts == ["start.org"]


def chain_handler(length: int, final: Callable[[], httpx.Response] = lambda: html_page(PAGE)) -> Handler:
    """/0 -> /1 -> ... -> /length (the page), each hop a redirect."""

    def handler(request: httpx.Request) -> httpx.Response:
        n = int(request.url.path.strip("/") or 0)
        return redirect(f"/{n + 1}") if n < length else final()

    return handler


def test_five_redirects_are_followed() -> None:
    result = make(chain_handler(5)).fetch("https://a.org/0")
    assert result.final_url == "https://a.org/5" and result.status == 200


def test_the_sixth_redirect_is_refused() -> None:
    requested: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requested.append(request.url.path)
        return chain_handler(6)(request)

    with pytest.raises(WebError, match="^denied: more than 5 redirects"):
        make(handler).fetch("https://a.org/0")
    assert requested == [f"/{i}" for i in range(6)]       # the page at /6 was never requested


def test_a_redirect_without_a_location_is_an_error() -> None:
    with pytest.raises(WebError, match="without saying where"):
        make(lambda r: httpx.Response(302, content=b"")).fetch("https://a.org/")


# ---------------------------------------------------------------------------------------------
# content types and sizes
# ---------------------------------------------------------------------------------------------
@pytest.mark.parametrize("ctype", ["image/png", "application/pdf", "application/octet-stream", "video/mp4", ""])
def test_a_non_text_content_type_is_refused_before_the_body_is_read(ctype: str) -> None:
    stream = ChunkStream(50)
    headers = {"content-type": ctype} if ctype else {}
    fetcher = make(lambda r: httpx.Response(200, headers=headers, stream=stream))
    with pytest.raises(WebError, match=r"^denied: not a text page \(" + (re.escape(ctype) if ctype else "no content type")):
        fetcher.fetch("https://a.org/file")
    assert stream.pulled == 0


def test_binary_content_labelled_as_text_is_refused() -> None:
    with pytest.raises(WebError, match="denied: not a text page"):
        make(lambda r: httpx.Response(200, headers={"content-type": "text/plain"}, content=b"MZ\x00\x01" * 50)
             ).fetch("https://a.org/x")


def test_the_body_is_read_at_most_2_mib() -> None:
    stream = ChunkStream(count=300)                              # 4.7 MiB on offer
    fetcher = make(lambda r: httpx.Response(200, headers={"content-type": "text/plain"}, stream=stream),
                   web_page_chars=10_000_000)
    result = fetcher.fetch("https://a.org/big")
    body = result.text.split("\n(the page is larger", 1)[0]
    assert len(body) == MAX_BODY_BYTES
    assert "larger than 2 MiB" in result.text
    assert stream.pulled <= MAX_BODY_BYTES // (16 * 1024) + 2    # it stopped pulling, it did not drain the rest


def test_a_body_of_exactly_the_cap_is_not_marked_truncated() -> None:
    stream = ChunkStream(count=MAX_BODY_BYTES // (16 * 1024))
    result = make(lambda r: httpx.Response(200, headers={"content-type": "text/plain"}, stream=stream),
                  web_page_chars=10_000_000).fetch("https://a.org/exact")
    assert len(result.text) == MAX_BODY_BYTES and "larger than" not in result.text


def test_the_text_is_cut_to_web_page_chars_with_a_note() -> None:
    words = " ".join(f"word{i}" for i in range(2000))
    result = make(lambda r: html_page(f"<html><body><p>{words}</p></body></html>"), web_page_chars=120).fetch("https://a.org/")
    match = re.search(r"\n\(cut: shows 120 of (\d+) chars\)$", result.text)
    assert match is not None and int(match.group(1)) > 120
    assert len(result.text) < 120 + 40
    short = make(lambda r: html_page("<html><body><p>tiny</p></body></html>"), web_page_chars=120).fetch("https://a.org/")
    assert "cut:" not in short.text


# ---------------------------------------------------------------------------------------------
# cancel and the wall-clock limit
# ---------------------------------------------------------------------------------------------
def test_cancel_set_mid_stream_stops_the_read() -> None:
    cancel = threading.Event()
    stream = ChunkStream(count=200, on_chunk=lambda i: cancel.set() if i == 2 else None)
    fetcher = make(lambda r: httpx.Response(200, headers={"content-type": "text/plain"}, stream=stream))
    with pytest.raises(WebError, match="^cancelled"):
        fetcher.fetch("https://a.org/slow", cancel=cancel)
    assert stream.pulled <= 4


def test_cancel_set_before_the_call_never_touches_the_network() -> None:
    cancel = threading.Event()
    cancel.set()
    fetcher = make(lambda r: pytest.fail("must not be requested"), resolver=no_resolution)
    with pytest.raises(WebError, match="^cancelled"):
        fetcher.fetch("https://a.org/", cancel=cancel)
    with pytest.raises(WebError, match="^cancelled"):
        fetcher.search("anything", cancel=cancel)


def test_a_server_that_drips_forever_hits_the_wall_clock_limit() -> None:
    class Clock:
        now = 0.0

        def __call__(self) -> float:
            return self.now

    clock = Clock()
    stream = ChunkStream(count=1000, on_chunk=lambda i: setattr(clock, "now", clock.now + 5.0))
    cfg = Config()
    cfg.capabilities.web_timeout_s = 10.0
    fetcher = WebFetcher(cfg, transport=httpx.MockTransport(
        lambda r: httpx.Response(200, headers={"content-type": "text/plain"}, stream=stream)),
        resolver=resolver_for(), clock=clock)
    with pytest.raises(WebError, match="timed out after 30 s"):
        fetcher.fetch("https://a.org/drip")
    assert stream.pulled < 10


class RawServer:
    """A raw-socket HTTP server on 127.0.0.1. ``drip``: answers 200 ``text/plain`` and then sends one
    byte every ``interval`` seconds for ever (a per-read timeout never trips, a 16 KiB buffer never
    fills); ``silent``: accepts, reads the request and never answers."""

    def __init__(self, mode: str, interval: float = 0.2) -> None:
        self.mode, self.interval = mode, interval
        self.stop = threading.Event()
        self.client_gone = threading.Event()                 # drip: a send failed, the client hung up
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.bind(("127.0.0.1", 0))
        self.sock.listen(8)
        self.sock.settimeout(0.2)
        self.port: int = self.sock.getsockname()[1]
        self.conns: list[socket.socket] = []
        threading.Thread(target=self._serve, daemon=True, name="raw-server").start()

    def _serve(self) -> None:
        while not self.stop.is_set():
            try:
                conn, _addr = self.sock.accept()
            except socket.timeout:
                continue
            except OSError:
                return
            self.conns.append(conn)
            threading.Thread(target=self._handle, args=(conn,), daemon=True).start()

    def _handle(self, conn: socket.socket) -> None:
        try:
            conn.recv(65536)
            if self.mode == "drip":
                conn.sendall(b"HTTP/1.1 200 OK\r\nContent-Type: text/plain\r\nConnection: close\r\n\r\n")
                while not self.stop.wait(self.interval):
                    conn.sendall(b"x")
            else:
                self.stop.wait()
        except OSError:
            self.client_gone.set()
        finally:
            conn.close()

    def close(self) -> None:
        self.stop.set()
        for sock in [self.sock, *self.conns]:
            try:
                sock.close()
            except OSError:
                pass


@pytest.fixture
def raw_server(monkeypatch: pytest.MonkeyPatch) -> Iterator[Callable[..., RawServer]]:
    monkeypatch.setenv("NO_PROXY", "*")                       # a proxy in the environment must not sit between us and 127.0.0.1
    made: list[RawServer] = []

    def start(mode: str, interval: float = 0.2) -> RawServer:
        made.append(RawServer(mode, interval))
        return made[-1]

    yield start
    for server in made:
        server.close()


def local_fetcher(**caps: Any) -> WebFetcher:
    """A real (no mock transport) fetcher that may reach 127.0.0.1."""
    cfg = Config()
    cfg.capabilities.web_allow_local = True
    for key, value in caps.items():
        setattr(cfg.capabilities, key, value)
    return WebFetcher(cfg)


@pytest.mark.parametrize("mode", ["drip", "silent"])
def test_cancel_ends_a_fetch_whatever_the_server_does(raw_server: Callable[..., RawServer], mode: str) -> None:
    """A server that dribbles a byte at a time, and one that accepts and never answers: the caller
    gets 'cancelled' about a second after the cancel was set, not when the server gives up."""
    server = raw_server(mode)
    fetcher = local_fetcher(web_timeout_s=30.0)               # 3x = 90 s: only the cancel can end this
    cancel = threading.Event()
    threading.Timer(1.0, cancel.set).start()
    started = time.monotonic()
    try:
        with pytest.raises(WebError, match="^cancelled$"):
            fetcher.fetch(f"http://127.0.0.1:{server.port}/", cancel=cancel)
        assert time.monotonic() - started < 2.5
        if mode == "drip":
            # the abandoned fetch thread notices the abort on its next socket read and lets go of the
            # connection (a chunked iter_bytes(n) would keep it buffering for hours)
            assert server.client_gone.wait(3), "the abandoned fetch is still reading"
    finally:
        fetcher.close()


def test_the_wall_clock_limit_ends_a_server_that_drips_for_ever(raw_server: Callable[..., RawServer]) -> None:
    server = raw_server("drip", interval=0.2)
    fetcher = local_fetcher(web_timeout_s=0.5)                # per-read timeout 0.5 s, whole fetch 1.5 s
    started = time.monotonic()
    try:
        with pytest.raises(WebError, match=r"^timed out after 1\.5 s$"):
            fetcher.fetch(f"http://127.0.0.1:{server.port}/")
        assert time.monotonic() - started < 3.5
    finally:
        fetcher.close()


def test_the_wall_clock_limit_ends_a_silent_server_without_waiting_for_the_read_timeout(
        raw_server: Callable[..., RawServer]) -> None:
    server = raw_server("silent")
    fetcher = local_fetcher(web_timeout_s=30.0)
    fetcher._clock = _SkippingClock()                         # 3x30 s of wall clock pass in a moment
    started = time.monotonic()
    try:
        with pytest.raises(WebError, match=r"^timed out after 90 s$"):
            fetcher.fetch(f"http://127.0.0.1:{server.port}/")
        assert time.monotonic() - started < 3.0
    finally:
        fetcher.close()


class _SkippingClock:
    """A clock that runs 50x fast, so a 90 s wall-clock limit expires after under two real seconds."""

    def __init__(self) -> None:
        self.start = time.monotonic()

    def __call__(self) -> float:
        return self.start + (time.monotonic() - self.start) * 50


def test_a_fetch_runs_on_its_own_thread_and_returns_the_page() -> None:
    names: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        names.append(threading.current_thread().name)
        return html_page(PAGE)

    assert make(handler).fetch("https://a.org/").title == "Example Page"
    assert names == ["web-fetch"] and threading.current_thread().name != "web-fetch"


def test_an_unexpected_error_in_the_fetch_thread_reaches_the_caller() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise RuntimeError("boom")

    with pytest.raises(RuntimeError, match="boom"):
        make(handler).fetch("https://a.org/")


# ---------------------------------------------------------------------------------------------
# hostile HTML must not pin the worker
# ---------------------------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]


def run_isolated(body: str, limit: float = 40.0) -> None:
    """Run ``body`` in a fresh interpreter and FAIL if it needs more than ``limit`` seconds.

    The regressions guarded here are single parser / regex calls that hold the GIL for minutes to
    hours, so a watchdog thread in this process could not fire (and the suite would hang): the child
    is killed at the limit instead."""
    code = f"import sys\nsys.path.insert(0, {str(ROOT)!r})\n" + textwrap.dedent(body)
    try:
        done = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=limit, check=False)
    except subprocess.TimeoutExpired:
        pytest.fail(f"still running after {limit:g} s: the HTML conversion is quadratic again")
    assert done.returncode == 0, done.stderr[-2000:]


@pytest.mark.parametrize("unit", ["<a ", '<a b=">"', '<a b="'], ids=["open-tag", "quoted-gt", "open-quote"])
def test_html_to_text_is_linear_on_a_page_of_unterminated_tags(unit: str) -> None:
    run_isolated(f"""
        from exocortex.context import html_to_text
        unit = {unit!r}
        html_to_text(unit * (2 * 1024 * 1024 // len(unit)))
    """)


FETCH_HOSTILE = """
    import httpx
    from exocortex.config import Config
    from exocortex.webfetch import WebFetcher

    page = ({page}).encode()
    fetcher = WebFetcher(Config(), resolver=lambda host, port: [(2, 1, 6, "", ("93.184.216.34", port))],
                         transport=httpx.MockTransport(lambda r: httpx.Response(200, headers={{"content-type": "text/html"}}, content=page)))
    {call}
"""


def test_a_page_of_unclosed_titles_is_not_quadratic() -> None:
    run_isolated(FETCH_HOSTILE.format(page='"<title>" * (256 * 1024 // 7)', call='assert fetcher.fetch("https://a.org/").status == 200'))


def test_a_hostile_search_page_is_not_quadratic() -> None:
    run_isolated(FETCH_HOSTILE.format(page="""'<a b=">"' * (512 * 1024 // 8)""", call='assert fetcher.search("x") == []'))


def test_a_huge_unterminated_tail_is_dropped_but_everything_before_it_is_kept() -> None:
    text = html_to_text("<h1>Kept</h1><p>before</p>" + "<a " * 5000)
    assert "# Kept" in text and "before" in text
    # a short unterminated tail is still handled by the parser itself, as before
    assert "before" in html_to_text("<p>before</p><a href=")


def test_attribute_values_with_markup_in_them_convert_as_they_always_did() -> None:
    page = '<html><body><p data-content="<b>hi</b>">text <b>bold</b></p><a href="/x" title="a > b">link</a></body></html>'
    assert html_to_text(page).split() == ["text", "bold", "link"]
    assert make(lambda r: html_page(page)).fetch("https://a.org/").text.split() == ["text", "bold", "link"]


def test_markup_beyond_512_kib_is_not_converted_and_the_result_says_so() -> None:
    page = "<html><body><h1>Head</h1>" + "<p>filler</p>" * 60_000 + "</body></html>"          # about 800 KiB
    result = make(lambda r: html_page(page), web_page_chars=200).fetch("https://a.org/")
    assert result.text.startswith("# Head")
    assert result.text.endswith("(the page's HTML is larger than 512 KiB: only the first part was read)")


# ---------------------------------------------------------------------------------------------
# the production resolver
# ---------------------------------------------------------------------------------------------
def test_the_default_resolver_is_socket_getaddrinfo(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[str, int]] = []
    fake = resolver_for({"evil.example": "10.0.0.5", "ok.example": PUBLIC_IP})

    def spy(host: str, port: int, *a: Any, **k: Any) -> list[Any]:
        calls.append((host, port))
        return fake(host, port)

    monkeypatch.setattr(socket, "getaddrinfo", spy)           # the default is bound in __init__: patch first
    fetcher = WebFetcher(Config(), transport=httpx.MockTransport(lambda r: html_page(PAGE)))    # no resolver=
    with pytest.raises(WebError, match=r"^denied: evil\.example resolves to"):
        fetcher.fetch("http://evil.example/")
    assert fetcher.fetch("https://ok.example/").status == 200
    assert [c for c in calls if c[0] in ("evil.example", "ok.example")] == [("evil.example", 80), ("ok.example", 443)]


def test_localhost_is_refused_by_the_real_resolver() -> None:
    """Environment-dependent by nature: where localhost does not resolve the error is 'could not resolve'."""
    fetcher = WebFetcher(Config(), transport=httpx.MockTransport(lambda r: pytest.fail("must not be requested")))
    with pytest.raises(WebError, match=r"^(denied: localhost resolves to|could not resolve localhost)"):
        fetcher.fetch("http://localhost/")


# ---------------------------------------------------------------------------------------------
# search
# ---------------------------------------------------------------------------------------------
DDG_PAGE = """<!DOCTYPE html><html><head><title>hello at DuckDuckGo</title></head><body>
<div class="serp__results"><div id="links" class="results">
 <div class="result results_links results_links_deep result--ad">
  <h2 class="result__title"><a rel="nofollow" class="result__a" href="https://duckduckgo.com/y.js?ad_domain=shop.example&amp;ad_provider=bing">Buy things now</a></h2>
  <a class="result__snippet" href="https://duckduckgo.com/y.js?ad=1">An advertisement.</a>
 </div>
 <div class="result results_links results_links_deep web-result">
  <div class="links_main links_deep result__body">
   <h2 class="result__title"><a rel="nofollow" class="result__a" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.com%2Fdocs%3Fa%3D1%26b%3D2&amp;rut=abc123">Example <b>Docs</b></a></h2>
   <div class="result__extras"><a class="result__url" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.com%2Fdocs">example.com/docs</a></div>
   <a class="result__snippet" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.com%2Fdocs">The <b>official</b> documentation &amp; guides.</a>
  </div>
 </div>
 <div class="result results_links results_links_deep web-result">
  <div class="links_main links_deep result__body">
   <h2 class="result__title"><a rel="nofollow" class="result__a" href="https://other.org/page">Other   Page</a></h2>
   <div class="result__snippet">Plain <i>div</i> snippet
      over two lines.</div>
  </div>
 </div>
 <div class="result results_links web-result">
  <h2 class="result__title"><a class="result__a" href="https://example.com/docs?a=1&amp;b=2">Duplicate of the first</a></h2>
 </div>
 <div class="result results_links web-result">
  <h2 class="result__title"><a class="result__a" href="https://blocked.net/x">Blocked Site</a></h2>
  <a class="result__snippet" href="https://blocked.net/x">Should be dropped.</a>
 </div>
 <div class="result results_links web-result">
  <h2 class="result__title"><a class="result__a" href="javascript:void(0)">Script link</a></h2>
 </div>
 <div class="result results_links web-result">
  <h2 class="result__title"><a class="result__a" href="https://third.org/">Third</a></h2>
 </div>
</div></div></body></html>"""


def search_handler(markup: str, seen: list[httpx.Request] | None = None) -> Handler:
    def handler(request: httpx.Request) -> httpx.Response:
        if seen is not None:
            seen.append(request)
        return html_page(markup)

    return handler


def test_search_parses_duckduckgo_results_and_unwraps_uddg() -> None:
    seen: list[httpx.Request] = []
    hits = make(search_handler(DDG_PAGE, seen), web_blocked_domains=["blocked.net"], web_search_results=10
                ).search("hello world & more")
    assert [h.url for h in hits] == ["https://example.com/docs?a=1&b=2", "https://other.org/page", "https://third.org/"]
    assert all(isinstance(h, SearchHit) for h in hits)
    assert hits[0].title == "Example Docs"
    assert hits[0].snippet == "The official documentation & guides."
    assert hits[1].title == "Other Page" and hits[1].snippet == "Plain div snippet over two lines."
    assert hits[2].snippet == ""
    # the request: the configured template with the query form-encoded
    assert len(seen) == 1
    assert seen[0].url.host == "html.duckduckgo.com" and seen[0].url.path == "/html/"
    assert seen[0].url.params["q"] == "hello world & more"
    assert "q=hello+world+%26+more" in str(seen[0].url)


def test_search_stops_at_web_search_results() -> None:
    hits = make(search_handler(DDG_PAGE), web_search_results=2).search("x")
    assert [h.url for h in hits] == ["https://example.com/docs?a=1&b=2", "https://other.org/page"]


def test_search_drops_the_engines_own_ads_and_hits_outside_the_domain_lists() -> None:
    urls = [h.url for h in make(search_handler(DDG_PAGE), web_allowed_domains=["html.duckduckgo.com", "other.org"],
                                web_search_results=10).search("x")]
    assert urls == ["https://other.org/page"]          # example.com / third.org / blocked.net: not allowed -> dropped


def test_search_query_is_cut_to_200_chars() -> None:
    seen: list[httpx.Request] = []
    make(search_handler(DDG_PAGE, seen)).search("q" * 500)
    assert seen[0].url.params["q"] == "q" * 200


def test_search_uses_the_configured_url_template() -> None:
    seen: list[httpx.Request] = []
    make(search_handler(DDG_PAGE, seen), web_search_url="https://search.example.net/s?term={query}&safe=1").search("a b")
    assert str(seen[0].url) == "https://search.example.net/s?term=a+b&safe=1"
    with pytest.raises(WebError, match="not a valid template"):
        make(search_handler(DDG_PAGE), web_search_url="https://x.net/?q={nope}").search("a")
    with pytest.raises(WebError, match="empty search query"):
        make(search_handler(DDG_PAGE)).search("   ")


GENERIC_PAGE = """<html><body>
<a href="/">Home</a>
<a href="https://duckduckgo.com/about">About the engine</a>
<a href="#top">Jump</a>
<a href="mailto:me@example.com">Mail</a>
<a href="javascript:void(0)">Script</a>
<a href="https://one.org/a">  First   result </a>
<a href="https://one.org/a">First result again</a>
<a href="https://two.org/b"><img src="x.png"></a>
<a href="//three.org/c">Third (protocol relative)</a>
<a href="relative/page">Fourth (relative)</a>
</body></html>"""


def test_search_falls_back_to_every_link_with_text_when_there_is_no_result_markup() -> None:
    hits = make(search_handler(GENERIC_PAGE), web_search_results=10).search("x")
    # "Home", "About" and the relative link all point back at the engine itself; "#top", mailto:,
    # javascript: and the text-less image link are not results; the duplicate is folded away
    assert [(h.title, h.url) for h in hits] == [
        ("First result", "https://one.org/a"),
        ("Third (protocol relative)", "https://three.org/c"),
    ]
    assert all(h.snippet == "" for h in hits)


def test_only_the_search_engines_own_domain_counts_as_its_own_links() -> None:
    page = '<a href="https://www.bbc.co.uk/news">BBC</a><a href="https://search.example.co.uk/about">About us</a>'            '<a href="https://example.co.uk/x">Example</a><a href="https://duckduckgo.com/about">DDG</a>'
    hits = make(search_handler(page), web_search_url="https://search.example.co.uk/?q={query}",
                web_search_results=10).search("x")
    # the engine's own host is skipped; its parent domain, other .co.uk sites and duckduckgo.com are not
    assert [h.url for h in hits] == ["https://www.bbc.co.uk/news", "https://example.co.uk/x", "https://duckduckgo.com/about"]


def test_search_goes_through_the_same_policy_as_a_fetch() -> None:
    page = search_handler(DDG_PAGE)
    # a non-empty allowed list must contain the search host itself
    with pytest.raises(WebError, match="html.duckduckgo.com is not on the allowed domain list"):
        make(page, web_allowed_domains=["example.com"]).search("x")
    # a parent domain matches the search host; the hits are filtered by the same list
    allowed = make(page, web_allowed_domains=["duckduckgo.com", "other.org"]).search("x")
    assert [h.url for h in allowed] == ["https://other.org/page"]
    with pytest.raises(WebError, match="blocked domain list"):
        make(page, web_blocked_domains=["duckduckgo.com"]).search("x")
    with pytest.raises(WebError, match="^denied: html.duckduckgo.com resolves to a private address"):
        make(page, resolver=resolver_for(default="10.0.0.9")).search("x")


def test_search_follows_its_redirects_under_the_policy() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.host == "html.duckduckgo.com":
            return redirect("https://blocked.net/search")
        return html_page(DDG_PAGE)

    with pytest.raises(WebError, match="blocked.net is on the blocked domain list"):
        make(handler, web_blocked_domains=["blocked.net"]).search("x")


def test_search_rejects_a_non_html_answer() -> None:
    with pytest.raises(WebError, match="not HTML"):
        make(lambda r: httpx.Response(200, headers={"content-type": "application/json"}, content=b"{}")).search("x")
