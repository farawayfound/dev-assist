"""Fetch a web page and search the web for the model (2026-10-07, capabilities).

The network half of the "Fetch web content" switch: :class:`WebFetcher` turns a URL into
readable text (:meth:`WebFetcher.fetch`) and a query into a handful of hits
(:meth:`WebFetcher.search`). It is Qt-free and model-free (stdlib + ``httpx`` + the landed
``exocortex.context.html_to_text``); ``exocortex.capabilities`` imports it lazily and decides WHEN a
call may happen (the ask box only, the exfiltration guard, the per-answer budget). This module
decides WHAT a call may reach, from ``cfg.capabilities`` read live on every call:

1. **Scheme** -- ``http`` / ``https`` only (``file:``, ``data:``, ``ftp:`` ... are refused).
2. **Domain lists** -- the host must not match ``web_blocked_domains`` and, when
   ``web_allowed_domains`` is not empty, must match it. Matching is by whole labels:
   ``example.com`` matches ``docs.example.com`` and never ``notexample.com``. The web SEARCH goes
   through the same check, so with a non-empty allowed list the search host
   (``html.duckduckgo.com`` by default) has to be on it too, while an EMPTY allowed list lets it
   through; every search hit's URL must pass the lists as well (a hit that does not is dropped,
   not refused).
3. **Address** -- unless ``web_allow_local``, the host is resolved and EVERY address must be a
   public one: loopback, private, link-local, multicast, reserved, unspecified, IPv4-mapped
   (``::ffff:a.b.c.d``) and IPv4-compatible IPv6, and ``100.64.0.0/10`` (carrier-grade NAT) are
   refused. A literal IP is checked the same way, and a host ``ipaddress`` cannot parse but that
   LOOKS numeric (``0x7f000001``, ``2130706433``, ``127.1``) is refused outright: the system
   resolver would read those as IPv4 addresses.

Every redirect hop is run through 1-3 again (``follow_redirects`` is off; at most 5 hops are
followed by hand), so an allowed page cannot bounce the fetcher to ``http://127.0.0.1/``.

**Known limitation: DNS rebinding.** The address check resolves the host, then ``httpx`` resolves
it AGAIN to connect; a hostile DNS server can answer "public" the first time and "127.0.0.1" the
second. Pinning the checked address would break TLS name checks with a plain ``httpx`` client, so
it is documented rather than fixed. The domain lists are the real control on a machine where that
matters, and the switch defaults to OFF.

**Cancel and the wall-clock limit.** ``web_timeout_s`` is the ``httpx`` timeout, and the whole fetch
(all hops) gets three times that as a wall-clock limit (a server dripping a byte at a time never
trips a per-read timeout). The fetch itself -- DNS, connect, the header wait, every hop and the
body -- runs on a short-lived ``web-fetch`` thread while the CALLING thread polls ``cancel`` and
that limit every 100 ms, so either one returns within about 0.1 s whatever the server is doing
(silent, dripping, a DNS lookup that hangs). The abandoned thread is told to stop (the body loop
looks at that on every socket read: ``iter_bytes()`` is not chunked) and otherwise dies at the
``httpx`` timeout; ``close()`` still unblocks it at shutdown.

Bodies are read streamed and cut at 2 MiB. ``Accept-Encoding: identity`` keeps a compressed-bomb
from expanding past the cap. ``text/html`` goes through ``html_to_text`` (fed at most 512 KiB of
markup: the stdlib parser costs time in proportion to its input and cannot be interrupted),
``text/*`` and ``application/json`` are kept as they are, anything else (an image, a PDF, a zip)
is refused before its body is read. The text is cut to ``web_page_chars``.

Policy refusals raise :class:`WebError` with a message starting ``denied: ``; every other failure
(HTTP error, timeout, unreachable host) is a :class:`WebError` with a readable message too. Nothing
else is raised and a traceback never reaches the model.
"""
from __future__ import annotations

import codecs
import html.parser
import ipaddress
import logging
import re
import socket
import threading
import time
import urllib.parse
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import httpx

from exocortex import __version__
from exocortex.config import Config
from exocortex.context import ContextError, html_to_text

log = logging.getLogger(__name__)

#: The most bytes read from one response; the rest is dropped (and the result says so).
MAX_BODY_BYTES = 2 * 1024 * 1024
#: Redirect hops followed by hand. The 6th redirect of a chain is refused.
MAX_REDIRECTS = 5
#: The most HTML markup handed to ``html_to_text`` / the result parser (the rest is not converted).
_MAX_HTML_CHARS = 512 * 1024
#: ``HTMLParser.close()`` rescans an unterminated tag from every later position (quadratic), while
#: ``feed()`` is linear. A tail this long is one tag that never ended, which HTML5 drops too.
_MAX_OPEN_TAIL = 4096
#: How often the calling thread looks at ``cancel`` and the wall-clock limit while a fetch runs.
_POLL_S = 0.1
_REDIRECT_STATUSES = frozenset({301, 302, 303, 307, 308})
#: Search titles / snippets are cut to these (the runner sanitises and cuts the whole result).
_TITLE_CHARS = 200
_SNIPPET_CHARS = 400
_CGNAT = ipaddress.ip_network("100.64.0.0/10")
_NAT64 = ipaddress.ip_network("64:ff9b::/96")
#: A host the system resolver would read as an IPv4 address in some legacy form: dotted parts that
#: are all decimal / octal / 0x-hex ("127.1", "0177.0.0.1", "0x7f000001", "2130706433").
_NUMERIC_HOST = re.compile(r"(?:0x[0-9a-f]*|[0-9]+)(?:\.(?:0x[0-9a-f]*|[0-9]+))*\.?", re.IGNORECASE)
_META_CHARSET = re.compile(rb"<meta[^>]+charset\s*=\s*[\"']?\s*([A-Za-z0-9_.:-]+)", re.IGNORECASE)
# Bounded on purpose: an unbounded ``(.*?)`` makes every unclosed "<title>" rescan the rest of the page.
_TITLE_TAG = re.compile(r"<title[^<>]{0,200}>(.{0,400}?)</title>", re.IGNORECASE | re.DOTALL)
_SNIPPET_CLASSES = frozenset({"result__snippet", "result-snippet"})

_HEADERS = {
    # DuckDuckGo's HTML endpoint serves a bot page to unfamiliar clients; a browser-shaped agent with
    # the app's name on the end is what the site expects from a person's machine.
    "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) "
                   f"Chrome/126.0 Safari/537.36 ExoCortex/{__version__}"),
    "Accept": "text/html,application/xhtml+xml,text/plain;q=0.9,application/json;q=0.8,*/*;q=0.1",
    "Accept-Language": "en",
    "Accept-Encoding": "identity",
}


class WebError(Exception):
    """A readable failure of a fetch or a search. A policy refusal starts with ``denied: ``."""


@dataclass(frozen=True)
class FetchResult:
    """One fetched page: its readable ``text`` (already cut to ``web_page_chars``), its ``title``
    ("" when it has none), the URL actually served after redirects and the HTTP ``status``."""

    text: str
    title: str
    final_url: str
    status: int


@dataclass(frozen=True)
class SearchHit:
    """One search result."""

    title: str
    url: str
    snippet: str


@dataclass
class _Page:
    """A response body as read by :meth:`WebFetcher._get`, before any conversion."""

    body: bytes
    media: str
    charset: str
    kind: str          # "html" | "text" | "json"
    status: int
    final_url: str
    truncated: bool


def _norm_domain(entry: object) -> str:
    """A domain-list entry as a bare lower-case host: ``*.Example.com`` / ``https://example.com/x``
    / ``.example.com.`` all become ``example.com``. "" for something unusable."""
    text = str(entry).strip().lower()
    if "://" in text:
        text = urllib.parse.urlsplit(text).hostname or ""
    text = text.split("/")[0]
    return text.lstrip("*").lstrip(".").rstrip(".")


def _host_matches(host: str, domain: str) -> bool:
    """Whole-label suffix match: ``docs.example.com`` matches ``example.com``; ``notexample.com`` does not."""
    return host == domain or host.endswith("." + domain)


def _address_problem(ip: ipaddress.IPv4Address | ipaddress.IPv6Address) -> str | None:
    """Why ``ip`` may not be fetched ("" -> None when it is a public address)."""
    if isinstance(ip, ipaddress.IPv6Address):
        if ip.ipv4_mapped is not None:
            return "an IPv4-mapped IPv6 address"
        if int(ip) >> 32 == 0 and int(ip) > 1:
            return "an IPv4-compatible IPv6 address"
        if ip in _NAT64:
            # DNS64 hands out these for ordinary sites on IPv6-only networks: judge the IPv4 inside.
            return _address_problem(ipaddress.IPv4Address(int(ip) & 0xFFFFFFFF))
    if ip.is_loopback:
        return "a loopback address"
    if ip.is_unspecified:
        return "an unspecified address"
    if ip.is_link_local:
        return "a link-local address"
    if ip.is_multicast:
        return "a multicast address"
    if ip.is_private:
        return "a private address"
    if ip.is_reserved:
        return "a reserved address"
    if isinstance(ip, ipaddress.IPv4Address) and ip in _CGNAT:
        return "a carrier-grade NAT address"
    if not ip.is_global:
        return "a non-public address"
    return None


def _parse_ip(host: str) -> ipaddress.IPv4Address | ipaddress.IPv6Address | None:
    try:
        return ipaddress.ip_address(host)
    except ValueError:
        return None


def _split_content_type(header: str) -> tuple[str, str]:
    """``'text/html; charset=UTF-8'`` -> ``('text/html', 'utf-8')``."""
    media, _, params = header.partition(";")
    charset = ""
    for part in params.split(";"):
        key, _, value = part.partition("=")
        if key.strip().lower() == "charset":
            charset = value.strip().strip("\"'").lower()
    return media.strip().lower(), charset


def _classify(media: str) -> str:
    """The kind of body a media type is ("html" / "text" / "json"), or "" when it is not text."""
    if media in ("text/html", "application/xhtml+xml"):
        return "html"
    if media == "application/json" or (media.startswith("application/") and media.endswith("+json")):
        return "json"
    if media.startswith("text/"):
        return "text"
    return ""


def _decode(body: bytes, charset: str, *, html_page: bool) -> str:
    """Bytes -> text with the declared charset (header, else the page's own ``<meta>``), UTF-8
    otherwise; undecodable bytes become U+FFFD. A byte-order mark is dropped."""
    name = charset
    if not name and html_page:
        match = _META_CHARSET.search(body[:2048])
        name = match.group(1).decode("ascii", "ignore").lower() if match else ""
    encoding = "utf-8"
    if name:
        try:
            encoding = codecs.lookup(name).name
        except LookupError:
            log.debug("unknown charset %r, decoding as utf-8", name)
    return body.decode(encoding, errors="replace").lstrip("﻿")


def _page_title(text: str, markup: str) -> str:
    """The title of a converted HTML page: the first ``# `` line (``html_to_text`` renders both
    ``<title>`` and ``<h1>`` that way), else the raw ``<title>`` element."""
    for line in text.split("\n"):
        if line.startswith("# ") and line[2:].strip():
            return line[2:].strip()[:_TITLE_CHARS]
    match = _TITLE_TAG.search(markup)
    if match:
        return " ".join(html.unescape(match.group(1)).split())[:_TITLE_CHARS]
    return ""


class _ResultParser(html.parser.HTMLParser):
    """Collects a search page's results.

    DuckDuckGo's HTML page marks each hit as ``<a class="result__a" href=...>title</a>`` followed by
    an element with class ``result__snippet``. ``hits`` holds those (href, title, snippet);
    ``links`` holds EVERY anchor with text, for the generic fallback when a page has no
    ``result__a`` (another search engine, or DuckDuckGo changing its markup).
    """

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.hits: list[dict[str, str]] = []
        self.links: list[dict[str, str]] = []
        self._anchor: dict[str, Any] | None = None
        self._snip_tag = ""
        self._snip_depth = 0
        self._snip_parts: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if self._snip_tag:
            if tag == self._snip_tag:
                self._snip_depth += 1
            return
        values = dict(attrs)
        classes = (values.get("class") or "").split()
        if _SNIPPET_CLASSES.intersection(classes):
            self._close_anchor()
            self._snip_tag, self._snip_depth, self._snip_parts = tag, 1, []
        elif tag == "a":
            self._close_anchor()
            href = values.get("href")
            if href:
                self._anchor = {"href": href, "text": [], "result": "result__a" in classes}

    def handle_endtag(self, tag: str) -> None:
        if self._snip_tag:
            if tag == self._snip_tag:
                self._snip_depth -= 1
                if self._snip_depth <= 0:
                    self._close_snippet()
            return
        if tag == "a":
            self._close_anchor()

    def handle_data(self, data: str) -> None:
        if self._snip_tag:
            self._snip_parts.append(data)
        elif self._anchor is not None:
            self._anchor["text"].append(data)

    def finish(self) -> None:
        """Close whatever the markup left open at the end of the page."""
        self._close_snippet()
        self._close_anchor()

    def _close_snippet(self) -> None:
        if not self._snip_tag:
            return
        text = " ".join("".join(self._snip_parts).split())
        self._snip_tag, self._snip_depth, self._snip_parts = "", 0, []
        # The snippet belongs to the hit just before it, once.
        if self.hits and not self.hits[-1]["snippet"]:
            self.hits[-1]["snippet"] = text

    def _close_anchor(self) -> None:
        anchor, self._anchor = self._anchor, None
        if anchor is None:
            return
        text = " ".join("".join(anchor["text"]).split())
        if anchor["result"]:
            self.hits.append({"href": anchor["href"], "title": text, "snippet": ""})
        if text:
            self.links.append({"href": anchor["href"], "title": text, "snippet": ""})


class WebFetcher:
    """Fetch pages and search the web under the ``[capabilities]`` web policy (module docstring).

    ``cfg`` is read live on every call (a changed domain list applies to the next fetch);
    ``transport`` lets tests substitute ``httpx.MockTransport``; ``resolver`` has the shape of
    ``socket.getaddrinfo`` (``resolver(host, port)`` -> list of ``(family, type, proto, canon,
    sockaddr)``; a plain address string is accepted too) and is how tests control what a host
    resolves to; ``clock`` is the monotonic clock of the wall-clock limit. Thread-safe: one
    instance serves the assist worker and the engine's threads.
    """

    def __init__(self, cfg: Config, *, transport: httpx.BaseTransport | None = None,
                 resolver: Callable[..., Any] | None = None,
                 clock: Callable[[], float] = time.monotonic) -> None:
        self._cfg = cfg
        self._resolver = resolver if resolver is not None else socket.getaddrinfo
        self._clock = clock
        self._client = httpx.Client(headers=_HEADERS, follow_redirects=False, transport=transport)

    # ------------------------------------------------------------------------------ public API
    def fetch(self, url: str, *, cancel: threading.Event | None = None) -> FetchResult:
        """Fetch ``url`` and return its readable text. Raises :class:`WebError`."""
        cc = self._cfg.capabilities
        page = self._get(url, cancel=cancel)
        text = _decode(page.body, page.charset, html_page=page.kind == "html")
        title = ""
        markup_cut = False
        if page.kind == "html":
            if len(text) > _MAX_HTML_CHARS:
                text, markup_cut = text[:_MAX_HTML_CHARS], True
            try:
                converted = html_to_text(text)
            except ContextError as exc:
                raise WebError(f"could not read the page: {exc}") from exc
            title = _page_title(converted, text)
            text = converted
        text = re.sub(r"\n{3,}", "\n\n", text.replace("\r\n", "\n").replace("\r", "\n")).strip()
        limit = max(1, int(cc.web_page_chars))
        total = len(text)
        if total > limit:
            text = f"{text[:limit].rstrip()}\n(cut: shows {limit} of {total} chars)"
        if page.truncated:
            text += f"\n(the page is larger than {MAX_BODY_BYTES // (1024 * 1024)} MiB: only the first part was read)"
        elif markup_cut:
            text += f"\n(the page's HTML is larger than {_MAX_HTML_CHARS // 1024} KiB: only the first part was read)"
        return FetchResult(text=text, title=title, final_url=page.final_url, status=page.status)

    def search(self, query: str, *, cancel: threading.Event | None = None) -> list[SearchHit]:
        """Search the web for ``query`` (cut to 200 chars) and return at most ``web_search_results``
        hits. The search page goes through the same policy as any fetch; hits whose URL fails the
        domain lists are dropped. Raises :class:`WebError`."""
        cc = self._cfg.capabilities
        text = str(query).strip()[:200]
        if not text:
            raise WebError("empty search query")
        try:
            url = cc.web_search_url.format(query=urllib.parse.quote_plus(text))
        except (KeyError, IndexError, ValueError) as exc:
            raise WebError(f"web_search_url is not a valid template (use {{query}} once): {exc}") from exc
        page = self._get(url, cancel=cancel)
        if page.kind != "html":
            raise WebError(f"the search page was not HTML ({page.media})")
        markup = _decode(page.body, page.charset, html_page=True)[:_MAX_HTML_CHARS]
        parser = _ResultParser()
        try:
            parser.feed(markup)
            if len(parser.rawdata) <= _MAX_OPEN_TAIL:
                parser.close()
        except Exception as exc:  # noqa: BLE001 - html.parser is lenient; a crash must still be a WebError
            raise WebError(f"could not read the search results: {exc}") from exc
        parser.finish()
        candidates = parser.hits or parser.links
        own = self._own_domains(url)
        hits: list[SearchHit] = []
        seen: set[str] = set()
        limit = max(1, int(cc.web_search_results))
        for cand in candidates:
            target = self._hit_url(cand["href"], page.final_url)
            if target is None or target in seen:
                continue
            host = (urllib.parse.urlsplit(target).hostname or "").lower().rstrip(".")
            if any(_host_matches(host, d) for d in own):
                continue                       # the engine's own navigation / ad redirects
            reason = self._domain_problem(host)
            if reason:
                log.debug("search hit dropped: %s", reason)
                continue
            seen.add(target)
            hits.append(SearchHit(title=(cand["title"] or target)[:_TITLE_CHARS], url=target,
                                  snippet=cand["snippet"][:_SNIPPET_CHARS]))
            if len(hits) >= limit:
                break
        return hits

    def close(self) -> None:
        """Release the connection pool."""
        try:
            self._client.close()
        except Exception:  # noqa: BLE001 - closing is best effort
            log.debug("closing the web client failed", exc_info=True)

    # ----------------------------------------------------------------------------------- policy
    def _domain_problem(self, host: str) -> str:
        """Why the domain lists refuse ``host`` ("" when they let it through)."""
        cc = self._cfg.capabilities
        for entry in cc.web_blocked_domains:
            domain = _norm_domain(entry)
            if domain and _host_matches(host, domain):
                return f"denied: {host} is on the blocked domain list"
        allowed = [d for d in (_norm_domain(e) for e in cc.web_allowed_domains) if d]
        if allowed and not any(_host_matches(host, d) for d in allowed):
            return f"denied: {host} is not on the allowed domain list"
        return ""

    def _check_address(self, host: str, port: int) -> None:
        """Refuse a host that is, or resolves to, a non-public address (``web_allow_local`` lifts it)."""
        if self._cfg.capabilities.web_allow_local:
            return
        literal = _parse_ip(host)
        if literal is not None:
            problem = _address_problem(literal)
            if problem:
                raise WebError(f"denied: {host} is {problem}")
            return
        if ":" in host:
            raise WebError(f"denied: {host} is not a valid IP address")
        if _NUMERIC_HOST.fullmatch(host):
            raise WebError(f"denied: {host} is a numeric IP address in a form that is not accepted")
        try:
            infos = self._resolver(host, port)
        except (OSError, UnicodeError) as exc:
            raise WebError(f"could not resolve {host}: {exc}") from exc
        addresses: list[str] = []
        for info in infos or ():
            addr = info[4][0] if isinstance(info, tuple) else info
            addr = str(addr).split("%")[0]
            if addr not in addresses:
                addresses.append(addr)
        if not addresses:
            raise WebError(f"could not resolve {host}")
        for addr in addresses:
            ip = _parse_ip(addr)
            if ip is None:
                raise WebError(f"denied: {host} resolved to something that is not an IP address ({addr})")
            problem = _address_problem(ip)
            if problem:
                raise WebError(f"denied: {host} resolves to {problem} ({addr})")

    def _check_url(self, raw: str) -> httpx.URL:
        """Run the whole policy on one URL (the request's, and each redirect hop's) -> the parsed URL."""
        text = str(raw).strip()
        if not text:
            raise WebError("denied: empty URL")
        try:
            url = httpx.URL(text)
        except (httpx.InvalidURL, ValueError, TypeError) as exc:
            raise WebError(f"denied: not a valid URL ({exc})") from exc
        if url.scheme not in ("http", "https"):
            raise WebError(f"denied: scheme '{url.scheme or text.split(':')[0]}' is not allowed (http and https only)")
        host = url.host.lower().rstrip(".")
        if not host:
            raise WebError("denied: the URL has no host")
        if url.userinfo:
            raise WebError("denied: URLs carrying a user name or password are not allowed")
        reason = self._domain_problem(host)
        if reason:
            raise WebError(reason)
        self._check_address(host, url.port or (443 if url.scheme == "https" else 80))
        return url

    @staticmethod
    def _own_domains(search_url: str) -> tuple[str, ...]:
        """The domains of the search engine itself, whose own links (navigation, ads) are not hits:
        the host the search was sent to, plus ``duckduckgo.com`` for DuckDuckGo (its ads redirect
        through ``duckduckgo.com/y.js``, not through the ``html.`` host)."""
        try:
            host = (httpx.URL(search_url).host or "").lower().rstrip(".")
        except (httpx.InvalidURL, ValueError):
            return ()
        if not host:
            return ()
        return (host, "duckduckgo.com") if host.endswith("duckduckgo.com") else (host,)

    @staticmethod
    def _hit_url(href: str, base: str) -> str | None:
        """A result link as an absolute http(s) URL, DuckDuckGo's ``/l/?uddg=<real url>`` redirect
        unwrapped; None for anything else (``javascript:``, ``mailto:``, a bare fragment)."""
        try:
            absolute = urllib.parse.urljoin(base, href.strip())
            parts = urllib.parse.urlsplit(absolute)
            host = (parts.hostname or "").lower()
            if host.endswith("duckduckgo.com"):
                wrapped = urllib.parse.parse_qs(parts.query).get("uddg")
                if wrapped:
                    absolute = wrapped[0]
                    parts = urllib.parse.urlsplit(absolute)
            if parts.scheme not in ("http", "https") or not parts.hostname:
                return None
        except ValueError:
            return None
        return absolute

    # --------------------------------------------------------------------------------- transport
    @staticmethod
    def _check_cancel(*events: threading.Event | None) -> None:
        if any(event is not None and event.is_set() for event in events):
            raise WebError("cancelled")

    def _get(self, url: str, *, cancel: threading.Event | None) -> _Page:
        """GET ``url`` (every hop, the body) on a ``web-fetch`` thread and wait for it from THIS one,
        looking at ``cancel`` and the wall-clock limit every 100 ms (module docstring). The thread
        is never joined: on a cancel / timeout it is told to stop and left to finish by itself."""
        self._check_cancel(cancel)
        timeout = max(0.5, float(self._cfg.capabilities.web_timeout_s))
        deadline = self._clock() + timeout * 3
        abort = threading.Event()
        done = threading.Event()
        outcome: list[_Page | BaseException] = []

        def work() -> None:
            try:
                outcome.append(self._get_blocking(url, cancel=cancel, abort=abort, timeout=timeout, deadline=deadline))
            except BaseException as exc:  # noqa: BLE001 - handed to the calling thread, which re-raises it
                outcome.append(exc)
            finally:
                done.set()

        threading.Thread(target=work, daemon=True, name="web-fetch").start()
        while not done.wait(_POLL_S):
            if cancel is not None and cancel.is_set():
                abort.set()
                raise WebError("cancelled")
            if self._clock() > deadline:
                abort.set()
                raise WebError(f"timed out after {timeout * 3:g} s")
        result = outcome[0]
        if isinstance(result, BaseException):
            raise result
        return result

    def _get_blocking(self, url: str, *, cancel: threading.Event | None, abort: threading.Event,
                      timeout: float, deadline: float) -> _Page:
        """GET ``url`` by hand-followed redirects (the policy runs on every hop) and read the body.
        Blocks; ``_get`` runs it on its own thread. ``abort`` is that thread's stop signal."""
        current = url
        for hop in range(MAX_REDIRECTS + 1):
            self._check_cancel(cancel, abort)
            if self._clock() > deadline:
                raise WebError(f"timed out after {timeout * 3:g} s")
            target = self._check_url(current)
            try:
                with self._client.stream("GET", target, timeout=timeout) as resp:
                    if resp.status_code in _REDIRECT_STATUSES:
                        location = resp.headers.get("location", "").strip()
                        if not location:
                            raise WebError(f"the server redirected (HTTP {resp.status_code}) without saying where")
                        if hop >= MAX_REDIRECTS:
                            raise WebError(f"denied: more than {MAX_REDIRECTS} redirects")
                        try:
                            current = str(target.join(location))
                        except (httpx.InvalidURL, ValueError) as exc:
                            raise WebError(f"denied: redirected to an invalid URL ({exc})") from exc
                        continue
                    return self._read(resp, target, (cancel, abort), deadline, timeout)
            except WebError:
                raise
            except httpx.InvalidURL as exc:
                # httpx BUILDS the next request of a redirect even with follow_redirects off, and a
                # Location like "data:..." makes that fail before we ever see the response.
                raise WebError(f"denied: the server sent an invalid redirect or URL ({exc})") from exc
            except httpx.TimeoutException as exc:
                raise WebError(f"timed out after {timeout:g} s") from exc
            except httpx.HTTPError as exc:
                raise WebError(f"request failed: {str(exc) or type(exc).__name__}") from exc
        raise WebError(f"denied: more than {MAX_REDIRECTS} redirects")   # unreachable; keeps the type checker calm

    def _read(self, resp: httpx.Response, target: httpx.URL, cancel: tuple[threading.Event | None, ...],
              deadline: float, timeout: float) -> _Page:
        """The final response: status and content type judged BEFORE the body, then the body streamed."""
        status = resp.status_code
        if status >= 400:
            raise WebError(f"the server answered HTTP {status}")
        if status < 200 or status >= 300:
            raise WebError(f"unexpected HTTP {status}")
        header = resp.headers.get("content-type", "")
        media, charset = _split_content_type(header)
        kind = _classify(media)
        if not kind:
            raise WebError(f"denied: not a text page ({media or 'no content type'})")
        body = bytearray()
        truncated = False
        # No chunk size: iter_bytes(n) buffers until n bytes arrived, so a byte-dripping server would
        # keep the checks below from ever running. Unchunked, they run on every socket read.
        for chunk in resp.iter_bytes():
            self._check_cancel(*cancel)
            if self._clock() > deadline:
                raise WebError(f"timed out after {timeout * 3:g} s")
            room = MAX_BODY_BYTES - len(body)
            if len(chunk) > room:
                body += chunk[:room]
                truncated = True
                break
            body += chunk
        self._check_cancel(*cancel)
        if b"\x00" in body[:4096]:
            raise WebError(f"denied: not a text page (binary content labelled {media})")
        return _Page(body=bytes(body), media=media, charset=charset, kind=kind, status=status,
                     final_url=str(target), truncated=truncated)
