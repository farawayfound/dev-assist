"""Tests for exocortex.config's [llm] ``server_url`` field (2026-09-27, portable-install): the
``effective_server_url`` helper, the ``base_url``/``ollama_url`` property shim built over it, and
``load_config``'s legacy ``base_url``/``ollama_url`` -> ``server_url`` migration.

Two URL fields ([llm].base_url for the openai backend, [llm].ollama_url for ollama) collapsed into
one ([llm].server_url) because a machine only ever talks to one such external server at a time.
Old config.toml files and old call sites (``cfg.llm.base_url = ...``) must keep working exactly as
before -- that is what every test below pins down.

review round two, 2026-09-27 (C1): also pins the loud/recoverable handling of the case where BOTH
legacy keys were customised to different addresses -- a WARNING naming both values plus the
config.toml lines to recover the dropped one, and the dropped value on the non-persisted
``LLMConfig.dropped_legacy_url`` (excluded from ``to_toml``, like ``Config.path``).
"""
from __future__ import annotations

import dataclasses
import logging
from pathlib import Path

import pytest

from exocortex.config import (
    STANDARD_SERVER_URLS,
    Config,
    LLMConfig,
    load_config,
    to_toml,
)


def _write(tmp_path: Path, content: str) -> Path:
    p = tmp_path / "config.toml"
    p.write_text(content, encoding="utf-8")
    return p


# ------------------------------------------------------------------------ effective_server_url ---
def test_effective_server_url_falls_back_to_standard_when_server_url_unset() -> None:
    cfg = LLMConfig()
    assert cfg.effective_server_url("ollama") == STANDARD_SERVER_URLS["ollama"]
    assert cfg.effective_server_url("openai") == STANDARD_SERVER_URLS["openai"]


def test_effective_server_url_has_no_standard_for_llamacpp() -> None:
    assert LLMConfig().effective_server_url("llamacpp") == ""


def test_effective_server_url_uses_server_url_verbatim_for_either_backend_when_set() -> None:
    cfg = LLMConfig(server_url="http://h:9000")
    assert cfg.effective_server_url("ollama") == "http://h:9000"
    assert cfg.effective_server_url("openai") == "http://h:9000"


def test_effective_server_url_strips_a_trailing_slash() -> None:
    cfg = LLMConfig(server_url="http://h:9000/")
    assert cfg.effective_server_url("openai") == "http://h:9000"


# ------------------------------------------------------------------------------ property shim ---
def test_base_url_and_ollama_url_getters_reflect_server_url() -> None:
    cfg = LLMConfig()
    assert cfg.base_url == STANDARD_SERVER_URLS["openai"]
    assert cfg.ollama_url == STANDARD_SERVER_URLS["ollama"]
    cfg.server_url = "http://h:1234"
    assert cfg.base_url == "http://h:1234"
    assert cfg.ollama_url == "http://h:1234"


def test_base_url_setter_writes_server_url() -> None:
    cfg = LLMConfig()
    cfg.base_url = "http://h:9"
    assert cfg.server_url == "http://h:9"
    assert cfg.ollama_url == "http://h:9"          # same underlying field, read back either way


def test_ollama_url_setter_writes_server_url() -> None:
    cfg = LLMConfig()
    cfg.ollama_url = "http://h:99"
    assert cfg.server_url == "http://h:99"
    assert cfg.base_url == "http://h:99"


def test_base_url_and_ollama_url_are_not_dataclass_fields() -> None:
    names = {f.name for f in dataclasses.fields(LLMConfig)}
    assert "base_url" not in names
    assert "ollama_url" not in names
    assert "server_url" in names
    assert "library_dir" in names
    assert "runtime_dir" in names


def test_constructing_with_base_url_or_ollama_url_kwargs_no_longer_works() -> None:
    """They are properties, not fields -- the dataclass-generated __init__ only accepts real
    fields. Call sites that used to construct with these kwargs must switch to server_url."""
    with pytest.raises(TypeError):
        LLMConfig(base_url="http://h:1")  # type: ignore[call-arg]
    with pytest.raises(TypeError):
        LLMConfig(ollama_url="http://h:1")  # type: ignore[call-arg]


# ------------------------------------------------------------------------------------ to_toml ---
def test_to_toml_omits_legacy_keys_and_writes_the_new_fields() -> None:
    text = to_toml(Config())
    assert "server_url" in text
    assert "base_url" not in text
    assert "ollama_url" not in text
    assert "library_dir" in text
    assert "runtime_dir" in text


# ------------------------------------------------------------------------- load_config: no-op ---
def test_migration_noop_when_neither_legacy_key_is_present(tmp_path: Path) -> None:
    p = _write(tmp_path, '[llm]\nbackend = "auto"\n')
    cfg = load_config(p)
    assert cfg.llm.server_url == ""
    assert cfg.llm.backend == "auto"


# --------------------------------------------------------------------- load_config: migration ---
def test_migration_ollama_backend_uses_ollama_url(tmp_path: Path) -> None:
    p = _write(
        tmp_path,
        '[llm]\nbackend = "ollama"\n'
        'base_url = "http://127.0.0.1:8081"\n'
        'ollama_url = "http://h:11434"\n',
    )
    cfg = load_config(p)
    assert cfg.llm.server_url == "http://h:11434"
    assert cfg.llm.ollama_url == "http://h:11434"


def test_migration_openai_backend_uses_base_url(tmp_path: Path) -> None:
    p = _write(
        tmp_path,
        '[llm]\nbackend = "openai"\n'
        'base_url = "http://h:8081"\n'
        'ollama_url = "http://127.0.0.1:11434"\n',
    )
    cfg = load_config(p)
    assert cfg.llm.server_url == "http://h:8081"
    assert cfg.llm.base_url == "http://h:8081"


def test_migration_auto_backend_prefers_ollama_url_when_it_is_the_customized_one(tmp_path: Path) -> None:
    p = _write(
        tmp_path,
        '[llm]\nbackend = "auto"\n'
        'base_url = "http://127.0.0.1:8081"\n'   # left at its default
        'ollama_url = "http://h:11434"\n',        # customized
    )
    cfg = load_config(p)
    assert cfg.llm.server_url == "http://h:11434"


def test_migration_auto_backend_prefers_base_url_when_it_is_the_customized_one(tmp_path: Path) -> None:
    p = _write(
        tmp_path,
        '[llm]\nbackend = "auto"\n'
        'base_url = "http://h:8081"\n'            # customized
        'ollama_url = "http://127.0.0.1:11434"\n',  # left at its default
    )
    cfg = load_config(p)
    assert cfg.llm.server_url == "http://h:8081"


def test_migration_both_still_at_default_yields_empty_server_url(tmp_path: Path) -> None:
    p = _write(
        tmp_path,
        '[llm]\nbackend = "auto"\n'
        'base_url = "http://127.0.0.1:8081"\n'
        'ollama_url = "http://127.0.0.1:11434"\n',
    )
    cfg = load_config(p)
    assert cfg.llm.server_url == ""


def test_migration_explicit_value_identical_to_standard_migrates_to_empty_for_a_pinned_backend(
    tmp_path: Path,
) -> None:
    """An explicit ollama_url identical to the standard address must not get hard-pinned into
    server_url -- it migrates to "" (still the standard address, via effective_server_url)."""
    p = _write(tmp_path, '[llm]\nbackend = "ollama"\nollama_url = "http://127.0.0.1:11434"\n')
    cfg = load_config(p)
    assert cfg.llm.server_url == ""
    assert cfg.llm.ollama_url == STANDARD_SERVER_URLS["ollama"]


def test_migration_only_base_url_present_treats_the_missing_ollama_url_as_its_default(
    tmp_path: Path,
) -> None:
    p = _write(tmp_path, '[llm]\nbackend = "auto"\nbase_url = "http://h:8081"\n')
    cfg = load_config(p)
    assert cfg.llm.server_url == "http://h:8081"


def test_migration_only_ollama_url_present_treats_the_missing_base_url_as_its_default(
    tmp_path: Path,
) -> None:
    p = _write(tmp_path, '[llm]\nbackend = "auto"\nollama_url = "http://h:11434"\n')
    cfg = load_config(p)
    assert cfg.llm.server_url == "http://h:11434"


def test_migration_server_url_present_wins_and_legacy_keys_are_dropped_silently(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    p = _write(
        tmp_path,
        '[llm]\nbackend = "auto"\n'
        'server_url = "http://real:9"\n'
        'base_url = "http://h:8081"\n'
        'ollama_url = "http://h:11434"\n',
    )
    with caplog.at_level(logging.INFO, logger="exocortex.config"):
        cfg = load_config(p)
    assert cfg.llm.server_url == "http://real:9"
    assert not any("migrated" in r.getMessage() for r in caplog.records)


def test_migration_logs_exactly_one_info_line(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    p = _write(tmp_path, '[llm]\nbackend = "ollama"\nollama_url = "http://h:11434"\n')
    with caplog.at_level(logging.INFO, logger="exocortex.config"):
        load_config(p)
    migrated = [r for r in caplog.records if "migrated" in r.getMessage()]
    assert len(migrated) == 1
    assert "server_url" in migrated[0].getMessage()
    assert migrated[0].levelno == logging.INFO


def test_migration_never_warns_about_the_legacy_keys_as_unknown(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    p = _write(
        tmp_path,
        '[llm]\nbackend = "ollama"\n'
        'base_url = "http://127.0.0.1:8081"\n'
        'ollama_url = "http://h:11434"\n',
    )
    with caplog.at_level(logging.WARNING, logger="exocortex.config"):
        load_config(p)
    assert not any("unknown key" in r.getMessage() for r in caplog.records)


def test_migrated_config_round_trips_through_to_toml_without_legacy_keys(tmp_path: Path) -> None:
    p = _write(
        tmp_path,
        '[llm]\nbackend = "ollama"\n'
        'base_url = "http://127.0.0.1:8081"\n'
        'ollama_url = "http://h:11434"\n',
    )
    cfg = load_config(p)
    text = to_toml(cfg)
    assert "server_url" in text
    assert "base_url" not in text
    assert "ollama_url" not in text


# ---------------------------------------------------------- C1: loud, recoverable dropped URL ---
# review round two, 2026-09-27 -- second review round finding C1: with backend "auto" and BOTH
# legacy keys customised to DIFFERENT non-standard addresses, only one survives into server_url by
# design (there is one field on purpose); the fix is making the loss loud (a WARNING naming both
# values, which was kept, and the exact two config.toml lines to recover the other) and recoverable
# (the dropped value recorded on the non-persisted LLMConfig.dropped_legacy_url).
def test_migration_auto_backend_both_customised_different_warns_and_records_dropped_value(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    p = _write(
        tmp_path,
        '[llm]\nbackend = "auto"\n'
        'base_url = "http://a:8081"\n'      # customised
        'ollama_url = "http://b:11434"\n',  # customised, different address
    )
    with caplog.at_level(logging.WARNING, logger="exocortex.config"):
        cfg = load_config(p)
    # auto prefers the first customised of (base_url, ollama_url) -- base_url here.
    assert cfg.llm.server_url == "http://a:8081"
    assert cfg.llm.dropped_legacy_url == "http://b:11434"
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1
    msg = warnings[0].getMessage()
    assert "http://a:8081" in msg          # names both old values
    assert "http://b:11434" in msg
    assert 'backend = "ollama"' in msg     # exact two lines to recover the dropped value
    assert 'server_url = "http://b:11434"' in msg


def test_migration_explicit_backend_both_customised_different_warns_about_the_other(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    p = _write(
        tmp_path,
        '[llm]\nbackend = "ollama"\n'
        'base_url = "http://a:8081"\n'      # customised -- not the matching backend, dropped
        'ollama_url = "http://b:11434"\n',  # customised, matches backend -- kept
    )
    with caplog.at_level(logging.WARNING, logger="exocortex.config"):
        cfg = load_config(p)
    assert cfg.llm.server_url == "http://b:11434"      # the matching one is kept
    assert cfg.llm.dropped_legacy_url == "http://a:8081"
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1
    msg = warnings[0].getMessage()
    assert "http://a:8081" in msg
    assert "http://b:11434" in msg
    assert 'backend = "openai"' in msg     # recovers the dropped (openai/base_url) address
    assert 'server_url = "http://a:8081"' in msg


def test_migration_only_one_customised_never_warns(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    p = _write(
        tmp_path,
        '[llm]\nbackend = "auto"\n'
        'base_url = "http://a:8081"\n'                   # customised
        'ollama_url = "http://127.0.0.1:11434"\n',        # left at its default -- nothing lost
    )
    with caplog.at_level(logging.WARNING, logger="exocortex.config"):
        cfg = load_config(p)
    assert cfg.llm.server_url == "http://a:8081"
    assert cfg.llm.dropped_legacy_url == ""
    assert not any(r.levelno == logging.WARNING for r in caplog.records)


def test_to_toml_never_writes_the_dropped_legacy_url_attribute() -> None:
    cfg = Config()
    cfg.llm.dropped_legacy_url = "http://dropped:1234"
    text = to_toml(cfg)
    assert "dropped_legacy_url" not in text
    assert "http://dropped:1234" not in text
