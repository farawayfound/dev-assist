"""The ``[capabilities]`` config section and the config hardening that came with it (2026-10-07):
defaults, the TOML round trip of the MCP server entries (an array of inline tables with nested
lists / tables), quoted keys and escaped control characters in the writer, fail-closed type
coercion in ``_apply`` and the secret redaction of ``config_as_dict``. Pure config tests."""
from __future__ import annotations

import tomllib
from pathlib import Path

import pytest

from exocortex import config as config_mod
from exocortex.config import (
    DEFAULT_EDIT_EXCLUDE,
    DEFAULT_READ_EXCLUDE,
    SAFE_COMMANDS,
    CapabilitiesConfig,
    Config,
    config_as_dict,
    load_config,
    save_config,
    to_toml,
)


# ---------------------------------------------------------------------------------------------
# defaults
# ---------------------------------------------------------------------------------------------
def test_defaults_match_the_owner_request() -> None:
    cc = Config().capabilities
    assert (cc.read_files, cc.edit_files, cc.execute_commands, cc.fetch_web, cc.mcp_servers) == (
        True, False, True, False, False)
    assert cc.apply_to == "questions"
    assert cc.read_roots == [] and cc.edit_roots == []
    assert cc.allowed_commands == list(SAFE_COMMANDS)
    assert cc.allow_all_commands is False
    assert cc.web_allow_local is False
    assert cc.mcp == []
    assert cc.active_map == ""


def test_list_defaults_are_independent_per_config() -> None:
    a, b = Config(), Config()
    a.capabilities.read_roots.append("D:/x")
    a.capabilities.allowed_commands.append("foo")
    assert b.capabilities.read_roots == []
    assert "foo" not in b.capabilities.allowed_commands


def test_safe_commands_hold_no_interpreter_and_no_test_runner() -> None:
    joined = " | ".join(SAFE_COMMANDS)
    for forbidden in ("python -c", "python -m", "pytest", "npm run", "npm test", "cargo build",
                      "powershell", "cmd", "node -e"):
        assert forbidden not in joined, forbidden
    assert "git status" in SAFE_COMMANDS and "dir" in SAFE_COMMANDS and "python --version" in SAFE_COMMANDS
    # every entry is a plain command prefix: no chaining / redirection characters
    for entry in SAFE_COMMANDS:
        assert not any(ch in entry for ch in "&|;<>^`$%\n"), entry


def test_safe_commands_hold_no_content_reader() -> None:
    """type / cat / head / tail / more / findstr / grep / rg / wc leaked around the read scope (wildcards,
    switches, recursion); read_file / search_text cover what they did, and a user can add one by hand."""
    first_words = {e.split()[0] for e in SAFE_COMMANDS}
    for reader in ("type", "cat", "head", "tail", "more", "findstr", "grep", "rg", "wc"):
        assert reader not in first_words, reader
    for kept in ("dir", "ls", "tree", "where", "which", "pwd", "hostname", "whoami", "ver", "date", "tasklist", "systeminfo"):
        assert kept in SAFE_COMMANDS, kept
    for kept in ("git status", "git log", "git diff", "git show", "git blame", "git remote -v", "pip list", "pip show",
                 "npm ls", "python --version", "node --version"):
        assert kept in SAFE_COMMANDS, kept


def test_default_excludes_cover_the_usual_credential_stores() -> None:
    for name in (".git", ".env", "*.pem", "id_rsa*", ".ssh", ".aws", ".npmrc", ".netrc", ".git-credentials"):
        assert name in DEFAULT_READ_EXCLUDE
    for name in (".git", "*.bat", "*.cmd", "*.ps1", "*.exe"):
        assert name in DEFAULT_EDIT_EXCLUDE


def test_maps_dir_follows_models_dir_unless_set(tmp_path: Path) -> None:
    cfg = Config()
    cfg.paths.models_dir = str(tmp_path / "models")
    assert cfg.maps_dir() == tmp_path / "models" / "maps"
    cfg.capabilities.maps_dir = str(tmp_path / "elsewhere")
    assert cfg.maps_dir() == tmp_path / "elsewhere"


# ---------------------------------------------------------------------------------------------
# the TOML round trip
# ---------------------------------------------------------------------------------------------
def test_section_round_trips_through_the_file(tmp_path: Path) -> None:
    cfg = Config()
    cc = cfg.capabilities
    cc.edit_files = True
    cc.read_roots = ["D:/Repos/one", "C:/Users/me/notes"]
    cc.allowed_commands = ["git status", "dir"]
    cc.web_allowed_domains = ["docs.python.org"]
    cc.mcp = [{"name": "fs", "transport": "stdio", "command": "npx",
               "args": ["-y", "@modelcontextprotocol/server-filesystem", "D:/x"],
               "env": {"A": "1"}, "enabled": True},
              {"name": "remote", "transport": "http", "url": "http://127.0.0.1:9000/mcp", "enabled": False}]
    cc.active_map = "My App"
    path = save_config(cfg, tmp_path / "config.toml")
    back = load_config(path).capabilities
    assert back.edit_files is True
    assert back.read_roots == cc.read_roots
    assert back.allowed_commands == ["git status", "dir"]
    assert back.web_allowed_domains == ["docs.python.org"]
    assert back.mcp == cc.mcp
    assert back.active_map == "My App"


def test_writer_quotes_keys_that_are_not_bare_and_escapes_control_characters(tmp_path: Path) -> None:
    cfg = Config()
    cfg.capabilities.mcp = [{"name": "x", "transport": "stdio", "command": "srv",
                             "env": {"ProgramFiles(x86)": "a\nb", "A B": "tab\there", "a.b": "dot", "DEL": "\x7f"}}]
    cfg.capabilities.web_search_url = "https://s/?q={query}\r\n"
    text = to_toml(cfg)
    data = tomllib.loads(text)     # must parse at all: a raw newline used to brick the next launch
    env = data["capabilities"]["mcp"][0]["env"]
    assert env == {"ProgramFiles(x86)": "a\nb", "A B": "tab\there", "a.b": "dot", "DEL": "\x7f"}
    assert data["capabilities"]["web_search_url"] == "https://s/?q={query}\r\n"


def test_writer_renders_none_as_an_empty_string() -> None:
    cfg = Config()
    cfg.capabilities.mcp = [{"name": "x", "url": None}]
    data = tomllib.loads(to_toml(cfg))
    assert data["capabilities"]["mcp"][0]["url"] == ""


# ---------------------------------------------------------------------------------------------
# fail-closed coercion
# ---------------------------------------------------------------------------------------------
def _load(tmp_path: Path, body: str) -> Config:
    p = tmp_path / "config.toml"
    p.write_text(body, encoding="utf-8")
    return load_config(p)


def test_a_string_for_a_bool_keeps_the_default_instead_of_becoming_true(tmp_path: Path, caplog) -> None:
    cfg = _load(tmp_path, '[capabilities]\nallow_all_commands = "false"\nedit_files = "yes"\n')
    assert cfg.capabilities.allow_all_commands is False
    assert cfg.capabilities.edit_files is False
    assert "expects true/false" in caplog.text


def test_zero_and_one_still_count_as_bools(tmp_path: Path) -> None:
    cfg = _load(tmp_path, "[capabilities]\nedit_files = 1\nread_files = 0\n")
    assert cfg.capabilities.edit_files is True
    assert cfg.capabilities.read_files is False


def test_a_string_for_a_list_keeps_the_default(tmp_path: Path, caplog) -> None:
    cfg = _load(tmp_path, '[capabilities]\nread_roots = "D:/Repos"\n')
    assert cfg.capabilities.read_roots == []
    assert "expects a list" in caplog.text


def test_a_list_for_a_table_keeps_the_default(tmp_path: Path, caplog) -> None:
    cfg = _load(tmp_path, "[llm]\nllama_env = [1, 2]\n")
    assert cfg.llm.llama_env == {}
    assert "expects a table" in caplog.text


def test_numbers_still_coerce_as_before(tmp_path: Path) -> None:
    cfg = _load(tmp_path, "[capabilities]\ntool_time_s = 12\nmax_tool_calls = 3.0\n")
    assert cfg.capabilities.tool_time_s == 12.0 and isinstance(cfg.capabilities.tool_time_s, float)
    assert cfg.capabilities.max_tool_calls == 3 and isinstance(cfg.capabilities.max_tool_calls, int)


# ---------------------------------------------------------------------------------------------
# redaction
# ---------------------------------------------------------------------------------------------
def test_config_as_dict_redacts_mcp_env_values_and_url_tokens() -> None:
    cfg = Config()
    cfg.capabilities.mcp = [{"name": "x", "transport": "http",
                             "url": "https://h/mcp?api_key=SECRET&x=1&token=T2#frag",
                             "env": {"HF_TOKEN": "hf_abc", "PATH": "C:/x"}, "enabled": True}]
    d = config_as_dict(cfg)
    spec = d["capabilities"]["mcp"][0]
    assert spec["env"] == {"HF_TOKEN": "***", "PATH": "***"}
    assert spec["url"] == "https://h/mcp?api_key=***&x=1&token=***#frag"
    assert spec["name"] == "x" and spec["enabled"] is True
    # the config object itself is untouched (to_toml must still write the real values)
    assert cfg.capabilities.mcp[0]["env"]["HF_TOKEN"] == "hf_abc"
    assert "hf_abc" in to_toml(cfg)


def test_config_as_dict_redacts_secrets_in_args_urls_and_the_search_url() -> None:
    cfg = Config()
    cfg.capabilities.web_search_url = "https://searx.example/search?q={query}&key=SEARXKEY&lang=en"
    cfg.capabilities.mcp = [
        {"name": "a", "transport": "stdio", "command": "npx",
         "args": ["-y", "srv", "--api-key", "sk-SECRET1", "--token=tok-SECRET2", "--no-cache", "postgresql://u:PGPASS@h/db",
                  "--password", "pw", "--db=mine"], "enabled": True},
        {"name": "b", "transport": "http", "url": "https://user:PWSECRET@h.example/mcp?token=qs&page=2",
         "headers": {"Authorization": "Bearer abc"}, "enabled": True},
    ]
    d = config_as_dict(cfg)
    a, b = d["capabilities"]["mcp"]
    assert a["args"] == ["-y", "srv", "--api-key", "***", "--token=***", "--no-cache", "postgresql://u:***@h/db",
                         "--password", "***", "--db=mine"]                      # only the secret goes, the rest stays readable
    assert b["url"] == "https://user:***@h.example/mcp?token=***&page=2"
    assert b["headers"] == {"Authorization": "***"}
    assert d["capabilities"]["web_search_url"] == "https://searx.example/search?q={query}&key=***&lang=en"
    blob = repr(d)
    for secret in ("sk-SECRET1", "tok-SECRET2", "PGPASS", "PWSECRET", "SEARXKEY", "Bearer abc", "=qs"):
        assert secret not in blob, secret
    # the config itself keeps the real values (to_toml writes them)
    assert cfg.capabilities.mcp[0]["args"][3] == "sk-SECRET1" and "sk-SECRET1" in to_toml(cfg)


def test_the_redaction_helpers_are_shared_with_the_capabilities_tab() -> None:
    assert config_mod._redact_text("https://u:p@h/x?auth=1") == "https://u:***@h/x?auth=***"
    assert config_mod._redact_args(("--secret", "x", "-v")) == ["--secret", "***", "-v"]
    assert config_mod._redact_args([]) == []
    assert config_mod._redact_text("plain text, no url") == "plain text, no url"


def test_config_as_dict_tolerates_odd_mcp_entries() -> None:
    cfg = Config()
    cfg.capabilities.mcp = ["not a dict", {"name": "y"}]
    d = config_as_dict(cfg)
    assert d["capabilities"]["mcp"] == ["not a dict", {"name": "y"}]


def test_section_sits_between_monitor_and_ui_in_the_file() -> None:
    text = to_toml(Config())
    assert text.index("[monitor]") < text.index("[capabilities]") < text.index("[ui]")


def test_capabilities_config_is_a_plain_dataclass_with_every_field_persisted() -> None:
    names = {f.name for f in config_mod.fields(CapabilitiesConfig)}
    assert {"read_files", "edit_files", "execute_commands", "fetch_web", "mcp_servers", "apply_to",
            "read_roots", "edit_roots", "allowed_commands", "allow_all_commands", "web_allowed_domains",
            "mcp", "maps_dir", "active_map", "map_tokens", "live_max_tool_calls"} <= names
    data = tomllib.loads(to_toml(Config()))["capabilities"]
    assert set(data) == names
