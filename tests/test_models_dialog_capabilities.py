"""Offscreen tests for the Models dialog's "Capabilities" tab (2026-10-07, capabilities).

Same shape as ``tests/test_models_dialog_context.py``: a :class:`FakeActions` records every call as
``(name, args, kwargs)`` and returns canned data (``None`` by default, like an idle engine); the
folder picker, the MCP server prompt, the map-name prompt and the confirmation are monkeypatched on
the instance so nothing opens a native window. Nothing here touches the real ``config.toml``, the
model library or a network: the dialog only edits the in-memory ``Config`` it is handed.
"""
from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import contextlib
import dataclasses
import logging
import time
from typing import Any

import pytest
from PySide6.QtCore import QTimer
from PySide6.QtTest import QTest
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDialogButtonBox,
    QLineEdit,
    QMessageBox,
    QPlainTextEdit,
)

from exocortex.bus import CapabilityMapEvent, ComponentStatusEvent, McpToolsEvent, ModelLibraryEvent
from exocortex import capabilities as capabilities_mod
from exocortex import mcp as mcp_mod
from exocortex.config import SAFE_COMMANDS, Config
from exocortex.ui import models as models_mod
from exocortex.ui.models import ModelsDialog


class FakeActions:
    """Records every call as ``(name, args, kwargs)``; returns canned data the test set up. The
    capability calls named in ``raise_on`` record themselves and then raise."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple[Any, ...], dict[str, Any]]] = []
        self.order: list[str] = []          # "index:<name>" etc., for tests that check sequencing
        self.maps_result: list[dict[str, object]] | None = None
        self.status_result: dict[str, object] | None = None
        self.status_sequence: list[dict[str, object] | None] = []   # popped one per read; the last one sticks
        self.raise_on: set[str] = set()

    def _record(self, name: str, args: tuple[Any, ...] = (), kwargs: dict[str, Any] | None = None) -> None:
        self.calls.append((name, args, dict(kwargs or {})))
        if name in self.raise_on:
            raise RuntimeError(f"{name} blew up")

    def calls_named(self, name: str) -> list[tuple[tuple[Any, ...], dict[str, Any]]]:
        return [(a, kw) for (n, a, kw) in self.calls if n == name]

    # ---- the capability surface (exocortex.actions.EngineActions, 2026-10-07) ----
    def capability_maps(self) -> list[dict[str, object]] | None:
        self._record("capability_maps")
        return self.maps_result

    def capability_status(self) -> dict[str, object] | None:
        self._record("capability_status")
        if self.status_sequence:
            self.status_result = self.status_sequence.pop(0) if len(self.status_sequence) > 1 else self.status_sequence[0]
        return self.status_result

    def index_capability_map(self, name: str) -> None:
        self.order.append(f"index:{name}")
        self._record("index_capability_map", (name,))

    def load_capability_map(self, name: str) -> None:
        self._record("load_capability_map", (name,))

    def delete_capability_map(self, name: str) -> None:
        self._record("delete_capability_map", (name,))

    def mcp_server_tools(self, name: str, request_id: int) -> None:
        self._record("mcp_server_tools", (name, request_id))

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_"):
            raise AttributeError(name)

        def f(*args: Any, **kwargs: Any) -> None:
            self._record(name, args, kwargs)
            return None

        return f


@pytest.fixture(scope="session")
def qapp() -> QApplication:
    return QApplication.instance() or QApplication(["exocortex-tests"])


@pytest.fixture()
def cfg(tmp_path) -> Config:
    c = Config()
    c.paths.models_dir = str(tmp_path / "models")      # maps_dir() resolves under tmp_path
    c.paths.sessions_dir = str(tmp_path / "sessions")
    return c


@pytest.fixture()
def actions() -> FakeActions:
    return FakeActions()


@pytest.fixture()
def make_dialog(qapp: QApplication, actions: FakeActions):
    made: list[ModelsDialog] = []

    def make(config: Config) -> ModelsDialog:
        d = ModelsDialog(config, actions)
        d.show()
        actions.calls.clear()
        made.append(d)
        return d

    yield make
    for d in made:
        d.close()


@pytest.fixture()
def dialog(make_dialog, cfg: Config) -> ModelsDialog:
    return make_dialog(cfg)


def _map_row(name: str, **overrides: object) -> dict[str, object]:
    base: dict[str, object] = {
        "name": name,
        "path": f"C:/maps/{name}.json",
        "roots": ["C:/work/proj"],
        "files": 12,
        "dirs": 3,
        "indexed_at": "2026-10-07T10:00:00",
        "truncated": False,
        "active": False,
        "settings": {},
    }
    base.update(overrides)
    return base


def _server(name: str = "fs", **overrides: object) -> dict[str, Any]:
    base: dict[str, Any] = {"name": name, "transport": "stdio", "command": "npx",
                            "args": ["-y", "server-filesystem"], "env": {}, "enabled": True}
    base.update(overrides)
    return base


def _cells(table, row: int) -> list[str]:
    return [table.item(row, c).text() for c in range(table.columnCount())]


def _lines(edit: QPlainTextEdit) -> list[str]:
    return edit.toPlainText().splitlines()


# ============================================================================================ tab
def test_capabilities_tab_sits_between_context_and_hardware(dialog: ModelsDialog) -> None:
    assert ModelsDialog.TABS.index("Capabilities") == ModelsDialog.TABS.index("Context") + 1
    assert ModelsDialog.TABS.index("Hardware") == ModelsDialog.TABS.index("Capabilities") + 1
    dialog.show_tab("Capabilities")
    assert dialog.tabs.tabText(dialog.tabs.currentIndex()) == "Capabilities"
    assert dialog.tabs.widget(dialog.tabs.currentIndex()).findChild(type(dialog.cap_scroll)) is dialog.cap_scroll


def test_the_widgets_the_contract_names_all_exist(dialog: ModelsDialog) -> None:
    for name in ("cap_read_cb", "cap_edit_cb", "cap_exec_cb", "cap_web_cb", "cap_mcp_cb",
                 "cap_read_roots", "cap_read_add_btn", "cap_read_remove_btn", "cap_read_exclude",
                 "cap_edit_roots", "cap_edit_add_btn", "cap_edit_remove_btn", "cap_edit_exclude",
                 "cap_commands", "cap_commands_restore_btn", "cap_web_allowed", "cap_web_blocked",
                 "cap_mcp_table", "cap_mcp_add_btn", "cap_mcp_edit_btn", "cap_mcp_remove_btn",
                 "cap_mcp_test_btn", "cap_maps_table", "cap_index_btn", "cap_map_load_btn",
                 "cap_map_reindex_btn", "cap_map_delete_btn", "cap_map_open_btn", "cap_map_status",
                 "cap_status", "cap_note", "cap_apply_btn", "cap_restore_btn", "_capabilities_form"):
        assert getattr(dialog, name, None) is not None, name
    assert dialog.cap_apply_btn.text() == "Apply"
    assert dialog.cap_restore_btn.text() == "Restore defaults"
    assert dialog.cap_index_btn.text().startswith("Pre-index scoped directories")


def test_the_form_covers_the_scalar_settings_but_not_the_switches(dialog: ModelsDialog) -> None:
    paths = {f.path for f in ModelsDialog._CAPABILITY_FIELDS}
    for key in ("apply_to", "max_tool_calls", "tool_result_chars", "tool_time_s", "live_max_tool_calls",
                "live_tool_result_chars", "live_tool_time_s", "read_max_file_chars", "edit_backup",
                "allow_all_commands", "command_timeout_s", "command_output_chars", "command_cwd",
                "web_allow_local", "web_max_fetches", "web_page_chars", "web_timeout_s", "web_search_url",
                "web_search_results", "mcp_timeout_s", "maps_dir", "map_max_files", "map_tokens"):
        assert f"capabilities.{key}" in paths, key
    for key in ("read_files", "edit_files", "execute_commands", "fetch_web", "mcp_servers"):
        assert f"capabilities.{key}" not in paths, key
    assert [f.path for f in ModelsDialog._CAPABILITY_FIELDS if f.restart] == []   # the engine reads every one live
    all_commands = next(f for f in ModelsDialog._CAPABILITY_FIELDS if f.key == "allow_all_commands")
    assert "runs arbitrary code" in all_commands.label


def test_the_five_switches_stay_one_row_when_wide_and_wrap_when_narrow(dialog: ModelsDialog) -> None:
    row = dialog.cap_read_cb.parentWidget().layout().itemAt(0).layout()   # the group's first item
    assert isinstance(row, models_mod._FlowLayout) and row.count() == 5
    wide, narrow = row.heightForWidth(2000), row.heightForWidth(150)
    one_line = dialog.cap_read_cb.sizeHint().height()
    assert wide == one_line                       # one row with room to spare
    assert narrow > wide                          # a window narrower than the row wraps it
    widest = max(cb.minimumSizeHint().width() for cb in dialog._cap_switches.values())
    assert row.minimumSize().width() == widest    # ... and can shrink to the widest single switch


def test_every_form_default_is_inside_its_widget_range() -> None:
    """A value the config holds by default must survive the round trip through its spin box,
    or an untouched tab would report changes on Apply."""
    cc = Config().capabilities
    for f in ModelsDialog._CAPABILITY_FIELDS:
        if f.kind in ("int", "float"):
            value = getattr(cc, f.key)
            assert f.lo <= value <= f.hi, f.path


# ============================================================================================ load
def test_defaults_are_shown_for_a_fresh_config(dialog: ModelsDialog) -> None:
    assert dialog.cap_read_cb.isChecked() is True
    assert dialog.cap_edit_cb.isChecked() is False
    assert dialog.cap_exec_cb.isChecked() is True
    assert dialog.cap_web_cb.isChecked() is False
    assert dialog.cap_mcp_cb.isChecked() is False
    assert dialog.cap_read_roots.count() == 0
    assert _lines(dialog.cap_commands) == list(SAFE_COMMANDS)
    assert ".git" in _lines(dialog.cap_read_exclude)
    assert dialog.cap_mcp_table.rowCount() == 0
    assert dialog._capabilities_changes() == {}      # an untouched tab reports nothing to apply


def test_load_fills_switches_lists_commands_and_the_form_from_a_custom_config(
    make_dialog, cfg: Config, tmp_path
) -> None:
    cc = cfg.capabilities
    cc.read_files, cc.edit_files, cc.execute_commands = False, True, False
    cc.fetch_web, cc.mcp_servers = True, True
    cc.read_roots = [str(tmp_path / "a"), str(tmp_path / "b")]
    cc.read_exclude = ["*.log", "private"]
    cc.edit_roots = [str(tmp_path / "out")]
    cc.edit_exclude = ["*.exe"]
    cc.allowed_commands = ["git status", "dir"]
    cc.web_allowed_domains = ["example.com"]
    cc.web_blocked_domains = ["bad.example.com", "tracker.net"]
    cc.mcp = [_server("fs"), _server("api", transport="http", url="https://h/mcp", enabled=False)]
    cc.apply_to, cc.tool_time_s, cc.max_tool_calls = "ask", 12.5, 3
    cc.allow_all_commands, cc.command_cwd = True, str(tmp_path / "a")
    cc.maps_dir = str(tmp_path / "maps")
    d = make_dialog(cfg)

    assert (d.cap_read_cb.isChecked(), d.cap_edit_cb.isChecked(), d.cap_exec_cb.isChecked(),
            d.cap_web_cb.isChecked(), d.cap_mcp_cb.isChecked()) == (False, True, False, True, True)
    assert [d.cap_read_roots.item(i).text() for i in range(d.cap_read_roots.count())] == cc.read_roots
    assert _lines(d.cap_read_exclude) == ["*.log", "private"]
    assert [d.cap_edit_roots.item(i).text() for i in range(d.cap_edit_roots.count())] == cc.edit_roots
    assert _lines(d.cap_edit_exclude) == ["*.exe"]
    assert _lines(d.cap_commands) == ["git status", "dir"]
    assert _lines(d.cap_web_allowed) == ["example.com"]
    assert _lines(d.cap_web_blocked) == ["bad.example.com", "tracker.net"]
    w = d._capabilities_form.widgets
    assert w["capabilities.apply_to"].currentText() == "ask"
    assert w["capabilities.tool_time_s"].value() == pytest.approx(12.5)
    assert w["capabilities.max_tool_calls"].value() == 3
    assert w["capabilities.allow_all_commands"].isChecked() is True
    assert w["capabilities.command_cwd"].text() == str(tmp_path / "a")
    assert w["capabilities.maps_dir"].text() == str(tmp_path / "maps")
    # the MCP table
    assert d.cap_mcp_table.rowCount() == 2
    assert [d.cap_mcp_table.horizontalHeaderItem(c).text() for c in range(5)] == [
        "Name", "Transport", "Command or URL", "Enabled", "Tools"]
    assert _cells(d.cap_mcp_table, 0) == ["fs", "stdio", "npx -y server-filesystem", "yes", ""]
    assert _cells(d.cap_mcp_table, 1) == ["api", "http", "https://h/mcp", "no", ""]
    # nothing was changed by loading
    assert d._capabilities_changes() == {}


def test_load_reads_the_maps_and_the_status_from_actions(dialog: ModelsDialog, actions: FakeActions) -> None:
    dialog.load()
    assert len(actions.calls_named("capability_maps")) == 1
    assert len(actions.calls_named("capability_status")) == 1


def test_loading_again_drops_unapplied_edits(dialog: ModelsDialog, cfg: Config) -> None:
    dialog.cap_edit_cb.setChecked(True)
    dialog.cap_commands.setPlainText("dir")
    assert dialog._capabilities_changes()
    dialog.load()
    assert dialog.cap_edit_cb.isChecked() is False
    assert _lines(dialog.cap_commands) == list(SAFE_COMMANDS)
    assert dialog._capabilities_changes() == {}


# ===================================================================================== folders
def test_add_folder_goes_through_the_picker_and_skips_duplicates(
    dialog: ModelsDialog, tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    a, b = str(tmp_path / "a"), str(tmp_path / "b")
    picks = iter([a, b, a, ""])
    monkeypatch.setattr(dialog, "_pick_directory", lambda: next(picks))
    for _ in range(4):
        dialog.cap_read_add_btn.click()
    assert [dialog.cap_read_roots.item(i).text() for i in range(dialog.cap_read_roots.count())] == [a, b]
    assert dialog.cap_edit_roots.count() == 0                # the other list is its own


def test_edit_scope_has_its_own_list_and_buttons(
    dialog: ModelsDialog, tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    out = str(tmp_path / "out")
    monkeypatch.setattr(dialog, "_pick_directory", lambda: out)
    dialog.cap_edit_add_btn.click()
    assert dialog.cap_edit_roots.count() == 1 and dialog.cap_read_roots.count() == 0
    dialog.cap_edit_roots.setCurrentRow(0)
    assert dialog.cap_edit_remove_btn.isEnabled()
    dialog.cap_edit_remove_btn.click()
    assert dialog.cap_edit_roots.count() == 0
    assert dialog.cap_edit_remove_btn.isEnabled() is False


def test_remove_drops_the_selected_folders_only(
    dialog: ModelsDialog, tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    roots = [str(tmp_path / n) for n in ("a", "b", "c")]
    dialog.cap_read_roots.addItems(roots)
    dialog.cap_read_roots.setCurrentRow(1)
    dialog.cap_read_remove_btn.click()
    assert [dialog.cap_read_roots.item(i).text() for i in range(dialog.cap_read_roots.count())] == [roots[0], roots[2]]


def test_a_cancelled_or_failing_picker_adds_nothing(
    dialog: ModelsDialog, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(dialog, "_pick_directory", lambda: "")
    dialog.cap_read_add_btn.click()
    assert dialog.cap_read_roots.count() == 0

    def boom() -> str:
        raise RuntimeError("picker blew up")

    monkeypatch.setattr(dialog, "_pick_directory", boom)
    dialog.cap_read_add_btn.click()
    assert dialog.cap_read_roots.count() == 0


def test_pick_directory_returns_a_string(dialog: ModelsDialog, monkeypatch: pytest.MonkeyPatch) -> None:
    from PySide6.QtWidgets import QFileDialog

    monkeypatch.setattr(QFileDialog, "getExistingDirectory", staticmethod(lambda *a, **kw: "C:/work/x"))
    assert dialog._pick_directory() == "C:/work/x"
    monkeypatch.setattr(QFileDialog, "getExistingDirectory", staticmethod(lambda *a, **kw: ""))
    assert dialog._pick_directory() == ""


# ====================================================================================== commands
def test_restore_safe_list_refills_the_commands_box_without_applying(
    dialog: ModelsDialog, cfg: Config
) -> None:
    emitted: list[dict] = []
    dialog.applied.connect(emitted.append)
    dialog.cap_commands.setPlainText("dir\nmystery-tool")
    dialog.cap_commands_restore_btn.click()
    assert _lines(dialog.cap_commands) == list(SAFE_COMMANDS)
    assert emitted == []
    assert cfg.capabilities.allowed_commands == list(SAFE_COMMANDS)


# ======================================================================================== apply
def test_apply_with_nothing_changed_emits_nothing(dialog: ModelsDialog) -> None:
    emitted: list[dict] = []
    dialog.applied.connect(emitted.append)
    dialog.cap_apply_btn.click()
    assert emitted == []
    assert dialog.cap_note.text() == ""


def test_apply_emits_only_the_changed_keys_and_writes_the_lists_back(
    dialog: ModelsDialog, cfg: Config, tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    emitted: list[dict] = []
    dialog.applied.connect(emitted.append)

    root = str(tmp_path / "proj")
    monkeypatch.setattr(dialog, "_pick_directory", lambda: root)
    dialog.cap_read_add_btn.click()                                    # a root added
    lines = _lines(dialog.cap_commands)
    lines[lines.index("git log")] = "git log -5"
    dialog.cap_commands.setPlainText("\n".join(lines))                 # a command line edited
    spec = _server("fs", env={"ROOT": root})
    monkeypatch.setattr(dialog, "_prompt_mcp_server", lambda existing: dict(spec))
    dialog.cap_mcp_add_btn.click()                                     # an MCP server added
    dialog.cap_edit_cb.setChecked(True)                                # a switch
    dialog._capabilities_form.widgets["capabilities.max_tool_calls"].setValue(6)   # a form scalar
    dialog.cap_web_blocked.setPlainText("  tracker.net  \n\n")        # blank lines dropped, whitespace stripped

    dialog.cap_apply_btn.click()

    assert len(emitted) == 1
    assert set(emitted[0]) == {
        "capabilities.read_roots", "capabilities.allowed_commands", "capabilities.mcp",
        "capabilities.edit_files", "capabilities.max_tool_calls", "capabilities.web_blocked_domains",
    }
    cc = cfg.capabilities
    assert cc.read_roots == [root]
    assert "git log -5" in cc.allowed_commands and "git log" not in cc.allowed_commands
    assert len(cc.allowed_commands) == len(SAFE_COMMANDS)
    assert cc.mcp == [spec]
    assert cc.edit_files is True and cc.max_tool_calls == 6
    assert cc.web_blocked_domains == ["tracker.net"]
    assert cc.read_exclude == list(Config().capabilities.read_exclude)    # untouched lists stay put
    assert emitted[0]["capabilities.read_roots"] == [root]
    assert dialog.cap_note.text() == "Applied."
    # a second Apply has nothing left to say
    dialog.cap_apply_btn.click()
    assert len(emitted) == 1 and dialog.cap_note.text() == ""


def test_apply_copies_the_server_specs_so_the_table_and_the_config_never_share_them(
    dialog: ModelsDialog, cfg: Config, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(dialog, "_prompt_mcp_server", lambda existing: _server("fs"))
    dialog.cap_mcp_add_btn.click()
    dialog.cap_apply_btn.click()
    dialog._mcp_specs[0]["enabled"] = False          # an unapplied edit of the working copy
    assert cfg.capabilities.mcp[0]["enabled"] is True
    assert "capabilities.mcp" in dialog._capabilities_changes()


def test_a_list_compares_as_a_list_whitespace_and_blank_lines_aside(dialog: ModelsDialog) -> None:
    noisy = "\n\n".join(f"  {c}  " for c in SAFE_COMMANDS) + "\n\n"
    dialog.cap_commands.setPlainText(noisy)
    assert dialog._capabilities_changes() == {}                          # same list: nothing to apply
    swapped = list(SAFE_COMMANDS)
    swapped[0], swapped[1] = swapped[1], swapped[0]
    dialog.cap_commands.setPlainText("\n".join(swapped))
    assert set(dialog._capabilities_changes()) == {"capabilities.allowed_commands"}   # order matters


def test_the_switches_are_applied_like_form_fields(dialog: ModelsDialog, cfg: Config) -> None:
    emitted: list[dict] = []
    dialog.applied.connect(emitted.append)
    dialog.cap_read_cb.setChecked(False)
    dialog.cap_exec_cb.setChecked(False)
    dialog.cap_web_cb.setChecked(True)
    dialog.cap_mcp_cb.setChecked(True)
    dialog.cap_apply_btn.click()
    assert emitted == [{"capabilities.read_files": False, "capabilities.execute_commands": False,
                        "capabilities.fetch_web": True, "capabilities.mcp_servers": True}]
    cc = cfg.capabilities
    assert (cc.read_files, cc.execute_commands, cc.fetch_web, cc.mcp_servers) == (False, False, True, True)


def test_the_maps_folder_applies_at_once_and_the_table_re_reads_the_new_folder(
    dialog: ModelsDialog, cfg: Config, actions: FakeActions, tmp_path
) -> None:
    emitted: list[dict] = []
    dialog.applied.connect(emitted.append)
    actions.maps_result = [_map_row("old-folder-map")]
    dialog._refresh_maps()
    assert dialog.cap_maps_table.rowCount() == 1
    actions.calls.clear()
    # what the engine does on the (direct) applied signal: it switches folders, so the old map is gone
    dialog.applied.connect(lambda changed: setattr(actions, "maps_result", []))

    dialog._capabilities_form.widgets["capabilities.maps_dir"].setText(str(tmp_path / "elsewhere"))
    dialog.cap_apply_btn.click()

    assert emitted == [{"capabilities.maps_dir": str(tmp_path / "elsewhere")}]
    assert dialog.cap_note.text() == "Applied."                    # no "takes effect at the next launch"
    assert cfg.capabilities.maps_dir == str(tmp_path / "elsewhere")
    assert len(actions.calls_named("capability_maps")) == 1        # re-read AFTER the engine switched
    assert dialog.cap_maps_table.rowCount() == 0

    dialog._capabilities_form.widgets["capabilities.web_page_chars"].setValue(5000)
    actions.calls.clear()
    dialog.cap_apply_btn.click()
    assert dialog.cap_note.text() == "Applied."
    assert actions.calls_named("capability_maps") == []            # only a maps_dir change re-reads the list


def test_a_field_marked_restart_still_gets_the_next_launch_note(
    dialog: ModelsDialog, monkeypatch: pytest.MonkeyPatch
) -> None:
    fields = tuple(dataclasses.replace(f, restart=True) if f.key == "web_page_chars" else f
                   for f in ModelsDialog._CAPABILITY_FIELDS)
    monkeypatch.setattr(ModelsDialog, "_CAPABILITY_FIELDS", fields)
    dialog._capabilities_form.widgets["capabilities.web_page_chars"].setValue(5000)
    dialog.cap_apply_btn.click()
    assert dialog.cap_note.text().startswith("Takes effect at the next launch")
    assert "Web page text kept" in dialog.cap_note.text()


def test_the_maps_folder_help_says_old_maps_are_not_moved(dialog: ModelsDialog) -> None:
    field = next(f for f in ModelsDialog._CAPABILITY_FIELDS if f.key == "maps_dir")
    assert "not moved" in field.help and "restart" not in field.help.lower()


def test_apply_refreshes_the_status_line(dialog: ModelsDialog, actions: FakeActions) -> None:
    actions.status_result = {"tools": ["list_dir", "read_file"], "read_roots": ["C:/w"], "mcp": {},
                             "active_map": "", "notes": []}
    dialog.cap_edit_cb.setChecked(True)
    dialog.cap_apply_btn.click()
    assert len(actions.calls_named("capability_status")) == 1
    assert "list_dir, read_file" in dialog.cap_status.text()


def test_apply_never_logs_an_mcp_servers_environment(
    dialog: ModelsDialog, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setattr(dialog, "_prompt_mcp_server", lambda existing: _server("s", env={"API_KEY": "hunter2"}))
    dialog.cap_mcp_add_btn.click()
    with caplog.at_level(logging.DEBUG):
        dialog.cap_apply_btn.click()
    assert "hunter2" not in caplog.text
    assert "capabilities.mcp" in caplog.text
    texts = [dialog.cap_mcp_table.item(0, c).text() for c in range(5)]
    assert not any("hunter2" in t for t in texts)


# ====================================================================================== restore
def test_restore_defaults_resets_the_tab_but_does_not_apply(
    make_dialog, cfg: Config, tmp_path
) -> None:
    cc = cfg.capabilities
    cc.read_files, cc.edit_files = False, True
    cc.read_roots = [str(tmp_path / "a")]
    cc.allowed_commands = ["dir"]
    cc.web_allowed_domains = ["example.com"]
    cc.mcp = [_server("fs")]
    cc.max_tool_calls = 9
    d = make_dialog(cfg)
    emitted: list[dict] = []
    d.applied.connect(emitted.append)

    d.cap_restore_btn.click()

    assert emitted == []
    assert (cc.read_files, cc.edit_files, cc.max_tool_calls) == (False, True, 9)    # the config is untouched
    assert cc.read_roots == [str(tmp_path / "a")] and cc.mcp == [_server("fs")]
    assert d.cap_read_cb.isChecked() is True and d.cap_edit_cb.isChecked() is False
    assert d.cap_read_roots.count() == 0
    assert _lines(d.cap_commands) == list(SAFE_COMMANDS)
    assert _lines(d.cap_web_allowed) == []
    assert d.cap_mcp_table.rowCount() == 0
    assert d._capabilities_form.widgets["capabilities.max_tool_calls"].value() == Config().capabilities.max_tool_calls
    assert "Apply" in d.cap_note.text()
    # ... and a following Apply is what writes the defaults
    d.cap_apply_btn.click()
    assert len(emitted) == 1
    assert cc.read_roots == [] and cc.mcp == [] and cc.read_files is True and cc.max_tool_calls == 4


# =========================================================================================== MCP
def test_add_edit_and_remove_a_server_through_the_prompt(
    dialog: ModelsDialog, cfg: Config, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: list[Any] = []
    answers = iter([_server("fs"), _server("web", transport="http", url="https://h/mcp"),
                    _server("fs2", args=["-y", "other"])])
    monkeypatch.setattr(dialog, "_prompt_mcp_server", lambda existing: (seen.append(existing), next(answers))[1])
    dialog.cap_mcp_add_btn.click()
    dialog.cap_mcp_add_btn.click()
    assert dialog.cap_mcp_table.rowCount() == 2
    assert seen == [None, None]                       # Add passes no existing server

    dialog.cap_mcp_table.setCurrentCell(0, 0)
    dialog.cap_mcp_edit_btn.click()
    assert seen[2]["name"] == "fs"                    # Edit passes the row's spec
    assert _cells(dialog.cap_mcp_table, 0)[0] == "fs2"
    assert _cells(dialog.cap_mcp_table, 0)[2] == "npx -y other"

    dialog.cap_mcp_table.setCurrentCell(1, 0)
    dialog.cap_mcp_remove_btn.click()
    assert [s["name"] for s in dialog._mcp_specs] == ["fs2"]
    assert cfg.capabilities.mcp == []                 # nothing is written before Apply


def test_a_cancelled_prompt_changes_nothing(dialog: ModelsDialog, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(dialog, "_prompt_mcp_server", lambda existing: None)
    dialog.cap_mcp_add_btn.click()
    assert dialog.cap_mcp_table.rowCount() == 0
    dialog._mcp_specs.append(_server("fs"))
    dialog._refresh_mcp_table()
    dialog.cap_mcp_table.setCurrentCell(0, 0)
    dialog.cap_mcp_edit_btn.click()
    assert dialog._mcp_specs == [_server("fs")]


def test_a_duplicate_server_name_is_refused(dialog: ModelsDialog, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(dialog, "_prompt_mcp_server", lambda existing: _server("fs"))
    dialog.cap_mcp_add_btn.click()
    dialog.cap_mcp_add_btn.click()
    assert dialog.cap_mcp_table.rowCount() == 1
    assert "already listed" in dialog.cap_note.text()


def test_the_row_buttons_follow_the_selection(dialog: ModelsDialog) -> None:
    dialog._mcp_specs.append(_server("fs"))
    dialog._refresh_mcp_table()
    dialog.cap_mcp_table.setCurrentCell(-1, -1)
    for b in (dialog.cap_mcp_edit_btn, dialog.cap_mcp_remove_btn, dialog.cap_mcp_test_btn):
        assert b.isEnabled() is False
    assert dialog.cap_mcp_add_btn.isEnabled() is True
    dialog.cap_mcp_table.setCurrentCell(0, 0)
    for b in (dialog.cap_mcp_edit_btn, dialog.cap_mcp_remove_btn, dialog.cap_mcp_test_btn):
        assert b.isEnabled() is True


def test_the_table_hides_secret_url_query_values(dialog: ModelsDialog) -> None:
    dialog._mcp_specs.append(_server("api", transport="http", url="https://h/mcp?token=abc123&x=1&api_key=zzz"))
    dialog._refresh_mcp_table()
    target = dialog.cap_mcp_table.item(0, 2).text()
    assert target == "https://h/mcp?token=***&x=1&api_key=***"
    assert "abc123" not in target and "zzz" not in target


def test_test_asks_the_engine_with_a_counter_and_the_event_fills_the_tools_column(
    make_dialog, cfg: Config, actions: FakeActions
) -> None:
    cfg.capabilities.mcp = [_server("fs"), _server("web", transport="http", url="https://h/mcp")]
    d = make_dialog(cfg)
    d.cap_mcp_table.setCurrentCell(0, 0)
    d.cap_mcp_test_btn.click()
    assert actions.calls_named("mcp_server_tools") == [(("fs", 1), {})]
    assert d.cap_mcp_table.item(0, 4).text().startswith("testing")

    d.cap_mcp_table.setCurrentCell(1, 0)
    d.cap_mcp_test_btn.click()
    assert actions.calls_named("mcp_server_tools")[-1] == (("web", 2), {})

    d.on_event(McpToolsEvent(request_id=1, server="fs",
                             tools=({"name": "read", "description": "r"}, {"name": "write", "description": "w"})))
    assert d.cap_mcp_table.item(0, 4).text() == "2 tools: read, write"
    assert "read" in d.cap_mcp_table.item(0, 4).text()
    d.on_event(McpToolsEvent(request_id=2, server="web", error="connection refused"))
    assert d.cap_mcp_table.item(1, 4).text() == "error: connection refused"
    assert d.cap_mcp_table.item(0, 4).text() == "2 tools: read, write"    # the other row is not disturbed


def test_a_stale_request_id_is_ignored(make_dialog, cfg: Config, actions: FakeActions) -> None:
    cfg.capabilities.mcp = [_server("fs")]
    d = make_dialog(cfg)
    d.cap_mcp_table.setCurrentCell(0, 0)
    d.cap_mcp_test_btn.click()          # request 1
    d.cap_mcp_test_btn.click()          # request 2 supersedes it
    assert [c[0] for c in actions.calls_named("mcp_server_tools")] == [("fs", 1), ("fs", 2)]

    d.on_event(McpToolsEvent(request_id=1, server="fs", tools=({"name": "old", "description": ""},)))
    assert d.cap_mcp_table.item(0, 4).text().startswith("testing")        # the old answer changed nothing
    d.on_event(McpToolsEvent(request_id=2, server="fs", tools=({"name": "new", "description": ""},)))
    assert d.cap_mcp_table.item(0, 4).text() == "1 tool: new"
    d.on_event(McpToolsEvent(request_id=2, server="fs", tools=({"name": "again", "description": ""},)))
    assert d.cap_mcp_table.item(0, 4).text() == "1 tool: new"             # an answer is taken once
    d.on_event(McpToolsEvent(request_id=9, server="ghost", tools=()))     # nobody asked: ignored, no crash


def test_a_long_tool_list_is_cut_and_zero_tools_say_so(make_dialog, cfg: Config) -> None:
    cfg.capabilities.mcp = [_server("fs")]
    d = make_dialog(cfg)
    d.cap_mcp_table.setCurrentCell(0, 0)
    d.cap_mcp_test_btn.click()
    names = [f"t{i}" for i in range(12)]
    d.on_event(McpToolsEvent(request_id=1, server="fs", tools=tuple({"name": n, "description": ""} for n in names)))
    text = d.cap_mcp_table.item(0, 4).text()
    assert text.startswith("12 tools: t0, t1")
    assert text.endswith("\u2026") and "t11" not in text
    d.cap_mcp_test_btn.click()
    d.on_event(McpToolsEvent(request_id=2, server="fs", tools=()))
    assert d.cap_mcp_table.item(0, 4).text() == "0 tools"


def test_testing_a_server_that_is_not_applied_yet_hints_at_apply(
    dialog: ModelsDialog, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(dialog, "_prompt_mcp_server", lambda existing: _server("new"))
    dialog.cap_mcp_add_btn.click()
    dialog.cap_mcp_test_btn.click()
    dialog.on_event(McpToolsEvent(request_id=1, server="new", error="no such server"))
    assert "Apply first" in dialog.cap_mcp_table.item(0, 4).text()


def test_a_removed_servers_late_answer_is_ignored(make_dialog, cfg: Config) -> None:
    cfg.capabilities.mcp = [_server("fs")]
    d = make_dialog(cfg)
    d.cap_mcp_table.setCurrentCell(0, 0)
    d.cap_mcp_test_btn.click()
    d.cap_mcp_remove_btn.click()
    d.on_event(McpToolsEvent(request_id=1, server="fs", tools=({"name": "x", "description": ""},)))
    assert d.cap_mcp_table.rowCount() == 0 and d._mcp_tools_text == {}


# -- the real prompt dialog (driven from a timer while exec() runs)
@contextlib.contextmanager
def _drive_modal(fn, stuck_ms: int = 2000):
    """Run ``fn(dialog)`` from a timer while a modal ``exec()`` blocks inside the ``with`` body.

    Qt only prints an exception raised in a timer callback, and a dialog that nobody closes keeps
    ``exec()`` blocking for ever (there is no pytest-timeout). So a failure in ``fn`` is recorded and
    the dialog rejected, a repeating guard rejects whatever modal is still open every ``stuck_ms``
    (the first tick can free a nested message box, the next the dialog itself), and the recorded
    error is re-raised when the block ends -- as a failure of the test, with the real cause."""
    errors: list[BaseException] = []

    def reject_active() -> None:
        w = QApplication.activeModalWidget()
        if w is not None:
            w.reject()

    def unstick() -> None:
        errors.append(TimeoutError("the driven modal dialog was still open"))
        reject_active()

    guard = QTimer()
    guard.setInterval(stuck_ms)
    guard.timeout.connect(unstick)
    guard.start()

    def run() -> None:
        try:
            dlg = QApplication.activeModalWidget()
            assert dlg is not None, "no modal dialog was open"
            fn(dlg)
        except BaseException as e:      # noqa: BLE001 - re-raised when the block ends
            errors.append(e)
            reject_active()             # on failure only: rejecting after a successful accept() flips exec() to Rejected

    QTimer.singleShot(0, run)
    try:
        yield
    finally:
        guard.stop()
    if errors:
        raise errors[0]


def _ok(dlg) -> None:
    dlg.findChild(QDialogButtonBox).button(QDialogButtonBox.StandardButton.Ok).click()


def test_the_real_prompt_builds_a_stdio_spec(dialog: ModelsDialog) -> None:
    def fill(dlg) -> None:
        name, command, url = dlg.findChildren(QLineEdit)[:3]
        args, env = dlg.findChildren(QPlainTextEdit)[:2]
        name.setText("fs")
        command.setText("npx")
        args.setPlainText("-y\n\n@scope/server-filesystem\nD:/work\n")
        env.setPlainText("ROOT=D:/work\n  TOKEN = abc=def  \n")
        _ok(dlg)

    with _drive_modal(fill):
        spec = dialog._prompt_mcp_server(None)
    assert spec == {"name": "fs", "transport": "stdio", "command": "npx",
                    "args": ["-y", "@scope/server-filesystem", "D:/work"],
                    "env": {"ROOT": "D:/work", "TOKEN": "abc=def"}, "enabled": True}


def test_the_real_prompt_builds_an_http_spec_and_keeps_unknown_keys(dialog: ModelsDialog) -> None:
    existing = {"name": "api", "transport": "stdio", "command": "x", "args": [], "env": {}, "enabled": False,
                "note": "kept"}

    def fill(dlg) -> None:
        dlg.findChild(QComboBox).setCurrentText("http")
        dlg.findChildren(QLineEdit)[2].setText("https://h/mcp")
        _ok(dlg)

    with _drive_modal(fill):
        spec = dialog._prompt_mcp_server(existing)
    assert spec == {"name": "api", "transport": "http", "url": "https://h/mcp", "env": {}, "enabled": False,
                    "note": "kept"}


def test_the_real_prompt_refuses_bad_input_and_can_be_cancelled(
    dialog: ModelsDialog, monkeypatch: pytest.MonkeyPatch
) -> None:
    warnings: list[str] = []
    monkeypatch.setattr(QMessageBox, "warning", staticmethod(lambda parent, title, text: warnings.append(text)))

    def fill(dlg) -> None:
        dlg.findChildren(QLineEdit)[0].setText("bad name!")        # not a valid name
        _ok(dlg)
        dlg.findChildren(QLineEdit)[0].setText("ok")
        dlg.findChildren(QLineEdit)[1].setText("")                 # a stdio server needs a command
        _ok(dlg)
        dlg.findChildren(QLineEdit)[1].setText("npx")
        dlg.findChildren(QPlainTextEdit)[1].setPlainText("NOEQUALS")   # env must be KEY=VALUE
        _ok(dlg)
        dlg.reject()

    with _drive_modal(fill):
        result = dialog._prompt_mcp_server(None)
    assert result is None
    assert len(warnings) == 3
    assert "name" in warnings[0].lower() and "command" in warnings[1].lower() and "KEY=VALUE" in warnings[2]


# =========================================================================================== maps
def test_map_rows_render(dialog: ModelsDialog, actions: FakeActions) -> None:
    actions.maps_result = [
        _map_row("proj", roots=["C:/work/proj", "C:/work/docs"], files=12, active=True),
        _map_row("big", files=4000, truncated=True, indexed_at="2026-10-06T23:59:59"),
    ]
    dialog._refresh_maps()
    t = dialog.cap_maps_table
    assert [t.horizontalHeaderItem(c).text() for c in range(t.columnCount())] == [
        "Name", "Roots", "Files", "Indexed", "Active"]
    assert _cells(t, 0) == ["proj", "C:/work/proj; C:/work/docs", "12", "2026-10-07 10:00", "active"]
    assert _cells(t, 1) == ["big", "C:/work/proj", "4000 (cut)", "2026-10-06 23:59", ""]
    assert "3 folders" in t.item(0, 0).toolTip()


def test_none_and_empty_map_lists_render_an_empty_table(dialog: ModelsDialog, actions: FakeActions) -> None:
    actions.maps_result = None
    dialog._refresh_maps()
    assert dialog.cap_maps_table.rowCount() == 0
    actions.maps_result = []
    dialog._refresh_maps()
    assert dialog.cap_maps_table.rowCount() == 0


def test_the_selected_map_is_kept_by_name_across_refreshes(dialog: ModelsDialog, actions: FakeActions) -> None:
    actions.maps_result = [_map_row("a"), _map_row("b"), _map_row("c")]
    dialog._refresh_maps()
    dialog.cap_maps_table.setCurrentCell(1, 0)
    actions.maps_result = [_map_row("z"), _map_row("a"), _map_row("b"), _map_row("c")]
    dialog.on_event(CapabilityMapEvent(reason="indexed", name="z", files=1, dirs=1))
    assert dialog.cap_maps_table.currentRow() == 2
    assert dialog._selected_map_row()["name"] == "b"


def test_the_map_buttons_follow_the_selection(dialog: ModelsDialog, actions: FakeActions) -> None:
    actions.maps_result = [_map_row("a")]
    dialog._refresh_maps()
    dialog.cap_maps_table.setCurrentCell(-1, -1)
    for b in (dialog.cap_map_load_btn, dialog.cap_map_reindex_btn, dialog.cap_map_delete_btn):
        assert b.isEnabled() is False
    assert dialog.cap_index_btn.isEnabled() and dialog.cap_map_open_btn.isEnabled()
    dialog.cap_maps_table.setCurrentCell(0, 0)
    for b in (dialog.cap_map_load_btn, dialog.cap_map_reindex_btn, dialog.cap_map_delete_btn):
        assert b.isEnabled() is True


def test_pre_index_with_no_readable_folder_says_so_and_calls_nothing(
    dialog: ModelsDialog, actions: FakeActions, monkeypatch: pytest.MonkeyPatch
) -> None:
    prompted: list[str] = []
    monkeypatch.setattr(dialog, "_prompt_map_name", lambda default: prompted.append(default) or "x")
    dialog.cap_index_btn.click()
    assert "readable folder" in dialog.cap_map_status.text()
    assert prompted == []
    assert actions.calls_named("index_capability_map") == []
    assert [n for (n, _a, _k) in actions.calls if n != "capability_status"] == []


def test_pre_index_applies_the_tab_first_then_indexes_under_the_prompted_name(
    dialog: ModelsDialog, cfg: Config, actions: FakeActions, tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = str(tmp_path / "myproject")
    monkeypatch.setattr(dialog, "_pick_directory", lambda: root)
    dialog.cap_read_add_btn.click()
    dialog.cap_edit_cb.setChecked(True)            # an unapplied change: the map must carry it
    prompted: list[str] = []
    monkeypatch.setattr(dialog, "_prompt_map_name", lambda default: prompted.append(default) or "My map")
    dialog.applied.connect(lambda changed: actions.order.append("applied"))

    dialog.cap_index_btn.click()

    assert prompted == ["myproject"]                          # default = the first root's folder name
    assert actions.order == ["applied", "index:My map"]       # applied BEFORE the engine is asked
    assert actions.calls_named("index_capability_map") == [(("My map",), {})]
    assert cfg.capabilities.read_roots == [root] and cfg.capabilities.edit_files is True


def test_pre_index_over_an_existing_map_asks_then_deletes_it_and_indexes_the_tabs_folders(
    dialog: ModelsDialog, cfg: Config, actions: FakeActions, tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    old_root, root = str(tmp_path / "old"), str(tmp_path / "new")
    actions.maps_result = [_map_row("proj", roots=[old_root])]
    dialog._refresh_maps()
    dialog.cap_read_roots.addItem(root)
    monkeypatch.setattr(dialog, "_prompt_map_name", lambda default: "proj")
    asked: list[tuple[str, str]] = []
    monkeypatch.setattr(dialog, "_confirm", lambda title, text: asked.append((title, text)) or True)
    dialog.applied.connect(lambda changed: actions.order.append("applied"))
    actions.delete_capability_map = lambda name: actions.order.append(f"delete:{name}")   # type: ignore[method-assign]

    dialog.cap_index_btn.click()

    assert len(asked) == 1 and asked[0][0] == "Replace map"
    assert "'proj'" in asked[0][1] and old_root in asked[0][1] and root in asked[0][1]
    assert "Replace it with the folders in the tab" in asked[0][1]
    # the tab is applied first (the map carries what it shows), then the old map goes, then the index runs
    assert actions.order == ["applied", "delete:proj", "index:proj"]
    assert cfg.capabilities.read_roots == [root]


def test_pre_index_over_an_existing_map_does_nothing_when_declined_or_the_confirm_fails(
    dialog: ModelsDialog, cfg: Config, actions: FakeActions, tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    emitted: list[dict] = []
    dialog.applied.connect(emitted.append)
    actions.maps_result = [_map_row("proj")]
    dialog._refresh_maps()
    dialog.cap_read_roots.addItem(str(tmp_path / "new"))
    monkeypatch.setattr(dialog, "_prompt_map_name", lambda default: "proj")

    monkeypatch.setattr(dialog, "_confirm", lambda title, text: False)
    dialog.cap_index_btn.click()

    def boom(*a: Any, **kw: Any) -> bool:
        raise RuntimeError("dialog blew up")

    monkeypatch.setattr(dialog, "_confirm", boom)
    dialog.cap_index_btn.click()

    assert actions.calls_named("delete_capability_map") == []
    assert actions.calls_named("index_capability_map") == []
    assert emitted == [] and cfg.capabilities.read_roots == []     # a No applies nothing either


def test_pre_index_over_an_existing_map_stops_when_the_old_one_cannot_be_deleted(
    dialog: ModelsDialog, actions: FakeActions, tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The engine would re-index the old map from ITS settings, ignoring the tab: say so instead."""
    actions.maps_result = [_map_row("proj")]
    dialog._refresh_maps()
    dialog.cap_read_roots.addItem(str(tmp_path / "new"))
    monkeypatch.setattr(dialog, "_prompt_map_name", lambda default: "proj")
    monkeypatch.setattr(dialog, "_confirm", lambda title, text: True)
    actions.raise_on = {"delete_capability_map"}
    dialog.cap_index_btn.click()
    assert "failed to replace proj" in dialog.cap_map_status.text()
    assert actions.calls_named("index_capability_map") == []


def test_pre_index_under_a_new_name_neither_asks_nor_deletes(
    dialog: ModelsDialog, actions: FakeActions, tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    actions.maps_result = [_map_row("proj")]
    dialog._refresh_maps()
    dialog.cap_read_roots.addItem(str(tmp_path / "new"))
    monkeypatch.setattr(dialog, "_prompt_map_name", lambda default: "Proj")      # a different name (case counts)
    monkeypatch.setattr(dialog, "_confirm", lambda title, text: pytest.fail("no confirmation expected"))
    dialog.cap_index_btn.click()
    assert actions.calls_named("delete_capability_map") == []
    assert actions.calls_named("index_capability_map") == [(("Proj",), {})]


def test_pre_index_with_a_cancelled_prompt_applies_and_calls_nothing(
    dialog: ModelsDialog, cfg: Config, actions: FakeActions, tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dialog.cap_read_roots.addItem(str(tmp_path / "proj"))
    monkeypatch.setattr(dialog, "_prompt_map_name", lambda default: None)
    dialog.cap_index_btn.click()
    assert actions.calls_named("index_capability_map") == []
    assert cfg.capabilities.read_roots == []


def test_the_map_name_prompt_returns_a_stripped_name_or_none(
    dialog: ModelsDialog, monkeypatch: pytest.MonkeyPatch
) -> None:
    from PySide6.QtWidgets import QInputDialog

    monkeypatch.setattr(QInputDialog, "getText", staticmethod(lambda *a, **kw: ("  proj  ", True)))
    assert dialog._prompt_map_name("d") == "proj"
    monkeypatch.setattr(QInputDialog, "getText", staticmethod(lambda *a, **kw: ("proj", False)))
    assert dialog._prompt_map_name("d") is None
    monkeypatch.setattr(QInputDialog, "getText", staticmethod(lambda *a, **kw: ("   ", True)))
    assert dialog._prompt_map_name("d") is None


def test_re_index_calls_the_selected_maps_name(
    dialog: ModelsDialog, actions: FakeActions
) -> None:
    actions.maps_result = [_map_row("a"), _map_row("b")]
    dialog._refresh_maps()
    dialog.cap_maps_table.setCurrentCell(1, 0)
    dialog.cap_map_reindex_btn.click()
    assert actions.calls_named("index_capability_map") == [(("b",), {})]


def test_re_index_keeps_the_maps_own_scope_and_applies_nothing_from_the_tab(
    dialog: ModelsDialog, cfg: Config, actions: FakeActions, tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The engine re-indexes an existing map from the settings it was saved with: the tab (other
    folders, an unapplied switch, a Restore defaults) must neither be applied nor asked about."""
    emitted: list[dict] = []
    dialog.applied.connect(emitted.append)
    actions.maps_result = [_map_row("alpha", roots=[str(tmp_path / "alpha")])]
    dialog._refresh_maps()
    dialog.cap_edit_cb.setChecked(True)                            # an unapplied edit
    dialog.cap_restore_btn.click()                                 # restored defaults (mcp = [], pending)
    dialog.cap_read_roots.addItem(str(tmp_path / "beta"))          # and other folders on the tab
    asked: list[str] = []
    monkeypatch.setattr(dialog, "_confirm", lambda title, text: asked.append(text) or True)
    dialog.cap_maps_table.setCurrentCell(0, 0)

    dialog.cap_map_reindex_btn.click()

    assert actions.calls_named("index_capability_map") == [(("alpha",), {})]
    assert emitted == [] and asked == []
    assert cfg.capabilities.read_roots == [] and cfg.capabilities.edit_files is False   # nothing was written
    assert "Apply" in dialog.cap_note.text()                       # the restore note is still pending


def test_the_re_index_button_says_it_uses_the_maps_own_folders(dialog: ModelsDialog) -> None:
    assert "made from" in dialog.cap_map_reindex_btn.toolTip()


def test_load_confirms_with_what_changes_and_then_calls(
    dialog: ModelsDialog, cfg: Config, actions: FakeActions, tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    r_read, r_edit = str(tmp_path / "src"), str(tmp_path / "out")
    actions.maps_result = [_map_row("proj", settings={
        "read_files": True,                        # unchanged: not listed
        "edit_files": True,                        # off -> on
        "execute_commands": False,                 # on -> off
        "read_roots": [r_read], "edit_roots": [r_edit],
        "allowed_commands": ["dir"],
    })]
    dialog._refresh_maps()
    dialog.cap_maps_table.setCurrentCell(0, 0)
    asked: list[tuple[str, str]] = []
    monkeypatch.setattr(dialog, "_confirm", lambda title, text: asked.append((title, text)) or True)

    dialog.cap_map_load_btn.click()

    assert len(asked) == 1
    title, text = asked[0]
    assert title == "Load map" and "'proj'" in text
    assert "Edit files: on (now off)" in text
    assert "Execute commands: off (now on)" in text
    assert "Read files" not in text
    assert r_read in text and r_edit in text
    assert actions.calls_named("load_capability_map") == [(("proj",), {})]


def test_load_lists_the_mcp_servers_it_would_flip_but_never_adds_one(
    make_dialog, cfg: Config, actions: FakeActions, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg.capabilities.mcp = [_server("fs", enabled=True), _server("web", enabled=False), _server("same")]
    actions.maps_result = [_map_row("m", settings={"mcp_enabled": ["web", "same", "ghost"]})]
    d = make_dialog(cfg)
    d.cap_maps_table.setCurrentCell(0, 0)
    asked: list[str] = []
    monkeypatch.setattr(d, "_confirm", lambda title, text: asked.append(text) or False)
    d.cap_map_load_btn.click()
    assert "MCP server 'fs': off (now on)" in asked[0]
    assert "MCP server 'web': on (now off)" in asked[0]
    assert "'same'" not in asked[0] and "ghost" not in asked[0]


def test_load_with_nothing_to_change_says_so(
    dialog: ModelsDialog, actions: FakeActions, monkeypatch: pytest.MonkeyPatch
) -> None:
    actions.maps_result = [_map_row("same", settings={"read_files": True, "edit_files": False})]
    dialog._refresh_maps()
    dialog.cap_maps_table.setCurrentCell(0, 0)
    asked: list[str] = []
    monkeypatch.setattr(dialog, "_confirm", lambda title, text: asked.append(text) or False)
    dialog.cap_map_load_btn.click()
    assert "leaves the switches and folders as they are" in asked[0]
    assert "never turns on" in asked[0]


def test_load_warns_that_unapplied_edits_are_replaced(
    dialog: ModelsDialog, actions: FakeActions, monkeypatch: pytest.MonkeyPatch
) -> None:
    actions.maps_result = [_map_row("m", settings={"edit_files": True})]
    dialog._refresh_maps()
    dialog.cap_maps_table.setCurrentCell(0, 0)
    asked: list[str] = []
    monkeypatch.setattr(dialog, "_confirm", lambda title, text: asked.append(text) or False)
    dialog.cap_map_load_btn.click()
    assert "not applied" not in asked[0]
    dialog.cap_web_cb.setChecked(True)
    dialog.cap_map_load_btn.click()
    assert "not applied" in asked[1]


def test_a_declined_confirmation_calls_nothing(
    dialog: ModelsDialog, actions: FakeActions, monkeypatch: pytest.MonkeyPatch
) -> None:
    actions.maps_result = [_map_row("proj")]
    dialog._refresh_maps()
    dialog.cap_maps_table.setCurrentCell(0, 0)
    monkeypatch.setattr(dialog, "_confirm", lambda title, text: False)
    dialog.cap_map_load_btn.click()
    dialog.cap_map_delete_btn.click()
    assert actions.calls_named("load_capability_map") == []
    assert actions.calls_named("delete_capability_map") == []


def test_delete_confirms_then_calls(
    dialog: ModelsDialog, actions: FakeActions, monkeypatch: pytest.MonkeyPatch
) -> None:
    actions.maps_result = [_map_row("a"), _map_row("b")]
    dialog._refresh_maps()
    dialog.cap_maps_table.setCurrentCell(1, 0)
    asked: list[tuple[str, str]] = []
    monkeypatch.setattr(dialog, "_confirm", lambda title, text: asked.append((title, text)) or True)
    dialog.cap_map_delete_btn.click()
    assert asked[0][0] == "Delete map" and "'b'" in asked[0][1]
    assert actions.calls_named("delete_capability_map") == [(("b",), {})]


def test_confirm_uses_a_yes_no_message_box(dialog: ModelsDialog, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(QMessageBox, "question", staticmethod(lambda *a, **kw: QMessageBox.StandardButton.Yes))
    assert dialog._confirm("t", "x") is True
    monkeypatch.setattr(QMessageBox, "question", staticmethod(lambda *a, **kw: QMessageBox.StandardButton.No))
    assert dialog._confirm("t", "x") is False


def test_open_folder_opens_the_maps_folder_when_it_exists(
    dialog: ModelsDialog, cfg: Config, tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    opened: list[object] = []
    monkeypatch.setattr(models_mod.QDesktopServices, "openUrl", staticmethod(lambda url: opened.append(url)))
    dialog.cap_map_open_btn.click()
    assert opened == [] and "does not exist" in dialog.cap_map_status.text()

    cfg.maps_dir().mkdir(parents=True)
    dialog.cap_map_open_btn.click()
    assert len(opened) == 1
    assert opened[0].toLocalFile().replace("\\", "/").endswith("models/maps")


# ============================================================================== map events
def test_map_events_update_the_status_text_per_reason(dialog: ModelsDialog, actions: FakeActions) -> None:
    dialog.on_event(CapabilityMapEvent(reason="indexing", name="proj"))
    assert dialog.cap_map_status.text() == "Indexing proj\u2026"

    actions.maps_result = [_map_row("proj", files=40, dirs=6, active=True)]
    dialog.on_event(CapabilityMapEvent(reason="indexed", name="proj", files=40, dirs=6))
    assert dialog.cap_map_status.text() == "Map 'proj': 40 files, 6 folders"
    assert dialog.cap_maps_table.rowCount() == 1                      # the list was re-read
    assert _cells(dialog.cap_maps_table, 0)[4] == "active"

    dialog.on_event(CapabilityMapEvent(reason="indexed", name="big", files=4000, dirs=90, detail="cut at 4000 files"))
    assert dialog.cap_map_status.text() == "Map 'big': 4000 files, 90 folders (cut at 4000 files)"

    dialog.on_event(CapabilityMapEvent(reason="loaded", name="proj"))
    assert dialog.cap_map_status.text() == "Loaded 'proj'"

    actions.maps_result = []
    dialog.on_event(CapabilityMapEvent(reason="deleted", name="proj"))
    assert dialog.cap_map_status.text() == "Deleted 'proj'"
    assert dialog.cap_maps_table.rowCount() == 0

    dialog.on_event(CapabilityMapEvent(reason="error: folder not found: C:/gone"))
    assert dialog.cap_map_status.text() == "error: folder not found: C:/gone"


def test_a_loaded_map_refills_the_tab_from_the_config_the_engine_just_rewrote(
    dialog: ModelsDialog, cfg: Config, actions: FakeActions, tmp_path
) -> None:
    emitted: list[dict] = []
    dialog.applied.connect(emitted.append)
    dialog.cap_apply_btn.click()
    cc = cfg.capabilities
    # what load_capability_map does on the engine side
    cc.read_files, cc.edit_files = False, True
    cc.read_roots = [str(tmp_path / "from-the-map")]
    cc.allowed_commands = ["dir"]
    cc.mcp = [_server("fs", enabled=False)]
    dialog.cap_note.setText("stale")

    dialog.on_event(CapabilityMapEvent(reason="loaded", name="proj"))

    assert dialog.cap_read_cb.isChecked() is False and dialog.cap_edit_cb.isChecked() is True
    assert dialog.cap_read_roots.item(0).text() == str(tmp_path / "from-the-map")
    assert _lines(dialog.cap_commands) == ["dir"]
    assert _cells(dialog.cap_mcp_table, 0)[3] == "no"
    assert dialog.cap_note.text() == ""
    # so a later Apply does not put the old values back
    assert dialog._capabilities_changes() == {}
    dialog.cap_apply_btn.click()
    assert emitted == [] and cc.read_files is False


def test_map_events_refresh_the_status_line(dialog: ModelsDialog, actions: FakeActions) -> None:
    dialog.on_event(CapabilityMapEvent(reason="indexed", name="proj", files=1, dirs=1))
    assert len(actions.calls_named("capability_status")) == 1


# ========================================================================================= status
def test_the_status_line_reports_tools_servers_map_and_notes(dialog: ModelsDialog, actions: FakeActions) -> None:
    actions.status_result = {
        "tools": ["list_dir", "read_file"], "read_roots": ["C:/w"],
        "mcp": {"fs": "connected", "web": "error: refused"}, "active_map": "proj",
        "notes": ["web: unreachable"],
    }
    dialog._refresh_capability_status()
    text = dialog.cap_status.text()
    assert "Offered to the ask box: list_dir, read_file" in text
    assert "fs connected" in text and "web error: refused" in text
    assert "Active map: proj" in text
    assert text.splitlines()[-1] == "web: unreachable"


def test_the_status_line_says_when_no_tool_is_offered(dialog: ModelsDialog, actions: FakeActions) -> None:
    actions.status_result = {"tools": [], "read_roots": [], "mcp": {}, "active_map": "",
                             "notes": ["no readable folder: the model gets no tools"]}
    dialog._refresh_capability_status()
    assert dialog.cap_status.text().startswith("No tools are offered right now")
    assert "no readable folder" in dialog.cap_status.text()


def test_a_none_or_malformed_status_leaves_the_line_empty(dialog: ModelsDialog, actions: FakeActions) -> None:
    for bad in (None, {}, "nonsense", {"tools": 5, "mcp": [1], "notes": "x"}):
        actions.status_result = bad     # type: ignore[assignment]
        dialog._refresh_capability_status()
    assert dialog.cap_status.text().startswith("No tools are offered right now")   # the last, malformed one
    actions.status_result = None
    dialog._refresh_capability_status()
    assert dialog.cap_status.text() == ""


# ====================================================================== tolerance: None / raising
def test_loading_survives_every_capability_call_raising(make_dialog, cfg: Config, actions: FakeActions) -> None:
    actions.raise_on = {"capability_maps", "capability_status"}
    d = make_dialog(cfg)                  # construction calls load(): must not raise
    d.load()
    assert d.cap_maps_table.rowCount() == 0
    assert d.cap_status.text() == ""
    assert "failed" in d.cap_map_status.text() or d.cap_map_status.text() == ""
    d.on_event(CapabilityMapEvent(reason="indexed", name="x", files=1, dirs=1))
    d.on_event(McpToolsEvent(request_id=1, server="x"))
    assert d.isVisible()


def test_failing_map_and_server_actions_are_swallowed_and_reported(
    make_dialog, cfg: Config, actions: FakeActions, tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg.capabilities.read_roots = [str(tmp_path / "proj")]
    cfg.capabilities.mcp = [_server("fs")]
    actions.maps_result = [_map_row("proj")]
    d = make_dialog(cfg)
    d.cap_maps_table.setCurrentCell(0, 0)
    d.cap_mcp_table.setCurrentCell(0, 0)
    monkeypatch.setattr(d, "_confirm", lambda title, text: True)
    monkeypatch.setattr(d, "_prompt_map_name", lambda default: "proj")
    actions.raise_on = {"index_capability_map", "load_capability_map", "delete_capability_map",
                        "mcp_server_tools"}

    d.cap_index_btn.click()
    assert "failed" in d.cap_map_status.text()
    d.cap_map_reindex_btn.click()
    assert "failed" in d.cap_map_status.text()
    d.cap_map_load_btn.click()
    assert "failed to load" in d.cap_map_status.text()
    d.cap_map_delete_btn.click()
    assert "failed to delete" in d.cap_map_status.text()
    d.cap_mcp_test_btn.click()
    assert d.cap_mcp_table.item(0, 4).text().startswith("error")
    assert d.isVisible()


def test_a_failing_confirm_or_prompt_never_reaches_the_engine(
    make_dialog, cfg: Config, actions: FakeActions, tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg.capabilities.read_roots = [str(tmp_path / "proj")]
    actions.maps_result = [_map_row("proj")]
    d = make_dialog(cfg)
    d.cap_maps_table.setCurrentCell(0, 0)

    def boom(*a: Any, **kw: Any) -> Any:
        raise RuntimeError("dialog blew up")

    monkeypatch.setattr(d, "_confirm", boom)
    monkeypatch.setattr(d, "_prompt_map_name", boom)
    monkeypatch.setattr(d, "_prompt_mcp_server", boom)
    d.cap_map_load_btn.click()
    d.cap_map_delete_btn.click()
    d.cap_index_btn.click()
    d.cap_mcp_add_btn.click()
    assert actions.calls_named("load_capability_map") == []
    assert actions.calls_named("delete_capability_map") == []
    assert actions.calls_named("index_capability_map") == []
    assert d._mcp_specs == []


def test_unrelated_events_still_work_next_to_the_capability_ones(dialog: ModelsDialog) -> None:
    dialog.on_event(ModelLibraryEvent(reason="loaded"))
    dialog.on_event(ComponentStatusEvent(component="llm", state="ready"))
    dialog.on_event(CapabilityMapEvent(reason="indexing", name="x"))
    dialog.on_event(McpToolsEvent(request_id=3, server="nobody"))
    assert dialog.cap_map_status.text() == "Indexing x\u2026"


# ============================================================================== status: size cap
_BUILTIN_TOOLS = ["edit_file", "fetch_url", "find_files", "list_dir", "read_file", "run_command",
                  "search_text", "web_search", "write_file"]


def _status_with(tools: list[str], **extra: object) -> dict[str, object]:
    return {"tools": tools, "read_roots": ["C:/w"], "mcp": {}, "active_map": "", "notes": [], **extra}


def test_a_big_mcp_server_cannot_flood_the_status_line() -> None:
    mcp_tools = [f"mcp:github:tool_{i}" for i in range(81)]
    builtin = ["find_files", "list_dir", "read_file", "search_text"]
    tools = sorted(builtin + mcp_tools)                     # the runner sorts: "mcp:" lands between list_dir and read_file
    text, state = models_mod._format_capability_status(_status_with(tools))
    first = text.splitlines()[0]
    assert state == ""
    for name in builtin:                                    # the built-in tools are never cut
        assert name in first
    assert first.count("mcp:github:") == 4                  # 8 names in all: the room left is filled with MCP ones
    assert first.endswith("+77 more")
    assert len(text) < 250


def test_the_built_in_tools_are_listed_in_full_even_past_the_name_cap() -> None:
    text, _state = models_mod._format_capability_status(_status_with(list(_BUILTIN_TOOLS)))
    assert text == "Offered to the ask box: " + ", ".join(_BUILTIN_TOOLS)         # nine names, no "more"
    text, _state = models_mod._format_capability_status(_status_with(_BUILTIN_TOOLS + ["mcp:s:a", "mcp:s:b"]))
    assert all(name in text for name in _BUILTIN_TOOLS)
    assert "mcp:s:a" not in text and text.splitlines()[0].endswith("+2 more")


def test_a_short_mcp_list_is_shown_whole() -> None:
    text, _state = models_mod._format_capability_status(_status_with(["list_dir", "mcp:fs:read", "read_file"]))
    assert text == "Offered to the ask box: list_dir, read_file, mcp:fs:read"


def test_a_status_with_only_mcp_tools_is_not_called_loading() -> None:
    text, state = models_mod._format_capability_status(_status_with(["mcp:fs:read"]))
    assert state == "" and "mcp:fs:read" in text
    assert models_mod._format_capability_status(_status_with([]))[1] == "loading"


# ============================================================================ status: live MCP states
_CONNECTING = _status_with(["read_file"], mcp={"fs": "connecting"})
_CONNECTED = _status_with(["read_file", "mcp:fs:echo"], mcp={"fs": "connected"})


def test_the_poll_runs_only_while_the_capabilities_tab_is_on_screen(dialog: ModelsDialog) -> None:
    timer = dialog._cap_poll_timer
    assert timer.interval() == 2000
    assert not timer.isActive()                      # the window opens on the Library tab
    dialog.show_tab("Capabilities")
    assert timer.isActive()
    dialog.show_tab("Context")
    assert not timer.isActive()
    dialog.show_tab("Capabilities")
    assert timer.isActive()
    dialog.hide()
    assert not timer.isActive()
    dialog.show()
    assert timer.isActive()                          # shown again on the tab it was left on
    dialog.close()
    assert not timer.isActive()


def test_a_poll_picks_up_a_server_that_finished_connecting(dialog: ModelsDialog, actions: FakeActions) -> None:
    """Apply returns while the engine's mcp-connect thread still runs: the line said "connecting" for ever."""
    actions.status_sequence = [_CONNECTING, _CONNECTED]
    dialog.show_tab("Capabilities")
    dialog._refresh_capability_status()
    assert "fs connecting" in dialog.cap_status.text() and "mcp:fs:echo" not in dialog.cap_status.text()
    dialog._poll_capability_status()
    assert "fs connected" in dialog.cap_status.text() and "mcp:fs:echo" in dialog.cap_status.text()


def test_the_timer_itself_refreshes_the_status_line(dialog: ModelsDialog, actions: FakeActions) -> None:
    actions.status_sequence = [_CONNECTING, _CONNECTED]
    dialog.show_tab("Capabilities")
    dialog._refresh_capability_status()
    dialog._cap_poll_timer.setInterval(10)
    dialog._cap_poll_timer.start()                   # restart with the short interval
    # a real timer: wait for the tick instead of for a fixed time (a loaded machine serves 10 ms
    # timers far slower than that)
    deadline = time.monotonic() + 10.0
    while "fs connected" not in dialog.cap_status.text() and time.monotonic() < deadline:
        QTest.qWait(20)
    assert "fs connected" in dialog.cap_status.text()


def test_a_stray_tick_off_screen_stops_the_timer_and_reads_nothing(
    dialog: ModelsDialog, actions: FakeActions
) -> None:
    dialog.show_tab("Capabilities")
    assert dialog._cap_poll_timer.isActive()
    dialog.tabs.blockSignals(True)                   # leave the tab without the sync
    dialog.tabs.setCurrentIndex(dialog.TABS.index("Hardware"))
    dialog.tabs.blockSignals(False)
    actions.calls.clear()
    dialog._poll_capability_status()
    assert not dialog._cap_poll_timer.isActive()
    assert actions.calls_named("capability_status") == []


def test_a_poll_that_finds_nothing_new_leaves_the_label_alone(dialog: ModelsDialog, actions: FakeActions) -> None:
    """A text selection in the status label must survive the 2 s refresh."""
    actions.status_result = dict(_CONNECTED)
    dialog.show_tab("Capabilities")
    writes: list[str] = []
    real = dialog.cap_status.setText
    dialog.cap_status.setText = lambda t: (writes.append(t), real(t))[1]      # type: ignore[method-assign]
    sheets: list[str] = []
    real_sheet = dialog.cap_status.setStyleSheet
    dialog.cap_status.setStyleSheet = lambda t: (sheets.append(t), real_sheet(t))[1]   # type: ignore[method-assign]
    dialog._refresh_capability_status()
    dialog._poll_capability_status()
    dialog._poll_capability_status()
    assert len(writes) == 1 and len(sheets) <= 1
    actions.status_result = dict(_CONNECTING)
    dialog._poll_capability_status()
    assert len(writes) == 2


def test_every_mcp_tools_event_re_reads_the_status_line(make_dialog, cfg: Config, actions: FakeActions) -> None:
    cfg.capabilities.mcp = [_server("fs")]
    actions.status_sequence = [_CONNECTING, _CONNECTED]
    d = make_dialog(cfg)
    assert "fs connecting" in d.cap_status.text()
    d.cap_mcp_table.setCurrentCell(0, 0)
    d.cap_mcp_test_btn.click()
    actions.calls.clear()
    d.on_event(McpToolsEvent(request_id=1, server="fs", tools=({"name": "echo", "description": ""},)))
    assert d.cap_mcp_table.item(0, 4).text() == "1 tool: echo"
    assert len(actions.calls_named("capability_status")) == 1
    assert "fs connected" in d.cap_status.text()
    # an answer the table ignores (stale / nobody asked) says the server state moved on too
    actions.status_sequence = [_status_with(["read_file"], mcp={"fs": "error: exited (1)"})]
    d.on_event(McpToolsEvent(request_id=99, server="ghost", tools=()))
    assert "fs error: exited (1)" in d.cap_status.text()


def test_apply_shows_the_servers_state_the_engine_reports_right_away(
    dialog: ModelsDialog, actions: FakeActions, monkeypatch: pytest.MonkeyPatch
) -> None:
    actions.status_result = _CONNECTING
    monkeypatch.setattr(dialog, "_prompt_mcp_server", lambda existing: _server("fs"))
    dialog.cap_mcp_add_btn.click()
    dialog.cap_mcp_cb.setChecked(True)
    dialog.cap_apply_btn.click()
    assert "fs connecting" in dialog.cap_status.text()


# ========================================================================= MCP entries read like the runner
_READER_SPECS = [
    {"name": "a", "command": "x"},
    {"name": "b", "url": "https://h/mcp"},
    {"name": "c", "transport": "HTTP", "url": "https://h/mcp"},
    {"name": "d", "transport": " stdio ", "command": "x"},
    {"name": "e", "url": "https://h/mcp", "command": "x"},
    {"name": "f", "transport": "", "url": "https://h/mcp"},
]


def test_kind_and_enabled_are_read_as_the_runner_reads_them() -> None:
    kind_of = getattr(mcp_mod, "server_kind", None)
    for spec in _READER_SPECS:
        if kind_of is not None:
            expected = kind_of(spec)
        else:
            try:
                server = mcp_mod.McpServer(spec, timeout_s=1.0)
            except TypeError:                       # the signature may drop the keyword
                server = mcp_mod.McpServer(spec)
            expected = getattr(server, "kind", None) or getattr(server, "_kind", None)
            if expected is None:
                pytest.skip("the runner exposes no transport reader to compare with")
        assert models_mod._mcp_kind(spec) == expected, spec
    enabled_of = getattr(capabilities_mod, "spec_enabled", None) or getattr(capabilities_mod, "_spec_enabled", None)
    if enabled_of is None:
        pytest.skip("the runner exposes no enabled-reader to compare with")
    for value in (None, True, False, 0, 1, 2, 0.0, 1.0, "false", "true", "no", "", [], {}):
        spec = {"name": "x", "command": "c"} if value is None else {"name": "x", "command": "c", "enabled": value}
        assert models_mod._mcp_enabled(spec) == enabled_of(spec), value


def test_the_table_reads_entries_as_the_runner_does(dialog: ModelsDialog) -> None:
    dialog._mcp_specs = [
        {"name": "web", "url": "https://h/mcp"},                                    # http by inference
        {"name": "up", "transport": "HTTP", "url": "https://h/mcp?token=abc"},       # case does not matter
        {"name": "off", "transport": "stdio", "command": "x", "enabled": "false"},   # fail-closed: disabled
        {"name": "flt", "command": "x", "enabled": 1.0},                             # a float is not a bool / int
        {"name": "one", "command": "x", "enabled": 1},
        {"name": "zero", "command": "x", "enabled": 0},
        {"name": "dflt", "command": "x", "args": ["-y", "p"]},
    ]
    dialog._refresh_mcp_table()
    cells = [_cells(dialog.cap_mcp_table, i) for i in range(len(dialog._mcp_specs))]
    assert cells[0] == ["web", "http", "https://h/mcp", "yes", ""]
    assert cells[1] == ["up", "http", "https://h/mcp?token=***", "yes", ""]
    assert [c[3] for c in cells[2:6]] == ["no", "no", "yes", "no"]
    assert cells[6] == ["dflt", "stdio", "x -y p", "yes", ""]


def test_edit_opens_an_inferred_http_entry_as_http_and_ok_changes_nothing(
    dialog: ModelsDialog
) -> None:
    old = {"name": "web", "url": "https://h/mcp"}
    seen: dict[str, Any] = {}

    def fill(dlg) -> None:
        seen["transport"] = dlg.findChild(QComboBox).currentText()
        seen["url"] = dlg.findChildren(QLineEdit)[2].text()
        _ok(dlg)

    with _drive_modal(fill):
        spec = dialog._prompt_mcp_server(dict(old))
    assert seen == {"transport": "http", "url": "https://h/mcp"}      # not refused as "a stdio server needs a command"
    assert spec == old


def test_edit_keeps_a_mixed_case_transport_and_a_hand_written_enabled_value(dialog: ModelsDialog) -> None:
    old = {"name": "web", "transport": "HTTP", "url": "https://h/mcp", "enabled": "false"}
    seen: dict[str, Any] = {}

    def fill(dlg) -> None:
        seen["transport"] = dlg.findChild(QComboBox).currentText()
        seen["enabled"] = dlg.findChild(QCheckBox).isChecked()
        _ok(dlg)

    with _drive_modal(fill):
        spec = dialog._prompt_mcp_server(dict(old))
    assert seen == {"transport": "http", "enabled": False}            # the runner fail-closed this entry: so does Edit
    assert spec == old                                                # ... and OK does not flip it to True

    def enable(dlg) -> None:
        dlg.findChild(QCheckBox).setChecked(True)
        _ok(dlg)

    with _drive_modal(enable):
        spec = dialog._prompt_mcp_server(dict(old))
    assert spec["enabled"] is True and spec["transport"] == "HTTP"    # a real change is written; the rest is kept


def test_edit_and_ok_on_a_bare_entry_adds_no_keys(dialog: ModelsDialog, cfg: Config) -> None:
    cfg.capabilities.mcp = [{"name": "fs", "transport": "stdio", "command": "npx"}]
    dialog.load()
    dialog.cap_mcp_table.setCurrentCell(0, 0)
    with _drive_modal(_ok):
        dialog.cap_mcp_edit_btn.click()
    assert dialog._mcp_specs == [{"name": "fs", "transport": "stdio", "command": "npx"}]
    assert dialog._capabilities_changes() == {}                        # nothing to apply: nothing was really edited


def test_edit_writes_a_key_the_entry_never_had_once_it_differs_from_the_default(dialog: ModelsDialog) -> None:
    old = {"name": "fs", "command": "npx"}

    def fill(dlg) -> None:
        dlg.findChildren(QPlainTextEdit)[0].setPlainText("-y\npkg")        # arguments
        dlg.findChildren(QPlainTextEdit)[1].setPlainText("K=v")            # environment
        dlg.findChild(QCheckBox).setChecked(False)
        _ok(dlg)

    with _drive_modal(fill):
        spec = dialog._prompt_mcp_server(dict(old))
    assert spec == {"name": "fs", "command": "npx", "args": ["-y", "pkg"], "env": {"K": "v"}, "enabled": False}


def test_edit_switching_the_transport_writes_it_down(dialog: ModelsDialog) -> None:
    old = {"name": "fs", "command": "npx"}                                 # stdio by inference

    def fill(dlg) -> None:
        dlg.findChild(QComboBox).setCurrentText("http")
        dlg.findChildren(QLineEdit)[2].setText("https://h/mcp")
        _ok(dlg)

    with _drive_modal(fill):
        spec = dialog._prompt_mcp_server(dict(old))
    assert spec == {"name": "fs", "transport": "http", "url": "https://h/mcp"}


def test_the_load_question_reads_enabled_like_the_runner(
    make_dialog, cfg: Config, actions: FakeActions, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg.capabilities.mcp = [_server("strict", enabled="false"), _server("float", enabled=1.0), _server("ok")]
    actions.maps_result = [_map_row("m", settings={"mcp_enabled": ["strict", "float", "ok"]})]
    d = make_dialog(cfg)
    d.cap_maps_table.setCurrentCell(0, 0)
    asked: list[str] = []
    monkeypatch.setattr(d, "_confirm", lambda title, text: asked.append(text) or False)
    d.cap_map_load_btn.click()
    assert "MCP server 'strict': on (now off)" in asked[0]
    assert "MCP server 'float': on (now off)" in asked[0]
    assert "'ok'" not in asked[0]


# ================================================================================ the modal driver
def test_a_failing_driver_fails_the_test_instead_of_hanging(dialog: ModelsDialog) -> None:
    def broken(dlg) -> None:
        raise AssertionError("the filler broke")

    with pytest.raises(AssertionError, match="the filler broke"):
        with _drive_modal(broken):
            result = dialog._prompt_mcp_server(None)
    assert not [w for w in QApplication.topLevelWidgets() if w.isVisible() and w.isModal()]


def test_a_driver_that_leaves_the_dialog_open_is_freed_by_the_guard(dialog: ModelsDialog) -> None:
    with pytest.raises(TimeoutError):
        with _drive_modal(lambda dlg: None, stuck_ms=100):
            dialog._prompt_mcp_server(None)


def test_a_nested_message_box_does_not_hang_the_driver(dialog: ModelsDialog) -> None:
    """The real, unpatched QMessageBox.warning blocks inside the filler (a validation regression)."""
    def fill(dlg) -> None:
        dlg.findChildren(QLineEdit)[0].setText("bad name!")
        _ok(dlg)                                           # opens the warning and blocks here

    with pytest.raises(TimeoutError):
        with _drive_modal(fill, stuck_ms=100):
            dialog._prompt_mcp_server(None)


def test_the_driver_leaves_a_dialog_that_accepted_alone(dialog: ModelsDialog) -> None:
    """Rejecting after a successful accept() would flip exec() to Rejected: only a failure rejects."""
    def fill(dlg) -> None:
        dlg.findChildren(QLineEdit)[0].setText("fs")
        dlg.findChildren(QLineEdit)[1].setText("npx")
        _ok(dlg)

    with _drive_modal(fill):
        spec = dialog._prompt_mcp_server(None)
    assert spec is not None and spec["name"] == "fs"
