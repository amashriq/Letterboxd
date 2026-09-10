# Unit tests for engine.paths.resolve_user_dir's short-name -> most-recent-
# dated-folder resolution and exact-folder-name override.

import pytest

import engine.paths as paths


def test_exact_folder_name_used_as_is(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "DATA_DIR", tmp_path)
    (tmp_path / "alice-2026-01-01").mkdir()
    assert paths.resolve_user_dir("alice-2026-01-01") == tmp_path / "alice-2026-01-01"


def test_short_name_resolves_to_most_recent_dated_folder(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "DATA_DIR", tmp_path)
    (tmp_path / "alice-2026-01-01").mkdir()
    (tmp_path / "alice-2026-06-15").mkdir()
    (tmp_path / "alice-2026-03-10").mkdir()
    assert paths.resolve_user_dir("alice") == tmp_path / "alice-2026-06-15"


def test_missing_user_raises_systemexit(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "DATA_DIR", tmp_path)
    with pytest.raises(SystemExit):
        paths.resolve_user_dir("nobody")
