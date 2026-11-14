"""Isolate template-library writes from the user's installed packages."""

import pytest

from character_dna.editors.converter import catalog, packages


@pytest.fixture(autouse=True)
def template_library(tmp_path, monkeypatch, addon):
    with monkeypatch.context() as patch:
        patch.setattr(packages, "get_user_data_folder", lambda: tmp_path / "user-data")
        catalog.refresh()
        yield packages.library_root()
