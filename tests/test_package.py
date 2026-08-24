import importlib
import importlib.metadata

import dwi2cond_xp
import pytest


def test_missing_distribution_metadata_uses_unknown_version(monkeypatch) -> None:
    def missing_version(name: str) -> str:
        raise importlib.metadata.PackageNotFoundError(name)

    with monkeypatch.context() as context:
        context.setattr(importlib.metadata, "version", missing_version)
        reloaded = importlib.reload(dwi2cond_xp)
        assert reloaded.__version__ == "0+unknown"
    importlib.reload(dwi2cond_xp)


def test_root_exports_are_loaded_only_when_requested() -> None:
    reloaded = importlib.reload(dwi2cond_xp)
    assert "load_gradients" not in reloaded.__dict__

    from dwi2cond_xp.gradients import load_gradients

    assert "load_gradients" in dir(reloaded)
    assert reloaded.load_gradients is load_gradients
    assert reloaded.__dict__["load_gradients"] is load_gradients
    with pytest.raises(AttributeError, match="missing"):
        getattr(reloaded, "missing")


def test_version_lookup_is_cached(monkeypatch) -> None:
    with monkeypatch.context() as context:
        context.setattr(importlib.metadata, "version", lambda _name: "9.8.7")
        reloaded = importlib.reload(dwi2cond_xp)
        assert reloaded.__version__ == "9.8.7"
        assert reloaded.__dict__["__version__"] == "9.8.7"
    importlib.reload(dwi2cond_xp)
