import importlib
import importlib.metadata

import dwi2cond_xp


def test_missing_distribution_metadata_uses_unknown_version(monkeypatch) -> None:
    def missing_version(name: str) -> str:
        raise importlib.metadata.PackageNotFoundError(name)

    with monkeypatch.context() as context:
        context.setattr(importlib.metadata, "version", missing_version)
        reloaded = importlib.reload(dwi2cond_xp)
        assert reloaded.__version__ == "0+unknown"
    importlib.reload(dwi2cond_xp)
