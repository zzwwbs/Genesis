"""The extension registry is swapped in whole, never truncated in place.

Its reader treats unparseable JSON as "no extensions" and returns silently, so a
write interrupted after truncation dropped every registered extension on the
next start, with nothing to say so.
"""

from __future__ import annotations

import os
from pathlib import Path

from genesis.service import GenesisService


def test_the_registry_is_replaced_atomically(tmp_path: Path, monkeypatch) -> None:
    service = GenesisService(tmp_path / "ws")
    try:
        store = service._extension_store
        store.write_text('[{"entry_point": "previous:entry"}]\n')
        seen: list[tuple[str, str]] = []
        real_replace = os.replace

        def spying_replace(source, destination):
            # At the moment of the swap the live file must still be intact.
            seen.append((Path(destination).read_text(), Path(source).name))
            return real_replace(source, destination)

        monkeypatch.setattr("genesis.service.os.replace", spying_replace)
        service._extensions = type("R", (), {"manifests": lambda self: []})()
        service._save_extensions()
        assert seen, "the registry was written in place rather than swapped in"
        assert "previous:entry" in seen[0][0]
        assert store.read_text().strip() == "[]"
        assert not any(p.name.endswith(".tmp") for p in store.parent.iterdir())
    finally:
        service.close()
