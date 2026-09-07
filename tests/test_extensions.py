import hashlib

import pytest

from genesis.extensions import ExtensionManifest, ExtensionRegistry


def test_extension_manifest_requires_compatible_integrity_and_explicit_enablement():
    material = b"trusted"
    manifest = ExtensionManifest(
        id="demo-extension",
        version="1.0.0",
        genesis_range=">=0.1,<0.2",
        schema_range=">=1.0,<2.0",
        capabilities=("executor",),
        entry_point="demo:factory",
        integrity_hash=hashlib.sha256(material).hexdigest(),
    )
    registry = ExtensionRegistry(genesis_version="0.1.0", schema_version="1.0")
    with pytest.raises(PermissionError):
        registry.register(manifest, lambda: "trusted")
    registry.register(manifest, lambda: "trusted", enabled=True, integrity_material=material)
    assert registry.get("demo-extension")() == "trusted"


def test_extension_registry_rejects_incompatible_or_duplicate_manifests():
    registry = ExtensionRegistry(genesis_version="0.1.0", schema_version="1.0")
    bad = ExtensionManifest(
        id="bad-extension",
        version="1.0.0",
        genesis_range=">=0.2",
        schema_range=">=1.0",
        capabilities=(),
        entry_point="bad:factory",
        integrity_hash="hash",
    )
    with pytest.raises(ValueError, match="compatible"):
        registry.register(bad, lambda: None, enabled=True)
