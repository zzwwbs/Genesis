from pathlib import Path

import pytest

from genesis.service import GenesisService


def test_retention_redacts_nested_provider_bodies_not_declared_values() -> None:
    payload = {
        "raw_response": "private",
        "parsed_response": {"secret": "private"},
        "provider_attempts": [{"raw_response": "private", "request_id": "request"}],
        "value": {"strategy": "declared scientific output"},
    }
    redacted = GenesisService._redact_raw_responses(payload)
    assert "private" not in str(redacted)
    assert redacted["provider_attempts"][0]["request_id"] == "request"
    assert redacted["value"] == payload["value"]
    assert payload["raw_response"] == "private"


def test_rejected_edit_preserves_entire_package(tmp_path: Path) -> None:
    service = GenesisService(tmp_path)
    try:
        service.create_specification({"id": "review", "title": "Before"})
        directory = service._specification_dir("review")
        before = {
            p.relative_to(directory): p.read_bytes() for p in directory.rglob("*") if p.is_file()
        }
        with pytest.raises(ValueError):
            service.update_specification(
                "review", {"title": "After", "prompts": {"bad/id": "invalid"}}, 1
            )
        assert before == {
            p.relative_to(directory): p.read_bytes() for p in directory.rglob("*") if p.is_file()
        }
    finally:
        service.close()


def test_removed_prompts_are_materialized(tmp_path: Path) -> None:
    service = GenesisService(tmp_path)
    try:
        service.create_specification({"id": "review", "prompts": {"old": "original"}})
        service.update_specification("review", {"prompts": {}}, 1)
        assert not (service._specification_dir("review") / "prompts/old.txt").exists()
    finally:
        service.close()


def test_failed_file_write_restores_package(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    service = GenesisService(tmp_path)
    try:
        service.create_specification({"id": "review", "title": "Before"})
        directory = service._specification_dir("review")
        before = {
            p.relative_to(directory): p.read_bytes() for p in directory.rglob("*") if p.is_file()
        }
        original = service._write_yaml
        count = 0

        def fail_second(path: Path, value: dict) -> None:
            nonlocal count
            count += 1
            if count == 2:
                raise OSError("injected write failure")
            original(path, value)

        monkeypatch.setattr(service, "_write_yaml", fail_second)
        with pytest.raises(OSError):
            service.update_specification("review", {"title": "After"}, 1)
        assert before == {
            p.relative_to(directory): p.read_bytes() for p in directory.rglob("*") if p.is_file()
        }
    finally:
        service.close()
