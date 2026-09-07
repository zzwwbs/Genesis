from pathlib import Path

from tools.check_traceability import check_traceability


def test_traceability_reports_normative_ids_without_evidence(tmp_path: Path) -> None:
    specification = tmp_path / "spec.md"
    matrix = tmp_path / "matrix.yaml"
    specification.write_text("## Requirements\nACC-001 authoring\nACC-002 replay\n")
    matrix.write_text("requirements:\n  ACC-001:\n    tests: [tests/test_smoke.py]\n")

    errors = check_traceability(specification, matrix)

    assert errors == ["ACC-002: missing traceability entry"]


def test_traceability_accepts_standard_yaml_indentation_and_lists(tmp_path: Path) -> None:
    specification = tmp_path / "spec.md"
    matrix = tmp_path / "matrix.yaml"
    evidence = tmp_path / "tests" / "test_behavior.py"
    evidence.parent.mkdir()
    evidence.write_text("# evidence\n")
    specification.write_text("ACC-001\n")
    matrix.write_text(
        "requirements:\n    ACC-001:\n      tests:\n        - tests/test_behavior.py\n"
    )

    assert check_traceability(specification, matrix, repository_root=tmp_path) == []


def test_traceability_reports_missing_evidence_path(tmp_path: Path) -> None:
    specification = tmp_path / "spec.md"
    matrix = tmp_path / "matrix.yaml"
    specification.write_text("ACC-001\n")
    matrix.write_text("requirements:\n  ACC-001:\n    tests: [tests/does_not_exist.py]\n")

    assert check_traceability(specification, matrix, repository_root=tmp_path) == [
        "ACC-001: evidence path does not exist: tests/does_not_exist.py"
    ]
