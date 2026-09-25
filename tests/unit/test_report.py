"""Tests for sweep result aggregation."""

import json
from pathlib import Path

import pytest

from langlab.analysis.report import (
    README_END,
    README_START,
    aggregate,
    create_report,
    to_markdown,
    update_readme,
)


def _write(root: Path, run: str, seed: int, metrics: dict) -> None:
    d = root / run / f"seed{seed}"
    d.mkdir(parents=True)
    (d / "results.json").write_text(
        json.dumps({"run": run, "seed": seed, "params": {}, "metrics": metrics})
    )


def test_aggregate_mean_std() -> None:
    rows = aggregate(
        [
            {"run": "a", "seed": 1, "metrics": {"iid_acc": 0.8, "topsim": 0.5}},
            {"run": "a", "seed": 2, "metrics": {"iid_acc": 1.0, "topsim": 0.7}},
            {"run": "b", "seed": 1, "metrics": {"iid_acc": 0.5}},
        ]
    )
    assert [r["run"] for r in rows] == ["a", "b"]
    assert rows[0]["n_seeds"] == 2
    assert rows[0]["iid_acc_mean"] == pytest.approx(0.9)
    assert rows[0]["topsim_std"] == pytest.approx(0.1414, abs=1e-3)
    assert rows[1]["iid_acc_std"] == 0.0

    md = to_markdown(rows)
    assert "| a | 2 | 90.0 ± 14.1% | 0.60 ± 0.14 |" in md
    assert "| b | 1 | 50.0 ± 0.0% | – |" in md


def test_create_report_and_readme(tmp_path: Path) -> None:
    _write(tmp_path, "mlp", 1, {"iid_acc": 1.0})
    _write(tmp_path, "mlp", 2, {"iid_acc": 0.5})
    readme = tmp_path / "README.md"
    readme.write_text(f"intro\n{README_START}\nold\n{README_END}\noutro\n")

    md = create_report(str(tmp_path), readme=str(readme))
    assert "75.0" in md
    assert (tmp_path / "summary.csv").exists()
    text = readme.read_text()
    assert "old" not in text and md in text and text.endswith("outro\n")


def test_update_readme_requires_markers(tmp_path: Path) -> None:
    readme = tmp_path / "README.md"
    readme.write_text("no markers")
    with pytest.raises(ValueError):
        update_readme(str(readme), "table")


def test_create_report_empty(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        create_report(str(tmp_path))
