"""Tests for population training and sweeps."""

import csv
import json
from pathlib import Path

import torch
from click.testing import CliRunner

from langlab.apps.cli import main
from langlab.training.population import train_population
from langlab.training.sweep import expand, run_sweep


def test_population_with_turnover(tmp_path: Path) -> None:
    metrics = train_population(
        n_steps=8,
        k=3,
        v=8,
        message_length=2,
        n_agents=2,
        lifespan=3,
        batch_size=4,
        hidden_size=16,
        heldout_pairs=[("red", "circle")],
        out_dir=str(tmp_path),
        eval_every=4,
        n_eval=20,
    )
    for key in ("iid_acc", "iid_acc_min", "compo_target_acc", "agreement", "topsim"):
        assert key in metrics
    assert metrics["iid_acc_min"] <= metrics["iid_acc"]
    with open(tmp_path / "metrics.csv") as f:
        rows = list(csv.DictReader(f))
    assert rows[-1]["generation"] == "2"
    ckpt = torch.load(tmp_path / "population.pt", weights_only=False)
    assert len(ckpt["speaker_state_dicts"]) == 2


def test_expand_sweep() -> None:
    specs = expand(
        {
            "base": {"n_steps": 5, "heldout_pairs": [["red", "circle"]]},
            "variants": [{"name": "a"}, {"name": "b", "runner": "population"}],
            "grid": {"v": [4, 8]},
            "seeds": [1, 2],
        }
    )
    assert len(specs) == 8
    assert {s["run"] for s in specs} == {"a_v=4", "a_v=8", "b_v=4", "b_v=8"}
    assert specs[0]["params"]["heldout_pairs"] == [("red", "circle")]
    assert {s["runner"] for s in specs} == {"train", "population"}


def test_run_sweep_resumes(tmp_path: Path) -> None:
    cfg = {
        "base": {
            "n_steps": 3,
            "k": 3,
            "v": 4,
            "message_length": 1,
            "batch_size": 4,
            "hidden_size": 16,
            "n_eval": 10,
        },
        "variants": [{"name": "tiny"}],
        "seeds": [1, 2],
    }
    md = run_sweep(cfg, str(tmp_path))
    assert "| tiny | 2 |" in md
    result = tmp_path / "tiny" / "seed1" / "results.json"
    data = json.loads(result.read_text())
    data["metrics"]["iid_acc"] = 0.123
    result.write_text(json.dumps(data))
    run_sweep(cfg, str(tmp_path))  # existing results are reused, not rerun
    assert json.loads(result.read_text())["metrics"]["iid_acc"] == 0.123


def test_cli_pop_train_sweep_report(tmp_path: Path) -> None:
    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "pop-train",
            "--steps",
            "3",
            "--k",
            "3",
            "--agents",
            "2",
            "--batch-size",
            "4",
            "--out-dir",
            str(tmp_path / "pop"),
        ],
    )
    assert result.exit_code == 0, result.output
    assert "agreement" in result.output

    cfg = tmp_path / "s.json"
    cfg.write_text(
        json.dumps(
            {
                "base": {"n_steps": 2, "k": 3, "batch_size": 4, "n_eval": 10},
                "variants": [{"name": "x"}],
                "seeds": [1],
            }
        )
    )
    out = tmp_path / "sweep"
    result = runner.invoke(main, ["sweep", str(cfg), "--out-dir", str(out)])
    assert result.exit_code == 0, result.output
    result = runner.invoke(main, ["report", str(out)])
    assert result.exit_code == 0 and "| x | 1 |" in result.output


def test_shared_listener_population(tmp_path: Path) -> None:
    metrics = train_population(
        n_steps=6,
        k=3,
        v=8,
        message_length=2,
        n_agents=3,
        n_listeners=1,
        lifespan=2,
        batch_size=4,
        hidden_size=16,
        out_dir=str(tmp_path),
        eval_every=2,
        n_eval=20,
    )
    assert "topsim_initial" in metrics and "agreement" in metrics
    with open(tmp_path / "metrics.csv") as f:
        rows = list(csv.DictReader(f))
    # evaluation at a replacement step measures the generation that just ended
    assert [(r["step"], r["generation"]) for r in rows] == [
        ("2", "0"),
        ("4", "1"),
        ("6", "2"),
    ]
    ckpt = torch.load(tmp_path / "population.pt", weights_only=False)
    assert len(ckpt["speaker_state_dicts"]) == 3
    assert len(ckpt["listener_state_dicts"]) == 1
