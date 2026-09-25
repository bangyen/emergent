"""Tests for training internals: REINFORCE loss, scene stream, heldout parsing."""

import csv
from itertools import islice
from pathlib import Path

import pytest
import torch
from click.testing import CliRunner

from langlab.apps.cli import main, parse_heldout
from langlab.data.data import SceneStream, heldout_objects
from langlab.training.train import compute_speaker_loss, train


def test_speaker_loss_uses_sent_tokens() -> None:
    # Argmax of logits is token 0, but the sent token is 1: the gradient must
    # push probability toward the rewarded token that was actually sent.
    logits = torch.tensor([[[2.0, 0.0, 0.0]]], requires_grad=True)
    tokens = torch.tensor([[1]])
    loss = compute_speaker_loss(logits, tokens, torch.tensor([1.0]), 0.0, 0.0)
    loss.backward()
    assert logits.grad is not None
    assert logits.grad[0, 0, 1] < 0  # descending increases logit of token 1
    assert logits.grad[0, 0, 0] > 0


def test_heldout_objects() -> None:
    keys = heldout_objects([("red", "circle")])
    assert keys == {("circle", "red", "small"), ("circle", "large", "red")}
    with pytest.raises(ValueError):
        heldout_objects([("red", "hexagon")])


def test_scene_stream_excludes_heldout_and_is_seeded() -> None:
    red_circle = {0, 3}  # one-hot indices for red, circle
    stream = SceneStream(k=5, seed=3, heldout_pairs=[("red", "circle")])
    scenes = list(islice(stream, 200))
    for scene, target in scenes:
        assert scene.shape == (5, 8)
        assert 0 <= target < 5
        for obj in scene:
            assert set(obj.nonzero().flatten().tolist()) & red_circle != red_circle

    again = list(
        islice(SceneStream(k=5, seed=3, heldout_pairs=[("red", "circle")]), 200)
    )
    assert all(torch.equal(a[0], b[0]) and a[1] == b[1] for a, b in zip(scenes, again))


def test_parse_heldout() -> None:
    assert parse_heldout(None) is None
    assert parse_heldout("red, circle,blue,square") == [
        ("red", "circle"),
        ("blue", "square"),
    ]


def test_train_with_heldout_writes_metrics(tmp_path: Path) -> None:
    metrics = train(
        n_steps=6,
        k=3,
        v=8,
        message_length=1,
        batch_size=4,
        hidden_size=16,
        heldout_pairs=[("red", "circle")],
        out_dir=str(tmp_path),
        eval_every=3,
        n_eval=20,
        log_every=2,
    )
    assert {"iid_acc", "compo_acc", "compo_target_acc", "topsim"} <= set(metrics)
    with open(tmp_path / "metrics.csv") as f:
        rows = list(csv.DictReader(f))
    assert [r["step"] for r in rows] == ["2", "3", "4", "6"]
    assert rows[1]["compo_acc"] != "" and rows[0]["compo_acc"] == ""

    ckpt = torch.load(tmp_path / "checkpoints" / "final_model.pt", weights_only=False)
    assert ckpt["heldout_pairs"] == [("red", "circle")]
    assert ckpt["k"] == 3


def test_cli_train_eval_plot(tmp_path: Path) -> None:
    runner = CliRunner()
    out = str(tmp_path)
    result = runner.invoke(
        main,
        [
            "train",
            "--steps",
            "4",
            "--batch-size",
            "4",
            "--k",
            "3",
            "--heldout",
            "red,circle",
            "--eval-every",
            "2",
            "--out-dir",
            out,
        ],
    )
    assert result.exit_code == 0, result.output
    assert "compo_acc" in result.output

    ckpt = str(tmp_path / "checkpoints" / "final_model.pt")
    result = runner.invoke(main, ["eval", "--ckpt", ckpt, "--split", "compo"])
    assert result.exit_code == 0, result.output

    png = str(tmp_path / "curve.png")
    result = runner.invoke(
        main, ["plot", "--metrics", str(tmp_path / "metrics.csv"), "--out", png]
    )
    assert result.exit_code == 0, result.output
    assert Path(png).exists()

    result = runner.invoke(main, ["train", "--steps", "1", "--heldout", "red"])
    assert result.exit_code != 0


def test_train_large_world_logs_language_metrics(tmp_path: Path) -> None:
    metrics = train(
        n_steps=4,
        k=6,
        v=8,
        message_length=2,
        batch_size=4,
        hidden_size=16,
        heldout_pairs=[("yellow", "star")],
        world="large",
        out_dir=str(tmp_path),
        eval_every=4,
        n_eval=20,
    )
    assert {"compo_target_acc", "topsim", "posdis", "n_messages"} <= set(metrics)
    ckpt = torch.load(tmp_path / "checkpoints" / "final_model.pt", weights_only=False)
    assert ckpt["world"] == "large" and ckpt["config"].object_dim == 16


def test_eval_splits_and_pragmatic(tmp_path: Path) -> None:
    from langlab.analysis.eval import evaluate

    train(
        n_steps=2,
        k=4,
        v=8,
        message_length=2,
        batch_size=4,
        hidden_size=16,
        heldout_pairs=[("red", "circle")],
        out_dir=str(tmp_path),
        n_eval=10,
    )
    ckpt = str(tmp_path / "checkpoints" / "final_model.pt")
    for split in ["train", "iid", "compo", "compo_target", "distractor"]:
        res = evaluate(ckpt, split=split, n_scenes=40)
        assert 0.0 <= res["acc"] <= 1.0 and "topsim" in res
    res = evaluate(ckpt, split="distractor", n_scenes=40, pragmatic=True)
    assert 0.0 <= res["acc"] <= 1.0
    with pytest.raises(ValueError):
        evaluate(ckpt, split="bogus")
