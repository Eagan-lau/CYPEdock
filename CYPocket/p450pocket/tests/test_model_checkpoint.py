from pathlib import Path

import numpy as np
import pytest
import torch

from p450pocket.checkpoint import load_model, save_training_checkpoint
from p450pocket.manifest import SampleRecord
from p450pocket.model import ModelConfig, ResidueBertClassifier
from p450pocket.training import TrainingConfig, run_training


def tiny_config() -> ModelConfig:
    return ModelConfig(
        feature_dim=100,
        num_labels=2,
        hidden_size=24,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=48,
        max_position_embeddings=512,
        chunk_size=512,
        architecture="tiny-test-classifier",
    )


@pytest.mark.parametrize("length", [511, 512, 513])
def test_chunk_boundaries_match_explicit_windows(length: int) -> None:
    torch.manual_seed(7)
    model = ResidueBertClassifier(tiny_config()).eval()
    features = torch.randn(1, length, 100)
    mask = torch.ones(1, length, dtype=torch.long)
    with torch.inference_mode():
        actual = model.forward_chunked(features, mask)
        expected = torch.cat(
            [
                model(features[:, start : start + 512], mask[:, start : start + 512])
                for start in range(0, length, 512)
            ],
            dim=1,
        )
    assert actual.shape == (1, length, 2)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_structured_checkpoint_is_safe_and_strict(tmp_path: Path) -> None:
    torch.manual_seed(11)
    model = ResidueBertClassifier(tiny_config())
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
    path = tmp_path / "checkpoint.pth"
    save_training_checkpoint(
        path,
        model,
        optimizer,
        scheduler=None,
        epoch=2,
        global_step=9,
        metrics={"pr_auc": 0.75, "loss": 0.25},
        training_config={"learning_rate": 1e-5},
        threshold=0.4,
    )
    restored, metadata = load_model(path, strict=True)
    assert restored.model_config == tiny_config()
    assert metadata["epoch"] == 2
    assert metadata["threshold"] == pytest.approx(0.4)
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, restored.state_dict()[name], rtol=0, atol=0)


def test_one_epoch_training_smoke(tmp_path: Path) -> None:
    torch.manual_seed(13)
    initial_model = ResidueBertClassifier(tiny_config())
    initialization = tmp_path / "initial.pth"
    save_training_checkpoint(
        initialization,
        initial_model,
        optimizer=None,
        scheduler=None,
        epoch=0,
        global_step=0,
        metrics={},
        training_config={},
    )

    records = []
    for sample_id, split, offset in (
        ("train-a", "train", 0),
        ("train-b", "train", 1),
        ("val-a", "val", 2),
    ):
        features = tmp_path / f"{sample_id}_features.npy"
        labels = tmp_path / f"{sample_id}_labels.npy"
        rng = np.random.RandomState(100 + offset)
        np.save(features, rng.normal(size=(4, 100)).astype(np.float32))
        np.save(labels, np.array([0, 1, 0, 1], dtype=np.int64))
        records.append(
            SampleRecord(sample_id, split, str(features), str(labels))
        )

    result = run_training(
        [record for record in records if record.split == "train"],
        [record for record in records if record.split == "val"],
        tmp_path / "run",
        TrainingConfig(
            epochs=1,
            batch_size=1,
            learning_rate=1e-4,
            monitor="pr_auc",
            patience=1,
            seed=17,
        ),
        device="cpu",
        initialization_checkpoint=initialization,
    )
    assert result["epochs_completed"] == 1
    assert Path(result["best_checkpoint"]).is_file()
    restored, metadata = load_model(result["best_checkpoint"], strict=True)
    assert restored.model_config == tiny_config()
    assert metadata["epoch"] == 0
