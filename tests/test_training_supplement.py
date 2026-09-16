import math
from pathlib import Path

import pytest

from tools.generate_nnunet_training_supplement import moving_average, parse_training_log


def test_parse_training_log_reads_nnunet_epoch_metrics(tmp_path: Path) -> None:
    log = tmp_path / "training_log.txt"
    log.write_text(
        "2026-01-01: Epoch 12\n"
        "2026-01-01: Current learning rate: 0.007\n"
        "2026-01-01: train_loss -0.75\n"
        "2026-01-01: val_loss -0.72\n"
        "2026-01-01: Pseudo dice [0.8, nan, 0.9]\n"
        "2026-01-01: Epoch time: 10.5 s\n",
        encoding="utf-8",
    )
    rows = parse_training_log(log)
    assert len(rows) == 1
    assert rows[0]["epoch"] == 12
    assert rows[0]["learning_rate"] == 0.007
    assert rows[0]["train_loss"] == -0.75
    assert rows[0]["val_loss"] == -0.72
    assert rows[0]["mean_pseudo_dice"] == pytest.approx(0.85)
    assert math.isnan(rows[0]["pseudo_dice"][1])
    assert rows[0]["epoch_time_seconds"] == 10.5


def test_moving_average_is_centered_and_handles_edges() -> None:
    assert moving_average([1.0, 2.0, 3.0, 4.0], 3) == [1.5, 2.0, 3.0, 3.5]
