from pathlib import Path

from meposer.config import load_config, parse_value

ROOT = Path(__file__).resolve().parents[1]


def test_parse_value_types():
    assert parse_value("1e-4") == 1e-4 and isinstance(parse_value("1e-4"), float)
    assert parse_value("32") == 32 and isinstance(parse_value("32"), int)
    assert parse_value("[imu, image]") == ["imu", "image"]
    assert parse_value("true") is True and parse_value("null") is None
    assert parse_value("runs/x/features") == "runs/x/features"


def test_overrides_and_inheritance():
    cfg = load_config(ROOT / "configs/cv_only.yaml", ["train.lr=3e-4", "model.temporal.blocks=1", "data.val_subjects=[\"0001\"]"])
    assert cfg.train.lr == 3e-4 and cfg.model.temporal.blocks == 1 and cfg.data.val_subjects == ["0001"]
    assert cfg.model.modalities == ["image"] and cfg.train.milestones == [10, 16]
    assert cfg.data.window == 32
