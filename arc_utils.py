"""Shared ARC dataset helpers for the Hone sandbox contract."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_input_data(input_dir: Path) -> dict[str, Any]:
    dataset_file = input_dir / "miner_current_dataset.json"
    if not dataset_file.exists():
        raise FileNotFoundError(f"missing dataset file: {dataset_file}")
    return json.loads(dataset_file.read_text())


def save_output_data(results: dict[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "results.json"
    output_file.write_text(json.dumps(results, indent=2, ensure_ascii=True))
