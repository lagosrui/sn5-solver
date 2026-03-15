"""Prep phase for SN5 Hone solver."""

from __future__ import annotations

import json
from pathlib import Path


def run_prep_phase(input_dir: Path, output_dir: Path) -> None:
    del input_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "phase": "prep",
        "status": "success",
        "message": "no-op prep completed",
    }
    (output_dir / "prep_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=True)
    )
