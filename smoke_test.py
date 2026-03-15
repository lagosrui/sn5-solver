"""Small local smoke test for the SN5 Hone solver package."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from arc_inference_phase import run_inference_phase


def main() -> int:
    dataset = {
        "tasks": [
            {
                "task_hash": "identity-demo",
                "train_examples": [
                    {
                        "input": [[1, 2], [3, 4]],
                        "output": [[1, 2], [3, 4]],
                    }
                ],
                "test_input": [[5, 6], [7, 8]],
                "metadata": {},
            },
            {
                "task_hash": "flip-h-demo",
                "train_examples": [
                    {
                        "input": [[1, 2, 3]],
                        "output": [[3, 2, 1]],
                    }
                ],
                "test_input": [[4, 5, 6]],
                "metadata": {},
            },
        ]
    }

    with tempfile.TemporaryDirectory() as temp_dir:
        base = Path(temp_dir)
        input_dir = base / "input"
        output_dir = base / "output"
        input_dir.mkdir(parents=True, exist_ok=True)
        (input_dir / "miner_current_dataset.json").write_text(
            json.dumps(dataset, indent=2)
        )

        run_inference_phase(input_dir, output_dir)
        payload = json.loads((output_dir / "results.json").read_text())
        assert payload["status"] == "success"
        assert payload["predictions"][0]["predicted_output"] == [[5, 6], [7, 8]]
        assert payload["predictions"][1]["predicted_output"] == [[6, 5, 4]]
        print("smoke test passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
