"""Inference phase for the SN5 Hone solver."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from arc_solver import ARCSolver
from arc_utils import load_input_data, save_output_data


def run_inference_phase(input_dir: Path, output_dir: Path) -> None:
    data = load_input_data(input_dir)
    tasks: list[dict[str, Any]] = data.get("tasks", [])
    solver = ARCSolver(use_vllm=True)

    predictions = []
    for problem_index, task in enumerate(tasks):
        prediction = solver.solve(
            train_examples=task.get("train_examples", []),
            test_input=task.get("test_input", []),
        )
        predictions.append(
            {
                "problem_index": problem_index,
                "task_hash": task.get("task_hash"),
                "predicted_output": prediction,
                "metadata": task.get("metadata", {}),
            }
        )

    save_output_data(
        {
            "phase": "inference",
            "status": "success",
            "predictions": predictions,
            "num_problems_solved": len(predictions),
        },
        output_dir,
    )
