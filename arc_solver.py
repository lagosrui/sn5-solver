"""ARC solver with program search, object-centric transforms, and optional vLLM fallback."""

from __future__ import annotations

import itertools
import json
import os
from collections import Counter, deque
from dataclasses import dataclass
from typing import Any, Callable


Grid = list[list[int]]
Example = dict[str, Any]
Transform = Callable[[Grid], Grid]


@dataclass(frozen=True)
class TransformSpec:
    name: str
    fn: Transform


class ARCSolver:
    def __init__(self, use_vllm: bool = True):
        self.use_vllm = use_vllm
        self.vllm_client = None
        self.vllm_model_name = None
        if use_vllm:
            self._init_vllm()

    def solve(self, train_examples: list[Example], test_input: Grid) -> Grid:
        program_solution = self._solve_via_program_search(train_examples, test_input)
        if program_solution is not None:
            return program_solution

        translation_solution = self._solve_via_translation(train_examples, test_input)
        if translation_solution is not None:
            return translation_solution

        analogy_solution = self._solve_via_example_analogy(train_examples, test_input)
        if analogy_solution is not None:
            return analogy_solution

        if self.vllm_client and self.vllm_model_name:
            predicted = self._solve_with_vllm(train_examples, test_input)
            if self._is_valid_grid(predicted):
                return predicted

        return self._majority_color_fill_guess(train_examples, test_input)

    def _init_vllm(self) -> None:
        try:
            from openai import OpenAI

            base_url = os.environ.get("VLLM_API_BASE", "http://vllm-container:8000")
            client = OpenAI(base_url=f"{base_url}/v1", api_key="dummy")
            models = client.models.list()
            if not models.data:
                return
            self.vllm_client = client
            self.vllm_model_name = models.data[0].id
        except Exception:
            self.vllm_client = None
            self.vllm_model_name = None

    def _solve_via_program_search(
        self, train_examples: list[Example], test_input: Grid
    ) -> Grid | None:
        transforms = self._candidate_transforms()

        for depth in (1, 2):
            for sequence in itertools.product(transforms, repeat=depth):
                exact_candidate = self._test_sequence(sequence, train_examples)
                if exact_candidate is not None:
                    return exact_candidate(test_input)

        for depth in (1, 2):
            for sequence in itertools.product(transforms, repeat=depth):
                recolor_candidate = self._test_sequence_with_color_map(
                    sequence, train_examples, test_input
                )
                if recolor_candidate is not None:
                    return recolor_candidate(test_input)
        return None

    def _test_sequence(
        self, sequence: tuple[TransformSpec, ...], train_examples: list[Example]
    ) -> Transform | None:
        for example in train_examples:
            transformed = self._apply_sequence(example["input"], sequence)
            if transformed != example["output"]:
                return None
        return lambda grid: self._apply_sequence(grid, sequence)

    def _test_sequence_with_color_map(
        self,
        sequence: tuple[TransformSpec, ...],
        train_examples: list[Example],
        test_input: Grid,
    ) -> Transform | None:
        merged_map: dict[int, int] = {}
        for example in train_examples:
            transformed = self._apply_sequence(example["input"], sequence)
            color_map = self._learn_color_map(transformed, example["output"])
            if color_map is None:
                return None
            for key, value in color_map.items():
                existing = merged_map.get(key)
                if existing is not None and existing != value:
                    return None
                merged_map[key] = value
        transformed_test = self._apply_sequence(test_input, sequence)
        transformed_test_colors = {value for row in transformed_test for value in row}
        if not transformed_test_colors.issubset(set(merged_map)):
            return None

        def program(grid: Grid) -> Grid:
            transformed = self._apply_sequence(grid, sequence)
            return self._apply_color_map(transformed, merged_map)

        return program

    def _solve_via_translation(
        self, train_examples: list[Example], test_input: Grid
    ) -> Grid | None:
        delta: tuple[int, int] | None = None
        for example in train_examples:
            inferred = self._infer_translation(example["input"], example["output"])
            if inferred is None:
                return None
            if delta is None:
                delta = inferred
            elif delta != inferred:
                return None
        if delta is None:
            return None
        shifted = self._shift_grid(test_input, delta[0], delta[1])
        return shifted if shifted is not None else None

    def _solve_via_example_analogy(
        self, train_examples: list[Example], test_input: Grid
    ) -> Grid | None:
        transforms = self._candidate_transforms()
        for example in train_examples:
            for depth in (1, 2):
                for sequence in itertools.product(transforms, repeat=depth):
                    if self._apply_sequence(example["input"], sequence) == test_input:
                        return self._apply_sequence(example["output"], sequence)

            delta = self._infer_translation(example["input"], test_input)
            if delta is not None:
                shifted = self._shift_grid(example["output"], delta[0], delta[1])
                if shifted is not None:
                    return shifted
        return None

    def _infer_translation(self, input_grid: Grid, output_grid: Grid) -> tuple[int, int] | None:
        if self._shape(input_grid) != self._shape(output_grid):
            return None
        input_points = self._non_background_points(input_grid)
        output_points = self._non_background_points(output_grid)
        if not input_points or len(input_points) != len(output_points):
            return None

        ordered_input = sorted(input_points)
        ordered_output = sorted(output_points)
        dx = ordered_output[0][0] - ordered_input[0][0]
        dy = ordered_output[0][1] - ordered_input[0][1]

        for (in_r, in_c, in_v), (out_r, out_c, out_v) in zip(ordered_input, ordered_output):
            if out_v != in_v:
                return None
            if out_r - in_r != dx or out_c - in_c != dy:
                return None
        return (dx, dy)

    def _solve_with_vllm(
        self, train_examples: list[Example], test_input: Grid
    ) -> Grid | None:
        prompt = self._build_prompt(train_examples, test_input)
        try:
            response = self.vllm_client.chat.completions.create(
                model=self.vllm_model_name,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "Solve the ARC task. Infer the rule from training examples, "
                            "prefer symbolic and compositional reasoning, and use candidate "
                            "grids as hints rather than blindly copying them. "
                            "Return only a JSON 2D integer array."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.0,
                max_tokens=2000,
            )
            content = response.choices[0].message.content.strip()
            if "```json" in content:
                content = content.split("```json", 1)[1].split("```", 1)[0].strip()
            elif "```" in content:
                content = content.split("```", 1)[1].split("```", 1)[0].strip()
            payload = json.loads(content)
            return payload if self._is_valid_grid(payload) else None
        except Exception:
            return None

    def _build_prompt(self, train_examples: list[Example], test_input: Grid) -> str:
        candidate_outputs = self._candidate_outputs_for_vllm(train_examples, test_input)
        parts = [
            "Infer the transformation from training examples and solve the test case.\n",
            "Reason about grid size, object counts, object positions, colors, symmetry, "
            "cropping, scaling, gravity, and composition of simple rules.\n",
        ]
        for idx, example in enumerate(train_examples, start=1):
            parts.append(f"Example {idx} input: {json.dumps(example['input'])}\n")
            parts.append(f"Example {idx} output: {json.dumps(example['output'])}\n")
        parts.append(f"Test input: {json.dumps(test_input)}\n")
        if candidate_outputs:
            parts.append(
                "Candidate outputs from symbolic heuristics. Use them only if they fit all training examples:\n"
            )
            for idx, candidate in enumerate(candidate_outputs, start=1):
                parts.append(
                    f"Candidate {idx} ({candidate['name']}): {json.dumps(candidate['grid'])}\n"
                )
        parts.append("Return only the output grid as JSON.")
        return "".join(parts)

    def _candidate_outputs_for_vllm(
        self, train_examples: list[Example], test_input: Grid, limit: int = 8
    ) -> list[dict[str, Any]]:
        shape_counts = Counter(self._shape(example["output"]) for example in train_examples)
        transforms = self._candidate_transforms()
        seen: set[str] = set()
        candidates: list[tuple[int, int, str, Grid]] = []

        for transform in transforms:
            grid = transform.fn(test_input)
            if not self._is_valid_grid(grid):
                continue
            key = json.dumps(grid, separators=(",", ":"))
            if key in seen:
                continue
            seen.add(key)
            candidates.append(
                (shape_counts.get(self._shape(grid), 0), 1, transform.name, grid)
            )

        for example_index, example in enumerate(train_examples, start=1):
            if self._is_valid_grid(example["output"]):
                key = json.dumps(example["output"], separators=(",", ":"))
                if key not in seen:
                    seen.add(key)
                    candidates.append(
                        (
                            shape_counts.get(self._shape(example["output"]), 0),
                            0,
                            f"train_output_{example_index}",
                            example["output"],
                        )
                    )

        majority_guess = self._majority_color_fill_guess(train_examples, test_input)
        key = json.dumps(majority_guess, separators=(",", ":"))
        if key not in seen:
            candidates.append((0, 9, "majority_fill", majority_guess))

        candidates.sort(key=lambda row: (-row[0], row[1], row[2]))
        return [
            {"name": name, "grid": grid}
            for _, _, name, grid in candidates[:limit]
        ]

    def _candidate_transforms(self) -> list[TransformSpec]:
        return [
            TransformSpec("identity", self._identity),
            TransformSpec("rotate_90", self._rotate_90),
            TransformSpec("rotate_180", self._rotate_180),
            TransformSpec("rotate_270", self._rotate_270),
            TransformSpec("flip_h", self._flip_h),
            TransformSpec("flip_v", self._flip_v),
            TransformSpec("transpose", self._transpose),
            TransformSpec("flip_antidiagonal", self._flip_antidiagonal),
            TransformSpec("crop_majority_bg", self._crop_majority_background),
            TransformSpec("crop_zero_bg", lambda grid: self._crop_background(grid, 0)),
            TransformSpec("trim_border", self._trim_uniform_border),
            TransformSpec("extract_largest_object", self._extract_largest_object),
            TransformSpec("extract_smallest_object", self._extract_smallest_object),
            TransformSpec("gravity_down", self._gravity_down),
            TransformSpec("gravity_up", self._gravity_up),
            TransformSpec("gravity_left", self._gravity_left),
            TransformSpec("gravity_right", self._gravity_right),
            TransformSpec("recenter", self._recenter),
            TransformSpec("downsample_2x", self._downsample_2x),
            TransformSpec("scale_x2", lambda grid: self._scale_grid(grid, 2)),
            TransformSpec("scale_x3", lambda grid: self._scale_grid(grid, 3)),
        ]

    def _apply_sequence(self, grid: Grid, sequence: tuple[TransformSpec, ...]) -> Grid:
        current = [row[:] for row in grid]
        for transform in sequence:
            current = transform.fn(current)
        return current

    def _learn_color_map(self, source: Grid, target: Grid) -> dict[int, int] | None:
        if self._shape(source) != self._shape(target):
            return None
        color_map: dict[int, int] = {}
        for row_idx, row in enumerate(source):
            for col_idx, value in enumerate(row):
                mapped = target[row_idx][col_idx]
                existing = color_map.get(value)
                if existing is not None and existing != mapped:
                    return None
                color_map[value] = mapped
        return color_map

    def _majority_color_fill_guess(
        self, train_examples: list[Example], test_input: Grid
    ) -> Grid:
        output_colors: list[int] = []
        for example in train_examples:
            output_colors.extend(value for row in example["output"] for value in row)
        fill = Counter(output_colors).most_common(1)[0][0] if output_colors else 0
        rows, cols = self._shape(test_input)
        return [[fill for _ in range(cols)] for _ in range(rows)]

    def _apply_color_map(self, grid: Grid, color_map: dict[int, int]) -> Grid:
        return [[color_map.get(value, value) for value in row] for row in grid]

    def _scale_grid(self, grid: Grid, factor: int) -> Grid:
        output: Grid = []
        for row in grid:
            expanded_row = []
            for value in row:
                expanded_row.extend([value] * factor)
            for _ in range(factor):
                output.append(expanded_row[:])
        return output

    def _crop_majority_background(self, grid: Grid) -> Grid:
        background = self._majority_color(grid)
        return self._crop_background(grid, background)

    def _trim_uniform_border(self, grid: Grid) -> Grid:
        current = [row[:] for row in grid]
        while len(current) > 1 and self._row_uniform(current[0]):
            current = current[1:]
        while len(current) > 1 and self._row_uniform(current[-1]):
            current = current[:-1]
        while current and len(current[0]) > 1 and self._col_uniform(current, 0):
            current = [row[1:] for row in current]
        while current and len(current[0]) > 1 and self._col_uniform(current, len(current[0]) - 1):
            current = [row[:-1] for row in current]
        return current

    def _extract_largest_object(self, grid: Grid) -> Grid:
        component = self._select_component(grid, largest=True)
        return self._component_to_grid(component) if component else grid

    def _extract_smallest_object(self, grid: Grid) -> Grid:
        component = self._select_component(grid, largest=False)
        return self._component_to_grid(component) if component else grid

    def _select_component(self, grid: Grid, largest: bool) -> list[tuple[int, int, int]] | None:
        components = self._connected_components(grid)
        if not components:
            return None
        components.sort(key=len, reverse=largest)
        return components[0]

    def _component_to_grid(self, component: list[tuple[int, int, int]]) -> Grid:
        min_r = min(r for r, _, _ in component)
        max_r = max(r for r, _, _ in component)
        min_c = min(c for _, c, _ in component)
        max_c = max(c for _, c, _ in component)
        rows = max_r - min_r + 1
        cols = max_c - min_c + 1
        grid = [[0 for _ in range(cols)] for _ in range(rows)]
        for row, col, value in component:
            grid[row - min_r][col - min_c] = value
        return grid

    def _connected_components(self, grid: Grid) -> list[list[tuple[int, int, int]]]:
        rows, cols = self._shape(grid)
        if rows == 0 or cols == 0:
            return []
        visited: set[tuple[int, int]] = set()
        components: list[list[tuple[int, int, int]]] = []
        background = self._background_color(grid)

        for row in range(rows):
            for col in range(cols):
                if (row, col) in visited or grid[row][col] == background:
                    continue
                color = grid[row][col]
                queue = deque([(row, col)])
                visited.add((row, col))
                component: list[tuple[int, int, int]] = []

                while queue:
                    cur_row, cur_col = queue.popleft()
                    component.append((cur_row, cur_col, grid[cur_row][cur_col]))
                    for next_row, next_col in self._neighbors(cur_row, cur_col, rows, cols):
                        if (next_row, next_col) in visited:
                            continue
                        if grid[next_row][next_col] != color:
                            continue
                        visited.add((next_row, next_col))
                        queue.append((next_row, next_col))

                components.append(component)

        return components

    def _crop_background(self, grid: Grid, background: int) -> Grid:
        positions = [
            (row_idx, col_idx)
            for row_idx, row in enumerate(grid)
            for col_idx, value in enumerate(row)
            if value != background
        ]
        if not positions:
            return [row[:] for row in grid]
        min_r = min(row for row, _ in positions)
        max_r = max(row for row, _ in positions)
        min_c = min(col for _, col in positions)
        max_c = max(col for _, col in positions)
        return [row[min_c : max_c + 1] for row in grid[min_r : max_r + 1]]

    def _shift_grid(self, grid: Grid, dx: int, dy: int) -> Grid | None:
        rows, cols = self._shape(grid)
        background = self._background_color(grid)
        output = [[background for _ in range(cols)] for _ in range(rows)]
        for row in range(rows):
            for col in range(cols):
                value = grid[row][col]
                if value == background:
                    continue
                target_row = row + dx
                target_col = col + dy
                if not (0 <= target_row < rows and 0 <= target_col < cols):
                    return None
                output[target_row][target_col] = value
        return output

    def _non_background_points(self, grid: Grid) -> list[tuple[int, int, int]]:
        background = self._background_color(grid)
        return sorted(
            (row_idx, col_idx, value)
            for row_idx, row in enumerate(grid)
            for col_idx, value in enumerate(row)
            if value != background
        )

    def _background_color(self, grid: Grid) -> int:
        values = [value for row in grid for value in row]
        if not values:
            return 0
        return 0 if 0 in values else Counter(values).most_common(1)[0][0]

    def _majority_color(self, grid: Grid) -> int:
        values = [value for row in grid for value in row]
        return Counter(values).most_common(1)[0][0] if values else 0

    def _row_uniform(self, row: list[int]) -> bool:
        return len(set(row)) == 1

    def _col_uniform(self, grid: Grid, index: int) -> bool:
        return len({row[index] for row in grid}) == 1

    def _neighbors(self, row: int, col: int, rows: int, cols: int):
        for delta_row, delta_col in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            next_row = row + delta_row
            next_col = col + delta_col
            if 0 <= next_row < rows and 0 <= next_col < cols:
                yield next_row, next_col

    def _identity(self, grid: Grid) -> Grid:
        return [row[:] for row in grid]

    def _rotate_90(self, grid: Grid) -> Grid:
        return [list(row) for row in zip(*grid[::-1])]

    def _rotate_180(self, grid: Grid) -> Grid:
        return [row[::-1] for row in grid[::-1]]

    def _rotate_270(self, grid: Grid) -> Grid:
        return [list(row) for row in zip(*grid)][::-1]

    def _flip_h(self, grid: Grid) -> Grid:
        return [row[::-1] for row in grid]

    def _flip_v(self, grid: Grid) -> Grid:
        return grid[::-1]

    def _transpose(self, grid: Grid) -> Grid:
        if not grid:
            return []
        return [list(row) for row in zip(*grid)]

    def _flip_antidiagonal(self, grid: Grid) -> Grid:
        return self._rotate_90(self._flip_v(grid))

    def _gravity_down(self, grid: Grid) -> Grid:
        rows, cols = self._shape(grid)
        background = self._background_color(grid)
        output = [[background for _ in range(cols)] for _ in range(rows)]
        for col in range(cols):
            write_row = rows - 1
            for row in range(rows - 1, -1, -1):
                if grid[row][col] != background:
                    output[write_row][col] = grid[row][col]
                    write_row -= 1
        return output

    def _gravity_up(self, grid: Grid) -> Grid:
        rows, cols = self._shape(grid)
        background = self._background_color(grid)
        output = [[background for _ in range(cols)] for _ in range(rows)]
        for col in range(cols):
            write_row = 0
            for row in range(rows):
                if grid[row][col] != background:
                    output[write_row][col] = grid[row][col]
                    write_row += 1
        return output

    def _gravity_left(self, grid: Grid) -> Grid:
        rows, cols = self._shape(grid)
        background = self._background_color(grid)
        output = [[background for _ in range(cols)] for _ in range(rows)]
        for row in range(rows):
            write_col = 0
            for col in range(cols):
                if grid[row][col] != background:
                    output[row][write_col] = grid[row][col]
                    write_col += 1
        return output

    def _gravity_right(self, grid: Grid) -> Grid:
        rows, cols = self._shape(grid)
        background = self._background_color(grid)
        output = [[background for _ in range(cols)] for _ in range(rows)]
        for row in range(rows):
            write_col = cols - 1
            for col in range(cols - 1, -1, -1):
                if grid[row][col] != background:
                    output[row][write_col] = grid[row][col]
                    write_col -= 1
        return output

    def _recenter(self, grid: Grid) -> Grid:
        rows, cols = self._shape(grid)
        background = self._background_color(grid)
        points = [
            (row_idx, col_idx, value)
            for row_idx, row in enumerate(grid)
            for col_idx, value in enumerate(row)
            if value != background
        ]
        if not points:
            return [row[:] for row in grid]
        min_r = min(row for row, _, _ in points)
        max_r = max(row for row, _, _ in points)
        min_c = min(col for _, col, _ in points)
        max_c = max(col for _, col, _ in points)
        content_h = max_r - min_r + 1
        content_w = max_c - min_c + 1
        start_r = (rows - content_h) // 2
        start_c = (cols - content_w) // 2
        output = [[background for _ in range(cols)] for _ in range(rows)]
        for row, col, value in points:
            output[start_r + (row - min_r)][start_c + (col - min_c)] = value
        return output

    def _downsample_2x(self, grid: Grid) -> Grid:
        rows, cols = self._shape(grid)
        if rows < 2 or cols < 2:
            return [row[:] for row in grid]
        return [[grid[row][col] for col in range(0, cols, 2)] for row in range(0, rows, 2)]

    def _shape(self, grid: Grid) -> tuple[int, int]:
        return (len(grid), len(grid[0]) if grid else 0)

    def _is_valid_grid(self, grid: Any) -> bool:
        if not isinstance(grid, list) or not grid:
            return False
        if not all(isinstance(row, list) and row for row in grid):
            return False
        width = len(grid[0])
        for row in grid:
            if len(row) != width:
                return False
            for value in row:
                if not isinstance(value, int) or value < 0 or value > 9:
                    return False
        return True
