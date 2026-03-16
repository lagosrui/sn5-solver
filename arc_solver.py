"""ARC solver with program search, object-centric transforms, and optional vLLM fallback."""

from __future__ import annotations

import itertools
import json
import os
from collections import Counter, deque
from copy import deepcopy
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

    # ── public entry ─────────────────────────────────────────────

    def solve(self, train_examples: list[Example], test_input: Grid) -> Grid:
        # 1. Pure color mapping (fast, exact)
        solution = self._solve_via_color_map_only(train_examples, test_input)
        if solution is not None:
            return solution

        # 2. Translation (specific, reliable)
        solution = self._solve_via_translation(train_examples, test_input)
        if solution is not None:
            return solution

        # 3. Program search (depth 1 always, depth 2 only with ≥2 examples)
        solution = self._solve_via_program_search(train_examples, test_input)
        if solution is not None:
            return solution

        # 4. Tiling / repeating pattern
        solution = self._solve_via_tiling(train_examples, test_input)
        if solution is not None:
            return solution

        # 5. Grid partitioning (split by separator lines, overlay sub-grids)
        solution = self._solve_via_grid_partition(train_examples, test_input)
        if solution is not None:
            return solution

        # 6. Symmetry completion
        solution = self._solve_via_symmetry(train_examples, test_input)
        if solution is not None:
            return solution

        # 7. Fill / flood fill based rules
        solution = self._solve_via_fill_rules(train_examples, test_input)
        if solution is not None:
            return solution

        # 8. Pixel-wise conditional rules
        solution = self._solve_via_pixel_rules(train_examples, test_input)
        if solution is not None:
            return solution

        # 9. Example analogy
        solution = self._solve_via_example_analogy(train_examples, test_input)
        if solution is not None:
            return solution

        # 10. Object-based transforms
        solution = self._solve_via_object_transforms(train_examples, test_input)
        if solution is not None:
            return solution

        # 11. D4 augmented search (try all 8 symmetries)
        solution = self._solve_via_d4_augmentation(train_examples, test_input)
        if solution is not None:
            return solution

        # 12. vLLM fallback
        if self.vllm_client and self.vllm_model_name:
            predicted = self._solve_with_vllm(train_examples, test_input)
            if self._is_valid_grid(predicted):
                return predicted

        return self._majority_color_fill_guess(train_examples, test_input)

    # ── vLLM ─────────────────────────────────────────────────────

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

    # ── strategy 1: program search ───────────────────────────────

    def _solve_via_program_search(
        self, train_examples: list[Example], test_input: Grid
    ) -> Grid | None:
        transforms = self._candidate_transforms()

        for depth in (1, 2):
            for sequence in itertools.product(transforms, repeat=depth):
                result = self._test_sequence(sequence, train_examples)
                if result is not None:
                    return result(test_input)

        for depth in (1, 2):
            for sequence in itertools.product(transforms, repeat=depth):
                result = self._test_sequence_with_color_map(
                    sequence, train_examples, test_input
                )
                if result is not None:
                    return result(test_input)
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
        transformed_test_colors = {v for row in transformed_test for v in row}
        if not transformed_test_colors.issubset(set(merged_map)):
            return None

        def program(grid: Grid) -> Grid:
            t = self._apply_sequence(grid, sequence)
            return self._apply_color_map(t, merged_map)
        return program

    # ── strategy 2: pure color map ───────────────────────────────

    def _solve_via_color_map_only(
        self, train_examples: list[Example], test_input: Grid
    ) -> Grid | None:
        merged: dict[int, int] = {}
        for ex in train_examples:
            if self._shape(ex["input"]) != self._shape(ex["output"]):
                return None
            cmap = self._learn_color_map(ex["input"], ex["output"])
            if cmap is None:
                return None
            for k, v in cmap.items():
                if k in merged and merged[k] != v:
                    return None
                merged[k] = v
        test_colors = {v for row in test_input for v in row}
        if not test_colors.issubset(set(merged)):
            return None
        return self._apply_color_map(test_input, merged)

    # ── strategy 3: translation ──────────────────────────────────

    def _solve_via_translation(
        self, train_examples: list[Example], test_input: Grid
    ) -> Grid | None:
        delta: tuple[int, int] | None = None
        for ex in train_examples:
            inferred = self._infer_translation(ex["input"], ex["output"])
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

    # ── strategy 4: tiling / repeating ───────────────────────────

    def _solve_via_tiling(
        self, train_examples: list[Example], test_input: Grid
    ) -> Grid | None:
        for ex in train_examples:
            ir, ic = self._shape(ex["input"])
            orr, oc = self._shape(ex["output"])
            if ir == 0 or ic == 0:
                return None
            if orr % ir != 0 or oc % ic != 0:
                return None
            rr, rc = orr // ir, oc // ic
            if rr < 1 or rc < 1 or (rr == 1 and rc == 1):
                return None
            tiled = self._tile_grid(ex["input"], rr, rc)
            if tiled != ex["output"]:
                return None
        # all examples match same tiling ratio
        ir2, ic2 = self._shape(train_examples[0]["input"])
        orr2, oc2 = self._shape(train_examples[0]["output"])
        rr, rc = orr2 // ir2, oc2 // ic2
        return self._tile_grid(test_input, rr, rc)

    def _tile_grid(self, grid: Grid, rows_repeat: int, cols_repeat: int) -> Grid:
        result: Grid = []
        for _ in range(rows_repeat):
            for row in grid:
                result.append(row * cols_repeat)
        return result

    # ── strategy 5: fill rules ───────────────────────────────────

    def _solve_via_fill_rules(
        self, train_examples: list[Example], test_input: Grid
    ) -> Grid | None:
        # Check if output = input with enclosed regions filled
        fill_color = self._detect_fill_color(train_examples)
        if fill_color is None:
            return None
        # Verify on all training examples
        for ex in train_examples:
            candidate = self._fill_enclosed_regions(ex["input"], fill_color)
            if candidate != ex["output"]:
                return None
        return self._fill_enclosed_regions(test_input, fill_color)

    # ── strategy 5: grid partition ─────────────────────────────

    def _solve_via_grid_partition(
        self, train_examples: list[Example], test_input: Grid
    ) -> Grid | None:
        # Try splitting grids by separator lines and combining sub-grids
        for op in ("xor", "or", "and"):
            result = self._try_partition_overlay(train_examples, test_input, op)
            if result is not None:
                return result
        return None

    def _try_partition_overlay(
        self, train_examples: list[Example], test_input: Grid, op: str
    ) -> Grid | None:
        for split_fn in (self._split_by_h_separator, self._split_by_v_separator):
            parts_match = True
            for ex in train_examples:
                sub_grids = split_fn(ex["input"])
                if sub_grids is None or len(sub_grids) < 2:
                    parts_match = False
                    break
                combined = self._overlay_grids(sub_grids, op)
                if combined is None or combined != ex["output"]:
                    parts_match = False
                    break
            if parts_match:
                test_parts = split_fn(test_input)
                if test_parts and len(test_parts) >= 2:
                    result = self._overlay_grids(test_parts, op)
                    if result is not None:
                        return result
        return None

    def _split_by_h_separator(self, grid: Grid) -> list[Grid] | None:
        """Split grid by horizontal separator rows (rows of uniform color)."""
        rows, cols = self._shape(grid)
        if rows < 3:
            return None
        sep_rows = []
        for r in range(rows):
            if len(set(grid[r])) == 1:
                sep_rows.append(r)
        if not sep_rows:
            return None
        parts: list[Grid] = []
        prev = 0
        for sr in sep_rows:
            if sr > prev:
                parts.append([row[:] for row in grid[prev:sr]])
            prev = sr + 1
        if prev < rows:
            parts.append([row[:] for row in grid[prev:]])
        # All parts must have same shape
        if len(parts) < 2:
            return None
        shapes = [self._shape(p) for p in parts]
        if len(set(shapes)) != 1:
            return None
        return parts

    def _split_by_v_separator(self, grid: Grid) -> list[Grid] | None:
        """Split grid by vertical separator columns."""
        transposed = self._transpose(grid)
        parts = self._split_by_h_separator(transposed)
        if parts is None:
            return None
        return [self._transpose(p) for p in parts]

    def _overlay_grids(self, grids: list[Grid], op: str) -> Grid | None:
        if not grids:
            return None
        base = grids[0]
        rows, cols = self._shape(base)
        for g in grids[1:]:
            if self._shape(g) != (rows, cols):
                return None
        result = [[0] * cols for _ in range(rows)]
        for r in range(rows):
            for c in range(cols):
                vals = [g[r][c] for g in grids]
                if op == "xor":
                    # Non-zero XOR: keep value that differs from background
                    nonzero = [v for v in vals if v != 0]
                    result[r][c] = nonzero[0] if len(nonzero) == 1 else (0 if not nonzero else max(set(nonzero), key=nonzero.count))
                elif op == "or":
                    result[r][c] = max(vals)
                elif op == "and":
                    result[r][c] = min(v for v in vals if v != 0) if all(v != 0 for v in vals) else 0
        return result

    # ── strategy 6: symmetry completion ──────────────────────────

    def _solve_via_symmetry(
        self, train_examples: list[Example], test_input: Grid
    ) -> Grid | None:
        for sym_fn in (self._complete_h_symmetry, self._complete_v_symmetry):
            all_match = True
            for ex in train_examples:
                if self._shape(ex["input"]) != self._shape(ex["output"]):
                    all_match = False
                    break
                candidate = sym_fn(ex["input"])
                if candidate != ex["output"]:
                    all_match = False
                    break
            if all_match:
                return sym_fn(test_input)
        return None

    def _complete_h_symmetry(self, grid: Grid) -> Grid:
        """Complete horizontal symmetry: mirror left half to right."""
        rows, cols = self._shape(grid)
        result = [r[:] for r in grid]
        mid = cols // 2
        for r in range(rows):
            for c in range(mid):
                mirror_c = cols - 1 - c
                if grid[r][c] != 0 and grid[r][mirror_c] == 0:
                    result[r][mirror_c] = grid[r][c]
                elif grid[r][mirror_c] != 0 and grid[r][c] == 0:
                    result[r][c] = grid[r][mirror_c]
        return result

    def _complete_v_symmetry(self, grid: Grid) -> Grid:
        """Complete vertical symmetry: mirror top half to bottom."""
        rows, cols = self._shape(grid)
        result = [r[:] for r in grid]
        mid = rows // 2
        for r in range(mid):
            mirror_r = rows - 1 - r
            for c in range(cols):
                if grid[r][c] != 0 and grid[mirror_r][c] == 0:
                    result[mirror_r][c] = grid[r][c]
                elif grid[mirror_r][c] != 0 and grid[r][c] == 0:
                    result[r][c] = grid[mirror_r][c]
        return result

    # ── strategy 11: D4 augmented search ─────────────────────────

    def _solve_via_d4_augmentation(
        self, train_examples: list[Example], test_input: Grid
    ) -> Grid | None:
        """Try solving under all 8 D4 symmetry transformations."""
        d4_transforms = [
            ("identity", self._identity, self._identity),
            ("rot90", self._rotate_90, self._rotate_270),
            ("rot180", self._rotate_180, self._rotate_180),
            ("rot270", self._rotate_270, self._rotate_90),
            ("flip_h", self._flip_h, self._flip_h),
            ("flip_v", self._flip_v, self._flip_v),
            ("transpose", self._transpose, self._transpose),
            ("anti_diag", self._flip_antidiagonal, self._flip_antidiagonal),
        ]
        for name, fwd, inv in d4_transforms:
            if name == "identity":
                continue  # Already tried
            aug_examples = [
                {"input": fwd(ex["input"]), "output": fwd(ex["output"])}
                for ex in train_examples
            ]
            aug_test = fwd(test_input)
            # Try only the fast strategies on augmented data
            for strategy in (
                self._solve_via_color_map_only,
                self._solve_via_translation,
                self._solve_via_tiling,
                self._solve_via_fill_rules,
            ):
                result = strategy(aug_examples, aug_test)
                if result is not None:
                    return inv(result)
        return None

    def _detect_fill_color(self, train_examples: list[Example]) -> int | None:
        """Detect which color is used to fill enclosed regions."""
        for ex in train_examples:
            inp, out = ex["input"], ex["output"]
            if self._shape(inp) != self._shape(out):
                return None
            diff_colors = set()
            for r in range(len(inp)):
                for c in range(len(inp[0])):
                    if inp[r][c] != out[r][c] and inp[r][c] == 0:
                        diff_colors.add(out[r][c])
            if len(diff_colors) == 1:
                return diff_colors.pop()
        return None

    def _fill_enclosed_regions(self, grid: Grid, fill_color: int) -> Grid:
        rows, cols = self._shape(grid)
        if rows == 0 or cols == 0:
            return [r[:] for r in grid]
        result = [r[:] for r in grid]
        # Find background cells NOT reachable from edges
        visited: set[tuple[int, int]] = set()
        queue: deque[tuple[int, int]] = deque()
        for r in range(rows):
            for c in [0, cols - 1]:
                if grid[r][c] == 0 and (r, c) not in visited:
                    visited.add((r, c))
                    queue.append((r, c))
        for c in range(cols):
            for r in [0, rows - 1]:
                if grid[r][c] == 0 and (r, c) not in visited:
                    visited.add((r, c))
                    queue.append((r, c))
        while queue:
            cr, cc = queue.popleft()
            for nr, nc in self._neighbors(cr, cc, rows, cols):
                if (nr, nc) not in visited and grid[nr][nc] == 0:
                    visited.add((nr, nc))
                    queue.append((nr, nc))
        # Fill enclosed background cells
        for r in range(rows):
            for c in range(cols):
                if grid[r][c] == 0 and (r, c) not in visited:
                    result[r][c] = fill_color
        return result

    # ── strategy 6: pixel-wise rules ─────────────────────────────

    def _solve_via_pixel_rules(
        self, train_examples: list[Example], test_input: Grid
    ) -> Grid | None:
        # Try: output[r][c] = f(input[r][c], neighbors)
        # Simple case: replace one color with another based on neighbor count
        if not train_examples:
            return None
        for ex in train_examples:
            if self._shape(ex["input"]) != self._shape(ex["output"]):
                return None

        # Try: replace bg cells adjacent to non-bg with a specific color
        bg = self._background_color(train_examples[0]["input"])
        replace_color = None
        for ex in train_examples:
            inp, out = ex["input"], ex["output"]
            rows, cols = self._shape(inp)
            for r in range(rows):
                for c in range(cols):
                    if inp[r][c] != out[r][c]:
                        if inp[r][c] != bg:
                            return None  # non-bg changed - too complex
                        # bg cell changed: check if adjacent to non-bg
                        has_nonbg_neighbor = any(
                            inp[nr][nc] != bg
                            for nr, nc in self._neighbors(r, c, rows, cols)
                        )
                        if not has_nonbg_neighbor:
                            return None
                        if replace_color is None:
                            replace_color = out[r][c]
                        elif replace_color != out[r][c]:
                            return None

        if replace_color is None:
            return None

        # Verify
        for ex in train_examples:
            candidate = self._apply_neighbor_fill(ex["input"], bg, replace_color)
            if candidate != ex["output"]:
                return None
        return self._apply_neighbor_fill(test_input, bg, replace_color)

    def _apply_neighbor_fill(self, grid: Grid, bg: int, fill: int) -> Grid:
        rows, cols = self._shape(grid)
        result = [r[:] for r in grid]
        for r in range(rows):
            for c in range(cols):
                if grid[r][c] == bg:
                    has_nonbg = any(
                        grid[nr][nc] != bg
                        for nr, nc in self._neighbors(r, c, rows, cols)
                    )
                    if has_nonbg:
                        result[r][c] = fill
        return result

    # ── strategy 7: example analogy ──────────────────────────────

    def _solve_via_example_analogy(
        self, train_examples: list[Example], test_input: Grid
    ) -> Grid | None:
        transforms = self._candidate_transforms()
        for ex in train_examples:
            for depth in (1, 2):
                for seq in itertools.product(transforms, repeat=depth):
                    if self._apply_sequence(ex["input"], seq) == test_input:
                        return self._apply_sequence(ex["output"], seq)
            delta = self._infer_translation(ex["input"], test_input)
            if delta is not None:
                shifted = self._shift_grid(ex["output"], delta[0], delta[1])
                if shifted is not None:
                    return shifted
        return None

    # ── strategy 8: object-based transforms ──────────────────────

    def _solve_via_object_transforms(
        self, train_examples: list[Example], test_input: Grid
    ) -> Grid | None:
        # Try: sort objects by size/position, recolor, or rearrange
        # Simple case: count objects → output is the count as a grid
        counts = []
        for ex in train_examples:
            comps = self._connected_components(ex["input"])
            out = ex["output"]
            if len(out) == 1 and len(out[0]) == 1:
                counts.append((len(comps), out[0][0]))
            else:
                counts.clear()
                break
        if counts and all(c == v for c, v in counts):
            test_comps = self._connected_components(test_input)
            return [[len(test_comps)]]

        # Try: extract the object that appears in all examples
        # (keep largest, keep smallest already in transforms)
        return None

    # ── candidate transforms ─────────────────────────────────────

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
            # New transforms
            TransformSpec("sort_rows_by_color", self._sort_rows_by_nonzero_count),
            TransformSpec("sort_cols_by_color", self._sort_cols_by_nonzero_count),
            TransformSpec("remove_bg_rows", self._remove_background_rows),
            TransformSpec("remove_bg_cols", self._remove_background_cols),
            TransformSpec("unique_rows", self._unique_rows),
            TransformSpec("unique_cols", self._unique_cols),
            TransformSpec("max_color_fill", self._fill_with_max_color),
            TransformSpec("hollow_rect", self._hollow_rectangle),
            TransformSpec("mirror_h_extend", self._mirror_horizontal_extend),
            TransformSpec("mirror_v_extend", self._mirror_vertical_extend),
            TransformSpec("replace_bg_with_most_freq", self._replace_bg_most_freq),
            TransformSpec("outline_objects", self._outline_objects),
            TransformSpec("tile_2x2", lambda g: self._tile_grid(g, 2, 2)),
            TransformSpec("tile_3x3", lambda g: self._tile_grid(g, 3, 3)),
            TransformSpec("tile_1x2", lambda g: self._tile_grid(g, 1, 2)),
            TransformSpec("tile_2x1", lambda g: self._tile_grid(g, 2, 1)),
            TransformSpec("border_1px", self._add_border),
            TransformSpec("compress_colors", self._compress_colors),
            TransformSpec("invert_colors", self._invert_colors),
            TransformSpec("keep_nonzero_only", self._keep_nonzero_mask),
            TransformSpec("downsample_3x", self._downsample_3x),
        ]

    # ── helpers ──────────────────────────────────────────────────

    def _apply_sequence(self, grid: Grid, sequence: tuple[TransformSpec, ...]) -> Grid:
        current = [row[:] for row in grid]
        for transform in sequence:
            current = transform.fn(current)
        return current

    def _learn_color_map(self, source: Grid, target: Grid) -> dict[int, int] | None:
        if self._shape(source) != self._shape(target):
            return None
        color_map: dict[int, int] = {}
        for ri, row in enumerate(source):
            for ci, val in enumerate(row):
                mapped = target[ri][ci]
                existing = color_map.get(val)
                if existing is not None and existing != mapped:
                    return None
                color_map[val] = mapped
        return color_map

    def _majority_color_fill_guess(
        self, train_examples: list[Example], test_input: Grid
    ) -> Grid:
        output_colors: list[int] = []
        for ex in train_examples:
            output_colors.extend(v for row in ex["output"] for v in row)
        fill = Counter(output_colors).most_common(1)[0][0] if output_colors else 0
        # Predict output size from training
        out_shapes = [self._shape(ex["output"]) for ex in train_examples]
        if len(set(out_shapes)) == 1:
            rows, cols = out_shapes[0]
        else:
            rows, cols = self._shape(test_input)
        return [[fill for _ in range(cols)] for _ in range(rows)]

    def _apply_color_map(self, grid: Grid, color_map: dict[int, int]) -> Grid:
        return [[color_map.get(v, v) for v in row] for row in grid]

    # ── geometric transforms ─────────────────────────────────────

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

    # ── scale / crop ─────────────────────────────────────────────

    def _scale_grid(self, grid: Grid, factor: int) -> Grid:
        output: Grid = []
        for row in grid:
            expanded = []
            for v in row:
                expanded.extend([v] * factor)
            for _ in range(factor):
                output.append(expanded[:])
        return output

    def _crop_majority_background(self, grid: Grid) -> Grid:
        return self._crop_background(grid, self._majority_color(grid))

    def _crop_background(self, grid: Grid, bg: int) -> Grid:
        positions = [
            (ri, ci) for ri, row in enumerate(grid)
            for ci, v in enumerate(row) if v != bg
        ]
        if not positions:
            return [r[:] for r in grid]
        min_r = min(r for r, _ in positions)
        max_r = max(r for r, _ in positions)
        min_c = min(c for _, c in positions)
        max_c = max(c for _, c in positions)
        return [row[min_c:max_c + 1] for row in grid[min_r:max_r + 1]]

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

    def _downsample_2x(self, grid: Grid) -> Grid:
        rows, cols = self._shape(grid)
        if rows < 2 or cols < 2:
            return [r[:] for r in grid]
        return [[grid[r][c] for c in range(0, cols, 2)] for r in range(0, rows, 2)]

    def _downsample_3x(self, grid: Grid) -> Grid:
        rows, cols = self._shape(grid)
        if rows < 3 or cols < 3:
            return [r[:] for r in grid]
        return [[grid[r][c] for c in range(0, cols, 3)] for r in range(0, rows, 3)]

    # ── gravity ──────────────────────────────────────────────────

    def _gravity_down(self, grid: Grid) -> Grid:
        rows, cols = self._shape(grid)
        bg = self._background_color(grid)
        out = [[bg] * cols for _ in range(rows)]
        for c in range(cols):
            wr = rows - 1
            for r in range(rows - 1, -1, -1):
                if grid[r][c] != bg:
                    out[wr][c] = grid[r][c]
                    wr -= 1
        return out

    def _gravity_up(self, grid: Grid) -> Grid:
        rows, cols = self._shape(grid)
        bg = self._background_color(grid)
        out = [[bg] * cols for _ in range(rows)]
        for c in range(cols):
            wr = 0
            for r in range(rows):
                if grid[r][c] != bg:
                    out[wr][c] = grid[r][c]
                    wr += 1
        return out

    def _gravity_left(self, grid: Grid) -> Grid:
        rows, cols = self._shape(grid)
        bg = self._background_color(grid)
        out = [[bg] * cols for _ in range(rows)]
        for r in range(rows):
            wc = 0
            for c in range(cols):
                if grid[r][c] != bg:
                    out[r][wc] = grid[r][c]
                    wc += 1
        return out

    def _gravity_right(self, grid: Grid) -> Grid:
        rows, cols = self._shape(grid)
        bg = self._background_color(grid)
        out = [[bg] * cols for _ in range(rows)]
        for r in range(rows):
            wc = cols - 1
            for c in range(cols - 1, -1, -1):
                if grid[r][c] != bg:
                    out[r][wc] = grid[r][c]
                    wc -= 1
        return out

    def _recenter(self, grid: Grid) -> Grid:
        rows, cols = self._shape(grid)
        bg = self._background_color(grid)
        pts = [(ri, ci, v) for ri, row in enumerate(grid) for ci, v in enumerate(row) if v != bg]
        if not pts:
            return [r[:] for r in grid]
        min_r = min(r for r, _, _ in pts)
        max_r = max(r for r, _, _ in pts)
        min_c = min(c for _, c, _ in pts)
        max_c = max(c for _, c, _ in pts)
        ch, cw = max_r - min_r + 1, max_c - min_c + 1
        sr, sc = (rows - ch) // 2, (cols - cw) // 2
        out = [[bg] * cols for _ in range(rows)]
        for r, c, v in pts:
            out[sr + (r - min_r)][sc + (c - min_c)] = v
        return out

    # ── object extraction ────────────────────────────────────────

    def _extract_largest_object(self, grid: Grid) -> Grid:
        comp = self._select_component(grid, largest=True)
        return self._component_to_grid(comp) if comp else grid

    def _extract_smallest_object(self, grid: Grid) -> Grid:
        comp = self._select_component(grid, largest=False)
        return self._component_to_grid(comp) if comp else grid

    def _select_component(self, grid: Grid, largest: bool) -> list[tuple[int, int, int]] | None:
        comps = self._connected_components(grid)
        if not comps:
            return None
        comps.sort(key=len, reverse=largest)
        return comps[0]

    def _component_to_grid(self, comp: list[tuple[int, int, int]]) -> Grid:
        min_r = min(r for r, _, _ in comp)
        max_r = max(r for r, _, _ in comp)
        min_c = min(c for _, c, _ in comp)
        max_c = max(c for _, c, _ in comp)
        h, w = max_r - min_r + 1, max_c - min_c + 1
        grid = [[0] * w for _ in range(h)]
        for r, c, v in comp:
            grid[r - min_r][c - min_c] = v
        return grid

    def _connected_components(self, grid: Grid) -> list[list[tuple[int, int, int]]]:
        rows, cols = self._shape(grid)
        if rows == 0 or cols == 0:
            return []
        visited: set[tuple[int, int]] = set()
        components: list[list[tuple[int, int, int]]] = []
        bg = self._background_color(grid)
        for r in range(rows):
            for c in range(cols):
                if (r, c) in visited or grid[r][c] == bg:
                    continue
                color = grid[r][c]
                queue = deque([(r, c)])
                visited.add((r, c))
                comp: list[tuple[int, int, int]] = []
                while queue:
                    cr, cc = queue.popleft()
                    comp.append((cr, cc, grid[cr][cc]))
                    for nr, nc in self._neighbors(cr, cc, rows, cols):
                        if (nr, nc) not in visited and grid[nr][nc] == color:
                            visited.add((nr, nc))
                            queue.append((nr, nc))
                components.append(comp)
        return components

    # ── NEW transforms ───────────────────────────────────────────

    def _sort_rows_by_nonzero_count(self, grid: Grid) -> Grid:
        bg = self._background_color(grid)
        return sorted(grid, key=lambda row: sum(1 for v in row if v != bg))

    def _sort_cols_by_nonzero_count(self, grid: Grid) -> Grid:
        return self._transpose(self._sort_rows_by_nonzero_count(self._transpose(grid)))

    def _remove_background_rows(self, grid: Grid) -> Grid:
        bg = self._background_color(grid)
        filtered = [row for row in grid if any(v != bg for v in row)]
        return filtered if filtered else [grid[0][:]]

    def _remove_background_cols(self, grid: Grid) -> Grid:
        return self._transpose(self._remove_background_rows(self._transpose(grid)))

    def _unique_rows(self, grid: Grid) -> Grid:
        seen: list[list[int]] = []
        for row in grid:
            if row not in seen:
                seen.append(row)
        return seen if seen else [grid[0][:]]

    def _unique_cols(self, grid: Grid) -> Grid:
        return self._transpose(self._unique_rows(self._transpose(grid)))

    def _fill_with_max_color(self, grid: Grid) -> Grid:
        bg = self._background_color(grid)
        colors = [v for row in grid for v in row if v != bg]
        if not colors:
            return [r[:] for r in grid]
        most_common = Counter(colors).most_common(1)[0][0]
        return [[most_common if v == bg else v for v in row] for row in grid]

    def _hollow_rectangle(self, grid: Grid) -> Grid:
        rows, cols = self._shape(grid)
        if rows < 3 or cols < 3:
            return [r[:] for r in grid]
        result = [r[:] for r in grid]
        bg = self._background_color(grid)
        for r in range(1, rows - 1):
            for c in range(1, cols - 1):
                if grid[r][c] != bg:
                    result[r][c] = bg
        return result

    def _mirror_horizontal_extend(self, grid: Grid) -> Grid:
        return [row + row[::-1] for row in grid]

    def _mirror_vertical_extend(self, grid: Grid) -> Grid:
        return grid + grid[::-1]

    def _replace_bg_most_freq(self, grid: Grid) -> Grid:
        bg = self._background_color(grid)
        colors = [v for row in grid for v in row if v != bg]
        if not colors:
            return [r[:] for r in grid]
        most = Counter(colors).most_common(1)[0][0]
        return [[most if v == bg else v for v in row] for row in grid]

    def _outline_objects(self, grid: Grid) -> Grid:
        rows, cols = self._shape(grid)
        bg = self._background_color(grid)
        result = [[bg] * cols for _ in range(rows)]
        for r in range(rows):
            for c in range(cols):
                if grid[r][c] != bg:
                    has_bg_neighbor = any(
                        grid[nr][nc] == bg
                        for nr, nc in self._neighbors(r, c, rows, cols)
                    ) or r == 0 or r == rows - 1 or c == 0 or c == cols - 1
                    if has_bg_neighbor:
                        result[r][c] = grid[r][c]
        return result

    def _add_border(self, grid: Grid) -> Grid:
        rows, cols = self._shape(grid)
        if rows == 0:
            return grid
        colors = [v for row in grid for v in row if v != 0]
        border_color = Counter(colors).most_common(1)[0][0] if colors else 1
        new_cols = cols + 2
        result: Grid = [[border_color] * new_cols]
        for row in grid:
            result.append([border_color] + row + [border_color])
        result.append([border_color] * new_cols)
        return result

    def _compress_colors(self, grid: Grid) -> Grid:
        """Map colors to 0, 1, 2, ... in order of first appearance."""
        mapping: dict[int, int] = {}
        counter = 0
        for row in grid:
            for v in row:
                if v not in mapping:
                    mapping[v] = counter
                    counter += 1
        return [[mapping[v] for v in row] for row in grid]

    def _invert_colors(self, grid: Grid) -> Grid:
        return [[9 - v for v in row] for row in grid]

    def _keep_nonzero_mask(self, grid: Grid) -> Grid:
        return [[1 if v != 0 else 0 for v in row] for row in grid]

    # ── translation helpers ──────────────────────────────────────

    def _infer_translation(self, inp: Grid, out: Grid) -> tuple[int, int] | None:
        if self._shape(inp) != self._shape(out):
            return None
        ip = self._non_background_points(inp)
        op = self._non_background_points(out)
        if not ip or len(ip) != len(op):
            return None
        si, so = sorted(ip), sorted(op)
        dx = so[0][0] - si[0][0]
        dy = so[0][1] - si[0][1]
        for (ir, ic, iv), (orr, oc, ov) in zip(si, so):
            if ov != iv or orr - ir != dx or oc - ic != dy:
                return None
        return (dx, dy)

    def _shift_grid(self, grid: Grid, dx: int, dy: int) -> Grid | None:
        rows, cols = self._shape(grid)
        bg = self._background_color(grid)
        out = [[bg] * cols for _ in range(rows)]
        for r in range(rows):
            for c in range(cols):
                if grid[r][c] == bg:
                    continue
                tr, tc = r + dx, c + dy
                if not (0 <= tr < rows and 0 <= tc < cols):
                    return None
                out[tr][tc] = grid[r][c]
        return out

    # ── vLLM ─────────────────────────────────────────────────────

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
                            "You are an expert at solving ARC-AGI puzzles. "
                            "Analyze the input-output pairs to find the transformation rule. "
                            "Apply it to the test input. Return ONLY a JSON 2D integer array."
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
        candidates = self._candidate_outputs_for_vllm(train_examples, test_input)
        parts = [
            "Find the transformation rule from these input-output examples.\n\n",
        ]
        for idx, ex in enumerate(train_examples, 1):
            parts.append(f"Example {idx} input:\n{self._grid_to_visual(ex['input'])}\n")
            parts.append(f"Example {idx} output:\n{self._grid_to_visual(ex['output'])}\n\n")
        parts.append(f"Test input:\n{self._grid_to_visual(test_input)}\n\n")
        if candidates:
            parts.append("Candidate outputs from heuristics (use only if correct):\n")
            for idx, c in enumerate(candidates[:4], 1):
                parts.append(f"Candidate {idx} ({c['name']}): {json.dumps(c['grid'])}\n")
            parts.append("\n")
        parts.append("Return the test output as a JSON 2D integer array. No explanation.")
        return "".join(parts)

    def _grid_to_visual(self, grid: Grid) -> str:
        return "\n".join(" ".join(str(v) for v in row) for row in grid)

    def _candidate_outputs_for_vllm(
        self, train_examples: list[Example], test_input: Grid, limit: int = 8
    ) -> list[dict[str, Any]]:
        shape_counts = Counter(self._shape(ex["output"]) for ex in train_examples)
        transforms = self._candidate_transforms()
        seen: set[str] = set()
        candidates: list[tuple[int, int, str, Grid]] = []
        for t in transforms:
            g = t.fn(test_input)
            if not self._is_valid_grid(g):
                continue
            key = json.dumps(g, separators=(",", ":"))
            if key in seen:
                continue
            seen.add(key)
            candidates.append((shape_counts.get(self._shape(g), 0), 1, t.name, g))
        candidates.sort(key=lambda x: (-x[0], x[1], x[2]))
        return [{"name": n, "grid": g} for _, _, n, g in candidates[:limit]]

    # ── utility ──────────────────────────────────────────────────

    def _non_background_points(self, grid: Grid) -> list[tuple[int, int, int]]:
        bg = self._background_color(grid)
        return sorted(
            (ri, ci, v) for ri, row in enumerate(grid)
            for ci, v in enumerate(row) if v != bg
        )

    def _background_color(self, grid: Grid) -> int:
        vals = [v for row in grid for v in row]
        if not vals:
            return 0
        return 0 if 0 in vals else Counter(vals).most_common(1)[0][0]

    def _majority_color(self, grid: Grid) -> int:
        vals = [v for row in grid for v in row]
        return Counter(vals).most_common(1)[0][0] if vals else 0

    def _row_uniform(self, row: list[int]) -> bool:
        return len(set(row)) == 1

    def _col_uniform(self, grid: Grid, idx: int) -> bool:
        return len({row[idx] for row in grid}) == 1

    def _neighbors(self, r: int, c: int, rows: int, cols: int):
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols:
                yield nr, nc

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
            for v in row:
                if not isinstance(v, int) or v < 0 or v > 9:
                    return False
        return True
