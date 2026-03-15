# SN5 Hone Solver

ARC-AGI solver for Bittensor Subnet 5 (Hone).

Validators clone this repo and run the solver in their sandbox.

## Strategy

1. **Program search** (depth 1-2): deterministic transforms (rotations, flips, crops, gravity, scaling, object extraction)
2. **Color mapping**: learn color remaps from training examples
3. **Translation detection**: detect shifted grids
4. **Example analogy**: apply transforms learned from examples to test input
5. **vLLM fallback** (optional): LLM model via sidecar container
6. **Majority fill**: last resort

## Files

| File | Purpose |
|------|---------|
| `arc_main.py` | Entrypoint for prep/inference phases |
| `arc_prep_phase.py` | Prep phase (downloads, setup) |
| `arc_inference_phase.py` | Inference phase (solve tasks) |
| `arc_solver.py` | Core solver with 21 transforms |
| `arc_utils.py` | Dataset I/O helpers |
| `smoke_test.py` | Local validation test |
| `Dockerfile` | Container packaging |
| `requirements.txt` | Python dependencies |

## Local Smoke Test

```bash
python3 smoke_test.py
```
