# A3 Atomic Executor

This repository executes batches of cookie pick-and-place in the A3 dual-arm MuJoCo simulator. A task planner or a JSON file supplies the source, group size, and destination for each batch. The executor computes gripper poses from scene geometry and runs one fixed pick-and-place behavior per batch.

The batch sequence comes from the caller. Within each batch, the executor sets the jaw aperture, approaches, aligns, descends, closes with force feedback, lifts, carries, aligns to the tray, lowers, releases, and retreats. It stops the episode if a motion or grasp guard rejects a step.

## CLAP experiment

The [CLAP action-predictor experiment](clap_a3/README.md) includes a replayable A3 observation, the official action checkpoint, and an adapter for one simulated move-and-hold prediction.

## Geometry modes

| Mode | Geometry input |
| --- | --- |
| `truth` | Cookie poses and target-tray geometry from MuJoCo. The source hint still selects the search region. |
| `rgbd` | A calibrated front RGB-D frame at the start of each batch. Color, depth, and geometric fitting estimate cookie poses and the tray floor and axes. |
| `both` | Paired episodes for each seed, with a fresh simulator reset for each geometry mode and the same task plan. |

Both modes use the same group selection, pose generation, inverse kinematics, motion control, and guards. RGB-D uses camera calibration, known cookie dimensions, and the tray slot layout. It estimates object geometry without a fallback to simulator object poses. The robot controller uses simulator robot feedback, and the evaluator reads simulator object state after execution to score both modes.

## Visual Planner

The visual Planner proposes one batch from a task instruction, the current front RGB image, and an optional overhead image. It adds a normalized coordinate grid to the front view. An optional JSON history carries previous batch intents and their `accepted` or `rejected` outcomes. The Planner request accepts these image and text inputs without a simulator interface.

After installing the Python dependencies, set `A3_PLANNER_API_BASE_URL` to an OpenAI-compatible API base URL, `A3_PLANNER_MODEL` to a vision-capable model ID, and `A3_PLANNER_API_KEY` to your credential in the process environment. The Planner sends the task and images to that endpoint. It uses the credential in the authorization header and writes the validated proposal to a JSON file.

```sh
python -m a3_atomic_executor.planner --front front.png --overhead overhead.png --task "Pack 20 cookies into the target box, 10 per column." --output proposal.json
```

The proposal carries a count and two pixel hints, or a stop decision. `--history` accepts an array of objects with `intent` (a previous `pick_and_place` proposal) and `outcome` (`accepted` or `rejected`). Python callers can use `plan_next()` to request the next proposal from a fresh image.

The Planner and the paired physical trial currently have separate entry points. A grounding step must resolve a Planner proposal's target pixel hint to `target_column` and `first_row` for the existing fixed-executor plan format. The paired trial below consumes resolved batches directly from JSON.

## Run a paired trial

Use Python 3.12 and a graphics backend that supports MuJoCo offscreen rendering. Install the Python dependencies in a fresh virtual environment from the root of this repository.

```sh
python -m pip install -r requirements.txt
git clone https://github.com/Eter0109/a3_dual_arm_sim.git ../a3_dual_arm_sim
git -C ../a3_dual_arm_sim checkout --detach 6ff458b0b73b78a3ffd794e8b37fdf297a23af48
python -m a3_atomic_executor --a3-root ../a3_dual_arm_sim --geometry both --plan examples/pack-20.json --seed 11 --output artifacts/pack-20-paired
```

Use `--a3-root` to point to an existing A3 simulator checkout. The revision above defines the supported fixture and controller interfaces. The simulator supplies its own robot assets under its applicable licenses. See [NOTICE.md](NOTICE.md).

Use `examples/one-batch.json` for a single 10-cookie unit. Choose a new output directory for each invocation. Add `--video` for MP4 previews. Repeat `--seed` to run more seed pairs, and use `--randomize-boxes` for box-pose variation. The bundled examples use pixel hints for the default scene. For shifted scenes use a plan with hints corresponding to that scene. Cookie randomization is disabled in this runner, so repeating seed values across the default fixed layout does not by itself provide varied scenes.

## Batch intent

The input has a `batches` array. Each element describes one consecutive group and one destination row span.

```json
{
  "count": 10,
  "source_end_pixel_1000": [658, 398],
  "target_center_pixel_1000": [295, 560],
  "target_column": 0,
  "first_row": 0
}
```

- `count` is the number of adjacent cookies to pick, from 1 to 10. The source hint should point near an exposed end of the intended group.
- Both pixel hints use `[x, y]` in the front view with the origin at the top left and a 0–1000 range along each image axis. The target hint should point near the destination tray floor.
- `target_column` is 0 or 1. `first_row` is zero-based along that column. The requested `count` must fit in its 10 rows, and row spans across batches must be disjoint.

The examples specify complete task plans. An upstream planner can produce the same JSON format.

## Results

`summary.json` records pass and attempt counts per geometry mode. Each trial contains `report.json` with the physical score, `events.json` with resolved geometry and failure information, and `control-trace.json` with per-tick gripper and end-effector feedback.

A pass requires all batches to finish, the expected number of released cookies in each target column, all remaining cookies in the source tray, and the inactive arm command to remain unchanged. The full 20-cookie run also requires the simulator's consecutive success ticks. The score checks column counts and the simulator's containment and stability criteria.

This initial executor is scoped to the A3 cookie fixture and uses one geometry capture per batch. The module is licensed under [MIT](LICENSE).
