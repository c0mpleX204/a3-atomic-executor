"""Run paired truth/RGB-D trials against the same fixed A3 executor."""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path
import sys
from time import perf_counter

import cv2
import numpy as np


def write_json(path, payload):
    path.write_text(json.dumps(payload, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def assess(env, intents, right_before):
    in_target = tuple(env.privileged_cookie_in_target(i) for i in range(len(env._cookie_bodies)))
    target_ids = np.flatnonzero(in_target).tolist()
    source_count = sum(env.privileged_cookie_in_source(i) for i in range(len(in_target)))
    column_counts = list(env.target_column_counts(in_target))
    expected_by_column = [sum(batch.count for batch in intents if batch.target_column == col) for col in (0, 1)]
    expected = sum(expected_by_column)
    released = bool(target_ids) and not any(any(env.privileged_left_finger_contacts(i)) for i in target_ids)
    right_unchanged = bool(np.array_equal(right_before, env.last_applied_action[8:]))
    stable = env.success_hold_count
    full_task_stable = expected != env.task_config.required_cookies or stable >= env.task_config.success_hold_steps
    physical_pass = (column_counts == expected_by_column and source_count == len(in_target) - expected
                     and released and right_unchanged and full_task_stable)
    return dict(physical_pass=physical_pass, expected_count=expected, target_count=len(target_ids),
                target_column_counts=column_counts, expected_column_counts=expected_by_column,
                target_ids=target_ids, source_count=source_count, released=released,
                inactive_arm_unchanged=right_unchanged, full_task_stable_ticks=stable,
                full_task_required_stable_ticks=env.task_config.success_hold_steps)


def trial(folder, mode, seed, intents, randomize_boxes, save_video):
    from a3_dual_arm_sim.tasks.cookie_2x10 import A3Cookie2x10Env, Cookie2x10TaskConfig
    from .arm import Rejected, make_robot
    from .behavior import PickAndPlace
    from .geometry import RGBDGeometry, TruthGeometry
    from .observation import A3RGBDSensor
    from .video import render_video

    folder.mkdir(parents=True, exist_ok=False)
    env = A3Cookie2x10Env(action_mode='cartesian_delta', render_cameras=False,
                          task_config=replace(Cookie2x10TaskConfig(), terminate_on_success=False))
    frames, events = [], []
    current_batch = 0
    last_recorded_tick = 0
    arm = None
    started = perf_counter()
    report = dict(seed=seed, geometry=mode, randomize_boxes=randomize_boxes,
                  batches_completed=0, success=False)

    def record(_observation_tick, phase):
        nonlocal last_recorded_tick
        if save_video and env._step_count % 2 == 0 and env._step_count > last_recorded_tick:
            frames.append((env.data.qpos.copy(), f'{current_batch}: {phase}'))
            last_recorded_tick = env._step_count

    def emit(**event):
        events.append(dict(event, batch=current_batch))

    try:
        env.reset(seed=seed, options={'randomize_cookies': False, 'randomize_boxes': randomize_boxes})
        right_before = env.last_applied_action[8:].copy()
        size = env.COOKIE_HALF_SIZE * 2
        slots = np.asarray(env.TARGET_SLOTS_LOCAL)
        arm = make_robot(env, record)
        if save_video:
            frames.append((env.data.qpos.copy(), 'initial'))
        try:
            with A3RGBDSensor(env.model, env.data, width=1600, height=1200) as sensor:
                geometry = TruthGeometry(env) if mode == 'truth' else RGBDGeometry(size)
                executor = PickAndPlace(arm, sensor, geometry, size, slots, emit)
                for current_batch, intent in enumerate(intents, 1):
                    emit(kind='intent', intent=asdict(intent))
                    print(f'{mode} seed {seed} batch {current_batch} ({intent.count} cookies)', flush=True)
                    executor.run(intent)
                    report['batches_completed'] += 1
        except Rejected as exc:
            arm.executor.stop()
            report['rejected'] = str(exc)
            emit(kind='rejected', reason=str(exc))

        # Scripted scoring runs after this executor has returned or stopped.
        report['score'] = assess(env, intents, right_before)
        report['success'] = report['batches_completed'] == len(intents) and report['score']['physical_pass']
        report['control_ticks'] = env._step_count
        report['elapsed_before_video_s'] = round(perf_counter() - started, 3)
        write_json(folder / 'events.json', events)
        write_json(folder / 'control-trace.json', arm.trace)
        write_json(folder / 'report.json', report)
        if save_video and env._step_count > 0:
            frames.append((env.data.qpos.copy(), 'passed' if report['success'] else 'stopped'))
            env.model.vis.global_.offwidth = max(640, env.model.vis.global_.offwidth)
            env.model.vis.global_.offheight = max(480, env.model.vis.global_.offheight)
            try:
                report['video_frames'] = render_video(env, frames, folder / 'preview.mp4', title=f'A3 | {mode}')
            except (RuntimeError, OSError, ValueError) as exc:
                report['video_error'] = str(exc)
            write_json(folder / 'report.json', report)
        print(f"{mode} seed {seed}: {'pass' if report['success'] else 'fail'} "
              f"{report['score']['target_count']}/{report['score']['expected_count']} "
              f"{report.get('rejected', '')}", flush=True)
        return report
    finally:
        env.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--a3-root', type=Path, help='Existing a3_dual_arm_sim source checkout')
    parser.add_argument('--geometry', choices=('truth', 'rgbd', 'both'), required=True)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--seed', type=int, action='append', help='Repeat to use more seeds (default: 11)')
    parser.add_argument('--randomize-boxes', action='store_true')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--video', action='store_true')
    args = parser.parse_args()
    if args.a3_root:
        src = args.a3_root.resolve() / 'src'
        if not (src / 'a3_dual_arm_sim' / 'tasks' / 'cookie_2x10.py').is_file():
            parser.error('--a3-root must contain src/a3_dual_arm_sim/tasks/cookie_2x10.py')
        sys.path.insert(0, str(src))
    try:
        from a3_dual_arm_sim.tasks.cookie_2x10 import A3Cookie2x10Env
    except ModuleNotFoundError as exc:
        if exc.name == 'a3_dual_arm_sim':
            parser.error('Install a3_dual_arm_sim or supply --a3-root')
        raise

    from .behavior import parse_plan

    seeds = args.seed or [11]
    if len(seeds) != len(set(seeds)):
        parser.error('Use each seed once')
    modes = ('truth', 'rgbd') if args.geometry == 'both' else (args.geometry,)
    plan = json.loads(args.plan.read_text(encoding='utf-8-sig'))
    try:
        intents = parse_plan(plan)
    except ValueError as exc:
        parser.error(str(exc))

    cv2.setNumThreads(1)
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(args.output / 'plan.json', plan)
    rows = []
    summary = dict(seeds=seeds, geometry=list(modes), randomize_boxes=args.randomize_boxes, trials=rows)
    for seed in seeds:
        for mode in modes:
            directory = args.output / f'seed-{seed}' / mode
            rows.append(trial(directory, mode, seed, intents, args.randomize_boxes, args.video))
            summary['success'] = {
                kind: {'passed': sum(row['success'] for row in rows if row['geometry'] == kind),
                       'attempted': sum(row['geometry'] == kind for row in rows)}
                for kind in modes
            }
            write_json(args.output / 'summary.json', summary)
    return 0 if all(row['success'] for row in rows) else 1


if __name__ == '__main__':
    sys.exit(main())
