"""Capture the Planner's target depth and the saved post-grasp A3 observation."""

import argparse
import json
from pathlib import Path

import numpy as np

from a3_atomic_executor.observation import A3RGBDSensor, backproject_world
from .replay import open_scene, restore


CAMERAS = ('front', 'left_wrist', 'right_wrist')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('a3-root', 'fixture', 'output'):
        parser.add_argument(f'--{name}', type=Path, required=True)
    args = parser.parse_args()
    with open_scene(args.a3_root, args.fixture) as (env, replay):
        from a3_atomic_executor.arm_executor import A3Executor

        intent = replay['planner_intent']
        count = intent['count']
        hint = np.asarray(intent['target_center_pixel_1000'], dtype=float)
        if hint.shape != (2,) or not np.all((hint >= 0) & (hint <= 1000)):
            raise ValueError('Expected a normalized target pixel')
        with A3RGBDSensor(env.model, env.data, width=640, height=480) as sensor:
            initial = sensor.capture('front')
            h, w = initial.depth_m.shape
            x, y = np.rint(hint * np.array([w - 1, h - 1]) / 1000).astype(int)
            anchor = backproject_world(initial)[y, x]
            if not np.isfinite(anchor).all():
                raise ValueError('Planner target pixel has no observed depth')
            restore(env, args.fixture / 'stopped-state.npz')
            tcp = A3Executor(env)
            views = [sensor.capture(name) for name in CAMERAS]
            observation = {f'{view.camera_id}_{key}': value for view in views for key, value in (
                ('rgb', view.rgb), ('depth_m', view.depth_m), ('K', view.K),
                ('camera_to_world', view.camera_to_output),
                ('world_points', backproject_world(view).astype(np.float32)),
            )}
            observation.update(target_world_m=anchor, current_tcp_world_wxyz=tcp.get_eef_pose(),
                               jaw_opening_m=tcp.get_gripper_opening() * .085,
                               instruction=f'Put the {count} held cookies upright into the indicated slots.',
                               planner_intent_json=json.dumps(intent), camera_names=np.asarray(CAMERAS),
                               scene=replay['scenario'], seed=replay['seed'], count=count,
                               target_pixel_1000=intent['target_center_pixel_1000'],
                               observation_sim_seconds=views[0].timestamp_s)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **observation)
    print(args.output)


if __name__ == '__main__':
    main()
