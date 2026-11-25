"""Replay one stored CLAP move-and-hold prediction in A3 MuJoCo."""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from a3_atomic_executor.video import render_video
from .replay import ReplayMotion, open_scene, restore, save_state


def execute(env, replay, prediction, output):
    frames = [(env.data.qpos.copy(), 'restored hold')]

    def record(tick, phase):
        if tick % 2 == 0:
            frames.append((env.data.qpos.copy(), phase))

    motion = ReplayMotion(env, replay['held_cookies'], record)
    tcp = motion.arm.executor.get_eef_pose()
    if not np.allclose(tcp, prediction['current_a3_tcp_world_wxyz'], atol=1e-6, rtol=0):
        raise ValueError('Restored TCP differs from the prediction input')
    grip = float(env.last_applied_action[7])
    opening = float(env.current_joint_action[7])
    counts = motion.target_counts()
    held = motion.attachments()
    center = np.asarray(prediction['target_position_world_m'])
    rotation = Rotation.from_quat(prediction['target_clap_tool_quaternion_xyzw']).as_matrix()
    goal = motion.arm.pose(center, rotation[:, 1], rotation[:, 2])
    above = goal.copy()
    above[2] = max(tcp[2], goal[2] + .066)
    preflight, phases = {}, []
    start = env.data.qpos[motion.qids].copy()
    for label, pose in (('approach', above), ('predicted pose', goal)):
        try:
            joints = motion.ik(pose, start)
            motion.check_segment(start, joints, opening, held)
        except motion.rejected as exc:
            preflight[label] = str(exc)
            break
        preflight[label] = 'pass'
        phases.append((label, pose, joints))
        start = joints

    execution_error = None
    try:
        for label, pose, joints in phases:
            motion.move(pose, joints, grip, label)
            if max(np.linalg.norm(actual[0] - initial[0])
                   for actual, initial in zip(motion.attachments(), held, strict=True)) > .017:
                raise motion.rejected('object_slipped_during_group_carry')
    except motion.rejected as exc:
        execution_error = str(exc)
    actual = motion.arm.executor.get_eef_pose()
    actual_pad = actual[:3] + Rotation.from_quat(actual[3:], scalar_first=True).apply(
        motion.arm.pad_midpoint_local)
    rotation_error = (Rotation.from_quat(goal[3:], scalar_first=True).inv() *
                      Rotation.from_quat(actual[3:], scalar_first=True)).magnitude()
    reached = (not execution_error and len(phases) == 2 and
               np.linalg.norm(actual[:3] - goal[:3]) < .0035 and rotation_error < .04)
    if reached:
        status = 'predicted pose reached'
    elif execution_error:
        status = execution_error
    elif len(phases) < 2:
        status = 'blocked: ' + next(reversed(preflight.values()))
    else:
        status = 'target tracking tolerance missed'
    frames.append((env.data.qpos.copy(), status))
    result = {
        'status': status, 'reached_prediction': reached, 'control_ticks': motion.ticks,
        'scene': replay['scenario'], 'seed': replay['seed'], 'preflight': preflight,
        'predicted_pad_center_world_m': center.tolist(),
        'clap_to_a3_assumption': 'Panda +Y to jaw axis, +Z to approach, tip at pad midpoint',
        'mapped_a3_tcp_world_wxyz': goal.tolist(), 'initial_tcp_world_wxyz': tcp.tolist(),
        'approach_tcp_world_wxyz': above.tolist(), 'final_tcp_world_wxyz': actual.tolist(),
        'final_pad_center_world_m': actual_pad.tolist(), 'gripper_open_class': 0, 'grip_command': grip,
        'pad_position_error_mm': float(1000 * np.linalg.norm(actual_pad - center)),
        'tcp_orientation_error_deg': float(np.rad2deg(rotation_error)),
        'relative_cookie_slip_mm': [float(1000 * np.linalg.norm(a[0] - b[0]))
                                   for a, b in zip(motion.attachments(), held, strict=True)],
        'finger_contacts': [list(map(bool, env.privileged_left_finger_contacts(i))) for i in motion.indices],
        'target_columns_before': counts, 'target_columns_after': motion.target_counts(),
    }
    output.mkdir(parents=True, exist_ok=False)
    save_state(env, output / 'final-state.npz')
    path = output / 'execution.json'
    path.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    framebuffer = env.model.vis.global_
    width, height = framebuffer.offwidth, framebuffer.offheight
    try:
        framebuffer.offwidth, framebuffer.offheight = max(640, width), max(480, height)
        result['video_frames'] = render_video(env, frames, output / 'execution.mp4', title='A3 | CLAP move-and-hold')
    finally:
        framebuffer.offwidth, framebuffer.offheight = width, height
    path.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(result, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('a3-root', 'fixture', 'prediction', 'output'):
        parser.add_argument(f'--{name}', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Choose a new output directory')
    prediction = json.loads(args.prediction.read_text(encoding='utf-8'))
    with open_scene(args.a3_root, args.fixture) as (env, replay):
        if (prediction['scene'] != replay['scenario'] or prediction['seed'] != replay['seed'] or
                prediction['planner_intent'] != replay['planner_intent']):
            raise ValueError('Prediction does not match the recorded plan and scene')
        if prediction['gripper_open_class'] != 0:
            raise ValueError('This replay accepts a move-and-hold prediction')
        restore(env, args.fixture / 'stopped-state.npz')
        execute(env, replay, prediction, args.output)


if __name__ == '__main__':
    main()
