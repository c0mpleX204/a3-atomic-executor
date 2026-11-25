"""Restore the recorded A3 grasp and run the same guarded joint interpolation."""

from contextlib import contextmanager
from dataclasses import replace
from hashlib import sha256
import json
from pathlib import Path
import sys

import mujoco as mj
import numpy as np


STATE = mj.mjtState.mjSTATE_INTEGRATION


@contextmanager
def open_scene(a3_root, fixture):
    source = a3_root.resolve() / 'src'
    sys.path.insert(0, str(source))
    from a3_dual_arm_sim.workflows.benchmark import CookieScenarioBenchmark
    import a3_dual_arm_sim

    package = source / 'a3_dual_arm_sim'
    if package != Path(a3_dual_arm_sim.__file__).resolve().parent:
        raise ValueError('The requested checkout did not supply the A3 simulator')
    replay = json.loads((fixture / 'replay.json').read_text(encoding='utf-8'))
    for name, digest in replay['simulator']['source_sha256'].items():
        if sha256((package / name).read_bytes()).hexdigest() != digest:
            raise ValueError(f'A3 revision mismatch in {name}')
    settings = replay['reset_options']
    benchmark = CookieScenarioBenchmark(
        replay['scenario'], profile=settings['randomization_profile'],
        randomize_boxes=settings['randomize_boxes'], randomize_cookies=settings['randomize_cookies'],
    )
    benchmark.config = replace(benchmark.config, horizon=6000)
    env = benchmark.create_env(render_cameras=False)
    try:
        env.reset(seed=replay['seed'], options=settings)
        yield env, replay
    finally:
        env.close()


def restore(env, path):
    with np.load(path, allow_pickle=False) as snapshot:
        mj.mj_setState(env.model, env.data, snapshot['state'], STATE)
        env._last_applied_action = snapshot['action'].copy()
        env._step_count = int(snapshot['env_steps'])
        env._success_hold_count = int(snapshot['success_hold_count'])
        env._scenario_hold_count = int(snapshot['official_hold_count'])
    env._safety_stop, env._safety_reason = False, None
    mj.mj_forward(env.model, env.data)


def save_state(env, path):
    state = np.empty(mj.mj_stateSize(env.model, STATE))
    mj.mj_getState(env.model, env.data, state, STATE)
    np.savez_compressed(path, state=state, action=env.last_applied_action,
                        env_steps=env._step_count, success_hold_count=env.success_hold_count,
                        official_hold_count=env._scenario_hold_count)


class ReplayMotion:
    def __init__(self, env, names, on_tick):
        from a3_atomic_executor.arm import make_robot, Rejected

        self.env, self.on_tick = env, on_tick
        self.model, self.data = env.model, env.data
        self.arm = make_robot(env, lambda *_: None)
        self.rejected = Rejected
        self.scratch = mj.MjData(self.model)
        self.qids = self.arm.executor._qpos
        self.site = self.arm.executor._site
        self.pads = set(env._left_finger_geoms)
        self.finger_qids = self.model.jnt_qposadr[list(env._finger_joints['L'])]
        self.bodies = tuple(self.model.body(name).id for name in names)
        self.indices = tuple(env._cookie_bodies.index(body) for body in self.bodies)
        self.arm_geoms = set()
        arm_bodies = {int(self.model.jnt_bodyid[self.model.joint(name).id]) for name in env._qpos_ids
                      if name.startswith('L_')}
        for geom, geom_body in enumerate(self.model.geom_bodyid):
            body = int(geom_body)
            while body and body not in arm_bodies:
                body = int(self.model.body_parentid[body])
            if body in arm_bodies:
                self.arm_geoms.add(geom)
        if not self.pads <= self.arm_geoms:
            raise RuntimeError('Left arm pads could not be identified')
        self.ticks = 0

    def ik(self, pose, seed):
        self.env._ik._work.qpos[:] = self.data.qpos
        joints = self.env.solve_ik(pose[:3], pose[3:], seed)
        self.scratch.qpos[:] = self.data.qpos
        self.scratch.qpos[self.qids] = joints
        mj.mj_kinematics(self.model, self.scratch)
        quat, error = np.empty(4), np.empty(3)
        mj.mju_mat2Quat(quat, self.scratch.site_xmat[self.site])
        mj.mju_subQuat(error, pose[3:], quat)
        if np.linalg.norm(self.scratch.site_xpos[self.site] - pose[:3]) > .0018 or np.linalg.norm(error) > .04:
            raise self.rejected('endpoint_ik')
        return joints

    def attachments(self):
        rotation = self.data.site_xmat[self.site].reshape(3, 3)
        return [(rotation.T @ (self.data.xpos[body] - self.data.site_xpos[self.site]),
                 rotation.T @ self.data.xmat[body].reshape(3, 3)) for body in self.bodies]

    def collision(self, data, penetration):
        for contact in data.contact:
            if contact.dist >= -penetration:
                continue
            a, b = int(contact.geom1), int(contact.geom2)
            a_body, b_body = int(self.model.geom_bodyid[a]), int(self.model.geom_bodyid[b])
            if a in self.arm_geoms or b in self.arm_geoms:
                pad_contact = (a in self.pads and b_body in self.bodies) or (b in self.pads and a_body in self.bodies)
                if pad_contact and contact.dist >= -.004:
                    continue
            elif (a_body in self.bodies) == (b_body in self.bodies):
                continue
            return '/'.join(self.model.geom(g).name or str(g) for g in (a, b))
        return None

    def check_segment(self, start, goal, opening, held):
        steps = max(2, int(np.ceil(np.max(np.abs(goal - start)) / .018)))
        for joints in np.linspace(start, goal, steps + 1)[1:]:
            self.scratch.qpos[:] = self.data.qpos
            self.scratch.qpos[self.qids] = joints
            self.scratch.qpos[self.finger_qids] = self.env.GRIPPER_RANGE_M * (1 - opening)
            mj.mj_kinematics(self.model, self.scratch)
            rotation = self.scratch.site_xmat[self.site].reshape(3, 3)
            for body, (offset, orientation) in zip(self.bodies, held, strict=True):
                addr = int(self.model.jnt_qposadr[self.model.body_jntadr[body]])
                self.scratch.qpos[addr:addr + 3] = self.scratch.site_xpos[self.site] + rotation @ offset
                mj.mju_mat2Quat(self.scratch.qpos[addr + 3:addr + 7], (rotation @ orientation).ravel())
            mj.mj_kinematics(self.model, self.scratch)
            mj.mj_comPos(self.model, self.scratch)
            mj.mj_collision(self.model, self.scratch)
            contact = self.collision(self.scratch, .00025)
            if contact:
                raise self.rejected('sampled_collision:' + contact)

    def step(self, joints, opening, phase):
        action = self.env.last_applied_action
        action[:7], action[7] = joints, opening
        _, _, terminated, truncated, info = self.env.step(action)
        self.ticks += 1
        self.on_tick(self.ticks, phase)
        if truncated or self.env._safety_stop or (terminated and not info.get('success')):
            raise self.rejected('environment_termination')
        contact = self.collision(self.data, .0015)
        if contact:
            raise self.rejected('physical_collision:' + contact)

    def move(self, pose, goal, opening, phase):
        start = self.env.last_applied_action[:7]
        steps = max(1, int(np.ceil(np.max(np.abs(goal - start)) / .018)))
        for joints in np.linspace(start, goal, steps + 1)[1:]:
            self.step(joints, opening, phase)
        for _ in range(18):
            if np.linalg.norm(self.data.site_xpos[self.site] - pose[:3]) < .0035:
                return
            self.step(goal, opening, phase)
        raise self.rejected('tracking_error')

    def target_counts(self):
        target = self.env._target_bin_body
        axis = self.data.xmat[target].reshape(3, 3)[:, 0]
        slots = np.unique(np.asarray(self.env.TARGET_SLOTS_LOCAL)[:, 0])
        counts = np.zeros(len(slots), dtype=int)
        for i in np.flatnonzero(self.env._active):
            if self.env._inside_bin(i, target):
                x = (self.data.xpos[self.env._cookie_bodies[i]] - self.data.xpos[target]) @ axis
                counts[np.argmin(np.abs(slots - x))] += 1
        return counts.tolist()
