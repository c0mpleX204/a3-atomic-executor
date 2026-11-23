"""A3 mechanical interface and shared force-guarded commands."""

import mujoco as mj
import numpy as np

from .arm_executor import A3Executor


class Rejected(RuntimeError):
    pass


def vector(value, n=3):
    value = np.asarray(value, dtype=float)
    if value.shape != (n,) or not np.isfinite(value).all():
        raise ValueError(f'Expected finite {n}-vector')
    return value


def basis(closing, approach):
    x, y = vector(closing), vector(approach)
    if min(np.linalg.norm(x), np.linalg.norm(y)) < 1e-8:
        raise ValueError('Axis direction has zero length')
    x = x / np.linalg.norm(x)
    y = y / np.linalg.norm(y)
    y = y - x * (x @ y)
    if np.linalg.norm(y) < .1:
        raise ValueError('Closing and approach axes must be independent')
    y /= np.linalg.norm(y)
    return np.column_stack((x, y, np.cross(x, y)))


class RobotOnlyView:
    """Task/property access is blocked; model/data are needed for existing FK/IK.

    This is an access audit, not a security sandbox: raw MuJoCo arrays still
    contain scene physics. Unmodified A3Executor only indexes robot kinematics.
    """
    ALLOWED = {'model', 'data', 'config', '_qpos_ids', '_dof_ids', '_eef_sites',
               '_ik', 'last_applied_action', 'current_joint_action', 'solve_ik',
               'step', '_safety_stop', '_safety_reason'}

    def __init__(self, env):
        self._wrapped = env
        self.accesses = set()

    def __getattr__(self, name):
        if name not in self.ALLOWED:
            raise AssertionError(f'executor requested nonmechanical property: {name}')
        self.accesses.add(name)
        if name == 'step':
            # The simulator may compute task truth internally, but its returned
            # scene observations/reward/info never cross the executor boundary.
            def step_without_task_result(action):
                self._wrapped.step(action)
                return None
            return step_without_task_result
        return getattr(self._wrapped, name)


class SensorArm:
    def __init__(self, executor, pad_midpoint_local, pad_basis_local, forces, on_tick):
        self.executor = executor
        self.pad_midpoint_local = pad_midpoint_local
        self.pad_basis_local = pad_basis_local
        self.forces = forces
        self.on_tick = on_tick
        self.trace = []

    def pose(self, center, jaw, approach):
        rotation = basis(jaw, approach) @ self.pad_basis_local.T
        quat = np.empty(4)
        mj.mju_mat2Quat(quat, rotation.ravel())
        return np.r_[center - rotation @ self.pad_midpoint_local, quat]

    def tick(self, result, phase):
        force = self.forces()
        self.trace.append({'phase': phase, 'status': result.status, 'opening': result.gripper_opening,
                           'finger_force_N': force.tolist(), 'eef_pose': result.eef_pose.tolist()})
        self.on_tick(len(self.trace), phase)
        if result.status == 'failure':
            raise Rejected(f'{phase}:{result.reason}')
        return force

    def move(self, phase, pose, opening=None, *, speed=.0035, unheld=False, tolerance=.0025,
             minimum_hold_force=None):
        low_force_steps = 0
        while True:
            result = self.executor.servo_to(pose, opening=opening, speed=speed, max_steps=140,
                                            position_tolerance=tolerance)
            force = self.tick(result, phase)
            if unheld and np.max(force) > 1.0:
                raise Rejected(f'{phase}:unexpected_finger_force_before_closure')
            if minimum_hold_force is not None:
                low_force_steps = low_force_steps + 1 if np.min(force) < minimum_hold_force else 0
                if low_force_steps >= 3:
                    raise Rejected(f'{phase}:bilateral_hold_force_lost')
            if result.status == 'reached':
                return result

    def close(self, opening, *, target_force=1.25):
        stable = 0
        for _ in range(90):
            if stable == 0:
                opening = max(0., opening - .005)
            result = self.executor.gripper_close(opening=opening, max_steps=100)
            force = self.tick(result, 'force_closure')
            if np.max(force) > 6.:
                raise Rejected('force_closure:force_limit')
            stable = stable + 1 if np.min(force) >= target_force else 0
            if stable == 6:
                return opening
        raise Rejected('force_closure:bilateral_force_not_sustained')

    def open(self, opening):
        for _ in range(35):
            result = self.executor.gripper_open(opening=opening, max_steps=36)
            self.tick(result, 'release')
            if result.status == 'reached':
                return
        raise Rejected('release:opening_not_reached')


def make_robot(env, on_tick):
    executor = A3Executor(RobotOnlyView(env))
    model, data = env.model, env.data
    site = executor._site
    transform = data.site_xmat[site].reshape(3, 3).T
    pads = data.geom_xpos[list(env._left_finger_geoms)]
    midpoint = pads.mean(axis=0)
    base = model.geom('L_2f85_base_collision').id
    pad_basis = basis(transform @ (pads[1] - pads[0]),
                      transform @ (midpoint - data.geom_xpos[base]))
    return SensorArm(executor, transform @ (midpoint - data.site_xpos[site]), pad_basis,
                    lambda: np.array([env._sensor(f'L_finger_{pad}_touch_sensor')[0]
                                      for pad in ('inner', 'outer')]), on_tick)
