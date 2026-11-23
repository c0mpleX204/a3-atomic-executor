"""Explicit, single-tick A3 geometry/actuator commands.

Pose: world [x, y, z, qw, qx, qy, qz], meters. Opening: 0 closed, 1 open.
Bind an already-reset A3 environment. No background execution or command queue.
The servo math is copied from A3CookieBatchExpert._servo; original files stay intact.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import mujoco
import numpy as np

from a3_dual_arm_sim.contracts import LEFT_JOINTS, RIGHT_JOINTS


@dataclass(frozen=True)
class CommandResult:
    status: Literal['running', 'reached', 'failure', 'holding', 'stopped']
    eef_pose: np.ndarray
    gripper_opening: float
    control_steps: int
    position_error_m: float | None = None
    rotation_error_rad: float | None = None
    reason: str | None = None


class A3Executor:
    """Execute only the current explicit command; all decisions stay with the caller.

    Each mutating call advances at most one 20 Hz simulation tick. Repeat the
    desired command from the external observation loop. A new command replaces
    the old budget; nothing runs between calls. Reached is geometric, never a
    statement about contact, holding an object, or completion of a larger goal.
    """

    def __init__(self, env, arm: Literal['L', 'R'] = 'L'):
        if arm not in ('L', 'R'):
            raise ValueError("arm must be 'L' or 'R'")
        self.env = env
        self.model, self.data = env.model, env.data
        self.arm = arm
        self._arm_index = 0 if arm == 'L' else 1
        self._action_offset = 7 * self._arm_index
        self._joint_offset = 8 * self._arm_index
        joints = LEFT_JOINTS if arm == 'L' else RIGHT_JOINTS
        self._qpos = np.array([env._qpos_ids[n] for n in joints])
        self._dofs = np.array([env._dof_ids[n] for n in joints])
        self._site = env._eef_sites[self._arm_index]
        self._work = mujoco.MjData(self.model)
        self._key = None
        self._steps = 0
        self._failed_reason = None

    @staticmethod
    def _pose(value):
        pose = np.asarray(value, dtype=np.float64).copy()
        if pose.shape != (7,) or not np.isfinite(pose).all():
            raise ValueError('target pose must be a finite 7-vector')
        norm = np.linalg.norm(pose[3:])
        if norm < 1e-12:
            raise ValueError('target quaternion has zero norm')
        pose[3:] /= norm
        if pose[3] < 0:
            pose[3:] *= -1
        return pose

    @staticmethod
    def _positive(value, name):
        if not np.isfinite(value) or value <= 0:
            raise ValueError(f'{name} must be finite and positive')
        return float(value)

    @staticmethod
    def _opening(value):
        value = float(value)
        if not np.isfinite(value) or not 0 <= value <= 1:
            raise ValueError('opening must be finite and in [0, 1]')
        return value

    def get_eef_pose(self):
        quat = np.empty(4)
        mujoco.mju_mat2Quat(quat, self.data.site_xmat[self._site])
        return np.r_[self.data.site_xpos[self._site].copy(), quat]

    def get_gripper_opening(self):
        return float(self.env.current_joint_action[self._joint_offset + 7])

    def _errors(self, target):
        actual = self.get_eef_pose()
        error = np.empty(3)
        mujoco.mju_subQuat(error, target[3:], actual[3:])
        return float(np.linalg.norm(target[:3] - actual[:3])), float(np.linalg.norm(error))

    def is_target_reached(self, target_eef_pose, *, position_tolerance=0.0025,
                          rotation_tolerance=0.045, velocity_tolerance=0.25):
        target = self._pose(target_eef_pose)
        for name, value in [('position_tolerance', position_tolerance),
                            ('rotation_tolerance', rotation_tolerance),
                            ('velocity_tolerance', velocity_tolerance)]:
            self._positive(value, name)
        pos, rot = self._errors(target)
        return bool(pos < position_tolerance and rot < rotation_tolerance
                    and np.linalg.norm(self.data.qvel[self._dofs]) < velocity_tolerance)

    def _result(self, status, *, target=None, reason=None):
        pos, rot = self._errors(target) if target is not None else (None, None)
        return CommandResult(status, self.get_eef_pose(), self.get_gripper_opening(),
                             self._steps, pos, rot, reason)

    def _begin(self, key, max_steps):
        if isinstance(max_steps, bool) or not isinstance(max_steps, int) or max_steps < 1:
            raise ValueError('max_steps must be a positive integer')
        if key != self._key:
            self._key, self._steps, self._failed_reason = key, 0, None
        if self._failed_reason:
            return self._failed_reason
        if self._steps >= max_steps:
            self._failed_reason = 'command_step_budget_exhausted'
            return self._failed_reason
        return None

    def _safety_reason(self):
        if self.env._safety_stop:
            return self.env._safety_reason or 'environment_safety_stop'
        if not np.isfinite(self.data.qpos[self._qpos]).all():
            return 'nonfinite_joint_state'
        return None

    def _neutral_delta(self, opening=None):
        action = np.zeros(14, dtype=np.float64)
        # Preserve both grippers and the inactive arm's last commanded targets.
        last = self.env.last_applied_action
        action[6], action[13] = 2 * last[7] - 1, 2 * last[15] - 1
        if opening is not None:
            action[self._action_offset + 6] = 2 * opening - 1
        return action

    def _step(self, action):
        # The environment owns the existing IK, physical simulation and limits.
        # Reward, scene observations, termination and all task info are ignored.
        self.env.step(action)
        self._steps += 1
        reason = self._safety_reason()
        if reason:
            self._failed_reason = reason
        return reason

    def _ik_residual_ok(self, target):
        self.env._ik._work.qpos[:] = self.data.qpos
        q = self.env.solve_ik(target[:3], target[3:],
                              self.env.last_applied_action[self._joint_offset:self._joint_offset + 7],
                              arm=self.arm)
        self._work.qpos[:] = self.data.qpos
        self._work.qpos[self._qpos] = q
        mujoco.mj_kinematics(self.model, self._work)
        quat, err = np.empty(4), np.empty(3)
        mujoco.mju_mat2Quat(quat, self._work.site_xmat[self._site])
        mujoco.mju_subQuat(err, target[3:], quat)
        return (np.linalg.norm(self._work.site_xpos[self._site] - target[:3]) < 0.0025
                and np.linalg.norm(err) < 0.045)

    def _servo(self, position, rotation, opening, speed=0.0024):
        """Copied Cartesian servo math; only robot binding and output mapping changed."""
        work = self._work
        work.qpos[:] = self.data.qpos
        for names, offset in ((LEFT_JOINTS, 0), (RIGHT_JOINTS, 8)):
            ids = [self.env._qpos_ids[n] for n in names]
            work.qpos[ids] = self.env.last_applied_action[offset:offset + 7]
        mujoco.mj_kinematics(self.model, work)
        cmd_pos = work.site_xpos[self._site].copy()
        target_pos = np.asarray(position, dtype=np.float64)
        pos_err = target_pos - cmd_pos
        dist = np.linalg.norm(pos_err)
        decel_dist = 0.020
        if dist < decel_dist and speed > 0.0020:
            effective_speed = max(0.0012, speed * (dist / decel_dist))
        else:
            effective_speed = speed
        step_dist = min(dist, effective_speed)
        delta_p = ((pos_err / dist) * (step_dist / self.env.config.cartesian_translation_scale_m)
                   if dist > 1e-6 else np.zeros(3))
        delta_p = np.clip(delta_p, -1.0, 1.0)
        target_q = np.asarray(rotation, dtype=np.float64)
        cur_q, rot_err = np.empty(4), np.empty(3)
        mujoco.mju_mat2Quat(cur_q, work.site_xmat[self._site])
        mujoco.mju_subQuat(rot_err, target_q, cur_q)
        rot_dist = np.linalg.norm(rot_err)
        max_rot = min(0.05, self.env.config.cartesian_rotation_scale_rad)
        step_rot = min(rot_dist, max_rot)
        delta_r = ((rot_err / rot_dist) * (step_rot / self.env.config.cartesian_rotation_scale_rad)
                   if rot_dist > 1e-6 else np.zeros(3))
        action = self._neutral_delta(opening)
        start = self._action_offset
        action[start:start + 3] = delta_p
        action[start + 3:start + 6] = np.clip(delta_r, -1.0, 1.0)
        return action

    def servo_to(self, target_eef_pose, *, opening=None, speed=0.0024, max_steps=240,
                 position_tolerance=0.0025, rotation_tolerance=0.045,
                 velocity_tolerance=0.25, check_ik=True):
        """One pose-control tick; speed is meters per tick (legacy convention).

        Repeating an identical target shares a finite step budget. A different
        target preempts immediately. Optional opening is also an explicit goal.
        IK preflight uses the original solver/seed, not a completeness guarantee.
        """
        try:
            target = self._pose(target_eef_pose)
            speed = self._positive(speed, 'speed')
            for name, val in [('position_tolerance', position_tolerance),
                              ('rotation_tolerance', rotation_tolerance),
                              ('velocity_tolerance', velocity_tolerance)]:
                self._positive(val, name)
            grip = (float(self.env.last_applied_action[self._joint_offset + 7])
                    if opening is None else self._opening(opening))
            # An omitted gripper target means preserve its last command; it is
            # excluded from the key so rate limiting cannot reset pose budgets.
            key = ('pose', target.tobytes(), opening, speed)
            new = key != self._key
            reason = self._begin(key, max_steps) or self._safety_reason()
            if reason:
                return self._result('failure', target=target, reason=reason)
            if new and check_ik and not self._ik_residual_ok(target):
                self._failed_reason = 'ik_target_residual_exceeds_tolerance'
                return self._result('failure', target=target, reason=self._failed_reason)
            reason = self._step(self._servo(target[:3], target[3:], grip, speed))
            reached = self.is_target_reached(
                target, position_tolerance=position_tolerance,
                rotation_tolerance=rotation_tolerance, velocity_tolerance=velocity_tolerance)
            reached = reached and (opening is None or abs(self.get_gripper_opening() - grip) < 0.008)
            if reason:
                return self._result('failure', target=target, reason=reason)
            if not reached and self._steps >= max_steps:
                self._failed_reason = 'command_step_budget_exhausted'
                return self._result('failure', target=target, reason=self._failed_reason)
            return self._result('reached' if reached else 'running', target=target)
        except (ValueError, FloatingPointError) as exc:
            self._key, self._steps, self._failed_reason = None, 0, str(exc)
            return self._result('failure', reason=str(exc))

    def servo_delta(self, delta_xyz, delta_rot, *, opening=None):
        """One increment in m / rotation-vector rad, using existing MuJoCo convention.

        Components are clipped to the environment's delta limits. No increments
        are queued, integrated, or repeated by this executor after the call.
        """
        try:
            xyz, rot = np.asarray(delta_xyz, dtype=float), np.asarray(delta_rot, dtype=float)
            if xyz.shape != (3,) or rot.shape != (3,) or not np.isfinite(np.r_[xyz, rot]).all():
                raise ValueError('delta_xyz and delta_rot must be finite 3-vectors')
            grip = None if opening is None else self._opening(opening)
            self._key, self._steps, self._failed_reason = None, 0, None
            reason = self._safety_reason()
            if reason:
                return self._result('failure', reason=reason)
            action = self._neutral_delta(grip)
            start = self._action_offset
            action[start:start + 3] = np.clip(xyz / self.env.config.cartesian_translation_scale_m, -1, 1)
            action[start + 3:start + 6] = np.clip(rot / self.env.config.cartesian_rotation_scale_rad, -1, 1)
            reason = self._step(action)
            return self._result('failure' if reason else 'running', reason=reason)
        except (ValueError, FloatingPointError) as exc:
            return self._result('failure', reason=str(exc))

    def _gripper(self, opening, max_steps, tolerance):
        try:
            opening = self._opening(opening)
            tolerance = self._positive(tolerance, 'tolerance')
            reason = self._begin(('gripper', opening), max_steps) or self._safety_reason()
            if reason:
                return self._result('failure', reason=reason)
            reason = self._step(self._neutral_delta(opening))
            reached = abs(self.get_gripper_opening() - opening) < tolerance
            if not reached and self._steps >= max_steps:
                self._failed_reason = reason = 'gripper_step_budget_exhausted'
            return self._result('failure' if reason else 'reached' if reached else 'running', reason=reason)
        except (ValueError, FloatingPointError) as exc:
            return self._result('failure', reason=str(exc))

    def gripper_open(self, *, opening=1.0, max_steps=100, tolerance=0.008):
        """One opening-control tick. No EEF movement or subsequent operation."""
        return self._gripper(opening, max_steps, tolerance)

    def gripper_close(self, *, opening=0.0, max_steps=100, tolerance=0.008):
        """One closing-control tick. Reached means opening reached, not grasp success."""
        return self._gripper(opening, max_steps, tolerance)

    def hold(self):
        """Cancel the prior command and preserve both arms/grippers' commands for one tick."""
        self._key, self._steps, self._failed_reason = None, 0, None
        reason = self._safety_reason()
        if not reason:
            reason = self._step(self._neutral_delta())
        return self._result('failure' if reason else 'holding', reason=reason)

    def stop(self):
        """Cancel motion and hold for one tick; a later explicit command may resume.

        This is a commanded hold, not a torque-off or hardware emergency stop.
        """
        result = self.hold()
        if result.status == 'failure':
            return result
        return self._result('stopped')
