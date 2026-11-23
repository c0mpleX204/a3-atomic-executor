"""A fixed physical pick-and-place action. Batch selection belongs to its caller."""
from __future__ import annotations

from dataclasses import dataclass
from numbers import Real

import numpy as np
from scipy.spatial.transform import Rotation

from .arm import Rejected
from .geometry import UP, resolve_source


@dataclass(frozen=True)
class BatchIntent:
    count: int
    source_end_pixel_1000: tuple[float, float]
    target_center_pixel_1000: tuple[float, float]
    target_column: int
    first_row: int


def pixel(value):
    if (not isinstance(value, list) or len(value) != 2
            or any(isinstance(x, bool) or not isinstance(x, Real) or not 0 <= x <= 1000 for x in value)):
        raise ValueError('Pixel hints must have two coordinates in [0, 1000]')
    return tuple(float(x) for x in value)


def parse_plan(plan):
    if not isinstance(plan, dict) or set(plan) != {'batches'} or not isinstance(plan['batches'], list):
        raise ValueError('Plan must contain a batches array')
    if not 1 <= len(plan['batches']) <= 20:
        raise ValueError('Plan must contain between 1 and 20 batches')
    fields = set(BatchIntent.__dataclass_fields__)
    intents, used = [], set()
    for batch in plan['batches']:
        if not isinstance(batch, dict) or set(batch) != fields:
            raise ValueError(f'Expected exactly these batch fields: {", ".join(sorted(fields))}')
        count, col, row = batch['count'], batch['target_column'], batch['first_row']
        if type(count) is not int or not 1 <= count <= 10:
            raise ValueError('Batch count must be an integer in [1, 10]')
        if type(col) is not int or col not in (0, 1) or type(row) is not int or not 0 <= row <= 10 - count:
            raise ValueError('Target column or row span is out of bounds')
        slots = {(col, i) for i in range(row, row + count)}
        if used.intersection(slots):
            raise ValueError('Two batches request overlapping target locations')
        used.update(slots)
        intents.append(BatchIntent(count, pixel(batch['source_end_pixel_1000']),
                                   pixel(batch['target_center_pixel_1000']), col, row))
    return intents


class PickAndPlace:
    def __init__(self, arm, sensor, geometry, cookie_size_m, slots, emit):
        self.arm, self.sensor, self.geometry = arm, sensor, geometry
        self.size, self.emit = np.asarray(cookie_size_m), emit
        self.columns, self.rows = (np.unique(np.asarray(slots)[:, i]) for i in (0, 1))
        if len(self.columns) != 2 or len(self.rows) != 10:
            raise ValueError('The fixed executor requires the A3 2x10 target slot layout')

    def source_path(self, source):
        arm = self.arm
        pick = source.centers_m.mean(axis=0) + UP * (self.size[2] / 2 - .001)
        center_heights = (.076, .026, 0.)
        current = Rotation.from_quat(arm.executor.get_eef_pose()[3:], scalar_first=True)
        candidates = []
        for tilt in (0, -15, 15, -30, 30, -45, 45):
            direction = Rotation.from_rotvec(source.closing_axis * np.deg2rad(tilt)).apply(-UP)
            for sign in (1, -1):
                points = [arm.pose(pick + UP * height, sign * source.closing_axis, direction)
                          for height in center_heights]
                orientation = Rotation.from_quat(points[0][3:], scalar_first=True)
                turn = (current.inv() * orientation).magnitude()
                candidates.append((turn, tilt, points))
        for _, tilt, points in sorted(candidates, key=lambda entry: entry[0]):
            if all(arm.executor._ik_residual_ok(pose) for pose in points):
                return tilt, points
        raise Rejected('None of the sampled grasp orientations passed endpoint IK')

    def run(self, intent):
        arm = self.arm
        frame = self.sensor.capture('front')
        source = resolve_source(self.geometry, frame, np.asarray(intent.source_end_pixel_1000), intent.count, self.size)
        target = self.geometry.target_tray(frame, np.asarray(intent.target_center_pixel_1000))
        target_axis = target.axes[:, 1]
        row_center = float(self.rows[intent.first_row:intent.first_row + intent.count].mean())
        floor = target.floor_m + target.axes @ [self.columns[intent.target_column], row_center, 0.]
        tilt, (hover, align, descend) = self.source_path(source)
        # The A3 2F-85 model uses a normalized jaw command and a nominal 85 mm full gap.
        aperture = max(.015, (source.span_m - .0038) / .085)
        if aperture > 1:
            raise Rejected('Measured group is too wide for the gripper')
        self.emit(kind='geometry', frame_id=frame.frame_id, target_column=intent.target_column,
                  source_centers_m=source.centers_m.tolist(), target_floor_point_m=floor.tolist(),
                  closing_axis_world=source.closing_axis.tolist(), approach_tilt_deg=tilt,
                  pregrasp_aperture=aperture, grasp_eef_pose=descend.tolist())

        arm.open(aperture)
        arm.move('approach', hover, unheld=True)
        arm.move('align', align, unheld=True)
        arm.move('descend', descend, speed=.001, tolerance=.0035, unheld=True)
        grip = arm.close(aperture, target_force=1.5)
        lift = arm.executor.get_eef_pose().copy()
        lift[2] += .052
        arm.move('lift', lift, speed=.0016, minimum_hold_force=.35)

        rotation = Rotation.from_quat(lift[3:], scalar_first=True)
        carry = np.r_[floor + UP * .070 - rotation.apply(arm.pad_midpoint_local), lift[3:]]
        carry[2] = max(carry[2], lift[2])
        arm.move('carry', carry, minimum_hold_force=.35)
        jaw = rotation.apply(arm.pad_basis_local[:, 0])
        if target_axis @ jaw < 0:
            target_axis = -target_axis
        yaw = np.arctan2(np.cross(jaw, target_axis)[2], target_axis @ jaw)
        aligned = Rotation.from_rotvec(UP * yaw) * rotation
        offset = aligned.apply(arm.pad_midpoint_local)
        quat = aligned.as_quat(scalar_first=True)
        above = np.r_[floor + UP * .075 - offset, quat]
        release = np.r_[floor + UP * (self.size[2] + .002) - offset, quat]
        arm.move('align_target', above, speed=.002, minimum_hold_force=.35)
        arm.move('lower', release, speed=.0015, minimum_hold_force=.35)
        arm.open(min(1., grip + .013 / .085))
        retreat = arm.executor.get_eef_pose().copy()
        retreat[2] = above[2]
        arm.move('retreat', retreat)
        for _ in range(28):
            arm.tick(arm.executor.hold(), 'settle')
        self.emit(kind='unit_completed', force_n=arm.forces().tolist())
