"""Target grounding from either calibrated RGB-D or simulator reference geometry."""
from dataclasses import dataclass
from collections import Counter

import cv2
import numpy as np

from .arm import Rejected
from .detector import AdaptiveCookieRGBD
from .observation import backproject_world

UP = np.array([0., 0., 1.])


def project_pixel(observation, world_point):
    point = np.asarray(world_point, dtype=float)
    if point.shape != (3,) or not np.isfinite(point).all():
        raise ValueError('Projection requires a finite world point')
    T = observation.camera_to_output
    camera_point = T[:3, :3].T @ (point - T[:3, 3])
    if camera_point[2] <= 0:
        return None
    pixel = observation.K @ camera_point
    return (pixel[:2] / pixel[2]).tolist()


def pixel_1000(frame, point):
    image_xy = project_pixel(frame, point)
    if image_xy is None:
        raise Rejected('observed_geometry_outside_view')
    return np.asarray(image_xy) * 1000 / (np.array(frame.rgb.shape[1::-1]) - 1)


def roi_bounds(frame, hint, radius):
    normalized = np.clip(np.r_[hint - radius, hint + radius], 0, 1000)
    h, w = frame.rgb.shape[:2]
    return np.rint(normalized * [w - 1, h - 1, w - 1, h - 1] / 1000).astype(int)


def candidate_groups(frame, poses, size, hint, count):
    if len(poses) == 0:
        return []
    poses = np.asarray(poses)
    long = np.median(poses[:, :3, 0], axis=0)
    long -= UP * (long @ UP)
    norm = np.linalg.norm(long)
    if norm < 1e-6:
        raise Rejected('source_cookie_axis_is_ambiguous')
    long /= norm
    jaw = np.cross(UP, long)
    centers = poses[:, :3, 3]
    strips = []
    for position in sorted(centers, key=lambda p: p @ long):
        if strips and (position - strips[-1][0]) @ long < .010:
            strips[-1].append(position)
        else:
            strips.append([position])
    candidates = []
    for strip in strips:
        if len(strip) < count:
            continue
        ordered = sorted(strip, key=lambda p: p @ jaw)
        for group in (ordered[:count], ordered[::-1][:count]):
            group = np.asarray(group)
            gaps = np.diff(np.sort(group @ jaw))
            if (gaps > size[1] * 2.05).any() or (gaps < size[1] * .7).any():
                continue
            thickness = size[1]
            span = float(np.ptp(group @ jaw) + thickness)
            if span - .0038 > .085:
                continue
            anchor = pixel_1000(frame, group[0] + UP * size[2] / 2)
            distance = float(np.linalg.norm(anchor - hint))
            if distance <= 120:
                candidates.append({'anchor_1000': anchor.tolist(), 'distance_1000': distance,
                                   'centers_world_m': group.tolist(), 'jaw_axis_world': jaw.tolist(),
                                   'span_m': span, 'neighbor_spacing_m': gaps.tolist(),
                                   })
    candidates.sort(key=lambda group: group['distance_1000'])
    return candidates


def estimate_tray(frame, hint):
    """Fit the visible rim footprint and a distinct lower interior plane."""
    x0, y0, x1, y1 = roi_bounds(frame, hint, np.array([155, 160]))
    image = frame.rgb[y0:y1, x0:x1].astype(np.int16)
    neutral = ((image.max(axis=2) - image.min(axis=2) < 35) & (image.min(axis=2) > 65))
    _, labels, stats, _ = cv2.connectedComponentsWithStats(neutral.astype(np.uint8), connectivity=8)
    world = backproject_world(frame)[y0:y1, x0:x1]
    proposals = []
    for label in range(1, len(stats)):
        if stats[label, cv2.CC_STAT_AREA] < 600:
            continue
        points = world[(labels == label) & np.isfinite(world).all(axis=2)]
        if len(points) < 600:
            continue
        levels = np.round(points[:, 2] / .0005).astype(int)
        height_counts = Counter(levels)
        for level, hits in height_counts.most_common(4):
            rim = points[levels == level]
            if hits < 500:
                continue
            center_xy, sides, angle = cv2.minAreaRect(rim[:, :2].astype(np.float32))
            long_size, short_size = max(sides), min(sides)
            if not (.084 <= long_size <= .16 and .056 <= short_size <= .125):
                continue
            long = np.deg2rad(angle + (90 if sides[1] > sides[0] else 0))
            long_axis = np.array([np.cos(long), np.sin(long), 0.])
            if long_axis[0] < 0:
                long_axis *= -1
            short_axis = np.cross(UP, long_axis)
            xy = points[:, :2] - center_xy
            below = ((level - levels >= 16) & (level - levels <= 75)
                     & (np.abs(xy @ long_axis[:2]) < long_size * .38)
                     & (np.abs(xy @ short_axis[:2]) < short_size * .38))
            if below.sum() < 500:
                continue
            lower, floor_hits = Counter(levels[below]).most_common(1)[0]
            if floor_hits < 350:
                continue
            floor_center = np.r_[center_xy, float(np.median(points[below & (levels == lower), 2]))]
            px = pixel_1000(frame, floor_center)
            proposals.append((float(np.linalg.norm(px - hint)), -hits, floor_center,
                              long_axis, long_size, short_size))
    if not proposals:
        raise Rejected('could_not_measure_target_floor_and_tray_axes_from_rgbd')
    _, _, center, long, lw, sw = min(proposals, key=lambda row: row[:2])
    return center, long, lw, sw


@dataclass(frozen=True)
class SourceGroup:
    centers_m: np.ndarray
    closing_axis: np.ndarray
    span_m: float


@dataclass(frozen=True)
class TargetTray:
    floor_m: np.ndarray
    axes: np.ndarray


class RGBDGeometry:
    """Estimates geometry from calibrated frames without a simulator handle."""

    def __init__(self, cookie_size_m):
        self.detector = AdaptiveCookieRGBD(full_size_m=cookie_size_m)

    def source_poses(self, frame, hint):
        observation = self.detector.estimate(
            frame.rgb, frame.depth_m, frame.K, frame.camera_to_output,
            roi=roi_bounds(frame, hint, np.array([240, 180])),
        )
        return observation.poses_world

    def target_tray(self, frame, hint):
        floor, long_axis, _, _ = estimate_tray(frame, hint)
        return TargetTray(floor, np.column_stack((long_axis, np.cross(UP, long_axis), UP)))


class TruthGeometry:
    """Reads the simulator's present object geometry through the same interface."""

    def __init__(self, env):
        self.env = env

    def source_poses(self, frame, hint):
        env = self.env
        bounds = roi_bounds(frame, hint, np.array([240, 180]))
        poses = []
        for index, body in enumerate(env._cookie_bodies):
            if not env.privileged_cookie_in_source(index):
                continue
            center = env.data.xpos[body]
            uv = project_pixel(frame, center + UP * env.COOKIE_HALF_SIZE[2])
            if uv is None or not np.all((bounds[:2] <= uv) & (uv < bounds[2:])):
                continue
            pose = np.eye(4)
            pose[:3, :3] = env.data.xmat[body].reshape(3, 3)
            pose[:3, 3] = center
            poses.append(pose)
        return poses

    def target_tray(self, frame, hint):
        env = self.env
        body = env._target_bin_body
        axes = env.data.xmat[body].reshape(3, 3).copy()
        config = env.config.cookie_transfer
        top_of_floor = config.target_floor_z_m + config.bin_wall_thickness_m / 2
        floor = env.data.xpos[body] + top_of_floor * axes[:, 2]
        return TargetTray(floor, axes)


def resolve_source(geometry, frame, hint, count, size):
    options = candidate_groups(frame, geometry.source_poses(frame, hint), size, hint, count)
    if not options:
        raise Rejected('Could not find the requested contiguous group near the source pixel')
    selected = options[0]
    return SourceGroup(
        np.asarray(selected['centers_world_m']), np.asarray(selected['jaw_axis_world']), selected['span_m'],
    )
