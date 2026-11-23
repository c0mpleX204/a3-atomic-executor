"""Opt-in, synchronous MuJoCo RGB-D sensor; no object poses or segmentation.

Capture only while the environment is idle (the simulator is not thread-safe).
The returned observation contains arrays, never a model/data handle.
"""
from __future__ import annotations

import mujoco
import numpy as np

from dataclasses import dataclass


@dataclass(frozen=True)
class RGBDObservation:
    frame_id: str
    camera_id: str
    timestamp_s: float
    depth_m: np.ndarray
    K: np.ndarray
    camera_to_output: np.ndarray
    output_frame: str
    rgb: np.ndarray | None = None



class A3RGBDSensor:
    def __init__(self, model, data, *, width=256, height=256):
        if not isinstance(width, int) or not isinstance(height, int) or min(width, height) < 2:
            raise ValueError("width/height must be integers >= 2")
        self.model, self.data = model, data
        self.width, self.height = width, height
        self._renderer = None
        self._sequence = 0

    def capture(self, camera_name="front") -> RGBDObservation:
        cid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name)
        if cid < 0:
            raise ValueError(f"unknown camera: {camera_name}")
        # This adapter supports the A3 perspective/fovy cameras only. Refuse
        # alternative projection models rather than silently invent calibration.
        if self.model.cam_projection[cid] != mujoco.mjtProjection.mjPROJ_PERSPECTIVE or np.any(self.model.cam_sensorsize[cid] != 0):
            raise ValueError("only perspective fovy cameras are supported")
        mujoco.mj_forward(self.model, self.data)
        timestamp = float(self.data.time)
        transform = np.eye(4)
        transform[:3, :3] = self.data.cam_xmat[cid].reshape(3, 3) @ np.diag([1., -1., -1.])
        transform[:3, 3] = self.data.cam_xpos[cid]
        focal = .5 * self.height / np.tan(np.deg2rad(self.model.cam_fovy[cid]) / 2)
        K = np.array([[focal, 0, (self.width - 1) / 2],
                      [0, focal, (self.height - 1) / 2], [0, 0, 1.]])
        if self._renderer is None:
            # MSAA depth resolves a subpixel sample rather than the nominal
            # pixel center on this GL backend. Use a private non-MSAA context
            # for registered geometric RGB-D; restore policy renderer settings.
            samples = self.model.vis.quality.offsamples
            buffer_width = self.model.vis.global_.offwidth
            buffer_height = self.model.vis.global_.offheight
            try:
                self.model.vis.quality.offsamples = 0
                self.model.vis.global_.offwidth = max(buffer_width, self.width)
                self.model.vis.global_.offheight = max(buffer_height, self.height)
                self._renderer = mujoco.Renderer(self.model, height=self.height, width=self.width)
            finally:
                self.model.vis.quality.offsamples = samples
                self.model.vis.global_.offwidth = buffer_width
                self.model.vis.global_.offheight = buffer_height
        renderer = self._renderer
        renderer.disable_segmentation_rendering()
        renderer.disable_depth_rendering()
        renderer.update_scene(self.data, camera=cid)
        rgb = renderer.render().copy()
        try:
            renderer.enable_depth_rendering()
            renderer.update_scene(self.data, camera=cid)
            depth = renderer.render().copy()
        finally:
            renderer.disable_depth_rendering()
        # Renderer already returns top-left origin and metric optical Z, not
        # normalized OpenGL depth. Far-plane pixels are missing observations.
        far = float(self.model.vis.map.zfar * self.model.stat.extent)
        depth = np.asarray(depth, dtype=np.float32)
        depth[~np.isfinite(depth) | (depth <= 0) | (depth >= far * .9999)] = np.nan
        self._sequence += 1
        return RGBDObservation(str(self._sequence), camera_name, timestamp, depth,
                               K, transform, "world", rgb)

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


def backproject_world(observation: RGBDObservation):
    """Return organized HxWx3 world points; invalid depth remains NaN."""
    height, width = observation.depth_m.shape
    v, u = np.mgrid[:height, :width]
    rays = np.stack([u, v, np.ones_like(u)], axis=-1) @ np.linalg.inv(observation.K).T
    camera_points = rays * observation.depth_m[..., None]
    T = observation.camera_to_output
    return camera_points @ T[:3, :3].T + T[:3, 3]
