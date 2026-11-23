"""Render recorded A3 simulation states in a separate MuJoCo data instance."""

def render_video(env, frames, path, *, title='A3 atomic executor'):
    import av
    import cv2
    import mujoco as mj
    import numpy as np
    from PIL import Image

    data = mj.MjData(env.model)
    detail = mj.MjvCamera()
    detail.type = mj.mjtCamera.mjCAMERA_FREE
    detail.distance, detail.azimuth, detail.elevation = .31, 135, -32
    site = env._eef_sites[0]
    with mj.Renderer(env.model, height=480, width=640) as renderer, av.open(str(path), mode='w') as file:
        stream = file.add_stream('libx264', rate=10)
        stream.width, stream.height, stream.pix_fmt = 1280, 528, 'yuv420p'
        stream.options = {'crf': '19', 'preset': 'veryfast'}
        for number, (state, phase) in enumerate(frames):
            # Play the accepted physical trajectory in a separate rendering
            # state. Rejected branches remain recorded in events.json.
            data.qpos[:] = state
            mj.mj_forward(env.model, data)
            renderer.update_scene(data, camera='front')
            front = renderer.render().copy()
            detail.lookat[:] = data.site_xpos[site] + [0, 0, -.04]
            renderer.update_scene(data, camera=detail)
            frame = np.zeros((528, 1280, 3), np.uint8)
            frame[48:, :640], frame[48:, 640:] = front, renderer.render()
            cv2.putText(frame, f'{title} | {phase} | {number / 10:.1f} s',
                        (12, 30), cv2.FONT_HERSHEY_SIMPLEX, .63, (255, 220, 0), 1)
            for packet in stream.encode(av.VideoFrame.from_ndarray(frame, format='rgb24')):
                file.mux(packet)
            if number in (0, len(frames) - 1):
                Image.fromarray(frame).save(path.parent / ('video-first.png' if number == 0 else 'video-final.png'))
        for packet in stream.encode(None):
            file.mux(packet)
    with av.open(str(path)) as file:
        decoded = sum(1 for _ in file.decode(video=0))
    if decoded != len(frames):
        raise RuntimeError('Video does not decode to the recorded frame count')
    return decoded
