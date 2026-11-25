"""Run the released CLAP fine policy on a saved A3 observation without simulating motion."""

import argparse
import ast
from importlib import import_module
from math import hypot
import json
import os
from pathlib import Path
import sys
from time import perf_counter
from types import ModuleType
from unittest.mock import patch

import numpy as np
from scipy.spatial.transform import Rotation
import torch
from transformers import AutoConfig, AutoModel, AutoTokenizer
import yaml


def render_points(pixels, depths, features, image_out, depth_out, blank_depth, blank_feature):
    """Use depth in millimeters and input order to resolve points at each pixel."""
    indices = torch.arange(pixels.numel(), device=pixels.device)
    valid = (pixels >= 0) & (pixels < depth_out.numel())
    empty = torch.iinfo(torch.long).max
    depth_mm = (depths[valid] * 1000).to(torch.long) & 0xffffffff
    # Bias the unsigned depth word so signed scatter-min preserves upstream ordering.
    packed = ((depth_mm - (1 << 31)) << 32) | indices[valid]
    winners = torch.full((depth_out.numel(),), empty, device=pixels.device, dtype=torch.long)
    winners.scatter_reduce_(0, pixels[valid], packed, reduce='amin')
    selected_depth = (winners >> 32) + (1 << 31)
    filled = selected_depth != 0xffffffff
    depth_out.fill_(blank_depth).view(-1)[filled] = selected_depth[filled].float() * .001
    feature_ids = (winners[filled] & 0xffffffff) % len(features)
    image_out.fill_(blank_feature).view(-1, features.shape[-1])[filled] = features[feature_ids]


def splat(batch, depth_in, image_in, depth_out, image_out, blank, focal, radius, kernel, ortho):
    """Use the nearest covering screen-space sample without crossing view boundaries."""
    height, width = depth_in.shape[0] // batch, depth_in.shape[1]
    depths = depth_in.view(batch, height, width)
    images = image_in.view(batch, height, width, -1)
    best = depth_out.view_as(depths)
    colors = image_out.view_as(images)
    best.copy_(depths)
    colors.copy_(images)
    half = kernel // 2
    for dx in range(-half, half + 1):
        for dy in range(-half, half + 1):
            dst = (slice(None), slice(max(0, -dy), min(height, height - dy)),
                   slice(max(0, -dx), min(width, width - dx)))
            src = (slice(None), slice(max(0, dy), min(height, height + dy)),
                   slice(max(0, dx), min(width, width + dx)))
            depth = depths[src]
            coverage = focal * radius if ortho else focal * radius / depth
            update = ((depth != blank) & (depth != 0) & (coverage > hypot(dx, dy)) &
                      ((best[dst] == blank) | (depth <= best[dst])))
            best[dst] = torch.where(update, depth, best[dst])
            colors[dst] = torch.where(update[..., None], images[src], colors[dst])


def load_upstream(upstream, device):
    # Keep upstream camera projections and MVT. Skip imports used only for GemBench training
    # and the bypassed coarse Qwen stage. The fine inference methods are unchanged.
    sys.path[:0] = [str(upstream), str(upstream / 'finegrained_policy'),
                    str(upstream / 'libs' / 'point-renderer')]
    try:
        if device.type == 'cpu':
            raise ImportError('The compiled CLAP renderer requires CUDA')
        import_module('point_renderer._C.render')
        renderer = 'upstream_cuda'
    except ImportError:
        package = ModuleType('point_renderer._C')
        backend = ModuleType('point_renderer._C.render')
        backend.render_feature_pointcloud_to_image = render_points
        backend.screen_space_splatting = splat
        package.render = backend
        sys.modules[package.__name__] = package
        sys.modules[backend.__name__] = backend
        renderer = 'torch_point_renderer'

    path = upstream / 'finegrained_policy' / 'model.py'
    source = path.read_text(encoding='utf-8')
    if device.type == 'cpu':
        box_renderer = import_module('point_renderer.rvt_renderer')
        timer = import_module('point_renderer.profiler').PerfTimer

        class CpuPointRenderer(box_renderer.PointRenderer):
            def __init__(self, device, perf_timer=False):
                self.device = device
                self.timer = timer(activate=perf_timer, show_memory=False, print_mode=perf_timer)

        box_renderer.PointRenderer = CpuPointRenderer
        source = source.replace('f"cuda:{rank}"', '"cpu"').replace("f'cuda:{rank}'", "'cpu'")
    module = ast.parse(source, filename=str(path))
    unused = {'finegrained_policy.dataset', 'swift.llm', 'swift.utils', 'utils.vlm_utils'}
    module.body = [node for node in module.body
                   if not (isinstance(node, ast.ImportFrom) and node.module in unused)
                   and not (isinstance(node, ast.Import) and [x.name for x in node.names] == ['clip'])]
    namespace = {'__name__': 'clap_inference_model', '__file__': str(path)}
    exec(compile(module, str(path), 'exec'), namespace)
    return namespace['MVT'], renderer


def load_checkpoint(args, mvt_class):
    config = yaml.safe_load(args.config.read_text(encoding='utf-8'))
    tokenizer = AutoTokenizer.from_pretrained(args.siglip_config, local_files_only=True)
    weights = torch.load(args.checkpoint, map_location='cpu', weights_only=True, mmap=True)['model_state']
    print(f'Loaded checkpoint index ({len(weights)} tensors)', flush=True)
    if not any(name.startswith('image_encoder.siglip.text_model.') for name in weights):
        raise ValueError('Official checkpoint is missing SigLIP text weights')
    backbone_config = AutoConfig.from_pretrained(args.siglip_config, local_files_only=True)
    hub_load = torch.hub.load

    def dino_from_checkpoint(*positional, **keywords):
        return hub_load(*positional, **keywords, pretrained=False)

    mvt_config = {**config['mvt'], 'xops': False}
    with patch.object(torch.hub, 'load', side_effect=dino_from_checkpoint), patch.object(
        AutoModel, 'from_pretrained', side_effect=lambda _: AutoModel.from_config(backbone_config),
    ):
        model = mvt_class(**mvt_config, debug=False, rank=0).to(args.device).eval()
    model.load_state_dict(weights, strict=True)
    return model, tokenizer


def decode_translation(model, logits):
    from point_renderer.rvt_ops import select_feat_from_hm

    # Match CLAP's full 224^3 search and sampling while limiting GPU peak memory.
    side = model.img_size
    grid = torch.linspace(-1 + 1 / side, 1 - 1 / side, side).to(logits.device)
    heatmaps = logits.reshape(model.num_img, 1, -1).softmax(-1).reshape(model.num_img, 1, side, side)
    best_score = torch.tensor(-1., device=logits.device)
    best_point = None
    for section in grid.split(4):
        points = torch.cartesian_prod(section, grid, grid)
        pixels = model.renderer.get_pt_loc_on_img(points[None], fix_cam=True)[0].transpose(0, 1)
        sampled, _, _ = select_feat_from_hm(pixels, heatmaps)
        score, index = sampled.mean(0).flatten().max(0)
        if score > best_score:
            best_score, best_point = score, points[index]
    if best_point is None:
        raise ValueError('CLAP translation heatmaps contain no valid candidate')
    return best_point[None]


def predict(args, model, tokenizer, observation):
    from utils import mvt_utils

    device = args.device
    cameras = observation['camera_names'].tolist()
    points = np.concatenate([observation[f'{camera}_world_points'].reshape(-1, 3) for camera in cameras])
    colors = np.concatenate([observation[f'{camera}_rgb'].reshape(-1, 3) for camera in cameras])
    lower, upper = np.array([-.3, -.5, .6]), np.array([.7, .5, 1.6])
    visible = np.isfinite(points).all(1) & (points >= lower).all(1) & (points <= upper).all(1)
    anchor = observation['target_world_m']
    if not visible.any() or not ((anchor >= lower) & (anchor <= upper)).all():
        raise ValueError('No valid observation or target inside the CLAP scene bounds')
    points = torch.as_tensor(points[visible], device=device, dtype=torch.float32)
    colors = torch.as_tensor(colors[visible], device=device, dtype=torch.float32) * (2 / 255) - 1
    pc, to_world = mvt_utils.place_pc_in_cube(points, with_mean_or_bounds=False,
                                              scene_bounds=np.r_[lower, upper].tolist())
    center = (lower + upper) / 2
    cube_scale = 2 / max(upper - lower)
    anchor_cube = torch.as_tensor((anchor - center) * cube_scale, device=device, dtype=torch.float32)[None]
    cropped, undo_zoom = mvt_utils.trans_pc([pc], anchor_cube, model.st_sca)
    image = model.render(cropped, [colors], img_aug=0, dyn_cam_info=None)
    text = str(observation['instruction'])
    tokens = tokenizer([text], padding='max_length', max_length=model.lang_len,
                       truncation=True, return_tensors='pt').to(device)
    text_features = model.image_encoder.siglip.text_model(**tokens).last_hidden_state.float()
    # RLBench uses [open bit, left prismatic m, right prismatic m, normalized step].
    # A3's coupled jaws use half the measured total gap in each slot; keyframe index is explicit.
    half_gap = np.clip(float(observation['jaw_opening_m']) / 2, 0., .04)
    progress = 1 - 2 * args.keyframe / (args.episode_keyframes - 1)
    robot_state = torch.tensor([[0., half_gap, half_gap, progress]], device=device, dtype=torch.float32)
    started = perf_counter()
    out = model.forward_stage_2(img=image, proprio=robot_state, step_tokens_embs=text_features,
                                step_lang_goal=text, lang_goal=text)
    position = to_world(undo_zoom(decode_translation(model, out['trans'])))[0].cpu().numpy()
    rotation_bins = [int(out[f'feat_{axis}'].argmax(1).item()) for axis in 'xyz']
    euler_deg = np.asarray(rotation_bins) * (360 / model.num_rot) - 180
    rotation_xyzw = Rotation.from_euler('xyz', euler_deg, degrees=True).as_quat()
    scores = out['feat_ex_rot'][0]
    if not np.isfinite(position).all() or not all(torch.isfinite(value).all() for value in (
        out['trans'], out['feat_x'], out['feat_y'], out['feat_z'], scores,
    )):
        raise ValueError('CLAP produced non-finite outputs')
    return {
        'scene': str(observation['scene']), 'seed': int(observation['seed']),
        'instruction': text, 'planner_intent': json.loads(str(observation['planner_intent_json'])),
        'coarse_anchor_world_m': anchor.tolist(),
        'current_a3_tcp_world_wxyz': observation['current_tcp_world_wxyz'].tolist(),
        'current_jaw_opening_m': float(observation['jaw_opening_m']),
        'clap_proprio': robot_state[0].tolist(), 'camera_names': cameras,
        'point_count_in_scene_bounds': int(visible.sum()),
        'point_count_in_zoom': int((cropped[0].abs() <= 1).all(1).sum().item()),
        'zoom_half_width_m': float(1 / (cube_scale * model.st_sca)),
        'current_tcp_inside_zoom': bool((np.abs(observation['current_tcp_world_wxyz'][:3] - anchor)
                                         <= 1 / (cube_scale * model.st_sca)).all()),
        'target_position_world_m': position.tolist(),
        'target_clap_tool_quaternion_xyzw': rotation_xyzw.tolist(),
        'target_clap_tool_quaternion_wxyz': rotation_xyzw[[3, 0, 1, 2]].tolist(),
        'gripper_open_class': int(scores[:2].argmax().item()),
        'clap_collision_class': int(scores[2:].argmax().item()),
        'position_to_coarse_anchor_mm': float(np.linalg.norm(position - anchor) * 1000),
        'fine_forward_and_decode_seconds': perf_counter() - started,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('upstream', 'checkpoint', 'config', 'siglip-config', 'observation', 'output'):
        parser.add_argument(f'--{name}', type=lambda value: Path(value).resolve(), required=True)
    parser.add_argument('--keyframe', type=int, default=2)
    parser.add_argument('--episode-keyframes', type=int, default=25)
    parser.add_argument('--device', choices=('cpu', 'cuda:0'), default='cuda:0')
    args = parser.parse_args()
    if not 0 <= args.keyframe < args.episode_keyframes or args.episode_keyframes < 2:
        parser.error('Invalid keyframe index or episode length')
    args.device = torch.device(args.device)
    if args.device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA PyTorch is required for this probe')
    torch.set_num_threads(4)
    os.chdir(args.upstream)
    model_class, renderer = load_upstream(args.upstream, args.device)
    print(f'Renderer: {renderer} on {args.device}', flush=True)
    with torch.inference_mode():
        model, tokenizer = load_checkpoint(args, model_class)
        print('Official model weights loaded strictly', flush=True)
        with np.load(args.observation, allow_pickle=False) as observation:
            report = predict(args, model, tokenizer, observation)
    report.update(renderer=renderer, device=str(args.device), checkpoint=args.checkpoint.name,
                  output_scope='A3-world position and CLAP-tool orientation, no A3 TCP calibration or motion')
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
