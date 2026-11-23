"""Propose one A3 batch intent from RGB images and observable history."""

from __future__ import annotations

import argparse
import base64
from dataclasses import asdict, dataclass
from io import BytesIO
import json
import os
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

from PIL import Image, ImageDraw


SYSTEM_PROMPT = """You plan cookie packing for the A3 parallel-jaw gripper. Propose exactly one next batch using the task and RGB images. The first image is the current front view. Read coordinates from its grid, with x increasing right and y increasing down, both from 0 to 1000. An optional second image shows the current overhead view for judging rows and counts.

A batch picks 1 to 10 adjacent upright cookies from an exposed end of one source strip and places them in one unoccupied target span. Choose a feasible batch that makes progress and leaves room for later placements. Use any supplied history of attempted intents and outcomes. Your output will be grounded by the Executor. Identify the outermost exposed cookie of the group and the center of the intended destination span in the front view. The Executor handles geometry, gripper motion, and verification.

Return one JSON object with exactly these keys for a batch:
{"action":"pick_and_place","count":5,"source_end_pixel_1000":[x,y],"target_center_pixel_1000":[x,y],"reason":"brief visual justification"}
If the task is visibly complete or you cannot identify an appropriate batch, return exactly {"action":"stop","reason":"brief explanation"}. Do not return object IDs, slot indices, 3D poses, or motion steps."""


@dataclass(frozen=True)
class BatchProposal:
    action: str
    count: int
    source_end_pixel_1000: tuple[float, float]
    target_center_pixel_1000: tuple[float, float]
    reason: str


@dataclass(frozen=True)
class StopProposal:
    action: str
    reason: str


class PlannerError(RuntimeError):
    """A safe-to-display configuration, transport, or response error."""


def _pixel(value: object) -> tuple[float, float]:
    if (not isinstance(value, (tuple, list)) or len(value) != 2
            or any(type(x) not in (int, float) or not 0 <= x <= 1000 for x in value)):
        raise PlannerError('Planner coordinates must be a pair of numbers in [0, 1000]')
    return float(value[0]), float(value[1])


def parse_proposal(value: object) -> BatchProposal | StopProposal:
    if not isinstance(value, dict):
        raise PlannerError('Planner must return a JSON object')
    reason = value.get('reason')
    if not isinstance(reason, str) or not reason.strip():
        raise PlannerError('Planner must return a nonempty reason')
    if value.get('action') == 'stop' and set(value) == {'action', 'reason'}:
        return StopProposal('stop', reason)
    if value.get('action') != 'pick_and_place' or set(value) != set(BatchProposal.__dataclass_fields__):
        raise PlannerError('Planner returned fields outside the batch intent contract')
    count = value['count']
    if type(count) is not int or not 1 <= count <= 10:
        raise PlannerError('Planner batch count must be an integer in [1, 10]')
    return BatchProposal('pick_and_place', count, _pixel(value['source_end_pixel_1000']),
                         _pixel(value['target_center_pixel_1000']), reason)


def _image_part(image: Image.Image, *, grid: bool = False) -> dict:
    rgb = image.convert('RGB')
    if grid:
        width, height = rgb.size
        if min(width, height) < 2:
            raise PlannerError('Planner image must be at least 2 by 2 pixels')
        draw = ImageDraw.Draw(rgb, 'RGBA')
        for value in range(100, 1000, 100):
            x = round((width - 1) * value / 1000)
            y = round((height - 1) * value / 1000)
            draw.line((x, 0, x, height - 1), fill=(255, 255, 255, 96))
            draw.line((0, y, width - 1, y), fill=(255, 255, 255, 96))
            draw.text((x, 3), str(value), anchor='mt', fill='white', stroke_width=1, stroke_fill='black')
            draw.text((3, y), str(value), anchor='lm', fill='white', stroke_width=1, stroke_fill='black')
    with BytesIO() as output:
        rgb.save(output, format='JPEG', quality=90)
        encoded = base64.b64encode(output.getvalue()).decode('ascii')
    return {'type': 'image_url', 'image_url': {'url': f'data:image/jpeg;base64,{encoded}'}}


def _history(value: object) -> list[dict]:
    if not isinstance(value, list):
        raise PlannerError('History must be a JSON array')
    recent = []
    for item in value[-6:]:
        if (not isinstance(item, dict) or set(item) != {'intent', 'outcome'}
                or item['outcome'] not in ('accepted', 'rejected')):
            raise PlannerError('Each history entry needs an intent and an accepted or rejected outcome')
        intent = parse_proposal(item['intent'])
        if not isinstance(intent, BatchProposal):
            raise PlannerError('History may contain attempted batches only')
        context = asdict(intent)
        del context['reason']
        recent.append({'intent': context, 'outcome': item['outcome']})
    return recent


def plan_next(task: str, front: Image.Image, *, overhead: Image.Image | None = None,
              history: list[dict] | None = None) -> BatchProposal | StopProposal:
    """Request one proposal without any simulator, depth, or object-state input."""
    base_url = os.environ.get('A3_PLANNER_API_BASE_URL', '').rstrip('/')
    model = os.environ.get('A3_PLANNER_MODEL', '').strip()
    api_key = os.environ.get('A3_PLANNER_API_KEY', '').strip()
    try:
        endpoint = urlsplit(base_url)
        valid_url = (endpoint.scheme in ('http', 'https') and endpoint.hostname
                     and endpoint.username is None and not endpoint.query and not endpoint.fragment)
    except ValueError:
        valid_url = False
    if not valid_url or not model or not api_key:
        raise PlannerError('Set A3_PLANNER_API_BASE_URL, A3_PLANNER_MODEL, and A3_PLANNER_API_KEY')
    if not task.strip():
        raise PlannerError('Task instruction cannot be empty')

    context = {'task': task, 'recent_attempts': _history([] if history is None else history)}
    content = [{'type': 'text', 'text': json.dumps(context, ensure_ascii=False)}, _image_part(front, grid=True)]
    if overhead is not None:
        content.append(_image_part(overhead))
    payload = {'model': model, 'messages': [{'role': 'system', 'content': SYSTEM_PROMPT},
                                           {'role': 'user', 'content': content}]}
    request = Request(f'{base_url}/chat/completions', json.dumps(payload).encode('utf-8'),
                      headers={'Content-Type': 'application/json'})
    request.add_unredirected_header('Authorization', f'Bearer {api_key}')
    try:
        with urlopen(request, timeout=90) as response:
            result = json.load(response)
    except HTTPError as exc:
        raise PlannerError(f'Planner request failed with HTTP {exc.code}') from None
    except (URLError, OSError, ValueError):
        raise PlannerError('Planner request failed or returned invalid JSON') from None
    try:
        choice = result['choices'][0]
        if choice.get('finish_reason') in ('length', 'content_filter'):
            raise PlannerError('Planner response was incomplete')
        proposal = json.loads(choice['message']['content'])
    except (AttributeError, KeyError, IndexError, TypeError, ValueError):
        raise PlannerError('Planner response did not contain a JSON proposal') from None
    parsed = parse_proposal(proposal)
    if api_key in parsed.reason:
        raise PlannerError('Planner response included sensitive configuration')
    return parsed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task', required=True)
    parser.add_argument('--front', required=True, type=Path)
    parser.add_argument('--overhead', type=Path)
    parser.add_argument('--history', type=Path, help='Optional JSON array of prior intents and outcomes')
    parser.add_argument('--output', required=True, type=Path, help='New JSON file for the proposal')
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Choose a new output file')
    try:
        history = json.loads(args.history.read_text(encoding='utf-8-sig')) if args.history else []
        with Image.open(args.front) as front:
            if args.overhead:
                with Image.open(args.overhead) as overhead:
                    proposal = plan_next(args.task, front, overhead=overhead, history=history)
            else:
                proposal = plan_next(args.task, front, history=history)
        with args.output.open('x', encoding='utf-8') as output:
            json.dump(asdict(proposal), output, indent=2, ensure_ascii=False, allow_nan=False)
            output.write('\n')
    except PlannerError as exc:
        parser.error(str(exc))
    except (OSError, ValueError):
        parser.error('Could not read the input files or write the proposal')


if __name__ == '__main__':
    main()
