"""Verify the Git LFS parts and reconstruct the unmodified official checkpoint."""

import argparse
from hashlib import sha256
import json
from pathlib import Path


def restore_weights(manifest_path, output):
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    whole = sha256()
    output.parent.mkdir(parents=True, exist_ok=True)
    pending = output.with_name(output.name + '.pending')
    if output.exists():
        raise FileExistsError(output)
    target = pending.open('xb')
    try:
        with target:
            for part in manifest['parts']:
                path = manifest_path.parent / part['file']
                if path.stat().st_size != part['bytes']:
                    raise ValueError(f'{path.name} has the wrong length. Run git lfs pull.')
                digest = sha256()
                with path.open('rb') as source:
                    for chunk in iter(lambda: source.read(8 * 1024**2), b''):
                        target.write(chunk)
                        digest.update(chunk)
                        whole.update(chunk)
                if digest.hexdigest() != part['sha256']:
                    raise ValueError(f'SHA-256 mismatch in {path.name}')
        if pending.stat().st_size != manifest['bytes'] or whole.hexdigest() != manifest['sha256']:
            raise ValueError('Reassembled checkpoint does not match the official one')
        pending.rename(output)
    except BaseException:
        pending.unlink(missing_ok=True)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, default=Path(__file__).parent / 'weights/manifest.json')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    restore_weights(args.manifest, args.output)
    print(args.output)


if __name__ == '__main__':
    main()
