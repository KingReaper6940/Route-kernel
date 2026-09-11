"""Package current source, including uncommitted work, without local secrets."""

import argparse
import gzip
import hashlib
import io
import json
from pathlib import Path
import tarfile

ROOT = Path(__file__).resolve().parents[1]


def source_files(root=ROOT):
    paths = [root / name for name in ('README.md', 'LICENSE', 'pyproject.toml')]
    for folder, extensions in (
        ('routekernel', {'.py', '.wgsl'}), ('tests', {'.py'}),
        ('remote', {'.py', '.json', '.txt'}),
    ):
        paths.extend(p for p in (root / folder).rglob('*') if p.suffix in extensions)
    paths.append(root / 'docs' / 'remote-nvidia.md')
    return sorted(p for p in paths if p.is_file() and not p.is_symlink()
                  and not any(part.startswith('.') or part == '__pycache__' for part in p.relative_to(root).parts)
                  and p.resolve().is_relative_to(root.resolve()))


def source_hashes(root=ROOT):
    return {p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in source_files(root)}


def build_bundle(output, root=ROOT):
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    manifest = source_hashes(root)
    # A deterministic archive with normalized ownership, paths, and timestamps.
    with output.open('xb') as raw, gzip.GzipFile(fileobj=raw, mode='wb', filename='', mtime=0) as gz:
        with tarfile.open(fileobj=gz, mode='w') as archive:
            entries = [(name, (root / name).read_bytes()) for name in manifest]
            entries.append(('BUNDLE-MANIFEST.json', (json.dumps(manifest, indent=2) + '\n').encode()))
            for name, content in entries:
                item = tarfile.TarInfo('routekernel-remote/' + name)
                item.size, item.mode, item.mtime = len(content), 0o644, 0
                archive.addfile(item, io.BytesIO(content))
    digest = hashlib.sha256(output.read_bytes()).hexdigest()
    output.with_name(output.name + '.sha256').write_text(f'{digest}  {output.name}\n', encoding='utf-8')
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'dist' / 'routekernel-remote.tar.gz')
    args = parser.parse_args(argv)
    try:
        manifest = build_bundle(args.output)
    except (OSError, ValueError) as exc:
        parser.exit(1, f'{exc}\nChoose a new output name to preserve an existing archive.\n')
    print(f'Packed {len(manifest)} current source files: {args.output}')
    print('Excluded: environments, .git, credentials, caches, and previous benchmark results.')


if __name__ == '__main__':
    main()
