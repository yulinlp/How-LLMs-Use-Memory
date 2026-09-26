from pathlib import Path
import hashlib
import json
import os

ROOT = Path(__file__).resolve().parents[1]
DATA_FILES = {
    'rpeval': 'rpval_implicit_unified.jsonl',
    'benchpres': 'benchpres_unified.jsonl',
    'steem': 'steem_tag_masked_unified.jsonl',
}
MODELS = ('Qwen3-4B', 'Qwen3-8B', 'Qwen3.5-4B', 'Qwen3.5-9B',
          'Llama-3.1-8B-Instruct', 'Llama-3.2-3B-Instruct')


def model_paths():
    base = Path(os.environ.get('DUET_MODEL_DIR', ROOT / 'models'))
    return {name: str(base / name) for name in MODELS}


def read_rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def group(row):
    return str(row.get('metadata', {}).get('query_id', row['sample_id']))


def write_once(path, content):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_text() != content:
            raise ValueError(f'Existing file differs; choose a new output directory: {path}')
    else:
        path.write_text(content)


def jsonl(rows):
    return ''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in rows)


def check_inputs(directory, benchmarks, manifest=None, splits=None):
    manifest = Path(manifest or ROOT / 'configs/paper_inputs.json')
    specs = json.loads(manifest.read_text())
    splits = json.loads(Path(splits or ROOT / 'configs/paper_splits.json').read_text())
    report = []
    for benchmark in benchmarks:
        spec = specs[benchmark]
        path = Path(directory) / spec['filename']
        content = path.read_bytes()
        if hashlib.sha256(content).hexdigest() != spec['sha256']:
            raise ValueError(f'{benchmark}: input differs from the paper snapshot: {path}')
        rows = read_rows(path)
        if [row['sample_id'] for row in rows] != spec['sample_ids']:
            raise ValueError(f'{benchmark}: sample order differs from the manifest')
        groups = {group(row) for row in rows}
        if not set(splits[benchmark]) < groups:
            raise ValueError(f'{benchmark}: invalid calibration groups')
        report.append(dict(benchmark=benchmark, samples=len(rows), groups=len(groups),
                           calibration_groups=len(splits[benchmark])))
    return report
