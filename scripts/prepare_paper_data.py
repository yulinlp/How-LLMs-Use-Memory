import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from steem_adapt.reproduction import DATA_FILES, check_inputs, jsonl, write_once


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument('--benchmark', choices=[*DATA_FILES, 'all'], default='all')
    p.add_argument('--rpeval-root', type=Path, default=ROOT/'RPEval')
    p.add_argument('--steem-root', type=Path, default=ROOT/'SteeM-Memory-Control')
    p.add_argument('--benchpres-parquet', type=Path)
    p.add_argument('--benchpres-revision', default='main')
    p.add_argument('--output-dir', type=Path, default=ROOT/'data/paper')
    a = p.parse_args(argv)
    benchmarks = list(DATA_FILES) if a.benchmark == 'all' else [a.benchmark]
    frozen = json.loads((ROOT/'configs/paper_sources.json').read_text())
    for name, root in [('rpeval', a.rpeval_root), ('steem', a.steem_root)]:
        if name == 'steem' and name not in benchmarks:
            continue
        for relative, digest in frozen[name]['files'].items():
            path = root/relative
            if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
                raise ValueError(f'Upstream source differs from the paper version: {path}')
    from native_judge_protocol import frozen_string, load_benchmark
    from steem_adapt.convert_benchmarks import convert_rpval, convert_benchpres
    with tempfile.TemporaryDirectory(prefix='duet-data-') as directory:
        stage = Path(directory)
        unified = stage/'unified'
        unified.mkdir()
        template = frozen_string(a.rpeval_root/'prompts/prompts.py', 'Personalized_responser_template')
        write_once(stage/'rpeval_prompt.txt', template)
        if 'rpeval' in benchmarks:
            convert_rpval(unified, a.rpeval_root)
            os.environ['RPEVAL_ROOT'] = str(a.rpeval_root.resolve())
            write_once(stage/'rpeval_native.jsonl', jsonl(
                dict(sample_id=sid, **row) for sid, row in load_benchmark(None).items()))
        if 'benchpres' in benchmarks:
            from datasets import load_dataset
            if a.benchpres_parquet:
                rows = load_dataset('parquet', data_files=str(a.benchpres_parquet), split='train')
            else:
                rows = load_dataset('sangyon/BenchPreS', revision=a.benchpres_revision, split='test')
            if not convert_benchpres(unified, rows):
                raise ValueError('BenchPreS conversion failed')
        if 'steem' in benchmarks:
            from steem_adapt.io import load_json_or_jsonl
            from steem_adapt.prepare_steem import split_by_project, expand_samples
            source = a.steem_root/'data_pipeline/context_merge/all_contexts.json.gz'
            instructions = a.steem_root/'memory_control_method/sft_rewrite/control_instruct.json'
            _, evaluation = split_by_project(load_json_or_jsonl(source), 13, .5)
            rows = expand_samples(evaluation[:200], [1,2,3,4,5], str(instructions), 13, False)
            write_once(stage/'steem_native.jsonl', jsonl(rows))
            subprocess.run([sys.executable, str(ROOT/'scripts/build_steem_latent_unified.py'),
                            '--input', str(stage/'steem_native.jsonl'), '--output',
                            str(unified/DATA_FILES['steem'])], check=True)
        report = check_inputs(unified, benchmarks)
        for path in stage.rglob('*'):
            if path.is_file():
                write_once(a.output_dir/path.relative_to(stage), path.read_text())
        for benchmark in benchmarks:
            write_once(a.output_dir/'sources'/f'{benchmark}.json', json.dumps(dict(
                benchmark=benchmark,
                upstream=frozen.get(benchmark, {}),
                benchpres_revision=a.benchpres_revision if benchmark == 'benchpres' else None,
            ), indent=2)+'\n')
        print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
