import argparse
import json
import math
import os
from pathlib import Path
import shlex
import subprocess
import sys

from steem_adapt.reproduction import ROOT, DATA_FILES, check_inputs, group, jsonl, model_paths, read_rows, write_once


def execute(command, plan=False, env=None):
    command = list(map(str, command))
    print(shlex.join(command), flush=True)
    if not plan:
        subprocess.run(command, cwd=ROOT, env=env, check=True)


def judge_command(a, generated, output, expected=None):
    native = {'rpeval': a.data/'rpeval_native.jsonl',
              'benchpres': a.data/'unified'/DATA_FILES['benchpres'],
              'steem': a.data/'steem_native.jsonl'}[a.benchmark]
    cmd = [sys.executable, ROOT/'scripts/compass_matrix_judge.py',
           '--input', generated, '--benchmark', a.benchmark,
           '--benchmark-data', native, '--output-dir', output,
           '--root', a.output, '--concurrency', a.concurrency, '--idle-timeout', 600,
           '--api-key-env', 'DUET_JUDGE_API_KEY']
    for endpoint in a.judge_endpoint:
        cmd += ['--endpoint', endpoint]
    if expected is not None:
        cmd += ['--expected-samples', expected]
    return list(map(str, cmd))


def main(argv=None):
    p = argparse.ArgumentParser(description='Reproduce the memory-use analysis and Duet experiments.')
    sub = p.add_subparsers(dest='command', required=True)
    data = sub.add_parser('prepare-data', help='Convert and verify the original benchmark inputs')
    data.add_argument('arguments', nargs=argparse.REMAINDER)
    check = sub.add_parser('check-data')
    check.add_argument('--data', type=Path, default=ROOT/'data/paper')
    check.add_argument('--benchmark', choices=[*DATA_FILES, 'all'], default='all')
    for name in ['rq1', 'main', 'rq2', 'judge', 'summarize']:
        cmd = sub.add_parser(name)
        cmd.add_argument('--benchmark', choices=DATA_FILES, default='rpeval')
        cmd.add_argument('--data', type=Path, default=ROOT/'data/paper')
        cmd.add_argument('--output', type=Path, default=ROOT/'runs/reproduction')
        cmd.add_argument('--rpeval-root', type=Path, default=ROOT/'RPEval')
        cmd.add_argument('--steem-root', type=Path, default=ROOT/'SteeM-Memory-Control')
        if name in ['rq1', 'main', 'rq2']:
            cmd.add_argument('--model', required=True, help='Local checkpoint or alias under DUET_MODEL_DIR')
            cmd.add_argument('--device', default='cuda:0')
            cmd.add_argument('--plan', action='store_true')
        if name in ['main', 'rq2']:
            cmd.add_argument('--alpha', type=float, default=1.)
            cmd.add_argument('--max-new-tokens', type=int)
            cmd.add_argument('--limit', type=int, help='Separate smoke run; do not use for paper scores')
        if name in ['main', 'rq2', 'judge']:
            cmd.add_argument('--judge-endpoint', action='append', default=[], metavar='URL[=CAP]')
            cmd.add_argument('--concurrency', type=int, default=16)
        if name in ['judge', 'summarize']:
            cmd.add_argument('--input', type=Path, required=True, help='Generated response JSONL')
        if name == 'summarize':
            cmd.add_argument('--judge-dir', type=Path, required=True)
    if argv is None:
        argv = sys.argv[1:]
    if argv and argv[0] == 'prepare-data':
        execute([sys.executable, ROOT/'scripts/prepare_paper_data.py', *argv[1:]])
        return
    a = p.parse_args(argv)
    a.data = a.data.resolve()
    if a.command == 'check-data':
        benchmarks = list(DATA_FILES) if a.benchmark == 'all' else [a.benchmark]
        print(json.dumps(check_inputs(a.data/'unified', benchmarks), indent=2))
        return
    a.output = a.output.resolve()
    env = dict(os.environ, RPEVAL_ROOT=str(a.rpeval_root.resolve()),
               RPEVAL_PROMPT_FILE=str(a.data/'rpeval_prompt.txt'),
               STEEM_RUBRIC_FILE=str(a.steem_root.resolve()/'memory_control_method/rubrics/dependence_rubrics_text.py'))
    if a.command == 'judge':
        if not a.judge_endpoint:
            p.error('Pass --judge-endpoint explicitly')
        execute(judge_command(a, a.input.resolve(), a.output), env=env)
        return
    if a.command == 'summarize':
        execute([sys.executable, ROOT/'scripts/summarize_reproduction.py', '--benchmark', a.benchmark,
                 '--data', a.data, '--input', a.input.resolve(), '--judge-dir', a.judge_dir.resolve(),
                 '--output', a.output], env=env)
        return
    check_inputs(a.data/'unified', [a.benchmark])
    checkpoint = Path(model_paths().get(a.model, a.model)).resolve()
    if not (checkpoint/'config.json').is_file():
        p.error(f'Checkpoint not found: {checkpoint}')
    if not (a.data/'rpeval_prompt.txt').is_file():
        p.error('Run prepare-data first')
    split = ROOT/'configs/paper_splits.json'
    if a.command == 'rq1':
        execute([sys.executable, ROOT/'scripts/reproduce_rq1.py', '--model', checkpoint,
                 '--benchmark', a.benchmark, '--data-dir', a.data/'unified',
                 '--split-manifest', split, '--device', a.device, '--output-root', a.output/'rq1'], a.plan, env)
        return
    if not math.isfinite(a.alpha) or a.alpha < 0:
        p.error('alpha must be finite and nonnegative')
    if a.limit is not None and a.limit < 1:
        p.error('limit must be positive')
    if a.command == 'rq2' and a.benchmark != 'rpeval':
        p.error('RQ2 uses RPEval')
    rows = read_rows(a.data/'unified'/DATA_FILES[a.benchmark])
    out = a.output/a.command/checkpoint.name/a.benchmark
    if a.limit is not None:
        out = out/f'smoke-{a.limit}'
    if a.command == 'main':
        reader_root = a.output/'readers'
        reader = reader_root/checkpoint.name/a.benchmark/'benchmark'
        execute([sys.executable, ROOT/'scripts/compass_matrix_reader.py', '--model', checkpoint,
                 '--benchmark', a.benchmark, '--reference', 'benchmark', '--stage', 'all',
                 '--data-dir', a.data/'unified', '--split-manifest', split,
                 '--device', a.device, '--output-root', reader_root], a.plan, env)
        source = reader/'evaluation.jsonl'
        predictions = reader/'predictions.jsonl'
        condition, subset = 'dac_full_vocab', 'heldout'
        cal = set(json.loads(split.read_text())[a.benchmark])
        expected = sum(group(r) not in cal for r in rows)
    else:
        source = a.data/'unified'/DATA_FILES[a.benchmark]
        predictions = out/'oracle.jsonl'
        values = {'ignore': -1., 'support': 0., 'dominate': 1.}
        if not a.plan:
            write_once(predictions, jsonl(dict(uid=f"{r['sample_id']}::{m['memory_id']}",
                sample_id=r['sample_id'], memory_id=m['memory_id'], split='heldout',
                steering_coefficient=values[m['gold_policy']]) for r in rows for m in r['memories']))
        condition, subset, expected = 'fixed_no_gate', 'all', len(rows)
    generated = out/f'alpha{a.alpha:g}.jsonl'
    cmd = [sys.executable, ROOT/'scripts/compass_matrix_generate.py', '--model', checkpoint,
           '--benchmark', a.benchmark, '--input', source, '--predictions', predictions,
           '--condition', condition, '--split', subset, '--alpha', a.alpha,
           '--batch-size', 1, '--device', a.device, '--output', generated]
    if a.max_new_tokens is not None:
        cmd += ['--max-new-tokens', a.max_new_tokens]
    if a.limit is not None:
        cmd += ['--limit', a.limit]
        expected = min(expected, a.limit)
    worker = None
    judge_dir = out/f'alpha{a.alpha:g}-judge'
    if a.judge_endpoint:
        jc = judge_command(a, generated, judge_dir, expected)
        if a.plan:
            execute(jc, True)
        else:
            worker = subprocess.Popen(jc, cwd=ROOT, env=env)
    try:
        execute(cmd, a.plan, env)
        if worker is not None:
            code = worker.wait()
            if code:
                raise subprocess.CalledProcessError(code, worker.args)
    finally:
        if worker is not None and worker.poll() is None:
            worker.terminate()
            try:
                worker.wait(timeout=10)
            except subprocess.TimeoutExpired:
                worker.kill()
                worker.wait()
    if a.judge_endpoint and not a.plan:
        execute([sys.executable, ROOT/'scripts/summarize_reproduction.py', '--benchmark', a.benchmark,
                 '--data', a.data, '--input', generated, '--judge-dir', judge_dir,
                 '--output', out/f'alpha{a.alpha:g}-metrics.json'], env=env)


if __name__ == '__main__':
    main()
