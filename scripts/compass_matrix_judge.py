"""Native matrix judge, one generation JSONL per scheduler invocation.

Example (paths are relative to cwd; --root is metadata only)::

    python scripts/compass_matrix_judge.py --input runs/.../condition.jsonl \
        --benchmark rpeval --output-dir runs/.../judge/condition \
        --expected-samples 100 --endpoint http://localhost:8000/v1=8

With --expected-samples, wait for newline-terminated samples, even if the input
does not exist yet. Without it, judge a snapshot of complete input lines.
Exit 0 means complete valid coverage; 2 means missing/invalid coverage. Each
invocation tries each unresolved judgment once, with bounded transport retries.
Restart to retry invalid judgments; successful judgments are never repeated.
Output: protocol.json, judge_records.jsonl (append-only), summary.json, and the
Router's judge_routing_events.jsonl. Use a distinct output directory per input.
Run --self-test for synthetic CPU tests with mocked transport (no services).
"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
import fcntl
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import uuid

import requests

ROOT = Path(__file__).resolve().parents[1]
MODEL = 'Qwen/Qwen3.8-27B'
VERSION = 1


def encoded(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False)


def sha(value):
    return hashlib.sha256(value.encode('utf-8')).hexdigest()


def sample_id(row):
    value = row.get('sample_id', row.get('uid'))
    if value is None or str(value) == '':
        raise ValueError('Missing sample_id/uid')
    return str(value)


def read_rows(path):
    with path.open(encoding='utf-8') as stream:
        return [json.loads(line) for line in stream if line.strip()]


class Native:
    def __init__(self, benchmark, data=None):
        self.benchmark = benchmark
        if benchmark == 'rpeval':
            import native_judge_protocol as protocol
            prompt_path = Path(os.environ.get('RPEVAL_ROOT', ROOT / 'RPEval')) / 'prompts/prompts_715.py'
            self.rp = dict(METRICS=protocol.METRICS, load_benchmark=protocol.load_benchmark, safe_json=protocol.safe_json,
                           valid_result=protocol.valid_result,
                           SINGLE_PROMPT=protocol.frozen_string(prompt_path, 'LLM_judge_prompt'),
                           MULTI_PROMPT=protocol.frozen_string(prompt_path, 'LLM_judge_multiple_prompt'))
            self.schema = protocol.schema_instruction
            self.items = self.rp['load_benchmark'](None) if data is None else None
        elif benchmark == 'benchpres':
            import native_judge_protocol
            self.bp = native_judge_protocol
            data = data or ROOT / 'data/unified/benchpres_unified.jsonl'
        else:
            import native_judge_protocol
            self.st = native_judge_protocol
            rubric = Path(os.environ.get('STEEM_RUBRIC_FILE', ROOT / 'SteeM-Memory-Control/memory_control_method/rubrics/dependence_rubrics_text.py'))
            self.st.RUBRICS_TEXT = self.st.frozen_string(rubric, 'RUBRICS_TEXT')
            # Native context/query/task, NOT the unified generation prompt.
            data = data or ROOT / 'data/eval_split_tag_masked.jsonl'
        if data is not None:
            self.items = {}
            for row in read_rows(Path(data)):
                sid = sample_id(row)
                if sid in self.items:
                    raise ValueError(f'Duplicate benchmark sample: {sid}')
                self.items[sid] = row

    def tasks(self, row):
        sid = sample_id(row)
        text = row['generated_text']
        if not isinstance(text, str):
            raise ValueError(f'{sid}: generated_text must be a string')
        item = self.items[sid]
        specs = []
        if self.benchmark == 'rpeval':
            single = item['kind'] == 'single'
            slots = 1 if single else len(item['persona'])
            if single:
                intent = {'ignore': '忽略偏好', 'support': '支持性偏好',
                          'supportive': '支持性偏好', 'dominate': '以偏好为主线',
                          'dominant': '以偏好为主线'}.get(item['intent_type'], item['intent_type'])
                prompt = self.rp['SINGLE_PROMPT'].format(
                    persona=item['persona'], question=item['question'], reply=text,
                    intent_type=intent, intent=item.get('intent') or item.get('reason', ''))
            else:
                prompt = self.rp['MULTI_PROMPT'].format(
                    persona=item['persona'], question=item['question'], reply=text,
                    intent=item['intent_type'], intent_type=item['intent_type'])
            specs.append(('native', None, self.schema(single, slots), prompt, True,
                          {'kind': item['kind'], 'slots': slots}))
        elif self.benchmark == 'benchpres':
            memories = set()
            for memory in item['memories']:
                mid = str(memory['memory_id'])
                if mid in memories:
                    raise ValueError(f'{sid}: duplicate memory_id {mid}')
                memories.add(mid)
                specs.append(('preference', mid, None,
                              self.bp.preference_prompt(str(memory['memory_text']), text),
                              True, {'gold_policy': str(memory['gold_policy']),
                                     'gold_apply': str(memory['gold_policy']) != 'ignore'}))
            specs.append(('completeness', None, None,
                          self.bp.completeness_prompt(str(item['query']), text), False, {}))
        else:
            system, prompt = self.st.prompt({**item, 'generated_text': text})
            specs.append(('native', None, system, prompt, True, {}))
        tasks = []
        for kind, mid, system, prompt, json_mode, extra in specs:
            messages = ([{'role': 'system', 'content': system}] if system is not None else [])
            messages.append({'role': 'user', 'content': prompt})
            payload = dict(model=MODEL, temperature=0, max_tokens=8192,
                           chat_template_kwargs={'enable_thinking': True}, messages=messages)
            if json_mode:
                payload['response_format'] = {'type': 'json_object'}
            tasks.append(dict(key=encoded([sid, kind, mid]), sample_id=sid, kind=kind,
                              memory_id=mid, benchmark=self.benchmark, payload=payload,
                              generation_sha256=sha(text), response_sha256=sha(text),
                              sample_sha256=sha(encoded(item)),
                              prompt_sha256=sha((system + '\n' if system is not None else '') + prompt),
                              payload_sha256=sha(encoded(payload)), native_metadata=extra))
        return tasks

    def parse(self, task, raw):
        if self.benchmark == 'rpeval':
            return self.rp['valid_result'](self.rp['safe_json'](raw), task['native_metadata']['slots'])
        if self.benchmark == 'steem':
            return self.st.parse(raw)
        if task['kind'] == 'preference':
            label = self.bp.parse_label(raw)
            return None if label is None else {'judge_label': label, 'follow': label == 'follow'}
        rating = self.bp.parse_rating(raw)
        return None if rating is None else {'rating': rating}


def judge(task, native, router, args):
    transport = []
    raw, result, error, endpoint = '', None, None, None
    stable = getattr(args, 'stable_fallback', False)
    no_thinking = getattr(args, 'no_thinking_retry', False)
    fallback_used = False
    for attempt in range(1 if stable else args.network_attempts):
        state = router.acquire()
        endpoint = state['endpoint']
        start, status, retry = time.monotonic(), None, False
        entry = {'attempt': attempt + 1, 'endpoint': endpoint}
        payload = dict(task['payload'])
        if no_thinking:
            payload.update(chat_template_kwargs={'enable_thinking':False},max_tokens=2048,repetition_penalty=1.05)
        if stable:
            payload['rid'] = 'compass-judge-' + uuid.uuid4().hex
            entry.update(request_id=payload['rid'], thinking=not no_thinking)
        try:
            # Use Router's capacity/cooldown accounting directly so its hidden
            # RoutedSession retries cannot multiply our retry bound or hide attempts.
            with requests.Session() as session:
                session.trust_env = False
                response = session.post(endpoint + '/chat/completions', json=payload,
                                        headers={'Authorization': 'Bearer ' + os.environ.get(args.api_key_env, 'EMPTY')},
                                        timeout=args.timeout)
            status = response.status_code
            entry['status_code'] = status
            retry = status in (408, 429, 500, 502, 503, 504)
            response.raise_for_status()
            body = response.json()
            choice = body['choices'][0]
            entry.update(usage=body.get('usage'), model=body.get('model'),
                         finish_reason=choice.get('finish_reason'), message=choice['message'])
            raw = choice['message'].get('content') or ''
            result = native.parse(task, raw)
            error = None if result is not None else 'Invalid native judgment schema'
        except (requests.ConnectionError, requests.Timeout) as exc:
            error, retry = f'{type(exc).__name__}: {exc}', True
        except Exception as exc:
            error = f'{type(exc).__name__}: {exc}'
        finally:
            if stable and error and (status is None or retry):
                try:
                    with requests.Session() as cancel:
                        cancel.trust_env = False
                        entry['cancel_status'] = cancel.post(endpoint.removesuffix('/v1') + '/abort_request',
                            json={'rid':payload['rid']}, timeout=15).status_code
                except Exception as exc:
                    entry['cancel_status'] = type(exc).__name__
            entry.update(seconds=time.monotonic() - start, error=error)
            transport.append(entry)
            router.release(state, entry['seconds'], status)
        if not retry or result is not None:
            break
    if stable and result is None:
        fallback_used = True
        state = router.acquire(); endpoint = state['endpoint']
        start = time.monotonic(); status = None
        payload = dict(task['payload'], max_tokens=2048,
                       chat_template_kwargs={'enable_thinking':False}, repetition_penalty=1.05,
                       rid='compass-fallback-' + uuid.uuid4().hex)
        entry = dict(attempt=len(transport)+1,endpoint=endpoint,request_id=payload['rid'],
                     thinking=False,max_tokens=2048,repetition_penalty=1.05,
                     reason='Primary request timed out, failed transport, or returned invalid/empty judgment')
        try:
            with requests.Session() as session:
                session.trust_env = False
                response = session.post(endpoint+'/chat/completions',json=payload,
                    headers={'Authorization':'Bearer '+os.environ.get(args.api_key_env,'EMPTY')},
                    timeout=min(args.timeout,600))
            status=response.status_code; response.raise_for_status()
            body=response.json(); choice=body['choices'][0]
            raw=choice['message'].get('content') or ''
            result=native.parse(task,raw)
            error=None if result is not None else 'Invalid native judgment schema after no-thinking fallback'
            entry.update(usage=body.get('usage'),finish_reason=choice.get('finish_reason'),message=choice['message'])
        except Exception as exc:
            error=f'{type(exc).__name__}: {exc}'
            try:
                with requests.Session() as cancel:
                    cancel.trust_env=False
                    entry['cancel_status']=cancel.post(endpoint.removesuffix('/v1')+'/abort_request',
                        json={'rid':payload['rid']},timeout=15).status_code
            except Exception as cancel_exc:
                entry['cancel_status']=type(cancel_exc).__name__
        finally:
            entry.update(status_code=status,seconds=time.monotonic()-start,error=error)
            transport.append(entry); router.release(state,entry['seconds'],status)
    record = {k: v for k, v in task.items() if k != 'payload'}
    record.update(status='ok' if result is not None else 'unknown', judge_result=result,
                  parsed=result, raw=raw, raw_judge_response=raw, judge_error=error,
                  endpoint=endpoint, judge_endpoint=endpoint, transport=transport,
                  judge_model=MODEL, thinking=not (fallback_used or no_thinking), temperature=0,
                  max_tokens=2048 if (fallback_used or no_thinking) else 8192, fallback_used=fallback_used,
                  no_thinking_retry=no_thinking,
                  protocol_version=VERSION, time=time.time())
    return record


HASHES = ('generation_sha256', 'sample_sha256', 'prompt_sha256', 'payload_sha256')


def same_sample(previous, current):
    for field in HASHES:
        if previous.get(field) != current.get(field):
            raise ValueError(f"Resume/input mismatch for {current['key']}: {field}")


def run(args, native, router_factory=None):
    from rq3_judge_router import Router
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    with (output / 'worker.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        protocol = dict(version=VERSION, input=str(args.input.resolve()), benchmark=args.benchmark,
                        benchmark_sha256=sha(encoded(native.items)), model=MODEL,
                        thinking=True, temperature=0, max_tokens=8192,
                        root=str(args.root.resolve()), retry='invalids retry on next invocation; bounded network attempts')
        path = output / 'protocol.json'
        if path.exists() and json.loads(path.read_text()) != protocol:
            raise ValueError('Output protocol differs; use a separate output directory')
        if not path.exists():
            path.write_text(encoded(protocol) + '\n')
        if getattr(args,'no_thinking_retry',False):
            (output/'no_thinking_retry.json').write_text(encoded(dict(time=time.time(),thinking=False,max_tokens=2048,
                repetition_penalty=1.05,reason='User authorized direct non-thinking retry of remaining invalid judgments',
                scope='Previously valid records unchanged; per-record settings authoritative'))+'\n')
        if getattr(args,'stable_fallback',False):
            policy=dict(version=1,primary_thinking=True,primary_max_tokens=8192,
                fallback_thinking=False,fallback_max_tokens=2048,repetition_penalty=1.05,
                rule='Same fallback for all methods: timeout/transport/empty/invalid primary; never score-dependent',
                unchanged='Benchmark prompt, rubric, generation text, temperature=0; valid existing scores reused',
                transport='Unique request IDs; best-effort exact-request cancellation after timeout',
                scope='New attempts only; per-record thinking/fallback metadata is authoritative')
            (output/'fallback_policy.json').write_text(encoded(policy)+'\n')
        journal = output / 'judge_records.jsonl'
        latest, valid, historical_invalids = {}, {}, 0
        recovery = output / 'journal_recovery.json'
        known_damaged = set(json.loads(recovery.read_text()) if recovery.exists() else [])
        if journal.exists():
            # Preserve a crash-torn final write byte-for-byte, isolate it with
            # a newline, and retry its key. Interior corruption is fatal.
            lines = journal.read_bytes().splitlines(keepends=True)
            for index, line in enumerate(lines):
                if index + 1 in known_damaged:
                    continue
                try:
                    record = json.loads(line)
                except (ValueError, UnicodeDecodeError):
                    if index != len(lines) - 1 or line.endswith(b'\n'):
                        raise ValueError(f'Corrupt judgment journal at line {index + 1}')
                    known_damaged.add(index + 1)
                    # Persist recovery before isolating the fragment, so another
                    # crash cannot turn it into an unexplained interior line.
                    recovery_temp = output / 'journal_recovery.json.tmp'
                    recovery_temp.write_text(encoded(sorted(known_damaged)) + '\n')
                    recovery_temp.replace(recovery)
                    continue
                if record.get('_journal_recovery'):
                    continue
                key = record['key']
                if key in latest:
                    same_sample(latest[key], record)
                latest[key] = record
                if record['status'] == 'ok':
                    valid[key] = record
                else:
                    historical_invalids += 1
            if lines and not lines[-1].endswith(b'\n'):
                with journal.open('ab') as stream:
                    stream.write(b'\n')
        # A sidecar remembers isolated torn lines without changing journal bytes.
        tasks, samples, attempted, pending = {}, {}, set(), {}
        offset, identity, router = 0, None, None
        started = time.monotonic()

        def ingest():
            nonlocal offset, identity
            if not args.input.exists():
                return
            stat = args.input.stat()
            now_identity = (stat.st_dev, stat.st_ino)
            if identity is not None and (identity != now_identity or stat.st_size < offset):
                raise ValueError('Streaming input was replaced or truncated')
            identity = now_identity
            with args.input.open('rb') as stream:
                stream.seek(offset)
                while True:
                    line = stream.readline()
                    if not line or not line.endswith(b'\n'):
                        break
                    offset = stream.tell()
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    sid = sample_id(row)
                    new_tasks = native.tasks(row)
                    if sid in samples and samples[sid] != sha(row['generated_text']):
                        raise ValueError(f'Changed generation for {sid}')
                    samples[sid] = sha(row['generated_text'])
                    for task in new_tasks:
                        key = task['key']
                        if key in latest:
                            same_sample(latest[key], task)
                            if key in valid and native.parse(task, valid[key]['raw']) != valid[key]['judge_result']:
                                raise ValueError(f'Native parser changed for {key}')
                        tasks[key] = task
            if args.expected_samples is not None and len(samples) > args.expected_samples:
                raise ValueError('Input exceeds --expected-samples')

        def summary():
            keys = set(tasks)
            good = keys & valid.keys()
            bad = {k for k in keys if k in latest and k not in valid}
            expected = args.expected_samples if args.expected_samples is not None else len(samples)
            incomplete_samples = {tasks[k]['sample_id'] for k in keys - good}
            good_samples = len(samples.keys() - incomplete_samples)
            result = dict(benchmark=args.benchmark, input=str(args.input), expected_samples=expected,
                          observed_samples=len(samples), valid_samples=good_samples,
                          missing_samples=max(0, expected - len(samples)), expected_judgments_observed=len(keys),
                          valid=len(good), invalid=len(bad), pending=len(keys - good - bad),
                          invalid_attempts=historical_invalids, recovered_journal_lines=len(known_damaged),
                          coverage=good_samples / expected if expected else 0,
                          complete=bool(samples) and len(samples) == expected and len(good) == len(keys),
                          valid_by_kind=dict(Counter(tasks[k]['kind'] for k in good)))
            temp = output / 'summary.json.tmp'
            temp.write_text(encoded(result) + '\n')
            temp.replace(output / 'summary.json')
            return result

        ingest()
        if args.expected_samples is None and not args.input.exists():
            raise FileNotFoundError(args.input)
        with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
            while True:
                if args.expected_samples is not None:
                    ingest()
                for key, task in tasks.items():
                    if len(pending) >= args.concurrency:
                        break
                    if key in valid or key in attempted:
                        continue
                    if router is None:
                        router = (router_factory or Router)(output, endpoints=args.endpoints, api_key_env=args.api_key_env)
                    pending[pool.submit(judge, task, native, router, args)] = key
                    attempted.add(key)
                if pending:
                    finished, _ = wait(pending, timeout=args.poll_interval, return_when=FIRST_COMPLETED)
                    for future in finished:
                        key = pending.pop(future)
                        record = future.result()
                        with journal.open('a', encoding='utf-8') as stream:
                            stream.write(encoded(record) + '\n')
                            stream.flush()
                            os.fsync(stream.fileno())
                        latest[key] = record
                        if record['status'] == 'ok':
                            valid[key] = record
                        else:
                            historical_invalids += 1
                report = summary()
                exhausted = all(k in valid or k in attempted for k in tasks)
                received = args.expected_samples is None or len(samples) == args.expected_samples
                if not pending and exhausted and received:
                    break
                if args.idle_timeout and time.monotonic() - started >= args.idle_timeout and not pending:
                    break
                if not pending:
                    time.sleep(args.poll_interval)
        if set(latest) - set(tasks):
            raise ValueError('Judgment journal contains samples absent from completed input')
        print(encoded(report), flush=True)
        return 0 if report['complete'] else 2


def cli(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--input', type=Path)
    parser.add_argument('--benchmark', choices=('rpeval', 'rpval', 'benchpres', 'bp', 'steem'))
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--benchmark-data', type=Path, help='Native benchmark JSONL override')
    parser.add_argument('--root', type=Path, default=Path('runs/compass_matrix_20260922'))
    parser.add_argument('--expected-samples', '--expected-count', type=int)
    parser.add_argument('--endpoint', action='append', default=[], metavar='URL[=CAP]',
                        help='Repeat for multiple services; default uses Router endpoints')
    parser.add_argument('--concurrency', type=int, default=16)
    parser.add_argument('--network-attempts', type=int, default=2)
    parser.add_argument('--timeout', type=float, default=900)
    parser.add_argument('--poll-interval', type=float, default=2)
    parser.add_argument('--idle-timeout', type=float, default=0, help='Stop waiting after this many seconds (0: unlimited)')
    parser.add_argument('--api-key-env', default='COMPASS_JUDGE_API_KEY')
    parser.add_argument('--stable-fallback',action=argparse.BooleanOptionalAction,default=True,
                        help='Uniform no-thinking retry after timeout or invalid native judgment')
    parser.add_argument('--self-test', action='store_true')
    parser.add_argument('--no-thinking-retry',action='store_true',help='Explicitly authorized non-thinking retry; preserves valid existing judgments')
    args = parser.parse_args(argv)
    if args.self_test:
        return self_test()
    if not all((args.input, args.benchmark, args.output_dir)):
        parser.error('--input, --benchmark and --output-dir are required')
    if not 1 <= args.concurrency <= 128 or not 1 <= args.network_attempts <= 10:
        parser.error('concurrency must be 1..128 and network-attempts 1..10')
    if args.timeout <= 0 or args.poll_interval <= 0 or args.idle_timeout < 0:
        parser.error('timeouts/poll interval must be positive (idle timeout permits zero)')
    if args.expected_samples is not None and args.expected_samples < 1:
        parser.error('--expected-samples must be positive')
    args.benchmark = {'rpval': 'rpeval', 'bp': 'benchpres'}.get(args.benchmark, args.benchmark)
    args.endpoints = []
    for value in args.endpoint:
        url, sep, cap = value.rpartition('=')
        url, cap = (url, int(cap)) if sep else (value, args.concurrency)
        if not url.startswith(('http://', 'https://')) or cap < 1:
            parser.error('endpoint must be http(s)://host:port/v1[=positive-cap]')
        args.endpoints.append((url.rstrip('/'), min(cap, args.concurrency)))
    args.endpoints = args.endpoints or None
    return run(args, Native(args.benchmark, args.benchmark_data))


def self_test():
    """Tests live here to keep the change strictly to one new script."""
    import tempfile
    import threading
    from types import SimpleNamespace
    from unittest.mock import patch

    class FakeRouter:
        def __init__(self, *a, **kw):
            pass

        def acquire(self):
            return {'endpoint': 'http://synthetic/v1'}

        def release(self, *a):
            pass

    def response(raw, status=200):
        result = requests.Response()
        result.status_code = status
        result._content = json.dumps({'choices': [{'message': {'content': raw},
                                                   'finish_reason': 'stop'}]}).encode()
        return result

    with tempfile.TemporaryDirectory(prefix='compass-judge-test-') as directory:
        base = Path(directory)
        data = base / 'benchmark.jsonl'
        data.write_text(encoded(dict(sample_id='s', query='task', memories=[
            dict(memory_id='m', memory_text='preference', gold_policy='ignore')])) + '\n')
        native = Native('benchpres', data)
        source = base / 'input.jsonl'
        row = dict(sample_id='s', generated_text='answer')
        source.write_text(encoded(row) + '\n')
        args = SimpleNamespace(input=source, output_dir=base / 'out', root=base,
                               benchmark='benchpres', expected_samples=1, concurrency=1,
                               endpoints=None, network_attempts=2, timeout=1,
                               poll_interval=.001, idle_timeout=.1, api_key_env='TEST_KEY_UNUSED')
        with patch.object(requests.Session, 'post', side_effect=[response('invalid'), response('Rating: [[4]]')]) as call:
            assert run(args, native, FakeRouter) == 2
            assert call.call_count == 2
        with patch.object(requests.Session, 'post', return_value=response('{"label":"do_not_follow"}')) as call:
            assert run(args, native, FakeRouter) == 0
            assert call.call_count == 1
        with patch.object(requests.Session, 'post') as call:
            assert run(args, native, FakeRouter) == 0
            call.assert_not_called()
        records = read_rows(args.output_dir / 'judge_records.jsonl')
        assert len(records) == 3 and records[0]['status'] == 'unknown'
        journal = args.output_dir / 'judge_records.jsonl'
        before = journal.read_bytes()
        with journal.open('ab') as stream:
            stream.write(b'{"key":"crash')
        for _ in range(2):
            with patch.object(requests.Session, 'post') as call:
                assert run(args, native, FakeRouter) == 0
                call.assert_not_called()
        assert journal.read_bytes().startswith(before + b'{"key":"crash\n')
        task = native.tasks(row)[0]
        with patch.object(requests.Session, 'post', side_effect=[requests.ConnectionError('synthetic'), response('{"label":"follow"}')]):
            result = judge(task, native, FakeRouter(), args)
            assert result['status'] == 'ok' and len(result['transport']) == 2
        with patch.object(requests.Session, 'post', side_effect=requests.Timeout('synthetic')) as call:
            assert judge(task, native, FakeRouter(), args)['status'] == 'unknown'
            assert call.call_count == 2
        with patch.object(requests.Session, 'post', side_effect=[response('', 503), response('{"label":"follow"}')]) as call:
            result = judge(task, native, FakeRouter(), args)
            assert result['status'] == 'ok' and call.call_count == 2
            assert result['transport'][0]['status_code'] == 503
        with patch.object(requests.Session, 'post', return_value=response('', 400)) as call:
            assert judge(task, native, FakeRouter(), args)['status'] == 'unknown'
            assert call.call_count == 1
        source.write_text(encoded(dict(row, generated_text='changed')) + '\n')
        try:
            run(args, native, FakeRouter)
            raise AssertionError('changed sample was accepted')
        except ValueError as exc:
            assert 'mismatch' in str(exc)
        source.write_text(encoded(row))  # incomplete streamed line is not scored
        args.output_dir = base / 'partial'
        with patch.object(requests.Session, 'post') as call:
            assert run(args, native, FakeRouter) == 2
            call.assert_not_called()
        def finish_line():
            with source.open('a') as stream:
                stream.write('\n')
        timer = threading.Timer(.02, finish_line)
        timer.start()
        try:
            with patch.object(requests.Session, 'post', side_effect=[response('{"label":"follow"}'), response('Rating: [[5]]')]):
                assert run(args, native, FakeRouter) == 0
        finally:
            timer.join()
        for benchmark in ('rpeval', 'steem'):
            adapter = Native(benchmark)
            sid = next(iter(adapter.items))
            item_task = adapter.tasks(dict(sample_id=sid, generated_text='synthetic answer'))[0]
            assert item_task['payload']['max_tokens'] == 8192
            assert item_task['payload']['chat_template_kwargs']['enable_thinking'] is True
            if benchmark == 'rpeval':
                raw = encoded(dict(match=True, reason='test', **{k: 3 for k in adapter.rp['METRICS']}))
                assert adapter.parse(item_task, raw)['match'] is True
                multi = next(k for k, v in adapter.items.items() if v['kind'] == 'multi' and len(v['persona']) > 1)
                mt = adapter.tasks(dict(sample_id=multi, generated_text='test'))[0]
                n = mt['native_metadata']['slots']
                raw = encoded(dict(match=f'{n}/{n}', full_match=True, reason='test', **{k: 3 for k in adapter.rp['METRICS']}))
                assert adapter.parse(mt, raw)['full_match'] is True
            else:
                assert adapter.parse(item_task, '{"overall_memory_dependence_score":3}') is not None
    print('Synthetic CPU tests passed (native adapters, retries, resume, hashes, partial lines).')
    return 0


if __name__ == '__main__':
    sys.exit(cli())
