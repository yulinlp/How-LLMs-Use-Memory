"""Bounded, latency-aware routing across the user's shared judge services."""
import json
import os
import threading
import time
from pathlib import Path
import requests

ENDPOINTS = [('http://localhost:8000/v1', 8)]


class Router:
    def __init__(self, base, endpoints=None, api_key_env='COMPASS_JUDGE_API_KEY'):
        endpoints = ENDPOINTS if endpoints is None else endpoints
        self.condition = threading.Condition()
        self.states = [dict(endpoint=u, cap=c, active=0, latency=30.,
                            count=0, cooldown=0., failures=0) for u, c in endpoints]
        configs = {}
        with requests.Session() as session:
            session.trust_env = False
            session.headers['Authorization'] = 'Bearer ' + os.environ.get(api_key_env, 'EMPTY')
            for url, cap in endpoints:
                try:
                    response = session.get(url.removesuffix('/v1') + '/get_server_info', timeout=5)
                    response.raise_for_status()
                except requests.RequestException as exc:
                    configs[url] = {'unavailable': type(exc).__name__}
                    state = next(s for s in self.states if s['endpoint'] == url)
                    state['cooldown'] = time.monotonic() + 300
                    state['latency'] = 90
                    continue
                args = response.json()['internal_states'][0]
                keys = ['served_model_name', 'model_path', 'dtype', 'quantization',
                        'tp_size', 'reasoning_parser', 'enable_deterministic_inference']
                configs[url] = {k: args.get(k) for k in keys}
                assert args['served_model_name'] == 'Qwen/Qwen3.8-27B'
                assert args['reasoning_parser'] == 'qwen3'
                assert args['enable_deterministic_inference']
        if not configs or all('unavailable' in value for value in configs.values()):
            raise ConnectionError('No compatible judge endpoint is reachable; check the URL, API key and SGLang /get_server_info endpoint')
        self.audit = Path(base) / 'judge_routing_events.jsonl'
        with self.audit.open('a') as stream:
            stream.write(json.dumps(dict(event='startup', time=time.time(),
                endpoints=endpoints, server_configs=configs,
                note='Transport change only; rubric and request sampling unchanged.')) + '\n')

    def acquire(self):
        with self.condition:
            while True:
                eligible = [s for s in self.states if s['active'] < s['cap']
                            and s['cooldown'] <= time.monotonic()]
                if eligible:
                    state = min(eligible, key=lambda s:
                        ((s['active'] + 1) * s['latency'] / s['cap'], s['count']))
                    state['active'] += 1
                    state['count'] += 1
                    return state
                self.condition.wait(timeout=1)

    def release(self, state, seconds, status):
        with self.condition:
            state['active'] -= 1
            if status is None or status == 429 or status >= 500:
                state['failures'] += 1
                state['cooldown'] = time.monotonic() + min(600, 60 * 2 ** min(state['failures'], 3))
                # An immediate connection failure is not a fast inference.
                state['latency'] = max(90, state['latency'])
            else:
                state['failures'] = 0
                state['latency'] = .8 * state['latency'] + .2 * seconds
            with self.audit.open('a') as stream:
                stream.write(json.dumps(dict(event='completed', endpoint=state['endpoint'],
                    time=time.time(), seconds=seconds, status=status)) + '\n')
            self.condition.notify_all()

    def Session(self):
        return RoutedSession(self)


class RoutedSession(requests.Session):
    def __init__(self, router):
        super().__init__()
        self.router = router
        self.judge_endpoint = None

    def post(self, url, **kwargs):
        for attempt in range(3):
            state = self.router.acquire()
            self.judge_endpoint = state['endpoint']
            start, status = time.monotonic(), None
            try:
                response = super().post(self.judge_endpoint + '/chat/completions', **kwargs)
                status = response.status_code
                if (status == 429 or status >= 500) and attempt < 2:
                    continue
                return response
            except requests.RequestException:
                if attempt == 2:
                    raise
            finally:
                self.router.release(state, time.monotonic() - start, status)
