import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from steem_adapt.reproduction import DATA_FILES, read_rows
from reproduction_helpers import repetition


def mean(values):
    return sum(values)/len(values) if values else None


def summarize(benchmark, generations, records, data):
    generated = {r['sample_id']: r for r in generations}
    if len(generated) != len(generations):
        raise ValueError('Duplicate generation sample IDs')
    if set(generated) - data.keys():
        raise ValueError('Generation sample absent from benchmark')
    valid = {}
    for record in records:
        sid = record['sample_id']
        if sid not in generated:
            raise ValueError('Judgment sample absent from generations')
        digest = hashlib.sha256(generated[sid]['generated_text'].encode()).hexdigest()
        if record.get('generation_sha256') != digest:
            raise ValueError(f'Judgment/generation mismatch: {sid}')
        if record['status'] == 'ok':
            key = (sid, record['kind'], record.get('memory_id'))
            if key in valid and valid[key]['judge_result'] != record['judge_result']:
                raise ValueError(f'Conflicting valid judgments: {key}')
            valid[key] = record
    complete = []
    for sid in generated:
        keys = ([(sid, 'preference', str(m['memory_id'])) for m in data[sid]['memories']]
                + [(sid, 'completeness', None)]) if benchmark == 'benchpres' else [(sid, 'native', None)]
        if all(k in valid for k in keys):
            complete.append(sid)
    result = dict(benchmark=benchmark, generated=len(generated), valid_samples=len(complete),
                  invalid_or_missing=len(generated)-len(complete), sample_ids=complete,
                  incomplete_sample_ids=sorted(set(generated)-set(complete)))
    if benchmark == 'rpeval':
        def subset(ids):
            rs = [valid[(sid,'native',None)]['judge_result'] for sid in ids]
            slots = sum(r.get('total_slots',1) for r in rs)
            return dict(n=len(rs), macro=mean([r.get('full_match',r.get('match')) for r in rs]),
                        micro=sum(r.get('matched_slots',int(r.get('match') is True)) for r in rs)/slots if slots else None)
        single = [sid for sid in complete if len(data[sid]['memories'])==1]
        multi = [sid for sid in complete if len(data[sid]['memories'])>1]
        ignore = [sid for sid in multi if all(m['gold_policy']=='ignore' for m in data[sid]['memories'])]
        result['metrics'] = dict(single=subset(single), multi=subset(multi),
                                all_ignore=subset(ignore), mixed=subset([s for s in multi if s not in ignore]))
    elif benchmark == 'benchpres':
        apply, suppress, tc = [], [], []
        for sid in complete:
            for memory in data[sid]['memories']:
                followed = valid[(sid,'preference',str(memory['memory_id']))]['judge_result']['follow']
                (suppress if memory['gold_policy']=='ignore' else apply).append(followed)
            tc.append(valid[(sid,'completeness',None)]['judge_result']['rating'])
        result['metrics'] = dict(AAR=mean(apply), MR=mean(suppress), TC=mean(tc))
    else:
        result['metrics'] = dict(MAE=mean([abs(valid[(sid,'native',None)]['judge_result']['overall_memory_dependence_score']
                                             - data[sid]['metadata']['gold_level']) for sid in complete]))
    result['generation_quality'] = dict(
        repeated=sum(repetition(r['generated_text']) is not None for r in generations),
        repetition_rate=mean([repetition(r['generated_text']) is not None for r in generations]),
        mean_characters=mean([len(r['generated_text']) for r in generations]))
    return result


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--benchmark',choices=DATA_FILES,required=True)
    p.add_argument('--input',type=Path,required=True)
    p.add_argument('--judge-dir',type=Path,required=True)
    p.add_argument('--data',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    data={r['sample_id']:r for r in read_rows(a.data/'unified'/DATA_FILES[a.benchmark])}
    recovery=a.judge_dir/'journal_recovery.json'
    damaged=set(json.loads(recovery.read_text())) if recovery.exists() else set()
    records=[json.loads(s) for n,s in enumerate((a.judge_dir/'judge_records.jsonl').read_text().splitlines(),1)
             if s.strip() and n not in damaged]
    records=[r for r in records if not r.get('_journal_recovery')]
    result=summarize(a.benchmark,read_rows(a.input),records,data)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if not k.endswith('sample_ids')},indent=2))


if __name__=='__main__':
    main()
