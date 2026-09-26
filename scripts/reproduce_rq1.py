"""Resume-safe full-population RQ1 replication for the four extra backbones.

Native prompts, all layers, original RQ1 calibration group IDs. No judge calls.
RPEval additionally extracts paired isolated signatures and evaluates them
with the same joint-calibrated centroids and layer, not a refitted reader.
"""
import argparse
import hashlib
import inspect
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'scripts')]
from compass_matrix_generate import MODEL_PATHS, DATA
from compass_matrix_reader import configure_reader_runtime


def save_json(path, value):
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2) + '\n')
    temp.replace(path)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', required=True)
    p.add_argument('--data-dir', type=Path, default=ROOT/'data/unified')
    p.add_argument('--split-manifest', type=Path, default=ROOT/'configs/paper_splits.json')
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--benchmark', required=True, choices=list(DATA))
    p.add_argument('--output-root', type=Path, required=True)
    p.add_argument('--limit', type=int, help='Smoke run only; uses separate output directory')
    a = p.parse_args()
    checkpoint = str(Path(MODEL_PATHS.get(a.model, a.model)).resolve())
    a.model = Path(checkpoint).name
    DATA[a.benchmark] = a.data_dir / DATA[a.benchmark].name
    out = a.output_root / a.model / a.benchmark
    if a.limit:
        out = out / 'smoke'
    out.mkdir(parents=True, exist_ok=True)
    rows = [json.loads(s) for s in DATA[a.benchmark].read_text().splitlines() if s.strip()]
    config = configure_reader_runtime(checkpoint)
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from steem_adapt.mccs_generate import render_latent_prompt
    torch.set_num_threads(2)
    depth = config.get('text_config', config)['num_hidden_layers']
    contract = dict(model=a.model, benchmark=a.benchmark, layers=depth,
        source_sha256=hashlib.sha256(DATA[a.benchmark].read_bytes()).hexdigest(),
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        renderer_sha256=hashlib.sha256((ROOT/'steem_adapt/mccs_generate.py').read_bytes()).hexdigest(),
        calibration_sha256=hashlib.sha256(a.split_manifest.read_bytes()).hexdigest(),
        checkpoint=checkpoint, config_sha256=hashlib.sha256((Path(checkpoint)/'config.json').read_bytes()).hexdigest(),
        extraction='all non-embedding hidden states at last prompt token; native templates',
        limit=a.limit)
    if (out/'contract.json').exists():
        assert json.loads((out/'contract.json').read_text()) == contract, 'Contract changed; use a fresh output root'
    else:
        save_json(out/'contract.json', contract)
    if a.limit:
        rows = rows[:a.limit]
    checkpoints = out/'rows'
    checkpoints.mkdir(exist_ok=True)
    tokenizer = model = None
    start = time.monotonic()

    def activation(row, removed):
        nonlocal tokenizer, model
        if model is None:
            tokenizer = AutoTokenizer.from_pretrained(checkpoint, local_files_only=True)
            cls = AutoModelForCausalLM
            if config.get('model_type') == 'qwen3_5':
                from transformers import Qwen3_5ForConditionalGeneration
                cls = Qwen3_5ForConditionalGeneration
            model = cls.from_pretrained(checkpoint, local_files_only=True,
                torch_dtype=torch.bfloat16, attn_implementation='eager').to(a.device).eval()
            print('MODEL_READY', a.model, a.benchmark, flush=True)
        prompt = render_latent_prompt(tokenizer, row, removed)
        # Match legacy RQ1 tokenizer behavior (including its default special tokens).
        batch = tokenizer(prompt, return_tensors='pt').to(a.device)
        positions = batch['attention_mask'].long().cumsum(-1)-1
        positions.masked_fill_(batch['attention_mask']==0, 0)
        extra = {}
        parameters = inspect.signature(model.forward).parameters
        for name in ['logits_to_keep', 'num_logits_to_keep']:
            if name in parameters:
                extra[name] = 1
                break
        with torch.inference_mode():
            result = model(**batch, position_ids=positions, output_hidden_states=True,
                           use_cache=False, **extra)
        assert len(result.hidden_states) == depth+1
        return torch.stack([h[0,-1].detach().cpu() for h in result.hidden_states[1:]])

    metadata, joint, isolated = [], [], []
    for n,row in enumerate(rows):
        path = checkpoints/f'{n:05d}.pt'
        if path.exists():
            item = torch.load(path, map_location='cpu', weights_only=False)
            assert item['sample_id']==row['sample_id']
        else:
            full = activation(row,set())
            js, iso = [], []
            for memory in row['memories']:
                mid = memory['memory_id']
                js.append((full-activation(row,{mid})).to(torch.float16))
                if a.benchmark=='rpeval':
                    if len(row['memories'])==1:
                        iso.append(js[-1])
                    else:
                        singleton = dict(row,memories=[memory])
                        iso.append((activation(singleton,set())-activation(singleton,{mid})).to(torch.float16))
            item = dict(sample_id=row['sample_id'], joint=torch.stack(js),
                        isolated=torch.stack(iso) if iso else None)
            temp=path.with_suffix('.tmp')
            torch.save(item,temp)
            temp.replace(path)
        joint.extend(item['joint'])
        if item['isolated'] is not None:
            isolated.extend(item['isolated'])
        metadata.extend(dict(sample_id=row['sample_id'], memory_id=m['memory_id'],
            group_id=str(row.get('metadata',{}).get('query_id',row['sample_id'])),
            uid=f"{row['sample_id']}::{m['memory_id']}",gold_policy=m['gold_policy']) for m in row['memories'])
        if n%5==0 or n+1==len(rows):
            print('PROGRESS',a.model,a.benchmark,n+1,len(rows),'seconds',round(time.monotonic()-start,1),flush=True)
    artifact=dict(signatures=torch.stack(joint),metadata=metadata,model=checkpoint,
                  definition='last-token h(full)-h(without-memory-i), frozen benchmark prompt')
    assert torch.isfinite(artifact['signatures']).all()
    torch.save(artifact,out/'joint.pt')
    if isolated:
        torch.save(dict(artifact,signatures=torch.stack(isolated),definition='last-token h(only-memory-i)-h(no-memory)'),out/'isolated.pt')
    if a.limit:
        print('SMOKE_OK',flush=True)
        return
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    import analyze_rq1_cross_benchmark as analysis
    calibration = set(json.loads(a.split_manifest.read_text())[a.benchmark])
    groups = {str(r.get('metadata', {}).get('query_id', r['sample_id'])) for r in rows}
    if not calibration < groups:
        raise ValueError('Calibration groups missing from data')
    split = dict(reference_group_ids=sorted(calibration), final_test_group_ids=sorted(groups-calibration),
                 selection=dict(source=str(a.split_manifest)))
    report=analysis.analyze(out/'joint.pt',a.model,a.benchmark,rows,[split],out,2)
    save_json(out/'report.json',report)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(5, 3.2), layout='constrained')
    for regime, curves in report['regime_curves'].items():
        ax.plot(range(1, depth+1), curves['macro_recall'], label=regime)
    selected = report['calibration_selected_layer']
    ax.axvline(selected, color='gray', linestyle='--', label=f'Calibration-selected: {selected}')
    ax.set(xlabel='Layer (1-indexed)', ylabel='Held-out macro recall', title=f'{a.model} / {a.benchmark}', ylim=(0,1))
    ax.legend()
    fig.savefig(out/'layer_readout_curves.pdf')
    plt.close(fig)
    if isolated:
        records=analysis.validate_artifact(artifact,analysis.records_for(rows,a.benchmark),a.model)
        ref,test=analysis.split_indices(records,split)
        classes=analysis.CONFIG[a.benchmark][1]
        labels=torch.tensor([classes.index(r['label']) for r in records])
        regimes=[r['regime'] for r in records]
        x=artifact['signatures'].float()
        z=torch.stack(isolated).float()
        combined=torch.cat([x,z],0)
        pred=analysis.predict(combined,torch.cat([labels,labels]),regimes+regimes,ref,
                              [len(records)+i for i in test],len(classes))
        multi=[p for p,i in enumerate(test) if regimes[i]=='multi']
        score=analysis.metrics(pred[multi],labels[[test[p] for p in multi]],classes)
        save_json(out/'paired_isolated.json',dict(calibration_selected_layer=report['calibration_selected_layer'],
            joint=report['regime_curves']['multi'],isolated=score,
            test_items=len(multi),reader='same joint calibration centroids, same selected layer'))
    save_json(out/'complete.json',dict(rows=len(rows),items=len(metadata),seconds=time.monotonic()-start))
    print('COMPLETE',a.model,a.benchmark,flush=True)


if __name__=='__main__':
    main()
