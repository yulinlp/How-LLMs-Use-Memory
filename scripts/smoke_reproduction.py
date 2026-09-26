import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]


def main():
    import torch
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import PreTrainedTokenizerFast, Qwen3Config, Qwen3ForCausalLM
    from steem_adapt.reproduction import jsonl
    torch.set_num_threads(1)
    with tempfile.TemporaryDirectory(prefix='duet-smoke-') as d:
        base=Path(d); model=base/'TinyQwen'; data=base/'data'; data.mkdir()
        vocabulary=['<pad>','<unk>','<eos>','user','system','assistant','memory','query','Ignore','Support','Dominate','a','b','c']
        tok=Tokenizer(WordLevel({x:i for i,x in enumerate(vocabulary)},unk_token='<unk>'))
        tok.pre_tokenizer=Whitespace()
        fast=PreTrainedTokenizerFast(tokenizer_object=tok,pad_token='<pad>',unk_token='<unk>',eos_token='<eos>')
        fast.chat_template="{% for m in messages %}{{ m['role'] }} {{ m['content'] }} {% endfor %}{% if add_generation_prompt %}assistant {% endif %}"
        fast.save_pretrained(model)
        torch.manual_seed(42)
        Qwen3ForCausalLM(Qwen3Config(vocab_size=len(vocabulary),hidden_size=32,intermediate_size=64,
            num_hidden_layers=2,num_attention_heads=4,num_key_value_heads=2,head_dim=8,
            max_position_embeddings=2048,eos_token_id=2,pad_token_id=0)).save_pretrained(model)
        rows=[]
        for i in range(18):
            labels=['ignore','support','dominate'] if i>=9 else [['ignore','support','dominate'][i%3]]
            rows.append(dict(source='rpval',sample_id=f'q{i}',query=f'query {i}',context='',metadata={},
                memories=[dict(memory_id=f'm{j}',memory_text=f'memory {j}',gold_policy=l,
                               gold_score={'ignore':0,'support':.5,'dominate':1}[l]) for j,l in enumerate(labels)]))
        (data/'rpval_implicit_unified.jsonl').write_text(jsonl(rows))
        split=base/'splits.json'; split.write_text(json.dumps(dict(rpeval=[f'q{i}' for i in [0,1,2,3,4,5,9,10]])))
        template=base/'prompt.txt'; template.write_text('memory {persona}\nquery {question}')
        env=dict(os.environ,RPEVAL_PROMPT_FILE=str(template),CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',HF_HUB_OFFLINE='1')
        def run(script,*args):
            subprocess.run([sys.executable,str(ROOT/'scripts'/script),*map(str,args)],cwd=base,env=env,check=True)
        run('compass_matrix_reader.py','--model',model,'--benchmark','rpeval','--reference','benchmark',
            '--stage','all','--device','cpu','--dtype','float32','--data-dir',data,'--split-manifest',split,'--output-root',base/'readers')
        reader=base/'readers/TinyQwen/rpeval/benchmark'
        for condition in ['fixed_no_gate','dac_full_vocab']:
            out=base/(condition+'.jsonl')
            args=['--model',model,'--benchmark','rpeval','--condition',condition,'--input',reader/'evaluation.jsonl',
                  '--predictions',reader/'predictions.jsonl','--split','heldout','--batch-size','1','--limit','4',
                  '--max-new-tokens','3','--output',out]
            run('compass_matrix_generate.py',*args)
            before=out.read_bytes()
            run('compass_matrix_generate.py',*args)
            assert before==out.read_bytes(), 'Resume duplicated outputs'
            records=[json.loads(s) for s in out.read_text().splitlines()]
            assert len(records)==4
        for benchmark in ['benchpres','steem']:
            example=dict(source=benchmark,sample_id='s',query='query',context='memory a\nmemory b\nquery\n',
                memories=[dict(memory_id='a',memory_text='memory a'),dict(memory_id='b',memory_text='memory b')])
            src=base/(benchmark+'.jsonl'); src.write_text(jsonl([example]))
            pred=base/(benchmark+'-predictions.jsonl')
            pred.write_text(jsonl([dict(uid='s::'+m,sample_id='s',memory_id=m,split='heldout',steering_coefficient=c)
                                  for m,c in [('a',-1.),('b',1.)]]))
            run('compass_matrix_generate.py','--model',model,'--benchmark',benchmark,'--condition','dac_full_vocab',
                '--input',src,'--predictions',pred,'--split','heldout','--batch-size','1',
                '--max-new-tokens','3','--output',base/(benchmark+'-generated.jsonl'))
        run('reproduce_rq1.py','--model',model,'--benchmark','rpeval','--data-dir',data,
            '--split-manifest',split,'--device','cpu','--output-root',base/'rq1')
        assert (base/'rq1/TinyQwen/rpeval/complete.json').exists()
        print('PASS: random tiny-model reader, fixed/Duet decoding, resume, and RQ1 extraction/readout. This is not a benchmark score.')


if __name__=='__main__':
    main()
