import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'scripts')]
from steem_adapt.reproduction import check_inputs, jsonl, write_once
from compass_matrix_generate import FrozenDirections, select_rows, prompt_fields
from compass_matrix_reader import nearest_numeric, isolated_pairs
from summarize_reproduction import summarize
import native_judge_protocol as native
import torch


class ReproductionTests(unittest.TestCase):
    def test_write_once_rejects_overwrite(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'x'
            write_once(p,'original'); write_once(p,'original')
            with self.assertRaises(ValueError): write_once(p,'changed')

    def test_inputs_reject_drift(self):
        with tempfile.TemporaryDirectory() as d:
            base=Path(d)
            rows=[dict(sample_id=str(i),metadata={}) for i in range(3)]
            content=jsonl(rows)
            (base/'data.jsonl').write_text(content)
            (base/'manifest.json').write_text(json.dumps(dict(rpeval=dict(filename='data.jsonl',
                sha256=hashlib.sha256(content.encode()).hexdigest(),sample_ids=['0','1','2']))))
            (base/'split.json').write_text(json.dumps(dict(rpeval=['0'])))
            report=check_inputs(base,['rpeval'],base/'manifest.json',base/'split.json')
            self.assertEqual(report[0]['calibration_groups'],1)
            (base/'data.jsonl').write_text(content+'\n')
            with self.assertRaises(ValueError): check_inputs(base,['rpeval'],base/'manifest.json',base/'split.json')

    def test_fixed_direction_keeps_prefill_difference(self):
        x,y=torch.randn(2,5,8),torch.randn(2,5,8)
        freeze=FrozenDirections(); freeze(x)
        z=freeze(y)
        torch.testing.assert_close(z[:,0],y[:,0])
        torch.testing.assert_close(z[:,1::2]-z[:,2::2],x[:,1::2]-x[:,2::2])

    def test_predictions_require_exact_coverage(self):
        row=dict(sample_id='s',memories=[dict(memory_id='m')])
        pred=dict(uid='s::m',sample_id='s',memory_id='m',split='heldout',steering_coefficient=1.)
        self.assertEqual(select_rows([row],[pred],'heldout'),[row])
        with self.assertRaises(ValueError): select_rows([row],[pred,pred],'heldout')
        with self.assertRaises(ValueError): select_rows([row],[dict(pred,steering_coefficient=float('nan'))],'heldout')

    def test_runtime_prompt_drops_labels(self):
        row=dict(source='rpval',sample_id='s',query='q',metadata={'answer':'secret'},
                 memories=[dict(memory_id='m',memory_text='text',gold_policy='ignore',gold_score=0)])
        clean=prompt_fields(row)
        self.assertNotIn('metadata',clean)
        self.assertEqual(set(clean['memories'][0]),{'memory_id','memory_text'})

    def test_reader_excludes_same_query_and_group(self):
        def meta(uid,g,q):
            return dict(uid=uid,sample_id=uid,memory_id='m',split='heldout',benchmark='rpeval',group_id=g,query_key=q)
        q=torch.tensor([[[1.,0.]]]); refs=torch.tensor([[[1.,0.]],[[.9,.1]],[[0.,1.]]])
        out=nearest_numeric(q,refs,torch.tensor([-1.,0.,1.]),[meta('q','a','same')],
                            [meta('1','a','other'),meta('2','b','same'),meta('3','c','new')])
        self.assertEqual(out[0]['reference_uids'],['3'])
        self.assertEqual(out[0]['steering_coefficient'],1.)

    def test_benchpres_keeps_full_row_for_deletion(self):
        row=dict(source='benchpres',memories=[dict(memory_id='a'),dict(memory_id='b')])
        pairs=list(isolated_pairs(row))
        self.assertEqual(pairs[0],(row,{'b'},{'a','b'}))

    def test_native_parsers(self):
        scores={k:3 for k in native.METRICS}
        self.assertTrue(native.valid_result(dict(scores,match=True),1)['match'])
        self.assertIsNone(native.valid_result(dict(scores,match='true'),1))
        self.assertFalse(native.valid_result(dict(scores,match='1/2',full_match=True),2)['match_consistent'])
        self.assertEqual(native.parse_label('{"label":"do_not_follow"}'),'do_not_follow')
        self.assertEqual(native.parse_rating('Reason\nRating: [[4]]'),4)
        self.assertIsNone(native.parse('{"overall_memory_dependence_score":6}'))

    def test_unreachable_judge_fails_before_queueing(self):
        import requests
        from rq3_judge_router import Router
        with tempfile.TemporaryDirectory() as d:
            with patch.object(requests.Session,'get',side_effect=requests.ConnectionError('mock')):
                with self.assertRaises(ConnectionError):
                    Router(d,endpoints=[('http://unused/v1',1)])

    def test_summary_weighted_micro_and_missing(self):
        data={sid:dict(memories=[dict(gold_policy='ignore') for _ in range(n)]) for sid,n in [('a',2),('b',4),('c',3)]}
        gs=[dict(sample_id=sid,generated_text='answer') for sid in data]
        rs=[dict(sample_id=sid,kind='native',memory_id=None,status='ok',
                 generation_sha256=hashlib.sha256(b'answer').hexdigest(),
                 judge_result=dict(full_match=False,matched_slots=m,total_slots=n))
            for sid,m,n in [('a',1,2),('b',1,4)]]
        result=summarize('rpeval',gs,rs,data)
        self.assertAlmostEqual(result['metrics']['multi']['micro'],2/6)
        self.assertEqual(result['invalid_or_missing'],1)
        rs[0]['generation_sha256']='wrong'
        with self.assertRaises(ValueError): summarize('rpeval',gs,rs,data)


if __name__=='__main__':
    unittest.main()
