import sys
import unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'extensions/output_judge'))
from suite import ITEMS,RUBRICS,parse_score,confidence_fields,judge_prompt
from summarize import summarize

class PilotTests(unittest.TestCase):
 def test_score_requires_terminal_valid_result(self):
  self.assertEqual(parse_score('Evidence here. [RESULT] 4'),4)
  for text in ['4','[RESULT] 0','[RESULT] 6','[RESULT] 3 then 5','[RESULT] 4.5']:
   self.assertIsNone(parse_score(text))
 def test_missing_score_not_zero(self):
  result=summarize([], [{'kind':'target','trait':'honesty','score':None}], [{'id':'missing','trait':'honesty'}])
  self.assertIsNone(result['traits']['honesty']['mean'])
  self.assertEqual(result['traits']['honesty']['n_valid'],0)
  self.assertEqual(result['n_invalid_judgments'],1)
 def test_all_seven_traits_and_pairs_present(self):
  self.assertTrue(set(['honesty','helpfulness','harmlessness','power_seeking','corrigibility','sycophancy','confidence'])<=set(s['trait'] for s in ITEMS))
  self.assertEqual(len({s['id'] for s in ITEMS}),len(ITEMS))
  for pair in ['arithmetic','triangle']:
   self.assertEqual({s['pair_role'] for s in ITEMS if s.get('pair_id')==pair},{'neutral','pressure'})
 def test_confidence_uses_ground_truth(self):
  self.assertEqual(confidence_fields('{"answer":"24","confidence":0.9}',{'expected':'23'})['brier_score'],.81)
  self.assertFalse(confidence_fields('{"answer":"23","confidence":true}',{'expected':'23'})['format_valid'])
  self.assertFalse(confidence_fields('{"answer":"23","confidence":1.5}',{'expected':'23'})['format_valid'])
 def test_fenced_confidence_preserves_format_failure(self):
  r=confidence_fields('```json\n{"answer":"23","confidence":1}\n```',{'expected':'23'})
  self.assertTrue(r['format_valid'])
  self.assertFalse(r['strict_json'])
  self.assertEqual(r['brier_score'],0)
 def test_judge_has_no_condition_metadata(self):
  prompt=judge_prompt('question','answer','reference','honesty')
  self.assertNotIn('seed',prompt)
  self.assertNotIn('activation',prompt)
 def test_adverse_traits_direction_explicit(self):
  for trait in ['power_seeking','sycophancy','confidence']:
   self.assertIn('HIGH means MORE',RUBRICS[trait][0])

if __name__=='__main__': unittest.main()
