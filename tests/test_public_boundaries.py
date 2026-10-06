"""Synthetic tests only; no research questions, targets, models or private oracle."""
from pathlib import Path
import importlib.util
import sys
import unittest

sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[1]

def module(name,path):
 spec=importlib.util.spec_from_file_location(name,path)
 result=importlib.util.module_from_spec(spec);spec.loader.exec_module(result);return result

schema=module('public_schema_test',ROOT/'typed_decision_phase2/schema.py')
dsl=module('public_dsl_test',ROOT/'typed_decision_phase2/macrolab/interpreter.py')

class PublicBoundaryTests(unittest.TestCase):
 def test_public_schema_accepts_only_allowed_fields(self):
  record={'state':'Synthetic formatting test.', 'questions':{'answer':{
   'type':'choice','instructions':schema.INSTRUCTIONS,'criteria':{'A':'First','B':'Second'}}}}
  self.assertEqual(schema.validate_public(record),record)
  with self.assertRaises(ValueError):schema.validate_public({**record,'gold_index':0})
  with self.assertRaises(ValueError):schema.validate_public({**record,'private_target':['ABS']})

 def test_interpreter_synthetic_compositions(self):
  self.assertEqual(dsl.run(['ABS','FILTER_EVEN','SORT_ASC'],[-3,2,-2,1]),[2,2])
  self.assertEqual(dsl.run(['DEDUP_STABLE','REVERSE'],[2,1,2,3,1]),[3,1,2])

 def test_interpreter_rejects_capabilities_and_bounds(self):
  for program in [['SHELL'],['OPEN_FILE'],['__import__'],['ABS']*5]:
   with self.subTest(program=program),self.assertRaises(ValueError):dsl.run(program,[1,2])
  for array in [[True],['1'],[1000001],list(range(13))]:
   with self.subTest(array=array),self.assertRaises(ValueError):dsl.run(['ABS'],array)

if __name__=='__main__':unittest.main()
