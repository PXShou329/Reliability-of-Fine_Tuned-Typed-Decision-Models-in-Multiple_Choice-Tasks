"""Public input schema. This module has no labels or evaluator dependencies."""
INSTRUCTIONS='Choose the best answer to the multiple-choice question.'

def validate_public(p):
    if set(p)!={'state','questions'} or not isinstance(p['state'],str): raise ValueError('public keys')
    if set(p['questions'])!={'answer'}: raise ValueError('question keys')
    q=p['questions']['answer']
    if set(q)!={'type','instructions','criteria'} or q['type']!='choice' or q['instructions']!=INSTRUCTIONS: raise ValueError('typed schema')
    if list(q['criteria'])!=[chr(65+i) for i in range(len(q['criteria']))]: raise ValueError('option ordering')
    if not all(isinstance(v,str) for v in q['criteria'].values()): raise ValueError('option values')
    return p
