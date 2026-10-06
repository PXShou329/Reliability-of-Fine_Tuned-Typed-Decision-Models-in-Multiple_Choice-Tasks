"""Validate the reviewed release manifest without importing experiment runners."""
from pathlib import Path
import ast,csv,hashlib,json,re

ROOT=Path(__file__).resolve().parents[1]
patterns={
 'private_key':r'-----BEGIN (?:RSA |EC |OPENSSH |DSA )?PRIVATE KEY-----',
 'github_token':r'\b(?:gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,})\b',
 'cloud_key':r'\b(?:AKIA|ASIA)[A-Z0-9]{16}\b',
 'email':r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b',
 'personal_path':r'(?i)(?:(?<![A-Z0-9])[A-Z]:[\\/]|/(?:Users|home)/)',
 'mobile':r'(?<!\d)09\d{2}[ -]?\d{3}[ -]?\d{3}(?!\d)',
}
with (ROOT/'PUBLIC_FILES_SHA256.csv').open(encoding='utf-8',newline='') as handle:
 records=list(csv.DictReader(handle))
reviewed={r['file'] for r in records}|{'PUBLIC_FILES_SHA256.csv'}
assert len(reviewed)==len(records)+1
for row in records:
 path=(ROOT/row['file']).resolve();assert path.is_relative_to(ROOT.resolve())
 assert path.is_file() and path.stat().st_size==int(row['bytes']),row['file']
 assert hashlib.sha256(path.read_bytes()).hexdigest()==row['sha256'],row['file']
 if path.suffix not in {'.png','.pdf'}:
  text=path.read_text(encoding='utf-8-sig')
  for name,pattern in patterns.items():assert not re.search(pattern,text),(row['file'],name)
  if path.suffix=='.py':ast.parse(text,filename=row['file'])
  if path.suffix=='.json':json.loads(text)
with (ROOT/'PUBLIC_CSV_CHECKS.json').open(encoding='utf-8') as handle:
 for record in json.load(handle)['files']:
  with (ROOT/record['public_file']).open(encoding='utf-8',newline='') as source:
   reader=csv.DictReader(source);rows=list(reader)
   assert len(rows)==record['rows'] and reader.fieldnames==record['kept_columns']
assert 'typed_decision_phase2/macrolab/oracle.py' not in reviewed
assert 'typed_decision_phase2/macrolab/generate.py' not in reviewed
assert 'typed_decision_mcqa_15h/prepare_data.py' not in reviewed
assert 'typed_decision_phase2/prepare_data.py' not in reviewed
print(f'PASS: {len(reviewed)} reviewed files; hashes, text patterns, syntax and aggregate row counts.')
print('This validates the publication package, not the original experiment or all possible secret encodings.')
