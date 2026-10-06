import hashlib
import json
import os
from pathlib import Path
from datetime import datetime,timezone

ROOT=Path(__file__).resolve().parent
T0="2026-10-05T17:32:00+00:00"
DEADLINE="2026-10-06T08:32:00+00:00"

def canonical(x): return json.dumps(x,sort_keys=True,separators=(",",":"),ensure_ascii=False,allow_nan=False)
def digest(x): return hashlib.sha256(canonical(x).encode()).hexdigest()
def file_hash(p):
    h=hashlib.sha256()
    with Path(p).open("rb") as f:
        while b:=f.read(4*1024*1024): h.update(b)
    return h.hexdigest()
def now(): return datetime.now(timezone.utc).isoformat()
def read(p): return json.loads(Path(p).read_text(encoding="utf-8-sig"))
def save(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    t=p.with_suffix(p.suffix+".tmp")
    with t.open('w',encoding='utf-8') as f:
        f.write(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False));f.flush();os.fsync(f.fileno())
    os.replace(t,p)
def write(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    t=p.with_suffix(p.suffix+".tmp");t.write_text(x,encoding="utf-8");os.replace(t,p)
def event(kind,**fields):
    p=ROOT/"events.jsonl"
    with p.open("a",encoding="utf-8") as f:
        f.write(canonical({"time":now(),"kind":kind,**fields})+"\n");f.flush();os.fsync(f.fileno())
