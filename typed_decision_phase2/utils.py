import os,sys,json,hashlib,csv,time
from pathlib import Path
from datetime import datetime,timezone
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parent;V1=ROOT.parent/'typed_decision_mcqa_15h'
os.environ['HF_HOME']=str(ROOT/'cache/huggingface');os.environ['TORCH_HOME']=str(ROOT/'cache/torch')
os.environ['CUDA_CACHE_PATH']=str(ROOT/'cache/cuda');os.environ['KEV_DATE_FACTS']='0'
os.environ['TOKENIZERS_PARALLELISM']='false';os.environ['HF_HUB_DISABLE_XET']='1'
def now():return datetime.now(timezone.utc).isoformat()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def canon(x):return json.dumps(x,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False)
def digest(x):return hashlib.sha256(canon(x).encode()).hexdigest()
def file_hash(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        while b:=f.read(8*1024*1024):h.update(b)
    return h.hexdigest()
def save(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+'.tmp')
    with tmp.open('w',encoding='utf-8') as f:f.write(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False));f.flush();os.fsync(f.fileno())
    os.replace(tmp,p)
def write(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(x,encoding='utf-8');os.replace(tmp,p)
def csv_write(p,rows):
    if not rows:return
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('w',encoding='utf-8-sig',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def event(kind,**kw):
    with (ROOT/'events.jsonl').open('a',encoding='utf-8') as f:f.write(canon({'time':now(),'kind':kind,**kw})+'\n');f.flush();os.fsync(f.fileno())
def decision(text):
    with (ROOT/'decisions.md').open('a',encoding='utf-8') as f:f.write('\n- '+now()+' '+text+'\n');f.flush()
def budget_ok(seconds=0):return datetime.now(timezone.utc).timestamp()+seconds<datetime.fromisoformat(read(ROOT/'deadline.json')['experiment_stop_utc']).timestamp()
def set_run(run,status,evidence='',reason=''):
    path=ROOT/'manifests/run_matrix.csv';rows=list(csv.DictReader(path.open(encoding='utf-8-sig')))
    for r in rows:
        if r['run_id']==run:r.update(status=status,evidence=evidence,reason=reason)
    csv_write(path,rows);s=read(ROOT/'status.json');s['runs'][run]=status;s['time']=now();s['phase']=run
    save(ROOT/'status.json',s);event('run_status',run_id=run,status=status,evidence=evidence,reason=reason)
