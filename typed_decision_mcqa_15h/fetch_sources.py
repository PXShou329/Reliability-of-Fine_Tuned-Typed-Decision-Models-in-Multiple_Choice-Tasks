"""Acquire official metadata and code for review, not execute downloaded notebooks."""
import urllib.request
import json
from concurrent.futures import ThreadPoolExecutor
from utils import ROOT,save,write,event,now,file_hash

def fetch(url):
    request=urllib.request.Request(url,headers={"User-Agent":"Typed-MCQA-research-preflight/1.0"})
    with urllib.request.urlopen(request,timeout=60) as r: return r.read()

repos={"laya":"https://huggingface.co/api/models/convaiinnovations/laya","clef_flash":"https://huggingface.co/api/models/Cloudflare/clef-flash","kev":"https://huggingface.co/api/models/jaredpalmer/kev-4b","arc":"https://huggingface.co/api/datasets/allenai/ai2_arc","csqa":"https://huggingface.co/api/datasets/tau/commonsense_qa","obqa":"https://huggingface.co/api/datasets/allenai/openbookqa","laya_code":"https://api.github.com/repos/NandhaKishorM/laya/commits/main"}
with ThreadPoolExecutor(max_workers=4) as pool:
    fetched=dict(zip(repos,pool.map(fetch,repos.values())))
metadata={}
for name,data in fetched.items():
    value=json.loads(data);save(ROOT/"sources"/(name+"_metadata.json"),value);metadata[name]=value
save(ROOT/"sources/revisions.json",{k:v["sha"] for k,v in metadata.items()})
laya_sha=metadata["laya_code"]["sha"]
urls={"laya_common.py":f"https://raw.githubusercontent.com/NandhaKishorM/laya/{laya_sha}/laya/common.py","laya_agent.py":f"https://raw.githubusercontent.com/NandhaKishorM/laya/{laya_sha}/laya/agent.py","laya_training.ipynb":f"https://raw.githubusercontent.com/NandhaKishorM/laya/{laya_sha}/notebooks/laya_finetune_typed_decisions_2xT4_kaggle.ipynb","laya_README.md":f"https://raw.githubusercontent.com/NandhaKishorM/laya/{laya_sha}/README.md","clef_joint_schema_model.py":f"https://huggingface.co/Cloudflare/clef-flash/resolve/{metadata['clef_flash']['sha']}/joint_schema_model.py","clef_README.md":f"https://huggingface.co/Cloudflare/clef-flash/resolve/{metadata['clef_flash']['sha']}/README.md","laya_model_README.md":f"https://huggingface.co/convaiinnovations/laya/resolve/{metadata['laya']['sha']}/README.md"}
download_manifest=[]
for name,url in urls.items():
    data=fetch(url);path=ROOT/"sources"/name;path.write_bytes(data)
    download_manifest.append({"file":name,"url":url,"sha256":file_hash(path),"fetched_at":now()})
notebook=json.loads((ROOT/"sources/laya_training.ipynb").read_text(encoding="utf-8"))
write(ROOT/"sources/laya_training_review.py","\n\n".join("".join(cell["source"]) for cell in notebook["cells"] if cell["cell_type"]=="code"))
save(ROOT/"sources/download_manifest.json",download_manifest)
print(json.dumps({k:v["sha"] for k,v in metadata.items()},indent=2))
