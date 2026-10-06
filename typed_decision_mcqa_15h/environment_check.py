import sys,platform,subprocess,importlib.metadata,psutil,torch
from utils import ROOT,save,now
pkgs=['torch','torchvision','laya','transformers','datasets','accelerate','huggingface_hub','bitsandbytes','safetensors','numpy','scipy','pillow']
versions={}
for p in pkgs:
    try: versions[p]=importlib.metadata.version(p)
    except importlib.metadata.PackageNotFoundError: versions[p]=None
manifest={'time':now(),'python':sys.version,'executable':sys.executable,'platform':platform.platform(),
          'packages':versions,'torch_cuda':torch.version.cuda,'cuda_available':torch.cuda.is_available(),
          'ram_total_bytes':psutil.virtual_memory().total,'ram_available_bytes':psutil.virtual_memory().available,
          'gpu_name':torch.cuda.get_device_name(0),'compute_capability':torch.cuda.get_device_capability(0),
          'gpu_total_bytes':torch.cuda.get_device_properties(0).total_memory,
          'torch_cuda_arch_list':torch.cuda.get_arch_list(),
          'nvidia_smi':subprocess.check_output(['nvidia-smi','--query-gpu=name,driver_version,memory.total,memory.used','--format=csv'],text=True),
          'network_policy':'public downloads only; inference and training entirely local; no uploads'}
x=torch.randn(128,128,device='cuda',dtype=torch.bfloat16);y=x@x;torch.cuda.synchronize()
manifest['bf16_matmul_smoke']={'passed':bool(torch.isfinite(y).all().item()),'dtype':str(y.dtype)}
save(ROOT/'environment_manifest.json',manifest)
lock=subprocess.check_output([sys.executable,'-m','pip','freeze'],text=True)
(ROOT/'requirements.lock.txt').write_text(lock,encoding='utf-8')
print(manifest,flush=True)
