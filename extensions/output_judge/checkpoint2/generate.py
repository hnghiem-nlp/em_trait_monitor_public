"""Inference only; immutable input manifest, durable per-model outputs, resumable."""
import modal,json
from pathlib import Path
HERE=Path(__file__).resolve().parent
app=modal.App('checkpoint2-output-evaluation')
cache=modal.Volume.from_name('cs2881-hw0-model-cache')
results=modal.Volume.from_name('cs2881-em-harmful-fraction')
image=(modal.Image.debian_slim(python_version='3.11').uv_pip_install('vllm==0.15.0')
 .env({'HF_HOME':'/cache/huggingface','TOKENIZERS_PARALLELISM':'false','PYTHONUNBUFFERED':'1'}))
@app.function(image=image,gpu='A100-80GB',timeout=24*3600,volumes={'/cache':cache,'/results':results},max_containers=1)
def generate(manifest):
 import hashlib,time
 from vllm import LLM,SamplingParams
 from vllm.lora.request import LoRARequest
 root=Path('/results/output_evaluation')/manifest['digest'][:16];root.mkdir(parents=True,exist_ok=True)
 (root/'manifest.json').write_text(json.dumps(manifest,indent=2));results.commit()
 pending=[m for m in manifest['models'] if not (root/(m['id']+'.json')).exists()]
 if pending:
  llm=LLM(model='Qwen/Qwen2.5-7B-Instruct',revision='a09a35458c702b33eeacc393d103063234e8bc28',dtype='bfloat16',enable_lora=True,max_lora_rank=16,max_model_len=4096,gpu_memory_utilization=.85,enforce_eager=True,seed=42,max_num_seqs=64)
  tok=llm.get_tokenizer();texts=[tok.apply_chat_template(p['messages'],tokenize=False,add_generation_prompt=True) for p in manifest['prompts']]
  params=SamplingParams(**manifest['generation'])
  for index,m in enumerate(pending,1):
   adapter=m['adapter'];start=time.time()
   request=LoRARequest(m['id'],index,adapter) if adapter else None
   outputs=llm.generate(texts,params,lora_request=request,use_tqdm=False)
   answers=[]
   for spec,res in zip(manifest['prompts'],outputs):
    r=res.outputs[0];answers.append({**spec,'response':r.text,'finish_reason':r.finish_reason,'truncated':r.finish_reason=='length','tokens':len(r.token_ids)})
   assert len(answers)==len(texts)
   adapter_hash=hashlib.sha256((Path(adapter)/'adapter_model.safetensors').read_bytes()).hexdigest() if adapter else None
   data={'model':m,'adapter_sha256':adapter_hash,'manifest_digest':manifest['digest'],'engine':'vllm==0.15.0','elapsed_seconds':time.time()-start,'answers':answers}
   (root/(m['id']+'.json')).write_text(json.dumps(data,indent=2));results.commit()
   print('COMPLETE',m['id'],'seconds',round(time.time()-start),'truncated',sum(x['truncated'] for x in answers),flush=True)
 return {'remote_dir':str(root),'models':[json.loads((root/(m['id']+'.json')).read_text()) for m in manifest['models']]}
@app.local_entrypoint()
def main():
 manifest=json.loads((HERE/'manifest.json').read_text());out=HERE/'runs'/manifest['digest'][:16];out.mkdir(parents=True,exist_ok=True)
 (out/'manifest.json').write_text(json.dumps(manifest,indent=2))
 data=generate.remote(manifest)
 for row in data['models']:(out/(row['model']['id']+'.json')).write_text(json.dumps(row,indent=2))
 print('OUTPUT',out,flush=True)
