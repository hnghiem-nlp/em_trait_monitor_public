"""One-shot read-only download of completed models; no recurring monitor."""
import json,modal
from pathlib import Path
HERE=Path(__file__).resolve().parent
app=modal.App('checkpoint2-validation-collect')
v=modal.Volume.from_name('cs2881-em-harmful-fraction')
@app.function(volumes={'/results':v},timeout=300)
def collect(digest):
 root=Path('/results/output_evaluation')/digest[:16]
 return {p.name:json.loads(p.read_text()) for p in root.glob('*.json')}
@app.local_entrypoint()
def main():
 m=json.loads((HERE/'manifest.json').read_text());out=HERE/'runs'/m['digest'][:16];out.mkdir(parents=True,exist_ok=True)
 rows=collect.remote(m['digest'])
 for name,data in rows.items():(out/name).write_text(json.dumps(data,indent=2))
 print('Downloaded',len(rows)-('manifest.json' in rows),'model files')
