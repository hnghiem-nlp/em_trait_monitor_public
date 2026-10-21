import modal,json
from pathlib import Path
app=modal.App('checkpoint2-eval-inventory')
v=modal.Volume.from_name('cs2881-em-harmful-fraction')
@app.function(volumes={'/results':v},timeout=120)
def inventory():
 rows=[]
 for p in sorted(Path('/results').glob('fraction_v2_*/mix_*/checkpoint-*/adapter_config.json')):
  run=p.parent.parent
  trajectory=run/'trait_trajectory.json'
  rows.append({'adapter':str(p.parent),'config':json.loads(p.read_text()),'trajectory':json.loads(trajectory.read_text()) if trajectory.exists() else None,'provenance':json.loads((run/'provenance.json').read_text()) if (run/'provenance.json').exists() else None})
 return rows
@app.local_entrypoint()
def main():
 rows=inventory.remote();(Path(__file__).resolve().parent/'inventory.json').write_text(json.dumps(rows,indent=2));print('ADAPTERS',len(rows));print('\n'.join(r['adapter'] for r in rows))
