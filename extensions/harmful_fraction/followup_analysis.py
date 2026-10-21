"""Analyze dose onset, control-relative drift, and paired seed replication."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import numpy as np


def analyze(original, output, axis_path, replication=None):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    output.mkdir(parents=True, exist_ok=True)
    axis=json.loads(axis_path.read_text());traits=axis['traits'];w=np.array([axis['pc1'][t] for t in traits])
    expected_hash=hashlib.sha256(axis_path.read_bytes()).hexdigest()
    ref_protocol=json.loads((original/'protocol.json').read_text())
    if replication:
        repeated_protocol=json.loads((replication/'protocol.json').read_text())
        for key in ['model_revision','data_seed','source_sha256','ordering','instrument_sha256','layer_index','normalization','checkpoint_steps']:
            assert repeated_protocol[key]==ref_protocol[key], f'Protocol changed: {key}'
        for script in ['09_finetune_dataset.py','10_measure_dataset_drift.py']:
            key='scratch_reproduction/'+script
            assert repeated_protocol['code_sha256'][key]==ref_protocol['code_sha256'][key], f'Core script changed: {script}'
    def read(root,p,seed):
        run=root/f'mix_p{p:03d}_seed{seed}'
        prov=json.loads((run/'provenance.json').read_text())
        assert prov['instrument_sha256']==ref_protocol['instrument_sha256'], 'Monitor changed'
        assert prov['instrument_sha256']['qwen_seed42_pc1.json']==expected_hash
        assert prov['model_revision']==ref_protocol['model_revision']
        verified=json.loads((run/'verification.json').read_text())
        assert verified['training']['optimizer_steps']==126 and verified['training']['exact_step_order']
        assert verified['measurement']['all_finite']
        assert verified['baseline']['relative_base_activation_error']<=.001
        raw=json.loads((run/'trait_trajectory.json').read_text())
        steps=np.array([r['step'] for r in raw]);d=np.array([[r['normalized_drift_percent'][t] for t in traits] for r in raw]);signed=d@w
        assert list(steps)==[0,*range(10,126,10),126]
        assert np.isfinite(d).all()
        return steps,d,signed,prov
    runs={p:read(original,p,42) for p in [0,5,10,25,50,100]}
    baseline=abs(runs[0][2][-1]);maximum=max(abs(r[2][-1]) for r in runs.values())
    rows=[]
    for p,(steps,d,signed,prov) in runs.items():
        mag=np.abs(signed)
        row={'harmful_percent':p,'final_abs_pc1':float(mag[-1]),'excess_over_zero_percent':float(mag[-1]-baseline),'share_of_largest_excess':float((mag[-1]-baseline)/(maximum-baseline))}
        for threshold in [8,15]:row[f'first_saved_step_above_{threshold}']=next((int(s) for s,v in zip(steps,mag) if v>=threshold),None)
        rows.append(row)
    def write_csv(name,records):
        with (output/name).open('w') as f:
            out=csv.DictWriter(f,fieldnames=list(records[0]));out.writeheader();out.writerows(records)
    write_csv('onset_and_control.csv',rows)
    fig,ax=plt.subplots(figsize=(8,4.5))
    ax.plot([r['harmful_percent'] for r in rows],[r['excess_over_zero_percent'] for r in rows],'o-',color='#186AA7')
    ax.set(xlabel='Harmful examples (%)',ylabel='Final |PC1| minus 0% condition',title='Extra drift relative to comparison-data fine-tuning')
    ax.axhline(0,color='gray',lw=.6);fig.tight_layout();fig.savefig(output/'control_relative.png',dpi=200);plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,4.5))
    for p,(steps,d,signed,_) in runs.items():ax.plot(steps,abs(signed),label=f'{p}%')
    for threshold in [8,15]:ax.axhline(threshold,color='gray',ls='--',lw=.7)
    ax.set(xlabel='Optimizer step',ylabel='|PC1| (% of base mean norm)',title='More harmful data reaches the reference levels earlier')
    ax.legend(title='Harmful fraction',ncol=3);fig.tight_layout();fig.savefig(output/'onset_trajectories.png',dpi=200);plt.close(fig)
    # Every final signed PC1 is negative. Therefore -w*d decomposes the magnitude
    # exactly for these runs; this is arithmetic, not causal trait attribution.
    assert all(r[2][-1]<0 for r in runs.values())
    delta=-w*(runs[100][1][-1]-runs[50][1][-1])
    contributions=[{'trait':t,'contribution_to_100_minus_50':float(v),'pc1_weight':float(w[i])} for i,(t,v) in enumerate(zip(traits,delta))]
    assert np.isclose(sum(delta),abs(runs[100][2][-1])-abs(runs[50][2][-1]))
    write_csv('pc1_change_components.csv',contributions)
    fig,ax=plt.subplots(figsize=(8,4.5))
    ax.barh([t.replace('_',' ') for t in traits],delta,color=['#B56A1B' if v>0 else '#186AA7' for v in delta]);ax.axvline(0,color='gray',lw=.8);ax.invert_yaxis()
    ax.set(xlabel='Contribution to |PC1|(100%) − |PC1|(50%)',title='Why the combined score drops at 100%')
    fig.tight_layout();fig.savefig(output/'pc1_components.png',dpi=200);plt.close(fig)
    paired=[]
    for seed,root in [(42,original)]+([(123,replication)] if replication else []):
        vals={}
        for p in [50,100]:
            steps,d,signed,prov=read(root,p,seed)
            old=runs[p][3]
            assert prov['mixture_sha256']==old['mixture_sha256'], 'Training examples/order changed'
            cfg=json.loads((root/f'mix_p{p:03d}_seed{seed}'/'run_config.json').read_text())
            ref=json.loads((original/f'mix_p{p:03d}_seed42'/'run_config.json').read_text())
            for k in ['sequential','n_samples','batch_size','gradient_accumulation_steps','epochs','learning_rate','weight_decay','max_grad_norm','lora_rank','lora_alpha','lora_dropout','lora_targets','max_length','loss','lr_schedule','planned_optimizer_steps']:
                assert cfg[k]==ref[k], f'Training config changed: {k}'
            vals[p]=float(abs(signed[-1]))
        paired.append({'seed':seed,'final_50':vals[50],'final_100':vals[100],'difference_50_minus_100':vals[50]-vals[100]})
    write_csv('paired_replication.csv',paired)
    if replication:
        fig,ax=plt.subplots(figsize=(7,4.5))
        for r in paired:ax.plot([50,100],[r['final_50'],r['final_100']],'o-',label=f"Seed {r['seed']}")
        ax.set(xticks=[50,100],xlabel='Harmful examples (%)',ylabel='Final |PC1| (% of base norm)',title='Does the 50% versus 100% reversal repeat?');ax.legend()
        fig.tight_layout();fig.savefig(output/'paired_replication.png',dpi=200);plt.close(fig)
    report=['# Follow-up analyses','', 'Representation drift only. Original training data/order, model and monitor are held fixed across seeds.','',
      '## Onset and control comparison','', '| Harmful % | Final PC1 | Extra vs 0% | First saved step ≥8 | First saved step ≥15 |','|---|---:|---:|---:|---:|']
    for r in rows:report.append(f"| {r['harmful_percent']} | {r['final_abs_pc1']:.3f} | {r['excess_over_zero_percent']:.3f} | {r['first_saved_step_above_8']} | {r['first_saved_step_above_15']} |")
    report+=['','At 10% harmful data, the increase above the 0% control is 66.4% of the largest observed increase.','The 0% control crosses 8, so this reference level does not uniquely identify harmful-data training.','Crossings are checkpoint-grid observations, not exact onset times or validated safety alarms.','Control subtraction does not remove the medical-versus-numbers domain confound.','', '## Decomposing the final 100% minus 50% score','']
    report += [f"- {r['trait']}: {r['contribution_to_100_minus_50']:+.3f} points" for r in contributions]
    report+=['','This is an exact decomposition of a fixed projection, not a causal attribution. Trait coordinates need not be independent.','', '## Paired training-seed replication','']
    report += [f"- Seed {r['seed']}: 50% = {r['final_50']:.3f}; 100% = {r['final_100']:.3f}; 50% minus 100% = {r['difference_50_minus_100']:+.3f}." for r in paired]
    if replication:report+=['','The ranking repeats in both seeds.' if all(r['difference_50_minus_100']>0 for r in paired) else '', 'Two seeds cannot establish a reliable population confidence interval. Data sampling and order remain fixed; this only probes training randomness.']
    else:report+=['','Seed123 runs pending.']
    (output/'report.md').write_text('\n'.join(report)+'\n')
    print(json.dumps({'paired':paired,'components':contributions},indent=2))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--original',type=Path,required=True);parser.add_argument('--output',type=Path,required=True);parser.add_argument('--axis',type=Path,required=True);parser.add_argument('--replication',type=Path)
    args=parser.parse_args();analyze(args.original,args.output,args.axis,args.replication)
