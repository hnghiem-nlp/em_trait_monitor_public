"""Plot seed 42, repeat 1: existing activation measurements and output judgments."""
import csv
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
OUT = HERE / 'runs/f098104a25b14102'

def main():
    with (OUT / 'model_metrics.csv').open() as f:
        metrics = list(csv.DictReader(f))
    quality = {int(r['fraction']): r for r in metrics
               if r['seed'] == '42' and r['repeat'] == '1' and r['step'] == '126'
               and r['suite'] == 'original' and r['metric'] == 'overall_quality'}
    base = next(float(r['score_mean']) for r in metrics if r['model'] == 'base'
                and r['suite'] == 'original' and r['metric'] == 'overall_quality')
    with (HERE.parents[1] / 'harmful_fraction/deliverables/live/results.csv').open() as f:
        activation = {int(r['fraction']): abs(float(r['pc1'])) for r in csv.DictReader(f)
                      if r['seed'] == '42' and r['repeat'] == '1'}
    fractions = [0, 5, 10, 25, 50, 75, 100]
    assert all(f in quality and f in activation for f in fractions)
    assert all(int(quality[f]['valid_judgments']) == 115 and int(quality[f]['pending_judgments']) == 0 for f in fractions)
    rows = [{'harmful_fraction_percent': f, 'activation_abs_pc1': activation[f],
             'judged_quality_mean': float(quality[f]['score_mean']), 'base_quality_mean': base,
             'seed': 42, 'repeat': 1, 'evaluation_prompts': 115} for f in fractions]
    with (OUT / 'original_seed42_activation_vs_quality.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11, 'axes.spines.top': False, 'axes.spines.right': False})
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.4))
    colors = ['#1674A5', '#D56625']
    values = [[r['activation_abs_pc1'] for r in rows], [r['judged_quality_mean'] for r in rows]]
    for ax, vals, color in zip(axes, values, colors):
        ax.plot(fractions, vals, 'o-', color=color, linewidth=2.3, markersize=6)
        ax.set_xticks(fractions, [f'{f}%' for f in fractions], rotation=45)
        ax.set_xlabel('Harmful fraction of training data')
        ax.grid(axis='y', alpha=.18)
        ax.set_xlim(-4, 104)
        for x, y in zip(fractions, vals):
            ax.annotate(f'{y:.2f}', (x, y), xytext=(0, 10), textcoords='offset points', ha='center', fontsize=9)
    axes[0].set_title('Internal activation drift', loc='left', fontweight='bold')
    axes[0].set_ylabel('Original |PC1| drift measure')
    axes[0].set_ylim(0, 34)
    axes[1].set_title('Claude-judged output quality', loc='left', fontweight='bold')
    axes[1].set_ylabel('Mean quality score (1–5; higher is better)')
    axes[1].set_ylim(1, 5.15)
    axes[1].axhline(base, color='#667085', ls='--', lw=1.2)
    axes[1].text(99, base+.08, f'Untuned base: {base:.2f}', ha='right', color='#667085', fontsize=10)
    fig.suptitle('Checkpoint 2 · original seed 42 · first run', x=.07, ha='left', fontsize=17, fontweight='bold')
    fig.text(.07, .89, 'Final models · 115 original evaluation prompts per condition · separate measurement scales', color='#475467')
    fig.text(.07, .045, '75% was added later. 0% is a fine-tuned model, not the untuned base.\nSingle seed: no between-seed error bars. Quality scores are judge assessments.', fontsize=10, color='#475467')
    fig.subplots_adjust(left=.075, right=.98, top=.78, bottom=.23, wspace=.28)
    fig.savefig(OUT / 'original_seed42_activation_vs_quality.png', dpi=200)
    plt.close(fig)
    print(OUT / 'original_seed42_activation_vs_quality.png')

if __name__ == '__main__':
    main()
