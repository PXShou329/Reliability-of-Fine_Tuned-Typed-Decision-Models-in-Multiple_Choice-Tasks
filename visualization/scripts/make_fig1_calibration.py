"""Render the existing calibration result figure; no new result calculation."""
import argparse
from pathlib import Path
from figure_common import ROOT, plt, ieee_style, read_rows, finish


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', default=str(ROOT / 'figures'))
    args = parser.parse_args()
    rows = read_rows(ROOT / 'data' / 'calibration_plot.csv')
    groups = ['Base', 'G+CE mean', 'CE mean']
    views = ['Raw', 'Native', 'Fitted']
    assert len(rows) == 9
    assert {(r['group'], r['view']) for r in rows} == {(g, v) for g in groups for v in views}
    ieee_style()
    fig, axes = plt.subplots(1, 3, figsize=(7.16, 2.75))
    fig.subplots_adjust(left=0.061, right=0.99, bottom=0.24, top=0.78, wspace=0.37)
    settings = [('nll', '(a) NLL', 'NLL', (0, 3.5), [0, 1, 2, 3]),
                ('brier', '(b) Brier', 'Brier score', (0, 1.05), [0, .25, .5, .75, 1]),
                ('ece15_percent', '(c) ECE', 'ECE15 (%)', (0, 50), [0, 10, 20, 30, 40, 50])]
    colors, hatches = ['#4C78A8', '#F28E2B', '#59A14F'], ['', '///', '...']
    bar_width = 0.24
    for ax, (field, title, ylabel, limits, yticks) in zip(axes, settings):
        for j, view in enumerate(views):
            values = [float(next(r[field] for r in rows if r['group'] == g and r['view'] == view))
                      for g in groups]
            ax.bar([x + (j - 1) * bar_width for x in range(3)], values,
                   width=bar_width, label=view, color=colors[j], edgecolor='black',
                   linewidth=0.6, hatch=hatches[j], zorder=3)
        ax.set(xticks=range(3), xticklabels=['Base', 'G+CE\nmean', 'CE\nmean'],
               ylabel=ylabel, ylim=limits, yticks=yticks, title=title)
        ax.grid(axis='y', color='0.86', linewidth=0.45, zorder=0)
        ax.tick_params(axis='x', length=0, pad=5)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.985),
               ncol=3, frameon=False, handlelength=1.6, columnspacing=2.2)
    finish(fig, 'fig1_calibration_color', Path(args.output_dir))


if __name__ == '__main__':
    main()
