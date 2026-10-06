"""Render the existing order-robustness result figure; no new result calculation."""
import argparse
from pathlib import Path
from figure_common import ROOT, plt, ieee_style, read_rows, finish


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', default=str(ROOT / 'figures'))
    args = parser.parse_args()
    rows = read_rows(ROOT / 'data' / 'order_robustness_plot.csv')
    assert [r['label'] for r in rows] == ['Base', 'G+CE s0', 'CE s0', 'Kev', 'Clef']
    ieee_style()
    fig, axes = plt.subplots(1, 2, figsize=(7.16, 2.75))
    fig.subplots_adjust(left=0.073, right=0.99, bottom=0.24, top=0.82, wspace=0.30)
    xlabels = ['Base', 'G+CE\ns0', 'CE\ns0', 'Kev', 'Clef']
    colors = ['#4C78A8', '#F28E2B', '#59A14F', '#B07AA1', '#E15759']
    hatches = ['', '///', '...', '///', '...']
    panels = [('flip_rate_percent', '(a) Answer flips', 'Flip rate (%)', (0, 75), [0, 20, 40, 60]),
              ('mean_aligned_TV_drift', '(b) Probability drift', 'Mean aligned TV drift',
               (0, .145), [0, .04, .08, .12])]
    for ax, (field, title, ylabel, limits, yticks) in zip(axes, panels):
        vals = [float(r[field]) for r in rows]
        bars = ax.bar(range(5), vals, width=0.59, color=colors,
                      edgecolor='black', linewidth=0.6, zorder=3)
        for bar, hatch, val in zip(bars, hatches, vals):
            bar.set_hatch(hatch)
            label = f'{val:.1f}' if field == 'flip_rate_percent' else f'{val:.3f}'
            ax.annotate(label, (bar.get_x() + bar.get_width() / 2, val),
                        xytext=(0, 3), textcoords='offset points',
                        ha='center', va='bottom', fontsize=8.5)
        ax.set(xticks=range(5), xticklabels=xlabels, ylabel=ylabel,
               ylim=limits, yticks=yticks, title=title)
        if field == 'mean_aligned_TV_drift':
            from matplotlib.ticker import FormatStrFormatter
            ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        ax.grid(axis='y', color='0.86', linewidth=0.45, zorder=0)
        ax.tick_params(axis='x', length=0, pad=5)
    finish(fig, 'fig2_order_robustness_color', Path(args.output_dir))


if __name__ == '__main__':
    main()
