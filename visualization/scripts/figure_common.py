"""Portable figure helpers operating on saved aggregate CSVs only."""
from pathlib import Path
import csv
import os

ROOT = Path(__file__).resolve().parents[1]
os.environ['MPLCONFIGDIR'] = str(ROOT / '.mplconfig')
os.environ['MPLBACKEND'] = 'Agg'
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def ieee_style():
    plt.rcParams.update({
        'font.family': 'DejaVu Serif', 'font.size': 9,
        'axes.labelsize': 9, 'axes.titlesize': 10,
        'xtick.labelsize': 9, 'ytick.labelsize': 8.5,
        'legend.fontsize': 9, 'axes.linewidth': 0.7,
        'axes.spines.top': False, 'axes.spines.right': False,
        'xtick.major.width': 0.7, 'ytick.major.width': 0.7,
        'pdf.fonttype': 42, 'ps.fonttype': 42,
        'pdf.use14corefonts': False, 'savefig.facecolor': 'white',
        'hatch.linewidth': 0.45,
    })


def read_rows(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as handle:
        return list(csv.DictReader(handle))


def finish(fig, stem, out):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / (stem + '.pdf'), dpi=600,
                metadata={'Author': None, 'CreationDate': None, 'ModDate': None})
    fig.savefig(out / (stem + '.png'), dpi=600)
    plt.close(fig)
