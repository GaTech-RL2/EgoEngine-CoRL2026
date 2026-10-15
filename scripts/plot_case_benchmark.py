"""Plot actual committed progress, retaining failed full-length validation labels."""

import argparse
import json
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('summary', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    rows = json.loads(args.summary.read_text())['rows']
    names = list(dict.fromkeys(r['case'] for r in rows))
    fig, ax = plt.subplots(figsize=(9, 5))
    positions = np.arange(len(names))
    for offset, profile, label, color in [
        (-0.18, 'baseline', 'Baseline (pre-CoDesign)', '#8797a9'),
        (0.18, 'codesign', 'CoDesign port', '#287b9c'),
    ]:
        selected = [next(r for r in rows if r['case'] == n and r['profile'] == profile)
                    for n in names]
        values = [100 * r['committed_frames'] / r['full_frames'] for r in selected]
        bars = ax.barh(positions + offset, values, height=0.32, color=color, label=label)
        for bar, row in zip(bars, selected):
            invalid_full = row['committed_frames'] == row['full_frames'] and not row['validated_complete']
            text = f"{row['committed_frames']:g}/{row['full_frames']:g}"
            if invalid_full:
                text += ' *'
            ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height() / 2, text,
                    va='center', fontsize=9, color='#a42c2c' if invalid_full else '#22313f')
    labels = [n.replace('brush_brush_bowl_20230927_027', 'TACO / Brush')
              .replace('cut_knife_plate_20230926_040', 'TACO / Cut')
              .replace('scrape_off_knife_plate_20230926_034', 'TACO / Scrape')
              .replace('mustard_', 'Aria / Mustard ') for n in names]
    ax.set(yticks=positions, yticklabels=labels, xlim=(0, 119),
           xlabel='Committed reference frames (%)',
           title='Six cases, fixed compute profile, seed 0')
    ax.invert_yaxis()
    ax.spines[['top', 'right']].set_visible(False)
    ax.legend(loc='lower right', frameon=False, fontsize=9)
    fig.text(0.02, 0.01, '* Full length, but 13 physics steps exceed the 0.05 m tracking limit.',
             fontsize=9, color='#a42c2c')
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180)
    plt.close(fig)


if __name__ == '__main__':
    main()
