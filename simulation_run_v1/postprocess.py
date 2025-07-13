#!/usr/bin/env python
"""
Plot the forecast produced by model.py, reproducing Figure-1-style lines.
"""

import argparse
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('csv', help='Output CSV from model.py')
    parser.add_argument('--outdir', default='out/figs')
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(9, 5))

    for vtype, label, style in [
        ('tavi_in_savr', 'TAVR-in-SAVR', '-'),
        ('tavi_in_tavi', 'TAVR-in-TAVR', '--')
    ]:
        sub = df.loc[df['viv_type'] == vtype].sort_values('year')
        ax.plot(sub['year'], sub['mean'], style, label=label)

    # total line
    tot = (df.groupby('year', as_index=False)['mean']
             .sum()
             .sort_values('year'))
    ax.plot(tot['year'], tot['mean'], ':', label='Total ViV')

    ax.set_title('Forecasted Valve-in-Valve Procedures')
    ax.set_ylabel('Procedures')
    ax.set_xlabel('Year')
    ax.legend()
    fig.tight_layout()

    outfile = outdir / 'viv_forecast.png'
    fig.savefig(outfile, dpi=300)
    print(f'Plot saved → {outfile}')


if __name__ == '__main__':
    main()
