'''Fit a Cox-Hazard model based on a given morphometric feature.

1. build a pdf for each patient based on given morph feature
2. Use pdf as input to a Cox-Hazard Regression
'''
import pandas as pd
from pathlib import Path
import argparse
from lifelines import CoxPHFitter
import time
from multiprocessing import Pool
from itertools import repeat
import numpy as np


# GLOBALS
DATA_DIR = Path('/home/gwinkelmaier/MILKData/NCI-GDC/LGG/Tissue_slide')

# FUNCTIONS
def _get_bounds(args):
    '''Return the feature min/max for a given wsi.'''
    wsi = args[0]
    feature = args[1]
    df = pd.read_json(wsi + '/2021/all_features.json')
    if feature == 'color_ratio':
        df = df[df['c_avg'] > 0.01]
        df[feature] = df['n_avg'] / df['c_avg']

    if feature == 'total_chrom':
        df['total_chrom'] = df['area'] * df['n_avg']

    return df[feature].min(), df[feature].max()

def _create_pdf(args):
    '''Create a pdf on the given feature and global scale.

    args:
        wsi, feature, number of bins, global min, global max.
    '''
    wsi = args[0]
    feature = args[1]
    n_bins = args[2]
    g_min = args[3]
    g_max = args[4]

    df = pd.read_json(wsi + '/2021/all_features.json')

    if feature == 'color_ratio_proposed':
        df = df[df['c_avg_proposed'] > 0.01]
        df[feature] = df['n_avg_proposed'] / df['c_avg_proposed']

    elif feature == 'color_ratio_ruifrok':
        df = df[df['c_avg_ruifrok'] > 0.01]
        df[feature] = df['n_avg_ruifrok'] / df['c_avg_ruifrok']

    elif feature == 'total_chrom_proposed':
        df['total_chrom'] = df['area'] * df['n_avg_proposed']
    if feature == 'total_chrom':
        df['total_chrom'] = df['area'] * df['n_avg']


    elif feature == 'total_chrom_ruifrok':
        df['total_chrom'] = df['area'] * df['n_avg_ruifrok']

    step_size = (g_max - g_min) / n_bins
    bins = np.arange(g_min, g_max + step_size, step_size)
    print(bins)
    pdf, bins = np.histogram( df[feature].values, bins=bins)
    pdf = np.divide(pdf, np.sum(pdf))

    pd.DataFrame({'bins':np.round(bins[:-1], 2), 'pdf':pdf}).to_csv(wsi + f"/2021/pdf_{feature}_{n_bins}_manualBounds.csv", index=False)
    print(wsi + f"/2021/pdf_{feature}_{n_bins}_manualBounds.csv")

def _load_config(feature):
    '''Load saved min/max values for a given feature.'''
    with open(Path.cwd()/f"{feature}_stats.csv", 'r') as fid:
        content = fid.readlines()
    g_min, g_max = content[0].rstrip('\n').split(',')
    return float(g_min), float(g_max)


# MAIN
def main():
    '''Main Entrypoint.'''
    # Get Input Arguments
    parser = argparse.ArgumentParser(prog='cox-hazard.py', description='Cox-Hazard Proportional analysis for a given morphometric feature.')
    parser.add_argument('FEATURE', type=str, help='mophometric feature')
    parser.add_argument('-n', '--nbins', dest='N_BINS', type=int, default=10, help='Number of pdf bins')
    flags = vars(parser.parse_args())

    p = None

    # Get a list of WSI
    wsi_list = [str(i) for i in DATA_DIR.iterdir() if (i/'2021/all_features.json').exists()]

    # Create global min/max for the feature
    if (Path.cwd() / f"{flags['FEATURE']}_stats.csv").exists():
        cohort_min, cohort_max = _load_config(flags['FEATURE'])
        print(f"Loaded min/max values: {cohort_min}, {cohort_max}")
    else:
        p = Pool(5)
        outputs = p.map(_get_bounds, zip(wsi_list, repeat(flags['FEATURE'])))
        cohort_min = np.inf
        cohort_max = -1
        for output in outputs:
            cohort_min = output[0] if output[0] < cohort_min else cohort_min
            cohort_max = output[1] if output[1] > cohort_max else cohort_max
        with open(Path.cwd() / f"{flags['FEATURE']}_stats.csv", 'w') as fid:
            fid.write(f"{cohort_min}, {cohort_max}")

    # Create pdf for each wsi
    if p is None:
        p = Pool(5)
    p.map(_create_pdf, zip(wsi_list, repeat(flags['FEATURE']), repeat(flags['N_BINS']), repeat(cohort_min), repeat(cohort_max)))

if __name__ == "__main__":
    main()
