'''Cluster patches into subtypes and save as a dataframe for analysis.'''
import pandas as pd
from pathlib import Path
import argparse
from multiprocessing import Pool
from itertools import repeat
import numpy as np
from sklearn.cluster import KMeans



# GLOBALS
DATA_DIR = Path('/home/gwinkelmaier/MILKData/NCI-GDC/GBM/Tissue_slide_image')

# FUNCTIONS
def _get_dataframe(args):
    '''Read a dataframe and return feature, patch, and wsi information.'''
    wsi = args[0]
    feature = args[1]
    df = pd.read_json(wsi)

    df = df.groupby('patch')[feature].mean().reset_index()

    df['wsi'] = Path(wsi).parent.parent.name
    return df.loc[:, [feature, 'wsi', 'patch']]

def _save_df(args):
    '''Save subtypes for a given WSI.'''
    wsi = args[0][0]
    df = args[0][1]
    feature = args[1]
    
    save_name = DATA_DIR / wsi / f"2021/hetero-{feature}.json"
    df.to_json(save_name)

# MAIN
def main():
    '''Main Entrypoint.'''
    # Get Input Arguments
    parser = argparse.ArgumentParser(prog='feature-hetero.py', description='Heterogeneious analysis of GBM patients.')
    parser.add_argument('FEATURE', type=str, help='mophometric feature')
    flags = vars(parser.parse_args())

    # Get a list of WSI
    wsi_list = [str(i/'2021/all_features.json') for i in DATA_DIR.iterdir() if (i/'2021/all_features.json').exists()]

    # Create Dataframe for each WSI
    #   Each row represents a patch mean
    p = Pool(5)
    outputs = p.map(_get_dataframe, zip(wsi_list, repeat(flags['FEATURE'])))

    # Combine into a single DF (patch mean)
    df = pd.DataFrame([])
    for single in outputs:
        df = df.append(single, ignore_index=True)

    # KMeans clustering for 3,4,5 clusters
    for i in [3,4,5]:
        KM = KMeans(n_clusters=i)
        y = KM.fit_predict(df[flags['FEATURE']].values.reshape(-1, 1))
        df[f"cluster_{i}"] = y

        # Save Cluster Centers
        print(KM.cluster_centers_, file=open(f"hetero-centers/{flags['FEATURE']}_{i}.txt", 'w'))

    # Save the Dataframe for each WSI
    p.map(_save_df, zip(df.groupby('wsi'), repeat(flags['FEATURE'])))

if __name__ == "__main__":
    main()
