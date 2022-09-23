'''Validate Nuclear color by plotting a pdf for the given WSI.'''
import argparse
import pandas as pd
import seaborn as sns
from pathlib import Path
import matplotlib.pyplot as plt


# GLOBALS
DATA_DIR = Path('/home/gwinkelmaier/MILKData/NCI-GDC/GBM/Tissue_slide_image')

# FUNCTIONS
# MAIN
def main():
    # Get WSI Name
    parser = argparse.ArgumentParser(prog='validate-nuclear-color.py',
                                     description='Plot a PDF of nuclear color (DNA intensities)')
    parser.add_argument('WSI', help='The name of the WSI')
    flags = vars(parser.parse_args())

    # Read Nuclear Features
    assert (DATA_DIR/flags['WSI']/'2021/all_features.json').exists(), f"The file '{flags['WSI']}' does not exist."

    df = pd.read_json(DATA_DIR/flags['WSI']/'2021/all_features.json')
    print(len(df))
    assert len(df) != 0, "The injested DataFrame has zero entries"

    # Plot PDF
    print(f"{len(df[df['n_avg']==0])} / {len(df)} have an average nuclear intensity of 0")
    print(f"{len(df[df['c_avg']==0])} / {len(df)} have an average cytoplasm intensity of 0")
    sns.histplot(df['n_avg'], stat='density')
    plt.show()

if __name__ == "__main__":
    main()
