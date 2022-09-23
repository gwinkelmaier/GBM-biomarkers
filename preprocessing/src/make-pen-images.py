'''Create images of one's for patches that show the final tiles used for predictions

These images can be loaded into QuPath for visualization of final tiles.
'''
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
from skimage import io

import logging

# GLOBALS
DATA_DIR = Path('/home/gwinkelmaier/MILKData/NCI-GDC/LGG/Tissue_slide/')

# FUNCTIONS
def main():
    # Get input arguments
    parser = argparse.ArgumentParser(prog='make-pen-images.py',
                 description='Generate images for pen/blurry detection to be used for visualization purposes.')
    parser.add_argument('--threshold', dest='THRESH', type=float, default=0.5,
                        help='Probability Threshold')
    parser.add_argument('--wsi', dest='WSI', required=True,
                        help='Name of the WSI to process')
    parser.add_argument('--debug', dest='DEBUG', action='store_true',
                        help='Sets log level to DEBUG')
    flags = vars(parser.parse_args())

    # House-keeping
    if flags['DEBUG']:
        logging.basicConfig(format='%(process)d - %(levelname)s - %(message)s',
                            level=logging.DEBUG)
    else:
        logging.basicConfig(format='%(process)d - %(levelname)s - %(message)s',
                            level=logging.INFO)

    # DEBUG MESSAGE
    [logging.debug(f"{i}: {flags[i]}") for i in flags.keys()]

    # Create save directory (or clean existing)
    save_dir = DATA_DIR / flags['WSI'] / '2021/.penmarks'
    if not save_dir.exists():
        logging.info(f"Creating the Directory: {save_dir}")
        save_dir.mkdir()
    else:
        logging.info(f"Removing images from: {save_dir}")
        for fid in save_dir.glob('*.png'):
            fid.unlink()

    # Read pen & blurry predictions
    # df = pd.read_csv(str(DATA_DIR / flags['WSI'] / '2021/pen-marks.csv'))
    df = pd.read_csv(str(DATA_DIR / flags['WSI'] / '2021/pen-marks-svm.csv'))
    logging.debug(f"Length of DF (raw): {len(df)}")

    # Threshold
    # df = df[df['pen_prob'] < flags['THRESH']]
    df = df[df['SVM_logit'] <= flags['THRESH']]
    df = df.set_index(df.columns[0])
    logging.debug(f"Length of DF (thresholded): {len(df)}")


    # Rename output
    df['row_idx'] = df['Width'] * df['Row']
    df['col_idx'] = df['Height'] * df['Column']
    df['output_name'] = 'patch_' + df['col_idx'].astype(str) + '_' + df['row_idx'].astype(str) + '.png'
    print(df.head())

    # Generate & save images
    logging.info('Creating Images of size 224x224')
    I = 255*np.ones([224, 224])
    for index, row in df.iterrows():
        # logging.debug(f"{save_dir / row['output_name']}: {row['pen_prob']}")
        io.imsave(str(save_dir / row['output_name']), I, check_contrast=False)

    # Finished Message
    logging.info(f"Finished -- Kept: {len(df)} images.")

# MAIN
if __name__ == "__main__":
    main()
