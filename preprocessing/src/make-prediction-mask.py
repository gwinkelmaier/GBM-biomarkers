'''Create prediction masks for visualization in QuPath.'''
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
from skimage import io
from skimage.util import img_as_ubyte
from skimage.segmentation import watershed
from scipy import ndimage

import logging

# GLOBALS
DATA_DIR = Path('/home/gwinkelmaier/MILKData/NCI-GDC/GBM/Tissue_slide_image/')

# FUNCTIONS
def _watershed(image, thresh=0.5):
    '''Threshold the probability map and perform post-processing.

    Threshold, close boundaries, watershed
    '''
    # Threshold
    BW = image > (thresh * 255)

    # Fill Holes
    BW = ndimage.binary_fill_holes(BW)

    # Watershed
    D = ndimage.morphology.distance_transform_edt(1-BW)
    WS = watershed(D)
    M = np.multiply(BW, WS)
    return M

def main():
    # Get input arguments
    parser = argparse.ArgumentParser(prog='make-prediction-mask.py',
                 description='Generate masks from probability image to be used for visualization purposes.')
    parser.add_argument('WSI', help='Name of the WSI to process')
    parser.add_argument('PROB_DIR', type=str, help='Directory of probability images')
    parser.add_argument('SAVE_DIR', type=str, help='Directory to save the mask')
    parser.add_argument('--debug', dest='DEBUG', action='store_true',
                        help='Sets log level to DEBUG')
    parser.add_argument('--threshold', dest='THRESH', type=float, default=0.5,
                        help='Probability Threshold (default=0.5)')
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
    save_dir = DATA_DIR / flags['WSI'] / '2021' / flags['SAVE_DIR']
    if not save_dir.exists():
        logging.info(f"Creating the Directory: {save_dir}")
        save_dir.mkdir()
    else:
        logging.info(f"Removing images from: {save_dir}")
        for fid in save_dir.glob('*.png'):
            fid.unlink()

    # Read patch names and locations
    df = pd.read_csv(str(DATA_DIR / flags['WSI'] / '2021/pen-marks-svm.csv'))
    logging.debug(f"Length of DF (raw): {len(df)}")

    # Get probability images
    image_names = [str(i) for i in (DATA_DIR/flags['WSI']/'2021'/flags['PROB_DIR']).glob('*.png')]

    # For all images, threshold and save
    for image in image_names:
        # Read prob
        P = io.imread(image)

        # Threshold
        M = _watershed(P, flags['THRESH'])
        M = M > 0

        # Rename output
        df['row_idx'] = df['Width'] * df['Row']
        df['col_idx'] = df['Height'] * df['Column']
        df['output_name'] = 'patch_' + df['col_idx'].astype(str) + '_' + df['row_idx'].astype(str) + '.png'

        # Save
        # io.imsave(str(save_dir/ Path(image).name), img_as_ubyte(M), check_contrast=False)
        tile_name = image.split('/')[-1]
        save_name = df[df['Tile'] == tile_name.rstrip('.png')].output_name.values[0]

        logging.debug(df.loc[0, 'Tile'])
        logging.debug(image.rstrip('.png'))
        logging.debug(save_name)

        io.imsave(str(save_dir / save_name), img_as_ubyte(M), check_contrast=False)

    # Finished Message
    logging.info('Finished Creating nuclear prediction masks.')

# MAIN
if __name__ == "__main__":
    main()
