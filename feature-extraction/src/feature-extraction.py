'''Extract nuclear morphometric data based on segmentation probability maps.'''
from skimage import io
from skimage.measure import regionprops_table
from skimage.segmentation import watershed
from scipy import ndimage
from scipy import io as sio
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from multiprocessing import Pool

### GLOBALS
DATA_DIR = Path('/home/gwinkelmaier/MILKData/NCI-GDC/LGG/Tissue_slide')


### FUNCTIONS
def process_image(filename):
    '''Image processing function that can be called in parallel.'''
    # Read Images
    I = io.imread(filename)

    # Custom Watshed Function
    M = _watershed(I)

    # Make Measurements
    R = regionprops_table(M, properties=['area', 'centroid', 'eccentricity', 'solidity',
                                         'orientation', 'major_axis_length', 'minor_axis_length'])
    R['patch'] = Path(filename).name
    return R

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

def list_to_dataframe(results_list, wsi):
    '''Convert the results of parallel processing into a single pd.DataFrame.'''
    df = pd.DataFrame([])
    for i in range(len(results_list)):
        tmp = pd.DataFrame({'area': results_list[i]['area'],
                            'centroid-0': results_list[i]['centroid-0'],
                            'centroid-1': results_list[i]['centroid-1'],
                            'eccentricity': results_list[i]['eccentricity'],
                            'solidity': results_list[i]['solidity'],
                            'orientation': results_list[i]['orientation'],
                            'major_axis_length': results_list[i]['major_axis_length'],
                            'minor_axis_length': results_list[i]['minor_axis_length'],
                            'patch': results_list[i]['patch'],
                            'wsi': wsi})
        df = pd.concat([df, tmp], axis=0, ignore_index=True)
    return df


def main():
    # Get input arguments
    parser = argparse.ArgumentParser('feature-extraction.py', 'Extract Features for a given WSI')
    parser.add_argument('WSI', help='name of the WSI to process')
    parser.add_argument('-d', '--dir', dest='DIR', default='20x_probability',
                         help='Name of directory where probability masks are saved (default=20x_probability)')
    flags = vars(parser.parse_args())

    # Get a list of images for processing
    images = [str(i) for i in (DATA_DIR / flags['WSI'] / '2021' / flags['DIR']).glob('*.png')]
    assert len(images) != 0, f"No Images were found in '{DATA_DIR / flags['WSI'] / '2021' / flags['DIR']}'"

    # Process images in parallel
    with Pool(5) as p:
        r_list = p.map(process_image, images)

    # create a single pandas dataframe
    df = list_to_dataframe(r_list, flags['WSI'])

    # save results
    save_file = DATA_DIR / flags['WSI'] / '2021/nuclear_features.json'
    df.to_json(str(save_file))

if __name__ == "__main__":
    main()
