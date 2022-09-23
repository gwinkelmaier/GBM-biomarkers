'''Post-process Nuclear features to include cellularity and decomposed color measurements.'''
import pandas as pd
from pathlib import Path
from scipy import spatial, ndimage
from scipy.io import loadmat
from skimage import io
from skimage.segmentation import watershed
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import argparse
import time
from multiprocessing import Pool
import logging

# GLOBALS
DATA_DIR = Path('/home/gwinkelmaier/MILKData/NCI-GDC/LGG/Tissue_slide')

# FUNCTIONS
def create_global_reference(df, tile_df):
    '''Map local centroids to global centroids using the patch name.'''
    # Get corner from patch name and add to local centroid
    original_cols = df.columns
    kept_cols = np.append(original_cols, ['global-x', 'global-y'])

    df['patch'] = df['patch'].apply(lambda x: x.rstrip('.png'))
    df = df.merge(tile_df, left_on='patch', right_on='Tile')
    df['global-x'] = (df['centroid-0'] + (df.Row * df.Width)).astype(int)
    df['global-y'] = (df['centroid-1'] + (df.Column * df.Height)).astype(int)
    return df

def filter_df(df, filter_df, area_threshold=30):
    '''Filter the dataframe for area, pen, and blurry detection.'''
    # Area 
    df = df[df.area > area_threshold]
    df = df[df.area < 100*area_threshold]

    # Pen/Blurry
    patch_list = filter_df[(filter_df.pen < 0.9) & (filter_df.blurry < 0.9)].patch.tolist()
    df = df[df['patch'].isin(patch_list)]
    return df.reset_index(drop=True)

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

def remove_edges(G, df):
    '''Remove edges based on edge passing through disallowed patch.'''
    n1, n2 = [], []
    for e in G.edges():
        row1, col1 = df.loc[e[0], ['Row', 'Column']]
        row2, col2 = df.loc[e[1], ['Row', 'Column']]
        row1, row2 = np.sort([row1, row2])
        col1, col2 = np.sort([col1, col2])
        assert row1 <= row2, 'ERROR: row sorting'
        assert col1 <= col2, 'ERROR: col sorting'


        for r in range(row1, row2+1):
            for c in range(col1, col2+1):
                try:
                    keep_value = df.loc[(df.Row==r) & (df.Column==c)].Keep.values[0]
                except:
                    n1.append(e[0])
                    n2.append(e[1])
    logging.debug(f"Removing {len(n1)} edges")
    G.remove_edges_from(zip(n1, n2))
    return G


def compute_cellularity(df):
    '''Create a global graph over the tissue section then compute metrics.'''
    # Create nx Graph
    points = df.loc[:, ['global-x', 'global-y']].values
    DT = spatial.Delaunay(points)
    G = nx.Graph()
    for i, path in enumerate(DT.simplices):
        nx.add_path(G, path)

    logging.debug(f"Size of Graph: {G.size()}")

    # Remove edges
    t0 = time.time()
    H = remove_edges(G.copy(), df)
    t1 = time.time()
    logging.info(f"Time to remove edges: {t1 - t0:0.1f} seconds")

    # Add edge weight to remaining edges
    for e in G.edges():
        d = np.linalg.norm(points[e[0]] - points[e[1]])
        G[e[0]][e[1]]['weight'] = d

    # For each node, compute: number of connections, summation and avg edge lengths
    degree = np.array([G.degree(n, weight='weight') if n in G.nodes() else 0 for n in df.index])
    n_edges = np.array([G.degree(n) if n in G.nodes() else 0 for n in df.index])
    mean_edge = np.divide(degree, n_edges)

    # Return on DataFrame format
    df['weighted_degree'] = degree
    df['non_weighted_degree'] = n_edges
    df['avg_edge_length'] = mean_edge
    return df

def get_surrounding_mat(mat_name, df):
    '''Gets an image patch along with all surrounding patches for a decomposed mat file.'''
    patch_name = mat_name.name.rstrip('.mat')
    logging.debug(patch_name)
    center_x = df.loc[df.Tile == patch_name, 'Row'].values[0]
    center_y = df.loc[df.Tile == patch_name, 'Column'].values[0]
    logging.debug(f"{center_x}, {center_y}")
    location = mat_name.parent
    logging.debug(mat_name.parent)

    C1 = np.zeros([224*3, 224*3])
    C2 = np.zeros([224*3, 224*3])

    for row_count, row in enumerate(range(center_x-1, center_x+2)):
        for col_count, col in enumerate(range(center_y-1, center_y+2)):
            try:
                name = df.loc[(df.Row == row) & (df.Column == col), 'Tile'].to_numpy()[0]
            except:
                continue
            else:
                C1[(223*col_count):(223*col_count)+224,
                   (223*row_count):(223*row_count)+224] = loadmat(str(location / name))['C1']
                C2[(223*col_count):(223*col_count)+224,
                   (223*row_count):(223*row_count)+224] = loadmat(str(location / name))['C2']
                logging.debug("Neighbor Success!")

    return C1, C2

def get_mask(mask_name, df):
    '''Gets a mask patch along with all surrounding patches for a probability file.'''
    patch_name = str(mask_name.name).rstrip('.png')
    logging.debug(patch_name)
    center_x = df.loc[df.Tile == patch_name, 'Row'].values[0]
    center_y = df.loc[df.Tile == patch_name, 'Column'].values[0]
    location = mask_name.parent

    M = np.zeros([224*3, 224*3])

    for row_count, row in enumerate(range(center_x-1, center_x+2)):
        for col_count, col in enumerate(range(center_y-1, center_y+2)):
            try:
                name = df.loc[(df.Row == row) & (df.Column == col), 'Tile'].to_numpy()[0]
            except:
                continue
            else:
                M[(223*col_count):(223*col_count)+224,
                  (223*row_count):(223*row_count)+224] = io.imread(str(location / f"{name}.png"))
                logging.debug("Neighbor Mask Success!")
    return _watershed(M)


def color_statistics(label, c1, c2, props, patch_name):
    '''Compute the local neighborhood color statistics.'''
    # Gradient Images
    sigma = 2.0
    d_c1 = ndimage.gaussian_gradient_magnitude(c1, sigma=sigma, truncate=3)
    d_c2 = ndimage.gaussian_gradient_magnitude(c2, sigma=sigma, truncate=3)

    for index, row in props.iterrows():
        # Create Window
        x = int(row['centroid-0'])
        y = int(row['centroid-1'])
        size = int(row['major_axis_length'])
        w_corner = [x - size, y - size]
        w_corner[0]= np.maximum(w_corner[0], 0) + 224
        w_corner[1]= np.maximum(w_corner[1], 0) + 224
        w_size = size*2
        w_corner[0] = np.minimum(w_corner[0], c1.shape[0]-w_size)
        w_corner[1] = np.minimum(w_corner[1], c1.shape[1]-w_size)

        label_index = label[int(row['centroid-0'])+224, int(row['centroid-1'])+224]

        # Crop Images
        label_crop = label[w_corner[0]:w_corner[0]+w_size,
                           w_corner[1]:w_corner[1]+w_size]
        mask_crop = np.where(label_crop>0, 1, 0)
        label_crop = np.where(label_crop==label_index, 1, 0)

        nuc_crop = c1[w_corner[0]:w_corner[0]+w_size,
                      w_corner[1]:w_corner[1]+w_size]
        nuc_d_crop = d_c1[w_corner[0]:w_corner[0]+w_size,
                          w_corner[1]:w_corner[1]+w_size]
        cyto_crop = c2[w_corner[0]:w_corner[0]+w_size,
                       w_corner[1]:w_corner[1]+w_size]
        cyto_d_crop = d_c2[w_corner[0]:w_corner[0]+w_size,
                           w_corner[1]:w_corner[1]+w_size]

        # DEBUG - VISUALS
        # fig, ax = plt.subplots(2,2)
        # ax[0][0].imshow(label_crop, cmap='gray')
        # ax[0][1].imshow(mask_crop, cmap='gray')
        # ax[1][0].imshow(nuc_crop, cmap='gray')
        # ax[1][1].imshow(cyto_crop, cmap='gray')
        # plt.show()

        try:
            nuc_d_masked = np.ma.masked_where(label_crop==0, nuc_d_crop)
            nuc_masked = np.ma.masked_where(label_crop==0, nuc_crop)
            cyto_d_masked = np.ma.masked_where(mask_crop==1, cyto_d_crop)
            cyto_masked = np.ma.masked_where(mask_crop==1, cyto_crop)
        except:
            print("Error in Color Statistics")
            print(x,y)
            print(w_size)
            print(nuc_d_masked.shape)
            print(label_crop.shape)
            exit()

        props.loc[index, 'n_avg'] = nuc_masked.mean()
        props.loc[index, 'n_grad'] = np.ma.median(nuc_d_masked)
        props.loc[index, 'c_avg'] = cyto_masked.mean()
        props.loc[index, 'c_grad'] = np.ma.median(cyto_d_masked)

    return props.loc[:, ['n_avg', 'n_grad', 'c_avg', 'c_grad']]

def compute_intensities(df, wsi):
    '''For each nuceli, make measurements on decomposed images.'''
    # Variables
    decomp_dir = DATA_DIR / wsi / '2021/decompose-navab'
    logging.debug(f"Decomposition Directory is set to: {decomp_dir}")
    N_group = len(df.groupby('patch'))

    # For each patch in the dataframe
    count = 0
    tmp_df = pd.DataFrame([])
    for patch, group in df.groupby('patch'):
        t0 = time.time()
        # Read Images
        filename = DATA_DIR / wsi / '2021/20220110-162436' / f"{patch}.png"
        logging.debug(f"compute_intensities - filename: {filename}")
        matname = DATA_DIR / wsi / '2021/decompose-navab' / f"{patch}.mat"
        logging.debug(f"compute_intensities - matname: {matname}")
        N, C = get_surrounding_mat(matname, df)
        L = get_mask(filename, df)

        # Get color statistics
        props = color_statistics(L, N, C, group, patch)

        # Build out tmp dataframe
        tmp_df = tmp_df.append(props)

        t1 = time.time()
        print(f"{count+1}/{N_group}: {t1 - t0:0.3f}")
        count += 1 

    # Combine with original DF
    df = df.join(tmp_df, how='outer')

    # Return appended dataframe
    return df

def compute_additional_intensities(df, wsi):
    '''For each nuceli, make measurements on decomposed images.'''
    # Variables
    decomp_dir = DATA_DIR / wsi / '2021/decompose-ruifrok'
    logging.debug(f"Decomposition Directory is set to: {decomp_dir}")
    N_group = len(df.groupby('patch'))

    # For each patch in the dataframe
    count = 0
    tmp_df = pd.DataFrame([])
    for patch, group in df.groupby('patch'):
        t0 = time.time()
        # Read Images
        filename = DATA_DIR / wsi / '2021/20220110-162436' / f"{patch}.png"
        logging.debug(f"compute_intensities - filename: {filename}")
        matname = DATA_DIR / wsi / '2021/decompose-ruifrok' / f"{patch}.mat"
        logging.debug(f"compute_intensities - matname: {matname}")
        N, C = get_surrounding_mat(matname, df)
        L = get_mask(filename, df)

        # Get color statistics
        props = color_statistics(L, N, C, group, patch)

        # Build out tmp dataframe
        tmp_df = tmp_df.append(props)

        t1 = time.time()
        print(f"{count+1}/{N_group}: {t1 - t0:0.3f}")
        count += 1 

    # Combine with original DF
    df = df.join(tmp_df, how='outer', lsuffix='_proposed', rsuffix='_ruifrok')

    # Return appended dataframe
    return df

# MAIN
def main():
    '''Main Entrypoint'''
    # Get Input Arguments
    parser = argparse.ArgumentParser(prog='cellularity', description='Compute a cellularity score for each nuclei')
    parser.add_argument('WSI', help='Name of the WSI to process.')
    parser.add_argument('-t', '--THRESHOLD', type=int, default=1000, help='Distance to threshold graph connections.')
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

    # Make sure that nuclear_feattures.json exists
    assert (DATA_DIR/flags['WSI']/'2021/nuclear_features.json').exists(), f"File '{DATA_DIR/flags['WSI']/'2021/nuclear_features.json'} not found!"

    # Read Nuclear information
    df = pd.read_json(DATA_DIR / flags['WSI'] / '2021/nuclear_features.json')
    assert len(df) != 0, f"File '{DATA_DIR/flags['WSI']/'2021/nuclear_features.json'} contains not data!"

    # Print Log message
    logging.info(f"Processing WSI: '{flags['WSI']}'")

    # Remove cells with areas that are too small (50 pixels)
    df = df[df.area > 30]

    # Read table of tile names with pen logits
    tile_df = pd.read_csv(DATA_DIR / flags['WSI'] / '2021/pen-marks-svm.csv')
    logging.debug(f"tile_df.columns = {tile_df.columns}")
    tile_df = tile_df[tile_df['SVM_logit'] == 0]  # Remove predicted pen marks

    # Combine dataframes and compute global x, y coords
    df = create_global_reference(df, tile_df)

    # Compute metrics of cellularity
    df = compute_cellularity(df)

    # Compute metrics of Decomposition
    df = compute_intensities(df, flags['WSI'])

    # Compute metrics of Decomposition
   # df = compute_additional_intensities(df, flags['WSI'])

    # Save
    df.to_json(DATA_DIR/flags['WSI']/'2021/all_features.json')

if __name__ == "__main__":
    main()

