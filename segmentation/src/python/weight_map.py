from scipy.ndimage.morphology import generate_binary_structure
from scipy.ndimage import gaussian_laplace, binary_erosion
import scipy.ndimage.filters as filter
from pathlib import Path
from scipy.ndimage.morphology import distance_transform_edt as dt
import numpy as np
from skimage.io import imread, imsave
from skimage import measure
import matplotlib.pyplot as plt

def _compute_potential(mask):
    M = mask==1.0

    BW = dt(1-M)

    L1 = gaussian_laplace(BW, sigma=1.0, mode='nearest')
    L2 = gaussian_laplace(BW, sigma=1.5, mode='nearest')
    L3 = gaussian_laplace(BW, sigma=2.0, mode='nearest')

    L = np.stack([L1, L2, L3], axis=2)
    P = L.min(axis=2)

    P = np.where(P>-0.1, 0, -40*P)
    P = np.where(M>0, 0, P)

    M = np.zeros(P.shape)
    b = 5
    M[b:-b, b:-b] = P[b:-b, b:-b]

    M /= np.max(M)
    return M

def _compute_unet_weights(M):
    L = measure.label(M==255)

    # function parameters
    sigma=5
    omega_n = 1

    if np.max(L)==0:
        D = np.zeros([M.shape[0], M.shape[1]])
        return (D)

    D1 = dt( L )
    D2 = np.zeros([M.shape[0], M.shape[1]])
    for i in range(np.max(L)):
        L2 = (L==0) | (L==(i+1))
        tmp = dt(L2)
        D2 = np.maximum(D2, tmp)
    # D = np.sort(D, axis=2)
    D = np.stack([D1,D2], axis=-1)

    try:
        O = omega_n * np.exp( -1*((D[:,:,0]+D[:,:,1])**2)/( 2*sigma**2 ) )
    except:
        O = omega_n * np.exp( -1*((D[:,:,0])**2)/( 2*sigma**2 ) )
    O = np.where(M!=255, O, 0)

    return( O )

def _weight_maps(M):
    '''
    Compute the weight maps for pixel-wise loss functions
        P: developed by Mina Khoshdeli (NET FUSION)
        U: develeped by Olaf Ronneberger (UNET)
    '''
    P = _compute_potential(M)
    U = _compute_unet_weights(M)
    return (P,U)

if __name__ == "__main__":
    _data_path = Path.cwd() / 'MinaDataPheno'
    for file in _data_path.glob('**/mask*.bmp'):
        # print(file)
        M = imread( str(file) )
        P,U = _weight_maps(M)
        imsave(str.replace(str(file), 'mask','pot'), P)
        imsave(str.replace(str(file), 'mask','unet'), U)
        print(f"finished with {file.name}")
