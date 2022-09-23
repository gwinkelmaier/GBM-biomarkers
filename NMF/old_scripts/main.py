"""Compute NMF pilot study.

Seed are taken from Nuclear Segmentation Masks.
"""
from skimage import io
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from sklearn import decomposition

def image_generator(data_path):
    image_names = [str(i) for i in (data_path / 'Images').glob('*.bmp')]
    mask_names = [i.replace('Images','Masks') for i in image_names]
    mask_names = [i.replace('image','mask') for i in mask_names]
    return (image_names, mask_names)

def background_mask(I):
    mask = np.where(I > 235, 1, 0)
    mask = np.sum(mask, axis=2)
    mask = np.where(mask != 3, 1, 0)
    return mask

def get_seeds(I, M):
    I2 = I[M == 255]
    I3 = I[M != 255]
    fg, bg = np.mean(I2, axis=0), np.mean(I3, axis=0)
    fg = np.reshape(fg, [1,-1])
    bg = np.reshape(bg, [1,-1])
    return fg, bg

def decompose(I, H):
    (x, y) = I.shape[:2]
    I = np.reshape(I, [-1,3])
    W = np.random.randn(I.shape[0], I.shape[1])
    model = decomposition.NMF(n_components=3, init='custom')
    D = model.fit_transform(I, W=W, H=H)
    return np.reshape(D, [x, y, 2])

def show_pair(I, M):
    # I = io.imread(image_name)
    # M = io.imread(mask_name)

    fig, ax = plt.subplots(1, 2)
    ax[0].imshow(I, cmap='gray')
    ax[1].imshow(M, cmap='gray')
    ax[0].axis('off')
    ax[1].axis('off')
    plt.show()

def show_triplet(I, D):
    # I = io.imread(image_name)
    # M = io.imread(mask_name)

    fig, ax = plt.subplots(3,1)
    ax[0].imshow(I, cmap='gray')
    ax[1].imshow(D[:,:,0], cmap='gray')
    ax[2].imshow(D[:,:,1], cmap='gray')


    ax[0].axis('off')
    ax[1].axis('off')
    ax[2].axis('off')
    plt.show()


if __name__ == "__main__":
    data_path = Path.cwd() / 'data'
    (i_list, m_list) = image_generator(data_path)
    images = zip(i_list,m_list)

    for (i,m) in images:
        I, M = io.imread(i), io.imread(m)

        mask = background_mask(I)

        (fg, bg) = get_seeds(I, M)

        W = np.concatenate([fg, bg], axis=0)

        D = decompose(I, W)

        show_triplet(I,D)
        # show_pair(D[:,:,2], D[:,:,1])
        # show_pair(I, mask)
        break
