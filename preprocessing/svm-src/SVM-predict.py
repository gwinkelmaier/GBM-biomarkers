"""Use a trained SVM model to make predictions on WSIs."""
import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from skimage import io

# GLOBALS
# DATA_DIR = Path("/home/gwinkelmaier/MILKData/NCI-GDC/GBM/Tissue_slide_image")
DATA_DIR = Path("/home/gwinkelmaier/MILKData/NCI-GDC/LGG/Tissue_slide")

# FUNCTIONS
def _img_to_histogram(image):
    """Return an image histogram as concatenated 1D vector"""
    I = image.reshape([-1, 3])
    R = I[:, 0]
    G = I[:, 1]
    B = I[:, 2]

    bins = np.arange(0, 256.0)
    hist_r, _ = np.histogram(R, bins, density=True)
    hist_g, _ = np.histogram(G, bins, density=True)
    hist_b, _ = np.histogram(B, bins, density=True)

    return np.concatenate([hist_r, hist_g, hist_b], axis=0)


def main():
    """Main Entrypoint"""
    # Input Args
    parser = argparse.ArgumentParser(
        prog="SVM-predict.py", description="SVM predictions on WSI"
    )
    parser.add_argument("--wsi", dest="WSI", type=str, required=True, help="WSI Folder")
    flags = vars(parser.parse_args())

    # List of WSI STBpatches
    image_names = [
        i
        # for i in (DATA_DIR / flags["WSI"] / "2021/STBpatches/").glob("**/*_tiles/*.png")
        for i in (DATA_DIR / flags["WSI"] / "2021/patches/").glob("**/*_tiles/*.png")
    ]

    # Load SVM model
    model = pickle.load(open("SVMmodel.pkl", "rb"))

    # Form histograms
    X = np.array(
        [_img_to_histogram(io.imread(filename)) for filename in image_names]
    ).reshape([len(image_names), -1])

    # Make Predictions
    y = model.predict(X)

    # Attach predictions to names
    logits_df = pd.DataFrame(
        data=y,
        index=[str(i.name).rstrip(".png") for i in image_names],
        columns=["SVM_logit"],
    )

    # Get MetaData File
    meta_name = (DATA_DIR / flags["WSI"] / "2021/STBpatches").glob(
        "**/tile_selection.tsv"
    )
    meta_df = pd.read_csv(next(meta_name), sep="\t")
    meta_df = meta_df[meta_df["Keep"] == 1]

    # Save
    meta_df = meta_df.merge(logits_df, left_on="Tile", right_index=True)
    meta_df.to_csv(str(DATA_DIR / flags["WSI"] / "2021/pen-marks-svm.csv"), index=False)


def debug():
    # Input Args
    parser = argparse.ArgumentParser(
        prog="SVM-predict.py", description="SVM predictions on WSI"
    )
    parser.add_argument("--wsi", dest="WSI", type=str, required=True, help="WSI Folder")
    flags = vars(parser.parse_args())
    meta_name = (DATA_DIR / flags["WSI"] / "2021/STBpatches").glob(
        "**/tile_selection.tsv"
    )
    meta_df = pd.read_csv(next(meta_name), sep="\t")
    meta_df = meta_df[meta_df["Keep"] == 1]
    print(len(meta_df))


# MAIN
if __name__ == "__main__":
    main()
    # debug()
