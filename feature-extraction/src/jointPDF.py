"""Create Joint Probability Denisty Functions"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

WSI_DIR = Path("/home/gwinkelmaier/MILKData/NCI-GDC/GBM/Tissue_slide_image/")
FEATURES = ["area", "n_grad_proposed", "solidity"]


def _getBins(features):
    """Get manually set uppper/lower bounds for each feature."""
    bounds = {}
    for f in features:
        with open(f"{f}_stats.csv", "r") as fid:
            lineList = [i.rstrip("\n").split(", ") for i in fid.readlines()]
            bounds[f] = [float(lineList[0][0]), float(lineList[0][1])]
    return bounds


def jointPDF(df, features, binsDict):
    group = df.groupby([i + "_bin" for i in features])
    hist = group[features[0]].count().values
    return np.divide(hist, np.sum(hist))


if __name__ == "__main__":
    """Main Entrypoint."""
    parser = argparse.ArgumentParser(
        prog="jointPDF.py",
        description="Create joint pdf of features for classification of idh1",
    )
    parser.add_argument(
        "-n", dest="N", default=10, type=int, help="Number of bins in the PDF"
    )
    FLAGS = vars(parser.parse_args())

    # Get a list of wsi names
    with open("file-list.txt", "r") as fid:
        wsiList = [i.rstrip("\n") for i in fid.readlines()]

    # Get PDF bounds
    boundsDict = _getBins(FEATURES)
    binsDict = {}
    for f in FEATURES:
        step_size = (boundsDict[f][1] - boundsDict[f][0]) / FLAGS["N"]
        binsDict[f] = np.arange(
            boundsDict[f][0], boundsDict[f][1] + step_size, step_size
        )

    # Initialize pdf list
    jointPDFDict = {}

    # For All WSI
    for count, wsi in enumerate(wsiList):
        # Read Dataframe
        try:
            df = pd.read_json(WSI_DIR / wsi / "2021/all_features.json").loc[:, FEATURES]
        except Exception:
            continue

        # Quantize into bins of 10
        for f in FEATURES:
            df[f"{f}_bin"] = pd.cut(df[f].values, binsDict[f])

        # Build Joint PDF
        jPDF_output = jointPDF(df, FEATURES, binsDict)

        # Add to master list
        jointPDFDict[wsi] = list(jPDF_output)

    with open(f"jointPDF/{'_'.join(FEATURES)}.json", "w") as fid:
        json.dump(jointPDFDict, fid)
