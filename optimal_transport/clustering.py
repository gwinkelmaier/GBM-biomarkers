"""Cluster patient based on their pdf values and optimal transport distance"""
import argparse
from pathlib import Path

import matplotlib
# from sklearn.cluster import KMeans
import numpy as np
import pandas as pd
# from lifelines import CoxPHFitter
from lifelines.fitters.kaplan_meier_fitter import KaplanMeierFitter
from lifelines.statistics import multivariate_logrank_test

matplotlib.use("Agg")
import itertools

import matplotlib.pyplot as plt
import ot
import seaborn as sns
from fastcluster import linkage
from scipy.cluster.hierarchy import dendrogram, fcluster
from scipy.spatial.distance import pdist, squareform

plt.rcParams["font.weight"] = "bold"
plt.rcParams["axes.labelweight"] = "bold"

# GLOBALS
DATA_DIR = Path("/home/gwinkelmaier/MILKData/NCI-GDC/LGG/Tissue_slide/")

# OUTSIDE UTILITIES
def seriation(Z, N, cur_index):
    """
    input:
        - Z is a hierarchical tree (dendrogram)
        - N is the number of points given to the clustering process
        - cur_index is the position in the tree for the recursive traversal
    output:
        - order implied by the hierarchical tree Z

    seriation computes the order implied by a hierarchical tree (dendrogram)
    """
    if cur_index < N:
        return [cur_index]
    else:
        left = int(Z[cur_index - N, 0])
        right = int(Z[cur_index - N, 1])
        return seriation(Z, N, left) + seriation(Z, N, right)


def compute_serial_matrix(dist_mat, n_clusters, save_name, method="ward"):
    """
    input:
        - dist_mat is a distance matrix
        - method = ["ward","single","average","complete"]
    output:
        - seriated_dist is the input dist_mat,
          but with re-ordered rows and columns
          according to the seriation, i.e. the
          order implied by the hierarchical tree
        - res_order is the order implied by
          the hierarhical tree
        - res_linkage is the hierarhical tree (dendrogram)

    compute_serial_matrix transforms a distance matrix into
    a sorted distance matrix according to the order implied
    by the hierarchical tree (dendrogram)
    """
    N = len(dist_mat)
    flat_dist_mat = squareform(dist_mat)
    res_linkage = linkage(flat_dist_mat, method=method, preserve_input=True)
    res_order = seriation(res_linkage, N, N + N - 2)
    seriated_dist = np.zeros((N, N))
    a, b = np.triu_indices(N, k=1)
    seriated_dist[a, b] = dist_mat[[res_order[i] for i in a], [res_order[j] for j in b]]
    seriated_dist[b, a] = seriated_dist[a, b]

    clusters = fcluster(res_linkage, n_clusters, criterion="maxclust")

    # Plot
    fig, ax = plt.subplots(1, 2, gridspec_kw={"width_ratios": [1, 3]})
    dendrogram(
        res_linkage,
        show_leaf_counts=False,
        no_labels=True,
        orientation="left",
        ax=ax[0],
    )
    ax[0].invert_yaxis()
    ax[1].imshow(seriated_dist / seriated_dist.max(), cmap="gray")
    ax[1].axis("off")
    fig.tight_layout()
    plt.subplots_adjust(wspace=0.0)
    plt.savefig(save_name)
    print(f"Saving at: {save_name}")

    return seriated_dist, res_order, res_linkage, clusters


# FUNCTIONS
def plot_kaplan_meier(df, save_name):
    """Plot a kaplan meier curve based on patient stratified by 'labels'."""
    N = len(df["label"].unique())
    fig, ax = plt.subplots(1, 1)
    for l in range(N):
        l += 1
        kmf = KaplanMeierFitter()
        kmf.fit(
            df.loc[df["label"] == l, "Survival (months)"],
            df.loc[df["label"] == l, "Vital status (1=dead)"],
            label=l,
        )
        kmf.plot(ax=ax)

    plt.savefig(save_name, dpi=300)


def plot_centroids(centroids, save_name):
    """Plot centoids (PDFs) to visualize the difference in clusters."""
    plt.figure()
    for i, c in enumerate(centroids):
        plt.plot(c, label=f"Centroid {i}")
    plt.legend()

    plt.savefig(save_name)


def create_ot_pdf_dist(bins):
    def ot_pdf_dist(x, y):
        M = ot.dist(bins.reshape(-1, 1), bins.reshape(-1, 1))
        M = np.divide(M, M.max())
        emd = ot.emd(x, y, M)
        return np.ma.masked_where(emd == 0, M, 0).mean()

    return ot_pdf_dist


# MAIN
def main():  # Get Input Arguments
    parser = argparse.ArgumentParser(
        prog="featuer-pdf-analysis.pdf", description="Subtype Analysis"
    )
    parser.add_argument("FEATURE", help="Name of feature to analyze")
    parser.add_argument(
        "-n", "--n_bins", dest="N_BINS", type=int, default=10, help="Number of PDF bins"
    )
    flags = vars(parser.parse_args())

    print("Running Job...")
    print(f"\tFEATURE: {flags['FEATURE']}")
    print(f"\tN_BINS: {flags['N_BINS']}")

    # Read WSI data
    wsi_list = [
        str(i / f"2021/pdf_{flags['FEATURE']}_{flags['N_BINS']}_manualBounds.csv")
        for i in DATA_DIR.iterdir()
        if (
            i / f"2021/pdf_{flags['FEATURE']}_{flags['N_BINS']}_manualBounds.csv"
        ).exists()
    ]
    print(len(wsi_list))
    pdf_df = pd.DataFrame([])
    for i, wsi in enumerate(wsi_list):
        tmp_df = pd.read_csv(wsi)
        if i == 0:
            bins = tmp_df.loc[:, "bins"].values
        tmp_df = tmp_df.T
        tmp_df.columns = tmp_df.loc["bins", :].values
        tmp_df = tmp_df.loc["pdf", :]
        tmp_df["wsi"] = Path(wsi).parent.parent.name
        pdf_df = pdf_df.append(tmp_df, ignore_index=True)
    print(len(pdf_df))

    # Read Clinical Data
    clinical_df = pd.read_csv("clinical_data.csv")
    clinical_df = clinical_df.loc[
        :, ["Case", "Survival (months)", "Vital status (1=dead)"]
    ]
    link_df = pd.read_csv("wsi-tcga.txt")
    link_df["tcga"] = link_df.tcga.apply(
        lambda x: x[:12]
    )  # Strip patient tag out of tcga label
    pdf_df = pdf_df.merge(link_df, how="left")
    pdf_df = pdf_df.merge(clinical_df, left_on="tcga", right_on="Case", how="left")

    # For each patient create a combined pdf based on given feature
    group_df = pdf_df.groupby("tcga").mean()
    cols = group_df.columns
    tmp_df = pdf_df.groupby("tcga").sum()
    tmp_df = tmp_df.drop(["Survival (months)", "Vital status (1=dead)"], axis=1)
    for index, row in tmp_df.iterrows():
        tmp_df.loc[index, :] = (row[:] / np.sum(row[:].values)).values
    combined_df = group_df.merge(
        tmp_df, suffixes=("_mean", "_pdf"), left_index=True, right_index=True
    )

    # Reconstruct group_df
    cols = [c for c in combined_df.columns if c[-4:] != "mean"]
    group_df = combined_df[cols].reset_index()

    # # check summation
    # validation = group_df.filter(regex='pdf$', axis=1)
    # validation = validation.sum(axis=1)
    # print(validation.head())
    # print(validation.describe())
    # exit()

    # Create distance matrix if non exists
    dist_fcn = create_ot_pdf_dist(bins)
    M = np.zeros([len(group_df), len(group_df)])
    for x, y in itertools.combinations(group_df.index, 2):
        pdf_x = group_df.filter(regex="pdf$", axis=1).iloc[x, :].values
        pdf_x = np.ascontiguousarray(pdf_x)
        pdf_y = group_df.filter(regex="pdf$", axis=1).iloc[y, :].values
        pdf_y = np.ascontiguousarray(pdf_y)
        M[x, y] = dist_fcn(pdf_x, pdf_y)

    M += M.T

    # Save
    with open(f"save_data/{flags['FEATURE']}_manualBounds.npy", "wb") as fid:
        np.save(fid, M)

    print(f"Finished Creating {flags['FEATURE']}_manualBounds.npy {M.shape}")

    # Clusters
    for cluster in [2, 3, 4]:
        savename = f"save_data/{flags['FEATURE']}_manualBounds.png"
        (M_ordered, _, _, flat_clusters) = compute_serial_matrix(M, cluster, savename)

        # Kaplan-Meir
        group_df["label"] = flat_clusters
        plot_kaplan_meier(
            group_df.dropna(),
            str(
                Path.cwd()
                / f"save_data/{flags['FEATURE']}_{cluster}_km_manualBounds.png"
            ),
        )

        # TODO: Outdated survival analysis
        # # Log-Rank
        # results = multivariate_logrank_test(
        #     group_df["Survival (months)"],
        #     group_df[f"label"],
        #     group_df["Vital status (1=dead)"],
        # )
        # results.print_summary()

        # print()
        # print(40 * "=")
        # print()


if __name__ == "__main__":
    main()
