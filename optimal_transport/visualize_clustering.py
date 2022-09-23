"""Cluster patient based on their pdf values and optimal transport distance"""
import argparse
import itertools
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
# from sklearn.cluster import KMeans
import numpy as np
import ot
import pandas as pd
import seaborn as sns
from fastcluster import linkage
# from lifelines import CoxPHFitter
from lifelines.fitters.kaplan_meier_fitter import KaplanMeierFitter
from lifelines.statistics import multivariate_logrank_test
from scipy.cluster.hierarchy import dendrogram, fcluster
from scipy.spatial.distance import pdist, squareform

plt.rcParams["font.weight"] = "bold"
plt.rcParams["axes.labelweight"] = "bold"

# GLOBALS
DATA_DIR = Path("/home/gwinkelmaier/MILKData/NCI-GDC/GBM/Tissue_slide_image/")

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


def compute_serial_matrix(dist_mat, n_clusters, method="ward"):
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

    return seriated_dist, res_order, res_linkage, clusters


# FUNCTIONS
def create_ot_pdf_dist(bins):
    def ot_pdf_dist(x, y):
        M = ot.dist(bins.reshape(-1, 1), bins.reshape(-1, 1))
        M = np.divide(M, M.max())
        emd = ot.emd(x, y, M)
        return np.ma.masked_where(emd == 0, M, 0).mean()

    return ot_pdf_dist


def plot_distribution(df, savename):
    """Plot combined pdf for each cluster."""

    # Combined PDF for each cluster
    plot_df = pd.DataFrame([])
    for cluster, pdf in df.groupby("label"):
        cluster_pdf = pdf.drop("label", axis=1).sum()
        cluster_pdf = np.divide(cluster_pdf, cluster_pdf.sum())
        cluster_pdf["label"] = cluster
        plot_df = plot_df.append(cluster_pdf, ignore_index=True)

    # Reshape DataFrame
    plot_df = pd.melt(plot_df, id_vars="label")

    # Plot and save
    sns.barplot(x="variable", y="value", hue="label", data=plot_df)
    plt.xticks(rotation=45)
    plt.savefig(savename, dpi=300)
    plt.clf()


# MAIN
def main():
    # Get Input Arguments
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

    # Read Clinical Data
    clinical_df = pd.read_csv("../clinical_data.csv")
    clinical_df = clinical_df.loc[
        :,
        [
            "Case",
            "Survival (months)",
            "Vital status (1=dead)",
            "Age (years at diagnosis)",
            "Gender",
        ],
    ]
    clinical_df.Gender = (clinical_df.Gender == "male").astype(int)

    link_df = pd.read_csv("../wsi-tcga.txt")
    link_df["tcga"] = link_df.tcga.apply(
        lambda x: x[:12]
    )  # Strip patient tag out of tcga label
    pdf_df = pdf_df.merge(link_df, how="left")
    pdf_df = pdf_df.merge(clinical_df, left_on="tcga", right_on="Case", how="left")

    # For each patient create a combined pdf based on given feature
    group_df_pdf = pdf_df.groupby("tcga").sum()
    group_df_vital = pdf_df.groupby("tcga").mean()
    group_df = group_df_pdf.iloc[:, :10].merge(group_df_vital.iloc[:, 10:], on="tcga")
    group_df = group_df.reset_index()

    # Create distance matrix if non exists
    if not (Path.cwd() / f"save_data/{flags['FEATURE']}_manualBounds.npy").exists():
        dist_fcn = create_ot_pdf_dist(bins)
        M = np.zeros([len(group_df), len(group_df)])
        for x, y in itertools.combinations(group_df.index, 2):
            pdf_x = group_df.loc[x, bins].values
            pdf_y = group_df.loc[y, bins].values
            M[x, y] = dist_fcn(pdf_x, pdf_y)

        M += M.T

        # Save
        with open(f"save_data/{flags['FEATURE']}_manualBounds.npy", "wb") as fid:
            np.save(fid, M)
    else:
        with open(f"save_data/{flags['FEATURE']}_manualBounds.npy", "rb") as fid:
            M = np.load(fid)

    print(f"Finished Creating/Loading {flags['FEATURE']}_manualBounds.npy {M.shape}")

    # Clusters
    for cluster in [2, 3, 4]:
        (M_ordered, _, _, flat_clusters) = compute_serial_matrix(M, cluster)

        # Plot cluster pdfs
        group_df["label"] = flat_clusters
        plot_distribution(
            group_df.drop(
                [
                    "tcga",
                    "Vital status (1=dead)",
                    "Survival (months)",
                    "Age (years at diagnosis)",
                    "Gender",
                ],
                axis=1,
            ),
            str(
                Path.cwd()
                / f"cluster_examples/{flags['FEATURE']}_{cluster}_manualBounds.png"
            ),
        )

        # Write cluster labels
        label_output = group_df.rename(columns={"Gender": "Gender (1=male)"})
        label_output.loc[
            :,
            [
                "label",
                "tcga",
                "Vital status (1=dead)",
                "Survival (months)",
                "Age (years at diagnosis)",
                "Gender (1=male)",
            ],
        ].to_csv(f"labels/{flags['FEATURE']}_{cluster}.csv", index=False)


if __name__ == "__main__":
    main()
