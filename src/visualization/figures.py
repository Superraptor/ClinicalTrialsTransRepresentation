"""
Publication-Quality Figure Generation Module
Generates high-resolution static plots (matplotlib/seaborn) and interactive web plots (plotly)
for the manuscript.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

logger = logging.getLogger(__name__)

# Aesthetic styling
plt.rcParams.update({
    "font.sans-serif": "DejaVu Sans",
    "font.family": "sans-serif",
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.titlesize": 14,
})
sns.set_theme(style="whitegrid", palette="deep")


def plot_historical_emergence(
    traj_df: pd.DataFrame,
    output_dir: str = "reports/figures",
) -> str:
    """
    Figure 1: Dual-axis plot of raw annual TGD trial volume and normalized rate per 1,000 trials.
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    file_png = out_path / "figure1_historical_emergence.png"

    fig, ax1 = plt.subplots(figsize=(10, 5.5))

    color_raw = "#1f77b4"
    color_rate = "#d62728"

    bars = ax1.bar(
        traj_df["year"],
        traj_df["tgd_trial_count"],
        color=color_raw,
        alpha=0.65,
        width=0.7,
        label="Raw TGD Trials (Annual)",
    )
    ax1.set_xlabel("Registration Year", fontweight="bold")
    ax1.set_ylabel("Annual TGD Trial Count", color=color_raw, fontweight="bold")
    ax1.tick_params(axis="y", labelcolor=color_raw)
    ax1.set_xticks(traj_df["year"])
    ax1.set_xticklabels(traj_df["year"], rotation=45, ha="right")

    ax2 = ax1.twinx()
    line = ax2.plot(
        traj_df["year"],
        traj_df["tgd_rate_per_1000_trials"],
        color=color_rate,
        marker="o",
        linewidth=2.5,
        markersize=5,
        label="Rate per 1,000 ClinicalTrials.gov Studies",
    )
    ax2.set_ylabel("TGD Trials per 1,000 Registered Studies", color=color_rate, fontweight="bold")
    ax2.tick_params(axis="y", labelcolor=color_rate)
    ax2.grid(False)

    if 2016 in traj_df["year"].values:
        ax1.axvline(x=2016, color="#2ca02c", linestyle="--", linewidth=1.5, alpha=0.8)
        ax1.text(2016.1, ax1.get_ylim()[1] * 0.85, "2016: 'Transgender Persons'\nMeSH Heading Introduced", color="#2ca02c", fontsize=8, fontweight="bold")

    plt.title("Historical Emergence and Relative Prevalence of TGD Research in ClinicalTrials.gov", fontweight="bold", pad=15)
    fig.tight_layout()
    plt.savefig(file_png, dpi=300, bbox_inches="tight")
    plt.close()
    
    logger.info(f"Saved Figure 1 to {file_png}")
    return str(file_png)


def plot_formal_vs_informal_capture(
    traj_df: pd.DataFrame,
    output_dir: str = "reports/figures",
) -> str:
    """
    Figure 2: Stacked proportional area / bar chart of formal metadata vs informal-only capture.
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    file_png = out_path / "figure2_formal_vs_informal_capture.png"

    fig, ax = plt.subplots(figsize=(10, 5.5))

    p1 = ax.bar(
        traj_df["year"],
        traj_df["formal_capture_count"],
        label="Formal Capture (MeSH / genderBased / Keywords)",
        color="#2b5c8f",
        alpha=0.85,
    )
    p2 = ax.bar(
        traj_df["year"],
        traj_df["informal_only_count"],
        bottom=traj_df["formal_capture_count"],
        label="Informal-Only (Hidden in Free-Text)",
        color="#e28743",
        alpha=0.85,
    )

    ax.set_xlabel("Registration Year", fontweight="bold")
    ax.set_ylabel("Number of Studies", fontweight="bold")
    ax.set_xticks(traj_df["year"])
    ax.set_xticklabels(traj_df["year"], rotation=45, ha="right")
    ax.legend(loc="upper left", frameon=True)
    plt.title("Formal vs. Informal Capture of Transgender & Gender-Diverse Studies Over Time", fontweight="bold", pad=15)
    
    fig.tight_layout()
    plt.savefig(file_png, dpi=300, bbox_inches="tight")
    plt.close()
    
    logger.info(f"Saved Figure 2 to {file_png}")
    return str(file_png)


def plot_terminology_trajectories(
    term_traj_df: pd.DataFrame,
    output_dir: str = "reports/figures",
) -> str:
    """
    Figure 3: Longitudinal trajectory of terminology categories over time.
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    file_png = out_path / "figure3_terminology_evolution.png"

    fig, ax = plt.subplots(figsize=(11, 6))

    cols = [c for c in term_traj_df.columns if c != "year"]
    palette = sns.color_palette("tab10", len(cols))

    for idx, col in enumerate(cols):
        ax.plot(
            term_traj_df["year"],
            term_traj_df[col],
            marker="o",
            linewidth=2.2,
            label=col.replace("_", " ").title(),
            color=palette[idx],
        )

    ax.set_xlabel("Year", fontweight="bold")
    ax.set_ylabel("Unique Studies Mentioning Term Category", fontweight="bold")
    ax.set_xticks(term_traj_df["year"])
    ax.set_xticklabels(term_traj_df["year"], rotation=45, ha="right")
    ax.legend(loc="upper left", frameon=True)
    plt.title("Epistemic Shift in TGD Terminology in ClinicalTrials.gov (2000–Present)", fontweight="bold", pad=15)

    fig.tight_layout()
    plt.savefig(file_png, dpi=300, bbox_inches="tight")
    plt.close()

    logger.info(f"Saved Figure 3 to {file_png}")
    return str(file_png)


def plot_unsupervised_clusters_umap(
    df: pd.DataFrame,
    cluster_summaries: Dict[int, Dict[str, Any]],
    output_dir: str = "reports/figures",
) -> str:
    """
    Figure 4: 2D UMAP scatter projection colored by unsupervised cluster label.
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    file_png = out_path / "figure4_unsupervised_clusters_umap.png"

    if "umap_x" not in df.columns or "umap_y" not in df.columns:
        logger.warning("UMAP coordinates not in dataframe. Skipping Figure 4.")
        return ""

    fig, ax = plt.subplots(figsize=(11, 8))

    unique_clusters = sorted(df["cluster_id"].unique())
    palette = sns.color_palette("husl", len(unique_clusters))

    for i, c_id in enumerate(unique_clusters):
        sub = df[df["cluster_id"] == c_id]
        c_label = cluster_summaries.get(c_id, {}).get("cluster_label", f"Cluster {c_id}")
        ax.scatter(
            sub["umap_x"],
            sub["umap_y"],
            label=f"{c_label} (n={len(sub)})",
            color=palette[i],
            alpha=0.75,
            s=40,
            edgecolors="none",
        )

        cx = sub["umap_x"].mean()
        cy = sub["umap_y"].mean()
        ax.annotate(
            f"C{c_id}",
            (cx, cy),
            fontsize=11,
            fontweight="bold",
            color="black",
            ha="center",
            va="center",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.85, edgecolor=palette[i]),
        )

    ax.set_title("Unsupervised Thematic Landscape of Transgender-Related Clinical Trials (UMAP)", fontweight="bold", pad=15)
    ax.set_xlabel("UMAP Dimension 1", fontweight="bold")
    ax.set_ylabel("UMAP Dimension 2", fontweight="bold")
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", frameon=True)

    fig.tight_layout()
    plt.savefig(file_png, dpi=300, bbox_inches="tight")
    plt.close()

    logger.info(f"Saved Figure 4 to {file_png}")
    return str(file_png)


def plot_cluster_trajectories(
    cluster_traj_df: pd.DataFrame,
    output_dir: str = "reports/figures",
) -> str:
    """
    Figure 5: Area / line chart showing relative evolution of each unsupervised cluster over time.
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    file_png = out_path / "figure5_cluster_trajectories.png"

    fig, ax = plt.subplots(figsize=(12, 6.5))

    cols = [c for c in cluster_traj_df.columns if c != "year"]
    palette = sns.color_palette("tab10", len(cols))

    for idx, col in enumerate(cols):
        ax.plot(
            cluster_traj_df["year"],
            cluster_traj_df[col],
            marker="s",
            linewidth=2.0,
            markersize=4,
            label=str(col),
            color=palette[idx % len(palette)],
        )

    ax.set_xlabel("Registration Year", fontweight="bold")
    ax.set_ylabel("Annual Registered Studies", fontweight="bold")
    ax.set_xticks(cluster_traj_df["year"])
    ax.set_xticklabels(cluster_traj_df["year"], rotation=45, ha="right")
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", frameon=True)
    plt.title("Longitudinal Emergence and Shift of Unsupervised Thematic Clusters (2000–Present)", fontweight="bold", pad=15)

    fig.tight_layout()
    plt.savefig(file_png, dpi=300, bbox_inches="tight")
    plt.close()

    logger.info(f"Saved Figure 5 to {file_png}")
    return str(file_png)


NIH_IC_ACRONYMS = {
    "Eunice Kennedy Shriver National Institute of Child Health and Human Development": "NICHD (Child Health & Human Dev)",
    "National Institute of Mental Health": "NIMH (Mental Health)",
    "National Institute on Drug Abuse": "NIDA (Drug Abuse)",
    "National Institute on Minority Health and Health Disparities": "NIMHD (Minority Health)",
    "National Center for Chronic Disease Prev and Health Promo": "CDC / NCCDPHP (Chronic Disease)",
    "National Institute of Nursing Research": "NINR (Nursing Research)",
    "National Institute of Allergy and Infectious Diseases": "NIAID (Infectious Diseases)",
    "National Institute on Alcohol Abuse and Alcoholism": "NIAAA (Alcohol Abuse)",
    "National Cancer Institute": "NCI (Cancer)",
    "National Heart, Lung, and Blood Institute": "NHLBI (Heart, Lung, Blood)",
    "National Institute of Diabetes and Digestive and Kidney Diseases": "NIDDK (Diabetes & Kidney)",
}


def plot_funding_dollars_and_sponsors(
    funding_res: Dict[str, Any],
    top_sponsors_df: pd.DataFrame,
    output_dir: str = "reports/figures",
) -> str:
    """
    Figure 6: Funding dollars ($M) allocation by NIH IC and top individual lead sponsors.
    Uses vertical stacked subplots with horizontal bars and abbreviated labels for maximum readability.
    Also outputs standalone Figure 6A and Figure 6B.
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    file_png = out_path / "figure6_funding_dollars_and_sponsors.png"
    file_6a_png = out_path / "figure6a_top_lead_sponsors.png"
    file_6b_png = out_path / "figure6b_nih_funding_by_institute.png"

    # Clean Top Sponsors Data
    top12_sponsors = top_sponsors_df.head(12).copy()
    sponsor_names = [n[:38] + ("..." if len(n) > 38 else "") for n in top12_sponsors["lead_sponsor_name"]]
    sponsor_counts = top12_sponsors["total_studies"].tolist()

    # Clean NIH IC Data
    ic_df = funding_res.get("ic_funding_df", pd.DataFrame())
    if len(ic_df) > 0:
        top_ics = ic_df.head(8).copy()
        ic_names = [NIH_IC_ACRONYMS.get(name, name[:35] + ("..." if len(name) > 35 else "")) for name in top_ics["administering_ic"]]
        ic_dollars = top_ics["total_funding_millions"].tolist()
    else:
        ic_names = ["No Documented Grants"]
        ic_dollars = [0.0]

    # --- 1. Combined 2-Row Figure (Figure 6) ---
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 10))

    # Panel A: Top Lead Sponsors
    y1 = np.arange(len(sponsor_names))
    bars1 = ax1.barh(y1, sponsor_counts, color="#1f77b4", alpha=0.85, height=0.65)
    ax1.set_yticks(y1)
    ax1.set_yticklabels(sponsor_names, fontsize=9.5)
    ax1.invert_yaxis()
    ax1.set_xlabel("Number of Registered Studies (N)", fontweight="bold", fontsize=10.5)
    ax1.set_title("A. Top 12 Individual Lead Sponsors / Institutions", fontweight="bold", fontsize=12, loc="left", pad=8)
    
    # Add value labels
    for bar in bars1:
        w = bar.get_width()
        ax1.text(w + 0.3, bar.get_y() + bar.get_height() / 2, f"{int(w)}", va="center", ha="left", fontsize=9, fontweight="bold", color="#1f77b4")
    ax1.set_xlim(0, max(sponsor_counts) + 2.5)

    # Panel B: NIH Funding by IC
    y2 = np.arange(len(ic_names))
    bars2 = ax2.barh(y2, ic_dollars, color="#2ca02c", alpha=0.85, height=0.65)
    ax2.set_yticks(y2)
    ax2.set_yticklabels(ic_names, fontsize=9.5)
    ax2.invert_yaxis()
    ax2.set_xlabel("Documented NIH Grant Funding ($ Millions USD)", fontweight="bold", fontsize=10.5)
    ax2.set_title("B. NIH Grant Allocations by Administering Institute / Center ($ Millions)", fontweight="bold", fontsize=12, loc="left", pad=8)

    # Add dollar labels
    for bar in bars2:
        w = bar.get_width()
        ax2.text(w + 0.8, bar.get_y() + bar.get_height() / 2, f"${w:.2f}M", va="center", ha="left", fontsize=9, fontweight="bold", color="#2ca02c")
    ax2.set_xlim(0, max(ic_dollars) + 6.0)

    fig.suptitle("Institutional Sponsorship and Documented NIH Grant Capital ($ USD)", fontweight="bold", fontsize=13.5, y=0.99)
    fig.tight_layout()
    plt.savefig(file_png, dpi=300, bbox_inches="tight")
    plt.close()

    # --- 2. Standalone Figure 6A ---
    fig_a, ax_a = plt.subplots(figsize=(9.5, 6))
    bars_a = ax_a.barh(y1, sponsor_counts, color="#1f77b4", alpha=0.85, height=0.65)
    ax_a.set_yticks(y1)
    ax_a.set_yticklabels(sponsor_names, fontsize=10)
    ax_a.invert_yaxis()
    ax_a.set_xlabel("Number of Registered Studies (N)", fontweight="bold", fontsize=11)
    ax_a.set_title("Top 12 Individual Lead Sponsors & Institutions in TGD Research", fontweight="bold", fontsize=13, pad=12)
    for bar in bars_a:
        w = bar.get_width()
        ax_a.text(w + 0.3, bar.get_y() + bar.get_height() / 2, f"{int(w)}", va="center", ha="left", fontsize=9.5, fontweight="bold", color="#1f77b4")
    ax_a.set_xlim(0, max(sponsor_counts) + 2.5)
    fig_a.tight_layout()
    plt.savefig(file_6a_png, dpi=300, bbox_inches="tight")
    plt.close()

    # --- 3. Standalone Figure 6B ---
    fig_b, ax_b = plt.subplots(figsize=(9.5, 5))
    bars_b = ax_b.barh(y2, ic_dollars, color="#2ca02c", alpha=0.85, height=0.65)
    ax_b.set_yticks(y2)
    ax_b.set_yticklabels(ic_names, fontsize=10)
    ax_b.invert_yaxis()
    ax_b.set_xlabel("Documented Grant Funding ($ Millions USD)", fontweight="bold", fontsize=11)
    ax_b.set_title("NIH Grant Funding by Administering Institute / Center ($ Millions)", fontweight="bold", fontsize=13, pad=12)
    for bar in bars_b:
        w = bar.get_width()
        ax_b.text(w + 0.8, bar.get_y() + bar.get_height() / 2, f"${w:.2f}M", va="center", ha="left", fontsize=9.5, fontweight="bold", color="#2ca02c")
    ax_b.set_xlim(0, max(ic_dollars) + 6.0)
    fig_b.tight_layout()
    plt.savefig(file_6b_png, dpi=300, bbox_inches="tight")
    plt.close()

    logger.info(f"Saved Figure 6 to {file_png}, Figure 6A to {file_6a_png}, and Figure 6B to {file_6b_png}")
    return str(file_png)


def plot_publication_yield_and_open_access(
    pub_metrics: Dict[str, Any],
    output_dir: str = "reports/figures",
) -> str:
    """
    Figure 7: Top publishing journals and open access coverage.
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    file_png = out_path / "figure7_publication_yield_and_lag.png"

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Left: Top Journals
    top_j = pub_metrics.get("top_journals_df", pd.DataFrame())
    if len(top_j) > 0:
        j_names = [j[:35] + ("..." if len(j) > 35 else "") for j in top_j["resolved_journal"].head(10)]
        ax1.barh(np.arange(len(j_names)), top_j["article_count"].head(10), color="#9467bd", alpha=0.85)
        ax1.set_yticks(np.arange(len(j_names)))
        ax1.set_yticklabels(j_names)
        ax1.invert_yaxis()
        ax1.set_xlabel("Articles Indexed (N)", fontweight="bold")
        ax1.set_title("Top Journals Publishing TGD Clinical Trial Results", fontweight="bold")
    else:
        ax1.text(0.5, 0.5, "No journal citations found", ha="center")

    # Right: Publication Yield Breakdown (Pie Chart)
    studies_with_pub = pub_metrics.get("studies_with_publications", 0)
    studies_without_pub = max(0, 828 - studies_with_pub)  # cohort size
    
    labels = ["With Published Results", "No Linked Publications"]
    sizes = [studies_with_pub, studies_without_pub]
    colors = ["#2ca02c", "#d62728"]
    
    ax2.pie(
        sizes,
        labels=labels,
        autopct="%1.1f%%",
        startangle=140,
        colors=colors,
        textprops={"fontweight": "bold"},
    )
    ax2.set_title(f"Overall Publication Rate\n(Unique PMIDs: {pub_metrics.get('unique_pmids', 0)})", fontweight="bold")

    fig.suptitle("Peer-Reviewed Scientific Publications & Journal Distribution", fontweight="bold", fontsize=15)
    fig.tight_layout()
    plt.savefig(file_png, dpi=300, bbox_inches="tight")
    plt.close()

    logger.info(f"Saved Figure 7 to {file_png}")
    return str(file_png)
