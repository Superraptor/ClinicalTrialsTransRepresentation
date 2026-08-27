"""
Comprehensive Investigator, Clinical Trial, and Publication Citation Network Mapping Module
Builds the complete multi-layer citation network across studies and PMIDs in the cohort.
Includes direct paper-to-paper cross-citation linkages (NCBI elink / Crossref).

Visualizations:
1. High-Resolution Publication Figure 8: Pruned to shared publications and cross-citations with
   community modularity partitioning, degree-based sizing, and anti-overlap force layout.
2. High-Performance WebGL Interactive App: Sigma.js with integrated Real-Time Force-Pushing
   Physics Simulation Engine (anti-collision, repulsion, spring tension, and filter toggles).
"""

import html
import json
import logging
import math
import urllib.parse
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Known ROR IDs for prominent research organizations in the cohort
KNOWN_ROR_REGISTRY = {
    "University of California, San Francisco": "https://ror.org/043mz5j54",
    "University of Colorado, Denver": "https://ror.org/03w4st659",
    "University of California, Los Angeles": "https://ror.org/046rm7j60",
    "University of California, San Diego": "https://ror.org/0168r3w48",
    "Yale University": "https://ror.org/03v76x132",
    "University of North Carolina, Chapel Hill": "https://ror.org/0130frc33",
    "University Hospital, Ghent": "https://ror.org/00cv9y106",
    "Columbia University": "https://ror.org/00hj8s172",
    "University of Michigan": "https://ror.org/00e2ke570",
    "University of Washington": "https://ror.org/00cvxb145",
    "Emory University": "https://ror.org/03czfpz43",
    "Duke University": "https://ror.org/00py81415",
    "Medical College of Wisconsin": "https://ror.org/01k5qdt14",
    "University of Wisconsin, Madison": "https://ror.org/01y2jtd41",
    "Cairo University": "https://ror.org/04m1kbh80",
    "Medical University of Vienna": "https://ror.org/034907x22",
    "Assistance Publique - Hôpitaux de Paris": "https://ror.org/00yhsmh38",
    "National Institute of Allergy and Infectious Diseases (NIAID)": "https://ror.org/043z4tv69",
    "National Institute of Mental Health": "https://ror.org/04xeg9z08",
    "Eunice Kennedy Shriver National Institute of Child Health and Human Development": "https://ror.org/04byxpc64",
}


def get_ror_url(institution_name: str) -> str:
    """Returns official ROR link or ROR registry search query URL."""
    clean_name = str(institution_name).strip()
    if clean_name in KNOWN_ROR_REGISTRY:
        return KNOWN_ROR_REGISTRY[clean_name]
    encoded = urllib.parse.quote(clean_name)
    return f"https://ror.org/search?query={encoded}"


def build_investigator_trial_publication_graph(
    studies_df: pd.DataFrame,
    awards_df: pd.DataFrame,
    pub_df: pd.DataFrame,
    crosslinks_df: Optional[pd.DataFrame] = None,
    include_all_pubs: bool = True,
    prune_unshared_pubs: bool = False,
    min_shared_trials: int = 2,
    max_pis: Optional[int] = None,
    max_pubs: Optional[int] = None,
) -> nx.Graph:
    """
    Constructs a multi-layer NetworkX graph connecting PIs, NCT trials, Lead Sponsors,
    PMIDs, and direct PMID-to-PMID citation cross-linkages.
    
    If prune_unshared_pubs=True, only includes publications that are referenced across >= min_shared_trials
    or are part of direct cross-citation linkages.
    """
    G = nx.Graph()
    cohort_ncts: Set[str] = set(studies_df["nct_id"].unique())

    # Find cross-cited PMIDs
    cross_cited_pmids: Set[str] = set()
    if crosslinks_df is not None and len(crosslinks_df) > 0:
        if "citing_pmid" in crosslinks_df.columns and "cited_pmid" in crosslinks_df.columns:
            cross_cited_pmids = set(crosslinks_df["citing_pmid"].astype(str)).union(
                set(crosslinks_df["cited_pmid"].astype(str))
            )

    # Calculate publication connection counts per PMID across cohort trials
    valid_pubs = pub_df[(pub_df["pmid"] != "") & (pub_df["nct_id"].isin(cohort_ncts))].copy()
    valid_pubs["pmid_str"] = valid_pubs["pmid"].astype(str)
    pmid_trial_counts: Dict[str, int] = valid_pubs.groupby("pmid_str")["nct_id"].nunique().to_dict()

    # 1. Add all trials in cohort
    for _, row in studies_df.iterrows():
        nct_id = row["nct_id"]
        title = str(row.get("brief_title", nct_id) or nct_id)
        year = str(row.get("analysis_year", "") or "")
        sponsor = str(row.get("lead_sponsor_name", "") or "").strip()
        phase = str(row.get("phase", "") or "Not Specified")
        cluster = str(row.get("cluster_label", "") or "Unclassified")

        G.add_node(
            nct_id,
            node_type="Trial",
            label=nct_id,
            title=title,
            year=year,
            size=14,
            color="#10b981",  # Emerald
            cluster=cluster,
            url=f"https://clinicaltrials.gov/study/{nct_id}",
            details=f"<b>Clinical Trial:</b> {nct_id} ({year})<br><b>Cluster:</b> {cluster}<br><b>Title:</b> {title[:95]}...<br><b>Phase:</b> {phase}",
        )

        # Connect Trial to Sponsor
        if sponsor and len(sponsor) > 2:
            sp_label = sponsor[:28] + ("..." if len(sponsor) > 28 else "")
            ror_link = get_ror_url(sponsor)
            if not G.has_node(sp_label):
                G.add_node(
                    sp_label,
                    node_type="Sponsor",
                    label=sp_label,
                    full_name=sponsor,
                    size=18,
                    color="#f59e0b",  # Amber
                    url=ror_link,
                    details=f"<b>Research Institution:</b> {sponsor}<br><b>Registry:</b> Research Organization Registry (ROR)",
                )
            G.add_edge(sp_label, nct_id, weight=1, edge_type="SPONSORED")

    # 2. PI to Trial edges from awards
    valid_awards = awards_df[(awards_df["contact_pi"] != "") & (awards_df["nct_id"].isin(cohort_ncts))].copy()
    if max_pis is not None:
        top_pis = set(valid_awards["contact_pi"].value_counts().head(max_pis).index)
        valid_awards = valid_awards[valid_awards["contact_pi"].isin(top_pis)]

    for _, row in valid_awards.iterrows():
        pi_name = row["contact_pi"].title().strip()
        nct_id = row["nct_id"]
        funding = float(row.get("award_amount_usd", 0) or 0)
        grant_num = str(row.get("project_num", "") or "")
        ic_name = str(row.get("administering_ic", "") or "")

        pi_pubmed_url = f"https://pubmed.ncbi.nlm.nih.gov/?term={urllib.parse.quote(pi_name)}"
        pi_reporter_url = f"https://reporter.nih.gov/search/projects/results?search={urllib.parse.quote(pi_name)}"

        if not G.has_node(pi_name):
            G.add_node(
                pi_name,
                node_type="PI",
                label=pi_name,
                size=22,
                color="#0ea5e9",  # Sky blue
                url=pi_reporter_url,
                alt_url=pi_pubmed_url,
                details=f"<b>Principal Investigator:</b> {pi_name}<br><b>Administering IC:</b> {ic_name}<br><b>Grant:</b> {grant_num}",
                total_funding=funding,
            )
        else:
            G.nodes[pi_name]["total_funding"] = G.nodes[pi_name].get("total_funding", 0) + funding

        G.add_edge(pi_name, nct_id, weight=1, edge_type="FUNDED_PI")

    # 3. Add Publications (Optionally pruned to shared / cross-cited only)
    valid_pubs = valid_pubs.drop_duplicates(subset=["nct_id", "pmid_str"])
    if not include_all_pubs and max_pubs is not None:
        valid_pubs = valid_pubs.head(max_pubs)

    for row in valid_pubs.itertuples(index=False):
        nct_id = getattr(row, "nct_id", "")
        pmid = str(getattr(row, "pmid_str", ""))
        doi = str(getattr(row, "doi", "") or "")
        journal = str(getattr(row, "resolved_journal", "") or "PubMed")
        pub_title = str(getattr(row, "resolved_title", "") or "")
        pub_year = str(getattr(row, "resolved_pub_year", "") or "")
        pub_label = f"PMID:{pmid}"

        trial_cnt = pmid_trial_counts.get(pmid, 1)
        is_shared = trial_cnt >= min_shared_trials
        is_cross = pmid in cross_cited_pmids

        if prune_unshared_pubs and not (is_shared or is_cross):
            continue

        pubmed_url = f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"
        doi_url = f"https://doi.org/{doi}" if doi else pubmed_url

        if not G.has_node(pub_label):
            node_color = "#ec4899" if is_cross else ("#8b5cf6" if is_shared else "#a855f7")
            G.add_node(
                pub_label,
                node_type="Publication",
                label=pub_label,
                pmid=pmid,
                size=14 if is_shared or is_cross else 9,
                color=node_color,
                is_shared=is_shared,
                is_cross_cited=is_cross,
                shared_trial_count=trial_cnt,
                url=pubmed_url,
                doi_url=doi_url,
                details=f"<b>PMID {pmid}</b> ({pub_year})<br><b>Journal:</b> {journal}<br><b>Shared across:</b> {trial_cnt} trials<br><b>Title:</b> {pub_title[:95]}...",
            )

        if G.has_node(nct_id):
            G.add_edge(nct_id, pub_label, weight=1, edge_type="PUBLISHED_RESULT")

    # 4. Direct Citation Cross-Linkages between PMIDs (Internal to cohort)
    if crosslinks_df is not None and len(crosslinks_df) > 0:
        for row in crosslinks_df.itertuples(index=False):
            citing_pmid = str(getattr(row, "citing_pmid", ""))
            cited_pmid = str(getattr(row, "cited_pmid", ""))
            citing = f"PMID:{citing_pmid}"
            cited = f"PMID:{cited_pmid}"
            if G.has_node(citing) and G.has_node(cited) and citing != cited:
                G.add_edge(citing, cited, weight=2, edge_type="DIRECT_CITATION", color="#f43f5e")

    logger.info(
        f"Constructed network graph (prune_unshared={prune_unshared_pubs}): "
        f"{G.number_of_nodes()} nodes and {G.number_of_edges()} edges."
    )
    return G


def compute_balanced_force_layout(G: nx.Graph, scale: float = 1200.0) -> Dict[str, Tuple[float, float]]:
    """
    Computes a balanced force-directed layout with generous repulsive spacing.
    Optimized for multi-thousand node graphs by layouting core backbone and radiating leaves.
    """
    if len(G.nodes) == 0:
        return {}

    # If graph is large, layout the non-leaf backbone first for extreme speed & quality
    if G.number_of_nodes() > 400:
        core_nodes = [
            n for n, d in G.nodes(data=True)
            if d.get("node_type") != "Publication" or d.get("is_shared") or d.get("is_cross_cited") or G.degree(n) > 1
        ]
        if len(core_nodes) > 10:
            core_subG = G.subgraph(core_nodes).copy()
            k_val = max(0.3, 3.0 / math.sqrt(max(1, len(core_nodes))))
            pos = nx.spring_layout(core_subG, k=k_val, iterations=40, scale=scale, seed=42)

            # Radiate leaf publications around parent trials
            leaf_nodes = [n for n in G.nodes if n not in pos]
            for leaf in leaf_nodes:
                neighbors = list(G.neighbors(leaf))
                if neighbors and neighbors[0] in pos:
                    px, py = pos[neighbors[0]]
                    angle = (hash(leaf) % 360) * (math.pi / 180.0)
                    r = 60.0 + (hash(leaf) % 40)
                    pos[leaf] = (px + r * math.cos(angle), py + r * math.sin(angle))
                else:
                    pos[leaf] = (0.0, 0.0)
            return pos

    k_val = max(0.25, 2.5 / math.sqrt(max(1, G.number_of_nodes())))
    return nx.spring_layout(G, k=k_val, iterations=35, scale=scale, seed=42)


def plot_static_citation_network(
    G: nx.Graph,
    output_dir: str = "reports/figures",
    studies_df: Optional[pd.DataFrame] = None,
    awards_df: Optional[pd.DataFrame] = None,
    pub_df: Optional[pd.DataFrame] = None,
    crosslinks_df: Optional[pd.DataFrame] = None,
) -> str:
    """
    Renders high-resolution publication static network graph (Figure 8).
    Prunes unshared single-degree leaf publications to highlight cross-trial collaborations,
    shared literature hubs, and direct cross-citation bridges.
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    file_png = out_path / "figure8_investigator_citation_network.png"

    # If full dataframes provided or G contains unshared leaf pubs, build pruned high-signal graph
    if studies_df is not None and awards_df is not None and pub_df is not None:
        plot_G = build_investigator_trial_publication_graph(
            studies_df=studies_df,
            awards_df=awards_df,
            pub_df=pub_df,
            crosslinks_df=crosslinks_df,
            prune_unshared_pubs=True,
            min_shared_trials=2,
        )
    else:
        # Prune G directly: remove publications with degree <= 1 and no direct citation
        plot_G = G.copy()
        nodes_to_remove = []
        for n, d in plot_G.nodes(data=True):
            if d.get("node_type") == "Publication":
                deg = plot_G.degree(n)
                is_cross = d.get("is_cross_cited", False)
                if deg <= 1 and not is_cross:
                    nodes_to_remove.append(n)
        plot_G.remove_nodes_from(nodes_to_remove)

    if len(plot_G.nodes) == 0:
        logger.warning("Pruned graph is empty. Using original graph.")
        plot_G = G

    fig, ax = plt.subplots(figsize=(18, 13), facecolor="#090d16")
    ax.set_facecolor("#090d16")

    # Balanced force layout with high repulsion
    k_spacing = max(0.35, 3.2 / math.sqrt(max(1, plot_G.number_of_nodes())))
    pos = nx.spring_layout(plot_G, k=k_spacing, iterations=40, seed=42)

    # Edge drawing
    std_edges = [(u, v) for u, v, d in plot_G.edges(data=True) if d.get("edge_type") != "DIRECT_CITATION"]
    cite_edges = [(u, v) for u, v, d in plot_G.edges(data=True) if d.get("edge_type") == "DIRECT_CITATION"]

    nx.draw_networkx_edges(
        plot_G,
        pos,
        edgelist=std_edges,
        alpha=0.28,
        edge_color="#475569",
        width=1.1,
        ax=ax,
    )

    if cite_edges:
        nx.draw_networkx_edges(
            plot_G,
            pos,
            edgelist=cite_edges,
            alpha=0.95,
            edge_color="#f43f5e",  # Vibrant Rose
            width=2.6,
            style="dashed",
            ax=ax,
        )

    # Node styles by category
    node_categories = {
        "PI": {"color": "#38bdf8", "shape": "o", "label": "Principal Investigator (NIH RePORTER)", "base_size": 220},
        "Trial": {"color": "#34d399", "shape": "s", "label": "Clinical Trial (ClinicalTrials.gov)", "base_size": 160},
        "Publication": {"color": "#a855f7", "shape": "^", "label": "Shared Cohort Publication (PubMed)", "base_size": 190},
        "Sponsor": {"color": "#fbbf24", "shape": "h", "label": "Research Institution (ROR Registry)", "base_size": 260},
    }

    for ntype, style in node_categories.items():
        nlist = [n for n, d in plot_G.nodes(data=True) if d.get("node_type") == ntype]
        if nlist:
            sizes = []
            for n in nlist:
                deg = plot_G.degree(n)
                d = plot_G.nodes[n]
                if ntype == "PI":
                    sizes.append(max(150, min(800, deg * 60 + 100)))
                elif ntype == "Publication":
                    is_cross = d.get("is_cross_cited", False)
                    sizes.append(380 if is_cross else max(120, min(600, deg * 70)))
                elif ntype == "Sponsor":
                    sizes.append(max(200, min(950, deg * 50 + 150)))
                else:
                    sizes.append(max(100, min(500, deg * 40 + 80)))

            nx.draw_networkx_nodes(
                plot_G,
                pos,
                nodelist=nlist,
                node_color=style["color"],
                node_shape=style["shape"],
                node_size=sizes,
                alpha=0.92,
                edgecolors="#ffffff",
                linewidths=1.0,
                label=style["label"],
                ax=ax,
            )

    # Node Labeling: Label top hubs, prominent institutions, and cross-cited publications
    degrees = dict(plot_G.degree())
    top_nodes = sorted(degrees, key=degrees.get, reverse=True)[:35]
    for n, d in plot_G.nodes(data=True):
        if d.get("is_cross_cited") and n not in top_nodes:
            top_nodes.append(n)

    labels = {n: str(plot_G.nodes[n].get("label", n)) for n in top_nodes if n in plot_G}

    # Text rendering with dark backdrop bbox for clean legibility
    for node, label_txt in labels.items():
        if node not in pos:
            continue
        x, y = pos[node]
        ax.text(
            x,
            y + 0.025,
            label_txt,
            fontsize=8.5,
            fontweight="bold",
            color="#f8fafc",
            ha="center",
            va="bottom",
            bbox=dict(boxstyle="round,pad=0.22", facecolor="#0f172a", edgecolor="#334155", alpha=0.88, linewidth=0.7),
        )

    # Title & Subtitle block
    ax.text(
        0.02,
        0.97,
        "Investigator, Trial, Institution & Publication Collaboration Network",
        transform=ax.transAxes,
        fontsize=16,
        fontweight="bold",
        color="#f8fafc",
        va="top",
    )
    ax.text(
        0.02,
        0.94,
        f"Pruned collaborative backbone: {plot_G.number_of_nodes()} core nodes, {len(cite_edges)} direct PMID cross-citations. Single-trial leaf citations omitted for visual clarity.",
        transform=ax.transAxes,
        fontsize=10.5,
        color="#94a3b8",
        va="top",
    )

    # Legend
    legend = ax.legend(
        loc="lower left",
        frameon=True,
        facecolor="#0f172a",
        edgecolor="#334155",
        fontsize=10,
        labelcolor="#f8fafc",
        bbox_to_anchor=(0.02, 0.02),
    )
    ax.axis("off")

    fig.tight_layout()
    plt.savefig(file_png, dpi=300, bbox_inches="tight", facecolor="#090d16")
    plt.close()

    logger.info(f"Saved optimized, pruned Figure 8 to {file_png}")
    return str(file_png)


def plot_interactive_citation_network(
    G: nx.Graph,
    output_dir: str = "reports/figures",
) -> str:
    """
    Renders an interactive standalone WebGL network graph with Sigma.js, Graphology,
    and a custom real-time Force-Pushing Physics Simulation Engine.
    Features:
    - Live force-pushing anti-collision algorithm with interactive repulsion sliders.
    - Filter toggles: All nodes vs Shared Publications only vs Trials/PIs only.
    - Fast auto-focus search and comprehensive node inspection sidebar.
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    file_html = out_path / "investigator_citation_network.html"

    if len(G.nodes) == 0:
        return ""

    pos = compute_balanced_force_layout(G, scale=1200.0)

    nodes_list = []
    for node, d in G.nodes(data=True):
        x, y = pos.get(node, (0.0, 0.0))
        deg = G.degree(node)
        ntype = d.get("node_type", "General")
        is_shared = d.get("is_shared", False)
        is_cross = d.get("is_cross_cited", False)

        size = max(4.0, min(24.0, math.sqrt(deg + 1) * 3.6))

        nodes_list.append({
            "key": str(node),
            "attributes": {
                "label": str(d.get("label", node)),
                "node_type": ntype,
                "x": float(x),
                "y": float(y),
                "size": size,
                "color": d.get("color", "#0ea5e9"),
                "is_shared": bool(is_shared),
                "is_cross_cited": bool(is_cross),
                "url": d.get("url", ""),
                "doi_url": d.get("doi_url", ""),
                "details": d.get("details", ""),
                "degree": deg,
            },
        })

    edges_list = []
    edge_id = 0
    direct_citations_count = 0
    for u, v, d in G.edges(data=True):
        etype = d.get("edge_type", "CONNECTED")
        if etype == "DIRECT_CITATION":
            direct_citations_count += 1
            edges_list.append({
                "key": f"e_{edge_id}",
                "source": str(u),
                "target": str(v),
                "attributes": {
                    "edge_type": "DIRECT_CITATION",
                    "color": "#f43f5e",
                    "size": 2.4,
                    "type": "arrow",
                },
            })
        else:
            edges_list.append({
                "key": f"e_{edge_id}",
                "source": str(u),
                "target": str(v),
                "attributes": {
                    "edge_type": etype,
                    "color": "#334155",
                    "size": 0.8,
                },
            })
        edge_id += 1

    graphology_data = {
        "options": {"type": "undirected", "multi": False},
        "nodes": nodes_list,
        "edges": edges_list,
    }
    graph_json = json.dumps(graphology_data)

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Clinical Trials Collaboration & Citation Network</title>
  
  <!-- Graphology & Sigma.js WebGL Bundles -->
  <script src="https://cdn.jsdelivr.net/npm/graphology@0.25.4/dist/graphology.umd.min.js"></script>
  <script src="https://cdn.jsdelivr.net/npm/sigma@2.4.0/build/sigma.min.js"></script>
  
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
  <style>
    * {{
      box-sizing: border-box;
    }}
    body {{
      font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
      margin: 0;
      padding: 0;
      display: flex;
      height: 100vh;
      background: #090d16;
      color: #f8fafc;
      overflow: hidden;
    }}
    #app-container {{
      flex: 1;
      position: relative;
      height: 100%;
    }}
    #sigma-container {{
      width: 100%;
      height: 100%;
      background: radial-gradient(circle at center, #111827 0%, #030712 100%);
    }}
    #controls-toolbar {{
      position: absolute;
      top: 16px;
      left: 16px;
      z-index: 10;
      background: rgba(15, 23, 42, 0.92);
      backdrop-filter: blur(14px);
      border: 1px solid #334155;
      border-radius: 12px;
      padding: 10px 16px;
      display: flex;
      flex-wrap: wrap;
      align-items: center;
      gap: 10px;
      box-shadow: 0 12px 30px rgba(0,0,0,0.6);
      max-width: calc(100% - 440px);
    }}
    .search-input {{
      background: #090d16;
      border: 1px solid #475569;
      border-radius: 6px;
      padding: 7px 12px;
      color: #f8fafc;
      font-size: 0.85rem;
      outline: none;
      width: 200px;
      transition: border-color 0.2s;
    }}
    .search-input:focus {{
      border-color: #38bdf8;
    }}
    .tool-btn {{
      background: #1e293b;
      border: 1px solid #475569;
      color: #e2e8f0;
      padding: 7px 12px;
      border-radius: 6px;
      font-size: 0.82rem;
      font-weight: 600;
      cursor: pointer;
      display: flex;
      align-items: center;
      gap: 6px;
      transition: all 0.2s;
    }}
    .tool-btn:hover {{
      background: #334155;
      border-color: #38bdf8;
      color: #38bdf8;
    }}
    .tool-btn.active {{
      background: #0284c7;
      color: white;
      border-color: #38bdf8;
    }}
    .filter-select {{
      background: #090d16;
      border: 1px solid #475569;
      color: #e2e8f0;
      padding: 7px 10px;
      border-radius: 6px;
      font-size: 0.82rem;
      outline: none;
    }}
    .slider-group {{
      display: flex;
      align-items: center;
      gap: 8px;
      font-size: 0.78rem;
      color: #94a3b8;
    }}
    .slider-group input[type=range] {{
      width: 85px;
      accent-color: #38bdf8;
      cursor: pointer;
    }}
    #sidebar {{
      width: 410px;
      height: 100%;
      background: #0f172a;
      border-left: 1px solid #1e293b;
      padding: 24px;
      display: flex;
      flex-direction: column;
      box-shadow: -4px 0 25px rgba(0,0,0,0.6);
      overflow-y: auto;
      z-index: 20;
    }}
    h2 {{
      font-size: 1.25rem;
      margin: 0 0 8px 0;
      color: #38bdf8;
    }}
    .subtitle {{
      font-size: 0.82rem;
      color: #94a3b8;
      margin-bottom: 16px;
      line-height: 1.4;
    }}
    .stat-badge-grid {{
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 8px;
      margin-bottom: 16px;
    }}
    .stat-badge {{
      background: #1e293b;
      border: 1px solid #334155;
      padding: 10px;
      border-radius: 8px;
      font-size: 0.8rem;
    }}
    .stat-badge b {{
      display: block;
      font-size: 1.1rem;
      color: #f1f5f9;
      margin-top: 2px;
    }}
    .legend-box {{
      background: #1e293b;
      border: 1px solid #334155;
      padding: 12px;
      border-radius: 8px;
      margin-bottom: 16px;
    }}
    .legend-title {{
      font-size: 0.75rem;
      font-weight: 700;
      text-transform: uppercase;
      letter-spacing: 0.5px;
      color: #94a3b8;
      margin-bottom: 8px;
    }}
    .legend-grid {{
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 8px;
      font-size: 0.82rem;
    }}
    .legend-item {{
      display: flex;
      align-items: center;
      gap: 8px;
    }}
    .dot {{
      width: 11px;
      height: 11px;
      border-radius: 50%;
      display: inline-block;
    }}
    #details-panel {{
      background: #1e293b;
      padding: 18px;
      border-radius: 8px;
      border: 1px solid #334155;
      flex: 1;
      display: flex;
      flex-direction: column;
    }}
    .action-btn {{
      display: block;
      width: 100%;
      text-align: center;
      background: #0284c7;
      color: white;
      text-decoration: none;
      padding: 10px 14px;
      border-radius: 6px;
      font-weight: 600;
      margin-top: 10px;
      font-size: 0.85rem;
      transition: background 0.2s;
    }}
    .action-btn:hover {{
      background: #0369a1;
    }}
    .btn-ror {{ background: #d97706; }}
    .btn-ror:hover {{ background: #b45309; }}
    .btn-pubmed {{ background: #7c3aed; }}
    .btn-pubmed:hover {{ background: #6d28d9; }}
  </style>
</head>
<body>
  <div id="app-container">
    <div id="controls-toolbar">
      <input type="text" id="search-input" class="search-input" placeholder="Search trials, PMIDs, PIs...">
      
      <button id="force-toggle-btn" class="tool-btn">▶ Run Force-Pushing</button>
      <button id="force-step-btn" class="tool-btn">⚡ Step 60x</button>
      <button id="reset-zoom" class="tool-btn">🔍 Reset View</button>

      <div class="slider-group">
        <label for="repulsion-slider">Repulsion:</label>
        <input type="range" id="repulsion-slider" min="200" max="6000" step="200" value="1800">
      </div>

      <select id="node-filter-select" class="filter-select">
        <option value="all">Show All Nodes</option>
        <option value="shared_pubs">Shared & Cross-Cited Pubs Only</option>
        <option value="trials_pis">Trials & PIs Only</option>
      </select>
    </div>
    <div id="sigma-container"></div>
  </div>

  <div id="sidebar">
    <h2>Citation & Collaboration Network</h2>
    <div class="subtitle">
      Powered by <b>Sigma.js WebGL</b> and integrated <b>Real-Time Force-Pushing Physics Engine</b>.
    </div>

    <div class="stat-badge-grid">
      <div class="stat-badge">Total Network Nodes<b>{len(nodes_list):,}</b></div>
      <div class="stat-badge">Direct Cross-Citations<b>{direct_citations_count:,}</b></div>
    </div>

    <div class="legend-box">
      <div class="legend-title">Node Categories</div>
      <div class="legend-grid">
        <div class="legend-item"><span class="dot" style="background:#10b981;"></span> Clinical Trial (NCT)</div>
        <div class="legend-item"><span class="dot" style="background:#8b5cf6;"></span> Publication (PMID)</div>
        <div class="legend-item"><span class="dot" style="background:#f59e0b;"></span> Institution (ROR)</div>
        <div class="legend-item"><span class="dot" style="background:#0ea5e9;"></span> Principal Investigator</div>
        <div class="legend-item" style="grid-column: 1 / -1;"><span class="dot" style="background:#f43f5e;"></span> Direct Citation (cites-work)</div>
      </div>
    </div>

    <div id="details-panel">
      <h3 id="node-title" style="margin-top:0; color:#f1f5f9; font-size:1.1rem;">Select a Node</h3>
      <p id="node-type" style="color:#38bdf8; font-weight:600; font-size:0.85rem; margin-top:2px;">Click any element in the WebGL canvas</p>
      <div id="node-desc" style="font-size:0.85rem; color:#cbd5e1; line-height:1.5; margin-top:8px;"></div>
      <div id="node-actions" style="margin-top:auto; padding-top:16px;"></div>
    </div>
  </div>

  <script>
    const rawGraphData = {graph_json};
    const GraphConstructor = graphology.Graph || graphology;
    const graph = new GraphConstructor();

    // Import nodes and edges
    graph.import(rawGraphData);

    const container = document.getElementById('sigma-container');

    // Initialize Sigma.js WebGL renderer
    const renderer = new Sigma(graph, container, {{
      renderEdgeLabels: false,
      enableEdgeEvents: true,
      minCameraRatio: 0.04,
      maxCameraRatio: 12,
      labelFont: 'Inter, sans-serif',
      labelWeight: '600',
      labelColor: {{ color: '#f8fafc' }},
      labelSize: 11,
      zIndex: true
    }});

    // === FORCE-PUSHING PHYSICS ENGINE ===
    let simulationRunning = false;
    let animFrameId = null;
    let repulsionStrength = 1800;
    const springTension = 0.0035;
    const centerGravity = 0.0008;
    const friction = 0.88;

    // Node velocity vectors
    const velocities = {{}};
    graph.forEachNode(node => {{
      velocities[node] = {{ vx: 0, vy: 0 }};
    }});

    function stepPhysicsSimulation(iterations = 1) {{
      const nodes = graph.nodes();
      const nodeCount = nodes.length;
      if (nodeCount === 0) return;

      const nodeData = nodes.map(n => ({{
        id: n,
        x: graph.getNodeAttribute(n, 'x'),
        y: graph.getNodeAttribute(n, 'y'),
        size: graph.getNodeAttribute(n, 'size') || 10,
        hidden: graph.getNodeAttribute(n, 'hidden') || false,
        deg: graph.degree(n) || 1
      }}));

      const cellSize = 180;

      for (let iter = 0; iter < iterations; iter++) {{
        // Build spatial grid
        const grid = {{}};
        for (let i = 0; i < nodeCount; i++) {{
          const a = nodeData[i];
          if (a.hidden) continue;
          const cx = Math.floor(a.x / cellSize);
          const cy = Math.floor(a.y / cellSize);
          const key = cx + ',' + cy;
          if (!grid[key]) grid[key] = [];
          grid[key].push(i);
        }}

        // 1. Repulsion & Anti-Collision Force-Pushing (Spatial grid accelerated)
        for (let i = 0; i < nodeCount; i++) {{
          const a = nodeData[i];
          if (a.hidden) continue;
          let fx = 0;
          let fy = 0;

          const cx = Math.floor(a.x / cellSize);
          const cy = Math.floor(a.y / cellSize);

          // Check 3x3 neighbor grid cells
          for (let ox = -1; ox <= 1; ox++) {{
            for (let oy = -1; oy <= 1; oy++) {{
              const cellList = grid[(cx + ox) + ',' + (cy + oy)];
              if (!cellList) continue;

              for (let k = 0; k < cellList.length; k++) {{
                const j = cellList[k];
                if (i === j) continue;
                const b = nodeData[j];

                const dx = a.x - b.x;
                const dy = a.y - b.y;
                const distSq = dx * dx + dy * dy + 1e-2;
                const dist = Math.sqrt(distSq);

                // Repulsion
                const repForce = (repulsionStrength * (1 + Math.log(a.deg + 1))) / distSq;
                fx += (dx / dist) * repForce;
                fy += (dy / dist) * repForce;

                // Anti-Collision Push
                const minAllowedDist = a.size + b.size + 16;
                if (dist < minAllowedDist) {{
                  const push = (minAllowedDist - dist) * 0.6;
                  fx += (dx / dist) * push;
                  fy += (dy / dist) * push;
                }}
              }}
            }}
          }}

          // 2. Center gravity
          fx -= a.x * centerGravity * (1 + a.deg * 0.4);
          fy -= a.y * centerGravity * (1 + a.deg * 0.4);

          const v = velocities[a.id] || (velocities[a.id] = {{ vx: 0, vy: 0 }});
          v.vx = (v.vx + fx) * friction;
          v.vy = (v.vy + fy) * friction;
        }}

        // 3. Spring Attraction along Edges
        graph.forEachEdge((edge, attrs, source, target) => {{
          const aHidden = graph.getNodeAttribute(source, 'hidden');
          const bHidden = graph.getNodeAttribute(target, 'hidden');
          if (aHidden || bHidden) return;

          const ax = graph.getNodeAttribute(source, 'x');
          const ay = graph.getNodeAttribute(source, 'y');
          const bx = graph.getNodeAttribute(target, 'x');
          const by = graph.getNodeAttribute(target, 'y');

          const dx = bx - ax;
          const dy = by - ay;
          const dist = Math.sqrt(dx * dx + dy * dy) + 1e-3;

          const targetDist = attrs.edge_type === 'DIRECT_CITATION' ? 100 : 170;
          const springForce = (dist - targetDist) * springTension;

          const sx = (dx / dist) * springForce;
          const sy = (dy / dist) * springForce;

          const va = velocities[source];
          const vb = velocities[target];
          if (va) {{ va.vx += sx; va.vy += sy; }}
          if (vb) {{ vb.vx -= sx; vb.vy -= sy; }}
        }});

        // 4. Update Node Positions
        for (let i = 0; i < nodeCount; i++) {{
          const item = nodeData[i];
          if (item.hidden) continue;
          const v = velocities[item.id];
          if (!v) continue;

          const maxStep = 40;
          const speed = Math.sqrt(v.vx * v.vx + v.vy * v.vy);
          if (speed > maxStep) {{
            v.vx = (v.vx / speed) * maxStep;
            v.vy = (v.vy / speed) * maxStep;
          }}

          item.x += v.vx;
          item.y += v.vy;
          graph.setNodeAttribute(item.id, 'x', item.x);
          graph.setNodeAttribute(item.id, 'y', item.y);
        }}
      }}

      renderer.refresh();
    }}

    function physicsLoop() {{
      if (simulationRunning) {{
        stepPhysicsSimulation(1);
        animFrameId = requestAnimationFrame(physicsLoop);
      }}
    }}

    // Toggle simulation
    const toggleBtn = document.getElementById('force-toggle-btn');
    toggleBtn.addEventListener('click', () => {{
      simulationRunning = !simulationRunning;
      if (simulationRunning) {{
        toggleBtn.classList.add('active');
        toggleBtn.innerText = '⏸ Pause Force-Pushing';
        physicsLoop();
      }} else {{
        toggleBtn.classList.remove('active');
        toggleBtn.innerText = '▶ Run Force-Pushing';
        if (animFrameId) cancelAnimationFrame(animFrameId);
      }}
    }});

    // Step button
    document.getElementById('force-step-btn').addEventListener('click', () => {{
      stepPhysicsSimulation(60);
    }});

    // Repulsion slider
    document.getElementById('repulsion-slider').addEventListener('input', (e) => {{
      repulsionStrength = parseFloat(e.target.value);
    }});

    // Filtering
    document.getElementById('node-filter-select').addEventListener('change', (e) => {{
      const mode = e.target.value;
      graph.forEachNode(node => {{
        const ntype = graph.getNodeAttribute(node, 'node_type');
        const isShared = graph.getNodeAttribute(node, 'is_shared');
        const isCross = graph.getNodeAttribute(node, 'is_cross_cited');

        let hidden = false;
        if (mode === 'shared_pubs') {{
          if (ntype === 'Publication' && !isShared && !isCross) {{
            hidden = true;
          }}
        }} else if (mode === 'trials_pis') {{
          if (ntype !== 'Trial' && ntype !== 'PI') {{
            hidden = true;
          }}
        }}
        graph.setNodeAttribute(node, 'hidden', hidden);
      }});
      renderer.refresh();
    }});

    // State & Selection
    let selectedNode = null;

    renderer.on('clickNode', ({{ node }}) => {{
      selectedNode = node;
      const attrs = graph.getNodeAttributes(node);

      document.getElementById('node-title').innerText = attrs.label || node;
      document.getElementById('node-type').innerText = 'Type: ' + (attrs.node_type || 'Node') + ' (Degree: ' + (attrs.degree || 0) + ')';
      document.getElementById('node-desc').innerHTML = attrs.details || '';

      let actionsHtml = '';
      if (attrs.node_type === 'Trial') {{
        actionsHtml = `<a class="action-btn" href="${{attrs.url}}" target="_blank">🌐 View on ClinicalTrials.gov (${{node}})</a>`;
      }} else if (attrs.node_type === 'Publication') {{
        actionsHtml = `<a class="action-btn btn-pubmed" href="${{attrs.url}}" target="_blank">📚 View on PubMed (${{node}})</a>`;
        if (attrs.doi_url) {{
          actionsHtml += `<a class="action-btn" style="background:#059669;" href="${{attrs.doi_url}}" target="_blank">🔗 Open Article DOI</a>`;
        }}
      }} else if (attrs.node_type === 'Sponsor') {{
        actionsHtml = `<a class="action-btn btn-ror" href="${{attrs.url}}" target="_blank">🏛️ Research Organization Registry (ROR)</a>`;
      }} else if (attrs.node_type === 'PI') {{
        actionsHtml = `<a class="action-btn" href="${{attrs.url}}" target="_blank">💼 View NIH RePORTER Profile</a>`;
      }}

      document.getElementById('node-actions').innerHTML = actionsHtml;

      const nodePos = renderer.getNodeDisplayData(node);
      if (nodePos) {{
        renderer.getCamera().animate({{ x: nodePos.x, y: nodePos.y, ratio: 0.25 }}, {{ duration: 500 }});
      }}
    }});

    renderer.on('doubleClickNode', ({{ node }}) => {{
      const attrs = graph.getNodeAttributes(node);
      if (attrs.url) {{
        window.open(attrs.url, '_blank');
      }}
    }});

    document.getElementById('reset-zoom').addEventListener('click', () => {{
      renderer.getCamera().animatedReset({{ duration: 500 }});
    }});

    document.getElementById('search-input').addEventListener('input', (e) => {{
      const query = e.target.value.toLowerCase().trim();
      if (!query) return;

      const match = graph.nodes().find(n => {{
        const label = (graph.getNodeAttribute(n, 'label') || '').toLowerCase();
        return label.includes(query) || n.toLowerCase().includes(query);
      }});

      if (match) {{
        const nodePos = renderer.getNodeDisplayData(match);
        if (nodePos) {{
          renderer.getCamera().animate({{ x: nodePos.x, y: nodePos.y, ratio: 0.25 }}, {{ duration: 500 }});
        }}
      }}
    }});

    // Auto-run initial 40 iterations of force-pushing on load for clean spacing
    setTimeout(() => {{
      stepPhysicsSimulation(40);
    }}, 200);
  </script>
</body>
</html>
"""

    with open(file_html, "w", encoding="utf-8") as f:
        f.write(html_content)

    logger.info(
        f"Saved Sigma.js WebGL network with Real-Time Force-Pushing engine "
        f"({len(nodes_list)} nodes, {direct_citations_count} cross-citations) to {file_html}"
    )
    return str(file_html)
