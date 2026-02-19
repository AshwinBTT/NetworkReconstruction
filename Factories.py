#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Factories.py
============

Factory-level extension for a weighted firm-to-firm network (v3_netRecon: factory extension).

This script takes a reconstructed *firm-level* weighted network and allocates each firm-to-firm
link across factories (production units) using:
  - great-circle distance (Haversine),
  - an exponential distance kernel G(d; τ),
  - and within-firm factory prominence weights p_i(a).

The construction preserves:
  - firm-level adjacency (no new firm pairs; no firm self-links),
  - and total firm-to-firm weights.

Usage
-----
By default the script uses the same paths as the project version. For GitHub use, you can
override paths/parameters from the command line:

    python3 Factories.py --factories-csv <factories.csv> --weighted-network <weighted_network.txt> --output <out.txt>
    python3 Factories.py --tau-km 150 --seed 1 ...

If you do not pass flags, the original defaults (input_files/... and output/...) are used.
"""


import os
import argparse
import math
from collections import defaultdict

import numpy as np
import pandas as pd


# --------------------------------------------------------------------
# Config
# --------------------------------------------------------------------
FACTORY_CSV_PRIMARY = "input_files/top100000_firms (1).csv"
FACTORY_CSV_FALLBACK = "input_files/top100000_firms.csv"
WEIGHTED_NETWORK_FILE = "input_files/weighted_network.txt"
OUTPUT_FILE = "output/factory_network_weighted.txt"

TAU_KM = 100.0            # τ in kilometers for G(d; τ) = exp(-d/τ)
EARTH_RADIUS_KM = 6371.0  # Earth radius for great-circle distance
RNG_SEED = 12345          # fixed seed for reproducibility


# --------------------------------------------------------------------
# Utilities
# --------------------------------------------------------------------
def log(msg: str) -> None:
    print(f"[LOG] {msg}", flush=True)


def normalize_cols(df: pd.DataFrame) -> pd.DataFrame:
    """Lowercase + underscore column names, strip whitespace."""
    df = df.copy()
    df.columns = (
        df.columns.astype(str)
        .str.strip()
        .str.lower()
        .str.replace(r"\s+", "_", regex=True)
    )
    return df


def haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Great-circle distance between two points (radians) in kilometers.

    Inputs are lat/lon *in radians*, consistent with the TeX description.
    """
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = (
        math.sin(dlat / 2.0) ** 2
        + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2.0) ** 2
    )
    # numerical guard
    a = min(1.0, max(0.0, a))
    return 2.0 * EARTH_RADIUS_KM * math.asin(math.sqrt(a))


def G(d_km: float) -> float:
    """
    Distance kernel G(d; τ) = exp(-d / τ) in (0, 1].
    """
    return math.exp(-d_km / TAU_KM)


# --------------------------------------------------------------------
# Load factories
# --------------------------------------------------------------------
def load_factories():
    """
    Load factory coordinates and firm IDs, normalize columns, and assign
    a factory index 0..N-1.

    Returns
    -------
    fac_df : DataFrame
        Columns:
            - firm_id (str)
            - lat_rad, lon_rad (float, radians)
            - factory_idx (int, 0-based)
    firm_to_factories : dict[str, list[int]]
        firm_id -> list of factory indices.
    """
    path = FACTORY_CSV_PRIMARY if os.path.exists(FACTORY_CSV_PRIMARY) else FACTORY_CSV_FALLBACK
    log(f"Reading factories from {path}")
    df = pd.read_csv(path)
    df = normalize_cols(df)

    id_col = next((c for c in df.columns if "ciq_id" in c or "firm" in c), None)
    lat_col = next((c for c in df.columns if "lat" in c), None)
    lon_col = next((c for c in df.columns if "lon" in c), None)
    if not id_col or not lat_col or not lon_col:
        raise ValueError("Could not automatically detect id/lat/lon columns in factory file.")

    # keep only necessary columns and drop rows with missing lat/lon/id
    df = df[[id_col, lat_col, lon_col]].dropna()
    df[id_col] = df[id_col].astype(str)

    # convert to radians (TeX requires (φ_a, λ_a) in radians)
    df["lat_rad"] = np.deg2rad(df[lat_col].to_numpy(dtype=float))
    df["lon_rad"] = np.deg2rad(df[lon_col].to_numpy(dtype=float))

    # assign factory indices 0..N-1
    n_factories = len(df)
    df["factory_idx"] = np.arange(n_factories, dtype=int)

    log(f"Factories loaded: {n_factories}")

    firm_to_factories: dict[str, list[int]] = defaultdict(list)
    for row in df.itertuples(index=False):
        firm_id = getattr(row, id_col)
        factory_idx = int(row.factory_idx)
        firm_to_factories[firm_id].append(factory_idx)

    fac_df = df.rename(columns={id_col: "firm_id"})
    return fac_df, firm_to_factories


# --------------------------------------------------------------------
# Load weighted firm network
# --------------------------------------------------------------------
def load_weighted_network(firm_to_factories):
    """
    Load weighted firm-level network and restrict to firms that have factories.

    We treat each line src dst weight as a single firm-level link (i -> j)
    with associated weight. If multiple rows appear for the same pair (src,dst),
    their weights are summed, so W[(src,dst)] is the total weight between
    those firms.

    Returns
    -------
    net_df : DataFrame
        Columns ["src", "dst", "weight"] (firm_id strings), after filtering.
    firms : list[str]
        Sorted list of firm_ids that have factories (some may have no edges).
    firm_index : dict[str, int]
        firm_id -> integer index 0..F-1.
    neigh_out : dict[int, list[int]]
        firm_idx -> sorted list of destination firm indices (firm-level adjacency).
    W : dict[(str, str), float]
        (src_firm, dst_firm) -> total weight.
    """
    if not os.path.exists(WEIGHTED_NETWORK_FILE):
        raise FileNotFoundError(WEIGHTED_NETWORK_FILE)

    df = pd.read_csv(WEIGHTED_NETWORK_FILE, sep=r"\s+", header=None, engine="python")
    if df.shape[1] != 3:
        raise ValueError("weighted_network.txt must have exactly 3 columns: src dst weight")

    df.columns = ["src", "dst", "weight"]
    df["src"] = df["src"].astype(str)
    df["dst"] = df["dst"].astype(str)

    # Exclude self-links at the firm level (no i -> i)
    df = df[df["src"] != df["dst"]]

    # Only keep firm pairs where both firms have at least one factory
    valid_firms = set(firm_to_factories.keys())
    df = df[df["src"].isin(valid_firms) & df["dst"].isin(valid_firms)]

    if df.empty:
        raise ValueError("No firm-level edges remain after filtering to firms with factories.")

    log(f"Weighted firm edges (after filtering rows): {len(df)}")

    # Aggregate weights per firm pair (src,dst)
    grouped = df.groupby(["src", "dst"], as_index=False)["weight"].sum()
    grouped["weight"] = grouped["weight"].astype(float)
    log(f"Unique firm-level links (after collapsing duplicates): {len(grouped)}")

    # Build weight dict W[(src, dst)] = total weight
    W: dict[tuple[str, str], float] = {}
    for row in grouped.itertuples(index=False):
        s = str(row.src)
        d = str(row.dst)
        w = float(row.weight)
        W[(s, d)] = w

    # Build firm list / indices: all firms that have factories (even if no edges)
    firms = sorted(valid_firms)
    firm_index = {f: i for i, f in enumerate(firms)}

    # Build outgoing neighbor lists from W
    neigh_out: dict[int, list[int]] = defaultdict(list)
    for (s, d), w in W.items():
        isrc = firm_index[s]
        idst = firm_index[d]
        neigh_out[isrc].append(idst)

    # Deduplicate and sort neighbors for consistency
    for i in neigh_out:
        neigh_out[i] = sorted(set(neigh_out[i]))

    # Return df of *unique* firm pairs, not raw rows
    net_df = grouped.rename(columns={"src": "src", "dst": "dst", "weight": "weight"})

    return net_df, firms, firm_index, neigh_out, W


# --------------------------------------------------------------------
# L_i(a) and prominence p_i(a)
# --------------------------------------------------------------------
def compute_L_p(fac_df: pd.DataFrame, firm_index: dict[str, int]):
    """
    Compute L_i(a) and p_i(a) as in the TeX:

      L_i(a) = sum_{j != i} sum_{b in A(j)} G(d(a,b); τ)

    and then normalize within each firm i to get p_i(a).

    This is O(N^2) in the number of factories and may be expensive for
    very large datasets; it is the direct literal implementation.
    """
    fac_df = fac_df.copy()
    fac_df["firm_idx"] = fac_df["firm_id"].map(firm_index)
    if fac_df["firm_idx"].isnull().any():
        raise ValueError("Some factories have firm_ids not present in firm_index.")

    lat = fac_df["lat_rad"].to_numpy(dtype=float)
    lon = fac_df["lon_rad"].to_numpy(dtype=float)
    firm_idx = fac_df["firm_idx"].to_numpy(dtype=int)

    n = len(fac_df)
    L = np.zeros(n, dtype=float)
    log("Computing prominence scores L_i(a) (O(N^2))...")

    # Direct N^2 computation, vectorized over the 'other factories' for each a
    for a in range(n):
        la = lat[a]
        loa = lon[a]
        fi = firm_idx[a]

        dlat = lat - la
        dlon = lon - loa
        aterm = np.sin(dlat / 2.0) ** 2 + np.cos(la) * np.cos(lat) * np.sin(dlon / 2.0) ** 2
        dkm = 2.0 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(np.clip(aterm, 0.0, 1.0)))
        g = np.exp(-dkm / TAU_KM)

        # exclude factories of the same firm j = i
        g[firm_idx == fi] = 0.0
        L[a] = g.sum()

    # Normalize within each firm to get p_i(a)
    p = np.zeros(n, dtype=float)
    unique_firms = np.unique(firm_idx)
    for i in unique_firms:
        idx = np.where(firm_idx == i)[0]
        tot = L[idx].sum()
        if tot > 0.0:
            p[idx] = L[idx] / tot
        else:
            # If a firm is completely isolated (L_i(a) = 0 for all a),
            # assign equal prominence to its factories.
            p[idx] = 1.0 / len(idx)

    return fac_df, L, p


# --------------------------------------------------------------------
# Assign edges according to the two-step procedure
# --------------------------------------------------------------------
def assign_edges(
    fac_df: pd.DataFrame,
    firms: list[str],
    neigh_out: dict[int, list[int]],
    p: np.ndarray,
) -> dict[tuple[int, int], list[tuple[int, int]]]:
    """
    Implement the factory-level assignment as in the TeX:

      - Step 1: activate all factories (two cases d_i^O >= k_i vs d_i^O < k_i).
      - Step 2: allocate remaining links using prominence p_i and a
        distance-based, capacity-aware connection rule (dynamic Q).

    Assumptions
    -----------
    - The weighted firm network is a simple directed graph at the firm level:
      at most one link (with some weight) from firm i to firm j. Then
      d_i^O is the number of distinct outgoing neighbors of i.

    Returns
    -------
    edges : dict[(i_idx, j_idx), list[(src_factory_idx, dst_factory_idx)]]
        For each firm pair (i_idx, j_idx), a list of factory-level edges.
    """
    rng = np.random.default_rng(RNG_SEED)

    # Prepare arrays for quick access
    lat = fac_df["lat_rad"].to_numpy(dtype=float)
    lon = fac_df["lon_rad"].to_numpy(dtype=float)
    firm_idx_arr = fac_df["firm_idx"].to_numpy(dtype=int)
    factory_idx_arr = fac_df["factory_idx"].to_numpy(dtype=int)

    n_factories = len(fac_df)
    n_firms = len(firms)

    # Build firm_idx -> list of factories A(i)
    firm_idx_to_factories: list[list[int]] = [[] for _ in range(n_firms)]
    for idx in range(n_factories):
        i = firm_idx_arr[idx]
        if 0 <= i < n_firms:
            a = int(factory_idx_arr[idx])
            firm_idx_to_factories[i].append(a)

    edges: dict[tuple[int, int], list[tuple[int, int]]] = defaultdict(list)

    # ----------------------------------------------------------------
    # Helper: sample destination factory b for a given source factory a,
    # given a set of candidate neighbor firms J_plus with capacity>0.
    # This is the implicit "row" of the dynamic Q matrix for factory a.
    # ----------------------------------------------------------------
    def sample_destination_for_a(a: int, candidate_js: list[int]) -> tuple[int, int] | None:
        """
        Given a source factory index a and a list of neighbor firm indices
        candidate_js with remaining capacity, sample a destination factory b.

        Returns
        -------
        (b, j) or None if no valid candidate exists.
        """
        if not candidate_js:
            return None

        lat_a = float(lat[a])
        lon_a = float(lon[a])

        cand_b: list[int] = []
        weights: list[float] = []

        for j in candidate_js:
            factories_j = firm_idx_to_factories[j]
            if not factories_j:
                continue
            for b in factories_j:
                d_km = haversine(lat_a, lon_a, float(lat[b]), float(lon[b]))
                w = G(d_km)
                if w > 0.0:
                    cand_b.append(b)
                    weights.append(w)

        if not cand_b:
            return None

        w_arr = np.asarray(weights, dtype=float)
        s = w_arr.sum()
        if s <= 0.0:
            probs = np.full_like(w_arr, 1.0 / len(w_arr), dtype=float)
        else:
            probs = w_arr / s

        idx_choice = rng.choice(len(cand_b), p=probs)
        b = int(cand_b[idx_choice])
        j = int(firm_idx_arr[b])  # firm index of destination
        return b, j

    # For each firm i, execute Step 1 and Step 2
    for i in range(n_firms):
        A_i = firm_idx_to_factories[i]
        if not A_i:
            # No factories for this firm -> nothing to assign
            continue

        out_js = neigh_out.get(i, [])
        if not out_js:
            # No outgoing firm-level links -> factories remain inactive
            continue

        k_i = len(A_i)          # number of factories
        d_iO = len(out_js)      # number of distinct outgoing firm neighbors (links)

        # A_i as array, randomized order to avoid positional bias
        A_i_arr = np.array(A_i, dtype=int)
        factory_order = rng.permutation(A_i_arr)

        # ------------------------------------------------------------
        # Case 1: d_i^O >= k_i
        # ------------------------------------------------------------
        if d_iO >= k_i:
            # cap_ij[j] is the remaining capacity (here: number of remaining links
            # from i to j). With the "one link per neighbor" assumption, this is
            # 0 or 1.
            cap_ij: dict[int, int] = {j: 1 for j in out_js}

            # -----------------------------
            # Step 1: activate factories
            # -----------------------------
            for a in factory_order:
                # Candidate neighbors with remaining capacity
                candidate_js = [j for j, cap in cap_ij.items() if cap > 0]
                if not candidate_js:
                    # No remaining capacity from i to any neighbor; we are done.
                    break

                dest = sample_destination_for_a(int(a), candidate_js)
                if dest is None:
                    continue
                b, j = dest

                edges[(i, j)].append((int(a), int(b)))
                cap_ij[j] -= 1
                # When cap_ij[j] hits 0, that firm j drops out of candidate_js
                # in subsequent calls, exactly as in the dynamic Q description.

            # After Step 1, some links (i -> j) may still have cap_ij[j] == 1.
            # These correspond to the remaining r_i = d_i^O - k_i links.

            # -----------------------------
            # Step 2: allocate remaining links using prominence p_i
            # -----------------------------
            # Restrict prominence vector to factories of firm i
            p_i = p[A_i_arr]
            s_p = p_i.sum()
            if s_p <= 0.0:
                p_i = np.full_like(p_i, 1.0 / len(p_i), dtype=float)
            else:
                p_i = p_i / s_p

            # Total remaining capacity r_i
            r_i = sum(cap for cap in cap_ij.values() if cap > 0)

            for _ in range(r_i):
                # If for some reason capacities are exhausted early, stop
                candidate_js = [j for j, cap in cap_ij.items() if cap > 0]
                if not candidate_js:
                    break

                # 1) pick source factory a from prominence distribution p_i
                idx_a = rng.choice(len(A_i_arr), p=p_i)
                a = int(A_i_arr[idx_a])

                # 2) pick destination factory b via distance-based kernel over
                #    all factories of neighbors with remaining capacity
                dest = sample_destination_for_a(a, candidate_js)
                if dest is None:
                    # No valid destination: skip this iteration
                    continue
                b, j = dest

                edges[(i, j)].append((a, b))
                cap_ij[j] -= 1

            # All for this firm i in the d_i^O >= k_i case.

        # ------------------------------------------------------------
        # Case 2: d_i^O < k_i (more factories than outgoing links)
        # ------------------------------------------------------------
        else:
            # Here, some firm-level links (i -> j) will be shared by multiple
            # factories. We still loop once over factories, but we ensure that
            # every firm-level neighbor j appears at least once, so that
            # W_ij > 0 is represented at the factory level and total weight
            # from i to j can be split evenly across those edges.
            use_count: dict[int, int] = {j: 0 for j in out_js}

            for idx_in_order, a in enumerate(factory_order):
                if idx_in_order < d_iO:
                    # For the first d_i^O factories, force coverage:
                    # assign each link (i -> j) at least once.
                    candidate_js = [j for j, cnt in use_count.items() if cnt == 0]
                    if not candidate_js:
                        candidate_js = out_js
                else:
                    # Additional factories: links can be shared across factories.
                    candidate_js = out_js

                if not candidate_js:
                    # No possible destinations for this factory
                    continue

                dest = sample_destination_for_a(int(a), candidate_js)
                if dest is None:
                    continue
                b, j = dest

                edges[(i, j)].append((int(a), int(b)))
                use_count[j] += 1

            # In this case, r_i = 0 (all firm-level links have been "used" at
            # least once), and Step 2 is not needed. We will later split
            # W_ij evenly across all factory edges corresponding to (i, j),
            # so that total flow between firms is preserved.

    return edges


# --------------------------------------------------------------------
# Write output
# --------------------------------------------------------------------
def write_output(
    fac_df: pd.DataFrame,
    firms: list[str],
    W: dict[tuple[str, str], float],
    edges: dict[tuple[int, int], list[tuple[int, int]]],
) -> None:
    """
    Write the factory-level network to OUTPUT_FILE.

    Each firm-level total weight W[src_firm, dst_firm] is split evenly
    across all factory edges (a -> b) corresponding to that firm pair.
    """
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

    # Convert radians back to degrees for output
    lat_deg = np.rad2deg(fac_df["lat_rad"].to_numpy(dtype=float))
    lon_deg = np.rad2deg(fac_df["lon_rad"].to_numpy(dtype=float))

    count = 0
    with open(OUTPUT_FILE, "w") as f:
        f.write("# src dst srcF dstF srcLat srcLon dstLat dstLon weight\n")
        for (i_idx, j_idx), ab_list in edges.items():
            src_firm = firms[i_idx]
            dst_firm = firms[j_idx]

            key = (src_firm, dst_firm)
            if key not in W:
                # This should not happen if edges were only created for valid firm pairs
                continue

            w_ij = W[key]
            m = len(ab_list)
            if m <= 0:
                # No factory edges for this firm pair; skip
                continue

            per_weight = w_ij / float(m)

            for a, b in ab_list:
                # a, b are factory indices (0..N-1)
                f.write(
                    f"{src_firm} {dst_firm} {a} {b} "
                    f"{lat_deg[a]:.6f} {lon_deg[a]:.6f} "
                    f"{lat_deg[b]:.6f} {lon_deg[b]:.6f} "
                    f"{per_weight:.10e}\n"
                )
                count += 1

    log(f"Wrote {count} factory edges to {OUTPUT_FILE}")


# --------------------------------------------------------------------
# Main
# --------------------------------------------------------------------
def main():
    fac_df, firm_to_factories = load_factories()
    net_df, firms, firm_index, neigh_out, W = load_weighted_network(firm_to_factories)

    # Compute L_i(a) and p_i(a), and attach firm_idx to fac_df
    fac_df, L, p = compute_L_p(fac_df, firm_index)

    # Assign factory-level edges according to the two-step procedure
    edges = assign_edges(fac_df, firms, neigh_out, p)

    # Write the resulting factory network with weights
    write_output(fac_df, firms, W, edges)
    log("Done.")


if __name__ == "__main__":
    main()
# --------------------------------------------------------------------
# GitHub-friendly CLI wrapper
# --------------------------------------------------------------------
def cli_main() -> None:
    """
    Thin wrapper around `main()` that lets you override paths and parameters
    from the command line. Core functions stay unchanged.
    """
    parser = argparse.ArgumentParser(
        description="Factory-level extension: allocate firm-level weighted links across factories using a distance kernel."
    )
    parser.add_argument("--factories-csv", default=FACTORY_CSV_PRIMARY,
                        help=f"Primary factories CSV (default: {FACTORY_CSV_PRIMARY})")
    parser.add_argument("--factories-csv-fallback", default=FACTORY_CSV_FALLBACK,
                        help=f"Fallback factories CSV if primary is missing (default: {FACTORY_CSV_FALLBACK})")
    parser.add_argument("--weighted-network", default=WEIGHTED_NETWORK_FILE,
                        help=f"Firm-level weighted edge list (default: {WEIGHTED_NETWORK_FILE})")
    parser.add_argument("--output", default=OUTPUT_FILE,
                        help=f"Output file for factory network (default: {OUTPUT_FILE})")
    parser.add_argument("--tau-km", type=float, default=TAU_KM,
                        help=f"Distance decay τ in km for exp(-d/τ) (default: {TAU_KM})")
    parser.add_argument("--seed", type=int, default=RNG_SEED,
                        help=f"Random seed (default: {RNG_SEED})")

    args = parser.parse_args()

    # Override module-level defaults (the existing functions read these globals).
    global FACTORY_CSV_PRIMARY, FACTORY_CSV_FALLBACK, WEIGHTED_NETWORK_FILE, OUTPUT_FILE, TAU_KM, RNG_SEED
    FACTORY_CSV_PRIMARY = args.factories_csv
    FACTORY_CSV_FALLBACK = args.factories_csv_fallback
    WEIGHTED_NETWORK_FILE = args.weighted_network
    OUTPUT_FILE = args.output
    TAU_KM = float(args.tau_km)
    RNG_SEED = int(args.seed)

    main()


if __name__ == "__main__":
    cli_main()
