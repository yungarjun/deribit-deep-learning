import pandas as pd
import numpy as np
import pyarrow.parquet as pq
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
import torch
import matplotlib.pyplot as plt
import scipy.sparse as sp
import cvxpy as cp

# Read in raw parquet
def read_parquet(file_path):
    """
    Reads a parquet file and returns a pandas DataFrame.
    """
    table = pq.read_table(file_path)
    df = table.to_pandas()
    return df

# Cleaning function for Deribit options data
def clean_deribit(df, r=0, q=0):

    # Seperate Option Type, Strike and Maturity
    df[['asset', 'expiry', 'strike', 'option_type']] = df['instrument_name'].str.split('-', expand=True)

    # Define maturity in years 'tau'
    df['expiry'] = pd.to_datetime(df['expiry'])
    reference_date = pd.to_datetime("2025-01-30")
    df['tau'] = (df['expiry'] - reference_date).dt.days / 365.25  # Crypto is traded 24/7

    df['F'] = df['underlying_price'] * np.exp((r - q) * df['tau'])

    # Filter for just calls and usd volume > 0 
    df = df[(df['option_type'] == 'C') & (df['stats_volume_usd'] > 0)]

    # Define mid price
    df['mid_price'] = ((df['best_bid_price'] + df['best_ask_price']) / 2) 

    # Convert strike to numeric
    df['strike'] = pd.to_numeric(df['strike'], errors='coerce')

    # Define log moneyness, 'm'
    df['m'] = np.log(df['strike'] / df['F'])

    df['c_tilde'] = df['mid_price']


    return df

def build_lattice_grid(df, n_tau=5, n_m=5, top_K=50):
    """
    From raw df (train only), learn:
      - tau_grid, m_grid
      - full nodes array
      - fitted NearestNeighbors on those nodes
    """
    # 1) learn tau_grid via KMeans
    unique_taus = np.sort(df['tau'].unique()).reshape(-1,1)
    tau_kmeans = KMeans(n_clusters=n_tau, random_state=0).fit(unique_taus)
    tau_grid   = tau_kmeans.cluster_centers_.flatten()
    df['tau_cluster'] = tau_kmeans.predict(df[['tau']])
    
    # 2) build m_grid for each tau from percentiles
    m_grid = {}
    for label, tau in enumerate(tau_grid):
        subset = df[df['tau_cluster']==label]
        lo, hi  = np.percentile(subset['m'], [1,99])
        m_grid[tau] = np.linspace(lo, hi, n_m)
    
    # 3) build nodes & NN
    nodes = np.vstack([ [tau, m] 
                        for tau in tau_grid 
                        for m in m_grid[tau] ])
    nn    = NearestNeighbors(n_neighbors=1).fit(nodes)
    
    return nn, nodes, tau_grid, m_grid

def build_lattice_grid(df, n_tau=5, n_m=5, top_K=50):
    """
    Learn a *sorted* τ-grid and consistent labels, then build m-grids, nodes, and a 1-NN snapper.
    """
    df = df.copy()

    # --- 1) KMeans on τ, then sort centers and RELABEL ---
    unique_taus = np.sort(df['tau'].unique()).reshape(-1, 1)
    km = KMeans(n_clusters=n_tau, random_state=0).fit(unique_taus)

    centers = km.cluster_centers_.flatten()          # unsorted centers
    order   = np.argsort(centers)                    # permutation that sorts them
    tau_grid = centers[order]                        # sorted centers, shape [n_tau]

    # map old label -> new sorted label
    old2new = {old: new for new, old in enumerate(order)}

    # label every row via KMeans.predict, then remap to sorted indices
    raw_lab = km.predict(df[['tau']].values)         # 0..n_tau-1 (arbitrary order)
    df['tau_cluster'] = np.array([old2new[l] for l in raw_lab], dtype=int)

    # --- 2) m-grid per sorted τ-cluster (use robust percentiles) ---
    m_grid = {}
    for i, τ in enumerate(tau_grid):
        sub = df[df['tau_cluster'] == i]
        if len(sub) == 0:
            # fallback if cluster empty (rare): borrow global range
            lo, hi = np.percentile(df['m'], [1, 99])
        else:
            lo, hi = np.percentile(sub['m'], [1, 99])
        m_grid[τ] = np.linspace(lo, hi, n_m)

    # --- 3) build nodes and a 1-NN snapper on the rectangular-in-index lattice ---
    nodes = np.vstack([[τ, m] for τ in tau_grid for m in m_grid[τ]])

    from sklearn.neighbors import NearestNeighbors
    nn = NearestNeighbors(n_neighbors=1).fit(nodes)

    return nn, nodes, tau_grid, m_grid


def apply_lattice(df, nn, nodes, tau_grid, m_grid, 
                  top_K=50, fill_method="linear"):
    """
    Given any df plus a prebuilt (nn, nodes, tau_grid, m_grid),
    snap-to-nodes, pick top_K by liquidity, pivot → C_sparse,
    drop never-observed, rebuild sub-nodes & sub-m_grid,
    then interpolate & fill → C_interp.
    """
    # snap each (τ,m) via nn
    pts = df[['tau','m']].values
    idx = nn.kneighbors(pts, return_distance=False)[:,0]
    df = df.assign(node_idx=idx,
                   lattice_tau=nodes[idx,0],
                   lattice_m=nodes[idx,1])
    
    # pick most liquid per timestamp/node
    best = (
      df.sort_values('stats_volume_usd', ascending=False)
        .drop_duplicates(['timestamp','node_idx'])
        # .assign(timestamp=pd.to_datetime(df['timestamp']))
    )

    best['timestamp'] = pd.to_datetime(best['timestamp'])
    
    # pivot to sparse C
    # top_nodes = df['node_idx'].value_counts().nlargest(top_K).index
    # sub       = best[best['node_idx'].isin(top_nodes)]
    top_nodes = best['node_idx'].value_counts().nlargest(top_K).index
    sub       = best[best['node_idx'].isin(top_nodes)]
    C_sparse  = sub.pivot_table('c_tilde','timestamp','node_idx')
    
    # drop never observed & rebuild sub-nodes, sub grids
    never_obs = C_sparse.columns[C_sparse.isna().all()]
    C_sparse  = C_sparse.drop(columns=never_obs)
    present   = C_sparse.columns.astype(int).to_numpy()
    nodes_sub = nodes[present]
    tau_sub   = np.unique(nodes_sub[:,0])
    m_sub     = {τ: sorted(nodes_sub[nodes_sub[:,0]==τ,1]) for τ in tau_sub}
    
    # interpolate & fill
    C_interp = (C_sparse
                  .interpolate(method=fill_method,axis=0)
                  .ffill().bfill())
    
    return C_interp, nodes_sub, tau_sub, m_sub
