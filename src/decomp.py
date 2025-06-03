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
# table = pq.read_table("/Users/167011/Documents/MQF/Thesis/Deribit_Data/deribit_options_2025-01-30_100k_rows.parquet")

table = pq.read_table("/Users/arjunshah/Documents/UTS/Thesis/neural-sdes/data/deribit_options_2025-01-30_100k_rows.parquet")


# Convert to pandas DataFrame
df = table.to_pandas()

# Seperate Option Type, Strike and Maturity
df[['asset', 'expiry', 'strike', 'option_type']] = df['instrument_name'].str.split('-', expand=True)

# Define maturity in years 'tau'
df['expiry'] = pd.to_datetime(df['expiry'])
reference_date = pd.to_datetime("2025-01-30")
df['tau'] = (df['expiry'] - reference_date).dt.days / 365.25 # Crypto is traded 24/7

# Filter for just calls and open_interest > 0 
df = df[(df['option_type'] == 'C') & (df['open_interest'] > 0)]

# Define mid price
df['mid_price'] = (df['best_bid_price'] + df['best_ask_price']) / 2

# Convert strike to numeric
df['strike'] = pd.to_numeric(df['strike'], errors = 'coerce')

# Define log moneyness, 'm'
df['m'] = np.log(df['strike'] / df['underlying_price'])

# # Choose tau grid based on Kmeans clustering

unique_taus = np.sort(df['tau'].unique()).reshape(-1,1)
# Finding different clusters of tau  (time to expiry)
tau_kmeans = KMeans(n_clusters=5, random_state=0).fit(unique_taus)
# Get the cluster centers and sort them
tau_grid = (tau_kmeans.cluster_centers_.flatten())
# Assign each quote to a cluster
df['tau_cluster'] = tau_kmeans.predict(df[['tau']])

# Moneyness grid for each tau (ttm)
m_grid = {}
for cluster_label, tau in enumerate(tau_grid):
    # Select all rows in that clust
    subset = df[df['tau_cluster'] == cluster_label]
    # Get min and max m for that cluster
    m_lo, m_hi = np.percentile(subset['m'], [1, 99])
    m_grid[tau] = np.linspace(m_lo, m_hi, 10) # here we are assuming each moneyness point is equally likely

print(tau_grid)

for τi in tau_grid:
    print(f"τ={τi:.4f} → m_grid={m_grid[τi]}")


# Building the list of all (tau_i, m_ij) nodes
nodes = []
for tau in tau_grid:
    for m_j in m_grid[tau]:
        nodes.append((tau, m_j))
nodes = np.array(nodes) # Shape (N_nodes, 2)

# Fit to nearest neighbor model on those nodes
nn = NearestNeighbors(n_neighbors=1, algorithm='auto').fit(nodes)

# Query eachquote's (tau, m) to find the nearest node
points = df[['tau', 'm']].values # Shape (N_quotes, 2)
distances, indices = nn.kneighbors(points)



# Store back into dataframe
df['node_idx'] = indices[:, 0] # which row of 'nodes' it snapped too
df['lattice_tau']   = nodes[df['node_idx'], 0] # which tau it snapped too
df['lattice_m']     = nodes[df['node_idx'], 1] # which m it snapped too

# Within each time-slice, pick the most liquid quote per node:
best = (
    df
    .sort_values('open_interest', ascending = False)
    .drop_duplicates(subset = ['timestamp', 'node_idx'])
    .reset_index(drop = True)
)

best = best[['timestamp', 'node_idx', 'lattice_tau', 'lattice_m', 'mid_price']]

best['timestamp'] = pd.to_datetime(best['timestamp'])

# Pivot to get a sparse lattice DataFrame (NaNs left in place)
counts = df['node_idx'].value_counts()
K = 50
top_nodes = counts.nlargest(K).index
df_sub    = df[df['node_idx'].isin(top_nodes)]
C_sparse = (
    df_sub
      .sort_values('open_interest', ascending=False)
      .drop_duplicates(['timestamp','node_idx'])
      .pivot_table(
         'mid_price',        # values
         'timestamp',        # index
         'node_idx',         # columns
         aggfunc='first'
      )
)

nodes = np.array(nodes)  # shape (50,2)
m_grid = m_grid          # dict mapping each tau → 10 m-values

# **NEW**: remember the full lattice
nodes_full  = nodes.copy()
m_grid_full = {τ: m_grid[τ].copy() for τ in m_grid}
# 0) Drop any node-columns that were never observed
never_obs = C_sparse.columns[C_sparse.isna().all()].astype(int)
if len(never_obs) > 0:
    print(f"Dropping never-observed nodes: {list(never_obs)}")
    C_sparse = C_sparse.drop(columns=never_obs)

# 1) Rebuild `nodes`, `tau_grid`, `m_grid` to match the surviving columns
#    (you must have kept your original `nodes_full` array and `m_grid_full` dict)
present_idx = C_sparse.columns.to_numpy().astype(int)

nodes = nodes_full[present_idx]   # keep only those node coordinates
tau_grid = np.unique(nodes[:,0])

m_grid = {
    τ: sorted(nodes[nodes[:,0]==τ, 1].tolist())
    for τ in tau_grid
}

# 2) Time-interpolate and fill
C_interp = (
    C_sparse
      .interpolate(method='linear', axis=0)
      .ffill()
      .bfill()
)


# ---------------------------------------------------------------------------------- 
# Now we decompose 
# ----------------------------------------------------------------------------------


from sklearn.decomposition import PCA
import torch
from clean_normalise_no_arb import build_noarb_constraints
# Load in saved tensors
surf_tensor = torch.load('surf_tensor_af.pt').numpy()
dt_tensor = torch.load('dt.pt').numpy()

# Center each nodes price time series
G_0 = surf_tensor.mean(axis=0)

# Subtract G0 from every row to form R (zero mean per node)
R = surf_tensor - G_0[None, :]

# Perform PCA on the centered data
d = 3
pca = PCA(n_components=d)
Xi = pca.fit_transform(R)

G = pca.components_.T  # Principal components

# --- Saving each of the things as a tensor ---

# 1) PCA scores
Xi_tensor = torch.from_numpy(Xi).float()        # shape (T+1, d)

# 2) (Optional) basis vectors, if you plan to reconstruct surfaces later
G_tensor  = torch.from_numpy(G).float()         # shape (n_nodes, d)

# 3) (Optional) the mean vector G0, if you need it for reconstruction
G0_tensor = torch.from_numpy(G_0).float()        # shape (n_nodes,)

torch.save(Xi_tensor, 'Xi_tensor.pt')
torch.save(G_tensor, 'G_tensor.pt')
torch.save(G0_tensor, 'G0_tensor.pt')


# A, b = build_noarb_constraints(nodes, tau_grid, m_grid)


# # --- Step 1: Reconstruct PCA surface at each t ---
# C_recon = Xi.dot(G.T) + G_0[None, :]    # shape (T, n_nodes)

# # --- Step 2: Check static‐arbitrage violations ---
# import numpy as np
# import scipy.sparse as sp

# tol = 1e-8
# E = A.dot(C_recon.T)                   # shape (n_cons, T)
# E_minus_b = E - b[:, None]             # shape (n_cons, T)
# violations = (E_minus_b < -tol)        # boolean array, True where A c_t < b

# # Count how many snapshots have ANY violation
# any_violation_per_t = np.any(violations, axis=0)  # shape (T,)
# PSAS = any_violation_per_t.mean()

# print(f"PSAS = {PSAS:.4%}   (i.e. {any_violation_per_t.sum()} out of {C_recon.shape[0]} violate)")

# for d in [3, 10, 15, 20, 25, 30, 35, 40, 45, 49]:
#     pca = PCA(n_components=d)
#     Xi  = pca.fit_transform(R)
#     G   = pca.components_.T
#     C_recon = Xi.dot(G.T) + G_0[None,:]
#     E = A.dot(C_recon.T)
#     PSAS_d = np.any(E - b[:,None] < -1e-8, axis=0).mean()
#     print(f"d={d:2d} → PSAS = {PSAS_d:.3%}")
#     slacks =  A.dot(C_recon.T) - b[:, None]        # shape (n_cons, T)
#     min_slack_per_t = slacks.min(axis=0)  # length T

#     print("Worst‐case (most negative) slacks for first few t:", min_slack_per_t[:10])
#     print("Overall min slack:", min_slack_per_t.min())   # how far below 0
#     print("Overall max slack:", min_slack_per_t.max())   # typically > 0
#     print("Mean slack:", np.mean(min_slack_per_t))