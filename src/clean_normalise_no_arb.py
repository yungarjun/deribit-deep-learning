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

# Extract the underlying price 
unique_times = df.drop_duplicates(subset='timestamp').reset_index(drop=True)
underlying = unique_times['index_price'].values

# Save the underlying price as a tensor
underlying_tensor = torch.tensor(underlying, dtype=torch.float32)
torch.save(underlying_tensor, "underlying_price.pt")


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


def build_noarb_constraints(nodes, tau_grid, m_grid):
    """
    nodes:     [n_nodes×2] array of (tau, m)
    tau_grid:  sorted list/array of taus
    m_grid:    dict { tau -> sorted array of m's at that tau }
    Returns A, b for A c >= b
    """
    n = nodes.shape[0]
    rows, cols, data, b = [], [], [], []
    cons_idx = 0

    # 1) monotone in tau
    for i in range(len(tau_grid)-1):
        τ0, τ1 = tau_grid[i], tau_grid[i+1]
        # only m’s that actually appear at both τ0 and τ1
        common_ms = sorted(set(m_grid[τ0]) & set(m_grid[τ1]))
        for m in common_ms:
            mask0 = np.isclose(nodes[:,0], τ0) & np.isclose(nodes[:,1], m)
            mask1 = np.isclose(nodes[:,0], τ1) & np.isclose(nodes[:,1], m)
            k1 = np.where(mask0)[0][0]
            k2 = np.where(mask1)[0][0]
            rows += [cons_idx, cons_idx]
            cols += [k2, k1]
            data += [ 1.0, -1.0]
            b.append(0.0)
            cons_idx += 1

    # 2) monotone in m
    for τ in tau_grid:
        ms = sorted(m_grid[τ])
        for j in range(len(ms)-1):
            m0, m1 = ms[j], ms[j+1]
            mask0 = np.isclose(nodes[:,0], τ) & np.isclose(nodes[:,1], m0)
            mask1 = np.isclose(nodes[:,0], τ) & np.isclose(nodes[:,1], m1)
            k0 = np.where(mask0)[0][0]
            k1 = np.where(mask1)[0][0]
            # C(m1) - C(m0) <= 0  ->  -1 at m1, +1 at m0
            rows += [cons_idx, cons_idx]
            cols += [k1, k0]
            data += [-1.0, 1.0]
            b.append(0.0)
            cons_idx += 1

    # 3) convexity in m
    for τ in tau_grid:
        ms = sorted(m_grid[τ])
        for j in range(1, len(ms)-1):
            m_prev, m_mid, m_next = ms[j-1], ms[j], ms[j+1]
            mask_p = np.isclose(nodes[:,0], τ) & np.isclose(nodes[:,1], m_prev)
            mask_m = np.isclose(nodes[:,0], τ) & np.isclose(nodes[:,1], m_mid)
            mask_n = np.isclose(nodes[:,0], τ) & np.isclose(nodes[:,1], m_next)
            k_prev = np.where(mask_p)[0][0]
            k_mid  = np.where(mask_m)[0][0]
            k_next = np.where(mask_n)[0][0]
            # C_prev - 2 C_mid + C_next >= 0
            rows += [cons_idx]*3
            cols += [k_prev, k_mid, k_next]
            data += [1.0, -2.0, 1.0]
            b.append(0.0)
            cons_idx += 1

    A = sp.csr_matrix((data, (rows, cols)), shape=(cons_idx, n))
    b = np.array(b, dtype=float)
    return A, b

A, b = build_noarb_constraints(nodes, tau_grid, m_grid)



# ensure it has every node-column 0…n_nodes-1
n_nodes = nodes.shape[0]
# C_interp = C_interp.reindex(columns=range(n_nodes))
total_nans = C_interp.isna().sum().sum()
print(f"Total NaNs in C_interp: {total_nans:,}")

# Which node-indices never got any data at all?
all_nan_cols = C_interp.columns[C_interp.isna().all()]
print("Nodes never observed:", list(all_nan_cols))

# --- 3) Loop over each timestamp, project onto the no‐arb polytope ---
C_arb = []
for t, row in C_interp.iterrows():
    c_raw = row.values.astype(float)           # length n_nodes, no NaNs

    # define and solve the QP:  minimize ‖c - c_raw‖²  s.t. A c ≥ b,  c ≥ 0
    c = cp.Variable(n_nodes)
    obj  = cp.Minimize(cp.sum_squares(c - c_raw))
    cons = [A @ c >= b, c >= 0]
    prob = cp.Problem(obj, cons)

    # try OSQP first (install via `pip install osqp` if needed)
    try:
        prob.solve(solver=cp.OSQP, verbose=False)
    except Exception:
        prob.solve(solver=cp.SCS, verbose=False, eps_abs=1e-5, eps_rel=1e-5)

    if prob.status in ["optimal", "optimal_inaccurate"]:
        C_arb.append(c.value)
    else:
        # fall back to raw if solver fails
        C_arb.append(c_raw)

# Stack into array and build DataFrame
C_arb = np.vstack(C_arb)                      # shape [T, n_nodes]
C_arb_df = pd.DataFrame(
    C_arb,
    index   = C_interp.index,
    columns = C_interp.columns
)

surf_tensor_af = torch.from_numpy(C_arb_df.values).float()
torch.save(surf_tensor_af, "surf_tensor_af.pt")

# We now extract the timesteps between observed option snapshot
timestamps = C_interp.index.to_series()

# Compute raw delta t in seconds (or fractions of a second) between each row
dt_secs = timestamps.diff().dt.total_seconds().to_numpy()

# the first value for dt is na so we just assume that its the same as the next value
dt_secs[0] = dt_secs[1]

dt_secs = dt_secs.astype(np.float32)

# Convert to tensor
dt_tensor = torch.from_numpy(dt_secs).float()

# Save
torch.save(dt_tensor, "dt.pt")




 # ---------------------------------------------------------------------------------- 
 # Checking for violations of the no-arbitrage conditions
 # ----------------------------------------------------------------------------------
# build an array of the global node‐indices that are actually present
present_idx   = C_arb_df.columns.to_numpy().astype(int)     # e.g. [0,1,3,4,...,49] but not 12
present_nodes = nodes_full[present_idx]                    # shape (n_present,2)

violations = {"tau<0":0, "dm>0":0, "d2m<0":0}
t0 = C_arb_df.index[0]   # just check at the first timestamp

# 1) monotonicity in m & convexity in m, _per_ τ
for τ in np.unique(present_nodes[:,0]):
    # find which present columns lie at this τ
    mask        = np.isclose(present_nodes[:,0], τ)
    cols_global = present_idx[mask]          # global column indices
    ms          = present_nodes[mask,1]
    order       = np.argsort(ms)
    cols        = cols_global[order]         # sorted by m

    prices = C_arb_df.loc[t0, cols].values   # length = len(cols)

    # ∂C/∂m ≤ 0  →  price[j+1] - price[j] ≤ 0
    diffs = np.diff(prices)
    violations["dm>0"] += np.sum(diffs > 1e-8)

    # ∂²C/∂m² ≥ 0  →  price[j-1] - 2 price[j] + price[j+1] ≥ 0
    d2 = prices[:-2] - 2*prices[1:-1] + prices[2:]
    violations["d2m<0"] += np.sum(d2 < -1e-8)

# 2) monotonicity in τ, _per_ m
for m in np.unique(present_nodes[:,1]):
    mask        = np.isclose(present_nodes[:,1], m)
    cols_global = present_idx[mask]
    taus        = present_nodes[mask,0]
    order       = np.argsort(taus)
    cols        = cols_global[order]

    prices = C_arb_df.loc[t0, cols].values

    # ∂C/∂τ ≥ 0  →  price[j+1] - price[j] ≥ 0
    diffs = np.diff(prices)
    violations["tau<0"] += np.sum(diffs < -1e-8)

print("Violations:", violations)

# import matplotlib.pyplot as plt
# import matplotlib.tri as mtri
# from mpl_toolkits.mplot3d import Axes3D  # noqa

# # 1) grab your repaired DataFrame and the full lattice
# col_idx       = C_arb_df.columns.to_numpy().astype(int)
# nodes_present = nodes_full[col_idx]   # shape (49,2) if you dropped one node

# taus_p = nodes_present[:,0]
# ms_p   = nodes_present[:,1]

# # 2) values at t₀
# t0_idx = 0
# y_obs  = C_arb_df.iloc[t0_idx].values

# # 3) triangulate & plot
# tri = mtri.Triangulation(taus_p, ms_p)
# fig = plt.figure(figsize=(8,6))
# ax  = fig.add_subplot(111, projection='3d')
# ax.plot_trisurf(tri, y_obs, linewidth=0.2, antialiased=True)
# ax.set_xlabel('τ (yrs)')
# ax.set_ylabel('m = log(K/F)')
# ax.set_zlabel('Call Price (arb‐free)')
# ax.set_title(f'Arb‐free Surface at t₀ (idx={t0_idx})')
# plt.tight_layout()
# plt.show()