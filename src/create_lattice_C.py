import pandas as pd
import numpy as np
import pyarrow.parquet as pq
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
import torch
import matplotlib.pyplot as plt

# Read in raw parquet
table = pq.read_table("/Users/167011/Documents/MQF/Thesis/Deribit_Data/deribit_options_2025-01-30_100k_rows.parquet")

# table = pq.read_table("/Users/arjunshah/Documents/UTS/Thesis/neural-sdes/data/deribit_options_2025-01-30_100k_rows.parquet")


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
tau_kmeans = KMeans(n_clusters=7, random_state=0).fit(unique_taus)
# Get the cluster centers and sort them
tau_grid = np.sort(tau_kmeans.cluster_centers_.flatten())
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

# Pivot to get a sparse lattice DataFrame (NaNs left in place)
C_sparse = best.pivot(
    index = 'timestamp',
    columns = 'node_idx',
    values = 'mid_price'
)

# Mask zeros as NaN so interpolation will fill them
C_sparse = C_sparse.replace(0, np.nan)

# Interpolate in time, then forward- and back-fill any edge NaNs
C_interp = (
    C_sparse
    .interpolate(method='linear', axis=0)  # linear interpolation along timestamps
    .ffill()                              # carry last valid forward
    .bfill()                              # fill any leading NaNs at the start
)

# Very no zeros remain 
print(C_interp.isna().sum().sum()) # should be 0

# Create vector of underlying prices, log price and log return

# keep underlying_info as a DataFrame with the price column
underlying_info = (
    df
    .drop_duplicates('timestamp')
    .sort_values('timestamp')
    .reset_index(drop=True)
    [['underlying_price']]           # <-- still a DataFrame
)

underlying_info['log_return'] = np.log(
    underlying_info['underlying_price']
    / underlying_info['underlying_price'].shift(1)
)
underlying_info['log_price'] = np.log(
    underlying_info['underlying_price']
)



surf_arr = C_interp.to_numpy()              # or C.values
# print(surf_arr)
surf_tensor = torch.from_numpy(surf_arr).float()
# print(surf_tensor.shape[0])

# torch.save(surf_tensor, "btc_surfaces.pt")

# 1) compute low/high m for each τ
taus = tau_grid
m_lo = np.array([m_grid[t].min() for t in taus])
m_hi = np.array([m_grid[t].max() for t in taus])

# 2) build polygon coordinates in (m, τ)-space
poly_m = np.concatenate([m_lo, m_hi[::-1]])
poly_tau = np.concatenate([taus, taus[::-1]])

# 3) collect all lattice nodes
nodes = np.vstack([
    np.column_stack([m_grid[t], np.full_like(m_grid[t], t)])
    for t in taus
])

# 4) plot
plt.figure(figsize=(6,4))
plt.fill(poly_m, poly_tau, color='lightblue', alpha=0.5, label=r'$\mathcal{R}_{\rm liq}$')
plt.scatter(nodes[:,0], nodes[:,1], color='k', s=30, label=r'$\mathcal{L}_{\rm liq}$')

plt.xlabel(r'$m = \ln(K/S)$')
plt.ylabel(r'$\tau$ (years)')
plt.title('Liquid Range and Lattice Nodes')
plt.legend(loc='upper right')
plt.grid(False)
plt.tight_layout()
plt.show()
