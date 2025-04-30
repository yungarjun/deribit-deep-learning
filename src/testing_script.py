import pandas as pd
import numpy as np
import pyarrow.parquet as pq
from sklearn.cluster import KMeans

# Read in raw parquet
table = pq.read_table('data/raw/deribit_options_2025-01-30_100k_rows.parquet')

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


# Convert strike to numeric
df['strike'] = pd.to_numeric(df['strike'], errors = 'coerce')

# Define log moneyness, 'm'
df['m'] = np.log(df['strike'] / df['underlying_price'])

# Choose tau grid based on Kmeans clustering
taus = df['tau'].values.reshape(-1, 1)

kmeans = KMeans(n_clusters = 7).fit(taus)
df['tau_cluster'] = kmeans.labels_

tau_grid = np.sort(kmeans.cluster_centers_.flatten())

m_grid = {}
for cluster_label, tau in enumerate(tau_grid):
    # Select all rows in that clust
    subset = df[df['tau_cluster'] == cluster_label]
    # Get min and max m for that cluster
    m_min, m_max = subset['m'].min(), subset['m'].max()
    # Create equally spaced m grid
    m_grid[tau] = np.linspace(m_min, m_max, 10)

print(tau_grid)
for τi in tau_grid:
    print(f"τ={τi:.4f} → m_grid={m_grid[τi]}")