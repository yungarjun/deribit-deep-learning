import pandas as pd
import numpy as np
import os
import sys
import pyarrow.parquet as pq

def build_surface():
#   C
#       A NumPy array of shape (T,  N)(T,N), where

#     T = number of distinct timestamps you observed, and

#     N = number of (τi,mj)(τi​,mj​) cells in your fixed grid (here 55 expiries × 99 moneyness bins = 45 columns).
#     Each entry C[t,n]C[t,n] is the (mean-aggregated) mid-price at time tt in grid cell nn.

#   grid
#       A pandas MultiIndex of length N with two levels:

#       time_to_maturity τi​

#       moneyness_clean mj​
#       So grid[n] == (τ_i, m_j) tells you exactly which expiry and which moneyness bucket corresponds to column nn of C.

# times
#       A 1-d array of length TT (dtype datetime64) giving the exact timestamps in chronological order.
#       In other words, times[t] is the date-time for row tt of C."
    ""
    # Read in raw parquet
    table = pq.read_table('data/raw/deribit_options_2025-01-30_100k_rows.parquet')

    # Convert to pandas DataFrame
    df = table.to_pandas()

    # Seperate Option Type, Strike and Maturity
    df[['asset', 'expiry', 'strike', 'option_type']] = df['instrument_name'].str.split('-', expand=True)

    # Define maturity in years
    df['expiry'] = pd.to_datetime(df['expiry'])
    reference_date = pd.to_datetime("2025-01-30")
    df['time_to_maturity'] = (df['expiry'] - reference_date).dt.days / 365.25 # Crypto is traded 24/7

    # Filter for just calls and open_interest > 0 
    df = df[(df['option_type'] == 'C') & (df['open_interest'] > 0)]




    # Convert strike to numeric
    df['strike'] = pd.to_numeric(df['strike'], errors = 'coerce')

    # Define moneyness
    df['moneyness_raw'] = np.log(df['strike'] / df['underlying_price'])

    # pick top n most liquid expiries
    top_n = 5
    top_taus = df['time_to_maturity'].value_counts().nlargest(top_n).index.tolist()

    # Define number of moneyness bins
    n_bins = 9

    # compute quantile-based moneyness grid for each expiry
    tau_bin_centers = {}
    for tau in top_taus:
        m = df.loc[df['time_to_maturity'] == tau, 'moneyness_raw']
        edges = np.quantile(m, np.linspace(0, 1, n_bins + 1))
        centers = (edges[:-1] + edges[1:]) / 2
        tau_bin_centers[tau] = centers

    # Assign moneyness bin to each option
    df['moneyness_clean'] = np.nan
    for tau, centers in tau_bin_centers.items():
        mask = df['time_to_maturity'] == tau
        m_vals = df.loc[mask, 'moneyness_raw'].to_numpy()

        # find index of nearest center for each m_val
        idx = np.abs(m_vals[:, None] - centers).argmin(axis=1)
        df.loc[mask, 'moneyness_clean'] = centers[idx]

    # Define mid_price
    df['mid_price'] = (df['best_bid_price'] + df['best_ask_price']) / 2


    # Create surface 
    surface = df.pivot_table(
        index = 'timestamp',
        columns = ['time_to_maturity', 'moneyness_clean'],
        values = 'mid_price',
        aggfunc = 'mean'
    )

    # fill holes by interpolation or ffill/backfill
    surface = surface.interpolate(method='linear', axis=1).ffill().bfill()

    # Return surface.values
    C = surface.values
    grid = surface.columns
    times = surface.index.to_numpy()

    return C, grid, times

    