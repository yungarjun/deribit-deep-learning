import pandas as pd
import numpy as np
import pyarrow.parquet as pq
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
import torch
import matplotlib.pyplot as plt
import scipy.sparse as sp
import cvxpy as cp
# from .create_lattice import clean_deribit, read_parquet



# The static arbitrage constraints
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



def projection(C_interp, nodes, tau_grid, m_grid):


    # C_interp, nodes, tau_grid, m_grid = build_lattice(df)

    A, b = build_noarb_constraints(nodes, tau_grid, m_grid)
    C_arb = []
    n_nodes = nodes.shape[0]
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

    return C_arb_df
    

def projection_fast_cvxpy(C_interp, nodes, tau_grid, m_grid):
    # 1) Build A, b once
    A, b = build_noarb_constraints(nodes, tau_grid, m_grid)
    n = nodes.shape[0]

    # 2) Create a Parameter for the raw vector
    c_raw = cp.Parameter(n)

    # 3) Define the problem once
    c_var = cp.Variable(n)
    objective = cp.Minimize(cp.sum_squares(c_var - c_raw))
    constraints = [A @ c_var >= b, c_var >= 0]
    prob = cp.Problem(objective, constraints)

    # 4) Solve repeatedly with warm_start
    C_arb = []
    for row in C_interp.values:
        c_raw.value = row
        prob.solve(
            solver=cp.OSQP,
            warm_start=True,
            eps_abs=1e-6,
            eps_rel=1e-6,
            verbose=False
        )
        C_arb.append(c_var.value)

    C_arb = np.vstack(C_arb)
    return pd.DataFrame(C_arb, index=C_interp.index, columns=C_interp.columns)
