import numpy as np
from sklearn.decomposition import PCA
from itertools import combinations
import cvxpy as cp

import numpy as np

def compute_surface_curl(C_interp, nodes, tau_grid, m_grid, gamma_sq=None, tol=1e-8):
    """
    Compute Z[t,j] = -∂_τ C - ½ γ² ∂_m C + ½ γ² ∂²_m C on your lattice.

    C_interp : pd.DataFrame, shape (T, N)
    nodes     : np.ndarray (N,2) of [tau_j, m_j]
    tau_grid  : array-like sorted unique taus
    m_grid    : dict { tau -> sorted list of m’s }
    gamma_sq  : optional array of length T; defaults to 1
    tol       : tolerance for matching floats

    Returns Z : np.ndarray, shape (T, N)
    """
    C = C_interp.values      # (T, N)
    T, N = C.shape
    if gamma_sq is None:
        gamma_sq = np.ones(T)

    tau_arr = np.asarray(tau_grid)

    # Precompute neighbor indices for each node j
    nbr = [{} for _ in range(N)]
    for j in range(N):
        τ_j, m_j = nodes[j]
        # find the index in tau_arr closest to τ_j
        iτ = int(np.argmin(np.abs(tau_arr - τ_j)))

        # τ-neighbors for central difference
        if 0 < iτ < len(tau_arr) - 1:
            τ_dn, τ_up = tau_arr[iτ - 1], tau_arr[iτ + 1]
            # find node at (τ_dn, m_j)
            candidates = np.where(
                (np.isclose(nodes[:,0], τ_dn, atol=tol)) &
                (np.isclose(nodes[:,1], m_j,   atol=tol))
            )[0]
            k_dn = candidates[0] if candidates.size else None
            # likewise for τ_up
            candidates = np.where(
                (np.isclose(nodes[:,0], τ_up, atol=tol)) &
                (np.isclose(nodes[:,1], m_j,   atol=tol))
            )[0]
            k_up = candidates[0] if candidates.size else None
        else:
            k_dn = k_up = None

        # m-neighbors at the same τ_j
        ms = m_grid[τ_j]
        # find index in ms closest to m_j
        im = int(np.argmin([abs(m - m_j) for m in ms]))
        if 0 < im < len(ms) - 1:
            m_l, m_r = ms[im - 1], ms[im + 1]
            # find node indices for those
            left = np.where(
                (np.isclose(nodes[:,0], τ_j, atol=tol)) &
                (np.isclose(nodes[:,1], m_l,  atol=tol))
            )[0]
            right = np.where(
                (np.isclose(nodes[:,0], τ_j, atol=tol)) &
                (np.isclose(nodes[:,1], m_r,  atol=tol))
            )[0]
            k_left = left[0] if left.size else None
            k_right= right[0] if right.size else None
        else:
            k_left = k_right = None

        nbr[j] = dict(tau_dn=k_dn, tau_up=k_up, m_left=k_left, m_right=k_right)

    # allocate output
    Z = np.zeros((T, N), dtype=float)

    # loop over time and nodes
    for t in range(T):
        C_t = C[t]
        g2  = gamma_sq[t]

        for j in range(N):
            nb = nbr[j]
            val = 0.0

            # ∂_τ
            if nb['tau_dn'] is not None and nb['tau_up'] is not None:
                dτ = nodes[nb['tau_up'],0] - nodes[nb['tau_dn'],0]
                val -= (C_t[nb['tau_up']] - C_t[nb['tau_dn']]) / dτ

            # ∂_m and ∂²_m
            if nb['m_left'] is not None and nb['m_right'] is not None:
                dm = nodes[nb['m_right'],1] - nodes[nb['m_left'],1]
                C_l, C_c, C_r = C_t[nb['m_left']], C_t[j], C_t[nb['m_right']]
                val += -0.5 * g2 * ( (C_r - C_l) / dm )
                val +=  0.5 * g2 * ( (C_r - 2*C_c + C_l) / (dm**2) )

            Z[t,j] = val

    return Z