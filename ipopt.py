#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import time
import math
import numpy as np
import pandas as pd
from numba import njit, prange
from numba.typed import List as NumbaList
import cyipopt

# ---------------------------- USER SETTINGS ----------------------------
MEAN_DEGREE = 25.0
BAND        = 0.10      # modulus band ±10% for seller constraints
SKIP_SELF   = True
UNDIRECTED  = False     # keep directed

# bounds in TeX (honored via sigmoid boxes)
ALPHA_RANGE = (0.30, 0.95)
KAPPA_RANGE = (0.30, 0.90)

# raking warm-start
RAKE_INNER_STEPS = 2
RAKE_OUTER_ALTS  = 6
ETA_LAMBDA       = 0.5
CLIP_LOG_RATIO   = 3.0

# IPOPT options
IPOPT_OPTIONS = {
    "tol": 1e-7,
    "constr_viol_tol": 1e-7,
    "compl_inf_tol": 1e-7,
    "acceptable_tol": 1e-5,
    "acceptable_constr_viol_tol": 1e-5,
    "max_iter": 2000,
    "mu_strategy": "adaptive",
    "mu_init": 1e-2,
    "hessian_approximation": "limited-memory",
    "nlp_scaling_method": "gradient-based",
    "linear_solver": "ma97",
    "print_level": 5,
}

# tiny floors
EPS_S = 1e-12
EPS_SIJ = 1e-15
LAMBDA_FLOOR = 1e-15

# deterministic RNG for Bernoulli draws
SEED64 = np.uint64(0x1234D00DCAFEBEEF)

os.environ.setdefault("NUMBA_NUM_THREADS", "16")


def log(msg):
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}")


# ---------------------------- DATA LOADER -----------------------------
def load_data():
    root = "input_files"

    files = {
        "firm": os.path.join(root, "FIRM_STRENGTHSM.xlsx"),
        "map":  os.path.join(root, "MAPPING_SM.xlsx"),
        "iot":  os.path.join(root, "IOT!.xlsx"),
        "sij":  os.path.join(root, "Sij!.xlsx"),
    }
    for _, p in files.items():
        if not os.path.exists(p):
            raise FileNotFoundError(f"Missing required file: {p}")

    firm_df = pd.read_excel(files["firm"])
    map_df  = pd.read_excel(files["map"])
    iot_df  = pd.read_excel(files["iot"], header=0, index_col=0)
    sij_df  = pd.read_excel(files["sij"], header=0, index_col=0)

    # clean
    firm_df.columns = firm_df.columns.str.strip()
    map_df.columns  = map_df.columns.str.strip()
    iot_df.index    = iot_df.index.astype(str).str.strip()
    iot_df.columns  = iot_df.columns.astype(str).str.strip()
    sij_df.index    = sij_df.index.astype(str).str.strip()
    sij_df.columns  = sij_df.columns.astype(str).str.strip()
    map_df["SIC"]   = map_df["SIC"].astype(str).str.strip()

    # align sectors
    common = np.intersect1d(map_df["SIC"].unique(), iot_df.index)
    common = np.intersect1d(common, sij_df.index)
    if common.size == 0:
        raise RuntimeError("No common sector codes across mapping/IOT/Sij.")

    # restrict
    iot_df = iot_df.loc[common, common].copy()
    sij_df = sij_df.loc[common, common].copy()
    map_df = map_df[map_df["SIC"].isin(common)].copy()

    # S in [0,1], same support with IOT (zeros coincide)
    S = sij_df.to_numpy(dtype=np.float64)
    S = np.clip(S, 0.0, 1.0)
    Iraw = iot_df.to_numpy(dtype=np.float64)
    Iraw = np.maximum(Iraw, 0.0)
    mask = (S > EPS_SIJ) & (Iraw > EPS_SIJ)
    S = np.where(mask, S, 0.0)
    I = np.where(mask, Iraw, 0.0)

    # column-normalize I (seller shares)
    colsum = I.sum(axis=0, keepdims=True)
    colsum = np.where(colsum == 0.0, 1.0, colsum)
    I = I / colsum

    # map firms to sectors
    sic_to_idx = {sic: i for i, sic in enumerate(common)}
    map_df["SIC_index"] = map_df["SIC"].map(sic_to_idx)
    firm_df = firm_df.set_index("Firm_ID")
    map_df  = map_df.set_index("Firm_ID")
    firm_df = firm_df.join(map_df[["SIC_index"]], how="inner").reset_index()

    firm_ids = firm_df["Firm_ID"].to_numpy()
    q_raw    = firm_df["Size"].astype(float).to_numpy()
    n        = len(firm_ids)
    f2s      = firm_df["SIC_index"].astype(np.int64).to_numpy()
    Sct      = len(common)

    # q in (0,1]
    q = q_raw / max(np.max(q_raw), 1.0)
    q = np.maximum(q, 1e-15)
    log_q = np.log(q)

    # sector->firms lists
    stf = [[] for _ in range(Sct)]
    for i in range(n):
        stf[f2s[i]].append(i)
    stf_numba = NumbaList([np.ascontiguousarray(np.array(idx, dtype=np.int64)) for idx in stf])

    # nonzero blocks by S support
    m_idx = []
    s_idx = []
    S_ms  = []
    I_ms  = []
    for k in range(Sct):
        for l in range(Sct):
            if S[k, l] > EPS_SIJ:
                m_idx.append(k)
                s_idx.append(l)
                S_ms.append(S[k, l])
                I_ms.append(I[k, l])
    if not m_idx:
        raise RuntimeError("No nonzero (k,l) blocks.")

    m_idx = np.ascontiguousarray(np.array(m_idx, dtype=np.int64))
    s_idx = np.ascontiguousarray(np.array(s_idx, dtype=np.int64))
    S_ms  = np.ascontiguousarray(np.array(S_ms, dtype=np.float64))
    I_ms  = np.ascontiguousarray(np.array(I_ms, dtype=np.float64))
    K     = len(m_idx)

    # blocks_by_l
    blocks_by_l = [np.where(s_idx == l)[0].astype(np.int64) for l in range(Sct)]

    # seller sizes s_l
    s_l = np.zeros(Sct, dtype=np.float64)
    for j in range(n):
        s_l[f2s[j]] += q[j]

    # precompute logS per block
    logS_ms = np.log(np.maximum(S_ms, EPS_SIJ))

    info = dict(
        sic_labels=list(common),
        n=n,
        num_sectors=Sct,
        K=K,
        S_ms=S_ms,
        logS_ms=logS_ms,
        I_ms=I_ms,
        m_idx=m_idx,
        s_idx=s_idx,
        q=q,
        log_q=log_q,
        firm_to_sector=f2s,
        stf=stf_numba,
        blocks_by_l=blocks_by_l,
        seller_sizes=s_l,
    )
    return info


# ----------------------- NUMBA KERNELS (BLOCK LOOP) -------------------
@njit(parallel=True, fastmath=True, cache=True)
def block_sums_and_sens(
    z,
    alpha,
    kappa,
    lambdas,   # scalars/array (lambdas length K)
    S_pow,
    logS_ms,   # per-block S_ms^kappa and log(S_ms)
    m_idx,
    s_idx,
    stf,       # block indices + sector->firms
    q,
    log_q,     # sizes & logs
    I_ms,      # I shares per-block
    skip_self,
    undirected,
):
    K = m_idx.shape[0]
    P      = np.zeros(K, dtype=np.float64)
    gP_z   = np.zeros(K, dtype=np.float64)
    gP_a   = np.zeros(K, dtype=np.float64)
    gP_k   = np.zeros(K, dtype=np.float64)
    gP_ll  = np.zeros(K, dtype=np.float64)

    Eblk   = np.zeros(K, dtype=np.float64)
    gE_z   = np.zeros(K, dtype=np.float64)
    gE_a   = np.zeros(K, dtype=np.float64)
    gE_k   = np.zeros(K, dtype=np.float64)
    gE_ll  = np.zeros(K, dtype=np.float64)

    # precompute q^alpha
    q_pow_a = np.exp(alpha * log_q)

    for b in prange(K):
        k = m_idx[b]
        l = s_idx[b]
        i_list = stf[k]
        j_list = stf[l]
        if i_list.shape[0] == 0 or j_list.shape[0] == 0:
            continue

        lam   = lambdas[b]
        s_pow = S_pow[b]
        Ikl   = I_ms[b]
        logS  = logS_ms[b]

        accP = 0.0
        acc_gP_z = 0.0
        acc_gP_a = 0.0
        acc_gP_k = 0.0
        acc_gP_ll = 0.0

        accE = 0.0
        acc_gE_z = 0.0
        acc_gE_a = 0.0
        acc_gE_k = 0.0
        acc_gE_ll = 0.0

        for ii in range(i_list.shape[0]):
            i = i_list[ii]
            qi_a = q_pow_a[i]
            qi   = q[i]
            lgqi = log_q[i]
            for jj in range(j_list.shape[0]):
                j = j_list[jj]
                if skip_self and i == j:
                    continue
                if undirected and j <= i:
                    continue
                x = z * lam * s_pow * qi_a * q_pow_a[j]

                if x > 1e6:
                    p = 1.0
                    w = 0.0
                else:
                    p = x / (1.0 + x)
                    w = p * (1.0 - p)

                lgqq = lgqi + log_q[j]

                accP      += p
                acc_gP_z  += w
                acc_gP_a  += w * lgqq
                acc_gP_k  += w * logS
                acc_gP_ll += w

                val       = Ikl * qi * p
                accE      += val
                wE        = Ikl * qi * w
                acc_gE_z  += wE
                acc_gE_a  += wE * lgqq
                acc_gE_k  += wE * logS
                acc_gE_ll += wE

        P[b]     = accP
        gP_z[b]  = acc_gP_z
        gP_a[b]  = acc_gP_a
        gP_k[b]  = acc_gP_k
        gP_ll[b] = acc_gP_ll

        Eblk[b]  = accE
        gE_z[b]  = acc_gE_z
        gE_a[b]  = acc_gE_a
        gE_k[b]  = acc_gE_k
        gE_ll[b] = acc_gE_ll

    return P, gP_z, gP_a, gP_k, gP_ll, Eblk, gE_z, gE_a, gE_k, gE_ll


# ----------------------- HELPER: SIGMOID BOX MAPS ---------------------
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def inv_sigmoid(y):
    return np.log(y / (1.0 - y))


def unpack_vars(u):
    # u = [zeta, a, c, ell_0..ell_{K-1}]
    zeta, a, c = u[0], u[1], u[2]
    z = np.exp(zeta)
    a_lo, a_hi = ALPHA_RANGE
    c_lo, c_hi = KAPPA_RANGE
    alpha = a_lo + (a_hi - a_lo) * sigmoid(a)
    kappa = c_lo + (c_hi - c_lo) * sigmoid(c)
    return z, alpha, kappa


def unpack_all(u, K):
    z, alpha, kappa = unpack_vars(u)
    ells = u[3:]
    if ells.shape[0] != K:
        raise RuntimeError("Bad u length.")
    lambdas = np.exp(ells)  # >0
    return z, alpha, kappa, lambdas


# ---------------------- WARM START (RAKE + BISECT) --------------------
def total_edges_fast(z, alpha, kappa, lambdas, ctx):
    S_pow = np.power(np.maximum(ctx["S_ms"], EPS_SIJ), kappa)
    P, *_ = block_sums_and_sens(
        z,
        alpha,
        kappa,
        lambdas,
        S_pow,
        ctx["logS_ms"],
        ctx["m_idx"],
        ctx["s_idx"],
        ctx["stf"],
        ctx["q"],
        ctx["log_q"],
        ctx["I_ms"],
        SKIP_SELF,
        UNDIRECTED,
    )
    return float(P.sum())


def bisect_z_for_degree(alpha, kappa, lambdas, ctx, z_guess=1.0):
    target = ctx["n"] * MEAN_DEGREE
    S_pow = np.power(np.maximum(ctx["S_ms"], EPS_SIJ), kappa)

    def f(z):
        P, *_ = block_sums_and_sens(
            z,
            alpha,
            kappa,
            lambdas,
            S_pow,
            ctx["logS_ms"],
            ctx["m_idx"],
            ctx["s_idx"],
            ctx["stf"],
            ctx["q"],
            ctx["log_q"],
            ctx["I_ms"],
            SKIP_SELF,
            UNDIRECTED,
        )
        return float(P.sum()) - target

    z_lo = max(z_guess, 1e-12)
    f_lo = f(z_lo)
    z_hi, f_hi = z_lo, f_lo
    it = 0
    if f_lo < 0:
        while f_hi < 0 and it < 60:
            z_hi *= 2.0
            f_hi = f(z_hi)
            it += 1
        if f_hi < 0:
            return None
    else:
        while f_lo > 0 and it < 60 and z_lo > 1e-300:
            z_lo *= 0.5
            f_lo = f(z_lo)
            it += 1
        if f_lo > 0:
            return None

    for _ in range(80):
        z_mid = 0.5 * (z_lo + z_hi)
        fm = f(z_mid)
        if abs(fm) <= 1e-6:
            return z_mid
        if fm < 0:
            z_lo = z_mid
        else:
            z_hi = z_mid
    return 0.5 * (z_lo + z_hi)


def update_lambda_column(l, lambdas, idxs_l, E_l, s_l, Wb,
                         eta=ETA_LAMBDA, clip_log=CLIP_LOG_RATIO):
    El = max(E_l[l], 1e-300)
    sl = max(s_l[l], 1e-300)
    step = np.clip(math.log(sl / El), -clip_log, clip_log)
    w = Wb[idxs_l].copy()
    w_sum = float(np.sum(w))
    if w_sum <= 0.0:
        a = np.full_like(w, 1.0 / max(len(w), 1))
    else:
        a = w / w_sum
    if len(idxs_l) > 0:
        scale = np.exp(eta * step * a)
        lambdas[idxs_l] *= scale
        np.maximum(lambdas[idxs_l], LAMBDA_FLOOR, out=lambdas[idxs_l])


def rake_lambdas_few_steps(z, alpha, kappa, lambdas, ctx, steps=RAKE_INNER_STEPS):
    S_pow = np.power(np.maximum(ctx["S_ms"], EPS_SIJ), kappa)
    for _ in range(steps):
        (
            P,
            gP_z,
            gP_a,
            gP_k,
            gP_ll,
            Eblk,
            gE_z,
            gE_a,
            gE_k,
            gE_ll,
        ) = block_sums_and_sens(
            z,
            alpha,
            kappa,
            lambdas,
            S_pow,
            ctx["logS_ms"],
            ctx["m_idx"],
            ctx["s_idx"],
            ctx["stf"],
            ctx["q"],
            ctx["log_q"],
            ctx["I_ms"],
            SKIP_SELF,
            UNDIRECTED,
        )
        Sct = ctx["num_sectors"]
        E_l = np.zeros(Sct, dtype=np.float64)
        Wb = gE_ll
        for b in range(ctx["K"]):
            E_l[ctx["s_idx"][b]] += Eblk[b]
        for l in range(Sct):
            idxs_l = ctx["blocks_by_l"][l]
            if idxs_l.size == 0 or ctx["seller_sizes"][l] <= 0.0:
                continue
            update_lambda_column(l, lambdas, idxs_l, E_l, ctx["seller_sizes"], Wb)
    return lambdas


def warm_start(ctx):
    alpha = 0.5
    kappa = 0.6
    lambdas = np.ones(ctx["K"], dtype=np.float64)
    z = bisect_z_for_degree(alpha, kappa, lambdas, ctx, z_guess=1.0)
    if z is None:
        z = 1.0
    for _ in range(RAKE_OUTER_ALTS):
        lambdas = rake_lambdas_few_steps(z, alpha, kappa, lambdas, ctx)
        z2 = bisect_z_for_degree(alpha, kappa, lambdas, ctx, z_guess=z)
        if z2 is None:
            break
        z = z2
    return z, alpha, kappa, lambdas


# ------------------------------ IPOPT NLP -----------------------------
class GravityNLP:
    def __init__(self, ctx):
        self.ctx = ctx
        self.nd = ctx["n"] * MEAN_DEGREE
        self._last_key = None
        self._cache = None

    def _eval_common(self, u):
        z, alpha, kappa, lambdas = unpack_all(u, self.ctx["K"])
        S_pow = np.power(np.maximum(self.ctx["S_ms"], EPS_SIJ), kappa)

        (
            P,
            gP_z,
            gP_a,
            gP_k,
            gP_ll,
            Eblk,
            gE_z,
            gE_a,
            gE_k,
            gE_ll,
        ) = block_sums_and_sens(
            z,
            alpha,
            kappa,
            lambdas,
            S_pow,
            self.ctx["logS_ms"],
            self.ctx["m_idx"],
            self.ctx["s_idx"],
            self.ctx["stf"],
            self.ctx["q"],
            self.ctx["log_q"],
            self.ctx["I_ms"],
            SKIP_SELF,
            UNDIRECTED,
        )

        total_edges = float(P.sum())
        gP_zeta = float(gP_z.sum())
        gP_ell = gP_ll.copy()
        gP_a_raw = float(gP_a.sum())
        gP_k_raw = float(gP_k.sum())

        Sct = self.ctx["num_sectors"]
        E_l = np.zeros(Sct, dtype=np.float64)
        gE_zeta = np.zeros(Sct, dtype=np.float64)
        gE_a_raw = np.zeros(Sct, dtype=np.float64)
        gE_k_raw = np.zeros(Sct, dtype=np.float64)
        gE_ell = np.zeros(Sct * self.ctx["K"], dtype=np.float64)

        for b in range(self.ctx["K"]):
            l = self.ctx["s_idx"][b]
            E_l[l] += Eblk[b]
            gE_zeta[l] += gE_z[b]
            gE_a_raw[l] += gE_a[b]
            gE_k_raw[l] += gE_k[b]
            gE_ell[l * self.ctx["K"] + b] += gE_ll[b]

        self._cache = dict(
            z=z,
            alpha=alpha,
            kappa=kappa,
            lambdas=lambdas,
            total_edges=total_edges,
            gP_zeta=gP_zeta,
            gP_ell=gP_ell,
            gP_a_raw=gP_a_raw,
            gP_k_raw=gP_k_raw,
            E_l=E_l,
            gE_zeta=gE_zeta,
            gE_a_raw=gE_a_raw,
            gE_k_raw=gE_k_raw,
            gE_ell=gE_ell,
        )
        return self._cache

    def objective(self, u):
        key = tuple(np.round(u, 12))
        if key != self._last_key:
            self._eval_common(u)
            self._last_key = key
        r = self._cache["total_edges"] / self.nd - 1.0
        return r * r

    def gradient(self, u):
        key = tuple(np.round(u, 12))
        if key != self._last_key:
            self._eval_common(u)
            self._last_key = key
        c = self._cache
        r = c["total_edges"] / self.nd - 1.0
        scale = 2.0 * r / self.nd

        a_lo, a_hi = ALPHA_RANGE
        c_lo, c_hi = KAPPA_RANGE
        a = u[1]
        cvar = u[2]
        s_a = sigmoid(a)
        s_c = sigmoid(cvar)
        dalpha_da = (a_hi - a_lo) * s_a * (1.0 - s_a)
        dkappa_dc = (c_hi - c_lo) * s_c * (1.0 - s_c)

        grad = np.zeros_like(u)
        grad[0] = scale * c["gP_zeta"]
        grad[1] = scale * c["gP_a_raw"] * dalpha_da
        grad[2] = scale * c["gP_k_raw"] * dkappa_dc
        grad[3:] = scale * c["gP_ell"]
        return grad

    def constraints(self, u):
        key = tuple(np.round(u, 12))
        if key != self._last_key:
            self._eval_common(u)
            self._last_key = key
        c = self._cache
        s_l = self.ctx["seller_sizes"]
        r = (c["E_l"] - s_l) / np.maximum(s_l, EPS_S)
        return np.concatenate([r - BAND, -r - BAND], axis=0)

    def jacobian(self, u):
        key = tuple(np.round(u, 12))
        if key != self._last_key:
            self._eval_common(u)
            self._last_key = key
        c = self._cache
        s_l = np.maximum(self.ctx["seller_sizes"], EPS_S)

        a_lo, a_hi = ALPHA_RANGE
        c_lo, c_hi = KAPPA_RANGE
        a = u[1]
        cvar = u[2]
        s_a = sigmoid(a)
        s_c = sigmoid(cvar)
        dalpha_da = (a_hi - a_lo) * s_a * (1.0 - s_a)
        dkappa_dc = (c_hi - c_lo) * s_c * (1.0 - s_c)

        Sct = self.ctx["num_sectors"]
        K = self.ctx["K"]
        J = np.zeros((2 * Sct, 3 + K), dtype=np.float64)

        for l in range(Sct):
            row_pos = l
            row_neg = Sct + l
            invs = 1.0 / s_l[l]

            d_z = c["gE_zeta"][l] * invs
            d_a = c["gE_a_raw"][l] * dalpha_da * invs
            d_c = c["gE_k_raw"][l] * dkappa_dc * invs
            d_ell = np.zeros(K, dtype=np.float64)
            base = l * K
            d_ell[:] = c["gE_ell"][base:base + K] * invs

            J[row_pos, 0] = d_z
            J[row_pos, 1] = d_a
            J[row_pos, 2] = d_c
            J[row_pos, 3:] = d_ell

            J[row_neg, :] = -J[row_pos, :]

        return J.ravel()

    def jacobianstructure(self):
        m = 2 * self.ctx["num_sectors"]
        n = 3 + self.ctx["K"]
        rows, cols = np.nonzero(np.ones((m, n), dtype=np.int8))
        return rows, cols


# -------------------- BERNOULLI SAMPLING (UNWEIGHTED) -----------------
@njit(fastmath=True, cache=True)
def _rand01_pair(i_idx, j_idx, seed):
    x = (np.uint64(i_idx) * np.uint64(0x9E3779B97F4A7C15)) ^ (np.uint64(j_idx) << np.uint64(1)) ^ seed
    x ^= (x >> np.uint64(30))
    x *= np.uint64(0xBF58476D1CE4E5B9)
    x ^= (x >> np.uint64(27))
    x *= np.uint64(0x94D049BB133111EB)
    x ^= (x >> np.uint64(31))
    return float(x) / 18446744073709551616.0  # 2**64


@njit(parallel=True, fastmath=True, cache=True)
def _count_block_edges(i_list, j_list, z, lam, s_pow, q_pow_alpha, skip_self, undirected):
    Li = i_list.shape[0]
    counts = np.zeros(Li, dtype=np.int64)
    for ii in prange(Li):
        i = i_list[ii]
        qi_a = q_pow_alpha[i]
        c = 0
        for jj in range(j_list.shape[0]):
            j = j_list[jj]
            if skip_self and i == j:
                continue
            if undirected and j <= i:
                continue
            x = z * lam * s_pow * qi_a * q_pow_alpha[j]
            p = 1.0 if x > 1e6 else x / (1.0 + x)
            if p <= 0.0:
                continue
            r = _rand01_pair(i, j, SEED64)
            if p >= r:
                c += 1
        counts[ii] = c
    return counts


@njit(parallel=True, fastmath=True, cache=True)
def _fill_block_edges(
    i_list,
    j_list,
    z,
    lam,
    s_pow,
    q_pow_alpha,
    skip_self,
    undirected,
    offsets,
    edges_out,
):
    Li = i_list.shape[0]
    for ii in prange(Li):
        i = i_list[ii]
        qi_a = q_pow_alpha[i]
        pos = offsets[ii]
        for jj in range(j_list.shape[0]):
            j = j_list[jj]
            if skip_self and i == j:
                continue
            if undirected and j <= i:
                continue
            x = z * lam * s_pow * qi_a * q_pow_alpha[j]
            p = 1.0 if x > 1e6 else x / (1.0 + x)
            if p <= 0.0:
                continue
            r = _rand01_pair(i, j, SEED64)
            if p >= r:
                edges_out[pos, 0] = i
                edges_out[pos, 1] = j
                pos += 1


def write_unweighted_network(info, z, alpha, kappa, lambdas, outfile="edgerr.txt"):
    log("Preparing Bernoulli draw with optimized parameters …")
    log(f"  z      = {z:.12e}")
    log(f"  alpha  = {alpha:.6f}")
    log(f"  kappa  = {kappa:.6f}")
    log(f"  K (blocks) = {info['K']}")
    np.set_printoptions(linewidth=140, threshold=np.inf, floatmode="maxprec_equal")
    print("  lambdas =", np.array(lambdas, dtype=np.float64))

    q_pow_alpha = np.ascontiguousarray(np.exp(alpha * info["log_q"]))
    s_pow_arr   = np.ascontiguousarray(np.power(np.maximum(info["S_ms"], EPS_SIJ), kappa))

    edges_chunks = []
    total_edges = 0
    t0 = time.time()
    log("Sampling Bernoulli edges… (two-pass per block)")
    for b in range(info["K"]):
        m = info["m_idx"][b]
        s = info["s_idx"][b]
        i_list = info["stf"][m]
        j_list = info["stf"][s]
        if i_list.shape[0] == 0 or j_list.shape[0] == 0:
            continue

        lam = max(lambdas[b], LAMBDA_FLOOR)
        counts = _count_block_edges(
            i_list,
            j_list,
            z,
            lam,
            s_pow_arr[b],
            q_pow_alpha,
            SKIP_SELF,
            UNDIRECTED,
        )
        block_E = int(counts.sum())
        if block_E == 0:
            continue

        offsets = np.zeros(counts.shape[0] + 1, dtype=np.int64)
        np.cumsum(counts, out=offsets[1:])
        edges_block = np.empty((block_E, 2), dtype=np.int64)

        _fill_block_edges(
            i_list,
            j_list,
            z,
            lam,
            s_pow_arr[b],
            q_pow_alpha,
            SKIP_SELF,
            UNDIRECTED,
            offsets,
            edges_block,
        )

        edges_chunks.append(edges_block)
        total_edges += block_E

        if ((b + 1) % 10 == 0) or (b + 1 == info["K"]):
            log(f"  processed {b+1}/{info['K']} blocks… cumulative edges={total_edges:,}")

    if not edges_chunks:
        log("No edges sampled; writing empty file.")
        open(outfile, "w").close()
        return

    edges = np.concatenate(edges_chunks, axis=0)
    edges_out = edges + 1
    with open(outfile, "w") as f:
        for i, j in edges_out:
            f.write(f"{i} {j}\n")

    dt = time.time() - t0
    log(f"Done. Wrote {edges.shape[0]:,} edges to {outfile} in {dt:.1f}s.")


# -------------------------------- MAIN --------------------------------
def main():
    info = load_data()
    n, Sct, K = info["n"], info["num_sectors"], info["K"]
    target = n * MEAN_DEGREE
    log(f"Loaded. n={n:,}, sectors={Sct}, nonzero blocks={K}")
    log(f"Target edges = {target:.1f}, BAND = ±{int(BAND * 100)}% (modulus)")

    z0, alpha0, kappa0, lam0 = warm_start(info)

    a_lo, a_hi = ALPHA_RANGE
    c_lo, c_hi = KAPPA_RANGE
    u0 = np.zeros(3 + K, dtype=np.float64)
    u0[0] = np.log(max(z0, 1e-12))
    u0[1] = inv_sigmoid((alpha0 - a_lo) / max(a_hi - a_lo, 1e-12))
    u0[2] = inv_sigmoid((kappa0 - c_lo) / max(c_hi - c_lo, 1e-12))
    u0[3:] = np.log(np.minimum(np.maximum(lam0, LAMBDA_FLOOR), 1e300))

    nlp = GravityNLP(info)
    m = 2 * Sct
    lb = -np.inf * np.ones_like(u0)
    ub = np.inf * np.ones_like(u0)

    cl = -np.inf * np.ones(m, dtype=np.float64)
    cu = np.zeros(m, dtype=np.float64)

    nlp_prob = cyipopt.Problem(
        n=len(u0),
        m=m,
        problem_obj=nlp,
        lb=lb,
        ub=ub,
        cl=cl,
        cu=cu,
    )

    for k, v in IPOPT_OPTIONS.items():
        nlp_prob.add_option(k, v)

    sol, info_out = nlp_prob.solve(u0)

    z, alpha, kappa, lambdas = unpack_all(sol, K)
    nlp._last_key = None
    _ = nlp.objective(sol)
    cache = nlp._cache
    total_edges = cache["total_edges"]
    r = (cache["E_l"] - info["seller_sizes"]) / np.maximum(info["seller_sizes"], EPS_S)

    print("\n=== Final parameters ===")
    print(f"z      = {z:.12e}")
    print(f"alpha  = {alpha:.6f}")
    print(f"kappa  = {kappa:.6f}")
    print(f"lambdas: K = {K}, min={lambdas.min():.3e}, max={lambdas.max():{'.3e'}}")

    print("\n=== Objective diagnostic ===")
    print(f"sum p_ij = {total_edges:.6f} | target = {target:.6f} | gap = {total_edges - target:+.6f}")
    print(f"objective f = {(total_edges / target - 1.0) ** 2:.6e}")

    print("\n=== Seller constraints (modulus band) ===")
    viol = 0
    for l in range(Sct):
        status = "OK" if abs(r[l]) <= BAND + 1e-12 else "VIOLATED"
        if status != "OK":
            viol += 1
        print(f"  l={l:2d}: rel={r[l]:+.6e}  [{status}]")
    print(f"\nViolated: {viol} / {Sct}")

    log("Starting Bernoulli draw using the EXACT optimized parameters above …")
    write_unweighted_network(info, z, alpha, kappa, lambdas, outfile="edges_unweighted")


if __name__ == "__main__":
    main()