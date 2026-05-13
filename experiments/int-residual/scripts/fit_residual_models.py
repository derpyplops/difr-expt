"""Fit and evaluate residual models on the flat per-cell residual dataset.

Each model takes some subset of {Y', s_X, s_W, family, top-k mass,
sum_pos, sum_neg, ...} and returns an integer prediction r̂. We then
evaluate (r̂ - r) on held-out prompts.

We deliberately stay simple and report a table:

  | model  | features      | data | abs_mean | p50 | p99 | p99.9 | worst | signed_p1 | signed_p99 | bit_exact% |

with the same row broken out by matmul family (q/k/v/o/gate/up/down) too.

Models implemented (per plan.md):
  R1: zero            (no correction)
  R2: global_constant (round(mean(r)))
  R3: per_family      (round(mean(r) per family))
  R4: affine_Y        (a + b·Y' with a,b learned in fp64, output rounded to int)
  R5: per_d           (per-output-column constant) — pooled across (block, family) and per family
  R7: scale_lut       (r̂ = round(α · s_X · s_W · Y')) — Luke's LUT critique
  R8: sign_split      (r̂ = a·sum_pos + b·sum_neg + c·Y', integer-rounded)
  R9: outlier_aware   (r̂ = a + b·Y' + c·n_clip_X + d·n_clip_W + e·x_absmax + f·w_absmax)

All models compose: feature_extractor → linear (or rule) → round-to-int.

The "fit" is always least-squares on the integer residual viewed as a
real-valued target, then we round predictions to int at eval time.
Bit-exact rate is computed in integer space.
"""

from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch


FAMILY_NAMES = {0: "q", 1: "k", 2: "v", 3: "o", 4: "gate", 5: "up", 6: "down", 7: "other"}


def load_dataset(
    files: list[Path], subsample_per_matmul: int = 0, seed: int = 0
) -> dict[str, torch.Tensor]:
    """Concatenate per-prompt flat datasets into one big in-memory table.

    If `subsample_per_matmul > 0`, retain only that many cells per
    (matmul_id, prompt) pair. Lets us fit residual models on a small
    representative sample of an otherwise-huge dataset without losing
    coverage across blocks/families.
    """
    chunks: dict[str, list] = defaultdict(list)
    meta = None
    rng = torch.Generator().manual_seed(seed)
    for fi, f in enumerate(files):
        blob = torch.load(f, map_location="cpu", weights_only=False)
        if subsample_per_matmul > 0:
            mid = blob["matmul_id"].to(torch.int64)
            n = len(mid)
            # Indexes-per-matmul: find segment boundaries assuming records are
            # appended in matmul order (true by construction of build_residuals).
            keep_mask = torch.zeros(n, dtype=torch.bool)
            # Build a stable per-matmul subsample without sorting the full
            # array each time: group via numpy.bincount equivalent.
            unique_ids = torch.unique(mid)
            for uid in unique_ids.tolist():
                idxs = (mid == uid).nonzero(as_tuple=True)[0]
                if idxs.numel() <= subsample_per_matmul:
                    keep_mask[idxs] = True
                else:
                    perm = torch.randperm(idxs.numel(), generator=rng)
                    keep_mask[idxs[perm[:subsample_per_matmul]]] = True
            for k, v in blob.items():
                if k == "meta":
                    if meta is None:
                        meta = v
                    continue
                if not torch.is_tensor(v):
                    continue
                chunks[k].append(v[keep_mask])
        else:
            for k, v in blob.items():
                if k == "meta":
                    if meta is None:
                        meta = v
                    continue
                if not torch.is_tensor(v):
                    continue
                chunks[k].append(v)
    out: dict[str, torch.Tensor] = {}
    for k, vs in chunks.items():
        out[k] = torch.cat(vs, dim=0)
    out["__meta__"] = meta
    return out


def fit_and_eval(
    train: dict[str, torch.Tensor],
    val:   dict[str, torch.Tensor],
    name: str,
    feature_fn,
    extra_train_kwargs: dict | None = None,
) -> dict:
    """feature_fn(data) -> (X, params_dict_or_None).

    If the model has params, we fit them by least squares on train data using
    feature_fn's returned X, and re-call feature_fn with the params on val.
    For purely rule-based models (no params), feature_fn returns
    (r_hat_train_or_val, None).

    Returns a dict of metrics.
    """
    # We use a uniform interface where the feature_fn produces r_hat directly,
    # and may be parameterized by fitting on the train slice first.
    r_hat_val, info = feature_fn(train=train, val=val)
    r_val = val["r"].to(torch.int64)
    err = r_hat_val.to(torch.int64) - r_val
    abs_err = err.abs()
    metrics = {
        "model": name,
        "info": info or {},
        "n_val_cells": int(r_val.numel()),
        "abs_mean": float(abs_err.float().mean().item()),
        "abs_p50":  float(torch.quantile(abs_err.float(), 0.5).item()),
        "abs_p99":  float(torch.quantile(abs_err.float(), 0.99).item()),
        "abs_p99.9": float(torch.quantile(abs_err.float(), 0.999).item()),
        "abs_worst": int(abs_err.max().item()),
        "signed_mean": float(err.float().mean().item()),
        "signed_p1":  float(torch.quantile(err.float(), 0.01).item()),
        "signed_p99": float(torch.quantile(err.float(), 0.99).item()),
        "bit_exact_frac": float((err == 0).sum().item()) / r_val.numel(),
    }
    # Per-family breakouts
    family = val["family"].to(torch.int64)
    fam_metrics = {}
    for fcode, fname in FAMILY_NAMES.items():
        mask = family == fcode
        n = mask.sum().item()
        if n == 0:
            continue
        ae = abs_err[mask].float()
        e = err[mask].float()
        fam_metrics[fname] = {
            "n": int(n),
            "abs_mean": float(ae.mean().item()),
            "abs_p99": float(torch.quantile(ae, 0.99).item()),
            "abs_worst": int(abs_err[mask].max().item()),
            "signed_mean": float(e.mean().item()),
            "bit_exact_frac": float((err[mask] == 0).sum().item()) / max(n, 1),
        }
    metrics["per_family"] = fam_metrics
    return metrics


# ---------------------------------------------------------------------------
# Residual models
# ---------------------------------------------------------------------------

def model_R1_zero():
    """r̂ = 0  — the no-correction baseline."""
    def fn(*, train, val):
        return torch.zeros_like(val["r"], dtype=torch.int64), {"params": "none"}
    return fn


def model_R2_global_constant():
    """r̂ = round(mean(r))  — global integer constant."""
    def fn(*, train, val):
        r = train["r"].float()
        a = int(round(r.mean().item()))
        return torch.full_like(val["r"], a, dtype=torch.int64), {"a": a}
    return fn


def model_R3_per_family():
    """r̂ = round(mean_r per family)."""
    def fn(*, train, val):
        r_train = train["r"].float()
        fam_train = train["family"].to(torch.int64)
        means = {}
        for fcode in FAMILY_NAMES:
            mask = fam_train == fcode
            if mask.sum() > 0:
                means[fcode] = int(round(r_train[mask].mean().item()))
            else:
                means[fcode] = 0
        fam_val = val["family"].to(torch.int64)
        r_hat = torch.zeros_like(val["r"], dtype=torch.int64)
        for fcode, a in means.items():
            r_hat[fam_val == fcode] = a
        return r_hat, {"means": {FAMILY_NAMES[k]: v for k, v in means.items()}}
    return fn


def model_R4_affine_Y():
    """r̂ = round(a + b · Y')  — fit a, b by LSQ on train cells."""
    def fn(*, train, val):
        Y_train = train["Y_prime"].to(torch.float64).numpy()
        r_train = train["r"].to(torch.float64).numpy()
        # LSQ: r = a + b Y → [1, Y] · [a; b] = r
        n = len(Y_train)
        # Use closed-form for 2-param: efficient and exact.
        sum_Y = Y_train.sum()
        sum_r = r_train.sum()
        sum_YY = (Y_train * Y_train).sum()
        sum_Yr = (Y_train * r_train).sum()
        # mean-subtraction for numerical stability:
        mean_Y = sum_Y / n
        mean_r = sum_r / n
        cov = (Y_train - mean_Y) * (r_train - mean_r)
        var = (Y_train - mean_Y) ** 2
        b = cov.sum() / max(var.sum(), 1e-30)
        a = mean_r - b * mean_Y
        Y_val = val["Y_prime"].to(torch.float64)
        r_hat_f = a + b * Y_val
        r_hat = torch.round(r_hat_f).to(torch.int64)
        return r_hat, {"a": float(a), "b": float(b)}
    return fn


def model_R5_per_d_per_family():
    """r̂[t, d] = mean_r per (family, d)  — captures per-output-column bias.

    Note: 'd' indices repeat across blocks within the same family (e.g. all
    q_proj outputs of block 0 share d=0,1,...). Pooling per (family, d)
    averages across blocks. We use (family, block, d) for a richer model.
    """
    def fn(*, train, val):
        # Group by (family, block, d). Use a hashed key to vectorize.
        fam_b = train["family"].to(torch.int64) * 1_000_000 \
              + train["block"].to(torch.int64) * 10_000 \
              + train["d"].to(torch.int64)
        r_train = train["r"].to(torch.float64)
        # Sort by group, then segment-mean
        sort_idx = torch.argsort(fam_b)
        keys_s = fam_b[sort_idx]
        r_s = r_train[sort_idx]
        # Find group boundaries
        change = torch.cat([torch.tensor([True]),
                            keys_s[1:] != keys_s[:-1]])
        # group start positions
        starts = torch.where(change)[0]
        ends = torch.cat([starts[1:], torch.tensor([len(keys_s)])])
        means = torch.zeros(len(starts), dtype=torch.float64)
        for i in range(len(starts)):
            means[i] = r_s[starts[i]:ends[i]].mean()
        keys_uniq = keys_s[starts]
        mean_map = dict(zip(keys_uniq.tolist(), means.tolist()))

        fam_b_val = val["family"].to(torch.int64) * 1_000_000 \
                  + val["block"].to(torch.int64) * 10_000 \
                  + val["d"].to(torch.int64)
        r_hat_f = torch.tensor(
            [mean_map.get(int(k), 0.0) for k in fam_b_val.tolist()],
            dtype=torch.float64,
        )
        r_hat = torch.round(r_hat_f).to(torch.int64)
        return r_hat, {"n_groups": len(starts)}
    return fn


def model_R7_scale_lut():
    """r̂ = round(α · s_X · s_W · Y')  — single learned scalar α.

    This is the "fp32 cast rounding" formula: the residual is roughly
    proportional to the fp32 reduction error, which scales with the operand
    scale product and Y'. Single-parameter ⇒ proof-cheap.
    """
    def fn(*, train, val):
        # Feature: s_X * s_W * Y'
        feat_train = (train["s_X"].to(torch.float64)
                      * train["s_W"].to(torch.float64)
                      * train["Y_prime"].to(torch.float64))
        r_train = train["r"].to(torch.float64)
        num = (feat_train * r_train).sum().item()
        den = (feat_train * feat_train).sum().item()
        alpha = num / max(den, 1e-30)
        feat_val = (val["s_X"].to(torch.float64)
                    * val["s_W"].to(torch.float64)
                    * val["Y_prime"].to(torch.float64))
        r_hat_f = alpha * feat_val
        r_hat = torch.round(r_hat_f).to(torch.int64)
        return r_hat, {"alpha": float(alpha)}
    return fn


def model_R7b_affine_in_scale_Yp():
    """r̂ = round(a + b · s_X · s_W · Y' + c · Y')  — Luke's LUT critique with
    affine + interaction. Still only 3 params; proof-trivial.
    """
    def fn(*, train, val):
        s_Yp_train = (train["s_X"].to(torch.float64)
                      * train["s_W"].to(torch.float64)
                      * train["Y_prime"].to(torch.float64))
        Yp_train = train["Y_prime"].to(torch.float64)
        r_train = train["r"].to(torch.float64)
        # LSQ on [1, s_Yp, Yp]
        N = len(r_train)
        ones = torch.ones(N, dtype=torch.float64)
        X = torch.stack([ones, s_Yp_train, Yp_train], dim=1).numpy()
        y = r_train.numpy()
        # Normal equations:
        XtX = X.T @ X
        Xty = X.T @ y
        coefs = np.linalg.solve(XtX, Xty)
        s_Yp_val = (val["s_X"].to(torch.float64)
                    * val["s_W"].to(torch.float64)
                    * val["Y_prime"].to(torch.float64))
        Yp_val = val["Y_prime"].to(torch.float64)
        r_hat_f = coefs[0] + coefs[1] * s_Yp_val + coefs[2] * Yp_val
        r_hat = torch.round(r_hat_f).to(torch.int64)
        return r_hat, {"a": float(coefs[0]), "b": float(coefs[1]), "c": float(coefs[2])}
    return fn


def _lsq_int_round(X: np.ndarray, y: np.ndarray, X_val: np.ndarray) -> np.ndarray:
    """Solve LSQ in float64, return integer-rounded predictions."""
    XtX = X.T @ X
    Xty = X.T @ y
    try:
        coefs = np.linalg.solve(XtX, Xty)
    except np.linalg.LinAlgError:
        coefs, *_ = np.linalg.lstsq(X, y, rcond=None)
    r_hat_f = X_val @ coefs
    return np.round(r_hat_f).astype(np.int64), coefs


def model_R8_sign_split():
    """r̂ = round(a + b·sum_pos + c·sum_neg + d·Y' + e·s_X·s_W·Y').

    Requires that compute_features was run (sum_pos/sum_neg present).
    """
    def fn(*, train, val):
        if "sum_pos" not in train:
            raise RuntimeError("R8 needs --compute-features in build_residuals")
        N = len(train["r"])
        ones = np.ones(N, dtype=np.float64)
        sp = train["sum_pos"].to(torch.float64).numpy()
        sn = train["sum_neg"].to(torch.float64).numpy()
        Yp = train["Y_prime"].to(torch.float64).numpy()
        sYp = (train["s_X"].to(torch.float64)
               * train["s_W"].to(torch.float64)
               * train["Y_prime"].to(torch.float64)).numpy()
        Xt = np.stack([ones, sp, sn, Yp, sYp], axis=1)
        y = train["r"].to(torch.float64).numpy()
        sp_v = val["sum_pos"].to(torch.float64).numpy()
        sn_v = val["sum_neg"].to(torch.float64).numpy()
        Yp_v = val["Y_prime"].to(torch.float64).numpy()
        sYp_v = (val["s_X"].to(torch.float64)
                 * val["s_W"].to(torch.float64)
                 * val["Y_prime"].to(torch.float64)).numpy()
        Xv = np.stack([np.ones(len(Yp_v), dtype=np.float64), sp_v, sn_v, Yp_v, sYp_v], axis=1)
        r_hat_np, coefs = _lsq_int_round(Xt, y, Xv)
        return torch.from_numpy(r_hat_np), {
            "a": float(coefs[0]), "b_sumpos": float(coefs[1]),
            "c_sumneg": float(coefs[2]), "d_Yp": float(coefs[3]),
            "e_sXsWYp": float(coefs[4]),
        }
    return fn


def model_R10_delta_summaries():
    """r̂ uses per-row/per-col delta reductions (cheap precomputed features).

    Predicts the leading-term proxy:
       r ≈ α·X̄·Σ_k δ_W + α·W̄·Σ_k δ_X + (small δ·δ)
    where α = MAX/FP8_MAX is the cast scale and X̄, W̄ are per-row/per-col
    means of X', W'. We feed the actual per-row & per-col sums separately so
    the regressor can disentangle.
    """
    def fn(*, train, val):
        N_tr = len(train["r"])
        N_va = len(val["r"])
        feats_tr_list = [
            np.ones(N_tr, dtype=np.float64),
            train["Y_prime"].to(torch.float64).numpy(),
            train["dX_sum_t"].to(torch.float64).numpy(),
            train["dW_sum_d"].to(torch.float64).numpy(),
            train["dX_abs_sum_t"].to(torch.float64).numpy(),
            train["dW_abs_sum_d"].to(torch.float64).numpy(),
            train["Xp_sum_t"].to(torch.float64).numpy(),
            train["Wp_sum_d"].to(torch.float64).numpy(),
            # cross-interactions that match the algebra:
            (train["Xp_sum_t"].to(torch.float64) * train["dW_sum_d"].to(torch.float64)).numpy(),
            (train["Wp_sum_d"].to(torch.float64) * train["dX_sum_t"].to(torch.float64)).numpy(),
        ]
        feats_va_list = [
            np.ones(N_va, dtype=np.float64),
            val["Y_prime"].to(torch.float64).numpy(),
            val["dX_sum_t"].to(torch.float64).numpy(),
            val["dW_sum_d"].to(torch.float64).numpy(),
            val["dX_abs_sum_t"].to(torch.float64).numpy(),
            val["dW_abs_sum_d"].to(torch.float64).numpy(),
            val["Xp_sum_t"].to(torch.float64).numpy(),
            val["Wp_sum_d"].to(torch.float64).numpy(),
            (val["Xp_sum_t"].to(torch.float64) * val["dW_sum_d"].to(torch.float64)).numpy(),
            (val["Wp_sum_d"].to(torch.float64) * val["dX_sum_t"].to(torch.float64)).numpy(),
        ]
        Xt = np.stack(feats_tr_list, axis=1)
        Xv = np.stack(feats_va_list, axis=1)
        y = train["r"].to(torch.float64).numpy()
        r_hat_np, coefs = _lsq_int_round(Xt, y, Xv)
        return torch.from_numpy(r_hat_np), {
            "coefs": coefs.tolist(),
            "names": ["1", "Y'", "Σδ_X[t]", "Σδ_W[d]", "Σ|δ_X[t]|", "Σ|δ_W[d]|",
                      "ΣX'[t]", "ΣW'[d]", "ΣX'·Σδ_W", "ΣW'·Σδ_X"],
        }
    return fn


def model_R10b_per_family_delta():
    """R10 but a separate regression per matmul family.

    Different layers have different magnitudes — pooling across families
    averages away structure. Per-family coefficients are still cheap
    (≤10 family × ≤10 features = ≤100 numbers).
    """
    def fn(*, train, val):
        fam_tr = train["family"].to(torch.int64).numpy()
        fam_va = val["family"].to(torch.int64).numpy()
        all_keys = ["Y_prime", "dX_sum_t", "dW_sum_d", "dX_abs_sum_t",
                    "dW_abs_sum_d", "Xp_sum_t", "Wp_sum_d"]
        def stack(d, idx_mask):
            n = idx_mask.sum()
            feats = [np.ones(n, dtype=np.float64)] + [
                d[k].to(torch.float64).numpy()[idx_mask] for k in all_keys
            ] + [
                (d["Xp_sum_t"].to(torch.float64) * d["dW_sum_d"].to(torch.float64)).numpy()[idx_mask],
                (d["Wp_sum_d"].to(torch.float64) * d["dX_sum_t"].to(torch.float64)).numpy()[idx_mask],
            ]
            return np.stack(feats, axis=1)
        r_hat = np.zeros(len(val["r"]), dtype=np.int64)
        info = {"per_family_coefs": {}}
        for fcode in FAMILY_NAMES:
            mtr = fam_tr == fcode
            mva = fam_va == fcode
            if mtr.sum() == 0 or mva.sum() == 0:
                continue
            Xt = stack(train, mtr)
            Xv = stack(val, mva)
            yt = train["r"].to(torch.float64).numpy()[mtr]
            r_hat_chunk, coefs = _lsq_int_round(Xt, yt, Xv)
            r_hat[mva] = r_hat_chunk
            info["per_family_coefs"][FAMILY_NAMES[fcode]] = coefs.tolist()
        return torch.from_numpy(r_hat), info
    return fn


def model_R11_mixed_dotprods():
    """r̂ uses the exact mixed dot products X'·δ_W, W'·δ_X, δ_X·δ_W.

    These are PER-CELL features that cost K-sum work each — same order as
    the matmul itself. The point is to bound how well a residual model
    *could* do if we paid for per-cell features. This is the ceiling.
    """
    def fn(*, train, val):
        if "Xp_dW" not in train:
            raise RuntimeError("R11 needs --compute-features in build_residuals")
        N_tr = len(train["r"])
        N_va = len(val["r"])
        feats_tr = np.stack([
            np.ones(N_tr, dtype=np.float64),
            train["Xp_dW"].to(torch.float64).numpy(),
            train["Wp_dX"].to(torch.float64).numpy(),
            train["dX_dW"].to(torch.float64).numpy(),
        ], axis=1)
        feats_va = np.stack([
            np.ones(N_va, dtype=np.float64),
            val["Xp_dW"].to(torch.float64).numpy(),
            val["Wp_dX"].to(torch.float64).numpy(),
            val["dX_dW"].to(torch.float64).numpy(),
        ], axis=1)
        y = train["r"].to(torch.float64).numpy()
        r_hat_np, coefs = _lsq_int_round(feats_tr, y, feats_va)
        return torch.from_numpy(r_hat_np), {
            "coefs": coefs.tolist(),
            "names": ["1", "X'·δ_W", "W'·δ_X", "δ_X·δ_W"],
        }
    return fn


def model_R11a_first_order_only():
    """R11 without the second-order term δ_X·δ_W.

    2 extra K-sum-per-cell features instead of 3 — same proof cost as
    2 extra matmuls on top of Y'. Tests how much accuracy the second-order
    term contributes (algebra says it's bounded by K·0.25, much smaller
    than the first-order terms).
    """
    def fn(*, train, val):
        if "Xp_dW" not in train:
            raise RuntimeError("R11a needs --compute-features in build_residuals")
        N_tr = len(train["r"])
        N_va = len(val["r"])
        feats_tr = np.stack([
            np.ones(N_tr, dtype=np.float64),
            train["Xp_dW"].to(torch.float64).numpy(),
            train["Wp_dX"].to(torch.float64).numpy(),
        ], axis=1)
        feats_va = np.stack([
            np.ones(N_va, dtype=np.float64),
            val["Xp_dW"].to(torch.float64).numpy(),
            val["Wp_dX"].to(torch.float64).numpy(),
        ], axis=1)
        y = train["r"].to(torch.float64).numpy()
        r_hat_np, coefs = _lsq_int_round(feats_tr, y, feats_va)
        return torch.from_numpy(r_hat_np), {
            "coefs": coefs.tolist(),
            "names": ["1", "X'·δ_W", "W'·δ_X"],
        }
    return fn


def model_R11b_only_XpdW():
    """1-matmul variant: keep only X'·δ_W (drop W'·δ_X and δ_X·δ_W).

    Tests whether one of the two first-order terms carries most of the
    signal. The two terms have similar magnitude in expectation
    (|β c_W δ_X| vs |β c_X δ_W|), so dropping one should roughly halve
    the explained variance and bring abs error closer to R1's ~273 baseline.
    Cost: 1 K-sum matmul over (X', δ_W).
    """
    def fn(*, train, val):
        if "Xp_dW" not in train:
            raise RuntimeError("R11b needs --compute-features in build_residuals")
        N_tr = len(train["r"])
        N_va = len(val["r"])
        feats_tr = np.stack([
            np.ones(N_tr, dtype=np.float64),
            train["Xp_dW"].to(torch.float64).numpy(),
        ], axis=1)
        feats_va = np.stack([
            np.ones(N_va, dtype=np.float64),
            val["Xp_dW"].to(torch.float64).numpy(),
        ], axis=1)
        y = train["r"].to(torch.float64).numpy()
        r_hat_np, coefs = _lsq_int_round(feats_tr, y, feats_va)
        return torch.from_numpy(r_hat_np), {
            "coefs": coefs.tolist(),
            "names": ["1", "X'·δ_W"],
        }
    return fn


def model_R11b_only_WpdX():
    """1-matmul variant: keep only W'·δ_X (drop X'·δ_W and δ_X·δ_W).

    Symmetric counterpart to model_R11b_only_XpdW. Useful for checking
    whether X- or W-side rounding dominates the residual on this teacher.
    """
    def fn(*, train, val):
        if "Wp_dX" not in train:
            raise RuntimeError("R11b_W needs --compute-features in build_residuals")
        N_tr = len(train["r"])
        N_va = len(val["r"])
        feats_tr = np.stack([
            np.ones(N_tr, dtype=np.float64),
            train["Wp_dX"].to(torch.float64).numpy(),
        ], axis=1)
        feats_va = np.stack([
            np.ones(N_va, dtype=np.float64),
            val["Wp_dX"].to(torch.float64).numpy(),
        ], axis=1)
        y = train["r"].to(torch.float64).numpy()
        r_hat_np, coefs = _lsq_int_round(feats_tr, y, feats_va)
        return torch.from_numpy(r_hat_np), {
            "coefs": coefs.tolist(),
            "names": ["1", "W'·δ_X"],
        }
    return fn


def model_R12a_mixed_signXp_dW():
    """Mixed-precision: keep X'·δ_W exact, sign-quantize the W'·δ_X side.

    Equivalent proof cost: 1 full K-sum matmul + 1 K-sum over {-1,0,+1}.
    The sign-matmul is cheaper to commit (1 bit per δ entry).
    Tests whether one of the two terms tolerates magnitude loss better
    than the other.
    """
    def fn(*, train, val):
        if "Xp_dW" not in train:
            raise RuntimeError("R12a needs --compute-features in build_residuals")
        N_tr = len(train["r"])
        N_va = len(val["r"])
        feats_tr = np.stack([
            np.ones(N_tr, dtype=np.float64),
            train["Xp_dW"].to(torch.float64).numpy(),
            train["signdX_Wp"].to(torch.float64).numpy(),
        ], axis=1)
        feats_va = np.stack([
            np.ones(N_va, dtype=np.float64),
            val["Xp_dW"].to(torch.float64).numpy(),
            val["signdX_Wp"].to(torch.float64).numpy(),
        ], axis=1)
        y = train["r"].to(torch.float64).numpy()
        r_hat_np, coefs = _lsq_int_round(feats_tr, y, feats_va)
        return torch.from_numpy(r_hat_np), {
            "coefs": coefs.tolist(),
            "names": ["1", "X'·δ_W (full)", "sign(δ_X)·W'"],
        }
    return fn


def model_R12a_mixed_Xp_signdW():
    """Symmetric counterpart: full W'·δ_X + sign-quantized X'·δ_W.

    Same cost profile as R12a_mixed_signXp_dW. Run both to see if one
    side's sign-quant is more lossy than the other.
    """
    def fn(*, train, val):
        if "Wp_dX" not in train:
            raise RuntimeError("R12a needs --compute-features in build_residuals")
        N_tr = len(train["r"])
        N_va = len(val["r"])
        feats_tr = np.stack([
            np.ones(N_tr, dtype=np.float64),
            train["Wp_dX"].to(torch.float64).numpy(),
            train["Xp_signdW"].to(torch.float64).numpy(),
        ], axis=1)
        feats_va = np.stack([
            np.ones(N_va, dtype=np.float64),
            val["Wp_dX"].to(torch.float64).numpy(),
            val["Xp_signdW"].to(torch.float64).numpy(),
        ], axis=1)
        y = train["r"].to(torch.float64).numpy()
        r_hat_np, coefs = _lsq_int_round(feats_tr, y, feats_va)
        return torch.from_numpy(r_hat_np), {
            "coefs": coefs.tolist(),
            "names": ["1", "W'·δ_X (full)", "X'·sign(δ_W)"],
        }
    return fn


def model_R12_sign_quantized_delta():
    """Approximation: replace δ with sign(δ) in the mixed dot products.

    |δ| ≤ 0.5 with mean magnitude near 0.25 (assuming uniform fractional
    parts). The 1-bit approximation drops magnitude information; expect
    significantly worse than R11 but with the cheapest possible cost:
    matmul over {-1, 0, +1} entries (≈ same circuit cost as a real matmul
    but smaller commitment).
    """
    def fn(*, train, val):
        if "Xp_signdW" not in train:
            raise RuntimeError("R12 needs --compute-features in build_residuals")
        N_tr = len(train["r"])
        N_va = len(val["r"])
        feats_tr = np.stack([
            np.ones(N_tr, dtype=np.float64),
            train["Xp_signdW"].to(torch.float64).numpy(),
            train["signdX_Wp"].to(torch.float64).numpy(),
        ], axis=1)
        feats_va = np.stack([
            np.ones(N_va, dtype=np.float64),
            val["Xp_signdW"].to(torch.float64).numpy(),
            val["signdX_Wp"].to(torch.float64).numpy(),
        ], axis=1)
        y = train["r"].to(torch.float64).numpy()
        r_hat_np, coefs = _lsq_int_round(feats_tr, y, feats_va)
        return torch.from_numpy(r_hat_np), {
            "coefs": coefs.tolist(),
            "names": ["1", "X'·sign(δ_W)", "sign(δ_X)·W'"],
        }
    return fn


def model_R11_per_family():
    """R11 but with per-family coefficients.

    Different layers have very different scale distributions; the
    algebra coefs are universal (1, 1, -1) but the floor magnitudes per
    family differ. Per-family fit lets the intercept absorb per-family
    accumulator-drift bias.
    """
    def fn(*, train, val):
        if "Xp_dW" not in train:
            raise RuntimeError("R11_per_family needs --compute-features")
        r_hat = np.zeros(len(val["r"]), dtype=np.int64)
        info = {"per_family_coefs": {}}
        fam_tr = train["family"].to(torch.int64).numpy()
        fam_va = val["family"].to(torch.int64).numpy()
        for fcode in FAMILY_NAMES:
            mtr = fam_tr == fcode
            mva = fam_va == fcode
            if mtr.sum() == 0 or mva.sum() == 0:
                continue
            feats_tr = np.stack([
                np.ones(mtr.sum(), dtype=np.float64),
                train["Xp_dW"].to(torch.float64).numpy()[mtr],
                train["Wp_dX"].to(torch.float64).numpy()[mtr],
                train["dX_dW"].to(torch.float64).numpy()[mtr],
            ], axis=1)
            feats_va = np.stack([
                np.ones(mva.sum(), dtype=np.float64),
                val["Xp_dW"].to(torch.float64).numpy()[mva],
                val["Wp_dX"].to(torch.float64).numpy()[mva],
                val["dX_dW"].to(torch.float64).numpy()[mva],
            ], axis=1)
            y = train["r"].to(torch.float64).numpy()[mtr]
            r_hat_np, coefs = _lsq_int_round(feats_tr, y, feats_va)
            r_hat[mva] = r_hat_np
            info["per_family_coefs"][FAMILY_NAMES[fcode]] = coefs.tolist()
        return torch.from_numpy(r_hat), info
    return fn


def model_R13_rank1():
    """**Zero-K-sum** rank-1 model.

    Hypothesis: most of W'·δ_X's signal is the rank-1 term
        δ̄_X[t] · Σ_k W'[d,k] ≈ (Σ_k δ_X[t,k] / K) · Wp_sum_d[d]
    where δ̄_X[t] is the per-token mean of δ_X (carries the skew from
    one-sided activations). Symmetric term for the W side.

    Both dX_sum_t and Wp_sum_d are O(K) commitments per row/col — committed
    ONCE per matmul, not per cell. So this entire model uses *zero* extra
    per-cell K-sums beyond Y'.

    Features per cell (t, d):
        1,
        dX_sum_t[t] · Wp_sum_d[d],
        dW_sum_d[d] · Xp_sum_t[t],
        Wp_sum_d[d] · dX_abs_sum_t[t],   (magnitude proxy)
        Xp_sum_t[t]  · dW_abs_sum_d[d],  (magnitude proxy)
    """
    def fn(*, train, val):
        if "dX_sum_t" not in train:
            raise RuntimeError("R13 needs build_residuals delta-summary features")
        N_tr = len(train["r"])
        N_va = len(val["r"])
        def feats(d, N):
            return np.stack([
                np.ones(N, dtype=np.float64),
                (d["dX_sum_t"].to(torch.float64) * d["Wp_sum_d"].to(torch.float64)).numpy(),
                (d["dW_sum_d"].to(torch.float64) * d["Xp_sum_t"].to(torch.float64)).numpy(),
                (d["Wp_sum_d"].to(torch.float64) * d["dX_abs_sum_t"].to(torch.float64)).numpy(),
                (d["Xp_sum_t"].to(torch.float64)  * d["dW_abs_sum_d"].to(torch.float64)).numpy(),
            ], axis=1)
        Xt = feats(train, N_tr)
        Xv = feats(val, N_va)
        y = train["r"].to(torch.float64).numpy()
        r_hat_np, coefs = _lsq_int_round(Xt, y, Xv)
        return torch.from_numpy(r_hat_np), {
            "coefs": coefs.tolist(),
            "names": ["1",
                      "dX_sum_t · Wp_sum_d",
                      "dW_sum_d · Xp_sum_t",
                      "Wp_sum_d · dX_abs_sum_t",
                      "Xp_sum_t · dW_abs_sum_d"],
        }
    return fn


def model_R13a_W_rank1_only():
    """Just the W-side rank-1 term: dX_sum_t · Wp_sum_d.

    Tests whether the entire W'·δ_X signal can be captured by per-token
    δ_X mean × per-column W' sum — a true zero-K-sum model.
    """
    def fn(*, train, val):
        if "dX_sum_t" not in train:
            raise RuntimeError("R13a needs delta-summary features")
        N_tr = len(train["r"])
        N_va = len(val["r"])
        def feats(d, N):
            return np.stack([
                np.ones(N, dtype=np.float64),
                (d["dX_sum_t"].to(torch.float64) * d["Wp_sum_d"].to(torch.float64)).numpy(),
            ], axis=1)
        Xt = feats(train, N_tr)
        Xv = feats(val, N_va)
        y = train["r"].to(torch.float64).numpy()
        r_hat_np, coefs = _lsq_int_round(Xt, y, Xv)
        return torch.from_numpy(r_hat_np), {
            "coefs": coefs.tolist(),
            "names": ["1", "dX_sum_t · Wp_sum_d"],
        }
    return fn


def model_R14_rank1_plus_WpdX():
    """Rank-1 + the full W'·δ_X term — tests whether per-cell W'·δ_X
    adds anything beyond the rank-1 approximation. If the asymmetry is
    purely rank-1, this should match R11b_W exactly.
    """
    def fn(*, train, val):
        if "Wp_dX" not in train:
            raise RuntimeError("R14 needs --compute-features algebra")
        N_tr = len(train["r"])
        N_va = len(val["r"])
        def feats(d, N):
            return np.stack([
                np.ones(N, dtype=np.float64),
                (d["dX_sum_t"].to(torch.float64) * d["Wp_sum_d"].to(torch.float64)).numpy(),
                (d["dW_sum_d"].to(torch.float64) * d["Xp_sum_t"].to(torch.float64)).numpy(),
                d["Wp_dX"].to(torch.float64).numpy(),
            ], axis=1)
        Xt = feats(train, N_tr)
        Xv = feats(val, N_va)
        y = train["r"].to(torch.float64).numpy()
        r_hat_np, coefs = _lsq_int_round(Xt, y, Xv)
        return torch.from_numpy(r_hat_np), {
            "coefs": coefs.tolist(),
            "names": ["1", "dX_sum_t·Wp_sum_d", "dW_sum_d·Xp_sum_t", "W'·δ_X"],
        }
    return fn


def model_R11_hybrid_byK():
    """Per-family hybrid: R11 (3 terms) on down_proj, R11a (2 terms) elsewhere.

    Motivation: δ_X·δ_W is a K-sum whose variance grows with K. On Qwen2.5-0.5B,
    K(down_proj)=4864 vs ~896 for the other 6 families, so the dropped second-
    order term hurts down_proj 2-3× harder. This hybrid pays the 3-matmul cost
    only on the 1/7 family that needs it.

    Weighted extra-matmul cost (per matmul invocation): R11=3, R11a=2,
    hybrid = (6 × 2 + 1 × 3) / 7 ≈ 2.14 — basically the cost of R11a.
    """
    def fn(*, train, val):
        if "Xp_dW" not in train:
            raise RuntimeError("R11_hybrid_byK needs --compute-features")
        DOWN = 6  # FAMILY_CODE['down']
        fam_tr = train["family"].to(torch.int64).numpy()
        fam_va = val["family"].to(torch.int64).numpy()
        r_hat = np.zeros(len(val["r"]), dtype=np.int64)
        info = {"strategy": "R11 on down_proj (family=6), R11a elsewhere"}

        # R11 (3 terms) on down_proj
        mtr = fam_tr == DOWN
        mva = fam_va == DOWN
        if mtr.sum() > 0 and mva.sum() > 0:
            feats_tr = np.stack([
                np.ones(mtr.sum(), dtype=np.float64),
                train["Xp_dW"].to(torch.float64).numpy()[mtr],
                train["Wp_dX"].to(torch.float64).numpy()[mtr],
                train["dX_dW"].to(torch.float64).numpy()[mtr],
            ], axis=1)
            feats_va = np.stack([
                np.ones(mva.sum(), dtype=np.float64),
                val["Xp_dW"].to(torch.float64).numpy()[mva],
                val["Wp_dX"].to(torch.float64).numpy()[mva],
                val["dX_dW"].to(torch.float64).numpy()[mva],
            ], axis=1)
            y = train["r"].to(torch.float64).numpy()[mtr]
            r_hat_chunk, coefs = _lsq_int_round(feats_tr, y, feats_va)
            r_hat[mva] = r_hat_chunk
            info["down_coefs"] = coefs.tolist()

        # R11a (2 first-order terms) on everything else
        mtr = fam_tr != DOWN
        mva = fam_va != DOWN
        if mtr.sum() > 0 and mva.sum() > 0:
            feats_tr = np.stack([
                np.ones(mtr.sum(), dtype=np.float64),
                train["Xp_dW"].to(torch.float64).numpy()[mtr],
                train["Wp_dX"].to(torch.float64).numpy()[mtr],
            ], axis=1)
            feats_va = np.stack([
                np.ones(mva.sum(), dtype=np.float64),
                val["Xp_dW"].to(torch.float64).numpy()[mva],
                val["Wp_dX"].to(torch.float64).numpy()[mva],
            ], axis=1)
            y = train["r"].to(torch.float64).numpy()[mtr]
            r_hat_chunk, coefs = _lsq_int_round(feats_tr, y, feats_va)
            r_hat[mva] = r_hat_chunk
            info["other_coefs"] = coefs.tolist()

        return torch.from_numpy(r_hat), info
    return fn


def model_R11_fixed_unit_coefs():
    """The proof-friendly version: r̂ = X'·δ_W + W'·δ_X - δ_X·δ_W (no LSQ).

    The algebra GUARANTEES these are the right coefs modulo accumulator
    drift and rounding. Skipping LSQ means no fitted parameters at all —
    fully rule-based. Saves the proof from having to commit to learned
    constants.
    """
    def fn(*, train, val):
        if "Xp_dW" not in train:
            raise RuntimeError("R11_fixed needs --compute-features")
        r_hat_f = (val["Xp_dW"].to(torch.float64)
                   + val["Wp_dX"].to(torch.float64)
                   - val["dX_dW"].to(torch.float64))
        r_hat = torch.round(r_hat_f).to(torch.int64)
        return r_hat, {"rule": "X'·δ_W + W'·δ_X - δ_X·δ_W (no fit)"}
    return fn


def model_R9_outlier_aware():
    """r̂ uses clip counts + absmax features alongside Y' and scale."""
    def fn(*, train, val):
        feats_train = [
            torch.ones(len(train["r"]), dtype=torch.float64).numpy(),
            train["Y_prime"].to(torch.float64).numpy(),
            (train["s_X"].to(torch.float64) * train["s_W"].to(torch.float64)
             * train["Y_prime"].to(torch.float64)).numpy(),
            train["n_clip_X"].to(torch.float64).numpy(),
            train["n_clip_W"].to(torch.float64).numpy(),
            train["x_absmax"].to(torch.float64).numpy(),
            train["w_absmax"].to(torch.float64).numpy(),
        ]
        Xt = np.stack(feats_train, axis=1)
        y = train["r"].to(torch.float64).numpy()
        feats_val = [
            torch.ones(len(val["r"]), dtype=torch.float64).numpy(),
            val["Y_prime"].to(torch.float64).numpy(),
            (val["s_X"].to(torch.float64) * val["s_W"].to(torch.float64)
             * val["Y_prime"].to(torch.float64)).numpy(),
            val["n_clip_X"].to(torch.float64).numpy(),
            val["n_clip_W"].to(torch.float64).numpy(),
            val["x_absmax"].to(torch.float64).numpy(),
            val["w_absmax"].to(torch.float64).numpy(),
        ]
        Xv = np.stack(feats_val, axis=1)
        r_hat_np, coefs = _lsq_int_round(Xt, y, Xv)
        return torch.from_numpy(r_hat_np), {
            "coefs": coefs.tolist(),
            "names": ["1", "Y'", "s·Y'", "n_clip_X", "n_clip_W", "x_absmax", "w_absmax"],
        }
    return fn


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--train-prompts", default="",
                    help="comma-separated prompt indices for training")
    ap.add_argument("--val-prompts", default="",
                    help="comma-separated prompt indices for eval")
    ap.add_argument("--auto-split", type=float, default=0.5,
                    help="if no explicit lists: fraction of prompts to train on")
    ap.add_argument("--models", default="R1,R2,R3,R4,R5,R7,R7b",
                    help="comma-separated model names")
    ap.add_argument("--out", required=True, help="output json with results table")
    ap.add_argument("--subsample-per-matmul-train", type=int, default=0,
                    help="if >0, sample N cells per (matmul_id, prompt) for training only")
    ap.add_argument("--subsample-per-matmul-val", type=int, default=0,
                    help="if >0, sample N cells per (matmul_id, prompt) for val only")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    files = sorted(data_dir.glob("residuals_prompt_*.pt"))
    if not files:
        raise SystemExit(f"No residuals_prompt_*.pt in {data_dir}")
    print(f"Found {len(files)} residual files")

    # split prompts
    prompt_idxs = list(range(len(files)))
    if args.train_prompts:
        tr = set(int(x) for x in args.train_prompts.split(","))
        va = set(int(x) for x in args.val_prompts.split(","))
    else:
        n_tr = max(1, int(args.auto_split * len(prompt_idxs)))
        tr = set(prompt_idxs[:n_tr])
        va = set(prompt_idxs[n_tr:]) if n_tr < len(prompt_idxs) else set([prompt_idxs[-1]])

    train_files = [files[i] for i in sorted(tr)]
    val_files = [files[i] for i in sorted(va)]
    print(f"train prompts={sorted(tr)}; val prompts={sorted(va)}")

    t0 = time.time()
    train = load_dataset(train_files)
    val   = load_dataset(val_files)
    print(f"Loaded train: {len(train['r'])} cells; val: {len(val['r'])} cells "
          f"({time.time() - t0:.1f}s)")

    factories = {
        "R1": model_R1_zero,
        "R2": model_R2_global_constant,
        "R3": model_R3_per_family,
        "R4": model_R4_affine_Y,
        "R5": model_R5_per_d_per_family,
        "R7": model_R7_scale_lut,
        "R7b": model_R7b_affine_in_scale_Yp,
        "R8": model_R8_sign_split,
        "R9": model_R9_outlier_aware,
        "R10": model_R10_delta_summaries,
        "R10b": model_R10b_per_family_delta,
        "R11": model_R11_mixed_dotprods,
        "R11a": model_R11a_first_order_only,
        "R11b_X": model_R11b_only_XpdW,
        "R11b_W": model_R11b_only_WpdX,
        "R11_fixed": model_R11_fixed_unit_coefs,
        "R11_hybrid_byK": model_R11_hybrid_byK,
        "R13": model_R13_rank1,
        "R13a_W": model_R13a_W_rank1_only,
        "R14": model_R14_rank1_plus_WpdX,
        "R11_per_family": model_R11_per_family,
        "R12": model_R12_sign_quantized_delta,
        "R12a_X": model_R12a_mixed_signXp_dW,
        "R12a_W": model_R12a_mixed_Xp_signdW,
    }
    requested = [m.strip() for m in args.models.split(",") if m.strip()]
    results = []
    for m in requested:
        if m not in factories:
            print(f"  skipping unknown model {m!r}")
            continue
        print(f"Fitting {m}...")
        try:
            metrics = fit_and_eval(train, val, m, factories[m]())
        except RuntimeError as e:
            print(f"  {m} failed: {e}")
            continue
        results.append(metrics)
        print(f"  {m}: abs_mean={metrics['abs_mean']:.2f} "
              f"p99={metrics['abs_p99']:.0f} worst={metrics['abs_worst']} "
              f"bit_exact={metrics['bit_exact_frac']:.4f}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({
        "data_dir": str(data_dir),
        "train_prompts": sorted(tr),
        "val_prompts": sorted(va),
        "results": results,
    }, indent=2))
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
