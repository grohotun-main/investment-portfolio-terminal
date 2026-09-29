"""Minimum-variance portfolio optimizer (Risk Simulation Phase 2).

Pure numpy/pandas — no scipy/cvxpy. The repo deliberately avoids scipy (see
parsers/options_pricer.py). Computes the global minimum-variance portfolio under
long-only, fully-invested, per-name-cap, and asset-class-floor constraints via
projected-gradient descent with a Dykstra projection onto the feasible set.

No disk I/O. build_covariance() takes a price frame argument and delegates to
risk_metrics.estimate_covariance so Sigma matches the rest of the Risk
Simulation tab.

Run tests from phase1_build/ with:
    py -m unittest tests.test_min_variance
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

import risk_metrics as rm


def _project_capped_simplex(v: np.ndarray, cap: float) -> np.ndarray:
    """Exact Euclidean projection of v onto {w : sum(w) = 1, 0 <= w_i <= cap}.

    g(tau) = sum(clip(v - tau, 0, cap)) is continuous, non-increasing, and
    piecewise-linear with breakpoints at {v_i} and {v_i - cap}. Between two
    consecutive breakpoints g is linear, so the unique tau with g(tau) = 1 is
    found exactly by bracketing it between adjacent breakpoints and interpolating
    (no iteration; O(n log n) from the sort).
    """
    v = np.asarray(v, dtype=float)
    n = v.size
    if cap * n < 1.0 - 1e-9:
        raise ValueError(f"infeasible capped simplex: cap*n = {cap * n:.6g} < 1")
    cap = min(cap, 1.0)
    bps = np.unique(np.concatenate((v, v - cap)))          # candidate thresholds, ascending
    g = np.array([float(np.clip(v - t, 0.0, cap).sum()) for t in bps])  # non-increasing in tau
    idx = int(np.searchsorted(-g, -1.0))                   # first breakpoint with g <= 1
    if idx <= 0:
        tau = bps[0]
    elif idx >= bps.size:
        tau = bps[-1]
    else:
        a, b, ga, gb = bps[idx - 1], bps[idx], g[idx - 1], g[idx]
        tau = a + (ga - 1.0) * (b - a) / (ga - gb) if ga != gb else a
    return np.clip(v - tau, 0.0, cap)


def _project_feasible(v: np.ndarray, cap: float,
                      groups: list[tuple[np.ndarray, float]],
                      cap_groups: list[tuple[np.ndarray, float]] | None = None,
                      *, iters: int = 300, tol: float = 1e-12) -> np.ndarray:
    """Project v onto the capped simplex intersected with the floor
    half-spaces {sum(w[idx]) >= floor} and class-cap half-spaces
    {sum(w[idx]) <= cap_b} via Dykstra's alternating projection.

    groups: list of (index array, floor). cap_groups: list of (index array,
    class cap) — the mirrored half-space, projected by removing the excess
    uniformly across the bucket. Empty groups and cap_groups -> plain
    capped-simplex projection (one set, returned after the first pass).
    """
    cap_groups = cap_groups or []
    x = np.asarray(v, dtype=float).copy()
    corrections = [np.zeros_like(x)
                   for _ in range(1 + len(groups) + len(cap_groups))]
    for _ in range(iters):
        c_change = 0.0
        # Set 0: capped simplex.
        z = x + corrections[0]
        y = _project_capped_simplex(z, cap)
        new_corr = z - y
        c_change += float(np.sum((new_corr - corrections[0]) ** 2))
        corrections[0] = new_corr
        x = y
        # Sets 1..k: each bucket floor half-space.
        for k, (idx, floor) in enumerate(groups, start=1):
            z = x + corrections[k]
            y = z.copy()
            shortfall = floor - float(y[idx].sum())
            if shortfall > 0.0:
                y[idx] = y[idx] + shortfall / idx.size
            new_corr = z - y
            c_change += float(np.sum((new_corr - corrections[k]) ** 2))
            corrections[k] = new_corr
            x = y
        # Sets k+1..: each bucket class-cap half-space (mirror of the floor
        # step: exact Euclidean projection removes the excess uniformly).
        for k, (idx, cap_b) in enumerate(cap_groups, start=1 + len(groups)):
            z = x + corrections[k]
            y = z.copy()
            excess = float(y[idx].sum()) - cap_b
            if excess > 0.0:
                y[idx] = y[idx] - excess / idx.size
            new_corr = z - y
            c_change += float(np.sum((new_corr - corrections[k]) ** 2))
            corrections[k] = new_corr
            x = y
        if c_change < tol:
            break
    return x


def _check_feasibility(n: int, cap: float,
                       group_sizes: dict[str, int],
                       floors: dict[str, float],
                       caps: dict[str, float] | None = None) -> str | None:
    """Return a human-readable reason the constraints are infeasible, or None.

    Bounds view: each bucket b has lower_b = floor_b (0 if unfloored) and
    upper_b = min(cap_b, size_b * cap) (cap_b = 1 if uncapped). The three
    checks below (lower <= upper per bucket, sum of lowers <= 1, sum of
    uppers >= 1) are necessary AND sufficient for a non-empty feasible set,
    so every infeasibility has a named message and the solver never runs on
    an empty set. With no class caps this reduces exactly to the pre-O1a
    checks (same messages).
    """
    caps = caps or {}
    if cap * n < 1.0 - 1e-9:
        return (f"Per-name cap {cap * 100:.0f}% x {n} holdings only reaches "
                f"{cap * n * 100:.0f}% - can't total 100%. Raise the cap.")
    total = sum(floors.values())
    if total > 1.0 + 1e-9:
        return (f"Asset-class floors sum to {total * 100:.0f}% (>100%). "
                "Lower a floor.")
    for bucket, floor in floors.items():
        cap_b = caps.get(bucket)
        if cap_b is not None and floor > cap_b + 1e-9:
            return (f"{bucket.replace('_', ' ').title()} floor "
                    f"{floor * 100:.0f}% exceeds its {cap_b * 100:.0f}% cap "
                    "- lower the floor or raise the cap.")
        capacity = group_sizes.get(bucket, 0) * cap
        if floor > capacity + 1e-9:
            return (f"{bucket.replace('_', ' ').title()} floor "
                    f"{floor * 100:.0f}% exceeds what "
                    f"{group_sizes.get(bucket, 0)} holding(s) can hold at the "
                    f"{cap * 100:.0f}% cap ({capacity * 100:.0f}%).")
    if caps:
        in_capped = sum(group_sizes.get(b, 0) for b in caps)
        reach = (sum(min(c, group_sizes.get(b, 0) * cap)
                     for b, c in caps.items())
                 + (n - in_capped) * cap)
        if reach < 1.0 - 1e-9:
            return ("Class caps leave too little room: caps and uncapped "
                    f"capacity reach only {reach * 100:.0f}% - can't total "
                    "100%. Raise a cap.")
    return None


def solve_min_variance(cov: pd.DataFrame, *, name_cap: float,
                       class_of: dict[str, str],
                       class_floors: dict[str, float],
                       class_caps: dict[str, float] | None = None,
                       mu: pd.Series | None = None,
                       risk_aversion: float = 1.0,
                       max_iter: int = 10000, tol: float = 1e-6) -> dict:
    """Global minimum-variance portfolio under long-only, fully-invested,
    per-name-cap, asset-class-floor, and asset-class-cap constraints.

    class_of values must be floor buckets ("equity"/"fixed_income"/"gold"/"other");
    call to_floor_bucket() on raw asset_class strings before passing.
    class_caps maps bucket -> max class weight (decimals); entries >= 1.0 are
    inert, 0.0 legally excludes the class.

    Objective: minimize w'Sigma*w (the shipped behaviour, mu=None) or the
    scalarized mean-variance objective risk_aversion*w'Sigma*w - w'mu when
    `mu` (annual expected returns, indexed like cov) is given. Sweeping
    risk_aversion traces the efficient frontier; the FEASIBLE SET is identical
    either way, so mu=None keeps today's exact code path bit-for-bit.

    Returns {weights: pd.Series|None, vol: float, converged: bool,
             feasible: bool, binding: {name_cap: [...], floors: {bucket: bool},
             class_caps: {bucket: bool}}, error: str|None}.
    """
    syms = list(cov.index)
    n = len(syms)
    empty_binding: dict = {"name_cap": [], "floors": {}, "class_caps": {}}
    sigma = np.asarray(cov.to_numpy(), dtype=float) if n else np.empty((0, 0))
    if n == 0 or not np.all(np.isfinite(sigma)):
        return {"weights": None, "vol": float("nan"), "converged": False,
                "feasible": False, "binding": empty_binding,
                "error": "Covariance is empty or contains non-finite values."}

    cap = float(name_cap)
    floors = {b: float(f) for b, f in (class_floors or {}).items()
              if f and float(f) > 0.0}
    caps_d = {b: float(c) for b, c in (class_caps or {}).items()
              if c is not None and float(c) < 1.0}
    all_buckets = set(floors) | set(caps_d)
    group_idx = {b: np.array([i for i, s in enumerate(syms)
                              if class_of.get(s) == b], dtype=int)
                 for b in all_buckets}
    err = _check_feasibility(
        n, cap, {b: group_idx[b].size for b in all_buckets}, floors, caps_d)
    if err:
        return {"weights": None, "vol": float("nan"), "converged": False,
                "feasible": False, "binding": empty_binding, "error": err}
    groups = [(group_idx[b], floors[b]) for b in floors]
    cap_groups = [(group_idx[b], caps_d[b]) for b in caps_d]

    evals = np.linalg.eigvalsh(sigma)
    lam_min, lam_max = float(evals[0]), float(evals[-1])
    if lam_min < -1e-10:
        return {"weights": None, "vol": float("nan"), "converged": False,
                "feasible": False, "binding": empty_binding,
                "error": "Covariance is not positive-semidefinite."}
    if mu is None:
        eta = 1.0 / (2.0 * lam_max) if lam_max > 0 else 1.0
        def grad(y):                                    # shipped objective
            return 2.0 * (sigma @ y)
    else:
        lam = float(risk_aversion)
        if not (math.isfinite(lam) and lam > 0.0):
            return {"weights": None, "vol": float("nan"), "converged": False,
                    "feasible": False, "binding": empty_binding,
                    "error": "risk_aversion must be a positive finite number."}
        mu_v = np.asarray(pd.Series(mu).reindex(syms).to_numpy(dtype=float))
        if not np.all(np.isfinite(mu_v)):
            return {"weights": None, "vol": float("nan"), "converged": False,
                    "feasible": False, "binding": empty_binding,
                    "error": "Expected returns are missing or non-finite for "
                             "one or more holdings."}
        eta = 1.0 / (2.0 * lam * lam_max) if lam_max > 0 else 1.0
        def grad(y):                    # d/dw [lam*w'Sw - w'mu]
            return 2.0 * lam * (sigma @ y) - mu_v

    # Accelerated projected gradient (FISTA) with O'Donoghue-Candes adaptive
    # gradient restart. This is a convex QP, so FISTA reaches the SAME unique
    # optimum as plain projected-gradient, but in ~sqrt(kappa) rather than
    # ~kappa iterations -- decisive on ill-conditioned real-book covariances
    # (kappa up to ~1e5). The restart keeps the iterate effectively monotone, so
    # the max-weight-change stopping test stays reliable (un-restarted FISTA can
    # ripple and stop prematurely at a non-optimal point).
    x = _project_feasible(np.full(n, 1.0 / n), cap, groups, cap_groups)
    y = x.copy()
    t = 1.0
    converged = False
    for _ in range(max_iter):
        x_new = _project_feasible(y - eta * grad(y), cap, groups, cap_groups)
        if np.max(np.abs(x_new - x)) < tol:
            x = x_new
            converged = True
            break
        if float((y - x_new) @ (x_new - x)) > 0.0:      # momentum opposed the step -> restart
            t = 1.0
            y = x_new
        else:
            t_new = 0.5 * (1.0 + math.sqrt(1.0 + 4.0 * t * t))
            y = x_new + ((t - 1.0) / t_new) * (x_new - x)
            t = t_new
        x = x_new
    w = x

    weights = pd.Series(w, index=syms)
    vol = float(np.sqrt(max(0.0, float(w @ sigma @ w))))
    # Binding = the optimum sits AT the constraint boundary (within tol).
    # Feasibility already guarantees the inequality side, so floors test from
    # above (<= floor + tol) and caps from below (>= cap - tol).
    binding = {
        "name_cap": [syms[i] for i in range(n) if w[i] >= cap - 1e-6],
        "floors": {b: bool(w[group_idx[b]].sum() <= floors[b] + 1e-6)
                   for b in floors},
        "class_caps": {b: bool(w[group_idx[b]].sum() >= caps_d[b] - 1e-6)
                       for b in caps_d},
    }
    return {"weights": weights, "vol": vol, "converged": converged,
            "feasible": True, "binding": binding, "error": None}


_EQUITY_CLASSES = {"equity", "equity_stock", "equity_etf",
                   "tax_loss_harvesting"}
_FIXED_CLASSES = {"fixed_income", "cash", "bond"}
_GOLD_CLASSES = {"gold", "commodity", "commodity_etf"}


def to_floor_bucket(asset_class: str) -> str:
    """Collapse a raw asset_class into a floor bucket.

    Unknown / proxy classes fall through to 'other' (no floor applied).
    """
    a = (asset_class or "").strip().lower()
    if a in _EQUITY_CLASSES:
        return "equity"
    if a in _FIXED_CLASSES:
        return "fixed_income"
    if a in _GOLD_CLASSES:
        return "gold"
    return "other"


def anchored_defaults(weights: pd.Series,
                      class_of: dict[str, str]) -> dict:
    """Slider defaults anchored to the current book, rounded so the current
    portfolio is feasible at the defaults: the per-name cap rounds UP to the
    next whole percent (current largest <= cap); the equity floor rounds DOWN
    (current equity >= floor).

    class_of maps symbol -> floor bucket ("equity"/"fixed_income"/"gold"/"other");
    apply to_floor_bucket() to raw asset_class values before calling.
    Raw asset_class strings will not match and yield a 0 equity floor.
    """
    if weights is None or len(weights) == 0:
        return {"cap_default": 1.0, "equity_floor_default": 0.0}
    held = weights[weights > 0]
    largest = float(held.max()) if len(held) else 1.0
    cap_default = min(1.0, math.ceil(largest * 100.0) / 100.0)
    equity = sum(float(v) for s, v in weights.items()
                 if class_of.get(s) == "equity")
    equity_floor_default = math.floor(equity * 100.0) / 100.0
    return {"cap_default": cap_default,
            "equity_floor_default": equity_floor_default}


def build_covariance(daily_prices: pd.DataFrame, symbols: list[str], *,
                     min_overlap_days: int = 252,
                     estimator: str = "ewma_lw") -> dict:
    """Annualized covariance over the modelable subset of `symbols`.

    A holding needs >= min_overlap_days + 1 of its OWN price history to be
    estimated; younger names (or names with no price column) are excluded and
    returned in `excluded` so the caller can hold them at their current weight
    rather than letting one young name collapse the common-overlap window.
    Among the mature names, takes the common-overlap window (dropna) and
    delegates to risk_metrics.estimate_covariance.

    Returns {cov: pd.DataFrame|None, n_days: int, excluded: list[str],
             error: str|None}.
    """
    in_universe = list(symbols)
    priced = [s for s in in_universe if s in daily_prices.columns]
    mature = [s for s in priced
              if int(daily_prices[s].notna().sum()) >= min_overlap_days + 1]
    excluded = [s for s in in_universe if s not in mature]
    if not mature:
        return {"cov": None, "n_days": 0, "excluded": excluded,
                "error": (f"No holdings have >= {min_overlap_days + 1} trading "
                          "days of history to model.")}
    px = daily_prices[mature].sort_index().dropna(how="any")
    if len(px) < min_overlap_days + 1:
        return {"cov": None, "n_days": 0, "excluded": excluded,
                "error": (f"Insufficient overlap among modelable holdings: "
                          f"{len(px)} trading days (need >= "
                          f"{min_overlap_days + 1}).")}
    rets = px.pct_change().dropna(how="any")
    res = rm.estimate_covariance(rets, estimator=estimator)
    return {"cov": res["cov"], "n_days": int(res["n_days"]),
            "excluded": excluded, "error": None}


def suggest_min_variance_grid(daily_prices: pd.DataFrame, weights: pd.Series,
                              class_of: dict[str, str], *, name_cap: float,
                              class_floors: dict[str, float],
                              class_caps: dict[str, float] | None = None,
                              mu: pd.Series | None = None,
                              risk_aversion: float = 1.0,
                              min_overlap_days: int = 252,
                              covres: dict | None = None) -> dict:
    """Build Sigma over the modelable holdings, solve, and return the grid
    payload + a status message. Holdings too young to model (see
    build_covariance) are held at their current weight; the remaining budget is
    optimized across the rest.

    class_of values must be floor buckets ("equity"/"fixed_income"/"gold"/
    "other"); call to_floor_bucket() on raw asset_class strings before passing.

    class_caps maps bucket -> max class weight on the FULL book (decimals);
    transformed into the mature sleeve the same way floors are. A cap the
    young sleeve alone already exceeds is an honest kind:'error'.

    Pass a prebuilt `covres` (a build_covariance result for the SAME
    daily_prices / weights universe / min_overlap_days) to skip the internal
    Σ build — the cap-sweep tracer builds Σ once and reuses it per solve.

    `mu` (annual expected returns over the full book) switches the sleeve solve
    to the scalarized objective. Because the sleeve is solved in u-space and
    scaled by mature_budget (mb), the solver is called at effective lambda
    = risk_aversion * mb -- the mb factored out of the objective. The young
    sleeve's expected return is a constant the solver never sees; recover it by
    dotting the returned full-book new_pct with mu.

    Returns {new_pct: dict[symbol -> percent]|None, kind: 'success'|'error',
             message: str, vol: float, cur_vol: float, converged: bool}.
    """
    fail = {"new_pct": None, "vol": float("nan"), "cur_vol": float("nan"),
            "converged": False}
    if covres is None:
        covres = build_covariance(daily_prices, list(weights.index),
                                  min_overlap_days=min_overlap_days)
    if covres["error"]:
        return {**fail, "kind": "error", "message": covres["error"]}
    cov = covres["cov"]
    mature = list(cov.index)

    young_w = {s: float(weights[s]) for s in covres["excluded"]
               if s in weights.index}
    young_budget = float(sum(young_w.values()))
    mature_budget = 1.0 - young_budget
    if mature_budget <= 1e-9:
        return {**fail, "kind": "error",
                "message": ("Almost all weight is in holdings too new to "
                            "model; nothing to optimize.")}

    # Solve the mature sleeve as a sub-portfolio summing to 1 (u), then scale
    # by mature_budget. Transform cap/floors into u-space so the post-scale
    # full-portfolio constraints hold (subtract the young sleeve's share).
    eff_cap = min(1.0, float(name_cap) / mature_budget)
    young_in_bucket: dict[str, float] = {}
    for s, w in young_w.items():
        b = class_of.get(s)
        young_in_bucket[b] = young_in_bucket.get(b, 0.0) + w
    eff_floors = {
        b: max(0.0, (float(f) - young_in_bucket.get(b, 0.0)) / mature_budget)
        for b, f in (class_floors or {}).items()
    }
    eff_caps: dict[str, float] = {}
    for b, c in (class_caps or {}).items():
        if c is None or float(c) >= 1.0:
            continue                                  # inert full-book cap
        c = float(c)
        young_b = young_in_bucket.get(b, 0.0)
        if young_b > c + 1e-9:
            return {**fail, "kind": "error",
                    "message": (f"Holdings too new to model already hold "
                                f"{young_b * 100:.0f}% of "
                                f"{b.replace('_', ' ')} - above its "
                                f"{c * 100:.0f}% cap. Raise the cap.")}
        eff = max(0.0, (c - young_b) / mature_budget)
        if eff < 1.0 - 1e-12:
            eff_caps[b] = eff          # eff >= 1 cannot bind inside the sleeve
    mature_class_of = {s: class_of.get(s) for s in mature}

    eff_mu = None if mu is None else pd.Series(mu, dtype=float).reindex(mature)
    res = solve_min_variance(cov, name_cap=eff_cap, class_of=mature_class_of,
                             class_floors=eff_floors, class_caps=eff_caps,
                             mu=eff_mu,
                             risk_aversion=float(risk_aversion) * mature_budget)
    if not res["feasible"]:
        return {**fail, "kind": "error", "message": res["error"]}

    u = res["weights"].to_numpy(dtype=float)
    u_sum = float(u.sum())
    if u_sum > 0:
        u = u / u_sum
    w_mature = u * mature_budget
    s_mat = cov.to_numpy(dtype=float)
    sim_vol = float(np.sqrt(max(0.0, float(w_mature @ s_mat @ w_mature))))
    w_cur = weights.reindex(cov.index).fillna(0.0).to_numpy(dtype=float)
    cur_vol = float(np.sqrt(max(0.0, float(w_cur @ s_mat @ w_cur))))

    new_pct = {s: young_w[s] * 100.0 for s in young_w}
    for s, w in zip(mature, w_mature):
        new_pct[s] = float(w) * 100.0

    caps = ", ".join(res["binding"]["name_cap"][:3]) or "none"
    floors_on = ", ".join(b.replace("_", " ") for b, on
                          in res["binding"]["floors"].items() if on) or "none"
    cc = res["binding"]["class_caps"]
    cc_on = ", ".join(b.replace("_", " ") for b, on in cc.items() if on)
    cc_clause = f"; class caps {cc_on or 'none'}" if cc else ""
    approx = "" if res["converged"] else " (approx - did not fully converge)"
    held = ""
    if young_w:
        held = (f" Held {len(young_w)} holding(s) too new to model at current "
                f"weight: {', '.join(sorted(young_w))}.")
    message = (f"Min-variance loaded: vol ~ {sim_vol * 100:.1f}% "
               f"(current {cur_vol * 100:.1f}%). Binding: cap on {caps}; "
               f"floors {floors_on}{cc_clause}{approx}.{held} "
               "Review/tweak below, then Run.")
    return {"new_pct": new_pct, "kind": "success", "message": message,
            "vol": sim_vol, "cur_vol": cur_vol, "converged": res["converged"]}
