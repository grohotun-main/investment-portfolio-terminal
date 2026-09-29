"""Equal-risk-contribution (risk-parity) portfolio optimizer
(Risk Simulation Phase 3).

Pure numpy/pandas — no scipy/cvxpy (the repo deliberately avoids scipy; see
parsers/options_pricer.py and parsers/min_variance.py). Computes the long-only,
fully-invested portfolio whose holdings contribute EQUALLY to portfolio
variance, via cyclic coordinate descent on the log-barrier ERC form. Supports
an optional per-name cap enforced by active-set pinning; no class floors.

No disk I/O. Reuses min_variance.build_covariance (objective-agnostic) so Sigma
matches the rest of the Risk Simulation tab and young holdings are handled the
same way as the min-variance suggest.

Run tests from phase1_build/ with:
    py -m unittest tests.test_risk_parity
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from min_variance import build_covariance, _check_feasibility


def _solve_erc_unconstrained(sigma: np.ndarray, *,
                             budgets: np.ndarray | None = None,
                             max_iter: int = 10000, tol: float = 1e-10):
    """Unconstrained long-only ERC on a covariance ndarray via cyclic coordinate
    descent. Returns (weights summing to 1, converged, rc_dispersion).

    budgets: per-name STRICTLY POSITIVE risk shares, length n (raises
    ValueError otherwise; renormalized to sum 1); None = uniform — today's
    equal-RC path, float ops unchanged. With budgets, the fixed point has
    rc_i/Σrc = b_i and rc_dispersion is the target-relative metric
    max|share_i/b_i − 1| (the uniform path keeps the original (max−min)/mean
    metric byte-for-byte)."""
    n = sigma.shape[0]
    diag = np.diag(sigma).copy()
    inv_n = 1.0 / n
    b_vec = None
    if budgets is not None:
        b_vec = np.asarray(budgets, dtype=float)
        if b_vec.shape != (n,) or not np.all(b_vec > 0.0):
            raise ValueError("budgets must be length-n and strictly positive")
        b_vec = b_vec / b_vec.sum()
    y = 1.0 / np.sqrt(diag)            # inverse-vol warm start
    converged = False
    rc_disp = float("nan")
    for _ in range(max_iter):
        for i in range(n):
            b = float(sigma[i] @ y) - diag[i] * y[i]   # (Σy)ᵢ − Σᵢᵢ·yᵢ
            target = inv_n if b_vec is None else b_vec[i]
            y[i] = ((-b + math.sqrt(b * b + 4.0 * diag[i] * target))
                    / (2.0 * diag[i]))
        w = y / y.sum()
        rc = w * (sigma @ w)
        if b_vec is None:
            mean_rc = float(rc.mean())
            rc_disp = (float((rc.max() - rc.min()) / mean_rc)
                       if mean_rc > 0 else 0.0)
        else:
            total = float(rc.sum())
            rc_disp = (float(np.max(np.abs(rc / total / b_vec - 1.0)))
                       if total > 0 else 0.0)
        if rc_disp < tol:
            converged = True
            break
    return y / y.sum(), converged, rc_disp


def _erc_name_pinned(sigma: np.ndarray, cap: float, *,
                     budgets: np.ndarray | None = None,
                     max_iter: int = 10000, tol: float = 1e-10):
    """The per-name active-set pinning loop (shipped in Phase 3), extracted
    for reuse by the O1b bucket-pinning wrapper: long-only ERC summing to 1
    with every name capped at `cap`. Returns (weights, converged,
    rc_dispersion over the free names, pinned index set).

    budgets: per-name risk shares over the FULL index; the free sub-solve
    receives the free names' budgets renormalized; None = uniform
    (bit-identical)."""
    n = sigma.shape[0]
    pinned: set[int] = set()
    w = np.zeros(n)
    converged, rc_disp = True, 0.0
    for _ in range(n + 1):             # <= n pinning rounds; bounded for safety
        free = [i for i in range(n) if i not in pinned]
        if not free:
            break  # unreachable when cap*n >= 1 (the last free name always fits)
        budget = 1.0 - cap * len(pinned)
        sub = sigma[np.ix_(free, free)]
        sub_budgets = None
        if budgets is not None:
            bf = budgets[free]
            sub_budgets = bf / bf.sum()
        u, converged, rc_disp = _solve_erc_unconstrained(
            sub, budgets=sub_budgets, max_iter=max_iter, tol=tol)
        w_free = u * budget
        over = [free[k] for k in range(len(free)) if w_free[k] > cap + 1e-9]
        if not over:
            for k, idx in enumerate(free):
                w[idx] = w_free[k]
            break
        pinned.update(over)
        for idx in over:
            w[idx] = cap
    return w, converged, rc_disp, pinned


def solve_risk_parity(cov: pd.DataFrame, *, name_cap: float = 1.0,
                      class_of: dict[str, str] | None = None,
                      class_caps: dict[str, float] | None = None,
                      class_risk_budgets: dict[str, float] | None = None,
                      max_iter: int = 10000, tol: float = 1e-10) -> dict:
    """Long-only, fully-invested equal-risk-contribution (ERC) portfolio with an
    optional per-name cap.

    Each unpinned holding contributes equally to portfolio variance (cyclic
    coordinate descent on the log-barrier form). The cap is enforced by
    active-set pinning: solve unconstrained ERC, pin any name whose weight exceeds
    `name_cap` at the cap, re-solve ERC on the free remainder scaled to the
    leftover budget, repeat until no free name exceeds the cap. Pinned names sit at
    the cap; free names keep equal risk contribution. `name_cap=1.0` (default) pins
    nothing -> pure unconstrained ERC.

    The free-set re-solve uses the free sub-covariance, ignoring pinned names'
    cross-covariance with the free set — exact when capped names are uncorrelated
    with the rest (the dominant-near-riskless-asset case), the same modelling choice
    the young-holding mature-sleeve solve makes.

    class_of maps symbol -> floor bucket; class_caps maps bucket -> max class
    weight (decimals, entries >= 1.0 inert, 0.0 legally excludes the class).
    binding gains class_caps: {bucket: bool}.

    Class caps are enforced one level up by the same pattern: solve the free
    set, pin every bucket whose sum exceeds its cap AT the cap, allocate
    within each pinned bucket by per-name-pinned ERC on its own sub-Σ,
    re-solve outside on the leftover budget, repeat (bounded by the bucket
    count). The outside re-solve ignores cross-covariance with pinned
    buckets — exact when the capped sleeve is near-uncorrelated with the
    rest, the case this feature exists for.

    class_risk_budgets maps bucket -> share of portfolio risk (decimals,
    ENTERED buckets only, each in (0, 1]); entered classes split their share
    equally among members, the remainder spreads equally per name across
    unset classes (plain ERC there). Budgets shape the objective; caps
    remain constraints — free-set solves renormalize the free names'
    budgets, and within-bucket solves stay uniform because a bucket is one
    class (equal b_i). Result gains budgets: pd.Series|None; binding gains
    budget_limited: budgeted buckets pinned by their weight cap.

    Returns {weights: pd.Series|None, vol: float, converged: bool, feasible: bool,
             rc_dispersion: float (over the free names),
             binding: {name_cap: [...], class_caps: {bucket: bool}},
             error: str|None}.
    """
    syms = list(cov.index)
    n = len(syms)
    bail = {"weights": None, "vol": float("nan"), "converged": False,
            "feasible": False, "rc_dispersion": float("nan"),
            "binding": {"name_cap": [], "class_caps": {}, "budget_limited": []},
            "budgets": None}
    sigma = np.asarray(cov.to_numpy(), dtype=float) if n else np.empty((0, 0))
    if n == 0 or not np.all(np.isfinite(sigma)):
        return {**bail,
                "error": "Covariance is empty or contains non-finite values."}
    cap = float(name_cap)
    caps_d = {b: float(c) for b, c in (class_caps or {}).items()
              if c is not None and float(c) < 1.0}
    cof = class_of or {}
    group_idx = {b: np.array([i for i, s in enumerate(syms)
                              if cof.get(s) == b], dtype=int)
                 for b in caps_d}
    err = _check_feasibility(n, cap,
                             {b: group_idx[b].size for b in caps_d},
                             {}, caps_d)
    if err:
        return {**bail, "error": err}
    budgets_in = {b: float(v)
                  for b, v in (class_risk_budgets or {}).items()}
    b_vec = None
    if budgets_in:
        if any(not math.isfinite(v) or v <= 0.0 for v in budgets_in.values()):
            return {**bail, "error": ("Risk budgets must be positive - use "
                                      "a 0% class cap to exclude a class.")}
        total_b = sum(budgets_in.values())
        if total_b > 1.0 + 1e-9:
            return {**bail,
                    "error": (f"Risk budgets sum to {total_b * 100:.0f}% "
                              "(>100%). Lower a budget.")}
        bidx = {b: np.array([i for i, s in enumerate(syms)
                             if cof.get(s) == b], dtype=int)
                for b in budgets_in}
        for b in budgets_in:
            if bidx[b].size == 0:
                return {**bail,
                        "error": (f"No modelable holdings in "
                                  f"{b.replace('_', ' ')} to carry its "
                                  "risk budget.")}
        in_budgeted = {int(i) for b in budgets_in for i in bidx[b]}
        unset = [i for i in range(n) if i not in in_budgeted]
        remainder = 1.0 - total_b
        if remainder > 1e-9 and not unset:
            return {**bail,
                    "error": (f"Risk budgets cover every class but sum to "
                              f"{total_b * 100:.0f}% - make them total 100% "
                              "or leave a class unset.")}
        if remainder <= 1e-9 and unset:
            labels = ", ".join(sorted({cof.get(syms[i]) or "other"
                                       for i in unset})).replace("_", " ")
            return {**bail,
                    "error": (f"Risk budgets total 100% but {labels} carry "
                              "none - lower a budget to leave them room.")}
        b_vec = np.zeros(n)
        for b, share in budgets_in.items():
            b_vec[bidx[b]] = share / bidx[b].size
        if unset:
            b_vec[unset] = remainder / len(unset)
        b_vec = b_vec / b_vec.sum()
    evals = np.linalg.eigvalsh(sigma)
    if float(evals[0]) <= 1e-10:
        return {**bail,
                "error": ("Covariance is not positive-definite; risk-parity "
                          "needs a strictly PD matrix.")}

    if caps_d:
        bucket_pinned: dict[str, float] = {}
        w = np.zeros(n)
        converged, rc_disp = True, 0.0
        for _ in range(len(caps_d) + 1):
            pinned_set = {int(i) for b in bucket_pinned
                          for i in group_idx[b]}
            free = [i for i in range(n) if i not in pinned_set]
            budget = 1.0 - sum(bucket_pinned.values())
            # Within-bucket allocation for every pinned bucket: the same
            # per-name-pinned ERC on the bucket's own sub-Σ, scaled to its
            # cap (a 0-cap bucket is simply zeroed). Ignores cross-Σ with
            # the outside — the same modelling choice the per-name pinning
            # makes, exact when the capped sleeve is near-uncorrelated.
            for b, cap_b in bucket_pinned.items():
                idx = group_idx[b]
                if cap_b <= 1e-12 or idx.size == 0:
                    w[idx] = 0.0
                    continue
                sub_b = sigma[np.ix_(idx, idx)]
                u_b, _, _, _ = _erc_name_pinned(
                    sub_b, min(1.0, cap / cap_b),
                    max_iter=max_iter, tol=tol)
                w[idx] = u_b * cap_b
            if not free or budget <= 1e-12:
                break
            sub = sigma[np.ix_(free, free)]
            sub_budgets = None
            if b_vec is not None:
                bf = b_vec[free]
                sub_budgets = bf / bf.sum()
            u, converged, rc_disp, _sub_pin = _erc_name_pinned(
                sub, min(1.0, cap / budget), budgets=sub_budgets,
                max_iter=max_iter, tol=tol)
            for k, i in enumerate(free):
                w[i] = u[k] * budget
            over = [b for b in caps_d if b not in bucket_pinned
                    and float(w[group_idx[b]].sum()) > caps_d[b] + 1e-9]
            if not over:
                break
            for b in over:                 # pin ALL over-cap buckets
                bucket_pinned[b] = caps_d[b]

        weights = pd.Series(w, index=syms)
        vol = float(np.sqrt(max(0.0, float(w @ sigma @ w))))
        binding = {
            "name_cap": [syms[i] for i in range(n) if w[i] >= cap - 1e-6],
            "class_caps": {b: bool(w[group_idx[b]].sum()
                                   >= caps_d[b] - 1e-6)
                           for b in caps_d},
            "budget_limited": sorted(b for b in budgets_in
                                     if b in bucket_pinned),
        }
        return {"weights": weights, "vol": vol, "converged": converged,
                "feasible": True, "rc_dispersion": rc_disp,
                "binding": binding, "error": None,
                "budgets": (pd.Series(b_vec, index=syms)
                            if b_vec is not None else None)}

    w, converged, rc_disp, pinned = _erc_name_pinned(
        sigma, cap, budgets=b_vec, max_iter=max_iter, tol=tol)

    weights = pd.Series(w, index=syms)
    vol = float(np.sqrt(max(0.0, float(w @ sigma @ w))))
    binding = {"name_cap": [syms[i] for i in sorted(pinned)],
               "class_caps": {}, "budget_limited": []}
    return {"weights": weights, "vol": vol, "converged": converged,
            "feasible": True, "rc_dispersion": rc_disp, "binding": binding,
            "error": None,
            "budgets": (pd.Series(b_vec, index=syms)
                        if b_vec is not None else None)}


def full_sigma_rc_dispersion(w: pd.Series, sigma: pd.DataFrame,
                             free_names) -> float:
    """Coefficient of variation (population std / mean) of risk contributions
    across ``free_names``, computed on the FULL covariance with the final
    weights.

    The ERC optimizer's own ``rc_dispersion`` is the reduced free-set
    sub-problem metric: it ignores the capped/pinned names' cross-covariance,
    so it can read ~0 even when the realized contributions across the
    equalized names still spread a couple percent. This reports the spread a
    user actually gets (WSB-4). Returns 0.0 with fewer than two free names or
    a non-positive mean contribution.
    """
    w = pd.Series(w).reindex(sigma.index).fillna(0.0).astype(float)
    rc = w.to_numpy() * (sigma.to_numpy() @ w.to_numpy())
    free = [n for n in sigma.index if n in set(free_names)]
    rc_free = pd.Series(rc, index=sigma.index).loc[free].to_numpy()
    if len(rc_free) < 2:
        return 0.0
    mean = float(rc_free.mean())
    if mean <= 0:
        return 0.0
    return float(np.std(rc_free) / mean)


def _fmt_realized(realized_pct: float) -> str:
    """Realized RC share for the budgets banner. Clamps the signed-zero
    window (|x| < 0.05 would print '-0.0%') and names a genuinely negative
    share — a bare minus number reads as a broken figure, not as 'this
    sleeve currently offsets book risk'."""
    if abs(realized_pct) < 0.05:
        return "0.0%"
    if realized_pct < 0:
        return f"{realized_pct:.1f}% — risk contribution currently negative"
    return f"{realized_pct:.1f}%"


def _gap_fragment(shares: pd.Series, b_ser: pd.Series | None,
                  free_names: list[str]) -> str:
    """' (max gap to target X.XX%)' over free names with strictly positive
    realized share, or '' when none qualify (a non-positive share makes the
    shares/target ratio meaningless, not merely large)."""
    names = [m for m in free_names if float(shares.get(m, 0.0)) > 0.0]
    if b_ser is None or not names:
        return ""
    gap = float(np.max(np.abs(
        shares[names].to_numpy() / b_ser[names].to_numpy() - 1.0))) * 100.0
    return f" (max gap to target {gap:.2f}%)"


def suggest_risk_parity_grid(daily_prices: pd.DataFrame, weights: pd.Series, *,
                             name_cap: float = 1.0,
                             class_of: dict[str, str] | None = None,
                             class_caps: dict[str, float] | None = None,
                             class_risk_budgets: dict[str, float] | None = None,
                             min_overlap_days: int = 252,
                             covres: dict | None = None) -> dict:
    """Build Σ over the modelable holdings (reusing min_variance.build_covariance),
    hold holdings too young to model at their current weight, run ERC on the
    mature sleeve, scale back, and return the grid payload + a status message.

    The per-name `name_cap` is transformed into the mature sleeve as
    `eff_cap = min(1, name_cap / mature_budget)` (the same transform the
    min-variance suggest uses), so the full-book cap still holds after the mature
    weights are scaled by `mature_budget = 1 − Σ(young weights)`. No class floors.

    class_caps maps bucket -> max class weight on the FULL book (decimals);
    transformed into the mature sleeve the same way the min-variance suggest
    transforms floors and caps. A cap the young sleeve alone already exceeds
    is an honest kind:'error'.

    class_risk_budgets maps bucket -> share of MODELED risk (entered buckets
    only; no young transform - young holdings are held at weight outside the
    budget math; a bucket whose only members are young errors honestly).

    Pass a prebuilt `covres` (a build_covariance result for the SAME
    daily_prices / weights universe / min_overlap_days) to skip the internal
    Σ build — the cap-sweep tracer builds Σ once and reuses it per solve.

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

    eff_cap = min(1.0, float(name_cap) / mature_budget)
    cof = class_of or {}
    young_in_bucket: dict[str, float] = {}
    for s, yw in young_w.items():
        b = cof.get(s)
        young_in_bucket[b] = young_in_bucket.get(b, 0.0) + yw
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
    mature_cof = {s: cof.get(s) for s in mature}
    res = solve_risk_parity(cov, name_cap=eff_cap, class_of=mature_cof,
                            class_caps=eff_caps,
                            class_risk_budgets=class_risk_budgets)
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

    capped = res["binding"]["name_cap"]
    cc = res["binding"]["class_caps"]
    bucket_capped = {m for m in mature
                    if cc.get(mature_cof.get(m), False)} if cc else set()
    # Honest spread on the FULL Sigma across the equalized (uncapped) names —
    # res["rc_dispersion"] is the reduced sub-problem metric and reads ~0
    # because it ignores the capped names' cross-covariance (WSB-4).
    _free_names = [m for m in mature
                   if m not in capped and m not in bucket_capped]
    disp_pct = full_sigma_rc_dispersion(
        pd.Series(w_mature, index=cov.index), cov, _free_names) * 100.0
    n_equalized = len(_free_names)
    cap_clause = (f" Capped at the per-name limit: {', '.join(sorted(capped))}."
                  if capped else "")
    cc_on = ", ".join(sorted(b.replace("_", " ")
                             for b, on in cc.items() if on))
    cc_clause = f" Class-cap held {cc_on} at their limits." if cc_on else ""
    approx = "" if res["converged"] else " (approx - did not fully converge)"
    held = ""
    if young_w:
        held = (f" Held {len(young_w)} holding(s) too new to model at current "
                f"weight: {', '.join(sorted(young_w))}.")

    rb = class_risk_budgets or {}
    rb_clause = ""
    rb_limited = ""
    if rb:
        b_ser = res["budgets"]
        rc_m = w_mature * (s_mat @ w_mature)
        total_rc = float(rc_m.sum())
        shares = pd.Series(rc_m / total_rc if total_rc > 0 else 0.0,
                           index=cov.index)
        parts = []
        for bkt in sorted(rb):
            members = [m for m in mature if mature_cof.get(m) == bkt]
            realized = float(shares[members].sum()) * 100.0
            parts.append(f"{bkt.replace('_', ' ')} {rb[bkt] * 100:.1f}% "
                         f"(realized {_fmt_realized(realized)})")
        rb_clause = " Risk budgets: " + ", ".join(parts) + "."
        limited = res["binding"]["budget_limited"]
        if limited:
            rb_limited = (" Risk budgets limited by caps for: "
                          + ", ".join(b.replace("_", " ") for b in limited)
                          + ".")

    if rb:
        shape_frag = (f"Risk contributions budget-shaped across "
                      f"{n_equalized} holdings"
                      + _gap_fragment(shares, b_ser, _free_names))
    else:
        shape_frag = (f"Risk contributions equalized across "
                      f"{n_equalized} holdings (spread within "
                      f"{disp_pct:.2f}% of the mean)")
    message = (f"Risk-parity loaded: vol ~ {sim_vol * 100:.1f}% "
               f"(current {cur_vol * 100:.1f}%). {shape_frag}{approx}."
               f"{cap_clause}{cc_clause}{rb_clause}{rb_limited}{held} "
               "Review/tweak below, then Run.")
    return {"new_pct": new_pct, "kind": "success", "message": message,
            "vol": sim_vol, "cur_vol": cur_vol, "converged": res["converged"]}
