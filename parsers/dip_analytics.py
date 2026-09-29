"""Buy-the-Dip analytics: drawdown depth/percentile, conditional further-fall,
conditional forward-return study, and dividend-yield lock. Pure functions over
date-indexed pandas Series — no I/O, no Streamlit. See
docs/superpowers/specs/2026-06-18-buy-the-dip-tab-design.md.

Return-basis convention: drawdown/percentile/further-fall on PRICE (close,
split-adjusted); forward-return on TOTAL RETURN (adj_close); yield on cash
distributions.
"""
from __future__ import annotations

import sys
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from risk_metrics import compute_drawdown_episodes  # noqa: E402
from turbulence import vol_regime  # noqa: E402

# Headline regime stats fall back to the full sample below this many in-regime
# entries — a 6-sample median is noise, not an estimate.
REGIME_MIN_N = 20

# Dip-buy verdict + bootstrap parameters (exposed, not magic numbers).
VERDICT_HORIZON = 252          # 12-month anchor for the verdict
BOOTSTRAP_N = 1000             # resamples for the Omega CI
BOOTSTRAP_BLOCK = 21           # stationary-bootstrap mean block ~ 1 trading month
BOOTSTRAP_CI = 0.90            # confidence level
BOOTSTRAP_SEED = 0             # fixed -> deterministic CIs (no Streamlit flicker)
VERDICT_MIN_EPISODES = 10      # fewer conditional outcomes -> inconclusive
VERDICT_SHALLOW_PCTILE = 50.0  # depth shallower than this -> "no edge yet"
VERDICT_STRONG_RR = 0.67       # reward/risk depth-rank >= this -> eligible "strong"

# The walk-forward referee (parsers/dip_backtest.py, registered run 2026-07-14)
# validated the verdict at a 10-year burn-in; the same run at a 5-year burn-in
# returned not_supported. 10y is therefore the history depth at which an edge
# claim has demonstrated support — NOT a proven minimum (the boundary between
# 5 and 10 was never located).
VERDICT_TRUSTED_HISTORY_YEARS = 10.0
# Bands that claim a dip-buy edge = exactly the set the referee's registered
# primary metric measured (mirrors dip_backtest.EDGE_BANDS; pinned equal by
# tests.test_dip_analytics.TestHistoryDepthCaveat).
VERDICT_EDGE_BANDS = frozenset({"strong", "neutral"})


def underwater(price: pd.Series) -> pd.Series:
    """Drawdown fraction series d_t = P_t / cummax(P_<=t) - 1 (<= 0)."""
    p = price.astype(float)
    return p / p.cummax() - 1.0


def drawdown_state(price: pd.Series, current_dd: float | None = None) -> dict:
    """Current drawdown depth and where it sits in the asset's own history.

    `pct_history_shallower` = "today's dip is deeper than X% of history".
    `pct_history_deeper`    = "sharper only (100-X)% of the time".
    """
    uw = underwater(price)
    dd = float(uw.iloc[-1]) if current_dd is None else float(current_dd)
    window = price.tail(252)
    return {
        "current_dd": dd,
        "pct_history_shallower": float((uw > dd).mean() * 100.0),
        "pct_history_deeper": float((uw < dd).mean() * 100.0),
        # fraction (e.g. -0.33), NOT a percent — unlike the pct_history_* siblings
        "frac_below_52w_high": float(price.iloc[-1] / window.max() - 1.0),
        "n_days": int(len(price)),
    }


def episodes_reaching(price: pd.Series, depth_frac: float) -> int:
    """Count distinct peak->trough->recovery episodes whose trough was at least
    `depth_frac` deep (depth_frac < 0). Reuses risk_metrics.compute_drawdown_episodes."""
    dates = pd.Series(price.index)
    eps = compute_drawdown_episodes(price.reset_index(drop=True), dates)
    thr = depth_frac * 100.0
    return int(sum(1 for e in eps if e["depth_pct"] <= thr))


def _apply_regime(mask: np.ndarray, in_regime: np.ndarray | None) -> np.ndarray:
    """AND a base entry mask with an optional regime mask. None = no-op."""
    if in_regime is None:
        return mask
    return mask & np.asarray(in_regime, dtype=bool)


def _further_fall_losses(price: pd.Series, current_dd: float,
                         in_regime: np.ndarray | None = None) -> np.ndarray:
    """Positive loss magnitudes (-further_fall) for completed entries. Shared by
    conditional_further_fall (quantiles) and the EVT tail fit. `in_regime`, when
    given, is a boolean array aligned to price restricting entries to those days."""
    p = price.astype(float).to_numpy()
    n = len(p)
    peak = np.maximum.accumulate(p)
    uw = p / peak - 1.0
    entry_mask = _apply_regime(uw <= current_dd, in_regime)
    losses = []
    for i in np.where(entry_mask)[0]:
        peak_i, trough, rec = peak[i], p[i], None
        for j in range(i + 1, n):
            if p[j] < trough:
                trough = p[j]
            if p[j] >= peak_i:
                rec = j
                break
        if rec is not None:
            losses.append(-(trough / p[i] - 1.0))     # positive magnitude
    return np.asarray(losses, dtype=float)


def conditional_further_fall(
    price: pd.Series,
    current_dd: float,
    quantiles: tuple = (0.5, 0.85, 0.95, 0.99),
    in_regime: np.ndarray | None = None,
) -> dict:
    """Distribution of the additional decline if you buy at today's depth.

    Entry set = historical days already at drawdown <= current_dd (optionally
    further restricted to `in_regime` days). For each, the decline to the trough
    before the drawdown RECOVERS. Entries that never recover are right-censored
    (excluded from the completed distribution, counted separately).
    Returned quantiles map severity: q=0.85 -> the worst-15% further fall.
    """
    losses = _further_fall_losses(price, current_dd, in_regime=in_regime)
    p = price.astype(float).to_numpy()
    peak = np.maximum.accumulate(p)
    uw = p / peak - 1.0
    entry_mask = _apply_regime(uw <= current_dd, in_regime)
    n_entries = int(entry_mask.sum())
    n_censored = n_entries - losses.size
    falls = -losses                                           # back to negative returns
    qmap = {q: (float(np.quantile(falls, 1.0 - q)) if falls.size else float("nan"))
            for q in quantiles}
    return {"quantiles": qmap, "n_complete": int(falls.size), "n_censored": int(n_censored)}


def conditional_recovery_time(price: pd.Series, current_dd: float,
                              in_regime: np.ndarray | None = None) -> dict:
    """Trading-days to break-even on a buy at dips at least this deep: from each
    entry day (drawdown <= current_dd, optionally regime-restricted) to the first
    later day the price regains the entry price. Entries that never regain it are
    right-censored. Returns median/p90 days + complete/censored counts."""
    p = price.astype(float).to_numpy()
    n = len(p)
    peak = np.maximum.accumulate(p)
    uw = p / peak - 1.0
    entry_mask = _apply_regime(uw <= current_dd, in_regime)
    days, n_censored = [], 0
    for i in np.where(entry_mask)[0]:
        entry_price = p[i]
        rec = None
        for j in range(i + 1, n):
            if p[j] >= entry_price:
                rec = j
                break
        if rec is None:
            n_censored += 1
        else:
            days.append(rec - i)
    d = np.asarray(days, dtype=float)
    return {
        "median_days": float(np.median(d)) if d.size else float("nan"),
        "p90_days": float(np.quantile(d, 0.90)) if d.size else float("nan"),
        "n_complete": int(d.size),
        "n_censored": int(n_censored),
    }


def time_underwater_caption(sym: str, band: str, current_dd: float, *,
                            median_text: str = "—", p90_text: str = "—",
                            n_complete: int = 0, n_censored: int = 0) -> str:
    """Markdown for the per-card 'Time underwater' line. Three branches:

    - ``band == "shallow"``: the current dip is in the shallower half of history,
      so "dips this deep" is ~every down-day and break-even is near-immediate — a
      "median 0 mo" reads like nothing, so say there is no real dip yet and name
      the depth (avoids the empty-looking zero TK hit on SPY/SCHD near highs).
    - ``n_complete == 0``: no dip this deep has ever regained break-even in-sample.
    - else: median / p90 months-to-break-even (+ censored note). ``median_text``
      and ``p90_text`` are pre-formatted by the caller (the regime dual-formatting
      stays at the call site)."""
    if band == "shallow":
        return (f"**Time underwater:** no real dip right now — {sym} is "
                f"{abs(current_dd) * 100:.1f}% off its high, so break-even is "
                f"near-immediate. A real months-to-recover figure appears once "
                f"the drawdown is deep.")
    if n_complete == 0:
        return (f"**Time underwater:** no dip this deep has returned to "
                f"break-even in {sym}'s history yet.")
    cens = (f" {n_censored} of {n_complete + n_censored} comparable dips "
            f"never did." if n_censored else "")
    return (f"**Time underwater:** after buying a dip this deep, {sym} historically "
            f"took a median **{median_text}** to get back to break-even — about "
            f"**{p90_text}** in the slow 1-in-10 case.{cens}")


def history_span_years(price: pd.Series) -> float:
    """Calendar span of the price index in years — the same KIND of measure the
    walk-forward referee's burn-in uses (``index[0] + DateOffset(years=N)``):
    span from the first bar, not a bar count. Not exactly equal — this reads
    ~0.5 days short of a DateOffset decade when the window holds only 2 leap
    days (3652d -> 9.99863y), so at the boundary the product errs toward
    disclosure. 0.0 for a degenerate series: a caption helper must never crash
    a card."""
    if len(price) < 2:
        return 0.0
    return float((price.index[-1] - price.index[0]).days / 365.25)


def history_depth_caveat(sym: str, band: str, years: float) -> str:
    """Disclosure for an edge claim resting on thin history, or "" when it does
    not apply (the band claims no edge, history is deep enough, or the span is
    unknown). Fires only on VERDICT_EDGE_BANDS: 'weak' says history has NOT
    rewarded the dip and 'shallow'/'inconclusive' decline to call, so the
    referee's edge finding does not speak to them — and caveating a 'don't buy'
    would read as licence to discount it. Disclosure only; the verdict math is
    untouched."""
    if band not in VERDICT_EDGE_BANDS:
        return ""
    if not np.isfinite(years) or years >= VERDICT_TRUSTED_HISTORY_YEARS:
        return ""
    # Round for accuracy, then CLAMP below the threshold. Plain rounding renders
    # all of [9.5, 10.0) as "10 years" -> "rests on only 10 years — demonstrated
    # only on 10+ years", contradicting itself at the exact boundary this
    # discloses. Plain flooring understates (the 1.9959y fixture -> "1 year").
    # The clamp makes n <= 9 unreachable-by-construction, so the sentence stays
    # coherent while the count stays honest.
    if years < 1.0:
        span = "under 1 year"
    else:
        n = min(round(years), int(VERDICT_TRUSTED_HISTORY_YEARS) - 1)
        span = "1 year" if n == 1 else f"{n} years"
    return (f"Thin history: this call rests on only {span} of {sym} data — the "
            f"walk-forward test demonstrated this verdict's edge only on "
            f"{VERDICT_TRUSTED_HISTORY_YEARS:.0f}+ years of history (at 5 years "
            f"it found none), so treat it as untested rather than trusted.")


def entry_index(price: pd.Series, current_dd: float) -> pd.Index:
    """Historical dates at drawdown at least as deep as current_dd."""
    uw = underwater(price)
    return price.index[uw <= current_dd]


def regime_conditioned_entries(
    price: pd.Series, current_dd: float, labels: pd.Series,
) -> tuple[pd.Index, str]:
    """Dip-entry dates (drawdown <= current_dd) that share TODAY's vol regime,
    plus today's regime label.

    `labels` is a turbulence.vol_regime() Series aligned to price.index
    ("calm"/"stressed"). The entry set is the same one `entry_index` returns,
    intersected with days carrying today's (= last day's) label, so the forward
    stats answer "dips this deep, in a tape like today's".
    """
    ent = entry_index(price, current_dd)
    today = str(labels.iloc[-1])
    same = labels.reindex(ent).to_numpy() == today
    return ent[same], today


def forward_returns(total_return: pd.Series, entry_idx: pd.Index,
                    horizon: int) -> np.ndarray:
    """Raw forward TOTAL-returns from each entry day at a single horizon
    (trading days). Entries without `horizon` days ahead are dropped."""
    tr = total_return.astype(float)
    pos = {ts: i for i, ts in enumerate(tr.index)}
    arr = tr.to_numpy()
    n = len(arr)
    out = []
    for ts in entry_idx:
        i = pos.get(ts)
        if i is None or i + horizon >= n:
            continue
        out.append(arr[i + horizon] / arr[i] - 1.0)
    return np.asarray(out, dtype=float)


def forward_return_stats(
    total_return: pd.Series,
    entry_idx: pd.Index,
    horizons: tuple = (21, 63, 126, 252),
) -> dict:
    """Forward TOTAL-return stats from each entry day, per horizon (trading days).

    Reports mean, median, hit-rate P(r>0), cond_loss (median return among entries
    that ended LOWER — the typical loss conditional on being down), p10 (the
    unconditional 1-in-10 downside) and worst (the min) per horizon. All are
    measured from the entry/buy price at the horizon, not peak-to-trough.
    """
    out: dict = {}
    for h in horizons:
        r = forward_returns(total_return, entry_idx, h)
        if r.size == 0:
            out[h] = {k: float("nan")
                      for k in ("mean", "median", "hit_rate", "cond_loss", "p10", "worst")}
            out[h]["n"] = 0
            continue
        down = r[r < 0.0]
        out[h] = {
            "mean": float(r.mean()),
            "median": float(np.median(r)),
            "hit_rate": float((r > 0).mean()),
            "cond_loss": float(np.median(down)) if down.size else float("nan"),
            "p10": float(np.quantile(r, 0.10)),
            "worst": float(r.min()),
            "n": int(r.size),
        }
    return out


def ttm_yield(price_latest: float, dividends: pd.Series, asof: pd.Timestamp) -> float:
    """Trailing-365d cash distributions / current price = yield locked at this price."""
    if not dividends.index.is_monotonic_increasing:
        dividends = dividends.sort_index()
    lo = asof - pd.Timedelta(days=365)
    ttm = float(dividends[(dividends.index > lo) & (dividends.index <= asof)].sum())
    return ttm / float(price_latest) if price_latest else float("nan")


def yield_history(price: pd.Series, dividends: pd.Series) -> pd.Series:
    """TTM-yield at each price date (TTM dividends / price)."""
    if dividends.empty:
        return pd.Series(float("nan"), index=price.index)
    div = dividends.sort_index()
    ttm = pd.Series(index=price.index, dtype=float)
    for ts in price.index:
        lo = ts - pd.Timedelta(days=365)
        ttm[ts] = div[(div.index > lo) & (div.index <= ts)].sum()
    return (ttm / price.astype(float)).astype(float)


def yield_percentile(price: pd.Series, dividends: pd.Series) -> dict:
    """Current TTM yield and where it sits in the asset's own yield history."""
    yh = yield_history(price, dividends).dropna()
    if yh.empty:
        return {"current_yield": float("nan"), "percentile": float("nan")}
    cur = float(yh.iloc[-1])
    return {"current_yield": cur, "percentile": float((yh <= cur).mean() * 100.0)}


def omega_ratio(returns: np.ndarray, threshold: float = 0.0) -> float:
    """Keating-Shadwick Omega: probability-weighted gains / losses above
    `threshold`. Boundary 1.0 = gain-mass equals loss-mass. inf when there are no
    losses; nan on an empty sample."""
    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]
    if r.size == 0:
        return float("nan")
    gains = float(np.clip(r - threshold, 0.0, None).sum())
    losses = float(np.clip(threshold - r, 0.0, None).sum())
    if losses == 0.0:
        return float("inf") if gains > 0.0 else float("nan")
    return gains / losses


def _stationary_bootstrap_indices(rng: np.random.Generator, n: int,
                                  p_new: float) -> np.ndarray:
    """One stationary-bootstrap resample's index vector (Politis-Romano).
    Vectorized, but draw order matches the original per-resample loop exactly
    (idx0, jumps, starts) so outputs are bit-identical: positions where
    jumps[t] is True (plus t=0) start a fresh block at starts[t] (idx0 at
    t=0); positions in between continue +1 modulo n."""
    idx0 = rng.integers(n)
    jumps = rng.random(n) < p_new
    starts = rng.integers(0, n, size=n)
    base = starts.copy()
    base[0] = idx0
    is_start = jumps.copy()
    is_start[0] = True
    start_pos = np.flatnonzero(is_start)
    seg = np.cumsum(is_start) - 1              # segment id per position
    offsets = np.arange(n) - start_pos[seg]    # steps since segment start
    return (base[start_pos][seg] + offsets) % n


def stationary_block_bootstrap_ci(stat_fn: Callable[[np.ndarray], float],
                                  series: np.ndarray, *,
                                  n_boot: int = BOOTSTRAP_N,
                                  expected_block: int = BOOTSTRAP_BLOCK,
                                  ci: float = BOOTSTRAP_CI,
                                  seed: int = BOOTSTRAP_SEED) -> dict:
    """Politis-Romano stationary-bootstrap CI for a statistic of a serially-
    dependent 1-D series. Geometric block lengths (mean = expected_block); blocks
    wrap circularly. Returns {"point","lo","hi"} (lo/hi nan when n<2). Non-finite
    resample statistics (e.g. Omega inf) are dropped before taking quantiles."""
    x = np.asarray(series, dtype=float)
    x = x[np.isfinite(x)]
    n = x.size
    point = float(stat_fn(x)) if n else float("nan")
    if n < 2:
        return {"point": point, "lo": float("nan"), "hi": float("nan")}
    rng = np.random.default_rng(seed)
    p_new = 1.0 / max(1, expected_block)
    stats = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        stats[b] = stat_fn(x[_stationary_bootstrap_indices(rng, n, p_new)])
    finite = stats[np.isfinite(stats)]
    if finite.size == 0:
        return {"point": point, "lo": float("nan"), "hi": float("nan")}
    lo_q = (1.0 - ci) / 2.0
    return {"point": point,
            "lo": float(np.quantile(finite, lo_q)),
            "hi": float(np.quantile(finite, 1.0 - lo_q))}


def reward_risk_depth_percentile(price: pd.Series, total_return: pd.Series,
                                 current_dd: float, *,
                                 horizon: int = VERDICT_HORIZON,
                                 n_grid: int = 10) -> float:
    """Self-calibration: rank today's reward/risk (Omega of forward returns from
    dips at least this deep) against the same Omega computed across a grid of dip
    depths over the asset's own history. Returns a fraction in [0, 1] (share of
    grid depths whose Omega is <= today's), 1.0 if today's set is all-gains, or
    nan when today's depth has no historical entries."""
    uw = underwater(price)
    deepest = float(uw.min())
    if not np.isfinite(deepest) or deepest >= 0.0:
        return float("nan")
    today = omega_ratio(forward_returns(total_return,
                                        entry_index(price, current_dd), horizon))
    if not np.isfinite(today):
        return 1.0 if today == float("inf") else float("nan")
    grid = np.linspace(deepest, 0.0, n_grid + 1)[:-1]     # drop the 0.0 endpoint
    omegas = []
    for d in grid:
        r = forward_returns(total_return, entry_index(price, float(d)), horizon)
        if r.size:
            omegas.append(omega_ratio(r))
    valid = np.asarray([o for o in omegas if np.isfinite(o)], dtype=float)
    if valid.size == 0:
        return float("nan")
    return float((valid <= today).mean())


def dip_buy_verdict(fwd_cond: np.ndarray, fwd_uncond: np.ndarray, *,
                    depth_pctile: float, rr_percentile: float,
                    n_recovered_further_fall: int,
                    ci: float = BOOTSTRAP_CI, n_boot: int = BOOTSTRAP_N,
                    seed: int = BOOTSTRAP_SEED,
                    min_episodes: int = VERDICT_MIN_EPISODES,
                    shallow_pctile: float = VERDICT_SHALLOW_PCTILE,
                    strong_rr: float = VERDICT_STRONG_RR) -> dict:
    """Statistically-driven dip-buy verdict. Omega(cond) with a stationary
    block-bootstrap CI; baseline Omega(uncond) as a fixed reference; edge = the
    difference (CI = the conditional CI shifted by the baseline). Bands: shallow /
    inconclusive / weak / neutral / strong. No hand-picked ratio cutoffs — the
    boundaries are Omega's structural 1.0 and the edge CI clearing 0; strength is
    self-calibrated via rr_percentile. A nan rr_percentile is treated as below
    the strong threshold (cannot claim 'strong' without self-calibration)."""
    cond = np.asarray(fwd_cond, dtype=float)
    cond = cond[np.isfinite(cond)]
    omega = omega_ratio(cond)
    baseline = omega_ratio(fwd_uncond)
    boot = stationary_block_bootstrap_ci(omega_ratio, cond, n_boot=n_boot,
                                         ci=ci, seed=seed)
    omega_ci = {"lo": boot["lo"], "hi": boot["hi"]}
    # baseline non-finite (e.g. all-positive unconditional history) -> no edge can
    # be claimed; the verdict falls through to "weak".
    has_base = np.isfinite(baseline)
    edge = (omega - baseline) if has_base else float("nan")
    edge_ci = ({"lo": omega_ci["lo"] - baseline, "hi": omega_ci["hi"] - baseline}
               if has_base else {"lo": float("nan"), "hi": float("nan")})
    n = int(cond.size)

    if depth_pctile < shallow_pctile:
        band = "shallow"
    elif n_recovered_further_fall == 0 or n < min_episodes:
        band = "inconclusive"
    elif np.isinf(omega):                       # no losing outcomes in-sample
        band = "strong" if rr_percentile >= strong_rr else "neutral"
    elif not np.isfinite(omega_ci["lo"]):
        band = "inconclusive"
    elif (np.isfinite(edge_ci["lo"]) and edge_ci["lo"] > 0.0
          and omega > 1.0 and rr_percentile >= strong_rr):
        band = "strong"
    elif edge > 0.0 and omega > 1.0:
        band = "neutral"
    else:
        band = "weak"

    return {"band": band, "omega": omega, "omega_ci": omega_ci,
            "baseline_omega": baseline, "edge": edge, "edge_ci": edge_ci,
            "rr_percentile": float(rr_percentile), "n": n}


def dip_verdict_block(price: pd.Series, tr: pd.Series,
                      horizons: tuple = (VERDICT_HORIZON,)) -> dict:
    """The verdict half of a dip card, as one pure orchestration seam: drawdown
    state -> entry sets (plain + vol-regime-conditioned) -> forward stats ->
    further-fall counts -> self-calibration -> dip_buy_verdict. Extracted from
    terminal/dip_service.dip_card_data (spec 2026-07-14) so the walk-forward
    referee replays EXACTLY the shipped pipeline on truncated inputs. Pure
    function of (price, tr) — no clock, no module state; that purity is what
    makes input truncation equal zero look-ahead.

    `horizons` must contain VERDICT_HORIZON (the verdict's regime-fallback
    rule reads the regime set's outcome count at that horizon)."""
    if VERDICT_HORIZON not in horizons:
        raise ValueError(f"horizons must include VERDICT_HORIZON "
                         f"({VERDICT_HORIZON}); got {horizons}")
    state = drawdown_state(price)
    ent = entry_index(price, state["current_dd"])
    labels = vol_regime(price)
    reg_ent, today_regime = regime_conditioned_entries(
        price, state["current_dd"], labels)
    in_reg = (labels == today_regime).to_numpy()

    fwd_full = forward_return_stats(tr, ent, horizons=horizons)
    fwd_reg = forward_return_stats(tr, reg_ent, horizons=horizons)

    ff_full = conditional_further_fall(price, state["current_dd"])
    ff_reg = conditional_further_fall(price, state["current_dd"],
                                      in_regime=in_reg)
    use_reg_ff = ff_reg["n_complete"] >= REGIME_MIN_N
    ff_head = ff_reg if use_reg_ff else ff_full

    H = VERDICT_HORIZON
    use_v = fwd_reg.get(H, {}).get("n", 0) >= REGIME_MIN_N
    v_ent = reg_ent if use_v else ent
    fwd_cond = forward_returns(tr, v_ent, H)
    fwd_uncond = forward_returns(tr, tr.index, H)
    rr_pct = reward_risk_depth_percentile(price, tr, state["current_dd"])
    verdict = dip_buy_verdict(
        fwd_cond, fwd_uncond, depth_pctile=state["pct_history_shallower"],
        rr_percentile=rr_pct, n_recovered_further_fall=ff_head["n_complete"])

    return {"state": state, "ent": ent, "labels": labels,
            "today_regime": today_regime, "reg_ent": reg_ent, "in_reg": in_reg,
            "fwd_full": fwd_full, "fwd_reg": fwd_reg, "ff_full": ff_full,
            "ff_reg": ff_reg, "use_reg_ff": use_reg_ff, "ff_head": ff_head,
            "fwd_cond": fwd_cond, "fwd_uncond": fwd_uncond, "rr_pct": rr_pct,
            "verdict": verdict, "history_years": history_span_years(price)}


def recovery_rate(price: pd.Series, min_depth_frac: float = -0.05) -> dict:
    """Of distinct dips at least `min_depth_frac` deep, the share that fully
    recovered to their prior peak in-sample. Backs the index-vs-single-name
    'does it even recover' framing."""
    dates = pd.Series(price.index)
    eps = compute_drawdown_episodes(price.reset_index(drop=True), dates)
    sig = [e for e in eps if e["depth_pct"] <= min_depth_frac * 100.0]
    rec = [e for e in sig if e["recovery_date"] is not None]
    return {
        "n_episodes": len(sig),
        "recovered": len(rec),
        "recovery_rate": (len(rec) / len(sig)) if sig else float("nan"),
    }
