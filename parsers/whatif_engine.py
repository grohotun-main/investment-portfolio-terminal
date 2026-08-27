"""
What-if Risk-tab engine: applies a hypothetical weight + candidate-ticker
scenario to the portfolio and returns before/after risk metrics.

DESIGN
  - Pure orchestration. The math lives in risk_metrics.py; this module
    composes it under a "weight vector + augmented price matrix" framing.
  - WhatIfScenario supports two modes: (a) pure reweight of existing
    holdings (candidate_ticker=None), or (b) add exactly one new ticker.
    Existing holdings may move up or down freely; there is no sink-only
    constraint. Invariants are enforced in validate().
  - compute_before_after returns a dict shaped {coverage, headline,
    detail, error}. Headline is small (cards on the tab); detail carries
    heavier frames for expander panels. error is a string when overlap
    history is too short to compute on.
  - History boundary: by default the engine restricts both before/after
    to the candidate's overlap window so the comparison is apples to
    apples on the same dates. Splicing (handled upstream in whatif_data)
    extends that window when the user opts in.

The Streamlit tab handles fetch/splice; this module receives a fully
prepared candidate_history Series and an unaugmented daily_prices frame,
augments internally, then dispatches existing risk_metrics functions.
"""
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import risk_metrics as rm  # noqa: E402
import whatif_data as wd  # noqa: E402


MIN_OVERLAP_DAYS = 252           # block scenarios with shorter overlap
MCR_DIVERSIFYING_RATIO = 0.95    # MCR/σ_p below this = diversifying
MCR_RISK_ADD_RATIO = 1.05        # above this = risk-adding (else neutral)
STRESS_Z_THRESHOLD = -1.5        # SPY log-ret z below this = stress day
STRESS_MIN_DAYS = 15             # min stress days for conditional metrics

MAX_CANDIDATES = 3


@dataclass(frozen=True)
class WhatIfCandidate:
    """One new ticker for a what-if scenario. proxy=None means no splice; a
    non-empty proxy back-fills the candidate's pre-inception history."""
    ticker: str
    proxy: str | None = None


@dataclass(frozen=True)
class WhatIfScenario:
    """A hypothetical reweighting. Two construction forms:
      - legacy scalar (Streamlit): candidate_ticker + splice_with_proxy + proxy_ticker
        (candidate_ticker=None -> pure reweight).
      - list (terminal): candidates=(WhatIfCandidate(...), ...), up to MAX_CANDIDATES.
    __post_init__ normalizes both into norm_candidates. Invariants in validate().
    """
    current_weights: pd.Series
    new_weights: pd.Series
    candidate_ticker: str | None = None
    splice_with_proxy: bool = False
    proxy_ticker: str | None = None
    candidates: tuple = ()

    def __post_init__(self) -> None:
        norm = tuple(self.candidates)
        # Orphaned splice_with_proxy=True (no candidate_ticker) is intentionally a no-op.
        if not norm and self.candidate_ticker:
            norm = (WhatIfCandidate(
                self.candidate_ticker,
                self.proxy_ticker if self.splice_with_proxy else None),)
        object.__setattr__(self, "_norm", norm)

    @property
    def norm_candidates(self) -> tuple:
        return self._norm

    def validate(self) -> None:
        cw, nw = self.current_weights, self.new_weights
        if not isinstance(cw, pd.Series) or not isinstance(nw, pd.Series):
            raise TypeError("weights must be pandas Series")
        if cw.empty or nw.empty:
            raise ValueError("weights must not be empty")
        for name, w in (("current_weights", cw), ("new_weights", nw)):
            s = float(w.sum())
            if not (0.999 <= s <= 1.001):
                raise ValueError(f"{name} sum must be ~1.0 (got {s:.6f})")
            if (w < -1e-9).any():
                raise ValueError(f"{name} has negative weight(s)")

        new_syms = [s for s in nw.index
                    if s not in cw.index and float(nw[s]) > 1e-9]
        if len(new_syms) > MAX_CANDIDATES:
            raise ValueError(
                f"up to {MAX_CANDIDATES} new tickers at a time "
                f"(got {len(new_syms)}: {', '.join(map(str, new_syms))})")

        cands = self.norm_candidates
        seen: set = set()
        for c in cands:
            t = c.ticker
            if not isinstance(t, str) or not t.strip():
                raise ValueError("candidate ticker must be a non-empty string")
            if t in seen:
                raise ValueError(f"duplicate candidate ticker {t!r}")
            seen.add(t)
            if t in cw.index:
                raise ValueError(
                    f"candidate {t!r} is already held — reweight it in the grid")
            if t not in nw.index or float(nw.get(t, 0.0)) <= 0:
                raise ValueError(f"candidate {t!r} must carry weight > 0 in new_weights")
            if c.proxy is not None:
                if not isinstance(c.proxy, str) or not c.proxy.strip():
                    raise ValueError(f"proxy for {t!r} must be a non-empty string")
                if c.proxy == t:
                    raise ValueError(f"proxy cannot equal candidate ({t!r})")
        # every genuinely-new symbol must be a declared candidate
        undeclared = [s for s in new_syms if s not in seen]
        if undeclared:
            raise ValueError(
                f"new ticker(s) {undeclared} present but not declared as candidates")

        union = cw.index.union(nw.index)
        delta = (nw.reindex(union).fillna(0.0) - cw.reindex(union).fillna(0.0)).abs()
        if float(delta.max()) <= 1e-9:
            raise ValueError("nothing to simulate: new_weights == current_weights")

    @property
    def candidate_weight(self) -> float:
        """First candidate's weight (legacy scalar accessor). 0 if none."""
        c = self.norm_candidates
        if not c or c[0].ticker not in self.new_weights.index:
            return 0.0
        return float(self.new_weights[c[0].ticker])

    @property
    def source_reductions(self) -> pd.Series:
        """Per-existing-ticker net reduction (current - new), aligned to
        current_weights. Negative entries mean that holding increased."""
        aligned = self.new_weights.reindex(self.current_weights.index).fillna(0.0)
        return self.current_weights - aligned


def compute_before_after(
    scenario: WhatIfScenario,
    daily_prices: pd.DataFrame,
    candidate_history: pd.Series | dict,
    bench_tr: pd.Series | None = None,
    rf_series: pd.Series | None = None,
    *,
    history_start: pd.Timestamp | None = None,
    benchmark_ticker: str = "SPY",
    min_overlap_days: int = MIN_OVERLAP_DAYS,
) -> dict:
    """Compute before/after risk metrics for a what-if scenario.

    Args:
      scenario: validated upstream OR validated here (we re-validate).
      daily_prices: wide-format existing universe price frame.
      candidate_history: candidate ticker(s) daily closes (post-splice if
        applicable). Either a single date-indexed Series (legacy Streamlit
        path, mapped to the first candidate) or a dict[ticker -> Series]
        (terminal path, one entry per scenario.norm_candidates entry).
      bench_tr: SPY total-return series for down-beta.
      rf_series: 3-month Treasury yield (decimal, annualized).
      history_start: optional global "history from" filter timestamp.
      benchmark_ticker: column in daily_prices used as stress condition.
      min_overlap_days: minimum overlap-window length to compute on.

    Returns dict shaped:
      {
        "coverage": {candidate_inception, candidates, overlap_start,
                     overlap_end, overlap_days, spliced, proxy_used},
        "headline": {metric: {before, after, delta}, ..., mcr_candidate,
                     mcr_verdict, mcr_candidates} OR None if blocked,
        "detail":   {heavy frames for expander panels} OR None,
        "error":    str OR None,
      }
    """
    scenario.validate()
    cands = scenario.norm_candidates
    cand_tickers = [c.ticker for c in cands]
    cw, nw = scenario.current_weights, scenario.new_weights

    # Weight-only — always available, even when overlap is too short.
    concentration = concentration_delta(cw, nw)

    # candidate_history: Series (legacy, first candidate) or
    # dict[ticker -> Series] (terminal, one per candidate).
    if isinstance(candidate_history, dict):
        hist_by_ticker = {t: candidate_history.get(t, pd.Series(dtype=float))
                          for t in cand_tickers}
    else:
        hist_by_ticker = ({cand_tickers[0]: candidate_history}
                          if cand_tickers else {})

    # ----- Build augmented prices + overlap window ---------------------
    if cand_tickers:
        augmented = wd.build_multi_augmented_price_matrix(
            daily_prices, hist_by_ticker).sort_index()
    else:
        augmented = daily_prices.sort_index()
    if history_start is not None:
        hs = pd.to_datetime(history_start)
        augmented = augmented.loc[augmented.index >= hs]

    # Overlap = rows where EVERY candidate column is present (intersection).
    if cand_tickers:
        overlap_prices = augmented.loc[
            augmented[cand_tickers].notna().all(axis=1)]
    else:
        overlap_prices = augmented
    overlap_days = int(len(overlap_prices))

    per_cand_cov = []
    for c in cands:
        h = hist_by_ticker.get(c.ticker, pd.Series(dtype=float))
        per_cand_cov.append({
            "ticker": c.ticker,
            "inception": (h.index.min() if not h.empty else None),
            "spliced": c.proxy is not None,
            "proxy": c.proxy,
        })
    coverage = {
        "candidate_inception": (per_cand_cov[0]["inception"]
                                if per_cand_cov else None),
        "candidates":    per_cand_cov,
        "overlap_start": (overlap_prices.index.min()
                          if not overlap_prices.empty else None),
        "overlap_end":   (overlap_prices.index.max()
                          if not overlap_prices.empty else None),
        "overlap_days":  overlap_days,
        # legacy scalar spliced/proxy_used: true if ANY candidate spliced /
        # first proxy — kept for Streamlit back-compat.
        "spliced":       any(c.proxy is not None for c in cands),
        "proxy_used":    next((c.proxy for c in cands if c.proxy is not None),
                              None),
    }

    if overlap_days < min_overlap_days:
        return {
            "coverage": coverage,
            "concentration": concentration,
            "headline": None,
            "detail":   None,
            "error": (
                f"Insufficient overlap: {overlap_days} trading days "
                f"(need >= {min_overlap_days}). Toggle splice with a proxy "
                "ticker, or pick a candidate with longer history."
            ),
        }

    # Before-state uses the same date window as overlap, so we're comparing
    # the same regime for both sides. The existing universe's columns are
    # whatever daily_prices has (no candidate column).
    before_prices = daily_prices.copy()
    if history_start is not None:
        before_prices = before_prices.loc[before_prices.index >= pd.to_datetime(history_start)]
    before_prices = before_prices.loc[
        (before_prices.index >= overlap_prices.index.min())
        & (before_prices.index <= overlap_prices.index.max())
    ]
    after_prices = overlap_prices

    # ----- Bundle 1: Vol & risk-adj return -----------------------------
    before_rc = rm.compute_risk_contributions(cw, before_prices)
    after_rc = rm.compute_risk_contributions(nw, after_prices)

    before_rets = rm.synthesize_portfolio_returns(cw, before_prices)
    after_rets = rm.synthesize_portfolio_returns(nw, after_prices)

    sharpe_before = _daily_sharpe(before_rets, rf_series)
    sharpe_after  = _daily_sharpe(after_rets, rf_series)
    sortino_before = _daily_sortino(before_rets, rf_series)
    sortino_after  = _daily_sortino(after_rets, rf_series)

    vol_before = before_rc["port_vol_ann"]
    vol_after  = after_rc["port_vol_ann"]

    # ----- Bundle 2: Diversification + MCR -----------------------------
    before_symbols = [s for s in cw.index if s in before_prices.columns]
    after_symbols  = [s for s in nw.index if s in after_prices.columns]
    corr_before = rm.compute_correlation_matrix(before_prices, before_symbols)
    corr_after  = rm.compute_correlation_matrix(after_prices, after_symbols)
    avg_corr_before = _avg_offdiagonal_corr(corr_before)
    avg_corr_after  = _avg_offdiagonal_corr(corr_after)

    per_sym = after_rc["per_symbol"]
    mcr_candidates = []
    for c in cands:
        mcr_v = np.nan
        if isinstance(per_sym, pd.DataFrame) and c.ticker in per_sym.index:
            v = per_sym.loc[c.ticker, "mctr_ann"]
            if pd.notna(v):
                mcr_v = float(v)
        mcr_candidates.append({"ticker": c.ticker, "mcr": mcr_v,
                               "verdict": _classify_mcr(mcr_v, vol_after)})
    # scalar back-compat = first candidate (or NaN/unknown)
    mcr_candidate = mcr_candidates[0]["mcr"] if mcr_candidates else np.nan
    mcr_verdict = mcr_candidates[0]["verdict"] if mcr_candidates else "unknown"

    # ----- Bundle 3: Tail / drawdown -----------------------------------
    var95_before, cvar95_before = rm.compute_var_cvar(before_rets, alpha=0.05)
    var95_after,  cvar95_after  = rm.compute_var_cvar(after_rets,  alpha=0.05)
    max_dd_before = _max_drawdown_pct(before_rets)
    max_dd_after  = _max_drawdown_pct(after_rets)

    # ----- Bundle 4: Stress / regime -----------------------------------
    stress_before = rm.compute_conditional_correlation_matrix(
        before_prices, before_symbols, condition_symbol=benchmark_ticker,
        z_threshold=STRESS_Z_THRESHOLD, min_stress_days=STRESS_MIN_DAYS,
    )
    stress_after = rm.compute_conditional_correlation_matrix(
        after_prices, after_symbols, condition_symbol=benchmark_ticker,
        z_threshold=STRESS_Z_THRESHOLD, min_stress_days=STRESS_MIN_DAYS,
    )
    stressed_corr_avg_before = _avg_offdiagonal_corr(stress_before["conditional"])
    stressed_corr_avg_after  = _avg_offdiagonal_corr(stress_after["conditional"])

    down_beta_before = down_beta_after = np.nan
    if bench_tr is not None and not bench_tr.empty:
        bench_rets = bench_tr.pct_change().dropna()
        if not bench_rets.empty:
            _, db_b = rm.compute_up_down_beta(
                before_rets,
                bench_rets.loc[bench_rets.index.isin(before_rets.index)],
            )
            _, db_a = rm.compute_up_down_beta(
                after_rets,
                bench_rets.loc[bench_rets.index.isin(after_rets.index)],
            )
            if pd.notna(db_b):
                down_beta_before = float(db_b)
            if pd.notna(db_a):
                down_beta_after = float(db_a)

    stressed_dr_before = _stressed_dr(
        before_prices, cw, benchmark_ticker)
    stressed_dr_after = _stressed_dr(
        after_prices, nw, benchmark_ticker)

    # ----- Assemble result --------------------------------------------
    headline = {
        "vol":               _pair(vol_before, vol_after),
        "sharpe":            _pair(sharpe_before, sharpe_after),
        "sortino":           _pair(sortino_before, sortino_after),
        "dr":                _pair(before_rc["dr"], after_rc["dr"]),
        "avg_pairwise_corr": _pair(avg_corr_before, avg_corr_after),
        "mcr_candidate":     (float(mcr_candidate)
                              if np.isfinite(mcr_candidate) else np.nan),
        "mcr_verdict":       mcr_verdict,
        "mcr_candidates":    mcr_candidates,
        "max_dd":            _pair(max_dd_before, max_dd_after),
        "var95":             _pair(var95_before, var95_after),
        "cvar95":            _pair(cvar95_before, cvar95_after),
        "stressed_corr_avg": _pair(stressed_corr_avg_before,
                                   stressed_corr_avg_after),
        "down_beta":         _pair(down_beta_before, down_beta_after),
        "stressed_dr":       _pair(stressed_dr_before, stressed_dr_after),
    }

    detail = {
        "corr_matrix_before":          corr_before,
        "corr_matrix_after":           corr_after,
        "stressed_corr_matrix_before": stress_before["conditional"],
        "stressed_corr_matrix_after":  stress_after["conditional"],
        "drawdown_curve_before":       _drawdown_curve(before_rets),
        "drawdown_curve_after":        _drawdown_curve(after_rets),
        "risk_contribution_before":    before_rc["per_symbol"],
        "risk_contribution_after":     after_rc["per_symbol"],
        "stress_meta_before": {
            "n_stress": stress_before["n_stress"],
            "n_full":   stress_before["n_full"],
            "enough":   stress_before["enough"],
        },
        "stress_meta_after": {
            "n_stress": stress_after["n_stress"],
            "n_full":   stress_after["n_full"],
            "enough":   stress_after["enough"],
        },
    }

    return {"coverage": coverage, "concentration": concentration,
            "headline": headline, "detail": detail, "error": None}


def concentration_delta(current_weights: pd.Series,
                        new_weights: pd.Series) -> dict:
    """Before/after concentration from the weight vectors alone.

    Weight-only (no prices / covariance) so it is always computable, even
    when the price-overlap window is too short for the risk metrics. Reuses
    rm.compute_concentration; Herfindahl is the reciprocal of effective-N by
    construction.
    """
    before = rm.compute_concentration(current_weights.astype(float))
    after  = rm.compute_concentration(new_weights.astype(float))

    def _hhi(c: dict) -> float:
        en = c.get("effective_n", np.nan)
        return float(1.0 / en) if (np.isfinite(en) and en > 0) else np.nan

    return {
        "effective_n": _pair(before["effective_n"], after["effective_n"]),
        "top5_pct":    _pair(before["top5_pct"],    after["top5_pct"]),
        "max_pct":     _pair(before["max_pct"],     after["max_pct"]),
        "herfindahl":  _pair(_hhi(before),          _hhi(after)),
    }


# ---------------------------------------------------------------------
# Helpers (engine-private)
# ---------------------------------------------------------------------

def _pair(before: float, after: float) -> dict:
    b = float(before) if np.isfinite(before) else np.nan
    a = float(after) if np.isfinite(after) else np.nan
    delta = (a - b) if (np.isfinite(b) and np.isfinite(a)) else np.nan
    return {"before": b, "after": a, "delta": delta}


def _avg_offdiagonal_corr(corr: pd.DataFrame) -> float:
    """Mean of unique off-diagonal entries; NaN-tolerant."""
    if corr is None or not isinstance(corr, pd.DataFrame) or corr.empty:
        return np.nan
    n = len(corr)
    if n < 2:
        return np.nan
    vals = corr.values[np.triu_indices(n, k=1)]
    vals = vals[np.isfinite(vals)]
    return float(vals.mean()) if vals.size else np.nan


def _classify_mcr(mcr: float, port_vol: float) -> str:
    """diversifying / neutral / risk_adding based on mcr/σ_p ratio.

    MCR below 95% of σ_p means adding the candidate at the margin pulls
    portfolio vol down; above 105% means it pulls it up; the 5% band in
    between absorbs estimation noise so a 0.1pp jitter doesn't get
    headlined as a decisive verdict.
    """
    if not (np.isfinite(mcr) and np.isfinite(port_vol)) or port_vol <= 0:
        return "unknown"
    ratio = mcr / port_vol
    if ratio < MCR_DIVERSIFYING_RATIO:
        return "diversifying"
    if ratio > MCR_RISK_ADD_RATIO:
        return "risk_adding"
    return "neutral"


def _max_drawdown_pct(daily_rets: pd.Series) -> float:
    """Worst peak-to-trough drawdown of (1+r).cumprod over the window."""
    if daily_rets is None or len(daily_rets) < 2:
        return np.nan
    wealth = (1.0 + daily_rets.fillna(0.0)).cumprod()
    if wealth.empty:
        return np.nan
    dd = wealth / wealth.cummax() - 1.0
    return float(dd.min())


def _drawdown_curve(daily_rets: pd.Series) -> pd.Series:
    if daily_rets is None or daily_rets.empty:
        return pd.Series(dtype=float, name="drawdown")
    wealth = (1.0 + daily_rets.fillna(0.0)).cumprod()
    return (wealth / wealth.cummax() - 1.0).rename("drawdown")


def _daily_sharpe(daily_rets: pd.Series,
                  rf_series: pd.Series | None) -> float:
    """Sharpe = (ann_mean − ann_rf_avg) / ann_vol."""
    if daily_rets is None or len(daily_rets) < 20:
        return np.nan
    ann_r = float(daily_rets.mean() * 252.0)
    ann_v = float(daily_rets.std(ddof=1) * np.sqrt(252.0))
    if not np.isfinite(ann_v) or ann_v <= 0:
        return np.nan
    return (ann_r - _ann_rf_avg(rf_series, daily_rets.index)) / ann_v


def _daily_sortino(daily_rets: pd.Series,
                   rf_series: pd.Series | None) -> float:
    """Sortino with full-sample downside (Sortino-Bawa convention):
    positive observations contribute 0, divisor is √252 × std over ALL obs."""
    if daily_rets is None or len(daily_rets) < 20:
        return np.nan
    ann_r = float(daily_rets.mean() * 252.0)
    downside = daily_rets.where(daily_rets < 0, 0.0)
    ann_dv = float(downside.std(ddof=1) * np.sqrt(252.0))
    if not np.isfinite(ann_dv) or ann_dv <= 0:
        return np.nan
    return (ann_r - _ann_rf_avg(rf_series, daily_rets.index)) / ann_dv


def _ann_rf_avg(rf_series: pd.Series | None,
                idx: pd.Index) -> float:
    """Average annualized RF over the daily-return index. 0 if missing."""
    if rf_series is None or rf_series.empty or idx is None or len(idx) == 0:
        return 0.0
    aligned = rf_series.reindex(idx).ffill()
    if aligned.dropna().empty:
        return 0.0
    return float(aligned.mean())


def _stressed_dr(daily_prices: pd.DataFrame,
                 weights: pd.Series,
                 condition_symbol: str = "SPY",
                 z_threshold: float = STRESS_Z_THRESHOLD,
                 min_stress_days: int = STRESS_MIN_DAYS) -> float:
    """Diversification Ratio computed on SPY-stress days only.

    Same DR formula as compute_risk_contributions, but the covariance is
    estimated on stress-day simple-return rows. Returns NaN if the
    universe, condition column, or stress-day count is insufficient.
    """
    if daily_prices is None or daily_prices.empty:
        return np.nan
    if condition_symbol not in daily_prices.columns:
        return np.nan
    common = [s for s in weights.index if s in daily_prices.columns]
    if len(common) < 2:
        return np.nan
    # Dedupe: if condition_symbol is already in the user's holdings (very
    # common when SPY is a portfolio position), `common + [condition_symbol]`
    # selects SPY twice and rets[condition_symbol] returns a DataFrame
    # instead of a Series — same dedup pattern compute_conditional_-
    # correlation_matrix uses.
    work_cols = list(dict.fromkeys(common + [condition_symbol]))
    px = daily_prices[work_cols].sort_index()
    rets = px.pct_change().iloc[1:]
    if rets.empty:
        return np.nan
    cond = rets[condition_symbol].dropna()
    if len(cond) < 20:
        return np.nan
    mu, sigma = float(cond.mean()), float(cond.std(ddof=1))
    if not np.isfinite(sigma) or sigma <= 0:
        return np.nan
    threshold = mu + z_threshold * sigma
    stress_idx = cond.index[cond <= threshold]
    if len(stress_idx) < min_stress_days:
        return np.nan
    stress_rets = rets.loc[stress_idx, common].fillna(0.0)
    w = weights[common].astype(float)
    if w.sum() <= 0:
        return np.nan
    w = w / w.sum()
    cov_d = stress_rets.cov() * 252.0
    if cov_d.empty:
        return np.nan
    w_arr = w.values
    var_p = float(w_arr @ cov_d.values @ w_arr)
    if var_p <= 0:
        return np.nan
    port_vol = float(np.sqrt(var_p))
    standalone_vols = np.sqrt(np.diag(cov_d.values))
    weighted_avg = float((w.values * standalone_vols).sum())
    return weighted_avg / port_vol if port_vol > 0 else np.nan
