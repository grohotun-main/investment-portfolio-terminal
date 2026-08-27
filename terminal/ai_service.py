"""AI narration layer — narration only, never computation (spec 2026-08-07).

Facts reducers compress existing service payloads into percent/ratio-only
dicts; scrub_gate() is the fail-closed privacy boundary (tickers +
percentages leave the machine, dollars and account ids never do);
generate() sends facts to claude-fable-5; the cache keys prose by
(section, scope, data_version) so generation happens only when the data
changed or the user forces it. No key / no anthropic package = off-state.
FastAPI-free by design: terminal/server.py owns all HTTP mapping.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import sys
import tempfile
import threading
import time
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "parsers"))

import pandas as pd

import _config
from attribution import position_return_contribution
from interim_stub import bench_stub_return, chain, ytd_to_date
# OCC / JPM display leg -> (underlying, put|call)
from synthesize_interim_positions import _parse_option_leg
from crash_betas import portfolio_crash_scenarios
from hedge_recommender import CRASH_WINDOWS
from risk_metrics import (_return_stats, compute_alpha_annual, compute_beta,
                          compute_calmar, compute_concentration,
                          compute_risk_contributions, compute_sharpe,
                          compute_sortino, compute_up_down_beta,
                          compute_var_cvar, rolling_active_stats,
                          window_drawdown_pct)
from lot_engine import build_key_resolvers, load_corporate_identity
from terminal import benchmark_service as bs
from terminal import dip_service as dps
from terminal import factor_service as fs
from terminal import health_service as hlth
from terminal import holdings_service as hs
from terminal import income_service as ins
from terminal import options_service as ops
from terminal import performance_service as ps
from terminal import risk_service as rs
from terminal import riskcontrib_service as rcs
from terminal import risksim_service as rss
from terminal import tax_service as txs

try:                       # Soft dependency: absence = AI panels off.
    import anthropic
except ImportError:        # pragma: no cover - exercised on bare machines
    anthropic = None

_LOG = logging.getLogger(__name__)

MODEL = "claude-fable-5"
_FALLBACK_BETA = "server-side-fallback-2026-07-01"
_CHAT_EFFORT = "medium"    # chat turns: routine Q&A over a ~20k-token pack (TK 2026-08-22; grew with tax_detail)


class AIError(Exception):
    """Base for narration-layer failures (all degrade to kind:'error')."""


class AIScrubError(AIError):
    """Outbound payload failed the privacy scrub — nothing was sent."""


class AIGenerationError(AIError):
    """The API call failed (network, refusal, or malformed response)."""


class AIDimError(AIError):
    """A seg-dimension value the section's payload does not offer
    (server maps to 422 — a malformed request, never kind:'error')."""


# --------------------------------------------------------------------------- #
# Privacy boundary. Builders emit percent/ratio facts by construction; this
# gate is defense-in-depth and fails CLOSED (raise -> route returns
# kind:'error'; the payload never leaves the process). Numbers are not
# scanned — a bare float is indistinguishable from a percent; the guarantee
# for numerics is the builders' whitelist construction.
# --------------------------------------------------------------------------- #
_DENY_KEY_RE = re.compile(
    r"(usd|dollar|amount|market_value|cost_basis|basis|proceeds|nav\b|"
    r"balance|value|cost|price|gain|loss|_gl\b|equity)",
    re.IGNORECASE)
# The percent/ratio family is legal by construction — deny tokens are
# exempted when the key itself names the unit. Reducers emitting a percent
# under a dollar-shaped name must use one of these suffixes.
_PCT_SUFFIXES = ("_pct", "_pp", "_ratio")
_DOLLAR_STR_RE = re.compile(r"\$\s*\d")
_DIGIT_RUN_RE = re.compile(r"\d{5,}")   # account-mask shape; years/dates are ≤4


def _scrub_walk(obj, path: str) -> None:
    if isinstance(obj, dict):
        for k, v in obj.items():
            ks = str(k)
            if (_DENY_KEY_RE.search(ks)
                    and not ks.lower().endswith(_PCT_SUFFIXES)):
                raise AIScrubError(f"denied key {k!r} at {path}")
            _scrub_walk(v, f"{path}.{k}")
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            _scrub_walk(v, f"{path}[{i}]")
    elif isinstance(obj, str):
        if _DOLLAR_STR_RE.search(obj):
            raise AIScrubError(f"dollar-formatted string at {path}")
        if _DIGIT_RUN_RE.search(obj):
            raise AIScrubError(f"5+ digit run (account-mask shape) at {path}")


def scrub_gate(payload: dict) -> dict:
    """Raise AIScrubError if the payload carries anything dollar- or
    account-shaped; return the payload unchanged otherwise."""
    _scrub_walk(payload, "facts")
    return payload


# --------------------------------------------------------------------------- #
# Facts reducers — one per section. Each consumes the EXISTING tab payload
# (engine math untouched) and emits a compact, percent/ratio-only dict.
# --------------------------------------------------------------------------- #
def _canon_filter(vals) -> list[str]:
    """Sorted non-'all' ids from a filter selection (str | list | None).
    Empty result == whole book. Frames-free (used to key the cache before
    load_frames)."""
    if isinstance(vals, str):
        vals = [vals]
    return sorted({v for v in (vals or []) if v and v != "all"})


def _filter_labels(frames: hs.Frames, account, asset_class):
    """Joined human labels for an ACTIVE account/class selection, else None.
    Mirrors the broker label already in scope (plain strings, scrub-safe)."""
    snap = hs._current_snap(frames)

    def _lab(vals, opts):
        ids = _canon_filter(vals)
        if not ids:
            return None
        by_id = {o["id"]: o.get("label", o["id"]) for o in opts}
        return " + ".join(by_id.get(i, i) for i in ids)

    return (_lab(account, hs._account_options(snap)[0]),
            _lab(asset_class, hs._class_options(snap)[0]))


def _scope_block(frames: hs.Frames, history_start: str,
                 account_label: str | None = None,
                 class_label: str | None = None) -> dict:
    label = (" + ".join(frames.broker_scope) if frames.broker_scope
             else "Fidelity + JPM")
    # history_start is threaded from the request — Frames carries no such
    # attribute, so deriving it here would silently lie to the model.
    out = {"broker": label, "history_start": history_start or "all"}
    if account_label:
        out["account"] = account_label
    if class_label:
        out["asset_class"] = class_label
    return out


def _num(v, nd=2):
    """Finite float rounded to nd, else None (facts carry null, never NaN)."""
    if v is None:
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return round(f, nd) if f == f and abs(f) != float("inf") else None


def _pct100(v, nd=2):
    """Decimal fraction -> rounded percent, None/NaN-safe (via _num).
    Promoted from three identical function-local copies (#383 rider)."""
    return None if v is None else _num(float(v) * 100.0, nd)


def _facts_factors(frames: hs.Frames, history_start: str,
                   broker=None, dims=None) -> dict:
    view = fs.build_factor_view(frames)
    state = view.get("state", {})
    if not state.get("available"):
        return {"section": "factors", "available": False,
                "scope": _scope_block(frames, history_start),
                "reason": state.get("unavailable")}
    d = dims or {}
    w = d.get("window") or state.get("default_window") or fs.WINDOWS[0]
    model = d.get("model") or state.get("default_model") or fs.DEFAULT_MODEL
    if "window" in d and w not in (state.get("windows") or []):
        raise AIDimError(f"unknown window {w!r}")
    if "model" in d and model not in (state.get("models") or []):
        raise AIDimError(f"unknown model {model!r}")
    blk = view["by_window"].get(w, {})
    mb = (blk.get("models") or {}).get(model, {})
    if not mb:
        return {"section": "factors", "available": False,
                "scope": _scope_block(frames, history_start),
                "window": w, "model": model, "reason": "window_unavailable"}
    wf = mb.get("waterfall") or {}
    return {
        "section": "factors", "available": True,
        "scope": _scope_block(frames, history_start),
        "window": w, "model": model,
        "alpha_by_model": [
            {"model": s["model"], "available": s["available"],
             "alpha_annual": s["value"], "ci_half_width": s["delta"],
             "stats": s["help"]}
            for s in blk.get("strip", [])],
        "fit": mb.get("metrics"),
        "low_obs_warning": mb.get("low_obs_warning"),
        "betas": mb.get("beta_numeric"),
        "attribution_pp": [{"label": i["label"], "pp": i["value_pp"]}
                           for i in wf.get("items", [])],
        "mean_return_pp": wf.get("total_pp"),
        "aligned_window": mb.get("window_caption"),
    }


# --------------------------------------------------------------------------- #
# Cache: data/ai_cache.json (gitignored via the data/ rule). Keyed by
# section|scope_key, valid while data_version matches. data_version hashes
# the stat signature of every top-level *.csv/*.json EXCEPT the cache file
# itself — any ingest, refresh, or close invalidates automatically. Cache
# hits never load frames. This caches AI PROSE only; it is not the declined
# frames/latency cache and touches no engine path.
# --------------------------------------------------------------------------- #
_CACHE_NAME = "ai_cache.json"


def data_version(data_dir) -> str:
    d = Path(data_dir)
    if not d.is_dir():
        raise FileNotFoundError(str(data_dir))
    sig = []
    for p in sorted(d.glob("*.csv")) + sorted(d.glob("*.json")):
        if p.name == _CACHE_NAME:
            continue
        st = p.stat()
        sig.append(f"{p.name}|{st.st_mtime_ns}|{st.st_size}")
    return hashlib.sha1("\n".join(sig).encode("utf-8")).hexdigest()


def scope_key(broker: list[str], history_start: str,
              dims: dict | None = None) -> str:
    """Canonical, UNAMBIGUOUS scope key. JSON-encoded so a single value
    containing a comma can never collide with a multi-value selection
    (which would make the 422 contract depend on cache warmth). Non-empty
    ``dims`` (seg-tracking, S3) merge as a sub-object; empty/None dims are
    OMITTED so every dimless key stays byte-identical to its S1 form and
    existing cache entries remain valid."""
    obj: dict = {"broker": sorted(broker or ["all"]),
                 "history_start": history_start or "all"}
    if dims:
        obj["dims"] = {str(k): str(v) for k, v in dims.items()}
    return json.dumps(obj, sort_keys=True, separators=(",", ":"))


def _cache_load(data_dir) -> dict:
    p = Path(data_dir) / _CACHE_NAME
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}    # corrupt cache = cold cache; regeneration heals it


def cache_get(data_dir, section: str, skey: str) -> dict | None:
    return _cache_load(data_dir).get(f"{section}|{skey}")


CACHE_FMT = 3   # bumped when the narrative TEXT contract changes (v2 =
                # structured JSON, 2026-08-20; v3 = to_date provisional
                # blocks, 2026-08-22). Freshness = dv AND fmt, so every
                # older entry regenerates once, lazily — scope keys and
                # goldens never move.


def entry_fresh(entry: dict | None, dv: str) -> bool:
    """Entry is servable as CURRENT: same data_version AND written under
    the current text-format contract. Anything else is a miss (the old
    text can still serve as STALE on generation failure)."""
    return bool(entry and entry.get("data_version") == dv
                and entry.get("fmt") == CACHE_FMT)


# The blob write is a read-modify-write of the WHOLE file — without a lock,
# two concurrent puts (different sections) silently drop one key (review
# S2-3, second order). Sync-def routes share one process's threadpool, so a
# threading.Lock suffices.
_CACHE_IO_LOCK = threading.Lock()


def cache_put(data_dir, section: str, skey: str, dv: str, text: str) -> dict:
    entry = {"text": text, "data_version": dv, "model": MODEL,
             "fmt": CACHE_FMT,
             "generated_at": time.strftime("%Y-%m-%d %H:%M")}
    with _CACHE_IO_LOCK:
        blob = _cache_load(data_dir)
        blob[f"{section}|{skey}"] = entry
        p = Path(data_dir) / _CACHE_NAME
        fd, tmp = tempfile.mkstemp(dir=str(p.parent), suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump(blob, fh, ensure_ascii=False, indent=1)
            os.replace(tmp, p)          # atomic on the same volume
        except BaseException:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise
    return entry


# Per-scope in-flight guard (review S2-3): a cold generation takes ~30s, and
# an impatient tab flip re-fires the GET — without coalescing, both requests
# miss and BOTH generate (double cost). The second waiter blocks on the key
# lock, re-checks the cache, and serves the first's result.
_GEN_LOCKS: dict = {}
_GEN_LOCKS_GUARD = threading.Lock()


def _gen_lock(key: str) -> threading.Lock:
    with _GEN_LOCKS_GUARD:
        return _GEN_LOCKS.setdefault(key, threading.Lock())


def generate_cached(data_dir, section: str, skey: str, dv: str, facts: dict,
                    *, client, force: bool = False):
    """Generate-or-serve under the per-scope lock. Returns (entry, was_hit).
    ``force`` skips the in-lock double-check (the Regenerate button must
    actually regenerate). Raises AIError like generate() when the (single)
    generation fails."""
    with _gen_lock(f"{section}|{skey}"):
        if not force:
            entry = cache_get(data_dir, section, skey)
            if entry_fresh(entry, dv):
                return entry, True
        text = generate(section, facts, client=client)
        return cache_put(data_dir, section, skey, dv, text), False


# --------------------------------------------------------------------------- #
# B1a: async generation. generate_cached still does the work under _gen_lock;
# these wrap it in a background thread + a tiny in-memory status registry so the
# request handler returns kind:"generating" immediately and the client polls.
# The registry holds only transient status ("running"/"error"), never text —
# a finished narrative lives in the cache, so a poll that succeeds is a cache
# hit. State is per-process (single-worker uvicorn); a restart drops running
# jobs and the next poll simply restarts generation.
# --------------------------------------------------------------------------- #
_JOBS: dict = {}
_JOBS_GUARD = threading.Lock()


def _spawn(fn):
    threading.Thread(target=fn, daemon=True).start()


_SPAWN = _spawn   # patched to `lambda fn: fn()` in tests for determinism


def job_status(section: str, skey: str) -> dict | None:
    """Snapshot of the background generation job for one scope, or None."""
    with _JOBS_GUARD:
        j = _JOBS.get(f"{section}|{skey}")
        return dict(j) if j else None


def clear_job(section: str, skey: str) -> None:
    with _JOBS_GUARD:
        _JOBS.pop(f"{section}|{skey}", None)


def start_generation(data_dir, section: str, skey: str, dv: str, facts: dict,
                     *, client, force: bool = False) -> None:
    """Ensure a background job is generating this scope's narrative. Idempotent:
    a second call while one runs is a no-op (the running-guard). Returns at once;
    the job runs generate_cached and records completion (clears the entry — the
    cache now holds the text) or failure (an "error" entry a poll surfaces)."""
    key = f"{section}|{skey}"
    with _JOBS_GUARD:
        if _JOBS.get(key, {}).get("status") == "running":
            return
        _JOBS[key] = {"status": "running"}

    def _run():
        try:
            generate_cached(data_dir, section, skey, dv, facts,
                            client=client, force=force)
            clear_job(section, skey)
        except Exception as e:          # noqa: BLE001 — broad except is
            # deliberate here too (see generate(), ~line 1021): ANY failure in
            # the generate_cached chain (AIError, but also e.g. a KeyError from
            # an unknown section, a json.dumps TypeError, a cache_put OSError)
            # must be recorded as a poll-visible error, never left silently
            # uncaught in the background thread — an uncaught exception here
            # would leave _JOBS[key] wedged at "running" forever, making
            # start_generation a permanent silent no-op for that scope.
            with _JOBS_GUARD:
                _JOBS[key] = {"status": "error", "error": str(e)}

    _SPAWN(_run)


# --------------------------------------------------------------------------- #
# v2-S2: the whole-book chat pack. The chat answers ONLY from these facts —
# every chat-visible section's reducer output (of the four chat-only
# sections, tax/holdings detail also read the data dir's raw files;
# options/health wrap service views), scrubbed as a whole
# (defense in depth on top of each reducer's own gate; a scrub failure
# fails the whole turn CLOSED). Sig-gated sections (frontier, risksim) are
# omitted by design — the chat says those runs aren't visible to it.
# Memoized by (data_version, canonical broker, history_start) so repeat
# turns are frames-free; bounded like the frontier memo.
# --------------------------------------------------------------------------- #
_CHAT_SECTIONS = ("portfolio", "performance", "benchmark", "risk",
                  "riskcontrib", "factors", "income", "tax", "dip",
                  "tax_detail", "holdings_detail", "options", "health")
_CHAT_PACK_MEMO: dict = {}
_CHAT_PACK_LOCK = threading.Lock()
_CHAT_PACK_MAX = 8      # ~90 KB each on the real book; LRU (a hit re-touches) so passive warms never evict the chat scope


def build_chat_pack(frames: hs.Frames, history_start: str,
                    broker: list[str] | None) -> dict:
    """One facts dict per chat-visible section, whole-pack scrubbed.
    The nine box sections go through build_facts; the four chat-only
    sections (specs 2026-08-22 + 2026-08-23) through _CHAT_DETAIL_REDUCERS —
    same signature, same per-section scrub, never registered in SECTIONS.
    Failures propagate (fail-closed): a scrub or reducer error aborts the
    turn rather than shipping a partial pack.

    The tax view is built ONCE here and threaded to its three consumers
    (portfolio posture / tax / tax_detail) via their tax_view= keyword —
    the #262 bundle= seam shape; it was built three times per pack (#380
    review, TK-raised). All three effective asofs are today in this
    production path, so one view serves verbatim; the box path
    (build_facts without tax_view) still self-builds."""
    tv = txs.build_tax_view(frames, frames.data_dir, broker=broker)
    shared_tv = {"portfolio": _facts_portfolio, "tax": _facts_tax,
                 "tax_detail": _facts_tax_detail}
    pack = {}
    for s in _CHAT_SECTIONS:
        if s in shared_tv:
            pack[s] = scrub_gate(shared_tv[s](frames, history_start,
                                              broker, None, tax_view=tv))
        elif s in _CHAT_DETAIL_REDUCERS:
            pack[s] = scrub_gate(_CHAT_DETAIL_REDUCERS[s](
                frames, history_start, broker, None))
        else:
            pack[s] = build_facts(s, frames, history_start=history_start,
                                  broker=broker, dims=None)
    return scrub_gate(pack)


def _chat_pack_key(dv: str, history_start: str, broker) -> tuple:
    return (dv, tuple(sorted(broker or ["all"])), history_start or "all")


def chat_pack_get(dv: str, history_start: str, broker) -> dict | None:
    """Memo lookup; a hit re-touches the entry (LRU) so the scope being
    chatted in outlives packs that tab-render warms filled passively."""
    key = _chat_pack_key(dv, history_start, broker)
    with _CHAT_PACK_LOCK:
        pack = _CHAT_PACK_MEMO.pop(key, None)
        if pack is not None:
            _CHAT_PACK_MEMO[key] = pack
        return pack


def chat_pack_build(frames: hs.Frames, dv: str, history_start: str,
                    broker) -> dict:
    pack = build_chat_pack(frames, history_start, broker)
    with _CHAT_PACK_LOCK:
        _CHAT_PACK_MEMO[_chat_pack_key(dv, history_start, broker)] = pack
        while len(_CHAT_PACK_MEMO) > _CHAT_PACK_MAX:
            _CHAT_PACK_MEMO.pop(next(iter(_CHAT_PACK_MEMO)))
    return pack


# --- pack pre-warm (2026-08-22) --------------------------------------------
# One threading.Lock per scope key, held for the duration of a build: a warm
# and a chat POST (or two warms) for the same scope never build twice — the
# second caller blocks on the lock and returns the pack the first memoized.
# "building" == the key's lock is currently held. Locks for stale
# data_versions are pruned after each successful build (bounded growth).
# _CHAT_PACK_FAILED is the warm path's negative memo: a scope whose build
# raised is not re-spawned by tab renders under the same data_version (a
# deterministic failure would otherwise burn ~1 min per render); the chat
# POST still rebuilds (surfacing the error), and a success clears the mark.
_CHAT_PACK_BUILDING: dict = {}
_CHAT_PACK_FAILED: dict = {}


def chat_pack_state(dv: str, history_start: str, broker) -> str:
    """'ready' (memo hit) | 'building' (a build holds the scope's lock) |
    'failed' (last warm raised under this data_version) | 'missing'."""
    if chat_pack_get(dv, history_start, broker) is not None:
        return "ready"
    key = _chat_pack_key(dv, history_start, broker)
    with _CHAT_PACK_LOCK:
        lock = _CHAT_PACK_BUILDING.get(key)
        failed = key in _CHAT_PACK_FAILED
    if lock is not None and lock.locked():
        return "building"
    return "failed" if failed else "missing"


def chat_pack_ensure(dv: str, history_start: str, broker, frames_fn) -> dict:
    """Memo hit -> the pack (frames_fn never called). Miss -> build under
    the scope's lock; a concurrent caller blocks, then returns the pack
    the first one memoized. frames_fn is lazy so the caller's load +
    validation stay synchronous and only the reducer work (~1 min on the
    real book) runs inside the lock. A build exception releases the lock,
    stores nothing and propagates (the next call rebuilds)."""
    key = _chat_pack_key(dv, history_start, broker)
    with _CHAT_PACK_LOCK:
        lock = _CHAT_PACK_BUILDING.setdefault(key, threading.Lock())
    with lock:                          # unheld on a memo hit: microseconds
        pack = chat_pack_get(dv, history_start, broker)
        if pack is None:
            pack = chat_pack_build(frames_fn(), dv, history_start, broker)
            with _CHAT_PACK_LOCK:
                _CHAT_PACK_FAILED.pop(key, None)
                for k in [k for k, lk in _CHAT_PACK_BUILDING.items()
                          if k[0] != dv and not lk.locked()]:
                    del _CHAT_PACK_BUILDING[k]
        return pack


def chat_pack_warm(dv: str, history_start: str, broker, frames_fn) -> None:
    """Best-effort background warm: ensure() with failures logged, never
    raised. A failure marks the scope 'failed' for this data_version so tab
    renders stop re-spawning it; the next chat turn still rebuilds."""
    key = _chat_pack_key(dv, history_start, broker)
    try:
        chat_pack_ensure(dv, history_start, broker, frames_fn)
    except Exception as e:      # noqa: BLE001 — warm is advisory; chat surfaces errors
        with _CHAT_PACK_LOCK:
            _CHAT_PACK_FAILED[key] = str(e)
        _LOG.warning("chat pack warm failed for %s: %s", key, e)


def start_chat(chat_id: str, messages: list[dict], pack: dict,
               *, client, detail_fn=None) -> None:
    """Background chat generation under the _JOBS registry (key
    chat|<id>). Idempotent while running. DELIBERATE deviation from the
    narration jobs' 'registry never holds text' rule: chat answers are
    uncached by design, so the finished entry carries the text and the
    first successful poll pops it (server-side clear_chat on read).
    ``detail_fn`` (full-gate S2) is threaded through to generate_chat's
    tool loop."""
    key = f"chat|{chat_id}"
    with _JOBS_GUARD:
        if _JOBS.get(key, {}).get("status") == "running":
            return
        _JOBS[key] = {"status": "running"}

    def _run():
        try:
            text = generate_chat(messages, pack, client=client,
                                 detail_fn=detail_fn)
            with _JOBS_GUARD:
                _JOBS[key] = {"status": "done", "text": text}
        except Exception as e:          # noqa: BLE001 — poll-visible error,
            # never a wedged 'running' (the start_generation precedent).
            with _JOBS_GUARD:
                _JOBS[key] = {"status": "error", "error": str(e)}

    _SPAWN(_run)


def chat_status(chat_id: str) -> dict | None:
    return job_status("chat", chat_id)


def clear_chat(chat_id: str) -> None:
    clear_job("chat", chat_id)


# --------------------------------------------------------------------------- #
# S2: the five-window portfolio-vs-benchmark grid (resolved benchmark, not
# SPY-specific). No such grid exists anywhere in the codebase (seam-mapped
# 2026-08-07) — this is the slice's one piece of new glue, composed entirely
# from risk_metrics primitives. All emitted percents are PERCENT-SCALE floats
# (−50.0 == −50%).
# --------------------------------------------------------------------------- #
def _win_slice_monthly(m: pd.Series, spec):
    if spec is None or m.empty:
        return m
    if spec == "ytd":
        idx = pd.DatetimeIndex(pd.to_datetime(m.index))
        return m[idx.year == idx.year.max()]
    return m.tail(int(spec))


def _window_row(label: str, m: pd.Series, bench_m: pd.Series,
                p_d: pd.Series, bench_d: pd.Series, rf, dwin, *,
                requested_months=None) -> dict:
    """One grid row. A window the scope cannot FILL is marked unavailable
    (spec: never present a 7-month slice as '3y' — review S2-1); available
    rows carry the daily observation count actually consumed so partial
    daily coverage is visible to the model. The 'benchmark' side is the
    RESOLVED benchmark (SPY or the 60/40 blend), not necessarily SPY."""
    n = int(len(m))
    if n < 2 or len(bench_m) < 2:
        return {"window": label, "available": False, "n_months": n}
    if requested_months is not None and n < int(requested_months):
        return {"window": label, "available": False, "n_months": n,
                "requested_months": int(requested_months)}

    def _side(s: pd.Series) -> dict:
        cagr, vol, _dvol, _n = _return_stats(s)
        return {"twr_cum_pct": round(float((1 + s).prod() - 1) * 100, 2),
                "twr_ann_pct": round(float(cagr) * 100, 2),
                "vol_ann_pct": round(float(vol) * 100, 2),
                "max_dd_pct": round(float(window_drawdown_pct(s).min()), 2),
                "sharpe": round(float(compute_sharpe(s, rf)), 2)}

    pa, sa = p_d.align(bench_d, join="inner")
    if dwin is not None:
        pa, sa = pa.tail(int(dwin)), sa.tail(int(dwin))
    beta = compute_beta(p_d, bench_d, window=dwin)
    corr = float(pa.corr(sa)) if len(pa) >= 2 else None
    return {"window": label, "available": True, "n_months": n,
            "n_days_used": int(len(pa)),
            "portfolio": _side(m), "benchmark": _side(bench_m),
            "beta": (round(float(beta), 2) if beta == beta else None),
            "correlation": (round(corr, 2) if corr is not None
                            and corr == corr else None)}


_FACTORS_INSTRUCTION = (
    "Explain what this factor regression says about the portfolio: where "
    "returns come from (market/size/value/momentum/quality tilts), whether "
    "alpha is statistically distinguishable from zero (use the t and R² in "
    "the stats strings), and what the attribution decomposition attributes "
    "the mean return to. If a smaller model shows alpha the full model "
    "does not, say what that implies. Name the analysis window in your "
    "first sentence — the reader may be looking at a different window on "
    "the chart above.")

# --------------------------------------------------------------------------- #
# S2: the AI ANALYSIS tab facts — composed from existing engines only.
# --------------------------------------------------------------------------- #
_PORTFOLIO_WINDOWS = [("Full history", None, None), ("5y", 60, 1260),
                      ("3y", 36, 756), ("1y", 12, 252), ("YTD", "ytd", "ytd")]


def _ytd_daily_count(p_d: pd.Series):
    if p_d.empty:
        return None
    idx = pd.DatetimeIndex(pd.to_datetime(p_d.index))
    return int((idx.year == idx.year.max()).sum()) or None


def _concentration_facts(b: dict, frames: hs.Frames) -> tuple[dict, float | None]:
    """Concentration + cash-weight facts from a risk bundle. Mirrors the
    Risk tab's concentration EXACTLY (risk_service._concentration:410-420):
    drop cash + options, fold TLH->SPY and the treasury ladder->SGOV, group
    by symbol — narratives must cite the numbers the Risk tab displays.
    Shared by the portfolio (S2) and risk (S3) reducers."""
    snap = b.get("latest_snap")
    if snap is None or snap.empty:
        return {}, None
    total_all = float(snap["market_value"].sum()) or 1.0
    cash_mv = float(snap.loc[snap["asset_class"] == "cash",
                             "market_value"].sum())
    cash_weight_pct = round(cash_mv / total_all * 100, 1)
    s = snap.copy()
    is_option = s["asset_class"].astype(str).str.startswith("option")
    s = s[(s["asset_class"] != "cash") & (~is_option)].copy()
    s.loc[s["asset_class"] == "tax_loss_harvesting", "symbol"] = "SPY"
    s.loc[s["bucket"] == "JPM Treasury Ladder", "symbol"] = "SGOV"
    s = s.dropna(subset=["symbol"])
    by_ticker = s.groupby("symbol")["market_value"].sum()
    if not len(by_ticker):
        return {}, cash_weight_pct
    c = compute_concentration(by_ticker)
    rc = compute_risk_contributions(b["weights"], frames.daily_prices,
                                    window=252)
    per = rc["per_symbol"].head(3)
    conc = {"effective_n": round(float(c["effective_n"]), 1),
            "top5_weight_pct": round(float(c["top5_pct"]), 1),
            "top10_weight_pct": round(float(c["top10_pct"]), 1),
            "max_weight_pct": round(float(c["max_pct"]), 1),
            "n_positions": int(c["n_positions"]),
            "top_risk_contributors": [
                {"ticker": str(sym),
                 "risk_share_pct": round(float(r["pctr_pct"]), 1),
                 "weight_pct": round(float(r["weight_pct"]), 1)}
                for sym, r in per.iterrows()]}
    return conc, cash_weight_pct


def _tax_posture_facts(tv: dict) -> dict:
    """Unrealized LT/ST split from a build_tax_view payload, reduced to the
    S2 scheme: shares of |unrealized| magnitude + gain/loss direction
    words — dollars never leave the process. Shared by the portfolio (S2)
    and tax (S3) reducers."""
    lots = tv.get("lots") or []
    st = sum(r["unrealized_gl"] for r in lots
             if r.get("term") == "short"
             and r.get("unrealized_gl") is not None)
    lt = sum(r["unrealized_gl"] for r in lots
             if r.get("term") == "long"
             and r.get("unrealized_gl") is not None)
    mag = abs(st) + abs(lt)
    if mag <= 0:
        return {"available": False}
    return {"available": True,
            "long_share_pct": round(abs(lt) / mag * 100, 1),
            "short_share_pct": round(abs(st) / mag * 100, 1),
            "long_net": "gain" if lt >= 0 else "loss",
            "short_net": "gain" if st >= 0 else "loss",
            "note": ("taxable accounts only; shares of unrealized "
                     "P&L magnitude")}


def _facts_portfolio(frames: hs.Frames, history_start: str,
                     broker=None, dims=None, *, asof=None,
                     tax_view=None) -> dict:
    """``asof`` is a TEST-ONLY keyword (the SECTIONS dispatcher passes
    positionals) pinning the today-dependent income and tax_posture
    blocks, forwarded to build_income_view's and build_tax_view's own
    asof seams. ``tax_view`` is the chat pack's request-scoped reuse seam
    (the #262 bundle= shape): a caller-prebuilt build_tax_view result for
    the same frames/broker/effective-asof; None -> self-build as before
    (the box path and every pinned test)."""
    resolved = (dims or {}).get("benchmark") or "spy"
    if resolved not in hs.BENCHMARKS:
        raise AIDimError(f"unknown benchmark {resolved!r}")
    if resolved == "60_40" and not hs._agg_available(frames):
        # AGG bond leg absent -> degrade to SPY. The GET /api/ai/portfolio route
        # resolves this before build_facts; a direct 60/40 regenerate reaches
        # here unresolved, so mirror the fallback to keep the facts honest.
        resolved = "spy"
    b = rs._bundle(frames, None, None, False, False, benchmark=resolved)
    m, bench_m = b["monthly"], b["spy_monthly"]
    p_d, bench_d = b["port_rets"], b["spy_rets"]
    base = {"section": "portfolio",
            "scope": _scope_block(frames, history_start),
            "benchmark": {"id": resolved, "label": hs.BENCHMARKS[resolved]}}
    if m is None or len(m) < 2:
        return {**base, "available": False, "reason": "no_monthly_series"}
    # INNER-align the monthly pair once (review S2-2): bench_monthly is a
    # strict subset of monthly in general, so independent slicing hands the
    # model a portfolio window against a different benchmark span. Every
    # window below covers identical months on both sides; NaN months (e.g.
    # the book's 2020-03 partial) drop from both.
    m, bench_m = m.align(bench_m, join="inner")
    ok = m.notna() & bench_m.notna()
    m, bench_m = m[ok], bench_m[ok]
    if len(m) < 2:
        return {**base, "available": False, "reason": "no_aligned_months"}
    rf_series = rs._load_rf(frames.data_dir)
    rf = rf_series if not rf_series.empty else rs.RF_FALLBACK_ANNUAL

    # Provisional stub period (spec 2026-08-22): each available window also
    # carries to-date cumulative figures (statement window chained with the
    # stub on the portfolio side, the benchmark TR over the stub dates on the
    # other); statement keys stay. Only when the benchmark series reaches it.
    stub = hs.interim_stub(frames)
    bench_stub = bench_stub_return(b.get("bench_tr"), stub)   # the bundle's own series
    stub_gap_note = None
    if stub is not None and bench_stub is None:
        # DA-C-10: name why the to_date blocks are absent instead of a
        # silent statement-anchored view next to a provisional Performance
        # pack (a stale benchmark leg truncates the blend's join).
        tr = b.get("bench_tr")
        _end = (pd.Timestamp(tr.index.max()).strftime("%Y-%m-%d")
                if tr is not None and len(tr) else "n/a")
        stub_gap_note = (f"benchmark series ends {_end}, before the "
                         "provisional period — to-date columns unavailable")
    if bench_stub is None:
        stub = None
    windows = []
    for label, mspec, dspec in _PORTFOLIO_WINDOWS:
        mm = _win_slice_monthly(m, mspec)
        bm = _win_slice_monthly(bench_m, mspec)
        dwin = _ytd_daily_count(p_d) if dspec == "ytd" else dspec
        req = mspec if isinstance(mspec, int) else None
        row = _window_row(label, mm, bm, p_d, bench_d, rf, dwin,
                          requested_months=req)
        if stub is not None and row.get("available"):
            p_cum = row["portfolio"]["twr_cum_pct"] / 100.0
            b_cum = row["benchmark"]["twr_cum_pct"] / 100.0
            if label == "YTD":
                p_td, b_td = ytd_to_date(p_cum, b_cum, stub, bench_stub)
            else:
                p_td, b_td = chain(p_cum, stub.return_pct), chain(b_cum, bench_stub)
            row["to_date"] = {
                "end": stub.end_date.strftime("%Y-%m-%d"),
                "portfolio_cum_pct": _num(p_td * 100),
                "benchmark_cum_pct": _num(b_td * 100),
                "stub_days": int(stub.days),
                "provisional": True,
            }
        windows.append(row)

    conc, cash_w = _concentration_facts(b, frames)
    if cash_w is not None:
        base["cash_weight_pct"] = cash_w

    # Income and tax posture are OPTIONAL facts blocks: their absence
    # degrades the narrative, never the tab (the windows grid is the
    # load-bearing content). Both emit available:False on any trouble and
    # the prompt handles it ("say so plainly").
    inc: dict = {"available": False}
    try:
        iv = ins.build_income_view(frames, asof=asof)
        kpis = (iv.get("forward") or {}).get("kpis") or []
        if (iv.get("forward") or {}).get("available") and len(kpis) >= 4:
            inc = {"available": True,
                   "forward_yield_on_covered_mv": kpis[1]["value"],
                   "yield_on_cost_pct": kpis[2]["value"],
                   "coverage_of_book": kpis[3]["value"],
                   "note": ("yields are measured against the COVERED subset "
                            "of the book (coverage_of_book gives its share); "
                            "options/cash/unfetched tickers excluded")}
    except Exception:
        pass

    tax: dict = {"available": False}
    try:
        tv = (tax_view if tax_view is not None
              else txs.build_tax_view(frames, frames.data_dir,
                                      broker=broker, asof=asof))
        tax = _tax_posture_facts(tv)
    except Exception:
        pass

    latest: dict = {}
    if len(m) and len(bench_m):
        pm, bm2 = m.align(bench_m, join="inner")
        if len(pm):
            mo = pd.Timestamp(pm.index[-1])
            latest = {"month": mo.strftime("%Y-%m"),
                      "portfolio_pct": round(float(pm.iloc[-1]) * 100, 2),
                      "bench_pct": round(float(bm2.iloc[-1]) * 100, 2),
                      "spread_pp": round(float(pm.iloc[-1] - bm2.iloc[-1])
                                         * 100, 2)}

    as_of = None
    if frames.positions is not None and not frames.positions.empty:
        as_of = str(pd.Timestamp(
            frames.positions["statement_date"].max()).date())
    return {**base, "available": True, "as_of": as_of, "windows": windows,
            "concentration": conc, "income": inc, "tax_posture": tax,
            "latest_month": latest,
            **({"stub": stub.as_facts()} if stub is not None else {}),
            **({"to_date_unavailable": stub_gap_note} if stub_gap_note
               else {})}


def _facts_risk(frames: hs.Frames, history_start: str,
                broker=None, dims=None) -> dict:
    """Risk Overview facts — the same primitives the Risk tab's blocks call
    (risk_service _risk_adjusted/_drawdown/_daily_vol/_beta), reduced to
    percent/ratio-only values. The tab's tiles are display-formatted (and
    carry dollar subs), so facts recompute from the bundle instead — the
    S2 _facts_portfolio precedent."""
    d = dims or {}
    account = d.get("account") or "all"
    asset_class = d.get("asset_class") or "all"
    acct_label, class_label = _filter_labels(frames, account, asset_class)
    base = {"section": "risk",
            "scope": _scope_block(frames, history_start, acct_label, class_label)}
    if frames.twr_portfolio is None or frames.twr_portfolio.empty:
        return {**base, "available": False, "reason": "no_twr"}
    # filter_requested ⟺ a non-'all' selection produced a label. Whole-book
    # keeps the EXACT current call (golden byte-identical); filtered uses the
    # resolved bundle — the same call build_risk_view makes for that filter.
    filtered = bool(acct_label or class_label)
    b = (rss._bundle_for(frames, account, asset_class) if filtered
         else rs._bundle(frames, None, None, False, False))
    m = b["monthly"].dropna()
    if len(m) < 2:
        return {**base, "available": False, "reason": "too_short"}
    rf_series = rs._load_rf(frames.data_dir)
    rf = rf_series if not rf_series.empty else rs.RF_FALLBACK_ANNUAL

    r_1y, r_3y = m.tail(12), m.tail(36)
    spy_m = b["spy_monthly"]
    spy_1y, spy_3y = spy_m.tail(12), spy_m.tail(36)
    risk_adjusted = {
        "sharpe_1y": _num(compute_sharpe(r_1y, rf)),
        "sharpe_3y": _num(compute_sharpe(r_3y, rf)),
        "sortino_1y": _num(compute_sortino(r_1y, rf)),
        "sortino_3y": _num(compute_sortino(r_3y, rf)),
        "calmar_3y": _num(compute_calmar(r_3y, window_drawdown_pct(r_3y))),
        "spy_sharpe_1y": (_num(compute_sharpe(spy_1y, rf))
                          if not spy_1y.empty else None),
        "spy_sharpe_3y": (_num(compute_sharpe(spy_3y, rf))
                          if not spy_3y.empty else None),
        "spy_sortino_1y": (_num(compute_sortino(spy_1y, rf))
                           if not spy_1y.empty else None),
        "spy_sortino_3y": (_num(compute_sortino(spy_3y, rf))
                           if not spy_3y.empty else None),
        "spy_calmar_3y": (_num(compute_calmar(spy_3y,
                                              window_drawdown_pct(spy_3y)))
                          if not spy_3y.empty else None),
        "months_used_1y": int(len(r_1y)),
        "months_used_3y": int(len(r_3y)),
    }
    dd_full, spy_dd_full = b["dd_full_pct"], b["spy_dd_full_pct"]
    drawdown = {
        "current_dd_pct": (_num(dd_full.iloc[-1])
                           if not dd_full.empty else None),
        "max_dd_1y_pct": (_num(window_drawdown_pct(r_1y).min())
                          if not r_1y.empty else None),
        "max_dd_3y_pct": (_num(window_drawdown_pct(r_3y).min())
                          if not r_3y.empty else None),
        "max_dd_itd_pct": (_num(dd_full.min())
                           if not dd_full.empty else None),
        "spy_current_dd_pct": (_num(spy_dd_full.iloc[-1])
                               if not spy_dd_full.empty else None),
        "spy_max_dd_itd_pct": (_num(spy_dd_full.min())
                               if not spy_dd_full.empty else None),
    }
    daily = None
    p_d, s_d = b["port_rets"], b["spy_rets"]
    if not frames.daily_prices.empty and not p_d.empty:
        var95, cvar95 = compute_var_cvar(p_d, alpha=0.05)
        if not s_d.empty:
            svar95, scvar95 = compute_var_cvar(s_d, alpha=0.05)
        else:
            svar95 = scvar95 = float("nan")
        r60, r252 = p_d.tail(60), p_d.tail(252)
        s60, s252 = s_d.tail(60), s_d.tail(252)
        rt2 = 252 ** 0.5
        daily = {
            "vol_60d_ann_pct": (_num(r60.std(ddof=1) * rt2 * 100)
                                if len(r60) >= 5 else None),
            "vol_252d_ann_pct": (_num(r252.std(ddof=1) * rt2 * 100)
                                 if len(r252) >= 5 else None),
            "spy_vol_60d_ann_pct": (_num(s60.std(ddof=1) * rt2 * 100)
                                    if len(s60) >= 5 else None),
            "spy_vol_252d_ann_pct": (_num(s252.std(ddof=1) * rt2 * 100)
                                     if len(s252) >= 5 else None),
            "var_95_1d_pct": _num(var95 * 100),
            "cvar_95_1d_pct": _num(cvar95 * 100),
            "spy_var_95_1d_pct": _num(svar95 * 100),
            "spy_cvar_95_1d_pct": _num(scvar95 * 100),
            "worst_day_pct": _num(p_d.min() * 100) if len(p_d) else None,
            "spy_worst_day_pct": (_num(s_d.min() * 100)
                                  if len(s_d) else None),
            "n_days_60d": int(len(r60)),
            "n_days_252d": int(len(r252)),
        }
        if not s_d.empty:
            daily["beta_60d"] = _num(compute_beta(p_d, s_d, window=60))
            daily["beta_252d"] = _num(compute_beta(p_d, s_d, window=252))
            up_b, dn_b = compute_up_down_beta(p_d, s_d, window=252)
            daily["up_beta_252d"] = _num(up_b)
            daily["down_beta_252d"] = _num(dn_b)
            daily["alpha_252d_ann_pct"] = _num(
                compute_alpha_annual(p_d, s_d, window=252) * 100)
    coverage = None
    sg = b["synthesis_gaps"]
    if sg is not None and not sg.empty:
        bad = sg[sg["pct_no_price"] > 5.0]   # mirror risk_service 751-763
        if not bad.empty:
            coverage = {"n_symbols": int(len(bad)),
                        "weight_pct": _num(bad["weight_pct"].sum(), 1)}
    conc, cash_w = _concentration_facts(b, frames)
    # v2-S4: historical crash-window replay (β-weighted implied drop; the
    # engine excludes NaN-beta names and renormalizes — the #137/#138 rule —
    # and reports the excluded weight). Whole replay is a MODELLING read.
    stress = portfolio_crash_scenarios(frames.daily_prices,
                                       b.get("weights"), CRASH_WINDOWS)
    out = {**base, "available": True,
           "risk_adjusted": risk_adjusted, "drawdown": drawdown,
           "daily": daily, "daily_available": daily is not None,
           "concentration": conc or None,
           "coverage_gaps": coverage, "stress": stress}
    if cash_w is not None:
        out["cash_weight_pct"] = cash_w
    return out


def _facts_riskcontrib(frames: hs.Frames, history_start: str,
                       broker=None, dims=None) -> dict:
    """Risk Contribution facts for the tracked (estimator, benchmark) dims.
    Reads the view's own raw/numeric channels (weight_vs_pctr arrays,
    portfolio.vol.raw, _cmp strings, dr_regime.character) — zero math.
    Invariance rule (spec §3.2): the combo is indexed at the DEFAULT
    ES-confidence and threshold, and nothing alpha- or threshold-dependent
    is emitted, so a click on those untracked segs can never contradict
    the box."""
    d = dims or {}
    account = d.get("account") or "all"
    asset_class = d.get("asset_class") or "all"
    acct_label, class_label = _filter_labels(frames, account, asset_class)
    base = {"section": "riskcontrib",
            "scope": _scope_block(frames, history_start, acct_label, class_label)}
    view = rcs.build_riskcontrib_view(frames, account=account,
                                      asset_class=asset_class)
    state = view.get("state", {})
    if not state.get("available"):
        return {**base, "available": False,
                "reason": state.get("unavailable")}
    controls = view["controls"]
    ests = [e["id"] for e in controls["estimators"]]
    benches = [x["id"] for x in controls["benchmarks"]]
    est = d.get("estimator") or ests[0]
    bench = d.get("benchmark") or benches[0]
    if est not in ests:
        raise AIDimError(f"unknown estimator {est!r}")
    if bench not in benches:
        raise AIDimError(f"unknown benchmark {bench!r}")
    alpha0 = controls["es_levels"][0]["id"]
    thr0 = controls["thresholds"][0]["id"]
    combo = view["combos"][f"{est}|{alpha0}|{thr0}"]
    wp = combo["weight_vs_pctr"]
    contributors = sorted(
        ({"ticker": str(s), "risk_share_pct": _num(p, 1),
          "weight_pct": _num(w, 1)}
         for s, w, p in zip(wp["symbols"], wp["weight"], wp["pctr"])
         if p is not None),
        key=lambda r: -(r["risk_share_pct"] or 0.0))[:5]
    # v2-S4: pair each risk share with its 1y return contribution (weights ×
    # 252d total return — Task-1 engine; approximate, disclosed in the
    # instruction). Weights from the same filtered bundle seam; depends only
    # on weights+prices, so the untracked-seg invariance rule holds.
    try:
        b5 = rss._bundle_for(frames, account, asset_class)
        attrib5 = position_return_contribution(
            frames.daily_prices, b5.get("weights"), {"252d": 252})["252d"]
    except Exception:                     # noqa: BLE001 — pairing is optional
        attrib5 = None
    for row in contributors:
        row["contrib_252d_pp"] = (
            _num(attrib5.loc[row["ticker"], "contrib_pp"], 2)
            if attrib5 is not None and row["ticker"] in attrib5.index
            else None)
    regime = {"available": False}
    dr_reg = view.get("dr_regime") or {}
    ch = dr_reg.get("character")
    if dr_reg.get("available") and ch:
        regime = {"available": True, "level": ch.get("level"),
                  "headline": ch.get("headline"),
                  "asymmetry_note": ch.get("asymmetry_note")}
    bench_block = None
    bvol = (((view.get("benchmarks") or {}).get(bench) or {})
            .get("vol") or {}).get(est)
    if bvol and bvol.get("value") is not None:
        bench_block = {"id": bench,
                       "bench_vol_pct": bvol["value"],
                       "delta_pp": bvol["delta"],
                       "dir": bvol["dir"]}
    raw_vol = combo["portfolio"]["vol"]["raw"]
    tiles = combo.get("top_tiles") or []
    return {**base, "available": True,
            "estimator": est, "benchmark": bench,
            "portfolio_vol_ann_pct": (_num(raw_vol * 100)
                                      if raw_vol is not None else None),
            "diversification_ratio": (tiles[-1].get("value")
                                      if tiles else None),
            "top_risk_contributors": contributors,
            "benchmark_vol": bench_block,
            "regime": regime,
            "treasury_ladder_present": bool(
                (view.get("treasury_ladder") or {}).get("present"))}


_RISKCONTRIB_INSTRUCTION = (
    "Name the broker scope (FACTS.scope), covariance estimator, and "
    "benchmark from FACTS in your first "
    "sentence — the reader may have other segs selected on the tab. "
    "Explain who drives portfolio risk: the top contributors' risk share "
    "vs dollar weight (call out any name whose risk share far exceeds its "
    "weight), how concentrated total risk is, the diversification ratio, "
    "portfolio volatility vs the selected benchmark (bench_vol_pct and "
    "delta_pp), and the regime block's headline and asymmetry note when "
    "available. Fields that are null or marked unavailable — say so "
    "plainly. "
    "If FACTS.scope names an account or asset_class filter, name that "
    "filter in your first sentence too — the narration describes that "
    "filtered slice, not the whole book. "
    "Each top contributor carries contrib_252d_pp — its approximate 1y "
    "return contribution (weight × total return, distributions "
    "reinvested): note names whose risk share is large but return "
    "contribution small or negative (risk budget spent without payoff), "
    "and the reverse. Approximate basis — total returns, current weights. ")


_RISK_INSTRUCTION = (
    "Answer: how risky is this book right now, absolutely and vs SPY? "
    "Name the broker scope and history window from FACTS.scope in your "
    "first sentence. Cover volatility (60d vs 252d — rising or falling), "
    "tail risk (VaR/CVaR vs SPY), drawdown state (current vs the 1y/3y/"
    "full-history maxima), risk-adjusted quality (Sharpe/Sortino/Calmar "
    "vs SPY), beta including the up/down split, and concentration "
    "(effective N, max weight, top risk contributors). If coverage_gaps "
    "is present, add one caveat sentence naming the affected weight "
    "percentage. Fields that are null are unavailable — say so plainly. "
    "The _1y/_3y/_60d/_252d suffixes name REQUESTED windows; "
    "months_used_* and n_days_* carry the portfolio side's actual "
    "observation counts (SPY figures window independently) — "
    "when actual falls short of requested, state the actual span instead "
    "of the window label. "
    "If FACTS.scope names an account or asset_class filter, name that "
    "filter in your first sentence too — the narration describes that "
    "filtered slice, not the whole book. "
    "FACTS.stress replays historical crash windows: for each scenario give "
    "the window, SPY's drop, and the book's beta-implied drop "
    "(implied_drop_pct) — a modelling replay, never a forecast; name the "
    "excluded weight when nonzero. If stress.available is false, say the "
    "replay lacks usable history. ")


def _facts_tax(frames: hs.Frames, history_start: str,
               broker=None, dims=None, *, asof=None,
               tax_view=None) -> dict:
    """Tax posture facts: counts, magnitude shares, direction words, and
    flags ONLY — never dollars (they scrub-fail by construction). The
    stale reason is reduced to a presence flag: reason strings can carry
    row counts that would trip the 5-digit account-mask guard.
    ``asof`` is a TEST-ONLY keyword (the SECTIONS dispatcher passes
    positionals) pinning the today-dependent lot-term surface (term
    split, ripening count), forwarded to build_tax_view's own asof
    seam."""
    base = {"section": "tax", "scope": _scope_block(frames, history_start)}
    tv = (tax_view if tax_view is not None
          else txs.build_tax_view(frames, frames.data_dir, broker=broker,
                                  asof=asof))
    if tv.get("kind") != "tax":
        return {**base, "available": False,
                "reason": str(tv.get("reason") or "tax_view_error")[:80]}
    meta = tv.get("meta") or {}
    lots = tv.get("lots") or []
    ripening = sum(1 for r in lots if _lot_ripening(r))   # the #320 predicate
    rz = (tv.get("summary") or {}).get("realized_ytd") or {}
    realized = {"available": False}
    # tax_service's own contract (docstring at _realized_ytd_inner): the
    # SUCCESS path always carries an "unavailable": None key too (an
    # explicit "the build is fine" marker, not merely a name reused on
    # failure) — a key-presence check is therefore always False and would
    # permanently hide real by-term data. Check the VALUE.
    if rz.get("unavailable") is None and rz.get("by_term"):
        by_term = rz["by_term"]
        mag = sum(abs(float(v.get("net") or 0.0)) for v in by_term.values())
        terms = {}
        for t, v in by_term.items():
            net = float(v.get("net") or 0.0)
            terms[t] = {"net_direction": "gain" if net >= 0 else "loss",
                        "share_pct": (round(abs(net) / mag * 100, 1)
                                      if mag > 0 else None)}
        total_net = float(rz.get("net") or 0.0)
        realized = {"available": True, "year": rz.get("year"),
                    "by_term": terms,
                    "total_direction": ("gain" if total_net >= 0
                                        else "loss"),
                    "options_uncovered_closes": int(
                        rz.get("options_uncovered") or 0),
                    "note": ("shares of realized net magnitude by term; "
                             "options included, uncovered option closes "
                             "counted separately")}
    hv = tv.get("harvest") or {}
    harvest = {"available": False}
    hsum = hv.get("summary") or {}
    if hv.get("candidates") is not None and hsum:
        sem = hv.get("semantics") or {}
        harvest = {"available": True,
                   "candidates": int(hsum.get("candidates") or 0),
                   "wash_blocked": int(hsum.get("blocked") or 0),
                   "ira_blocked": int(hsum.get("ira_blocked") or 0),
                   "window_observed_pct": sem.get("window_observed_pct"),
                   "clear_means": sem.get("clear_means"),
                   "stale_note": sem.get("stale_note")}
    return {**base, "available": True,
            "ledger": {"stale": bool(meta.get("stale")),
                       "stale_reason_present": bool(
                           meta.get("stale_reason"))},
            "unrealized": _tax_posture_facts(tv),
            "realized_ytd": realized,
            "harvest": harvest,
            "ripening_to_long_within_60d": int(ripening)}


_TAX_INSTRUCTION = (
    "Factual posture summary ONLY — you must never recommend selling, "
    "harvesting, holding, or timing anything; describe what is. Name the "
    "broker scope from FACTS.scope in your first sentence. Cover: the "
    "unrealized long/short-term magnitude split and directions; realized "
    "year-to-date by term (directions and magnitude shares — FACTS "
    "carries no dollar amounts, cite shares and counts only); the "
    "harvest scan counts (candidates, wash-blocked, IRA-blocked) with "
    "the observability caveat — a 'clear' verdict covers only the "
    "observed fraction of the window (window_observed_pct); the count "
    "of short-term gain lots within 60 days of turning long-term; and, "
    "if ledger.stale is true, one sentence noting the numbers may lag "
    "until the ledger rebuilds.")


_DETAIL_RIPENING_DAYS = 60      # the #320 predicate _facts_tax already uses


_WASH_RANK = {"clear": 0, "blocked": 1}   # any other status ranks as non-clear


def _lot_ripening(r: dict, days: int = _DETAIL_RIPENING_DAYS) -> bool:
    """The #320 predicate: a short-term GAIN lot within `days` of turning
    long-term — shared by _facts_tax's count and _facts_tax_detail's rows."""
    return (r.get("term") == "short"
            and r.get("days_to_long_term") is not None
            and r["days_to_long_term"] <= days
            and (r.get("unrealized_gl") or 0) > 0)


def _sleeve_account_ids() -> frozenset:
    """The direct-index sleeve's account id(s) from config — the accounts
    whose in-window churn must not make a name lot-actionable. Empty when
    unconfigured (CI's config_example ships TLH_ACCOUNT_ID = \"\")."""
    tlh = str(getattr(hs.cfg, "TLH_ACCOUNT_ID", "") or "")
    return frozenset({tlh}) if tlh else frozenset()


def _facts_tax_detail(frames: hs.Frames, history_start: str,
                      broker=None, dims=None, *, asof=None,
                      tax_view=None) -> dict:
    """Chat-only per-ticker tax facts (spec 2026-08-22). Three blocks:
    `wash_calendar` from the transaction ledger alone (statement + interim
    rows — fresh through tx_frontier and independent of the lot ledger),
    `harvest` candidates grouped by ticker, and `lots` term detail for
    actionable tickers only. Dollar-free by construction. ``asof`` is the
    test seam (the pack dispatch passes none -> today); ``tax_view`` is
    the pack's request-scoped reuse seam (see _facts_portfolio)."""
    base = {"section": "tax_detail",
            "scope": _scope_block(frames, history_start)}
    asof_d = asof or date.today()
    labels = getattr(hs, "ACCOUNT_DISPLAY", {}) or {}
    ddir = Path(frames.data_dir)

    try:
        raw_tx = pd.read_csv(ddir / "transactions.csv")
        raw_pos = pd.read_csv(ddir / "positions.csv")
        interim_path = ddir / "transactions_interim.csv"
        if interim_path.exists():
            interim = pd.read_csv(interim_path)
            if not interim.empty:
                raw_tx = pd.concat([raw_tx, interim], ignore_index=True)
        fold, _splits = load_corporate_identity()
        resolver, cusip_resolver = build_key_resolvers(raw_tx, raw_pos, fold)
        calendar = txs.wash_calendar(raw_tx, asof=asof_d, labels=labels,
                                     resolver=resolver,
                                     cusip_resolver=cusip_resolver,
                                     fold=fold,
                                     sleeve_accounts=_sleeve_account_ids())
    except (FileNotFoundError, KeyError, ValueError, TypeError,
            AttributeError, IndexError) as exc:
        return {**base, "available": False,
                "reason": f"wash_calendar_{type(exc).__name__}"}

    tv = (tax_view if tax_view is not None
          else txs.build_tax_view(frames, frames.data_dir, broker=broker,
                                  asof=asof_d))
    tax_ok = tv.get("kind") == "tax"
    tax_reason = (None if tax_ok
                  else str(tv.get("reason") or "tax_view_error")[:80])
    meta = tv.get("meta") or {}
    ledger = {"stale": bool(meta.get("stale")) if tax_ok else True,
              "stale_reason_present": (bool(meta.get("stale_reason"))
                                       if tax_ok else True)}

    # -- harvest candidates grouped by ticker ------------------------------
    hv = (tv.get("harvest") or {}) if tax_ok else {}
    cands = hv.get("candidates")
    sem = hv.get("semantics") or {}
    groups: dict[str, dict] = {}
    n_omitted = 0
    for c in cands or []:
        t = str(c.get("symbol") or c.get("instrument_key") or "")
        if not txs._ticker_like(t):
            n_omitted += 1
            continue
        g = groups.setdefault(t, {"ticker": t, "accounts": set(),
                                  "terms": set(), "wash_status": "clear",
                                  "window_ends": None,
                                  "blocking_buy_dates": set(),
                                  "ira_blocked": False})
        g["accounts"].add(txs._scrub_safe_label(
            str(c.get("account_label") or ""), {}))
        g["terms"].add(str(c.get("term") or ""))
        st = str(c.get("wash_status") or "unknown")
        if _WASH_RANK.get(st, 1) > _WASH_RANK.get(g["wash_status"], 1):
            g["wash_status"] = st                          # blocked wins
        we = c.get("window_ends")
        if we:
            g["window_ends"] = max(filter(None, [g["window_ends"], str(we)]))
        for b in c.get("blocking_buys") or []:
            if b.get("date"):
                g["blocking_buy_dates"].add(str(b["date"]))
        g["ira_blocked"] = g["ira_blocked"] or bool(c.get("is_ira_blocked"))
    harvest_ok = bool(tax_ok and cands is not None and hv.get("summary"))
    harvest = {"available": harvest_ok,
               **({"reason": tax_reason} if tax_reason else
                  ({} if harvest_ok else
                   {"reason": "harvest_scan_unavailable"})),
               "candidates": int((hv.get("summary") or {}).get("candidates")
                                 or 0),
               "candidates_omitted": int(n_omitted),
               "window_observed_pct": sem.get("window_observed_pct"),
               "clear_means": sem.get("clear_means"),
               "tickers": [txs._omit_empty(
                               {"ticker": g["ticker"],
                                "accounts": sorted(g["accounts"]),
                                "terms": sorted(g["terms"]),
                                "wash_status": g["wash_status"],
                                "window_ends": g["window_ends"],
                                "blocking_buy_dates": sorted(g["blocking_buy_dates"]),
                                "ira_blocked": g["ira_blocked"]})
                           for g in (groups[k] for k in sorted(groups))]}

    # -- lot term detail, actionable tickers only --------------------------
    lots_rows = (tv.get("lots") or []) if tax_ok else []
    keys_all = {str(r.get("symbol") or r.get("instrument_key") or "")
                for r in lots_rows} - {""}
    keys_unnamed = {k for k in keys_all if not txs._ticker_like(k)}
    per: dict[str, dict] = {}
    for r in lots_rows:
        t = str(r.get("symbol") or r.get("instrument_key") or "")
        if not txs._ticker_like(t):
            continue
        p = per.setdefault(t, {"ticker": t, "accounts": set(), "lots": 0,
                               "short_lots": 0, "long_lots": 0,
                               "marked_lots": 0,
                               "min_days_to_long_term": None,
                               "_ugl": 0.0, "_ugl_seen": False,
                               "ripening_within_60d": 0})
        p["accounts"].add(txs._scrub_safe_label(
            str(r.get("account_label") or ""), {}))
        p["lots"] += 1
        term = r.get("term")
        if term == "short":
            p["short_lots"] += 1
        elif term == "long":
            p["long_lots"] += 1
        d = r.get("days_to_long_term")
        if term == "short" and d is not None:
            p["min_days_to_long_term"] = (
                int(d) if p["min_days_to_long_term"] is None
                else min(p["min_days_to_long_term"], int(d)))
            if _lot_ripening(r):
                p["ripening_within_60d"] += 1
        ugl = r.get("unrealized_gl")
        if ugl is not None:
            p["_ugl"] += float(ugl)
            p["_ugl_seen"] = True
            p["marked_lots"] += 1
    # Sleeve rider (#380 review, −9.9 KB): a name whose only in-window
    # trades are the sleeve's own churn does not qualify via the
    # traded-in-window arm (candidates/ripening arms untouched).
    calendar_tickers = {t["ticker"] for t in calendar["tickers"]
                        if not t.get("sleeve_only")}
    actionable = (set(groups) | calendar_tickers
                  | {t for t, p in per.items() if p["ripening_within_60d"]})
    lot_rows = []
    for t in sorted(per):
        if t not in actionable:
            continue
        p = per[t]
        row = txs._omit_empty({
            "ticker": t, "accounts": sorted(p["accounts"]),
            "lots": p["lots"], "short_lots": p["short_lots"],
            "long_lots": p["long_lots"],
            "marked_lots": p["marked_lots"],
            "unrealized_direction": ((("gain" if p["_ugl"] >= 0 else "loss")
                                      if p["_ugl_seen"] else None)),
            "ripening_within_60d": p["ripening_within_60d"]})
        if p["min_days_to_long_term"] is not None:
            # 0 is meaningful here (flips long-term today) — kept outside
            # the generic zero-omission
            row["min_days_to_long_term"] = p["min_days_to_long_term"]
        lot_rows.append(row)
    lots_block = {
        "available": bool(tax_ok),
        **({"reason": tax_reason} if tax_reason else {}),
        "coverage_note": (
            f"per-ticker lot detail covers {len(lot_rows)} of "
            f"{len(keys_all) - len(keys_unnamed)} ticker-keyed positions "
            f"with open lots ({len(keys_unnamed)} cusip-keyed bills/notes "
            f"or unnamed positions excluded); actionable = harvest "
            f"candidates, lots within {_DETAIL_RIPENING_DAYS} days of "
            f"long-term, tickers traded inside the wash window — "
            f"sleeve-only traded names are not lot-actionable; absent "
            f"row keys mean zero or none; ask the Tax tab for the rest"),
        "tickers": lot_rows}

    return {**base, "available": True, "as_of": str(asof_d),
            "ledger": ledger, "wash_calendar": calendar,
            "harvest": harvest, "lots": lots_block}


def _pct_label_to_num(label) -> float | None:
    """The holdings table's percent strings ("75.2%", "—") as finite numbers
    (facts carry null, never NaN/inf)."""
    if not isinstance(label, str):
        return None
    s = label.strip().rstrip("%").replace(",", "")
    try:
        return _num(float(s), 2)
    except ValueError:
        return None


def _facts_holdings_detail(frames: hs.Frames, history_start: str,
                           broker=None, dims=None, *, asof=None) -> dict:
    """Chat-only position list (spec 2026-08-22 §4.3): ticker, account
    label, class label, weight of the scoped book, unrealized percentage
    and its direction — never quantities, values or bases. The direct-index
    sleeve and the treasury ladder stay the Holdings tab's aggregated
    lines; option contracts list under their underlying ticker."""
    base = {"section": "holdings_detail",
            "scope": _scope_block(frames, history_start)}
    view = hs.build_holdings_view(frames)
    rows = ((view.get("positions") or {}).get("rows")) or []
    positions: list[dict] = []
    omitted_weight = 0.0
    for r in rows:
        ticker = str(r.get("symbol") or "")
        klass = str(r.get("class_label") or "")
        if klass.lower().startswith("option"):
            leg = _parse_option_leg(ticker)    # OCC or JPM display symbol
            if leg is not None:
                ticker = leg[0]                # the contract -> its underlying
        if txs._TICKER_DIGIT_RUN_RE.search(ticker):
            # an identifier the scrub would reject and no underlying to
            # name: omit the row, keep its weight visible
            omitted_weight += float(_num(r.get("weight_pct"), 2) or 0.0)
            continue
        positions.append({
            "ticker": ticker,
            "account": txs._scrub_safe_label(str(r.get("account") or ""), {}),
            "class": klass,
            "weight_pct": _num(r.get("weight_pct"), 2),
            "unrealized_pct": _pct_label_to_num(r.get("ugl_pct")),
            "unrealized_direction": str(r.get("ugl_pct_dir") or "flat")})
    available = bool(positions)
    return {**base, "available": available,
            **({} if available else {"reason": "no_nameable_positions"}),
            "as_of": (view.get("meta") or {}).get("as_of"),
            "omitted_weight_pct": _num(omitted_weight, 2),
            "note": ("weights are of the scoped book at the latest snapshot; "
                     "the direct-index sleeve and the treasury ladder appear "
                     "as one aggregated line each; option contracts are listed "
                     "under their underlying ticker with their option class "
                     "(strike, expiry and direction are not distinguished here "
                     "— see the Options tab); positions whose identifier cannot "
                     "be named are omitted and their weight reported in "
                     "omitted_weight_pct; no share counts or dollar figures by "
                     "design"),
            "positions": positions}


def _facts_options(frames: hs.Frames, history_start: str,
                   broker=None, dims=None, *, asof=None, now=None) -> dict:
    """Chat-only Options Hedging summary (spec 2026-08-23 S1). Wraps
    build_options_view (empty states, aggregates -> pct-of-NAV, IV
    percentile, staleness chips) plus per-contract rows from
    _assemble_opt_tbl under option_book_aggregates' live-mask rule
    (qty != 0 and expiry >= asof). Dollar-free by construction: dollar
    aggregates stay local, only percents/counts/dates leave. ``asof`` /
    ``now`` are the test seams (pack dispatch passes neither -> today)."""
    base = {"section": "options",
            "scope": _scope_block(frames, history_start)}
    asof_d = asof or date.today()
    now_ts = now if now is not None else pd.Timestamp.now(tz="UTC")
    try:
        view = ops.build_options_view(frames, today=asof_d, now=now_ts)
    except (FileNotFoundError, KeyError, ValueError, TypeError,
            AttributeError, IndexError) as exc:
        return {**base, "available": False,
                "reason": f"options_view_{type(exc).__name__}"}

    stale_src = view.get("staleness") or {}
    staleness = {k: {"fetched_at": (stale_src.get(k) or {}).get("fetched_at"),
                     "chip": (stale_src.get(k) or {}).get("chip")}
                 for k in ("snapshot", "atm_iv")}
    if view.get("empty"):
        return {**base, "available": True, "empty": True,
                "empty_message": view.get("empty_message"),
                "staleness": staleness}

    agg_src = view.get("aggregates") or {}
    # Dollar aggregates are consumed LOCALLY to derive percents; the
    # implied NAV is the same one the view divided by (back-out from its
    # own notional/notional_pct pair), so no new seam and no drift.
    notional = agg_src.get("notional_protected")
    notional_pct = agg_src.get("notional_pct_nav")
    at_risk = agg_src.get("premium_at_risk")
    at_risk_pct = None
    if (notional and notional_pct and at_risk is not None
            and notional > 0 and notional_pct > 0):
        implied_nav = notional / (notional_pct / 100.0)
        at_risk_pct = _num(at_risk / implied_nav * 100.0, 3)
    wiv = agg_src.get("weighted_iv")
    aggregates = {
        "notional_coverage_pct": _num(notional_pct, 1),
        "premium_at_risk_pct": at_risk_pct,
        "pnl_on_cost_pct": _num(agg_src.get("pnl_pct_cost"), 1),
        "weighted_dte": _num(agg_src.get("weighted_dte"), 0),
        "weighted_iv_pct": _num(wiv * 100.0, 1) if wiv is not None else None,
        "n_live": None,          # filled from the live-mask rows below
        "n_excluded": int(agg_src.get("n_excluded") or 0),
        "greeks_missing": bool(agg_src.get("greeks_missing")),
    }

    try:
        tbl = ops._assemble_opt_tbl(frames, today=asof_d)
    except (FileNotFoundError, KeyError, ValueError, TypeError,
            AttributeError, IndexError) as exc:
        return {**base, "available": False,
                "reason": f"opt_tbl_{type(exc).__name__}"}
    qty = pd.to_numeric(tbl["quantity"], errors="coerce").fillna(0.0)
    exp = pd.to_datetime(tbl["expiry"], errors="coerce")
    live = tbl[(qty != 0) & exp.notna()
               & (exp >= pd.Timestamp(asof_d))]
    rows = []
    for _, r in live.iterrows():
        strike = r.get("strike")
        spot = r.get("spot")
        mny = None
        if pd.notna(strike) and pd.notna(spot) and float(spot) > 0:
            mny = _num((float(strike) / float(spot) - 1.0) * 100.0, 1)
        q = float(pd.to_numeric(r.get("quantity"), errors="coerce") or 0.0)
        rows.append({"underlying": str(r.get("underlying")),
                     "opt_type": str(r.get("opt_type")),
                     "side": "long" if q > 0 else "short",
                     "contracts": int(abs(q)),
                     "expiry": pd.Timestamp(r.get("expiry"))
                              .date().isoformat(),
                     "dte": int(r.get("dte") or 0),
                     "strike_vs_spot_pct": mny})
    rows.sort(key=lambda x: (x["underlying"], x["expiry"], x["opt_type"]))
    aggregates["n_live"] = len(rows)

    iv_src = view.get("iv_percentile") or {}
    return {**base, "available": True, "empty": False,
            "contracts": rows,
            "aggregates": aggregates,
            "iv_percentile": {
                "caption": iv_src.get("caption"),
                "window_days": iv_src.get("window_days"),
                "last_percentile": iv_src.get("last_percentile")},
            "staleness": staleness,
            "note": ("exposure as % of scoped NAV; dollar notionals, "
                     "premiums, and P&L never leave the machine")}


def _facts_health(frames: hs.Frames, history_start: str,
                  broker=None, dims=None, *, asof=None) -> dict:
    """Chat-only Data Health summary (spec 2026-08-23 S1, Update #1).
    Structured reconciliation verdicts from the HealthReport dataclass —
    NOT health_rows_to_table (its rows format Extracted/Reported/dollar-
    delta strings) and NOT the headline text (the red case embeds the
    literal ">$10k" threshold, which _DOLLAR_STR_RE rejects). diff_pct
    only; dollar diffs never leave the machine. ``asof`` is the test
    seam; `today` feeds only unrendered day-counts, so the section is
    calendar-stable by construction (tested)."""
    base = {"section": "health",
            "scope": _scope_block(frames, history_start)}
    asof_d = asof or date.today()
    try:
        report = hlth.build_health_report_for_frames(frames, today=asof_d)
    except (FileNotFoundError, KeyError, ValueError, TypeError,
            AttributeError, IndexError) as exc:
        return {**base, "available": False,
                "reason": f"health_report_{type(exc).__name__}"}
    level, _text = hlth.format_health_headline(report)
    rows = []
    for a in report.accounts:
        if a.state == "carried":
            verdict = "carried (lagging)" if a.lagging else "carried"
        else:
            verdict = a.band
        row = {"account": txs._scrub_safe_label(str(a.label or ""), {}),
               "broker": a.broker, "state": a.state,
               "last_verified": (str(a.last_verified_month)
                                 if a.last_verified_month else None),
               "verdict": verdict}
        if a.state != "carried":
            row["diff_pct"] = _num(a.diff_pct, 2)
        rows.append(row)
    return {**base, "available": True,
            "headline_level": level,
            "recon_available": bool(report.recon_available),
            "as_of_month": (str(report.as_of_month)
                            if report.as_of_month else None),
            "unreconciled_months": [str(m) for m in
                                    (report.unreconciled_months or [])],
            "summary": {"n_ok": int(report.n_ok),
                        "n_known": int(report.n_known),
                        "n_watch": int(report.n_watch),
                        "n_error": int(report.n_error),
                        "n_carried": int(report.n_carried),
                        "worst_level": report.worst_level},
            "accounts": rows,
            "note": ("extracted vs broker-reported reconciliation; "
                     "diff_pct only — dollar diffs never leave the "
                     "machine")}


# --------------------------------------------------------------------------- #
# Full-gate S2: fetch_detail topic reducers. Chat-only depth the pack
# truncates, served on demand inside a chat turn (spec 2026-08-23 §5).
# Same scrub law as the pack; every result passes scrub_gate before it
# reaches the API. Never registered in SECTIONS or _CHAT_SECTIONS.
# --------------------------------------------------------------------------- #
def _detail_riskcontrib(frames: hs.Frames, history_start: str,
                        broker=None, ticker=None, *, asof=None) -> dict:
    """Full per-name PCTR table (the summary's top-5 is cut from this).
    Whole table, or one ticker. Percent-only; no truncation."""
    base = {"topic": "riskcontrib",
            "scope": _scope_block(frames, history_start)}
    view = rcs.build_riskcontrib_view(frames, account="all",
                                      asset_class="all")
    state = view.get("state", {})
    if not state.get("available"):
        return {**base, "available": False,
                "reason": state.get("unavailable")}
    controls = view["controls"]
    est = controls["estimators"][0]["id"]
    combo = view["combos"][
        f"{est}|{controls['es_levels'][0]['id']}"
        f"|{controls['thresholds'][0]['id']}"]
    wp = combo["weight_vs_pctr"]
    rows = [{"ticker": str(s), "weight_pct": _num(w, 2),
             "risk_share_pct": _num(p, 2)}
            for s, w, p in zip(wp["symbols"], wp["weight"], wp["pctr"])]
    rows.sort(key=lambda r: -(r["risk_share_pct"] or 0.0))
    names_total = len(rows)      # PRE-filter table size, not len(rows)
    if ticker:
        rows = [r for r in rows if r["ticker"].upper() == ticker]
        if not rows:
            return {**base, "available": True, "found": False,
                    "ticker": ticker,
                    "note": "not in the scoped book's risk table"}
    return {**base, "available": True, "found": True, "estimator": est,
            "names_total": names_total, "rows": rows,
            "note": ("risk_share_pct = share of portfolio risk (PCTR) at "
                     "the default estimator; weight_pct = portfolio "
                     "weight; full table, no truncation; covers "
                     "price-history names only — sleeves and the "
                     "treasury ladder sit outside this table")}


def _detail_income(frames: hs.Frames, history_start: str,
                   broker=None, ticker=None, *, asof=None) -> dict:
    """All forward payers (the summary keeps only the top payer + top-5
    share): per-name share of projected 12m income, weight, yield, YoC."""
    base = {"topic": "income", "scope": _scope_block(frames, history_start)}
    asof_d = asof or date.today()
    try:
        div_hist = ins.load_div_history(Path(frames.data_dir))
        fwd_df, roll, _bts = ins.forward_rollup(
            frames.positions_monthly, div_hist, asof_d)
    except (FileNotFoundError, KeyError, ValueError, TypeError,
            AttributeError, IndexError) as exc:
        return {**base, "available": False,
                "reason": f"forward_{type(exc).__name__}"}
    if fwd_df is None or fwd_df.empty:
        return {**base, "available": True, "found": False,
                "payers_total": 0, "rows": [],
                "note": "no projected income in the scoped book"}
    payers = ins.forward_payers(fwd_df, float(roll.get("nav") or 0.0))
    tot = float(payers["projected"].sum()) if len(payers) else 0.0
    rows = []
    for _, r in payers.iterrows():
        pm = r.get("pct_mv")
        rows.append({
            "symbol": str(r.get("symbol")),
            "share_of_forward_pct": (
                _num(float(r["projected"]) / tot * 100.0, 2)
                if tot > 0 else None),
            "weight_pct": _num(pm * 100.0, 2) if pm is not None else None,
            "yield_pct": _pct100(r.get("yield_mv")),
            "yoc_pct": _pct100(r.get("yield_cost"))})
    payers_total = len(rows)     # PRE-filter payer count, not len(rows)
    if ticker:
        rows = [x for x in rows if x["symbol"].upper() == ticker]
        if not rows:
            return {**base, "available": True, "found": False,
                    "ticker": ticker, "payers_total": payers_total,
                    "note": ("no projected income for this name in the "
                             "scoped book (not held, or pays nothing)")}
    return {**base, "available": True, "found": True,
            "payers_total": payers_total, "rows": rows,
            "note": ("forward 12m projection; share_of_forward_pct = "
                     "share of total projected income; sleeves appear "
                     "as one line each")}


def _detail_lots(frames: hs.Frames, history_start: str,
                 broker=None, ticker=None, *, asof=None) -> dict:
    """Every open lot for ONE ticker (the pack's tax_detail is
    actionable-only). Dollar amounts stay local: unrealized is emitted as
    % of the lot's remaining basis; no share counts."""
    base = {"topic": "lots", "scope": _scope_block(frames, history_start),
            "ticker": ticker}
    asof_d = asof or date.today()
    tv = txs.build_tax_view(frames, frames.data_dir, broker=broker,
                            asof=asof_d)
    if tv.get("kind") != "tax":
        return {**base, "available": False,
                "reason": str(tv.get("reason") or "tax_view_error")[:80]}
    rows = []
    for r in (tv.get("lots") or []):
        t = str(r.get("symbol") or r.get("instrument_key") or "")
        if t.upper() != ticker:
            continue
        ugl = r.get("unrealized_gl")
        b = r.get("basis_remaining")
        upct = None
        if ugl is not None and b is not None:
            try:
                upct = _num(float(ugl) / float(b) * 100.0, 1) if float(b) else None
            except (TypeError, ValueError):
                upct = None
        d = r.get("days_to_long_term")
        rows.append({
            "account": txs._scrub_safe_label(
                str(r.get("account_label") or ""), {}),
            "acquired_date": (str(r.get("acquired_date"))
                              if r.get("acquired_date") is not None
                              else None),
            "term": r.get("term"),
            "days_to_long_term": int(d) if d is not None else None,
            "unrealized_pct": upct,
            # NOT "priced": the substring "price" trips _DENY_KEY_RE.
            "marked": r.get("market_value") is not None})
    if not rows:
        return {**base, "available": True, "found": False,
                "stale": bool((tv.get("meta") or {}).get("stale")),
                "note": ("no open lots for this ticker in the scoped "
                         "ledger; if stale, the ledger may lag recent "
                         "buys — see the transactions topic; taxable "
                         "accounts only — the name may be held only in "
                         "an IRA")}
    rows.sort(key=lambda x: x["acquired_date"] or "")
    return {**base, "available": True, "found": True, "n_lots": len(rows),
            "stale": bool((tv.get("meta") or {}).get("stale")),
            "rows": rows,
            "note": ("all open lots for the ticker; unrealized_pct = "
                     "unrealized gain/loss as % of the lot's remaining "
                     "basis; wash flags are not on this row — see the "
                     "transactions topic or FACTS.tax_detail; share "
                     "counts and dollar amounts never leave the machine"
                     "; taxable accounts only — IRA lots are excluded")}


def _detail_transactions(frames: hs.Frames, history_start: str,
                         broker=None, ticker=None, *, asof=None) -> dict:
    """One ticker's rows inside the trailing wash window (statement +
    interim), via the wash scanner's own keying — dates/types/accounts
    and broker wash flags only; no quantities (spec Update S2)."""
    base = {"topic": "transactions",
            "scope": _scope_block(frames, history_start), "ticker": ticker}
    asof_d = asof or date.today()
    ddir = Path(frames.data_dir)
    labels = getattr(hs, "ACCOUNT_DISPLAY", {}) or {}
    try:
        raw_tx = pd.read_csv(ddir / "transactions.csv")
        interim_path = ddir / "transactions_interim.csv"
        if interim_path.exists():
            interim = pd.read_csv(interim_path)
            if not interim.empty:
                raw_tx = pd.concat([raw_tx, interim], ignore_index=True)
        raw_pos = pd.read_csv(ddir / "positions.csv")
        fold, _splits = load_corporate_identity()
        resolver, cusip_resolver = build_key_resolvers(raw_tx, raw_pos,
                                                       fold)
    except (FileNotFoundError, KeyError, ValueError, TypeError,
            AttributeError, IndexError) as exc:
        return {**base, "available": False,
                "reason": f"ledger_{type(exc).__name__}"}
    raw_tx = raw_tx.reset_index(drop=True)
    asof_ts = pd.Timestamp(asof_d).normalize()
    lo = asof_ts - pd.Timedelta(days=txs.WINDOW_DAYS)
    if "tax_flag" in raw_tx.columns:
        flagged = (raw_tx["tax_flag"].astype(str).str.strip()
                   .str.upper().eq("W"))
    else:
        flagged = pd.Series(False, index=raw_tx.index)
    rows = []
    for types, kind in ((None, "acq"), (("sell",), "sell")):
        kw = {} if types is None else {"types": types}
        df = txs.keyed_acquisitions(raw_tx, resolver, cusip_resolver,
                                    fold, **kw)
        if df.empty:
            continue
        df = df[(df["wash_date"] >= lo) & (df["wash_date"] <= asof_ts)]
        df = df[df["instrument_key"].astype(str).str.upper() == ticker]
        for _, r in df.iterrows():
            row = {"date": str(pd.Timestamp(r["wash_date"]).date()),
                   "type": str(r["transaction_type"]),
                   "account": txs._scrub_safe_label(
                       str(r["account_id"]), labels)}
            if kind == "sell":
                row["broker_flagged_wash"] = bool(
                    flagged.reindex([r["source_row"]]).fillna(False)
                    .iloc[0])
            rows.append(row)
    rows.sort(key=lambda x: x["date"])
    if not rows:
        return {**base, "available": True, "found": False,
                "window_days": int(txs.WINDOW_DAYS),
                "as_of": str(asof_ts.date()),
                "note": ("whole book — wash rules cross accounts and "
                         "brokers; no in-window activity for this "
                         "ticker")}
    return {**base, "available": True, "found": True,
            "window_days": int(txs.WINDOW_DAYS),
            "as_of": str(asof_ts.date()), "n_rows": len(rows),
            "rows": rows,
            "note": ("whole book — wash rules cross accounts and "
                     "brokers; trailing wash window ending at as_of; "
                     "statement + interim rows; option rows excluded by "
                     "the scanner; quantities and dollar amounts never "
                     "leave the machine")}


def _detail_performance(frames: hs.Frames, history_start: str,
                        broker=None, ticker=None, *, asof=None) -> dict:
    """Full monthly TWR series + calendar-year compounding + every
    account's cumulative row (the summary keeps only top/bottom).
    ``asof`` accepted-unused: the series is statement-anchored and
    date-stable. Dollar columns on the per-account frame are never
    read."""
    base = {"topic": "performance",
            "scope": _scope_block(frames, history_start)}
    try:
        (port_view, _bf, _cf, selected_account_ids,
         account_active, _ca) = ps.twr_view_for(frames, "all", "all")
    except (FileNotFoundError, KeyError, ValueError, TypeError,
            AttributeError, IndexError) as exc:
        return {**base, "available": False,
                "reason": f"twr_{type(exc).__name__}"}
    if port_view is None or port_view.empty:
        return {**base, "available": False, "reason": "no_twr"}
    pv = port_view.copy()
    # port_view["month"] is a PeriodIndex (freq="M") — pd.to_datetime()
    # rejects PeriodDtype directly; to_timestamp() is the period-native
    # conversion (verified against the fixture before coding this).
    pv["month"] = pv["month"].dt.to_timestamp()
    monthly = [{"month": ts.strftime("%Y-%m"),
                "return_pct": _num(r * 100.0, 2)}
               for ts, r in zip(pv["month"], pv["return_pct"])]
    by_year = []
    for year, grp in pv.groupby(pv["month"].dt.year):
        # The portfolio's first tracked month has no prior NAV, so
        # compute_twr seeds return_pct=NaN there (mirrors the dropna
        # precedent at per_account_raw / compute_twr's port_valid) — an
        # inception-year NaN must not poison the whole year to null.
        vals = grp["return_pct"].dropna()
        if vals.empty:
            continue
        cum = 1.0
        for r in vals:
            cum *= (1.0 + float(r))
        by_year.append({"year": int(year),
                        "return_pct": _num((cum - 1.0) * 100.0, 2),
                        "months": int(len(vals))})
    accounts = []
    accounts_failed = False
    try:
        pa = ps.per_account_raw(frames.twr_account, frames.irr_table,
                                selected_account_ids, account_active)
        for _, r in pa.iterrows():
            accounts.append({"account": txs._scrub_safe_label(
                                 str(r["account_label"] or ""), {}),
                             "cum_twr_pct": _num(float(r["cum_twr"])
                                                 * 100.0, 1),
                             "months": int(r["months"])})
    except (KeyError, ValueError, TypeError, AttributeError):
        accounts = []
        accounts_failed = True
    return {**base, "available": True,
            "n_months": len(monthly), "monthly": monthly,
            "by_year": by_year, "accounts": accounts,
            **({"accounts_unavailable": True} if accounts_failed else {}),
            "note": ("monthly and calendar-year TWR to the last statement "
                     "month-end; per-account cumulative rows; dollar NAVs "
                     "and flows never leave the machine")}


def _detail_dip(frames: hs.Frames, history_start: str,
                broker=None, ticker=None, *, asof=None) -> dict:
    """The registered walk-forward referee table (committed artifact,
    its own ticker only) + skipped-symbol disclosure. Per-symbol verdict
    cards already ride in FACTS.dip — not repeated here. ``asof``
    accepted-unused (file-anchored)."""
    base = {"topic": "dip", "scope": _scope_block(frames, history_start)}
    try:
        hist, divs = dps._load_dip_csvs(frames.data_dir)
        art = dps._registered_artifact()
        referee = (dps._referee_block(str(art.get("ticker")))
                   if art else None)
    except (FileNotFoundError, KeyError, ValueError, TypeError,
            AttributeError, IndexError) as exc:
        return {**base, "available": False,
                "reason": f"dip_{type(exc).__name__}"}
    skipped = []
    if not hist.empty:
        for sym in sorted(hist["symbol"].unique()):
            price = dps.dip_adhoc.slice_symbol(hist, divs, sym)[0]
            if len(price) < dps.dip_adhoc.MIN_HISTORY_DAYS:
                skipped.append(str(sym))
    return {**base, "available": True,
            "referee": referee,
            "referee_ticker": (str(art.get("ticker")) if art else None),
            "skipped_symbols": skipped,
            "note": ("registered out-of-sample referee for the artifact's "
                     "own ticker only; per-symbol verdicts live in "
                     "FACTS.dip")}


def _detail_factor(frames: hs.Frames, history_start: str,
                   broker=None, ticker=None, *, asof=None) -> dict:
    """Per window x model regression detail: n/R2/adjR2, the numeric beta
    table (beta/se/t per factor), the low-observation warning, and the
    aligned-window caption. Rolling series / waterfall / attribution /
    per-holding stay out (chart bulk — the S1 iv-series rule); alpha
    rides in FACTS.factors. ``asof`` accepted-unused (date-stable
    view)."""
    base = {"topic": "factor", "scope": _scope_block(frames, history_start)}
    try:
        view = fs.build_factor_view(frames)
    except (FileNotFoundError, KeyError, ValueError, TypeError,
            AttributeError, IndexError) as exc:
        return {**base, "available": False,
                "reason": f"factor_{type(exc).__name__}"}
    # VERIFY-FIRST probe (task-2-brief): build_factor_view's window
    # container is keyed "by_window", and its empty state lives under
    # state.available/state.unavailable — NOT a top-level empty/
    # empty_reason pair. Confirmed against _facts_factors, same file.
    state = view.get("state") or {}
    if not state.get("available"):
        return {**base, "available": False,
                "reason": str(state.get("unavailable") or "unavailable")[:80]}
    windows = []
    for wlabel, wblock in (view.get("by_window") or {}).items():
        if wblock.get("aligned_empty"):
            continue
        models = []
        for mname, mb in (wblock.get("models") or {}).items():
            if not mb.get("available"):
                continue
            met = mb.get("metrics") or {}
            models.append({
                "model": str(mname),
                "n": met.get("n"), "r2": met.get("r2"),
                "adj_r2": met.get("adj_r2"),
                "low_obs_warning": mb.get("low_obs_warning"),
                "window_caption": mb.get("window_caption"),
                "betas": [{"factor": str(b.get("factor")),
                           "beta": _num(b.get("beta"), 3),
                           "se": _num(b.get("se"), 3),
                           "t": _num(b.get("t"), 2)}
                          for b in (mb.get("beta_numeric") or [])]})
        if models:
            windows.append({"window": str(wlabel), "models": models})
    if not windows:
        return {**base, "available": False, "reason": "no_fitted_models"}
    return {**base, "available": True, "windows": windows,
            "note": ("monthly-regression detail per window and model; "
                     "alpha and the summary verdict live in "
                     "FACTS.factors")}


def _detail_options_contracts(frames: hs.Frames, history_start: str,
                              broker=None, ticker=None, *, asof=None) -> dict:
    """Per-contract rows behind FACTS.options: the S1 live-mask idiom
    plus per-contract P&L as % of that contract's cost. Optional
    ``ticker`` filters by underlying. ``asof``/`now` threading matches
    S1 (DTE/liveness); `now` has no test seam here (only staleness chips
    depend on it, and this topic does not surface staleness)."""
    base = {"topic": "options_contracts",
            "scope": _scope_block(frames, history_start)}
    asof_d = asof or date.today()
    try:
        view = ops.build_options_view(frames, today=asof_d,
                                      now=pd.Timestamp.now(tz="UTC"))
    except (FileNotFoundError, KeyError, ValueError, TypeError,
            AttributeError, IndexError) as exc:
        return {**base, "available": False,
                "reason": f"options_view_{type(exc).__name__}"}
    if view.get("empty"):
        return {**base, "available": True, "empty": True,
                "empty_message": view.get("empty_message")}
    try:
        tbl = ops._assemble_opt_tbl(frames, today=asof_d)
    except (FileNotFoundError, KeyError, ValueError, TypeError,
            AttributeError, IndexError) as exc:
        return {**base, "available": False,
                "reason": f"opt_tbl_{type(exc).__name__}"}
    qty = pd.to_numeric(tbl["quantity"], errors="coerce").fillna(0.0)
    exp = pd.to_datetime(tbl["expiry"], errors="coerce")
    live = tbl[(qty != 0) & exp.notna()
               & (exp >= pd.Timestamp(asof_d))]
    rows = []
    for _, r in live.iterrows():
        strike = r.get("strike")
        spot = r.get("spot")
        mny = None
        if pd.notna(strike) and pd.notna(spot) and float(spot) > 0:
            mny = _num((float(strike) / float(spot) - 1.0) * 100.0, 1)
        q = float(pd.to_numeric(r.get("quantity"), errors="coerce") or 0.0)
        mv = r.get("market_value")
        cb = r.get("cost_basis_total")
        pnl_pct = None
        if mv is not None and cb is not None:
            try:
                pnl_pct = (_num((float(mv) / float(cb) - 1.0) * 100.0, 1)
                           if float(cb) else None)
            except (TypeError, ValueError):
                pnl_pct = None
        rows.append({"underlying": str(r.get("underlying")),
                     "opt_type": str(r.get("opt_type")),
                     "side": "long" if q > 0 else "short",
                     "contracts": int(abs(q)),
                     "expiry": pd.Timestamp(r.get("expiry"))
                              .date().isoformat(),
                     "dte": int(r.get("dte") or 0),
                     "strike_vs_spot_pct": mny,
                     "pnl_on_cost_pct": pnl_pct})
    live_total = len(rows)       # PRE-filter live-contract count
    if ticker:
        rows = [x for x in rows if x["underlying"].upper() == ticker]
        if not rows:
            return {**base, "available": True, "empty": False,
                    "found": False, "ticker": ticker,
                    "live_total": live_total,
                    "note": "no live contracts under this underlying"}
    rows.sort(key=lambda x: (x["underlying"], x["expiry"], x["opt_type"]))
    return {**base, "available": True, "empty": False, "found": True,
            "live_total": live_total, "contracts": rows,
            "note": ("live contracts only; pnl_on_cost_pct is per "
                     "contract vs its own cost; dollar values never "
                     "leave the machine")}


def _detail_health(frames: hs.Frames, history_start: str,
                   broker=None, ticker=None, *, asof=None) -> dict:
    """The S1 health section's rows plus per-account ``lagging`` — the
    full non-dollar reconciliation surface. Structured fields only (the
    S1 headline-text rule); labels through the scrub-safe guard."""
    base = {"topic": "health", "scope": _scope_block(frames, history_start)}
    asof_d = asof or date.today()
    try:
        report = hlth.build_health_report_for_frames(frames, today=asof_d)
    except (FileNotFoundError, KeyError, ValueError, TypeError,
            AttributeError, IndexError) as exc:
        return {**base, "available": False,
                "reason": f"health_report_{type(exc).__name__}"}
    level, _text = hlth.format_health_headline(report)
    rows = []
    for a in report.accounts:
        if a.state == "carried":
            verdict = "carried (lagging)" if a.lagging else "carried"
        else:
            verdict = a.band
        row = {"account": txs._scrub_safe_label(str(a.label or ""), {}),
               "broker": a.broker, "state": a.state,
               "lagging": bool(a.lagging),
               "last_verified": (str(a.last_verified_month)
                                 if a.last_verified_month else None),
               "verdict": verdict}
        if a.state != "carried":
            row["diff_pct"] = _num(a.diff_pct, 2)
        rows.append(row)
    return {**base, "available": True,
            "headline_level": level,
            "recon_available": bool(report.recon_available),
            "as_of_month": (str(report.as_of_month)
                            if report.as_of_month else None),
            "unreconciled_months": [str(m) for m in
                                    (report.unreconciled_months or [])],
            "summary": {"n_ok": int(report.n_ok),
                        "n_known": int(report.n_known),
                        "n_watch": int(report.n_watch),
                        "n_error": int(report.n_error),
                        "n_carried": int(report.n_carried),
                        "worst_level": report.worst_level},
            "accounts": rows,
            "note": ("full reconciliation rows incl. lagging; diff_pct "
                     "only — dollar diffs never leave the machine")}


_DETAIL_TOPICS: dict = {"riskcontrib": _detail_riskcontrib,
                        "income": _detail_income,
                        "lots": _detail_lots,
                        "transactions": _detail_transactions,
                        "performance": _detail_performance,
                        "dip": _detail_dip,
                        "factor": _detail_factor,
                        "options_contracts": _detail_options_contracts,
                        "health": _detail_health}
_DETAIL_NEEDS_TICKER = {"lots", "transactions"}


def normalize_detail_ticker(ticker) -> str | None:
    """The ONE ticker normalization for fetch_detail: upper-cased and
    stripped, None for blank/missing. Shared by run_detail's dispatch and
    the chat route's per-turn success memo, so "strl" and "STRL" are one
    memo entry (the raw-key memo double-paid ~20 s reducers — #382
    rider)."""
    return (str(ticker).strip().upper()
            if ticker and str(ticker).strip() else None)


def run_detail(frames: hs.Frames, history_start: str, broker,
               topic: str, ticker, *, asof=None) -> dict:
    """fetch_detail dispatch: normalize the ticker, route to the topic
    reducer, scrub-gate the result. Unknown topic / missing required
    ticker return a structured {"error": ...} (the loop turns it into an
    is_error tool_result); a scrub rejection RAISES (fail the turn
    closed — the pack contract)."""
    fn = _DETAIL_TOPICS.get(topic)
    if fn is None:
        return {"error": f"unknown topic {topic!r}"}
    t = normalize_detail_ticker(ticker)
    if t and (_DIGIT_RUN_RE.search(t) or _DOLLAR_STR_RE.search(t)):
        return {"error": "not a ticker-shaped symbol"}
    if topic in _DETAIL_NEEDS_TICKER and not t:
        return {"error": "ticker required for this topic"}
    return scrub_gate(fn(frames, history_start, broker, t, asof=asof))


def _facts_performance(frames: hs.Frames, history_start: str,
                       broker=None, dims=None) -> dict:
    """Performance facts — the tab's own raw seams (twr_view_for +
    headline_raw / cashflows_raw / per_account_raw), reduced to percent/
    ratio/count values. Dollar channels (flow sums, NAV columns) are
    deliberately absent; only counts and direction words survive (the
    tax-box precedent). B2 filter idiom: account/class arrive via ``dims``
    and route through the SAME resolve+synthesize path
    build_performance_view uses — box==tab by construction. IRR mirrors
    the tab exactly: headline_raw returns NaN under a Holdings filter, so
    irr_pct is null precisely when the tab shows "—"."""
    d = dims or {}
    account = d.get("account") or "all"
    asset_class = d.get("asset_class") or "all"
    acct_label, class_label = _filter_labels(frames, account, asset_class)
    base = {"section": "performance",
            "scope": _scope_block(frames, history_start, acct_label,
                                  class_label)}
    (port_view, _bf, _cf, selected_account_ids,
     account_active, class_active) = ps.twr_view_for(frames, account,
                                                     asset_class)
    holdings_filter_active = account_active or class_active
    stub = None if holdings_filter_active else hs.interim_stub(frames)
    hd = ps.headline_raw(port_view, frames.irr_table, holdings_filter_active,
                         stub=stub)
    if hd is None:
        return {**base, "available": False, "reason": "no_twr"}

    irr_pct = _pct100(hd["irr"]) if hd["irr"] == hd["irr"] else None
    ann_pct = _pct100(hd["ann"])
    headline = {
        "cum_twr_pct": _pct100(hd["cum"]),
        "ann_twr_pct": ann_pct,
        "n_months": int(hd["n"]),
        "irr_pct": irr_pct,
        # Named for its computation (IRR − TWR); the shipped B3 name
        # twr_irr_gap_pp read backwards (final-review Minor, renamed here).
        "irr_minus_twr_pp": (_num(irr_pct - ann_pct)
                             if irr_pct is not None and ann_pct is not None
                             else None),
        "best_month_pct": _pct100(hd["best_ret"], 1),
        "best_month": (hd["best_month"].strftime("%b %Y")
                       if hd["best_month"] is not None else None),
        "worst_month_pct": _pct100(hd["worst_ret"], 1),
        "worst_month": (hd["worst_month"].strftime("%b %Y")
                        if hd["worst_month"] is not None else None),
    }
    if "cum_to_date" in hd:
        # Provisional stub period (spec 2026-08-22): additive block, the
        # statement-basis keys above stay as they are.
        headline["to_date"] = {
            "end": hd["to_date"],
            "cum_twr_pct": _pct100(hd["cum_to_date"]),
            "ann_twr_pct": _pct100(hd["ann_to_date"]),
            "stub_return_pct": _pct100(hd["stub"].return_pct),
            "stub_days": int(hd["stub"].days),
            "provisional": True,
        }
    drawdown = {
        # twr_dd_pct is already in percent units (headline_raw's mdd too).
        "current_dd_pct": _num(float(port_view["twr_dd_pct"].iloc[-1])),
        "max_dd_pct": _num(hd["mdd"]),
        "max_dd_month": pd.Timestamp(
            hd["max_dd_month"].to_timestamp()).strftime("%b %Y"),
    }
    valid = port_view.dropna(subset=["return_pct"])
    periodic = {"n_months": int(len(valid))}
    if len(valid):
        wins = int((valid["return_pct"] >= 0).sum())
        periodic.update(n_wins=wins,
                        win_rate_pct=_num(wins / len(valid) * 100.0, 1))
        yr_ret, yr_dt = ps.aggregate_periodic_returns(
            valid["return_pct"], valid["month_end"], "Y")
        years = [(pd.Timestamp(dt2).strftime("%Y"), float(v))
                 for dt2, v in zip(yr_dt, yr_ret) if v == v]
        if years:
            by = max(years, key=lambda t: t[1])
            wy = min(years, key=lambda t: t[1])
            periodic.update(best_year=by[0], best_year_pct=_pct100(by[1], 1),
                            worst_year=wy[0],
                            worst_year_pct=_pct100(wy[1], 1))
    out = {**base, "available": True, "headline": headline,
           "drawdown": drawdown, "periodic": periodic,
           "synthesized_twr": bool(holdings_filter_active)}
    cf = ps.cashflows_raw(frames.transactions, port_view,
                          selected_account_ids, account_active, class_active)
    if cf is not None:   # hidden under a class filter — mirror the tab
        net = cf["net"]
        out["cashflows"] = {
            "n_deposits": int(cf["n_deposits"]),
            "n_withdrawals": int(cf["n_withdrawals"]),
            "net_flow_direction": ("inflow" if net > 0 else
                                   "outflow" if net < 0 else "flat"),
            "synthetic_onboarding": bool(cf["synth_total"] > 0),
        }
    pa = ps.per_account_raw(frames.twr_account, frames.irr_table,
                            selected_account_ids, account_active)
    if not pa.empty:     # sorted cum-TWR descending by construction
        top, bot = pa.iloc[0], pa.iloc[-1]
        out["per_account"] = {
            "n_accounts": int(len(pa)),
            "top": {"account": str(top["account_label"]),
                    "cum_twr_pct": _pct100(top["cum_twr"], 1),
                    "months": int(top["months"])},
            "bottom": {"account": str(bot["account_label"]),
                       "cum_twr_pct": _pct100(bot["cum_twr"], 1),
                       "months": int(bot["months"])},
        }
    # v2-S4: per-position return contribution (approximate — total-return
    # basis, current weights; the engine documents both). Weights come from
    # the SAME filtered bundle seam the risk reducer uses (B2: box==tab on
    # filtered scopes). Generating-path-only cost.
    contributors: dict = {"available": False, "reason": "unavailable"}
    try:
        b4 = (rss._bundle_for(frames, account, asset_class)
              if holdings_filter_active
              else rs._bundle(frames, None, None, False, False))
        w4 = b4.get("weights")
    except Exception:                    # bundle unavailable on thin slices
        w4 = None
    if w4 is not None and len(w4) and not frames.daily_prices.empty:
        attrib = position_return_contribution(
            frames.daily_prices, w4,
            {"60d": 60, "ytd": "ytd", "252d": 252})

        def _rows(df, tail=False):
            part = df.tail(3).iloc[::-1] if tail else df.head(3)
            return [{"ticker": str(sym),
                     "weight_pct": _num(r["weight_pct"], 1),
                     "return_pct": _num(r["return_pct"], 1),
                     "contrib_pp": _num(r["contrib_pp"], 2)}
                    for sym, r in part.iterrows()]

        wins = {}
        for wl, df in attrib.items():
            if df.empty:
                wins[wl] = {"available": False}
                continue
            top_rows = _rows(df)
            top_set = {r["ticker"] for r in top_rows}
            bottom_rows = [r for r in _rows(df, tail=True)
                           if r["ticker"] not in top_set]
            wins[wl] = {"top": top_rows, "bottom": bottom_rows,
                        "excluded_weight_pct": _num(
                            df.attrs.get("excluded_weight_pct"), 1)}
            n_nan = int(df.attrs.get("n_dropped_nan_weights") or 0)
            if n_nan:
                wins[wl]["n_dropped_nan_weights"] = n_nan
        contributors = {
            "available": True,
            "method": ("approximate: total-return basis (distributions "
                      "reinvested at the ex-date, split-scaled), current "
                      "weights held constant"),
            "windows": wins}
    out["contributors"] = contributors
    return out


def _facts_benchmark(frames: hs.Frames, history_start: str,
                     broker=None, dims=None) -> dict:
    """Benchmark-tab facts — the vs-benchmark trailing-window returns table
    reduced to dollar-free percent/pp values, read from the SAME
    benchmark_service seam the tab renders (box==tab). account/asset_class
    arrive via ``dims`` (the server merges them for filter-threaded
    sections) and route through build_benchmark_view unchanged, so the box
    narrates exactly the filtered slice the table shows. Dollar channels
    (growth-of-$100k, wealth headline) are deliberately omitted."""
    d = dims or {}
    bench = d.get("benchmark") or "auto"
    if bench not in {"auto", "spy", "60_40"}:
        raise AIDimError(f"unknown benchmark {bench!r}")
    account = d.get("account") or "all"
    asset_class = d.get("asset_class") or "all"
    acct_label, class_label = _filter_labels(frames, account, asset_class)
    view = bs.build_benchmark_view(frames, account=account,
                                   asset_class=asset_class, benchmark=bench)
    meta = view["meta"]
    b = meta["benchmark"]
    base = {"section": "benchmark",
            "scope": _scope_block(frames, history_start, acct_label, class_label),
            "benchmark": {"id": b["id"], "label": b["label"], "short": b["short"],
                          # named to dodge the "nav" substring the facts
                          # dollar-scrub test bans (it also matches inside
                          # "unavailable") while keeping the same semantics
                          # as the view's meta.benchmark.unavailable_fallback.
                          "fallback_to_spy": bool(b["unavailable_fallback"])},
            "holdings_filter_active": bool(meta["holdings_filter_active"])}
    if meta["state"] != "ok" or not view.get("returns_table"):
        return {**base, "available": False, "reason": meta["state"]}

    def _p(v, nd=2):
        return None if v is None else _num(float(v) * 100.0, nd)

    rows = []
    for r in view["returns_table"]["rows"]:
        if not r["available"]:
            rows.append({"window": r["label"], "annualized": bool(r["annualized"]),
                         "available": False})
            continue
        row = {"window": r["label"], "annualized": bool(r["annualized"]),
               "available": True,
               "port_return_pct": _p(r["port"]),
               "bench_return_pct": _p(r["bench"]),
               "spread_pp": _p(r["spread"]),
               "port_vol_pct": _p(r["port_vol"]),
               "bench_vol_pct": _p(r["bench_vol"])}
        if r.get("provisional"):
            # Provisional stub period (spec 2026-08-22): additive to-date keys.
            row.update({"port_to_date_pct": _p(r["port_to_date"]),
                        "bench_to_date_pct": _p(r["bench_to_date"]),
                        "spread_to_date_pp": _p(r["spread_to_date"]),
                        "to_date": r["to_date"], "provisional": True})
        rows.append(row)
    # Rolling-12mo consistency (hit rate / TE / IR): reuse the SAME aligned
    # monthly pair the view's own periodic-monthly chart already computed
    # from port_view/tr_lookup (view["periodic"]["monthly"]), rather than
    # re-deriving one from frames.twr_portfolio (_facts_portfolio's route) —
    # that whole-book series would be wrong under an account/asset_class
    # filter. This seam is already filter-consistent by construction (built
    # from the same possibly-filtered port_view above), so no separate
    # filtered-scope gate is needed here: a scope too thin to align lands in
    # the meta["state"] != "ok" early-return above, before this line runs;
    # a scope that reaches "ok" but has <12 aligned months degrades honestly
    # via rolling_active_stats' own available:False/n_months branch.
    mo = (view.get("periodic") or {}).get("monthly") or {}
    port_m = pd.Series({r["x"]: r["v"] / 100.0 for r in mo.get("port") or []})
    bench_m = pd.Series({r["x"]: r["v"] / 100.0 for r in mo.get("bench") or []})
    cons = rolling_active_stats(port_m, bench_m, window=12)
    cons["window_months"] = 12
    if not cons["available"]:
        cons.setdefault("reason", "too_short")  # matches _facts_risk's term

    w = meta.get("window") or {}
    # DA-C-10: when the view refused the stub because the benchmark series
    # stops short (its caption names the gap and no row is provisional),
    # relay the reason so the box doesn't sit statement-anchored next to a
    # provisional Performance box with no explanation.
    gap_cap = (view.get("returns_table") or {}).get("caption")
    return {**base, "available": True,
            "window": {"n_months": w.get("n_months"), "years": w.get("years"),
                       "start": w.get("start"), "end": w.get("end")},
            "returns": rows, "consistency": cons,
            **({"to_date_unavailable": gap_cap}
               if gap_cap and "stub" not in view else {})}


_PERFORMANCE_INSTRUCTION = (
    "Answer: how has this book performed, and where did the return come "
    "from? Name the broker scope and history window from FACTS.scope in "
    "your first sentence. Distinguish time-weighted return "
    "(headline.cum_twr_pct / ann_twr_pct — investment performance with "
    "deposits stripped out) from money-weighted IRR (headline.irr_pct) "
    "whenever both are present, and note irr_minus_twr_pp (IRR minus TWR "
    "— positive means the money-weighted return is ahead), citing "
    "cashflows.synthetic_onboarding as a partial explanation when true. "
    "Cover the drawdown state (current_dd_pct vs max_dd_pct and its "
    "month), the win rate with best/worst month and year, and the "
    "per-account spread (top vs bottom account by cumulative TWR — name "
    "the accounts). If headline.irr_pct is null, say the tab hides IRR "
    "under this filter — never estimate one. If synthesized_twr is true, "
    "state that filtered returns are synthesized from daily returns "
    "within each statement window and can shift slightly from the "
    "unfiltered view. Fields that are null are unavailable — say so "
    "plainly. "
    "If FACTS.scope names an account or asset_class filter, name that "
    "filter in your first sentence too — the narration describes that "
    "filtered slice, not the whole book. "
    "FACTS.contributors names the top/bottom return contributors per "
    "window (contrib_pp = weight × total return, distributions "
    "reinvested): cite the leaders and laggards for at least one window, "
    "ALWAYS with the basis caveat — approximate, total return, current "
    "weights. If excluded_weight_pct is nonzero, say that weight was "
    "excluded. ")


_BENCHMARK_INSTRUCTION = (
    "Answer: how has this book done versus its benchmark? "
    "FACTS.benchmark.label names the benchmark (SPY, or a 60/40 SPY/AGG "
    "blend) — NAME it in your first sentence, with the broker scope and "
    "history window from FACTS.scope. Walk the trailing-window returns "
    "(FACTS.returns): for each available window give the portfolio return, "
    "the benchmark return, and the spread in percentage points (spread_pp; "
    "positive means the book is ahead), and compare the portfolio's "
    "annualized volatility to the benchmark's (port_vol_pct vs "
    "bench_vol_pct) — note where the book took more or less risk than the "
    "benchmark for the return it earned. Call out the windows where the "
    "book led vs lagged and any disagreement between short and long "
    "windows. If FACTS.benchmark.fallback_to_spy is true, note the "
    "60/40 blend fell back to SPY because AGG data was unavailable. "
    "Windows marked available:false are not fully covered — say so, never "
    "extrapolate. If FACTS.holdings_filter_active is true (FACTS.scope "
    "names an account or asset-class filter), state in sentence one that "
    "this compares the filtered slice, not the whole book. Describe only "
    "— never recommend buying, selling, or switching benchmarks. "
    "FACTS.consistency reports rolling-12-month reliability: hit_rate_pct "
    "(share of rolling years the book beat the benchmark), "
    "tracking_error_pct, and information_ratio — say whether the book "
    "beats reliably or in bursts. If consistency.available is false, "
    "skip it. ")


def _facts_frontier(frames: hs.Frames, history_start: str,
                    broker=None, dims=None) -> dict:
    """Efficient-frontier facts — read from the MEMOIZED frontier result (the
    ~80s sweep is never re-run here) keyed by the ``sig`` the frontier POST
    returned and the FE echoes. Dollar-free by construction: vols, expected
    returns (CAPM estimates), a gap in pp, effective-N, and the rf/ERP/assumed-
    beta caveat. A cold memo (server restarted since the run) -> available:false,
    reason:"stale"."""
    d = dims or {}
    memo = rss.frontier_memo_get(d.get("sig"))
    if not memo:
        return {"section": "frontier",
                "scope": _scope_block(frames, history_start),
                "available": False, "reason": "stale"}
    payload = memo["payload"] or {}
    acct_label, class_label = _filter_labels(frames, memo["account"],
                                             memo["asset_class"])
    base = {"section": "frontier",
            "scope": _scope_block(frames, history_start, acct_label, class_label)}
    series = (payload.get("series") or [{}])[0].get("points") or []
    cur = payload.get("current")
    if payload.get("error") or not series or not cur:
        return {**base, "available": False,
                "reason": "error" if payload.get("error") else "empty"}

    def _p(v):
        return _num(None if v is None else float(v) * 100.0)

    capm = payload.get("capm") or {}
    capm_facts = {"rf_pct": _num(capm.get("rf_pct")),
                  "rf_src": capm.get("rf_src"),
                  "erp_pct": _num(capm.get("erp_pct")),
                  "beta_years": capm.get("beta_years"),
                  "assumed_beta_names": list(capm.get("assumed_beta_names") or [])}
    best = min(series, key=lambda p: abs(p["vol"] - cur["vol"]))
    gap_pp = _num((best["exp_return"] - cur["exp_return"]) * 100.0)
    vols = [p["vol"] for p in series]
    ers = [p["exp_return"] for p in series]
    lo = series[vols.index(min(vols))]
    hi = series[ers.index(max(ers))]
    return {**base, "available": True, "capm": capm_facts,
            "current": {"vol_pct": _p(cur["vol"]),
                        "exp_return_pct": _p(cur["exp_return"]),
                        "effective_n": _num(cur.get("effective_n"))},
            "at_your_vol": {"point_vol_pct": _p(best["vol"]),
                            "frontier_exp_return_pct": _p(best["exp_return"]),
                            "gap_pp": gap_pp,
                            "effective_n": _num(best.get("effective_n")),
                            "max_weight_pct": _p(best.get("max_weight"))},
            "range": {"min_vol_pct": _p(lo["vol"]),
                      "min_vol_exp_return_pct": _p(lo["exp_return"]),
                      "max_return_pct": _p(hi["exp_return"]),
                      "max_return_vol_pct": _p(hi["vol"]),
                      "n_points": len(series)},
            "markers": [{"key": m["key"], "label": m["label"],
                         "vol_pct": _p(m["vol"]),
                         "exp_return_pct": _p(m["exp_return"])}
                        for m in (payload.get("markers") or [])],
            "skipped_n": payload.get("skipped_n", 0)}


_FRONTIER_INSTRUCTION = (
    "Answer: what does this efficient frontier show, and where does the book "
    "sit on it? In sentence one name the broker/history scope from FACTS.scope "
    "(and any account/asset_class filter it lists) and say the frontier is the "
    "highest expected return achievable at each volatility level under the live "
    "per-name cap and class floors/caps. State plainly that expected returns "
    "are CAPM ESTIMATES (risk-free rate + beta x equity risk premium): cite "
    "FACTS.capm.erp_pct as the ERP and, if FACTS.capm.assumed_beta_names is "
    "non-empty, name those holdings as ones whose beta was assumed 1.0 (a "
    "modelling caveat, not a measured fact). Describe where the current book "
    "sits (FACTS.current.vol_pct / exp_return_pct) and the gap at your vol "
    "(FACTS.at_your_vol.gap_pp — positive means the frontier offers about that "
    "many more percentage points of expected return at your current risk "
    "level), and note the concentration trade-off (FACTS.at_your_vol.effective_n "
    "vs FACTS.current.effective_n — the frontier point typically holds fewer "
    "names). Give the range from FACTS.range (min-variance vol/return to the "
    "max-return corner). If FACTS.skipped_n > 0, mention some points were "
    "infeasible under the constraints. Values that are null are unavailable — "
    "say so, never estimate. Describe only — never recommend buying, selling, "
    "or a specific reweight.")


def _facts_risksim(frames: hs.Frames, history_start: str,
                   broker=None, dims=None) -> dict:
    """What-if reweight facts — read from the MEMOIZED simulate result (the
    before/after risk is never re-computed here) keyed by the ``sig`` the
    /simulate POST returned and the FE echoes. Dollar-free by construction (the
    memoized block is already unit-normalized numeric). A cold memo (server
    restarted since the run, or an error run that never memoized) ->
    available:false, reason:"stale"."""
    d = dims or {}
    memo = rss.simulate_memo_get(d.get("sig"))
    if not memo:
        return {"section": "risksim",
                "scope": _scope_block(frames, history_start),
                "available": False, "reason": "stale"}
    acct_label, class_label = _filter_labels(frames, memo["account"],
                                             memo["asset_class"])
    # Spread facts FIRST so the control keys below always win — memo["facts"]
    # is produced by a different module (_simulate_ai_facts), so guarding
    # against a future key collision here is free defensive margin.
    return {**memo["facts"],
            "section": "risksim",
            "scope": _scope_block(frames, history_start, acct_label, class_label),
            "available": True}


_RISKSIM_INSTRUCTION = (
    "Answer: what does this simulated reweight do to the portfolio's risk? In "
    "sentence one name the broker/history scope from FACTS.scope (and any "
    "account/asset_class filter it lists) and state plainly that this is a "
    "HYPOTHETICAL what-if reweight the user is modelling — describe the modelled "
    "before -> after change, do not treat it as a real position. Walk the risk "
    "block: portfolio volatility (FACTS.vol_pct before/after/delta, annualized "
    "percent), Sharpe and Sortino (FACTS.sharpe / FACTS.sortino). Then "
    "diversification: Diversification Ratio (FACTS.dr), down-beta vs SPY "
    "(FACTS.down_beta), and Effective N (FACTS.effective_n). Then concentration: "
    "top-5 and max single-name weight (FACTS.top5_pct / FACTS.max_pct, already "
    "in percent). Then the tail: max drawdown, 95% VaR and CVaR "
    "(FACTS.max_dd_pct / var95_pct / cvar95_pct, daily where noted). Then stress: "
    "conditional average correlation and stressed DR (FACTS.stressed_corr_avg / "
    "stressed_dr). For each, say whether the after-state is higher or lower and "
    "whether that is an improvement: LOWER volatility, concentration, down-beta, "
    "VaR/CVaR and drawdown magnitude are better; HIGHER Effective N and "
    "Diversification Ratio are better. Name the biggest position changes from "
    "FACTS.weight_moves (ticker, before_pct -> after_pct). For each entry in "
    "FACTS.candidates (each a not-yet-held name being considered), report its "
    "MCR verdict (entry.verdict — diversifying / neutral / risk-adding) as a "
    "modelling read, not advice. Values that are null are unavailable — say so, "
    "never estimate. Describe only — never recommend buying, selling, or a "
    "specific reweight.")


def _facts_income(frames: hs.Frames, history_start: str,
                  broker=None, dims=None, *, asof=None) -> dict:
    """Income facts — fully dollar-free by construction (the tab is the
    most dollar-dense in the app): yields, coverage, shares, ratios,
    counts, and dates only. Whole-book like the tab (`/api/income` takes
    no account/class); ``dims`` account/class are ignored on purpose.
    ``asof`` is a TEST-ONLY keyword (the SECTIONS dispatcher passes
    positionals) pinning the today-dependent windows, mirroring
    build_income_view's own asof seam."""
    base = {"section": "income", "scope": _scope_block(frames, history_start)}
    transactions = frames.transactions
    inc_ts = ins.income_timeseries(transactions)
    div_hist = ins.load_div_history(Path(frames.data_dir))
    today_date = asof or date.today()
    today_ts = pd.Timestamp(today_date).normalize()
    fwd_df, roll, book_ts = ins.forward_rollup(frames.positions_monthly,
                                               div_hist, today_date)

    received = {"available": not bool(inc_ts.empty)}
    if received["available"]:
        # Trailing-12M month-bucket slice (same (asof-365d, asof] bounds as
        # trailing_income, applied to inc_ts's month-start index — the same
        # monthly-bucketed basis the tab's components chart displays).
        t12 = inc_ts[(inc_ts.index > today_ts - pd.Timedelta(days=365))
                     & (inc_ts.index <= today_ts)]
        div_s = float(t12["dividends"].sum()) if "dividends" in t12 else 0.0
        int_s = float(t12["interest"].sum()) if "interest" in t12 else 0.0
        wh_s = (float(t12["withholding"].sum())
                if "withholding" in t12 else 0.0)
        gross = abs(div_s) + abs(int_s)
        if gross > 0:
            received.update(
                dividends_share_pct=_num(abs(div_s) / gross * 100.0, 1),
                interest_share_pct=_num(abs(int_s) / gross * 100.0, 1),
                withholding_of_gross_pct=_num(abs(wh_s) / gross * 100.0, 1))
        if div_hist:
            t12m_actual = ins.trailing_income(transactions, today_ts)
            proj = float(roll.get("projected_12m") or 0.0)
            if t12m_actual > 0:
                received["projected_vs_ttm_ratio"] = _num(proj / t12m_actual, 2)
        # Latest-full-year growth (the B3-descoped trend fact, TK-approved
        # definition 2026-08-14): a calendar year is FULL only if the
        # transaction ledger already covered its Jan 1 — the partial first
        # year would otherwise overstate growth — and it precedes asof's
        # year. Needs two consecutive full years and a positive prior-year
        # total; omitted otherwise (short histories stay honest).
        when = pd.to_datetime(transactions["settlement_date"],
                              errors="coerce")
        if "trade_date" in transactions.columns:
            when = when.fillna(pd.to_datetime(transactions["trade_date"],
                                              errors="coerce"))
        first_txn = when.min()
        latest_full = today_ts.year - 1
        if (pd.notna(first_txn)
                and first_txn <= pd.Timestamp(latest_full - 1, 1, 1)):
            yr = inc_ts["net"].groupby(inc_ts.index.year).sum()
            cur = float(yr.get(latest_full, 0.0))
            prev = float(yr.get(latest_full - 1, 0.0))
            if prev > 0:
                received["latest_full_year_growth_pct"] = _num(
                    (cur / prev - 1.0) * 100.0, 1)

    forward = {"available": bool(div_hist)}
    if forward["available"]:
        forward.update(
            yield_on_covered_mv_pct=_pct100(roll["yield_on_covered_mv"]),
            yield_on_covered_cost_pct=_pct100(roll["yield_on_covered_cost"]),
            coverage_of_book_pct=_pct100(roll["coverage_pct_nav"], 1))
        if not fwd_df.empty:
            payers = ins.forward_payers(fwd_df, roll["nav"])
            forward["n_payers"] = int(len(payers))
            tot = float(payers["projected"].sum()) if len(payers) else 0.0
            if len(payers) and tot > 0:
                top = payers.iloc[0]
                forward.update(
                    top_payer=str(top["symbol"]),
                    top_payer_share_pct=_num(
                        float(top["projected"]) / tot * 100.0, 1),
                    top5_share_pct=_num(
                        float(payers["projected"].head(5).sum())
                        / tot * 100.0, 1))
        else:
            forward["n_payers"] = 0
        ex_max = ins.latest_ex_date_through(div_hist, today_ts)
        if ex_max is not None:
            forward["history_through"] = ex_max.strftime("%b %d, %Y")

    out = {**base, "available": True, "received": received,
           "forward": forward}
    if book_ts is not None:
        out["book_date"] = pd.Timestamp(book_ts).strftime("%b %d, %Y")
    return out


_INCOME_INSTRUCTION = (
    "Factual income-posture summary ONLY — you must never recommend "
    "buying, selling, or reaching for yield; describe what is. Name the "
    "broker scope from FACTS.scope in your first sentence. FACTS carries "
    "no dollar amounts — cite only the yields, shares, ratios, counts, "
    "and dates present. Cover: the forward yield on covered market value "
    "and on cost (name the covered-MV basis explicitly), and "
    "coverage_of_book_pct — state plainly that the uncovered remainder "
    "(options, cash, unfetched tickers) sits outside these yields; payer "
    "concentration (top_payer with top_payer_share_pct, top5_share_pct, "
    "n_payers); the received mix (dividends vs interest shares and the "
    "withholding drag — received dividends include return-of-capital "
    "distributions on held shares; ROC is tax character, not a different "
    "kind of yield); and projected_vs_ttm_ratio — above 1 means the "
    "model projects more than the trailing year actually paid. "
    "latest_full_year_growth_pct, when present, compares the latest "
    "complete calendar year of received income to the prior complete "
    "year (both fully covered by the ledger); when absent, the history "
    "is too short for a full-year comparison — do not infer a trend. If "
    "received.available or forward.available is false, say which half is "
    "unavailable and how that limits the read. Cite book_date / "
    "history_through when discussing staleness.")


def _facts_dip(frames: hs.Frames, history_start: str,
               broker=None, dims=None) -> dict:
    """Buy-the-Dip facts — whole-market. Re-runs the SAME per-symbol
    engine orchestration the tab's cards use (dip_service.dip_card_data)
    and reads its raw fields; no display strings parsed (the _facts_risk
    recompute precedent). Broker/history_start ride in scope for plumbing
    uniformity but never change these facts — the cards are whole-market
    by construction, and ``dims`` account/class are ignored on purpose.
    Cost: one extra card build per cold generation (async path only)."""
    base = {"section": "dip", "scope": _scope_block(frames, history_start)}
    hist, divs = dps._load_dip_csvs(frames.data_dir)
    if hist.empty:
        return {**base, "available": False, "reason": "no_dip_history"}
    watch = sorted(hist["symbol"].unique(),
                   key=lambda s: (s not in dps._TRIO, s))
    vintage = str(pd.to_datetime(hist["date"]).max().date())
    symbols, skipped = [], []
    for sym in watch:
        price, tr, dser = dps.dip_adhoc.slice_symbol(hist, divs, sym)
        if len(price) < dps.dip_adhoc.MIN_HISTORY_DAYS:
            skipped.append(str(sym))
            continue
        d = dps.dip_card_data(sym, price, tr, dser)
        v, st = d.verdict, d.state
        om = v["omega"]
        symbols.append({
            "ticker": str(sym),
            "band": str(v["band"]),
            "n_outcomes": int(v["n"]),
            "omega": _num(om, 1),                    # None when inf/NaN
            "omega_infinite": bool(om == float("inf")),
            "baseline_omega": _num(v["baseline_omega"], 1),
            "omega_ci_lo": _num(v["omega_ci"]["lo"], 1),
            "omega_ci_hi": _num(v["omega_ci"]["hi"], 1),
            "current_dd_pct": _num(st["current_dd"] * 100.0, 1),
            "pct_history_shallower": _num(st["pct_history_shallower"], 1),
            "n_episodes_this_deep": int(d.n_ep),
            "today_regime": str(d.today_regime),
            "history_years": _num(d.history_years, 1),
        })
    # Display order = the tab's (trio in its own order, then alpha) — apply
    # to the finished list exactly like build_dip_view sorts its cards.
    symbols.sort(key=lambda c: (c["ticker"] not in dps._TRIO,
                                dps._TRIO.index(c["ticker"])
                                if c["ticker"] in dps._TRIO else 0,
                                c["ticker"]))
    tb = dps.turbulence_snapshot(frames.daily_prices)
    turbulence = ({"available": True, "regime": str(tb["regime"]),
                   "percentile": _num(tb["percentile"], 1),
                   "n_days": int(tb["n"])}
                  if tb else {"available": False})
    referee = None
    art = dps._registered_artifact()
    if art is not None:
        cur = next((s for s in symbols
                    if s["ticker"] == art.get("ticker")), None)
        rec = ((art.get("referee") or {}).get(cur["band"])
               if cur else None)
        if cur and rec:
            referee = {
                "ticker": str(art["ticker"]), "band": cur["band"],
                "outcome": str((art.get("primary") or {}).get("outcome")),
                "n_days": int(rec["n_days"]),
                "n_episodes": int(rec["n_episodes"]),
                "median_12m_pct": (_num(rec["med_252"] * 100.0, 1)
                                   if rec.get("med_252") is not None
                                   else None),
                "hit_12m_pct": (_num(rec["hit_252"] * 100.0, 1)
                                if rec.get("hit_252") is not None
                                else None),
                "omega_12m": _num(rec.get("omega_252"), 1),
                # The registered artifact stores an infinite realized Omega
                # as omega_252:null + omega_252_inf:true; without this flag
                # the strongest possible record reads as unavailable.
                "omega_12m_infinite": bool(rec.get("omega_252_inf")),
            }
    return {**base, "available": True, "vintage": vintage,
            "turbulence": turbulence, "symbols": symbols,
            "skipped_short_history": skipped, "referee": referee}


_DIP_INSTRUCTION = (
    "Narrate the tab's dip verdicts — you must never issue your own buy "
    "or don't-buy advice beyond the engine's displayed band for each "
    "ticker; the bands ARE the verdicts. Say in your first sentence that "
    "these are whole-market dip statistics, independent of the account "
    "filters. For each ticker in FACTS.symbols cover: the band, the "
    "current drawdown (current_dd_pct) and how it ranks "
    "(pct_history_shallower — the share of history with a shallower "
    "drawdown), the reward-to-risk (omega vs baseline_omega with the "
    "omega_ci range; omega_infinite true means no losing 12-month "
    "outcomes in-sample), n_outcomes, and today_regime. Always carry the "
    "caveat that these are in-sample historical statistics, not "
    "forecasts. Where history_years is under 10, add that sub-10-year "
    "history is too short to trust deep-dip edges for that ticker. Name "
    "the turbulence regime and its percentile when available. If "
    "FACTS.referee is present, cite that ticker's registered walk-forward "
    "track record for its current band (hit_12m_pct, median_12m_pct, "
    "n_episodes; omega_12m_infinite true means the registered record had "
    "no losing 12-month outcomes for that band) — the out-of-sample "
    "evidence line. Name any "
    "skipped_short_history tickers as not evaluated. If vintage is "
    "well in the past, note the data is only current through it.")


def _facts_brief(frames: hs.Frames, history_start: str,
                 broker=None, dims=None, *, asof=None) -> dict:
    """Executive brief (v2-S3) — ONE whole-book read COMPOSED from the
    existing section reducers (the seams; zero new math). Whole-book v1:
    dims deliberately unthreaded. Each block carries its sub-reducer's
    own availability honestly; the brief itself is available whenever
    performance or risk is. Cost = five sub-reducers on the generating
    path only (seconds — the _facts_dip re-run precedent). ``asof`` is
    TEST-ONLY (dispatcher passes positionals), forwarded to the income
    reducer so the golden pins its today-dependent windows."""
    base = {"section": "brief", "scope": _scope_block(frames, history_start)}
    perf = _facts_performance(frames, history_start, broker, None)
    risk = _facts_risk(frames, history_start, broker, None)
    bench = _facts_benchmark(frames, history_start, broker, None)
    inc = _facts_income(frames, history_start, broker, None, asof=asof)
    tax = _facts_tax(frames, history_start, broker, None, asof=asof)

    def _sub(src: dict, keys: tuple) -> dict:
        if not src.get("available"):
            return {"available": False, "reason": src.get("reason")}
        out = {"available": True}
        for k in keys:
            if k in src:
                out[k] = src[k]
        return out

    vs_benchmark = {"available": False,
                    "reason": bench.get("reason") or "unavailable"}
    if bench.get("available"):
        rows = [r for r in (bench.get("returns") or [])
                if r.get("available")]
        vs_benchmark = {"available": bool(rows),
                        "benchmark_short": bench["benchmark"]["short"],
                        "fallback_to_spy":
                            bench["benchmark"]["fallback_to_spy"],
                        "windows": rows,
                        "consistency": bench.get("consistency")}

    available = bool(perf.get("available") or risk.get("available"))
    out = {**base, "available": available,
           "performance": _sub(perf, ("headline", "drawdown",
                                      "synthesized_twr", "contributors")),
           "risk": _sub(risk, ("daily", "daily_available",
                               "concentration", "coverage_gaps",
                               "cash_weight_pct", "stress")),
           "vs_benchmark": vs_benchmark,
           "income": _sub(inc, ("forward", "received", "book_date")),
           "tax": _sub(tax, ("unrealized", "realized_ytd", "harvest",
                             "ripening_to_long_within_60d", "ledger"))}
    if not available:
        out["reason"] = "no_data"
    return out


_BRIEF_INSTRUCTION = (
    "Answer: how is the whole book doing right now? One executive read "
    "for the owner, composed ONLY from FACTS — name the broker scope and "
    "history window from FACTS.scope in the headline or first bullet. "
    "headline: the single-sentence state of the book (performance + risk "
    "posture, with the 1-2 numbers that decide it). bullets: ONE point "
    "each, in this order where available — performance (TWR vs IRR, "
    "drawdown state), risk level (volatility vs SPY, beta), "
    "concentration (effective N, the top name's risk share), "
    "vs-benchmark (FACTS.vs_benchmark.windows spreads, naming "
    "benchmark_short), income posture (forward yield on covered MV, "
    "coverage of book), tax posture (unrealized long/short split "
    "directions, ripening count, harvest counts) "
    "— including the top return contributor/detractor, the rolling "
    "hit rate vs the benchmark, and the worst crash-replay implied "
    "drop where those blocks are available. A block with "
    "available false is skipped silently unless its absence changes the "
    "read — then one plain clause. watch: 2-4 forward-looking items "
    "grounded in FACTS numbers (short-window divergences, ripening "
    "lots, concentration, coverage gaps) — observations, never advice. "
    "Describe only — never recommend buying, selling, or timing trades.")

_BRIEF_SCHEMA = {
    "type": "object",
    "properties": {
        "headline": {"type": "string"},
        "bullets": {"type": "array", "items": {"type": "string"},
                    "minItems": 1},
        "watch": {"type": "array", "items": {"type": "string"},
                  "minItems": 1},
    },
    "required": ["headline", "bullets", "watch"],
    "additionalProperties": False,
}

# Shared by every narration/chat prompt (spec 2026-08-22): one sentence, one
# place, so the brief, the boxes and the chat present the stub identically.
_TO_DATE_CLAUSE = (
    " When a to_date block is present, lead with the to-date figure and say "
    "plainly that the final period is provisional (marked to live prices, "
    "unaudited until the statement lands).")

_BRIEF_SYSTEM = (
    "You are the narration layer of a private portfolio dashboard. You "
    "receive a FACTS JSON of pre-computed analytics (percentages, ratios, "
    "tickers). Respond ONLY with a JSON object: {\"headline\": string, "
    "\"bullets\": [string, ...], \"watch\": [string, ...]}.\n"
    "headline: the whole-book state in ONE sentence (at most 25 words) "
    "carrying the 1-2 numbers that decide it.\n"
    "bullets: 4-6 items, each ONE fact-grounded point, at most 22 words.\n"
    "watch: 2-4 forward-looking observations grounded in FACTS numbers, "
    "at most 20 words each — never advice.\n"
    "Rules: use ONLY numbers present in FACTS (rounding is fine); never "
    "invent or recompute figures; if a block is marked unavailable, skip "
    "it or say so plainly rather than extrapolating; describe and "
    "explain — never recommend buying, selling, or timing trades; no "
    "greetings, no disclaimers."
    + _TO_DATE_CLAUSE)


_NARRATIVE_FIELDS = ("verdict", "why", "changes", "watch")

_PORTFOLIO_SYSTEM = (
    "You are the narration layer of a private portfolio dashboard. You "
    "receive a FACTS JSON of pre-computed analytics (percentages, ratios, "
    "tickers). Respond ONLY with a JSON object with exactly these fields: "
    '"verdict" (string), "why" (array of strings), "changes" (string), '
    '"watch" (array of strings).\n'
    "verdict / changes: single plain sentences, at most 40 words each.\n"
    "why / watch: 2-5 short bullet strings, ONE point each, at most 20 "
    "words per item.\n"
    "Rules: use ONLY numbers present in FACTS (rounding is fine); never "
    "invent or recompute figures; if a window or block is marked "
    "unavailable, say so plainly rather than extrapolating; describe and "
    "explain — never recommend buying, selling, or timing trades; no "
    "markdown, no greetings.")

_PORTFOLIO_INSTRUCTION = (
    "Answer the owner's standing question: is this portfolio riskier than "
    "the selected benchmark, and why? FACTS.benchmark.label names the "
    "benchmark (SPY, or a 60/40 SPY/AGG blend) — NAME it in your verdict "
    "and treat every 'benchmark' figure as that series. verdict: the direct "
    "answer with the 2-3 numbers that drive it (beta, volatility vs the "
    "benchmark, max drawdown), naming the window(s) you rely on and noting "
    "when short windows disagree with the full history. why: the mechanism "
    "— concentration, top risk contributors, cash weight, correlation. "
    "changes: the latest month in context. watch: 2-3 concrete "
    "percent-grounded items worth watching.")

_PORTFOLIO_SCHEMA = {
    "type": "object",
    "properties": {
        "verdict": {"type": "string"},
        "why": {"type": "array", "items": {"type": "string"},
                "minItems": 1},
        "changes": {"type": "string"},
        "watch": {"type": "array", "items": {"type": "string"},
                  "minItems": 1},
    },
    "required": ["verdict", "why", "changes", "watch"],
    "additionalProperties": False,
}


def parse_narrative(text: str):
    """The four narrative fields from a generation, or None when malformed
    (schema enforcement makes that near-impossible; this is the seatbelt).
    v2 shape: verdict/changes strings, why/watch non-empty string arrays —
    the legacy all-strings shape parses as None (pre-v2 cache entries are
    already excluded from the fresh path by CACHE_FMT)."""
    try:
        d = json.loads(text)
    except (TypeError, json.JSONDecodeError):
        return None
    if not (isinstance(d, dict) and set(d) == set(_NARRATIVE_FIELDS)):
        return None
    if not all(isinstance(d[k], str) for k in ("verdict", "changes")):
        return None
    if not all(isinstance(d[k], list) and d[k]
               and all(isinstance(x, str) for x in d[k])
               for k in ("why", "watch")):
        return None
    return {k: d[k] for k in _NARRATIVE_FIELDS}


def portfolio_questions(bench_short: str) -> dict:
    """The AI-tab per-field questions (A-feedback 2026-08-20), server-side
    so the RESOLVED benchmark name lands in the verdict question."""
    return {"verdict": f"Is this portfolio riskier than {bench_short}?",
            "why": "What's driving that?",
            "changes": "What changed in the latest month?",
            "watch": "What should I keep an eye on?"}


_BOX_SCHEMA = {
    "type": "object",
    "properties": {
        "headline": {"type": "string"},
        "bullets": {"type": "array", "items": {"type": "string"},
                    "minItems": 1},
        "caveat": {"type": "string"},
    },
    "required": ["headline", "bullets"],
    "additionalProperties": False,
}


def parse_box(text: str):
    """{headline, bullets[], caveat|None, watch|None} from a structured
    box generation, or None when malformed — which after the CACHE_FMT
    bump means a legacy prose entry surfacing via the STALE path (the FE
    renders those as a plain paragraph). ``watch`` (S3 brief) follows the
    bullets rules; box schemas that omit it never emit it. Schema
    enforcement makes malformed fresh generations near-impossible; this
    is the seatbelt."""
    try:
        d = json.loads(text)
    except (TypeError, json.JSONDecodeError):
        return None
    if not (isinstance(d, dict)
            and set(d) <= {"headline", "bullets", "caveat", "watch"}
            and isinstance(d.get("headline"), str)
            and isinstance(d.get("bullets"), list) and d["bullets"]
            and all(isinstance(b, str) for b in d["bullets"])):
        return None
    cav = d.get("caveat")
    if cav is not None and not isinstance(cav, str):
        return None
    watch = d.get("watch")
    if watch is not None and not (isinstance(watch, list) and watch
                                  and all(isinstance(w, str)
                                          for w in watch)):
        return None
    return {"headline": d["headline"], "bullets": list(d["bullets"]),
            "caveat": cav, "watch": list(watch) if watch else None}


# Chat-only facts sections (specs 2026-08-22 + 2026-08-23): reducers with
# the standard signature that exist for the chat pack alone — never in
# SECTIONS, so the explain / regenerate routes keep 422-ing them as
# unknown sections.
_CHAT_DETAIL_REDUCERS = {"tax_detail": _facts_tax_detail,
                         "holdings_detail": _facts_holdings_detail,
                         "options": _facts_options,
                         "health": _facts_health}


SECTIONS: dict[str, dict] = {
    "factors": {"reduce": _facts_factors, "instruction": _FACTORS_INSTRUCTION,
                "question": "Where do returns come from, and is there real alpha?",
                "dims": ("window", "model")},
    "portfolio": {"reduce": _facts_portfolio,
                  "instruction": _PORTFOLIO_INSTRUCTION,
                  "question": "Is this portfolio riskier than its benchmark?",
                  "system": _PORTFOLIO_SYSTEM,
                  "schema": _PORTFOLIO_SCHEMA,
                  "dims": ("benchmark",)},
    "risk": {"reduce": _facts_risk, "instruction": _RISK_INSTRUCTION,
             "question": ("How risky is the book right now — absolutely "
                          "and vs SPY?")},
    "riskcontrib": {"reduce": _facts_riskcontrib,
                    "instruction": _RISKCONTRIB_INSTRUCTION,
                    "question": ("Who drives portfolio risk, and how "
                                 "concentrated is it?"),
                    "dims": ("estimator", "benchmark")},
    "tax": {"reduce": _facts_tax, "instruction": _TAX_INSTRUCTION,
            "question": "What is the current tax posture?"},
    "performance": {"reduce": _facts_performance,
                    "instruction": _PERFORMANCE_INSTRUCTION,
                    "question": ("How has this book performed, and where "
                                 "did the return come from?")},
    "benchmark": {"reduce": _facts_benchmark,
                  "instruction": _BENCHMARK_INSTRUCTION,
                  "question": ("How has the book done versus its benchmark, "
                               "and at what risk?"),
                  "dims": ("benchmark",)},
    "frontier": {"reduce": _facts_frontier,
                 "instruction": _FRONTIER_INSTRUCTION,
                 "question": ("Where does the book sit relative to the "
                              "efficient frontier?"),
                 "dims": ("sig",)},
    "risksim": {"reduce": _facts_risksim,
                "instruction": _RISKSIM_INSTRUCTION,
                "question": ("What would this simulated reweight do to the "
                             "book's risk?"),
                "dims": ("sig",)},
    "income": {"reduce": _facts_income, "instruction": _INCOME_INSTRUCTION,
               "question": ("What income does the book generate, and how "
                            "reliable is it?")},
    "dip": {"reduce": _facts_dip, "instruction": _DIP_INSTRUCTION,
            "question": ("What do today's dip statistics say for each "
                         "watched ticker?")},
    "brief": {"reduce": _facts_brief, "instruction": _BRIEF_INSTRUCTION,
              "question": "How is the whole book doing right now?",
              "system": _BRIEF_SYSTEM,
              "schema": _BRIEF_SCHEMA},
}


# --------------------------------------------------------------------------- #
# S2: display tables + tab meta (server-formatted, strings only — the tab
# renders these regardless of AI state).
# --------------------------------------------------------------------------- #
def _window_cols(short: str) -> list[str]:
    return ["Window", "TWR (cum)", f"{short} (cum)", "TWR (ann)",
            f"{short} (ann)", "Vol", f"{short} vol", "Beta", "Corr",
            "Max DD", f"{short} DD", "Sharpe", "Months"]


def _fmt_pct(v, signed=True) -> str:
    if v is None or v != v:
        return "—"
    return f"{v:+.1f}%" if signed else f"{v:.1f}%"


def _fmt_num(v) -> str:
    return "—" if v is None or v != v else f"{v:.2f}"


def portfolio_display(facts: dict) -> dict:
    short = hs.BENCH_SHORT.get((facts.get("benchmark") or {}).get("id"), "SPY")
    cols = _window_cols(short)
    rows = []
    for w in facts.get("windows", []):
        if not w.get("available"):
            row = {c: "—" for c in cols}
            row["Window"] = w["window"]
            row["Months"] = str(w.get("n_months", 0))
            rows.append(row)
            continue
        p, s = w["portfolio"], w["benchmark"]
        rows.append({
            "Window": w["window"],
            "TWR (cum)": _fmt_pct(p["twr_cum_pct"]),
            f"{short} (cum)": _fmt_pct(s["twr_cum_pct"]),
            "TWR (ann)": _fmt_pct(p["twr_ann_pct"]),
            f"{short} (ann)": _fmt_pct(s["twr_ann_pct"]),
            "Vol": _fmt_pct(p["vol_ann_pct"], signed=False),
            f"{short} vol": _fmt_pct(s["vol_ann_pct"], signed=False),
            "Beta": _fmt_num(w["beta"]),
            "Corr": _fmt_num(w["correlation"]),
            "Max DD": _fmt_pct(p["max_dd_pct"]),
            f"{short} DD": _fmt_pct(s["max_dd_pct"]),
            "Sharpe": _fmt_num(p["sharpe"]),
            "Months": str(w["n_months"]),
        })
    tiles = []
    c = facts.get("concentration") or {}
    if c:
        tiles += [{"label": "Effective N", "value": f"{c['effective_n']:.1f}"},
                  {"label": "Top-10 weight",
                   "value": _fmt_pct(c["top10_weight_pct"], signed=False)},
                  {"label": "Max position",
                   "value": _fmt_pct(c["max_weight_pct"], signed=False)}]
        top = c.get("top_risk_contributors") or []
        if top:
            tiles.append({"label": "Top risk",
                          "value": (f"{top[0]['ticker']} "
                                    f"{top[0]['risk_share_pct']:.0f}% of risk")})
    if facts.get("cash_weight_pct") is not None:
        tiles.append({"label": "Cash weight",
                      "value": _fmt_pct(facts["cash_weight_pct"],
                                        signed=False)})
    inc = facts.get("income") or {}
    if inc.get("available"):
        tiles.append({"label": "Fwd yield (covered)",
                      "value": str(inc["forward_yield_on_covered_mv"])})
    tx = facts.get("tax_posture") or {}
    if tx.get("available"):
        tiles.append({"label": "Unrealized LT share",
                      "value": (_fmt_pct(tx["long_share_pct"], signed=False)
                                + f" ({tx['long_net']})")})
    return {"window_table": {"columns": cols, "rows": rows},
            "tiles": tiles}


def portfolio_meta(frames: hs.Frames) -> dict:
    snap_all = hs._current_snap(frames)
    return {"accounts": hs._account_options(snap_all)[0],
            "classes": hs._class_options(snap_all)[0],
            "brokers": hs._broker_options(snap_all)[0],
            "available_dates": list(frames.available_dates),
            "synthetic": "synth" in str(frames.data_dir).lower(),
            "filter": {"account": "all", "asset_class": "all",
                       "broker": "all"}}


def build_facts(section: str, frames: hs.Frames, *,
                history_start: str = "all", broker=None,
                dims: dict | None = None) -> dict:
    """Reduce the section's tab payload to its facts dict and scrub it.
    ``history_start`` and ``broker`` are the REQUEST values — they must be
    threaded in because Frames carries neither (S1 review catch #2).
    ``dims`` are the section's tracked seg values (S3). KeyError on unknown
    section, AIDimError on a dim value the payload does not offer
    (server -> 422 for both)."""
    facts = SECTIONS[section]["reduce"](frames, history_start, broker, dims)
    return scrub_gate(facts)


# --------------------------------------------------------------------------- #
# Claude client. generate() is narration-only best-effort: ANY failure
# (transport, API status, refusal, empty content) raises AIGenerationError,
# which the route degrades to kind:'error' (+ stale text if cached). The
# broad except is deliberate and non-silent — every failure surfaces typed.
# --------------------------------------------------------------------------- #
_SYSTEM_PROMPT = (
    "You are the narration layer of a private portfolio dashboard. You "
    "receive a FACTS JSON of pre-computed analytics (percentages, ratios, "
    "tickers). Respond ONLY with a JSON object: {\"headline\": string, "
    "\"bullets\": [string, ...], \"caveat\": string (optional)}.\n"
    "headline: the direct answer to the section's question in ONE sentence "
    "(at most 25 words) carrying the 1-3 numbers that decide it.\n"
    "bullets: 3-6 items, each ONE fact-grounded point, at most 20 words.\n"
    "caveat: include ONLY when a window or block is unavailable or shorter "
    "than requested — one sentence naming it (at most 30 words); omit the "
    "key otherwise.\n"
    "Rules: use ONLY numbers present in FACTS (rounding is fine); never "
    "invent or recompute figures; if a field or window is marked "
    "unavailable, say so plainly rather than extrapolating; describe and "
    "explain — never recommend buying, selling, or timing trades; no "
    "greetings, no disclaimers."
    + _TO_DATE_CLAUSE)


_CLIENT_MEMO: tuple | None = None      # (api_key, client) — see resolve_client
_CLIENT_MEMO_LOCK = threading.Lock()


def resolve_client():
    """Real anthropic client, or None => AI off (no key / no package).
    The built client is memoized module-wide keyed by the API key:
    construction loads the Windows cert store (~0.7-1 s on the real box),
    which used to run per /api/ai/* request (#371 review ledger). A
    missing key is never cached (a key set later is picked up) and a
    changed key rebuilds. The anthropic client is thread-safe, so one
    instance serves concurrent routes. Norton TLS interception is already
    handled: importing _config injected truststore into SSL at module
    import."""
    global _CLIENT_MEMO
    key = _config.get_anthropic_key()
    if not key or anthropic is None:
        return None
    with _CLIENT_MEMO_LOCK:
        if _CLIENT_MEMO is None or _CLIENT_MEMO[0] != key:
            _CLIENT_MEMO = (key, anthropic.Anthropic(api_key=key))
        return _CLIENT_MEMO[1]


def _extract_text(msg) -> str:
    """First text block of a response, stripped. Raises AIGenerationError
    on refusal or empty content — shared by generate() and generate_chat()
    so both surfaces fail identically."""
    if getattr(msg, "stop_reason", None) == "refusal":
        raise AIGenerationError("model declined the request (refusal)")
    text = next((b.text for b in getattr(msg, "content", [])
                 if getattr(b, "type", "") == "text"), "").strip()
    if not text:
        raise AIGenerationError("empty response")
    return text


def generate(section: str, facts: dict, *, client) -> str:
    sec = SECTIONS[section]
    prompt = (sec["instruction"] + "\n\nFACTS:\n"
              + json.dumps(facts, sort_keys=True, ensure_ascii=False))
    schema = sec.get("schema") or _BOX_SCHEMA
    try:
        msg = client.beta.messages.create(
            model=MODEL,
            max_tokens=16000,  # Fable 5: always-on thinking shares the output budget
            betas=[_FALLBACK_BETA],
            fallbacks="default",
            system=sec.get("system", _SYSTEM_PROMPT),
            messages=[{"role": "user", "content": prompt}],
            output_config={"format": {"type": "json_schema",
                                      "schema": schema}},
        )
    except Exception as e:               # noqa: BLE001 — typed re-raise, see above
        raise AIGenerationError(f"API call failed: {e}") from e
    return _extract_text(msg)


_CHAT_SYSTEM = (
    "You are the conversational analyst of a private portfolio dashboard. "
    "You receive a FACTS JSON — pre-computed analytics for the whole book "
    "(percentages, ratios, tickers; one block per dashboard section) — "
    "followed by the owner's questions.\n"
    "Rules: answer ONLY from FACTS and fetch_detail results; never "
    "invent, estimate, or recompute "
    "figures (rounding is fine). If the answer is not in FACTS, say so "
    "plainly and name what data or dashboard section would be needed. "
    "FACTS carries no dollar amounts by design — dollar values never "
    "leave the owner's machine; when asked about dollars, say exactly "
    "that and answer in percentage or ratio terms instead. The efficient-"
    "frontier and what-if simulation runs are not visible to you — point "
    "the owner at those tabs for run-specific questions. Fields marked "
    "unavailable are unavailable — say so rather than extrapolating. "
    "Describe and explain — never recommend buying, selling, or timing "
    "trades. Plain text, no markdown; keep answers under 200 words "
    "unless the question genuinely demands more."
    " FACTS.tax_detail carries per-ticker wash-window dates derived from "
    "the transaction ledger (a trailing window of window_days ending at "
    "as_of, fresh through tx_frontier, whole book): wash_if_sold_before "
    "is the last day a loss sale is still washed by the latest purchase "
    "— only when shares other than the ones sold were acquired in the "
    "window; a single lot round-tripped within the window is not washed "
    "by its own purchase — wash_if_rebought_before the last day a "
    "repurchase would wash a loss on the latest sale; whether a past "
    "sale realized a loss is not stated — cite broker_flagged_wash_sells "
    "instead. Its lots block covers actionable tickers only (see "
    "coverage_note). FACTS.holdings_detail lists nameable positions with "
    "weights of the scoped book (unnameable identifiers are omitted, "
    "their weight in omitted_weight_pct; option contracts appear under "
    "their underlying); the direct-index sleeve and the treasury ladder "
    "appear as one line each."
    " FACTS.options summarizes the live option book: per-contract rows "
    "(underlying, put/call, long/short, expiry, DTE, strike vs spot %) "
    "and exposure as % of scoped NAV; when empty, empty_message says "
    "why. FACTS.health reports whether extracted holdings reconcile "
    "with broker-reported totals (headline_level, per-account verdicts, "
    "diff_pct only) — cite it when asked if today's numbers can be "
    "trusted."
    " You have one tool, fetch_detail(topic, ticker): call it when the "
    "question needs a per-name or full-table detail FACTS truncates — "
    "riskcontrib (every name's risk share), income (every payer's "
    "projected share/yield), lots (every open TAXABLE-account lot for "
    "one ticker; IRA lots excluded), transactions (one ticker's "
    "wash-window activity), performance (full monthly and yearly TWR "
    "plus every account's cumulative return), dip (the registered "
    "walk-forward referee table), factor (per-window regression betas), "
    "options_contracts (each live option contract), health (full "
    "reconciliation rows). "
    "Answer directly from FACTS when it already suffices; fetch at "
    "most what the question needs. Tool results follow the same rules "
    "as FACTS: percentages, dates, counts — never dollars, never "
    "share quantities."
    + _TO_DATE_CLAUSE)


# Full-gate S2 (spec §5): the one chat tool. Stable bytes — part of the
# prompt-cache prefix (tools render before system); the enum grows once
# in S3 (one cache re-prime per deploy, stable after).
_FETCH_DETAIL_TOOL = {
    "name": "fetch_detail",
    "description": (
        "Fetch per-name or full-table dashboard detail that FACTS "
        "truncates. Use when the question needs a name or row not in "
        "FACTS. Results are percentages/ratios/dates only — no dollar "
        "amounts, no share quantities."),
    "strict": True,
    "input_schema": {
        "type": "object",
        "properties": {
            "topic": {"type": "string",
                      "enum": ["riskcontrib", "income", "lots",
                               "transactions", "performance", "dip",
                               "factor", "options_contracts",
                               "health"]},
            "ticker": {"type": "string"},
        },
        "required": ["topic"],
        "additionalProperties": False,
    },
}
_CHAT_TOOL_ROUNDS = 3     # productive rounds per turn; then one nudge, then fail


def _tool_result(tool_use_id: str, payload: dict) -> dict:
    """One tool_result block. allow_nan=False is the leak guard (S1
    final-review carry-over): a NaN that slipped a reducer raises here
    and fails the turn closed instead of shipping literal NaN tokens."""
    is_err = isinstance(payload, dict) and "error" in payload
    block = {"type": "tool_result", "tool_use_id": tool_use_id,
             "content": json.dumps(payload, sort_keys=True,
                                   ensure_ascii=False, allow_nan=False)}
    if is_err:
        block["is_error"] = True
    return block


def generate_chat(messages: list[dict], pack: dict, *, client,
                  detail_fn=None) -> str:
    """One chat turn: the scrubbed whole-book pack rides as the FIRST
    user message's leading block with cache_control (multi-turn re-sends
    hit the prompt cache; the pack is byte-stable per memo entry), the
    conversation follows verbatim. With ``detail_fn`` (full-gate S2) the
    request carries the fetch_detail tool and this loops on
    stop_reason=="tool_use": assistant content replayed VERBATIM
    (thinking blocks unchanged — same model), every block executed, all
    results in ONE user message; a result dict carrying "error" ships as
    is_error (turn continues); AIScrubError from detail_fn fails the
    turn closed; _CHAT_TOOL_ROUNDS productive rounds, then one
    budget-exhausted nudge round, then AIGenerationError. Tool
    exchanges are turn-local — the stored answer is final text only.
    detail_fn=None keeps the legacy single-shot request (no tools key).
    Raises AIGenerationError exactly like generate()."""
    pack_block = {"type": "text",
                  "text": "FACTS:\n" + json.dumps(pack, sort_keys=True,
                                                  ensure_ascii=False),
                  "cache_control": {"type": "ephemeral"}}
    convo = []
    for i, m in enumerate(messages):
        if i == 0:
            convo.append({"role": "user",
                          "content": [pack_block,
                                      {"type": "text",
                                       "text": m["content"]}]})
        else:
            convo.append({"role": m["role"], "content": m["content"]})
    kwargs = dict(model=MODEL, max_tokens=16000, betas=[_FALLBACK_BETA],
                  fallbacks="default", system=_CHAT_SYSTEM,
                  output_config={"effort": _CHAT_EFFORT})
    if detail_fn is not None:
        kwargs["tools"] = [_FETCH_DETAIL_TOOL]
    rounds = 0
    while True:
        try:
            msg = client.beta.messages.create(messages=convo, **kwargs)
        except Exception as e:           # noqa: BLE001 — typed re-raise
            raise AIGenerationError(f"API call failed: {e}") from e
        if getattr(msg, "stop_reason", None) != "tool_use":
            return _extract_text(msg)
        blocks = [b for b in msg.content
                  if getattr(b, "type", "") == "tool_use"]
        if detail_fn is None or not blocks:
            raise AIGenerationError("unexpected tool_use stop")
        if rounds > _CHAT_TOOL_ROUNDS:          # nudge already spent
            raise AIGenerationError("tool budget exceeded")
        convo.append({"role": "assistant", "content": msg.content})
        exhausted = rounds >= _CHAT_TOOL_ROUNDS
        results = []
        try:
            for b in blocks:
                if exhausted:
                    results.append(_tool_result(b.id, {"error": (
                        "detail budget exhausted — answer now from FACTS "
                        "and the detail already fetched")}))
                    continue
                inp = b.input if isinstance(b.input, dict) else {}
                try:
                    out = detail_fn(str(inp.get("topic") or ""),
                                    inp.get("ticker"))
                except AIScrubError as e:
                    raise AIGenerationError(
                        f"detail scrub failed: {e}") from e
                results.append(_tool_result(b.id, out))
        except ValueError as e:          # NaN/Inf in a tool result
            raise AIGenerationError(
                f"tool result not JSON-safe: {e}") from e
        convo.append({"role": "user", "content": results})
        rounds += 1
