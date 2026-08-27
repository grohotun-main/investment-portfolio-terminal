"""FastAPI app for the MERIDIAN Portfolio Terminal — Holdings slice.

Loopback-only (see ``terminal/__main__.py`` for the bind). Serves the vanilla
static front-end **and** the JSON API from the same origin, so there is no CORS
surface (spec §7.2). The engine is reused through ``terminal.holdings_service``;
this module adds only the HTTP shell + typed input validation.

Security posture (spec §7):
  * ``docs_url`` / ``redoc_url`` disabled — no interactive schema surface.
  * No CORS middleware — single origin by design.
  * A loopback guard (``_loopback_guard``) rejects non-loopback ``Host`` headers
    (DNS-rebinding) and cross-origin ``Origin`` headers (CSRF) before any route
    runs, so the loopback bind can't be turned against a browser.
  * Every query param is typed + range-bounded, and ``as_of`` / ``account`` /
    ``asset_class`` are validated against the *known* option sets for the loaded
    data; anything unknown is rejected with ``422`` rather than passed onward.
The host bind lives in ``__main__`` only — importing this module never binds.
"""
from __future__ import annotations

import json
import math
import os
import uuid
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field

from terminal import holdings_service as hs
from terminal import performance_service as ps
from terminal import benchmark_service as bs
from terminal import health_service as hes
from terminal import income_service as ins
from terminal import factor_service as fs
from terminal import dip_service as ds
from terminal import risk_service as rs
from terminal import riskcontrib_service as rcs
from terminal import risksim_service as rss
from terminal import options_service as ops
from terminal import actions_service as acts
from terminal import chrome_service as cs
from terminal import tax_service as txs
from terminal import ai_service as ai
from terminal.tokens import root_css

STATIC = Path(__file__).resolve().parent / "static"

app = FastAPI(title="MERIDIAN Portfolio Terminal", docs_url=None, redoc_url=None)


@app.middleware("http")
async def _static_no_cache(request: Request, call_next):
    """Static assets must revalidate on every load (``no-cache`` ≠ ``no-store``:
    the StaticFiles ETag still yields 304s). Without any Cache-Control header,
    browsers heuristic-cache ``/app.js`` and serve a stale front-end after a
    merge."""
    resp = await call_next(request)
    path = request.url.path
    if path == "/" or path.endswith((".js", ".css", ".html")):
        resp.headers["Cache-Control"] = "no-cache"
    return resp


# --- Loopback trust boundary (spec §7.1) ------------------------------------
# The server binds 127.0.0.1 only, but loopback is NOT a boundary against code
# running in the user's own browser: a malicious page can POST to 127.0.0.1
# (CSRF — the browser hides the response, but a state-changing action still
# fires) or use DNS rebinding to READ the API cross-origin (full holdings /
# dollar figures). Two header checks close that gap without any auth surface:
#   * Host must be loopback  -> defeats DNS rebinding, whose request carries the
#     attacker's own hostname (Host: attacker.com) even after it resolves to
#     127.0.0.1.
#   * Origin, when present, must be loopback -> rejects cross-origin state-
#     changing POSTs before the route (and thus the parser job) runs.
# "testserver" is the in-process Starlette TestClient authority; no DNS name
# resolves to it, so a browser can never send it — allowing it grants no bypass.
_ALLOWED_HOST_NAMES = {"127.0.0.1", "localhost", "::1", "testserver"}


def _header_hostname(value: str) -> str:
    """Bare lowercase hostname from a Host/Origin header (scheme, port and path
    stripped; IPv6 brackets unwrapped)."""
    v = value.strip()
    if "://" in v:
        v = v.split("://", 1)[1]
    v = v.split("/", 1)[0]
    if v.startswith("["):                       # IPv6 literal e.g. [::1]:8502
        return v[1:v.index("]")].lower() if "]" in v else v.lower()
    return (v.rsplit(":", 1)[0] if ":" in v else v).lower()


@app.middleware("http")
async def _loopback_guard(request: Request, call_next):
    """Reject non-loopback Host (DNS-rebinding) and cross-origin requests (CSRF)
    before any route runs. See the trust-boundary note above."""
    if _header_hostname(request.headers.get("host", "")) not in _ALLOWED_HOST_NAMES:
        return Response("forbidden: non-loopback Host header", status_code=403)
    origin = request.headers.get("origin")
    if origin is not None and _header_hostname(origin) not in _ALLOWED_HOST_NAMES:
        return Response("forbidden: cross-origin request", status_code=403)
    return await call_next(request)


# --- NaN/Inf-safe validation-error responses ---------------------------------
# A pydantic constraint (Field(gt=..., allow_inf_nan=False), a plain `str`
# type, etc.) correctly REJECTS a client's NaN/Infinity request value with a
# RequestValidationError — but FastAPI's default handler for that exception
# echoes the raw offending value back inside the 422 body's error detail, and
# Starlette's JSONResponse.render hardcodes allow_nan=False, so *rendering the
# rejection* raises inside the exception handler itself and the whole request
# crashes instead of returning 422 (discovered on POST /api/tax/estimate's
# TaxSimLeg.qty and TaxOverrides.filing_status; the identical exposure predates
# this route on any other Field(gt=/lt=/ge=/le=) numeric constraint, e.g.
# OptimizeBody/TraceBody's cap_pct). This app-wide handler keeps the exact
# default {"detail": [...]} shape and 422 status — it only walks the error
# list and replaces any non-finite float (however deeply nested) with its
# repr so the response can always serialize.
def _finite_scrub(obj):
    if isinstance(obj, float) and not math.isfinite(obj):
        return repr(obj)                        # "nan" / "inf" / "-inf"
    if isinstance(obj, dict):
        return {k: _finite_scrub(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_finite_scrub(v) for v in obj]
    return obj


@app.exception_handler(RequestValidationError)
async def _validation_error_handler(request: Request,
                                     exc: RequestValidationError) -> Response:
    detail = _finite_scrub(jsonable_encoder(exc.errors()))
    return Response(json.dumps({"detail": detail}, allow_nan=False),
                    status_code=422, media_type="application/json")


def _data_dir() -> str:
    """The data dir the engine reads — the same ``APP_DATA_DIR`` seam Streamlit
    and the test suite point at (fixture vs real ``data/``)."""
    return os.environ.get("APP_DATA_DIR", "data")


def _validate_filter_ids(values: list[str], known_ids: set[str], kind: str) -> None:
    """422 if any requested filter id is neither the 'all' sentinel nor a known
    option id. Shared by every route that exposes account/asset_class."""
    for v in values:
        if v != "all" and v not in known_ids:
            raise HTTPException(status_code=422, detail=f"unknown {kind}")


def _validate_bucket_pcts(known: set[str], *, floors: dict, caps: dict,
                          budgets: dict | None = None) -> None:
    """422 on unknown floor/cap/budget bucket keys or out-of-range percents.
    Shared by the optimize/trace/frontier routes (frontier passes no budgets).
    Floors and caps are inclusive [0, 100]; budgets are (0, 100] — an explicit
    0 means "omit, not send". Detail strings are pinned by route tests."""
    for kind, entries, lo_excl in (("floor", floors, False),
                                   ("cap", caps, False),
                                   ("budget", budgets or {}, True)):
        for k, v in entries.items():
            if k not in known:
                raise HTTPException(status_code=422,
                                    detail=f"unknown {kind} bucket: {k}")
            v = float(v)
            ok = (0.0 < v <= 100.0) if lo_excl else (0.0 <= v <= 100.0)
            if not ok:
                raise HTTPException(status_code=422,
                                    detail=f"{kind} out of range")


def _finalize_broker_meta(view: dict, broker_opts: list[dict], broker: list[str]) -> dict:
    """Overwrite meta.brokers with the FULL pre-narrowing option list (so the
    picker never loses a choice once narrowed) and meta.filter.broker with the
    echoed selection (each build_*_view's own 'all' default is wrong once
    broker != 'all'). Reuses broker_opts computed BEFORE apply_global_filters
    narrowed frames. No-op under the default/canonical selection.

    Only the tab-landing views carry meta.brokers + meta.filter — the risksim
    POST compute results, /api/dip/lookup, and /api/options/recommend do NOT
    surface the broker picker payload, so this helper is intentionally not
    called on those routes (its home is the 11 GET tab views)."""
    view["meta"]["brokers"] = broker_opts
    view["meta"]["filter"]["broker"] = hs._filter_echo(hs._normalize_filter_ids(broker))
    return view


def _finalize_history_meta(view: dict, hist_opts: list[dict],
                           history_start: str) -> dict:
    """Add meta.history_starts (FULL option list, so the picker never locks out)
    + echo the selection. Companion to _finalize_broker_meta; called on the same
    11 GET tab views (the POST compute / lookup / recommend routes don't surface
    the picker)."""
    view["meta"]["history_starts"] = hist_opts
    view["meta"]["filter"]["history_start"] = history_start or "all"
    return view


@app.get("/api/holdings")
def holdings(
    as_of: str | None = None,
    account: list[str] = Query(["all"]),
    asset_class: list[str] = Query(["all"]),
    broker: list[str] = Query(["all"]),
    history_start: str = Query("all"),
    top_n: int = Query(15, ge=5, le=30),
    search: str = Query("", max_length=64),
):
    """Return the Holdings view for the requested as-of / filters.

    Validates ``as_of`` against the data's available dates and ``account`` /
    ``asset_class`` against the known filter-option ids before building the
    final view — unknown values raise ``422`` and never reach the engine.
    ``broker`` is the GLOBAL filter (spec: filter-parity S2a) — validated and
    applied (narrowing ``frames``) before the account/class options are read,
    so an out-of-broker account id also 422s. ``history_start`` (S3) is the
    GLOBAL history-start cutoff, validated + applied alongside ``broker``.
    """
    try:
        frames = hs.load_frames(_data_dir())
    except FileNotFoundError:
        # A wrong/missing APP_DATA_DIR (or a data dir lacking positions.csv) is
        # a misconfiguration, not empty data — surface a clean 503 rather than a
        # 500 stack trace, and do NOT mask it into an empty view.
        raise HTTPException(
            status_code=503,
            detail="data directory or positions.csv not found",
        )

    broker_opts = hs._broker_options(hs._current_snap(frames))[0]
    broker_ids = {o["id"] for o in broker_opts}
    _validate_filter_ids(broker, broker_ids, "broker")
    hist_opts = hs._history_start_options(frames)
    _validate_filter_ids([history_start], {o["id"] for o in hist_opts}, "history_start")
    frames = hs.apply_global_filters(frames, broker, history_start)

    if as_of is not None and as_of not in frames.available_dates:
        raise HTTPException(status_code=422, detail="unknown as_of")

    # Validate the requested filters against the option-id sets the view's meta
    # would carry — computed without a throwaway full-view build.
    account_ids, class_ids = hs.filter_option_ids(frames, as_of=as_of)

    _validate_filter_ids(account, account_ids, "account")
    _validate_filter_ids(asset_class, class_ids, "asset_class")

    view = hs.build_holdings_view(
        frames, as_of=as_of, account=account, asset_class=asset_class,
        top_n=top_n, search=search,
    )
    _finalize_broker_meta(view, broker_opts, broker)
    _finalize_history_meta(view, hist_opts, history_start)
    # Strict serialization: a NaN/Infinity token is invalid JSON that
    # ``json.dumps`` emits silently (and a browser ``JSON.parse`` rejects).
    # ``allow_nan=False`` makes any future NaN leak fail loudly server-side
    # instead of shipping a malformed body.
    return Response(json.dumps(view, allow_nan=False),
                    media_type="application/json")


@app.get("/api/performance")
def performance(account: list[str] = Query(["all"]), asset_class: list[str] = Query(["all"]),
                broker: list[str] = Query(["all"]), history_start: str = Query("all")):
    """Return the Performance view for the requested filters. Validates
    account / asset_class against the known option-id sets (422 otherwise).
    ``broker`` is the GLOBAL filter — validated + applied before account/class.
    ``history_start`` (S3) narrows the same way, applied alongside ``broker``."""
    try:
        frames = hs.load_frames(_data_dir())
    except FileNotFoundError:
        raise HTTPException(status_code=503,
                            detail="data directory or positions.csv not found")
    broker_opts = hs._broker_options(hs._current_snap(frames))[0]
    broker_ids = {o["id"] for o in broker_opts}
    _validate_filter_ids(broker, broker_ids, "broker")
    hist_opts = hs._history_start_options(frames)
    _validate_filter_ids([history_start], {o["id"] for o in hist_opts}, "history_start")
    frames = hs.apply_global_filters(frames, broker, history_start)
    account_ids, class_ids = hs.filter_option_ids(frames)
    _validate_filter_ids(account, account_ids, "account")
    _validate_filter_ids(asset_class, class_ids, "asset_class")
    view = ps.build_performance_view(frames, account=account, asset_class=asset_class)
    _finalize_broker_meta(view, broker_opts, broker)
    _finalize_history_meta(view, hist_opts, history_start)
    return Response(json.dumps(view, allow_nan=False),
                    media_type="application/json")


@app.get("/api/benchmark")
def benchmark(account: list[str] = Query(["all"]), asset_class: list[str] = Query(["all"]),
              broker: list[str] = Query(["all"]), history_start: str = Query("all"),
              benchmark: str = Query("auto")):
    """Return the Performance-vs-Benchmark view for the requested filters.
    Validates account / asset_class against the known option-id sets (422
    otherwise); missing data dir -> 503. ``broker`` is the GLOBAL filter —
    validated + applied before account/class. ``history_start`` (S3) narrows
    the same way, applied alongside ``broker``. ``benchmark`` (auto/spy/60_40)
    selects the comparison series; ``auto`` defers to resolve_benchmark's
    broker-scope default. Unknown values -> 422, including the ``all``
    sentinel (accepted elsewhere as a wildcard, but not a valid benchmark id)
    — resolve_benchmark itself would silently treat any unrecognized value as
    auto, so this route is the only rejection point."""
    try:
        frames = hs.load_frames(_data_dir())
    except FileNotFoundError:
        raise HTTPException(status_code=503,
                            detail="data directory or positions.csv not found")
    broker_opts = hs._broker_options(hs._current_snap(frames))[0]
    broker_ids = {o["id"] for o in broker_opts}
    _validate_filter_ids(broker, broker_ids, "broker")
    hist_opts = hs._history_start_options(frames)
    _validate_filter_ids([history_start], {o["id"] for o in hist_opts}, "history_start")
    if benchmark not in {"auto", "spy", "60_40"}:
        # NOT _validate_filter_ids: that helper exempts the "all" sentinel
        # (correct for account/asset_class/broker, which have an "all"
        # wildcard) — benchmark has no such wildcard, so "all" must 422 too.
        raise HTTPException(status_code=422, detail="unknown benchmark")
    frames = hs.apply_global_filters(frames, broker, history_start)
    account_ids, class_ids = hs.filter_option_ids(frames)
    _validate_filter_ids(account, account_ids, "account")
    _validate_filter_ids(asset_class, class_ids, "asset_class")
    view = bs.build_benchmark_view(frames, account=account, asset_class=asset_class,
                                   benchmark=benchmark)
    _finalize_broker_meta(view, broker_opts, broker)
    _finalize_history_meta(view, hist_opts, history_start)
    return Response(json.dumps(view, allow_nan=False),
                    media_type="application/json")


@app.get("/api/risk")
def risk(account: list[str] = Query(["all"]),
         asset_class: list[str] = Query(["all"]),
         broker: list[str] = Query(["all"]),
         history_start: str = Query("all")) -> Response:
    """Return the Risk Overview view for the requested filters (risk-adjusted
    ratios, drawdown, concentration, daily vol/VaR/CVaR, beta). Validates
    account / asset_class against the known option-id sets (422 otherwise); the
    Risk tab always uses the latest snapshot, so there is no as_of param. Missing
    data dir -> 503. ``broker`` is the GLOBAL filter — validated + applied
    before account/class. ``history_start`` (S3) narrows the same way, applied
    alongside ``broker``."""
    try:
        frames = hs.load_frames(_data_dir())
    except FileNotFoundError:
        raise HTTPException(status_code=503,
                            detail="data directory or positions.csv not found")
    broker_opts = hs._broker_options(hs._current_snap(frames))[0]
    broker_ids = {o["id"] for o in broker_opts}
    _validate_filter_ids(broker, broker_ids, "broker")
    hist_opts = hs._history_start_options(frames)
    _validate_filter_ids([history_start], {o["id"] for o in hist_opts}, "history_start")
    frames = hs.apply_global_filters(frames, broker, history_start)
    account_ids, class_ids = hs.filter_option_ids(frames)
    _validate_filter_ids(account, account_ids, "account")
    _validate_filter_ids(asset_class, class_ids, "asset_class")
    view = rs.build_risk_view(frames, account=account, asset_class=asset_class)
    _finalize_broker_meta(view, broker_opts, broker)
    _finalize_history_meta(view, hist_opts, history_start)
    return Response(json.dumps(view, allow_nan=False),
                    media_type="application/json")


@app.get("/api/riskcontrib")
def riskcontrib(account: list[str] = Query(["all"]),
                asset_class: list[str] = Query(["all"]),
                broker: list[str] = Query(["all"]),
                history_start: str = Query("all")) -> Response:
    """Return the Risk Contribution view (Slice 1: per-position vol/downside/ES
    decomposition). Validates account / asset_class against the known option-id
    sets (422 otherwise); like /api/risk, no as_of (latest snapshot). 503 on
    missing data dir. ``broker`` is the GLOBAL filter — validated + applied
    before account/class. ``history_start`` (S3) narrows the same way, applied
    alongside ``broker``."""
    try:
        frames = hs.load_frames(_data_dir())
    except FileNotFoundError:
        raise HTTPException(status_code=503,
                            detail="data directory or positions.csv not found")
    broker_opts = hs._broker_options(hs._current_snap(frames))[0]
    broker_ids = {o["id"] for o in broker_opts}
    _validate_filter_ids(broker, broker_ids, "broker")
    hist_opts = hs._history_start_options(frames)
    _validate_filter_ids([history_start], {o["id"] for o in hist_opts}, "history_start")
    frames = hs.apply_global_filters(frames, broker, history_start)
    account_ids, class_ids = hs.filter_option_ids(frames)
    _validate_filter_ids(account, account_ids, "account")
    _validate_filter_ids(asset_class, class_ids, "asset_class")
    view = rcs.build_riskcontrib_view(frames, account=account, asset_class=asset_class)
    _finalize_broker_meta(view, broker_opts, broker)
    _finalize_history_meta(view, hist_opts, history_start)
    return Response(json.dumps(view, allow_nan=False), media_type="application/json")


@app.get("/api/health")
def health(broker: list[str] = Query(["all"]),
           history_start: str = Query("all")) -> Response:
    """Return the Data Health (ingest reconciliation) view. Whole-book — no
    Account / Asset-class filter params (the report ignores them). ``broker``
    is the GLOBAL filter (spec: filter-parity S2a) and DOES apply here — it
    narrows the book before reconciliation. ``history_start`` (S3) narrows the
    same way, applied alongside ``broker``. Missing data dir -> 503."""
    try:
        frames = hs.load_frames(_data_dir())
    except FileNotFoundError:
        raise HTTPException(status_code=503,
                            detail="data directory or positions.csv not found")
    broker_opts = hs._broker_options(hs._current_snap(frames))[0]
    broker_ids = {o["id"] for o in broker_opts}
    _validate_filter_ids(broker, broker_ids, "broker")
    hist_opts = hs._history_start_options(frames)
    _validate_filter_ids([history_start], {o["id"] for o in hist_opts}, "history_start")
    frames = hs.apply_global_filters(frames, broker, history_start)
    view = hes.build_health_view(frames)
    _finalize_broker_meta(view, broker_opts, broker)
    _finalize_history_meta(view, hist_opts, history_start)
    return Response(json.dumps(view, allow_nan=False),
                    media_type="application/json")


@app.get("/api/income")
def income(broker: list[str] = Query(["all"]),
           history_start: str = Query("all")) -> Response:
    """Return the Income view (actual income + forward projection). Whole-book —
    no Account / Asset-class filter params (income ignores them). ``broker`` is
    the GLOBAL filter and DOES apply — it narrows the book before the income
    rollup. ``history_start`` (S3) narrows the same way, applied alongside
    ``broker``. Missing data dir -> 503."""
    try:
        frames = hs.load_frames(_data_dir())
    except FileNotFoundError:
        raise HTTPException(status_code=503,
                            detail="data directory or positions.csv not found")
    broker_opts = hs._broker_options(hs._current_snap(frames))[0]
    broker_ids = {o["id"] for o in broker_opts}
    _validate_filter_ids(broker, broker_ids, "broker")
    hist_opts = hs._history_start_options(frames)
    _validate_filter_ids([history_start], {o["id"] for o in hist_opts}, "history_start")
    frames = hs.apply_global_filters(frames, broker, history_start)
    view = ins.build_income_view(frames)
    _finalize_broker_meta(view, broker_opts, broker)
    _finalize_history_meta(view, hist_opts, history_start)
    return Response(json.dumps(view, allow_nan=False),
                    media_type="application/json")


@app.get("/api/factor")
def factor(broker: list[str] = Query(["all"]),
           history_start: str = Query("all")) -> Response:
    """Return the Factor Analysis view (alpha-by-model strip, betas, attribution
    waterfall + over-time, rolling betas, per-holding + cross-check). Whole-book —
    no Account / Asset-class filter params (regressions always run on the full
    filtered book). ``broker`` is the GLOBAL filter and DOES apply — it narrows
    the book before the regressions. ``history_start`` (S3) narrows the same
    way, applied alongside ``broker``. Missing data dir -> 503."""
    try:
        frames = hs.load_frames(_data_dir())
    except FileNotFoundError:
        raise HTTPException(status_code=503,
                            detail="data directory or positions.csv not found")
    broker_opts = hs._broker_options(hs._current_snap(frames))[0]
    broker_ids = {o["id"] for o in broker_opts}
    _validate_filter_ids(broker, broker_ids, "broker")
    hist_opts = hs._history_start_options(frames)
    _validate_filter_ids([history_start], {o["id"] for o in hist_opts}, "history_start")
    frames = hs.apply_global_filters(frames, broker, history_start)
    view = fs.build_factor_view(frames)
    _finalize_broker_meta(view, broker_opts, broker)
    _finalize_history_meta(view, hist_opts, history_start)
    return Response(json.dumps(view, allow_nan=False),
                    media_type="application/json")


class AiRegenBody(BaseModel):
    model_config = ConfigDict(extra="forbid")
    section: str
    broker: list[str] = ["all"]
    history_start: str = "all"
    account: list[str] = ["all"]
    asset_class: list[str] = ["all"]
    window: str | None = None
    model: str | None = None
    estimator: str | None = None
    benchmark: str | None = None
    sig: str | None = None


_CHAT_MAX_MSGS = 12
_CHAT_MAX_CHARS = 2000


class AiChatMsg(BaseModel):
    model_config = ConfigDict(extra="forbid")
    role: str
    content: str


class AiChatBody(BaseModel):
    model_config = ConfigDict(extra="forbid")
    messages: list[AiChatMsg]
    broker: list[str] = ["all"]
    history_start: str = "all"


class AiChatWarmBody(BaseModel):
    model_config = ConfigDict(extra="forbid")
    broker: list[str] = ["all"]
    history_start: str = "all"


def _dims_or_422(section: str, raw: dict) -> dict | None:
    """Drop unsupplied dims; 422 when a supplied dim NAME is not declared
    by the section. Name validation is route-level (no frames load) so it
    can never depend on cache warmth; VALUE validation happens in the
    reducer on the generating path (AIDimError -> 422), and an invalid
    value can never have been cached."""
    if section not in ai.SECTIONS:
        raise HTTPException(status_code=422, detail="unknown section")
    dims = {k: v for k, v in raw.items() if v is not None}
    allowed = ai.SECTIONS[section].get("dims") or ()
    unknown = sorted(set(dims) - set(allowed))
    if unknown:
        raise HTTPException(
            status_code=422,
            detail=f"section {section!r} takes no dim(s): {', '.join(unknown)}")
    return dims or None


def _ai_response(section: str, broker: list[str], history_start: str,
                 force: bool, dims: dict | None = None,
                 account: list[str] | None = None,
                 asset_class: list[str] | None = None) -> Response:
    """Shared explain/regenerate implementation. Cache hits skip load_frames
    entirely (fast box loads); the miss path validates filters exactly like
    /api/factor. Generation failure degrades to kind:'error' with stale text
    when available — transient API trouble is never a 5xx.

    DELIBERATE: filter validation runs only on the generating (miss+client)
    path, where frames are loaded anyway. A cache hit with a bogus filter is
    impossible (the unambiguous scope_key never had an entry generated for
    it), and the no-key off-state returns enabled:false without validating —
    a 200, not the 422 a validated route gives. Off = quiet, by design."""
    if section not in ai.SECTIONS:
        raise HTTPException(status_code=422, detail="unknown section")
    ddir = _data_dir()
    # B2 (+B3 performance): account/class are tracked cache dims for the
    # filter-threaded sections ONLY, and ONLY when a real filter is requested
    # (frames-free — cache hits must not load frames). Whole-book / other
    # sections (income and dip are whole-book/whole-market by design): dims
    # unchanged -> key stable.
    if section in ("risk", "riskcontrib", "performance", "benchmark"):
        acct_ids = ai._canon_filter(account)
        class_ids = ai._canon_filter(asset_class)
        if acct_ids or class_ids:
            dims = dict(dims or {})
            if acct_ids:
                dims["account"] = acct_ids
            if class_ids:
                dims["asset_class"] = class_ids
    skey = ai.scope_key(broker, history_start, dims)
    payload = {"enabled": True, "kind": "ok", "section": section,
               "question": ai.SECTIONS[section].get("question"),
               "text": None, "error": None, "generated_at": None,
               "model": None, "cached": False, "stale": False,
               "data_version": None}

    def _done(entry: dict, *, cached: bool, stale: bool, kind="ok",
              error=None) -> Response:
        payload.update({"kind": kind, "error": error, "cached": cached,
                        "stale": stale, "text": entry.get("text"),
                        "generated_at": entry.get("generated_at"),
                        "model": entry.get("model"),
                        "data_version": entry.get("data_version")})
        if section == "portfolio":
            payload["narrative"] = ai.parse_narrative(entry.get("text") or "")
            if payload["narrative"] is None and payload["kind"] == "ok":
                payload.update(kind="error", error="cached narrative malformed")
        else:
            # Box sections: parsed structured body (None = legacy prose;
            # the FE renders text as a plain paragraph then). Never flips
            # kind — prose is a valid degraded state, not an error.
            payload["structured"] = ai.parse_box(entry.get("text") or "")
        return Response(json.dumps(payload, allow_nan=False),
                        media_type="application/json")

    def _generating() -> Response:
        payload.update(kind="generating", text=None, error=None,
                       cached=False, stale=False)
        return Response(json.dumps(payload, allow_nan=False),
                        media_type="application/json")

    try:
        dv = ai.data_version(ddir)
    except FileNotFoundError:
        raise HTTPException(status_code=503,
                            detail="data directory or positions.csv not found")

    st = ai.job_status(section, skey)            # B1b: all sections async
    if st and st["status"] == "running":
        # An in-flight job (incl. a force regenerate) must win over any same-dv
        # cache entry, else a Regenerate poll serves the PRE-regen text.
        return _generating()

    cached = ai.cache_get(ddir, section, skey)   # computed before BOTH checks below

    if st and st["status"] == "error":
        # A recorded failure wins over a fresh same-dv cache (else a poll silently
        # re-serves the pre-regen text as ok and the error entry leaks).
        ai.clear_job(section, skey)
        if cached:
            return _done(cached, cached=True, stale=True, kind="error",
                         error=st["error"])
        return _done({}, cached=False, stale=False, kind="error",
                     error=st["error"])

    if cached and ai.entry_fresh(cached, dv) and not force:
        return _done(cached, cached=True, stale=False)

    client = ai.resolve_client()
    if client is None:
        payload["enabled"] = False
        return Response(json.dumps(payload, allow_nan=False),
                        media_type="application/json")

    try:
        frames = hs.load_frames(ddir)
    except FileNotFoundError:
        raise HTTPException(status_code=503,
                            detail="data directory or positions.csv not found")
    broker_opts = hs._broker_options(hs._current_snap(frames))[0]
    _validate_filter_ids(broker, {o["id"] for o in broker_opts}, "broker")
    hist_opts = hs._history_start_options(frames)
    _validate_filter_ids([history_start], {o["id"] for o in hist_opts},
                         "history_start")
    snap = hs._current_snap(frames)
    _validate_filter_ids(account or ["all"],
                         {o["id"] for o in hs._account_options(snap)[0]},
                         "account")
    _validate_filter_ids(asset_class or ["all"],
                         {o["id"] for o in hs._class_options(snap)[0]},
                         "asset_class")
    frames = hs.apply_global_filters(frames, broker, history_start)
    try:
        facts = ai.build_facts(section, frames, history_start=history_start,
                               broker=broker, dims=dims)
    except ai.AIDimError as e:
        raise HTTPException(status_code=422, detail=str(e))

    ai.start_generation(ddir, section, skey, dv, facts, client=client, force=force)
    return _generating()


@app.get("/api/ai/explain")
def ai_explain(section: str = Query(..., max_length=32),
               broker: list[str] = Query(["all"]),
               history_start: str = Query("all"),
               account: list[str] = Query(["all"]),
               asset_class: list[str] = Query(["all"]),
               window: str | None = Query(None, max_length=32),
               model: str | None = Query(None, max_length=32),
               estimator: str | None = Query(None, max_length=32),
               benchmark: str | None = Query(None, max_length=32),
               sig: str | None = Query(None, max_length=64)) -> Response:
    """Cached-first AI narration for one section. Unknown section/filter/
    dim -> 422; missing data dir -> 503; no key/package -> 200 enabled:false."""
    dims = _dims_or_422(section, {"window": window, "model": model,
                                  "estimator": estimator,
                                  "benchmark": benchmark, "sig": sig})
    return _ai_response(section, broker, history_start, force=False,
                        dims=dims, account=account, asset_class=asset_class)


@app.post("/api/ai/regenerate")
def ai_regenerate(body: AiRegenBody) -> Response:
    """Force regeneration for one section+scope (the box's Regenerate)."""
    dims = _dims_or_422(body.section,
                        {"window": body.window, "model": body.model,
                         "estimator": body.estimator,
                         "benchmark": body.benchmark, "sig": body.sig})
    return _ai_response(body.section, body.broker, body.history_start,
                        force=True, dims=dims,
                        account=body.account, asset_class=body.asset_class)


def _chat_scope_frames_fn(ddir: str, broker: list[str], history_start: str):
    """Shared chat/warm miss path: load frames (503), validate the global
    filter ids (422), and return a LAZY frames_fn for ai.chat_pack_ensure —
    one derivation, so the warm's memo key always equals the chat's."""
    try:
        frames = hs.load_frames(ddir)
    except FileNotFoundError:
        raise HTTPException(status_code=503,
                            detail="data directory or positions.csv not found")
    broker_opts = hs._broker_options(hs._current_snap(frames))[0]
    _validate_filter_ids(broker, {o["id"] for o in broker_opts}, "broker")
    hist_opts = hs._history_start_options(frames)
    _validate_filter_ids([history_start], {o["id"] for o in hist_opts},
                         "history_start")
    broker = list(broker)
    return lambda: hs.apply_global_filters(frames, broker, history_start)


def _chat_detail_fn(ddir, broker, history_start):
    """fetch_detail executor for one chat turn (full-gate S2). Lazy:
    frames load on the FIRST tool call only — pack-memo-hit turns that
    never fetch stay frames-free — then memoized for the turn. Successful
    (topic, ticker) results are also memoized in state — riskcontrib
    alone costs ~21-26s on the real book, and a repeat/nudge-round call
    must not pay full price twice; errors are never cached so a
    transient failure can retry. Reducer failures are contained to an
    {"error": ...} result (the loop ships is_error and the turn
    continues); AIScrubError propagates (the turn fails closed — never
    ship unscrubbed bytes)."""
    state: dict = {}
    broker = list(broker)

    def detail_fn(topic, ticker):
        # Memo key normalized exactly as run_detail will ("strl" ==
        # "STRL" == " Strl "); the raw ticker still flows through so
        # run_detail stays the single normalization/validation point.
        key = (topic, ai.normalize_detail_ticker(ticker))
        if key in state:
            return state[key]
        try:
            if "frames" not in state:
                state["frames"] = hs.apply_global_filters(
                    hs.load_frames(ddir), broker, history_start)
            out = ai.run_detail(state["frames"], history_start, broker,
                                topic, ticker)
            if "error" not in out:
                state[key] = out
            return out
        except ai.AIScrubError:
            raise
        except Exception as e:      # noqa: BLE001 — containment by contract
            return {"error": f"{topic} failed: {type(e).__name__}"}

    return detail_fn


@app.post("/api/ai/chat")
def ai_chat(body: AiChatBody) -> Response:
    """Start one chat turn (v2-S2). Off-state = 200 enabled:false WITHOUT
    semantic validation (doctrine). The pack memo (pre-warmed by
    /api/ai/chat/warm when the AI tab renders) makes repeat turns
    frames-free; a miss builds under the scope's lock, so a turn landing
    during a warm waits on that build instead of duplicating it. Broker/
    history ids are validated on the miss path only, exactly like
    _ai_response's generating path. Answers are never cached — the job
    entry carries the text and the first successful poll pops it."""
    client = ai.resolve_client()
    if client is None:
        return Response(json.dumps({"enabled": False}),
                        media_type="application/json")
    msgs = [{"role": m.role, "content": m.content} for m in body.messages]
    msgs = msgs[-_CHAT_MAX_MSGS:]
    # Drop only a stray leading ASSISTANT turn (the 12-message window can
    # start mid-conversation on a reply); a leading role that is neither
    # user nor assistant is not trim spillover, it's malformed input, and
    # must reach the "bad role" 422 below rather than be silently eaten.
    while msgs and msgs[0]["role"] == "assistant":
        msgs.pop(0)
    if not msgs:
        raise HTTPException(status_code=422, detail="messages required")
    if any(m["role"] not in ("user", "assistant") for m in msgs):
        raise HTTPException(status_code=422, detail="bad role")
    if msgs[-1]["role"] != "user":
        raise HTTPException(status_code=422,
                            detail="last message must be role user")
    # USER messages only: assistant echoes are server-authored context the
    # FE resends verbatim — capping them wedged the chat after one long
    # answer (final-review catch; spec Update 2026-08-20).
    if any(len(m["content"]) > _CHAT_MAX_CHARS
           for m in msgs if m["role"] == "user"):
        raise HTTPException(status_code=422, detail="message too long")
    if any(not m["content"].strip() for m in msgs):
        raise HTTPException(status_code=422, detail="empty message")
    ddir = _data_dir()
    try:
        dv = ai.data_version(ddir)
    except FileNotFoundError:
        raise HTTPException(status_code=503,
                            detail="data directory or positions.csv not found")
    pack = ai.chat_pack_get(dv, body.history_start, body.broker)
    if pack is None:
        pack = ai.chat_pack_ensure(            # waits on an in-flight warm
            dv, body.history_start, body.broker,
            _chat_scope_frames_fn(ddir, body.broker, body.history_start))
    chat_id = uuid.uuid4().hex[:16]
    ai.start_chat(chat_id, msgs, pack, client=client,
                  detail_fn=_chat_detail_fn(ddir, body.broker,
                                            body.history_start))
    return Response(json.dumps({"enabled": True, "kind": "generating",
                                "chat_id": chat_id}),
                    media_type="application/json")


@app.get("/api/ai/chat")
def ai_chat_poll(id: str = Query(..., max_length=64)) -> Response:
    """Poll one chat turn. Terminal reads (ok/error) POP the entry —
    answers are ephemeral by design; the FE holds the history."""
    st = ai.chat_status(id)
    if st is None:
        raise HTTPException(status_code=404, detail="unknown chat id")
    if st["status"] == "running":
        return Response(json.dumps({"kind": "generating"}),
                        media_type="application/json")
    ai.clear_chat(id)
    if st["status"] == "error":
        return Response(json.dumps({"kind": "error", "text": None,
                                    "error": st["error"]}),
                        media_type="application/json")
    return Response(json.dumps({"kind": "ok", "text": st["text"],
                                "model": ai.MODEL}),
                    media_type="application/json")


@app.post("/api/ai/chat/warm")
def ai_chat_warm(body: AiChatWarmBody) -> Response:
    """Pre-build the chat facts pack for one scope — fired when the AI tab
    renders so the first turn costs only model time (~1 min of reducer
    work on the real book otherwise). Off-state = 200 enabled:false before
    validation (doctrine); memo hit -> ready; a build already holding the
    scope's lock -> building (no second thread); miss -> ids validated on
    loaded frames (422), reducers run in a background thread -> building;
    a scope whose last warm raised under this data_version -> failed (not
    re-spawned — the next chat turn rebuilds and surfaces the error)."""
    if ai.resolve_client() is None:
        return Response(json.dumps({"enabled": False}),
                        media_type="application/json")
    ddir = _data_dir()
    try:
        dv = ai.data_version(ddir)
    except FileNotFoundError:
        raise HTTPException(status_code=503,
                            detail="data directory or positions.csv not found")
    state = ai.chat_pack_state(dv, body.history_start, body.broker)
    if state == "missing":
        frames_fn = _chat_scope_frames_fn(ddir, body.broker, body.history_start)
        broker, history_start = list(body.broker), body.history_start
        # Re-check after the load/validate window so two warms for the same
        # scope landing together spawn one thread, not two (review catch).
        state = ai.chat_pack_state(dv, history_start, broker)
        if state == "missing":
            ai._SPAWN(lambda: ai.chat_pack_warm(dv, history_start, broker,
                                                frames_fn))
            state = "building"
    return Response(json.dumps({"enabled": True, "status": state}),
                    media_type="application/json")


@app.get("/api/ai/portfolio")
def ai_portfolio(broker: list[str] = Query(["all"]),
                 history_start: str = Query("all"),
                 benchmark: str = Query("auto")) -> Response:
    """The AI ANALYSIS tab payload. Facts + display tables are computed
    fresh per request and ALWAYS served when data exists; the narrative
    rides the S1 cache under section 'portfolio', keyed by the RESOLVED
    benchmark so SPY and 60/40 prose cache separately. ``benchmark``
    (auto/spy/60_40) selects the comparison series (auto -> resolve_benchmark
    per broker scope). Unknown benchmark/filter -> 422; missing data dir ->
    503."""
    if benchmark not in {"auto", "spy", "60_40"}:
        raise HTTPException(status_code=422, detail="unknown benchmark")
    try:
        frames = hs.load_frames(_data_dir())
    except FileNotFoundError:
        raise HTTPException(status_code=503,
                            detail="data directory or positions.csv not found")
    broker_opts = hs._broker_options(hs._current_snap(frames))[0]
    _validate_filter_ids(broker, {o["id"] for o in broker_opts}, "broker")
    hist_opts = hs._history_start_options(frames)
    _validate_filter_ids([history_start], {o["id"] for o in hist_opts},
                         "history_start")
    meta = ai.portfolio_meta(frames)
    frames_f = hs.apply_global_filters(frames, broker, history_start)
    # Resolve auto -> concrete from the broker scope; 60/40 with no AGG data
    # degrades to SPY (state stays ok) — mirrors benchmark_service.
    resolved = hs.resolve_benchmark(benchmark, frames_f.broker_scope)
    unavailable_fallback = False
    if resolved == "60_40" and not hs._agg_available(frames_f):
        resolved, unavailable_fallback = "spy", True
    meta["benchmark"] = {"id": resolved, "label": hs.BENCHMARKS[resolved],
                         "short": hs.BENCH_SHORT[resolved],
                         "requested": benchmark,
                         "unavailable_fallback": unavailable_fallback}
    dims = {"benchmark": resolved}
    facts = ai.build_facts("portfolio", frames_f,
                           history_start=history_start, broker=broker,
                           dims=dims)
    display = ai.portfolio_display(facts)

    ddir = _data_dir()
    skey = ai.scope_key(broker, history_start, dims)
    dv = ai.data_version(ddir)
    payload = {"enabled": True, "kind": "ok", "error": None,
               "narrative": None,
               "questions": ai.portfolio_questions(meta["benchmark"]["short"]),
               "narrative_meta": {"generated_at": None, "model": None,
                                  "cached": False, "stale": False,
                                  "data_version": None},
               "facts": facts, "display": display, "meta": meta}

    def _mount(entry: dict, *, cached: bool, stale: bool) -> None:
        payload["narrative"] = ai.parse_narrative(entry.get("text") or "")
        payload["narrative_meta"] = {
            "generated_at": entry.get("generated_at"),
            "model": entry.get("model"), "cached": cached, "stale": stale,
            "data_version": entry.get("data_version")}

    st = ai.job_status("portfolio", skey)
    if st and st["status"] == "running":
        payload["kind"] = "generating"            # narrative null; the FE polls
    else:
        cached = ai.cache_get(ddir, "portfolio", skey)
        if st and st["status"] == "error":
            ai.clear_job("portfolio", skey)
            if cached:
                _mount(cached, cached=True, stale=True)
            payload.update(kind="error", error=st["error"])
        elif cached and ai.entry_fresh(cached, dv):
            _mount(cached, cached=True, stale=False)
            if payload["narrative"] is None:
                payload.update(kind="error", error="cached narrative malformed")
        else:
            client = ai.resolve_client()
            if client is None:
                payload["enabled"] = False
            else:
                ai.start_generation(ddir, "portfolio", skey, dv, facts,
                                    client=client)
                payload["kind"] = "generating"

    _finalize_broker_meta(payload, broker_opts, broker)
    _finalize_history_meta(payload, hist_opts, history_start)
    return Response(json.dumps(payload, allow_nan=False),
                    media_type="application/json")


@app.get("/api/tax")
def tax(broker: list[str] = Query(["all"]),
        history_start: str = Query("all")) -> Response:
    """Return the Tax view (open lots from the gated ledger, slice 2b).

    Whole-book — no Account / Asset-class filter params: the GLOBAL account
    picker lists the IRAs a tax view must exclude, so in-tab account/term/
    evidence filters are client-side (spec Update 2026-07-27). ``broker``
    is validated + applied (narrows to that broker's taxable accounts).
    ``history_start`` is accepted + validated for the global contract's
    uniformity but inert here — the ledger is a today-snapshot (the same
    fact behind build_lots' --as-of write refusal). Missing data dir ->
    503; a missing lots.csv is a kind:"error" view (valid request, ledger
    not built yet)."""
    try:
        frames = hs.load_frames(_data_dir())
    except FileNotFoundError:
        raise HTTPException(status_code=503,
                            detail="data directory or positions.csv not found")
    broker_opts = hs._broker_options(hs._current_snap(frames))[0]
    broker_ids = {o["id"] for o in broker_opts}
    _validate_filter_ids(broker, broker_ids, "broker")
    hist_opts = hs._history_start_options(frames)
    _validate_filter_ids([history_start], {o["id"] for o in hist_opts}, "history_start")
    view = txs.build_tax_view(frames, _data_dir(), broker=broker)
    _finalize_broker_meta(view, broker_opts, broker)
    _finalize_history_meta(view, hist_opts, history_start)
    return Response(json.dumps(view, allow_nan=False),
                    media_type="application/json")


class TaxSimLeg(BaseModel):
    """One hypothetical sale leg: an open lot (by the view's lot_id)
    and a quantity. Both fields carry their natural declarative
    constraint — ``qty`` positive AND finite (``allow_inf_nan=False``
    rejects NaN/Inf the same way ``gt=0`` rejects zero/negative) — and
    the app-wide RequestValidationError handler above is what makes
    that safe: without it, a NaN/Inf value would still be correctly
    REJECTED here, but rendering the 422 (which echoes the raw
    offending value) would crash instead."""
    model_config = ConfigDict(extra="forbid")
    lot_id: int = Field(ge=0)
    qty: float = Field(gt=0, allow_inf_nan=False)


class TaxOverrides(BaseModel):
    """Request-scoped TAX_PROFILE overlay; never persisted server-side.
    ``extra='forbid'`` rejects unknown fields."""
    model_config = ConfigDict(extra="forbid")
    filing_status: str | None = None
    w2_income: float | None = None
    state: str | None = None
    deduction: str | float | None = None
    carryforward_loss: float | None = None
    qualified_dividend_pct: float | None = None
    unknown_term_assumption: str | None = None


class TaxEstimateBody(BaseModel):
    model_config = ConfigDict(extra="forbid")
    overrides: TaxOverrides = Field(default_factory=TaxOverrides)
    sim: list[TaxSimLeg] = Field(default_factory=list)


@app.post("/api/tax/estimate")
def tax_estimate_route(body: TaxEstimateBody) -> Response:
    """Year tax estimate (+ sell simulator). Malformed shape -> 422
    (sim leg shape/range is pydantic-enforced on TaxSimLeg itself, made
    crash-safe by the app-wide validation-error handler above);
    unconfigured profile / unsupported state / missing realized block ->
    200 kind:"error" (valid request, named degrade); invalid sim legs ->
    200 with baseline + sim_rejected (all-or-nothing, spec §5.1).
    503 on missing data dir."""
    # TaxOverrides fields are plain `float | None` / `str | float | None`
    # (no Field constraint — None must stay valid, and a bare `deduction`
    # dollar amount has no natural upper bound), so a NaN/Inf value sails
    # through pydantic unrejected; only `deduction`'s numeric member needs
    # the isinstance guard (a "standard" string is a valid, non-numeric
    # deduction and must skip this check untouched).
    for name in ("w2_income", "carryforward_loss",
                 "qualified_dividend_pct", "deduction"):
        v = getattr(body.overrides, name)
        if (isinstance(v, (int, float)) and not isinstance(v, bool)
                and not math.isfinite(v)):
            raise HTTPException(status_code=422,
                                detail=f"{name} must be finite")
    try:
        frames = hs.load_frames(_data_dir())
    except FileNotFoundError:
        raise HTTPException(status_code=503,
                            detail="data directory or positions.csv not found")
    ov = {k: v for k, v in body.overrides.model_dump().items()
          if v is not None}
    view = txs.build_tax_estimate(
        frames, _data_dir(), overrides=ov,
        sim=[leg.model_dump() for leg in body.sim])
    return Response(json.dumps(view, allow_nan=False),
                    media_type="application/json")


@app.get("/api/dip")
def dip(broker: list[str] = Query(["all"]),
        history_start: str = Query("all")) -> Response:
    """Return the Buy the Dip view (turbulence banner + legend + the SPY/SCHD/GLD
    auto cards). Whole-book — no Account / Asset-class filter params (the cards
    are about the dip history, not the portfolio). ``broker`` is accepted +
    validated + applied for the global contract's uniformity, though the cards
    are whole-market and do not depend on ``frames.positions``. ``history_start``
    (S3) is accepted + validated + applied the same way. Missing data dir
    -> 503. The ad-hoc live-ticker lookup is a separate (PR B) surface, not
    served here."""
    try:
        frames = hs.load_frames(_data_dir())
    except FileNotFoundError:
        raise HTTPException(status_code=503,
                            detail="data directory or positions.csv not found")
    broker_opts = hs._broker_options(hs._current_snap(frames))[0]
    broker_ids = {o["id"] for o in broker_opts}
    _validate_filter_ids(broker, broker_ids, "broker")
    hist_opts = hs._history_start_options(frames)
    _validate_filter_ids([history_start], {o["id"] for o in hist_opts}, "history_start")
    frames = hs.apply_global_filters(frames, broker, history_start)
    view = ds.build_dip_view(frames)
    _finalize_broker_meta(view, broker_opts, broker)
    _finalize_history_meta(view, hist_opts, history_start)
    return Response(json.dumps(view, allow_nan=False),
                    media_type="application/json")


@app.get("/api/dip/lookup")
def dip_lookup(
    ticker: str = Query(..., min_length=1, max_length=10,
                        pattern=r"^[A-Za-z0-9.\-]{1,10}$"),
    broker: list[str] = Query(["all"]),
    history_start: str = Query("all"),
) -> Response:
    """Resolve one typed ad-hoc ticker to a dip-card payload (live Yahoo fetch,
    sidecar-cached). The ``ticker`` allowlist (422 before this body runs) is the
    security control for this first network-on-user-input surface — the value
    only fills a symbol slot in yfinance's fixed Yahoo endpoint; the 10-char cap
    also keeps it under yfinance's 12-char ISIN side-channel. ``broker`` is
    accepted + validated + applied for the global contract's uniformity — a
    harmless no-op here since the ad-hoc lookup is whole-market. ``history_start``
    (S3) is accepted + validated + applied the same way. Missing data
    dir -> 503."""
    try:
        frames = hs.load_frames(_data_dir())
    except FileNotFoundError:
        raise HTTPException(status_code=503,
                            detail="data directory or positions.csv not found")
    broker_opts = hs._broker_options(hs._current_snap(frames))[0]
    broker_ids = {o["id"] for o in broker_opts}
    _validate_filter_ids(broker, broker_ids, "broker")
    hist_opts = hs._history_start_options(frames)
    _validate_filter_ids([history_start], {o["id"] for o in hist_opts}, "history_start")
    frames = hs.apply_global_filters(frames, broker, history_start)
    view = ds.build_dip_lookup(frames, ticker)
    return Response(json.dumps(view, allow_nan=False),
                    media_type="application/json")


@app.get("/api/risksim")
def risksim(account: list[str] = Query(["all"]),
            asset_class: list[str] = Query(["all"]),
            broker: list[str] = Query(["all"]),
            history_start: str = Query("all")) -> Response:
    """Return the Risk Simulation seed (current-weights grid + guards) for the
    requested filters. Validates account / asset_class against the known
    option-id sets (422 otherwise); no as_of (latest snapshot). 503 on missing
    data dir. ``broker`` is the GLOBAL filter — validated + applied before
    account/class. ``history_start`` (S3) narrows the same way, applied
    alongside ``broker``. The Run compute is POST /api/risksim/simulate."""
    try:
        frames = hs.load_frames(_data_dir())
    except FileNotFoundError:
        raise HTTPException(status_code=503,
                            detail="data directory or positions.csv not found")
    broker_opts = hs._broker_options(hs._current_snap(frames))[0]
    broker_ids = {o["id"] for o in broker_opts}
    _validate_filter_ids(broker, broker_ids, "broker")
    hist_opts = hs._history_start_options(frames)
    _validate_filter_ids([history_start], {o["id"] for o in hist_opts}, "history_start")
    frames = hs.apply_global_filters(frames, broker, history_start)
    account_ids, class_ids = hs.filter_option_ids(frames)
    _validate_filter_ids(account, account_ids, "account")
    _validate_filter_ids(asset_class, class_ids, "asset_class")
    view = rss.build_risksim_view(frames, account=account, asset_class=asset_class)
    _finalize_broker_meta(view, broker_opts, broker)
    _finalize_history_meta(view, hist_opts, history_start)
    return Response(json.dumps(view, allow_nan=False), media_type="application/json")


@app.get("/api/risksim/progress")
def risksim_progress() -> Response:
    """Sweep progress for the trace/frontier progress bar (polled ~1s by the
    FE during a POST). Reads only the in-process slot — no frames load, no
    data-dir dependency, so never 503. Idle: {"running": false}."""
    return Response(json.dumps(rss.sweep_progress(), allow_nan=False),
                    media_type="application/json")


# NOTE: as of the filter-parity multi-select slice, ``account`` / ``asset_class``
# on all three POST bodies below are ``list[str]`` (scalar ids collapse to a
# one-element list; ``["all"]`` = unfiltered). The front-end (``static/app.js``)
# sends them as JSON arrays for its multi-select, matching the repeated-value
# ``?account=..&account=..`` the GET routes accept — Task 5 wires that UI.
# ``broker`` (added in S2a) is the GLOBAL filter and joins them on all three
# bodies: validated + applied to ``frames`` before the account/class gate.
# ``history_start`` (added in S3) is the GLOBAL history-start cutoff and joins
# them the same way, threaded into the same apply_global_filters call.
class Candidate(BaseModel):
    """One what-if candidate ticker for the optimizer opportunity set. Loopback
    single-origin. ``ticker`` required (1-10 allowlisted chars), ``asset_class``
    one of the four floor buckets, ``proxy`` optional (short-history splice
    source). Same charset control the simulate route uses for network-on-user-
    input tickers."""
    model_config = ConfigDict(extra="forbid")
    ticker: str = Field(pattern=r"^[A-Za-z0-9.\-]{1,10}$")
    asset_class: Literal["equity", "fixed_income", "gold", "other"]
    proxy: str = Field(default="", pattern=r"^[A-Za-z0-9.\-]{0,10}$")


class SimulateBody(BaseModel):
    """POST /api/risksim/simulate body. Loopback single-origin: no CSRF surface.
    ``extra='forbid'`` rejects unknown fields; weight values are finite floats in
    [0, 1000] (a loose garbage ceiling — real grid cells are <= 100).
    ``candidates`` (up to 3, reusing the ``Candidate`` model) models not-held
    names (fetched on Run); each candidate's optional ``proxy`` back-fills a
    short history with a splice source. Tickers are allowlisted to
    ``^[A-Za-z0-9.-]{0,10}$`` (blank allowed) — the same control
    ``/api/dip/lookup`` uses for its network-on-user-input surface."""
    model_config = ConfigDict(extra="forbid")
    account: list[str] = Field(default_factory=lambda: ["all"])
    asset_class: list[str] = Field(default_factory=lambda: ["all"])
    broker: list[str] = Field(default_factory=lambda: ["all"])
    history_start: str = "all"
    weights: dict[str, float] = Field(default_factory=dict)
    candidates: list[Candidate] = Field(default_factory=list, max_length=3)


@app.post("/api/risksim/simulate")
def risksim_simulate(body: SimulateBody) -> Response:
    """Run one what-if reweight and return before/after risk. Universe gate: the
    posted weights must be a SUBSET of the current filtered holdings PLUS the
    posted ``candidates`` tickers (unknown ticker -> 422; a not-held candidate is
    expected to appear in ``weights`` and is excluded from this gate rather than
    rejected). Domain failures (already-held candidate, bad proxy, empty fetch,
    short overlap, over-allocation) come back as a 200 with an ``error`` field
    (a valid request that can't simulate), not a 500 — the handler only forwards
    the candidate fields to ``run_simulation``, which owns those checks. 503 on
    missing data dir."""
    try:
        frames = hs.load_frames(_data_dir())
    except FileNotFoundError:
        raise HTTPException(status_code=503,
                            detail="data directory or positions.csv not found")
    broker_opts = hs._broker_options(hs._current_snap(frames))[0]
    broker_ids = {o["id"] for o in broker_opts}
    _validate_filter_ids(body.broker, broker_ids, "broker")
    hist_opts = hs._history_start_options(frames)
    _validate_filter_ids([body.history_start], {o["id"] for o in hist_opts}, "history_start")
    frames = hs.apply_global_filters(frames, body.broker, body.history_start)

    account_ids, class_ids = hs.filter_option_ids(frames)
    _validate_filter_ids(body.account, account_ids, "account")
    _validate_filter_ids(body.asset_class, class_ids, "asset_class")

    for t, v in body.weights.items():
        if not math.isfinite(v) or v < 0.0 or v > 1000.0:
            raise HTTPException(status_code=422,
                                detail=f"weight out of range for {t}")

    cand_set = {str(c.ticker).strip().upper() for c in body.candidates}
    bundle = rss._bundle_for(frames, body.account, body.asset_class)
    universe = set(bundle["weights"].index.map(str))
    unknown = [t for t in body.weights
              if t not in universe and str(t).strip().upper() not in cand_set]
    if unknown:
        raise HTTPException(status_code=422,
                            detail=f"unknown ticker(s): {', '.join(sorted(unknown))}")

    cand_dicts = [c.model_dump() for c in body.candidates]
    view = rss.run_simulation(frames, body.weights,
                              account=body.account, asset_class=body.asset_class,
                              candidates=cand_dicts, bundle=bundle)
    # Memoize the numeric before/after facts so the tab-scoped AI box narrates
    # this exact run (no re-simulate) via _facts_risksim; the FE echoes the sig
    # on the AI request. Popped from the wire so the response shape is unchanged
    # apart from `sig`. Error runs carry neither facts nor sig -> no box.
    facts = view.pop("ai_facts", None)
    if view.get("error") is None and facts is not None:
        sig = rss.simulate_sig(
            data_version=ai.data_version(_data_dir()),
            broker=body.broker, history_start=body.history_start,
            account=body.account, asset_class=body.asset_class,
            weights=body.weights, candidates=cand_dicts)
        rss.simulate_memo_put(sig, facts, body.account, body.asset_class)
        view = {**view, "sig": sig}
    return Response(json.dumps(view, allow_nan=False), media_type="application/json")


class OptimizeBody(BaseModel):
    """POST /api/risksim/optimize body. Loopback single-origin: no CSRF surface.
    ``extra='forbid'`` rejects unknown fields; cap in [1,100], floors and caps
    each in [0,100], budgets each in (0,100] mirror the Streamlit slider ranges
    (floor, cap, and budget keys validated in-handler against the seed's
    buckets)."""
    model_config = ConfigDict(extra="forbid")
    account: list[str] = Field(default_factory=lambda: ["all"])
    asset_class: list[str] = Field(default_factory=lambda: ["all"])
    broker: list[str] = Field(default_factory=lambda: ["all"])
    history_start: str = "all"
    optimizer: Literal["min_variance", "risk_parity"]
    cap_pct: float = Field(ge=1.0, le=100.0)
    floors: dict[str, float] = Field(default_factory=dict)
    caps: dict[str, float] = Field(default_factory=dict)
    budgets: dict[str, float] = Field(default_factory=dict)
    candidates: list[Candidate] = Field(default_factory=list, max_length=3)


@app.post("/api/risksim/optimize")
def risksim_optimize(body: OptimizeBody) -> Response:
    """Suggest min-variance / risk-parity weights for the grid. Universe gate on
    account/class (422); floor, cap, and budget keys must be among the seed's
    buckets; floors and caps each in [0,100], budgets in (0,100] (422).
    Infeasible-but-well-formed inputs return 200 with
    kind='error'. No external fetch unless ``candidates`` are present (then
    each is fetched cache-first via the offline/Polygon seam, same
    network-on-user-input surface as ``/simulate``). 503 on missing data dir.
    ``broker`` is the GLOBAL filter — validated + applied before account/class."""
    try:
        frames = hs.load_frames(_data_dir())
    except FileNotFoundError:
        raise HTTPException(status_code=503,
                            detail="data directory or positions.csv not found")
    broker_opts = hs._broker_options(hs._current_snap(frames))[0]
    broker_ids = {o["id"] for o in broker_opts}
    _validate_filter_ids(body.broker, broker_ids, "broker")
    hist_opts = hs._history_start_options(frames)
    _validate_filter_ids([body.history_start], {o["id"] for o in hist_opts}, "history_start")
    frames = hs.apply_global_filters(frames, body.broker, body.history_start)

    account_ids, class_ids = hs.filter_option_ids(frames)
    _validate_filter_ids(body.account, account_ids, "account")
    _validate_filter_ids(body.asset_class, class_ids, "asset_class")
    bundle = rss._bundle_for(frames, body.account, body.asset_class)
    known = rss.floor_buckets_for(frames, bundle)
    if known:
        known = known | {c.asset_class for c in body.candidates
                         if c.asset_class != "other"}
    _validate_bucket_pcts(known, floors=body.floors, caps=body.caps,
                          budgets=body.budgets)
    out = rss.run_optimize(frames, optimizer=body.optimizer, cap_pct=body.cap_pct,
                           floors=body.floors, caps=body.caps, budgets=body.budgets,
                           account=body.account, asset_class=body.asset_class,
                           candidates=[c.model_dump() for c in body.candidates],
                           bundle=bundle)
    return Response(json.dumps(out, allow_nan=False), media_type="application/json")


class TraceBody(BaseModel):
    """POST /api/risksim/trace body. Loopback single-origin: no CSRF surface.
    ``extra='forbid'`` rejects unknown fields; cap in [1,100], floors and caps
    each in [0,100], budgets each in (0,100] mirror the Streamlit slider ranges
    (floor, cap, and budget keys validated in-handler). No ``optimizer`` field
    — the trace sweeps BOTH optimizers."""
    model_config = ConfigDict(extra="forbid")
    account: list[str] = Field(default_factory=lambda: ["all"])
    asset_class: list[str] = Field(default_factory=lambda: ["all"])
    broker: list[str] = Field(default_factory=lambda: ["all"])
    history_start: str = "all"
    cap_pct: float = Field(ge=1.0, le=100.0)
    floors: dict[str, float] = Field(default_factory=dict)
    caps: dict[str, float] = Field(default_factory=dict)
    budgets: dict[str, float] = Field(default_factory=dict)
    candidates: list[Candidate] = Field(default_factory=list, max_length=3)


@app.post("/api/risksim/trace")
def risksim_trace(body: TraceBody) -> Response:
    """Sweep the cap ladder across both optimizers and return (vol, Effective N)
    per (cap, optimizer) for the concentration-tradeoff chart. Universe gate on
    account/class (422); floor, cap, and budget keys must be among the seed's
    buckets; floors and caps each in [0,100], budgets in (0,100] (422).
    Well-formed-but-can't-build-Σ returns 200 with ``error`` (an
    individually infeasible cap just raises skipped_n). No external fetch
    unless ``candidates`` are present (then each is fetched cache-first via
    the offline/Polygon seam, same network-on-user-input surface as
    ``/simulate``). 503 on missing data dir. ``broker`` is the GLOBAL filter —
    validated + applied before account/class."""
    try:
        frames = hs.load_frames(_data_dir())
    except FileNotFoundError:
        raise HTTPException(status_code=503,
                            detail="data directory or positions.csv not found")
    broker_opts = hs._broker_options(hs._current_snap(frames))[0]
    broker_ids = {o["id"] for o in broker_opts}
    _validate_filter_ids(body.broker, broker_ids, "broker")
    hist_opts = hs._history_start_options(frames)
    _validate_filter_ids([body.history_start], {o["id"] for o in hist_opts}, "history_start")
    frames = hs.apply_global_filters(frames, body.broker, body.history_start)

    account_ids, class_ids = hs.filter_option_ids(frames)
    _validate_filter_ids(body.account, account_ids, "account")
    _validate_filter_ids(body.asset_class, class_ids, "asset_class")
    bundle = rss._bundle_for(frames, body.account, body.asset_class)
    known = rss.floor_buckets_for(frames, bundle)
    if known:
        known = known | {c.asset_class for c in body.candidates
                         if c.asset_class != "other"}
    _validate_bucket_pcts(known, floors=body.floors, caps=body.caps,
                          budgets=body.budgets)
    out = rss.run_trace(frames, cap_pct=body.cap_pct, floors=body.floors,
                        caps=body.caps, budgets=body.budgets, account=body.account,
                        asset_class=body.asset_class,
                        candidates=[c.model_dump() for c in body.candidates],
                        bundle=bundle)
    return Response(json.dumps(out, allow_nan=False), media_type="application/json")


class FrontierBody(BaseModel):
    """POST /api/risksim/frontier body. Loopback single-origin: no CSRF surface.
    ``extra='forbid'`` rejects unknown fields; cap in [1,100], floors and caps
    each in [0,100] (keys validated in-handler), erp in (0,20]. No ``budgets``
    field — risk budgets are a risk-parity concept and the frontier is
    min-variance-family only."""
    model_config = ConfigDict(extra="forbid")
    account: list[str] = Field(default_factory=lambda: ["all"])
    asset_class: list[str] = Field(default_factory=lambda: ["all"])
    broker: list[str] = Field(default_factory=lambda: ["all"])
    history_start: str = "all"
    cap_pct: float = Field(ge=1.0, le=100.0)
    floors: dict[str, float] = Field(default_factory=dict)
    caps: dict[str, float] = Field(default_factory=dict)
    erp_pct: float = Field(default=4.5, gt=0.0, le=20.0)
    candidates: list[Candidate] = Field(default_factory=list, max_length=3)


@app.post("/api/risksim/frontier")
def risksim_frontier(body: FrontierBody) -> Response:
    """Sweep the risk-aversion ladder under the live constraints and return
    (vol, E[r]) per point plus current-book / min-variance / ERC markers for the
    efficient-frontier chart. Expected returns are CAPM-implied estimates — the
    caption discloses rf, ERP, the beta window, and any assumed betas. Universe
    gate on account/class (422); floor and cap keys must be among the seed's
    buckets, each in [0,100] (422). Missing factor file or unbuildable Σ returns
    200 with ``error`` (an individually infeasible point just raises skipped_n).
    No external fetch unless ``candidates`` are present (then each is fetched
    cache-first via the offline/Polygon seam, same network-on-user-input
    surface as ``/simulate``). 503 on missing data dir. ``broker`` is the
    GLOBAL filter — validated + applied before account/class."""
    try:
        frames = hs.load_frames(_data_dir())
    except FileNotFoundError:
        raise HTTPException(status_code=503,
                            detail="data directory or positions.csv not found")
    broker_opts = hs._broker_options(hs._current_snap(frames))[0]
    broker_ids = {o["id"] for o in broker_opts}
    _validate_filter_ids(body.broker, broker_ids, "broker")
    hist_opts = hs._history_start_options(frames)
    _validate_filter_ids([body.history_start], {o["id"] for o in hist_opts},
                         "history_start")
    frames = hs.apply_global_filters(frames, body.broker, body.history_start)

    account_ids, class_ids = hs.filter_option_ids(frames)
    _validate_filter_ids(body.account, account_ids, "account")
    _validate_filter_ids(body.asset_class, class_ids, "asset_class")
    bundle = rss._bundle_for(frames, body.account, body.asset_class)
    known = rss.floor_buckets_for(frames, bundle)
    if known:
        known = known | {c.asset_class for c in body.candidates
                         if c.asset_class != "other"}
    _validate_bucket_pcts(known, floors=body.floors, caps=body.caps)
    out = rss.run_frontier(frames, cap_pct=body.cap_pct, floors=body.floors,
                           caps=body.caps, erp_pct=body.erp_pct,
                           account=body.account, asset_class=body.asset_class,
                           candidates=[c.model_dump() for c in body.candidates],
                           bundle=bundle)
    # Memoize the result so the tab-scoped AI summary box reads it (no ~80s
    # re-run) via _facts_frontier; the FE echoes this sig on the AI request.
    sig = rss.frontier_sig(data_version=ai.data_version(_data_dir()),
                           broker=body.broker, history_start=body.history_start,
                           account=body.account, asset_class=body.asset_class,
                           cap_pct=body.cap_pct, floors=body.floors,
                           caps=body.caps, erp_pct=body.erp_pct)
    rss.frontier_memo_put(sig, out, body.account, body.asset_class)
    out = {**out, "sig": sig}
    return Response(json.dumps(out, allow_nan=False),
                    media_type="application/json")


@app.get("/api/options")
def options(broker: list[str] = Query(["all"]),
            history_start: str = Query("all")) -> Response:
    """Return the Options Hedging read-half view (staleness chips, aggregate
    exposure tiles, IV-percentile gauge/sparkline). Whole-book — no Account /
    Asset-class filter params (the tab ignores them). ``broker`` is the GLOBAL
    filter and DOES apply — it narrows the book before the aggregates.
    ``history_start`` (S3) narrows the same way, applied alongside ``broker``.
    Missing data dir -> 503."""
    try:
        frames = hs.load_frames(_data_dir())
    except FileNotFoundError:
        raise HTTPException(status_code=503,
                            detail="data directory or positions.csv not found")
    broker_opts = hs._broker_options(hs._current_snap(frames))[0]
    broker_ids = {o["id"] for o in broker_opts}
    _validate_filter_ids(broker, broker_ids, "broker")
    hist_opts = hs._history_start_options(frames)
    _validate_filter_ids([history_start], {o["id"] for o in hist_opts}, "history_start")
    frames = hs.apply_global_filters(frames, broker, history_start)
    view = ops.build_options_view(frames)
    _finalize_broker_meta(view, broker_opts, broker)
    _finalize_history_meta(view, hist_opts, history_start)
    return Response(json.dumps(view, allow_nan=False),
                    media_type="application/json")


@app.get("/api/options/recommend")
def options_recommend(
    mode: str = Query(..., pattern=r"^[AB]$"),
    target: float = Query(..., ge=0.0, le=1.0),
    broker: list[str] = Query(["all"]),
    history_start: str = Query("all"),
) -> Response:
    """Return the hedge recommendation for one (mode, target). ``mode`` in {A,B}
    and ``target`` in that mode's allowed set (422 otherwise). Whole-book. Domain
    failures (no universe, chain fetch) come back 200 with warnings/chain_error.
    ``broker`` is the GLOBAL filter — validated + applied before the
    recommendation is built. ``history_start`` (S3) narrows the same way,
    applied alongside ``broker``. 503 on missing data dir."""
    allowed = ops.REC_TARGETS.get(mode, [])
    if not any(abs(target - t) < 1e-9 for t in allowed):
        raise HTTPException(status_code=422, detail="target not in mode's set")
    try:
        frames = hs.load_frames(_data_dir())
    except FileNotFoundError:
        raise HTTPException(status_code=503,
                            detail="data directory or positions.csv not found")
    broker_opts = hs._broker_options(hs._current_snap(frames))[0]
    broker_ids = {o["id"] for o in broker_opts}
    _validate_filter_ids(broker, broker_ids, "broker")
    hist_opts = hs._history_start_options(frames)
    _validate_filter_ids([history_start], {o["id"] for o in hist_opts}, "history_start")
    frames = hs.apply_global_filters(frames, broker, history_start)
    # snap target to the canonical decimal so label lookup + golden are exact
    target = next(t for t in allowed if abs(target - t) < 1e-9)
    view = ops.build_recommend_view(frames, mode=mode, target=target)
    return Response(json.dumps(view, allow_nan=False),
                    media_type="application/json")


@app.get("/api/chrome")
def chrome(broker: list[str] = Query(["all"]),
           history_start: str = Query("all")) -> Response:
    """Global chrome (QA-polish S7): staleness warnings, data-sources panel,
    regime badge, footer. Whole-book like app.py's sidebar chrome — ``broker``
    + ``history_start`` (the global filters) apply; Account/Asset-class do
    not. Missing data dir -> 503."""
    try:
        frames = hs.load_frames(_data_dir())
    except FileNotFoundError:
        raise HTTPException(status_code=503,
                            detail="data directory or positions.csv not found")
    broker_opts = hs._broker_options(hs._current_snap(frames))[0]
    _validate_filter_ids(broker, {o["id"] for o in broker_opts}, "broker")
    hist_opts = hs._history_start_options(frames)
    _validate_filter_ids([history_start], {o["id"] for o in hist_opts}, "history_start")
    frames = hs.apply_global_filters(frames, broker, history_start)
    view = cs.build_chrome_view(frames, _data_dir())
    return Response(json.dumps(view, allow_nan=False),
                    media_type="application/json")


@app.get("/api/actions")
def actions_roster() -> Response:
    """Action-button roster + current job state (QA-polish S6). These routes
    never touch the data dir — the next tab fetch reloads frames per-request,
    so a finished job's CSVs appear without a restart."""
    return Response(json.dumps(acts.actions_view(), allow_nan=False),
                    media_type="application/json")


@app.get("/api/actions/status")
def actions_status() -> Response:
    """Job-state snapshot — the front-end's 1-second poll target."""
    return Response(json.dumps(acts.status(), allow_nan=False),
                    media_type="application/json")


@app.post("/api/actions/{action_id}")
def actions_start(action_id: str) -> dict:
    """Start a parser action. 422 unknown id · 409 while another job runs.
    Single-flight is enforced in the service (the parsers contend on the same
    data/ CSVs); the server is loopback-only, so the trust domain matches
    Streamlit's sidebar buttons."""
    if action_id not in acts.ACTIONS:
        raise HTTPException(status_code=422, detail="unknown action")
    ok, why = acts.start(action_id)
    if not ok:
        raise HTTPException(status_code=409, detail=why)
    return {"started": True}


@app.get("/tokens.css")
def tokens_css() -> Response:
    """The MERIDIAN design tokens as a ``:root{…}`` CSS block (from theme.py)."""
    return Response(root_css(), media_type="text/css")


@app.get("/")
def index() -> FileResponse:
    """The static front-end shell."""
    return FileResponse(STATIC / "index.html")


# Mount static LAST so the explicit routes above win on overlapping paths.
app.mount("/", StaticFiles(directory=STATIC), name="static")
