/* ============================================================================
   MERIDIAN Portfolio Terminal — Holdings front-end (vanilla, no framework).
   On load: fetch('/api/holdings' + location.search) and render every section.
   Selects (as-of / account / asset-class) re-fetch; search filters client-side.
   ============================================================================ */
'use strict';

/* ---- tiny DOM helpers ---- */
const $ = (id) => document.getElementById(id);

function el(tag, attrs, children) {
  const node = document.createElement(tag);
  if (attrs) {
    for (const [k, v] of Object.entries(attrs)) {
      if (v == null || v === false) continue;
      if (k === 'class') node.className = v;
      else if (k === 'html') node.innerHTML = v;
      else if (k === 'text') node.textContent = v;
      else node.setAttribute(k, v);
    }
  }
  if (children != null) {
    for (const c of [].concat(children)) {
      if (c == null) continue;
      node.appendChild(typeof c === 'string' ? document.createTextNode(c) : c);
    }
  }
  return node;
}
const SVGNS = 'http://www.w3.org/2000/svg';
function svgEl(tag, attrs) {
  const node = document.createElementNS(SVGNS, tag);
  if (attrs) for (const [k, v] of Object.entries(attrs)) {
    if (v != null) node.setAttribute(k, v);
  }
  return node;
}

/* ============ SORTABLE TABLE HEADERS (shared by every terminal tab) =======
   Click a header to sort; click again to reverse; a third click restores
   the table's natural (server/builder) order. One helper + a one-line
   opt-in per table instead of a bespoke sort per tab -- same shape as the
   chart-interactivity pass's shared niceTicks/attachAxes/attachCrosshair
   (#237-#240). See makeSortable() below for the call contract. */

/* Sort state per table, keyed by _sortKeyFor() -- survives re-renders (a
   filter change rebuilds the whole tbody) so the user's column/direction
   choice sticks until THEY change it. Tables with no id are sorted but not
   remembered (nothing stable to key an entry on). */
const _sortState = new Map();

/* Three Tax views (open lots / harvest / realized) all render into the
   SAME #tax-lots element with DIFFERENT column sets. Keying purely by
   element id would let a remembered column INDEX from one view silently
   re-apply to a different view's columns after a view switch (column 3 is
   "Qty" in open lots, "Closes" in realized -- same index, unrelated
   meaning). Folding the header labels into the key partitions state by
   column-set automatically: switching views (different labels) starts
   that view natural; switching back restores that view's OWN remembered
   sort; a plain filter change (same view, same labels) keeps the sort.
   No bookkeeping needed at the three view-switch call sites. */
function _sortKeyFor(tableEl, labels) {
  return tableEl.id + '::' + labels.join('|');
}

/* Strip decoration so what remains is Number()-parseable: a leading chip
   arrow (buildChip's up/down glyph -- redundant with the sign already in
   the text), a trailing dagger (statement-mark provenance, not
   magnitude), currency/thousands/percent/plus signs and whitespace, and
   the U+2212 minus MERIDIAN's signed-money helper renders negative
   amounts with (holdings_service._signed_money) -- Number() only
   recognizes ASCII '-'. */
function _sortClean(text) {
  return text
    .replace(/^[▲▼]/, '')
    .replace(/†$/, '')
    .replace(/[$,%+\s]/g, '')
    .replace(/^−/, '-');
}

/* Parse one cell's text into a sort value. Missing is '—', empty, or (once
   decoration is stripped) nothing left but a bare sign -- a few server
   formatters (income_service's local money()/_pct() helpers) render a
   plain '-' for missing instead of the app-standard '—'; both mean "not
   known" and must sink the same way in BOTH directions, or a column of
   unknowns floating to the top under descending would read as an extreme.
   This is an honesty rule, not a taste call. */
function _sortParse(raw) {
  const trimmed = (raw == null ? '' : String(raw)).trim();
  if (trimmed === '' || trimmed === '—') {
    return { missing: true, num: null, text: trimmed };
  }
  const cleaned = _sortClean(trimmed);
  // test the CLEANED value too, not just the raw one above: a decorated
  // missing marker (a chip arrow glued on -- '▲—', or a dagger -- '—†')
  // fails the raw test but strips down to a bare em dash or a bare sign,
  // and either one still means "not known" (FIX 3, slice-1 review; not
  // reachable in slice 1's five tables, but this parser is the contract
  // for the ~16 tables coming in slice 2).
  if (cleaned === '—' || /^[+-]?$/.test(cleaned)) {
    return { missing: true, num: null, text: trimmed };
  }
  const n = Number(cleaned);
  return { missing: false, num: Number.isFinite(n) ? n : null, text: trimmed };
}

/* Value for one block's leader cell at colIndex. opts.key, if given, runs
   FIRST and may return: undefined (no opinion -- fall through to the
   normal resolution below), null (explicitly missing), a finite number
   (used as-is), or anything else (re-parsed as text). Next td.dataset.sort,
   then the rendered text. */
function _sortCellValue(block, colIndex, opts) {
  const td = block.leader.children[colIndex];
  if (!td) return { missing: true, num: null, text: '' };
  if (opts.key) {
    const v = opts.key(td, colIndex);
    if (v !== undefined) {
      if (v === null) return { missing: true, num: null, text: '' };
      if (typeof v === 'number') {
        return Number.isFinite(v) ? { missing: false, num: v, text: String(v) }
          : { missing: true, num: null, text: '' };
      }
      return _sortParse(v);
    }
  }
  if (td.dataset.sort !== undefined) return _sortParse(td.dataset.sort);
  return _sortParse(td.textContent);
}

/* A column sorts numerically only if EVERY non-missing leader cell parses
   as a number; one stray non-numeric cell (free text, a date string)
   degrades the WHOLE column to localeCompare rather than an order that
   silently mixes numbers and strings. Computed fresh per sort from the
   CURRENT (possibly filtered) rows, never cached across renders. */
function _sortColumnIsNumeric(blocks, colIndex, opts) {
  for (const b of blocks) {
    const v = _sortCellValue(b, colIndex, opts);
    if (!v.missing && v.num == null) return false;
  }
  return true;
}

function _sortComparator(colIndex, dir, opts, numeric) {
  return (a, b) => {
    const va = _sortCellValue(a, colIndex, opts);
    const vb = _sortCellValue(b, colIndex, opts);
    if (va.missing !== vb.missing) return va.missing ? 1 : -1;   // missing always last
    if (va.missing) return 0;
    if (numeric) return dir === 'desc' ? vb.num - va.num : va.num - vb.num;
    const cmp = va.text.localeCompare(vb.text);
    return dir === 'desc' ? -cmp : cmp;
  };
}

/* Split tbody rows into pinned rows (opts.pinned -- never sorted, always
   last, e.g. a totals row) and blocks (opts.detail -- a leader row plus
   every immediately-following detail row). BLOCKS, not individual rows,
   are the sort unit: the Tax open-lots table renders an expanded
   rollup's per-lot detail as tr.tax-det rows directly under their
   instrument, and a plain row-level sort would scatter those lots under
   whatever instrument happens to land above them. */
function _sortBlocks(tbody, opts) {
  const pinned = [];
  const blocks = [];
  let current = null;
  Array.from(tbody.children).forEach((tr) => {
    if (opts.pinned && tr.matches(opts.pinned)) {
      pinned.push(tr);
      current = null;
      return;
    }
    if (opts.detail && current && tr.matches(opts.detail)) {
      current.details.push(tr);
      return;
    }
    current = { leader: tr, details: [] };
    blocks.push(current);
  });
  return { pinned, blocks };
}

function _sortApplyOrder(tbody, pinned, orderedBlocks) {
  orderedBlocks.forEach((b) => {
    tbody.appendChild(b.leader);
    b.details.forEach((d) => tbody.appendChild(d));
  });
  pinned.forEach((tr) => tbody.appendChild(tr));
}

/* Make a rendered table's headers clickable to sort. Call AFTER thead and
   tbody are populated -- it reads the rows once at attach and reapplies
   any remembered sort immediately.
     opts.pinned  CSS selector, rows that never sort (default '.total-row')
     opts.detail  CSS selector, rows that travel with the row above
                  (default '.tax-det')
     opts.skip    column indexes that get no sort affordance at all --
                  their cells are controls or free-form evidence, not a
                  single comparable value
     opts.key     (td, colIndex) => value override, see _sortCellValue
   INVARIANT: call only on a freshly built tbody (table.innerHTML = '' then
   rebuilt -- what all five current call sites do). naturalBlocks below is
   captured from the DOM once, at attach; if a future table ever re-attaches
   onto rows left over from a PRIOR sort instead of a rebuild, that sorted
   order silently becomes the new "natural" order and the third click
   (restore natural) can never get back the real one. */
function makeSortable(tableEl, opts) {
  opts = Object.assign({ pinned: '.total-row', detail: '.tax-det',
                         skip: [], key: null }, opts || {});
  const theadRow = tableEl.querySelector('thead tr');
  const tbody = tableEl.querySelector('tbody');
  if (!theadRow || !tbody) return;
  const skipSet = new Set(opts.skip);
  const ths = Array.from(theadRow.children);
  // capture each header's plain label ONCE: survives THIS attach appending
  // an indicator glyph to th.textContent, and survives a future caller
  // reusing the same th node across renders instead of rebuilding it
  const labels = ths.map((th) => (th.dataset.sortLabel !== undefined
    ? th.dataset.sortLabel : th.textContent.trim()));
  ths.forEach((th, i) => { th.dataset.sortLabel = labels[i]; });

  const { pinned, blocks: naturalBlocks } = _sortBlocks(tbody, opts);
  if (naturalBlocks.length === 0) {
    // a filter emptied the table, or left only pinned rows (e.g. a totals
    // row with nothing to total) -- nothing to sort, so no header may
    // offer to: a clickable arrow over zero rows is the same lying
    // affordance as a sort control over a degrade callout (FIX 2).
    ths.forEach((th) => {
      th.classList.remove('sortable');
      th.removeAttribute('tabindex');
      th.removeAttribute('role');
      th.removeAttribute('aria-sort');
      th.style.cursor = '';
      th.onclick = null;
      th.onkeydown = null;
    });
    return;
  }
  // one dedicated indicator child per header, created once and reused for
  // every renderHeaders() call in this attach (a click cycle re-renders
  // headers up to 3 times) -- keeps the header's OWN content (label, and
  // in slice 2 a possible tooltip span or abbr) untouched forever after
  // attach, instead of th.textContent overwriting it wholesale (FIX 4).
  // Fresh th per render in practice (see the docblock invariant above),
  // so this never finds a leftover span to reuse today, but the lookup
  // keeps a future re-attach from stacking a second one.
  const indicators = ths.map((th) => {
    let span = th.querySelector('.sort-ind');
    if (!span) {
      span = el('span', { class: 'sort-ind' });
      th.appendChild(span);
    }
    return span;
  });
  const hasId = !!tableEl.id;
  const stateKey = hasId ? _sortKeyFor(tableEl, labels) : null;
  const localState = { col: null, dir: null };   // used when !hasId
  const getState = () => (hasId
    ? (_sortState.get(stateKey) || { col: null, dir: null }) : localState);
  const setState = (s) => {
    if (!hasId) { localState.col = s.col; localState.dir = s.dir; return; }
    if (s.col == null) _sortState.delete(stateKey);
    else _sortState.set(stateKey, s);
  };

  function renderHeaders() {
    const st = getState();
    ths.forEach((th, i) => {
      if (skipSet.has(i)) return;
      const active = i === st.col;
      indicators[i].textContent = active
        ? (st.dir === 'desc' ? ' ▼' : ' ▲') : '';
      th.setAttribute('aria-sort', active
        ? (st.dir === 'desc' ? 'descending' : 'ascending') : 'none');
    });
  }

  function applyState() {
    const st = getState();
    let ordered = naturalBlocks;
    if (st.col != null) {
      const numeric = _sortColumnIsNumeric(naturalBlocks, st.col, opts);
      ordered = naturalBlocks.slice()
        .sort(_sortComparator(st.col, st.dir, opts, numeric));
    }
    _sortApplyOrder(tbody, pinned, ordered);
    renderHeaders();
  }

  function activate(colIndex) {
    const cur = getState();
    const numeric = _sortColumnIsNumeric(naturalBlocks, colIndex, opts);
    let nextDir;
    if (cur.col !== colIndex) {
      // first click on THIS column: biggest number first (TK's stated
      // case -- deepest loss / biggest gain), text starts A-Z
      nextDir = numeric ? 'desc' : 'asc';
    } else {
      const seq = numeric ? ['desc', 'asc', null] : ['asc', 'desc', null];
      nextDir = seq[(seq.indexOf(cur.dir) + 1) % seq.length];
    }
    setState({ col: nextDir == null ? null : colIndex, dir: nextDir });
    applyState();
  }

  // Rebuild every header's affordance from scratch on each attach (render
  // functions re-run on every filter change). Assigning .onclick/
  // .onkeydown -- not addEventListener -- replaces any prior handler
  // instead of stacking a second one, so repeat attaches never double-fire.
  ths.forEach((th, i) => {
    if (skipSet.has(i)) {
      th.classList.remove('sortable');
      th.removeAttribute('tabindex');
      th.removeAttribute('role');
      th.removeAttribute('aria-sort');
      th.style.cursor = '';
      th.onclick = null;
      th.onkeydown = null;
      return;
    }
    // no role="button" here (FIX 6): it OVERRIDES a <th>'s implicit
    // columnheader role, and aria-sort is only defined on
    // columnheader/rowheader -- the pair would silently void the very
    // attribute it sits beside and drop the header-to-cell association
    // for every cell in the column. Leave the implicit role alone.
    th.classList.add('sortable');
    th.style.cursor = 'pointer';
    th.setAttribute('tabindex', '0');
    th.onclick = () => activate(i);
    th.onkeydown = (e) => {
      if (e.key === 'Enter' || e.key === ' ' || e.key === 'Spacebar') {
        e.preventDefault();
        activate(i);
      }
    };
  });

  applyState();   // re-apply a remembered sort immediately, or stay natural
}

/* "Jun 2026" -> a monotonic sort key (Python's strftime("%b %Y") -- the one
   non-ISO date format any terminal table renders; ISO ("2026-06-30") already
   sorts correctly as plain text, per the value-extraction contract above).
   Returns undefined (no opinion) for anything else, so a non-matching cell
   -- '—', or a table-specific sentinel word the caller checks itself --
   falls through to normal parsing. Shared by renderHealthTable's "Last
   verified" and renderRiskEpisodes' Peak/Trough/Recovery. */
const _MON3 = { Jan: 0, Feb: 1, Mar: 2, Apr: 3, May: 4, Jun: 5,
                Jul: 6, Aug: 7, Sep: 8, Oct: 9, Nov: 10, Dec: 11 };
function _monYearKey(text) {
  const m = /^([A-Za-z]{3})\s+(\d{4})$/.exec((text || '').trim());
  return (m && (m[1] in _MON3)) ? Number(m[2]) * 12 + _MON3[m[1]] : undefined;
}

/* dir ('up'|'down'|'flat') -> chip class + arrow glyph */
function chipClass(dir) {
  return dir === 'up' ? 'chip-up' : dir === 'down' ? 'chip-down' : 'chip-flat';
}
function chipArrow(dir) {
  return dir === 'up' ? '▲' : dir === 'down' ? '▼' : '';
}

/* Build a P&L chip pill. `small` is for the KPI-tape variant. `colorDir`
   overrides the COLOR only (the arrow still follows `dir`) — used to mirror the
   Streamlit "$100K vs SPY" card, which is always green with a directional arrow
   (app.py's delta_color="inverse" trick). Defaults to `dir`. */
function buildChip(text, dir, small, colorDir) {
  const cls = 'chip ' + chipClass(colorDir || dir) + (small ? ' chip-sm' : '');
  const kids = [];
  const arrow = chipArrow(dir);
  if (arrow) kids.push(el('span', { class: 'chip-arrow' }, arrow));
  kids.push(document.createTextNode(text));
  return el('span', { class: cls }, kids);
}

/* ============ CHROME (selects + synthetic badge + rail notice) ============ */
function renderChrome(meta) {
  // synthetic badge
  $('badge-synthetic').hidden = !meta.synthetic;

  // As-of select
  const asof = $('sel-asof');
  asof.innerHTML = '';
  (meta.available_dates || []).forEach((d) => {
    const o = el('option', { value: d }, d);
    if (d === meta.as_of) o.selected = true;
    asof.appendChild(o);
  });

  // History select (global filter, like broker) + clamp As-of to its cutoff.
  clampAsOfToHistory(populateHistorySelect(meta));

  // Account + Asset-class multi-select pills (seed selection from meta.filter).
  filterSel.account = (meta.filter.account === 'all' || !meta.filter.account)
    ? [] : [].concat(meta.filter.account);
  filterSel.asset_class = (meta.filter.asset_class === 'all' || !meta.filter.asset_class)
    ? [] : [].concat(meta.filter.asset_class);
  filterSel.broker = (meta.filter.broker === 'all' || !meta.filter.broker)
    ? [] : [].concat(meta.filter.broker);
  buildMultiSelect('ms-account', 'account', 'All accounts', meta.accounts || []);
  buildMultiSelect('ms-class', 'asset_class', 'All asset classes', meta.classes || []);
  const bopts = _brokerOpts(meta);
  buildMultiSelect('ms-broker', 'broker', _brokerAllLabel(bopts), bopts);
}

/* Display-layer capitalization derived by rule (prettyBroker); ids stay
   lowercase for the API. Already-cased server labels pass through unchanged. */
function _brokerOpts(meta) {
  return (meta.brokers || []).map((b) => ({ ...b, label: prettyBroker(b.label || b.id) }));
}

/* The broker pill's nothing-checked state is the server's REAL-broker default
   (test/demo brokers are opt-in, never rolled in), so label it with the real
   broker names — "All brokers" read as if the test rows below were included. */
function _brokerAllLabel(opts) {
  const real = opts.filter((o) => !o.test).map((o) => o.label);
  return real.length ? real.join(' + ') : 'All brokers';
}

/* Populate the History pill (single-select GLOBAL filter, like broker — NOT
   Holdings-only like As-of). meta.history_starts is always the FULL option
   list (the server never narrows it, so the picker can't lock itself out);
   meta.filter.history_start echoes the active selection. Called from
   renderChrome (Holdings) and ensureFilterSelects (every other tab) so the
   pill is populated regardless of which tab is landed on first. Returns the
   resolved current selection so callers can feed clampAsOfToHistory. */
let _histDefaultApplied = false;   // S0: apply the 2021+ first-load default once

function populateHistorySelect(meta) {
  const hist = $('sel-history');
  if (!hist) return 'all';
  const hopts = (meta.history_starts && meta.history_starts.length)
    ? meta.history_starts : [{ id: 'all', label: 'All history' }];
  let cur = (meta.filter && meta.filter.history_start) || 'all';
  /* S0 (TK 2026-08-07): first load defaults to 2021+ instead of full history —
     only over the untouched server default ('all'), and only when the option
     exists (short-history data falls back to All honestly). The deferred
     onHistoryChange re-runs the exact manual-selection path (As-of clamp +
     global refetch) after this render completes. Any explicit user choice
     afterwards — All history included — sticks for the session. */
  if (!_histDefaultApplied) {
    _histDefaultApplied = true;
    if (cur === 'all' && hopts.some((o) => o.id === '2021+')) {
      cur = '2021+';
      setTimeout(onHistoryChange, 0);
    }
  }
  hist.innerHTML = '';
  hopts.forEach((o) => {
    const opt = el('option', { value: o.id }, o.label);
    if (o.id === cur) opt.selected = true;
    hist.appendChild(opt);
  });
  return cur;
}

/* Hide sel-asof dates before Jan 1 of the History-start cutoff year (mirrors
   app.py's `_asof_cutoff`, app.py:1203-1227). The server's meta.available_dates
   is NOT pre-clamped by history_start (apply_global_filters narrows
   positions/twr_account/etc. but Frames.available_dates is a load-time-cached
   field it never touches — same field on every route), so this is the only
   place the As-of picker gets narrowed. Mirrors app.py's own fallback too: if
   the cutoff would empty the picker entirely, leave every option in place
   rather than strand it with nothing selectable. */
function clampAsOfToHistory(cur) {
  const asofSel = $('sel-asof');
  if (!asofSel || !cur || cur === 'all') return;
  const y = parseInt(cur, 10);
  if (isNaN(y)) return;
  const cutoff = new Date(y, 0, 1);
  const options = [...asofSel.options];
  const survivors = options.filter((o) => !o.value || new Date(o.value) >= cutoff);
  if (!survivors.length) return;
  options.forEach((o) => {
    if (o.value && new Date(o.value) < cutoff) o.remove();
  });
}

/* rail "carried forward" amber notice card (bottom of the rail) */
function renderRailNotice(carried) {
  const wrap = $('rail-notice');
  if (!carried || !carried.length) { wrap.hidden = true; return; }
  wrap.hidden = false;
  const n = carried.length;
  $('rail-notice-title').textContent =
    n + ' account' + (n === 1 ? '' : 's') + ' carried fwd';
  const body = $('rail-notice-body');
  body.innerHTML = '';
  carried.forEach((c, i) => {
    if (i) body.appendChild(document.createTextNode(' · '));
    body.appendChild(el('b', null, c.account));
    body.appendChild(document.createTextNode(' (' + c.as_of + ')'));
  });
  body.appendChild(document.createTextNode(' shown from last-known holdings.'));
}

/* ============ KPI TAPE ============ */
function renderTape(tape) {
  const root = $('tape');
  root.innerHTML = '';
  (tape || []).forEach((cell) => {
    const valCls = 'tape-value' + (cell.color ? ' ' + cell.color : '');
    const kids = [
      el('div', { class: 'tape-label' }, cell.label),
      el('div', { class: valCls }, cell.value),
    ];
    if (cell.chip) {
      kids.push(el('div', { class: 'tape-chip-row' }, [
        buildChip(cell.chip.text, cell.chip.dir, true),
        cell.sub ? el('span', { class: 'chip-after' }, cell.sub) : null,
      ]));
    } else if (cell.sub) {
      kids.push(el('div', { class: 'tape-sub' }, cell.sub));
    }
    root.appendChild(el('div', { class: 'tape-cell' }, kids));
  });
}

/* ============ HEALTH BANNER ============ */
function renderHealth(health) {
  const root = $('health');
  root.className = 'callout ' + (health.level === 'warning' ? 'callout-warn' : 'callout-health');
  const icon = health.level === 'warning' ? '⚠' : '✓';
  root.innerHTML = '';
  root.appendChild(el('span', { class: 'callout-icon' }, icon));
  // strip a leading glyph the engine may already include, to avoid doubling
  const text = String(health.text || '').replace(/^[✓⚠⚑⌖]\s*/, '');
  root.appendChild(el('span', { class: 'callout-text' }, text));
}

/* ============ SNAPSHOT (5 KPI cards) ============ */
/* A value-delta card (vs. prior period / YTD): signed % headline, ▲▼ chip with
   the $ delta, and the reference date. `d` is the server's
   {pct, dir, abs, prior_label} block — same shape for both cards. */
function deltaCard(label, d) {
  const v = d || {};
  const dir = v.dir || 'flat';
  const valCls = 'kpi-value ' + (dir === 'up' ? 'gain' : dir === 'down' ? 'loss' : '');
  return el('div', { class: 'kpi' }, [
    el('div', { class: 'kpi-label' }, label),
    el('div', { class: valCls }, (dir === 'up' ? '+' : '') + (v.pct || '')),
    el('div', { class: 'kpi-chip-row' }, [
      buildChip(v.abs || '', dir, false),
      v.prior_label ? el('span', { class: 'chip-after' }, v.prior_label) : null,
    ]),
  ]);
}

function renderSnapshot(snap, asLabel) {
  $('snapshot-note').textContent = 'As of ' + asLabel + ' · marked to live prices';

  const grid = el('div', { class: 'snapshot-grid' });

  // card 1: portfolio value + sparkline
  const spark = svgEl('svg', {
    viewBox: '0 0 120 36', width: '120', height: '36',
    preserveAspectRatio: 'none', class: 'kpi-spark',
  });
  spark.appendChild(svgEl('path', {
    d: 'M0,30 L18,26 L34,28 L52,18 L70,22 L88,12 L106,9 L120,4',
    fill: 'none', stroke: '#4DA3F5', 'stroke-width': '1.6',
  }));
  grid.appendChild(el('div', { class: 'kpi' }, [
    el('div', { class: 'kpi-label' }, 'Portfolio value'),
    el('div', { class: 'kpi-value' }, snap.portfolio_value),
    // "of $X unfiltered · ..." under an account/class filter (Streamlit parity)
    el('div', { class: 'kpi-sub' }, snap.portfolio_value_sub || 'marked to live prices'),
    spark,
  ]));

  // card 2: vs prior period; card 3: YTD (vs the last prior-year snapshot).
  // Both are VALUE deltas in the same filter scope as card 1.
  grid.appendChild(deltaCard('vs. prior period', snap.vs_prior));
  grid.appendChild(deltaCard('YTD', snap.ytd));

  // card 4: accounts
  const ac = snap.accounts || {};
  grid.appendChild(el('div', { class: 'kpi' }, [
    el('div', { class: 'kpi-label' }, 'Accounts'),
    el('div', { class: 'kpi-value' }, String(ac.value)),
    el('div', { class: 'kpi-sub' }, ac.sub || ''),
  ]));

  // card 5: holdings (symbols / rows)
  const h = snap.holdings || {};
  grid.appendChild(el('div', { class: 'kpi' }, [
    el('div', { class: 'kpi-label' }, 'Holdings'),
    el('div', { class: 'kpi-value' }, [
      String(h.symbols),
      el('span', { class: 'kpi-value-sub' }, ' / ' + h.rows + ' rows'),
    ]),
    el('div', { class: 'kpi-sub' }, 'distinct symbols'),
  ]));

  const root = $('snapshot');
  root.innerHTML = '';
  root.appendChild(grid);
}

/* carried-forward blue callout under the snapshot */
function renderCarriedCallout(carried) {
  const root = $('carried');
  if (!carried || !carried.length) { root.hidden = true; return; }
  root.hidden = false;
  root.className = 'callout callout-blue';
  root.innerHTML = '';
  root.appendChild(el('span', { class: 'callout-icon' }, '⌖'));
  const txt = el('span', { class: 'callout-text' });
  txt.appendChild(document.createTextNode(
    'Carried forward (last-known holdings; no fresh statement this period): '));
  carried.forEach((c, i) => {
    if (i) txt.appendChild(document.createTextNode(' · '));
    txt.appendChild(el('b', null, c.account));
    txt.appendChild(document.createTextNode(' — ' + c.as_of));
  });
  txt.appendChild(document.createTextNode('.'));
  root.appendChild(txt);
}

/* ============ ALLOCATION BY CLASS (donut + legend) ============ */
function renderAllocClass(a) {
  const card = $('alloc-class');
  card.innerHTML = '';
  card.appendChild(el('div', { class: 'card-head' }, [
    el('div', { class: 'card-head-title' }, 'Allocation by asset class'),
    el('div', { class: 'card-head-note' }, (a.total_label || '') + ' · ' + a.n + ' classes'),
  ]));

  // donut: r=46, circumference = 2*pi*46 ≈ 288.83
  const R = 46;
  const C = 2 * Math.PI * R;
  const svg = svgEl('svg', { viewBox: '0 0 120 120', width: '150', height: '150', class: 'donut' });
  // track
  svg.appendChild(svgEl('circle', {
    cx: 60, cy: 60, r: R, fill: 'none', stroke: '#232E3C', 'stroke-width': 15,
  }));
  // segments. Non-positive slices (a short-options class with negative MV)
  // MUST be skipped: a negative dash length is invalid SVG, the browser
  // ignores the whole dasharray, and that circle paints a FULL ring over
  // every other segment — the "donut is one color" bug. They stay in the
  // legend (Plotly's pie drops them from the ring the same way).
  let offset = 0; // accumulated arc length already consumed
  (a.slices || []).forEach((s) => {
    const len = (s.pct / 100) * C;
    if (!(len > 0.05)) return;
    svg.appendChild(svgEl('circle', {
      cx: 60, cy: 60, r: R, fill: 'none', stroke: s.color, 'stroke-width': 15,
      'stroke-dasharray': len.toFixed(1) + ' ' + (C - len).toFixed(1),
      'stroke-dashoffset': offset === 0 ? null : (-offset).toFixed(1),
      transform: 'rotate(-90 60 60)',
    }));
    offset += len;
  });
  // centre labels
  svg.appendChild(svgEl('text', {
    x: 60, y: 57, 'text-anchor': 'middle', fill: '#EAF0F8', 'font-size': 14,
    'font-family': "'IBM Plex Mono',monospace", 'font-weight': 600,
  })).textContent = a.total_label || '';
  const sub = svgEl('text', {
    x: 60, y: 70, 'text-anchor': 'middle', fill: '#6B7786', 'font-size': 7,
    'font-family': "'IBM Plex Mono',monospace", 'letter-spacing': 1,
  });
  sub.textContent = a.n + ' CLASSES';
  svg.appendChild(sub);

  const legend = el('div', { class: 'legend' });
  (a.slices || []).forEach((s) => {
    legend.appendChild(el('div', { class: 'legend-row' }, [
      el('span', { class: 'legend-name' }, [
        el('span', { class: 'legend-swatch', style: 'background:' + s.color }),
        s.label,
      ]),
      el('span', { class: 'legend-pct' }, Math.round(s.pct) + '%'),
      el('span', { class: 'legend-val' }, s.value),
    ]));
  });

  card.appendChild(el('div', { class: 'donut-row' }, [svg, legend]));
}

/* ============ ALLOCATION BY ACCOUNT (bar rows) ============ */
function renderAllocAccount(a) {
  const card = $('alloc-account');
  card.innerHTML = '';
  card.appendChild(el('div', { class: 'card-head' }, [
    el('div', { class: 'card-head-title' }, 'Allocation by account'),
    el('div', { class: 'card-head-note' }, a.n + ' accounts'),
  ]));

  const brokers = new Map(); // broker -> color (for the footer legend)
  (a.rows || []).forEach((r) => {
    if (!brokers.has(r.broker)) brokers.set(r.broker, r.color);
    card.appendChild(el('div', { class: 'bar-row' }, [
      el('span', { class: 'bar-name' }, r.label),
      el('span', { class: 'bar' }, [
        el('span', { class: 'bar-fill', style: 'width:' + r.bar + '%;background:' + r.color }),
      ]),
      el('span', { class: 'bar-pct' }, r.pct.toFixed(1) + '%'),
    ]));
  });

  const foot = el('div', { class: 'legend-foot' });
  for (const [broker, color] of brokers) {
    foot.appendChild(el('span', { class: 'legend-foot-item' }, [
      el('span', { class: 'legend-swatch', style: 'background:' + color }),
      prettyBroker(broker),
    ]));
  }
  card.appendChild(foot);
}
function prettyBroker(b) {
  /* Rule-based (mirrors the service's _broker_display_label): a short
     all-lower id (<= 3 chars) reads as an initialism; anything else gets
     its first letter upper-cased. No broker name is ever hardcoded. */
  if (!b) return '';
  if (/^[a-z]{1,3}$/.test(b)) return b.toUpperCase();
  return b.charAt(0).toUpperCase() + b.slice(1);
}

/* ============ TOP HOLDINGS ============ */
function renderTop(t) {
  const card = $('top');
  card.innerHTML = '';
  card.appendChild(el('div', { class: 'top-head' }, [
    el('div', { class: 'top-head-title' }, 'Top holdings by weight'),
    (() => {
      // real control (was display-only text): refetch Holdings with top_n
      const sel = el('select', { class: 'topn-select' });
      [5, 10, 15, 20, 25, 30].forEach((n) => {
        const o = el('option', { value: String(n) }, 'Show top ' + n);
        if (n === t.top_n) o.selected = true;
        sel.appendChild(o);
      });
      sel.addEventListener('change', () => { _topN = +sel.value; fetchTab('holdings'); });
      return el('div', { class: 'top-topn' }, [sel]);
    })(),
  ]));

  const grid = el('div', { class: 'top-grid' });
  // Column-major fill (TK): read DOWN column 1 (#1, #2, ...) then column 2 —
  // the auto-fit row-major default put #1 and #2 side by side.
  const nRows = (t.rows || []).length;
  grid.style.gridAutoFlow = 'column';
  grid.style.gridTemplateColumns = 'repeat(2, minmax(280px, 1fr))';
  grid.style.gridTemplateRows = 'repeat(' + Math.max(1, Math.ceil(nRows / 2)) + ', auto)';
  (t.rows || []).forEach((r) => {
    grid.appendChild(el('div', { class: 'top-row' }, [
      el('span', { class: 'top-sym' }, r.symbol),
      el('span', { class: 'bar' }, [
        el('span', { class: 'bar-fill', style: 'width:' + r.bar + '%;background:' + r.color }),
      ]),
      el('span', { class: 'top-pct' }, r.pct.toFixed(1) + '%'),
    ]));
  });
  card.appendChild(grid);
}

/* ============ TOP-LEVEL RENDER + DATA FLOW ============ */
function renderHoldings(data) {
  renderChrome(data.meta);
  renderRailNotice(data.carried_forward);
  renderTape(data.tape);
  renderHealth(data.health);
  renderSnapshot(data.snapshot, data.meta.as_of_label);
  renderCarriedCallout(data.carried_forward);
  renderAllocClass(data.alloc_class);
  renderAllocAccount(data.alloc_account);
  renderTop(data.top_holdings);
  // All-holdings table removed per TK (QA-polish S8) — data.positions goes
  // unrendered; Streamlit + the CSVs keep the full table.
}

/* ===== multi-select filter pills (Account / Asset-class) ===== */
// Selected ids per filter; [] means "all". Read by currentQuery/fetchTab.
const filterSel = { account: [], asset_class: [], broker: [] };

function _msSummary(allLabel, options, selected) {
  if (!selected.length) return allLabel;
  if (selected.length === 1) {
    const o = options.find((x) => x.id === selected[0]);
    return o ? o.label : selected[0];
  }
  return selected.length + ' selected';
}

/* A 422 whose detail names a filter we hold a selection for ("unknown
   account" / "unknown asset_class" — an id outside the current broker /
   as-of scope): drop that selection back to "all" (pill included) and return
   the notice to show; null when the 422 is about something else. */
function _resetUnknownFilter(detail) {
  const m = /^unknown (account|asset_class)$/.exec(detail || '');
  if (!m || !filterSel[m[1]].length) return null;
  const key = m[1];
  filterSel[key] = [];
  const host = $(key === 'account' ? 'ms-account' : 'ms-class');
  const allCb = host && host.querySelector('.ms-opt-all input');
  // Re-checking "all" runs the pill's own change handler (clears the boxes,
  // rewrites the summary) without the close-popover refetch.
  if (allCb) { allCb.checked = true; allCb.dispatchEvent(new Event('change')); }
  return (key === 'account' ? 'Account' : 'Asset-class')
    + ' filter reset to all — the previous selection is not part of the current broker / as-of scope.';
}

/* Build (or rebuild) a checklist-dropdown pill into host `#hostId`.
   options: [{id,label}]. key: 'account'|'asset_class'. Refetches the active
   tab when the popover closes IF the selection changed. */
function buildMultiSelect(hostId, key, allLabel, options) {
  const host = $(hostId);
  if (host._msOnDoc) { document.removeEventListener('click', host._msOnDoc); host._msOnDoc = null; }
  host.innerHTML = '';
  const selected = filterSel[key];
  const summary = el('span', { class: 'pill-summary' }, _msSummary(allLabel, options, selected));
  const caret = el('span', { class: 'pill-caret' }, '▾');
  const pop = el('div', { class: 'ms-popover', hidden: true });

  const allRow = el('label', { class: 'ms-opt ms-opt-all' });
  const allCb = el('input', { type: 'checkbox' });
  allCb.checked = selected.length === 0;
  allRow.appendChild(allCb);
  allRow.appendChild(el('span', {}, allLabel));
  pop.appendChild(allRow);

  const boxes = [];
  options.forEach((o) => {
    const row = el('label', { class: 'ms-opt' });
    const cb = el('input', { type: 'checkbox' });
    cb.checked = selected.includes(o.id);
    cb.dataset.id = o.id;
    boxes.push(cb);
    row.appendChild(cb);
    row.appendChild(el('span', {}, o.label));
    pop.appendChild(row);
    cb.addEventListener('change', () => {
      if (cb.checked) allCb.checked = false;
      syncFromBoxes();
    });
  });

  allCb.addEventListener('change', () => {
    if (allCb.checked) { boxes.forEach((b) => { b.checked = false; }); }
    syncFromBoxes();
  });

  function syncFromBoxes() {
    const picked = boxes.filter((b) => b.checked).map((b) => b.dataset.id);
    filterSel[key] = picked;                 // [] => all
    if (!picked.length) allCb.checked = true;
    summary.textContent = _msSummary(allLabel, options, picked);
  }

  let openSel = selected.slice();
  function openPop() {
    openSel = filterSel[key].slice();
    pop.hidden = false;
    setTimeout(() => document.addEventListener('click', onDoc), 0);
    host._msOnDoc = onDoc;
  }
  function closePop() {
    pop.hidden = true;
    document.removeEventListener('click', onDoc);
    host._msOnDoc = null;
    const now = filterSel[key];
    const changed = now.length !== openSel.length
      || now.some((x) => !openSel.includes(x));
    // broker feeds the global chrome (KPI tape, status, footer) too, so it
    // takes the full onFilterChange; account/class are tab-only filters.
    if (changed) { if (key === 'broker') onFilterChange(); else fetchTab(activeTab); }
  }
  function onDoc(e) { if (!host.contains(e.target)) closePop(); }

  host.appendChild(summary);
  host.appendChild(caret);
  host.appendChild(pop);
  host.onclick = (e) => {
    if (host.contains(e.target) && pop.hidden) { openPop(); }
  };
}

/* Current History-start selection ('all' or 'YYYY+'), read straight off the
   pill's <select> — a plain single-select needs no filterSel-style shadow
   state; the DOM already holds the truth (mirrors how sel-asof's own value
   is read directly rather than mirrored into filterSel). */
function currentHistoryStart() {
  const sel = $('sel-history');
  return (sel && sel.value) || 'all';
}

let _topN = null;   // Holdings "Show top N" (server default 15 when unset)
const _benchState = { bench: 'auto' };   // Auto until the user explicitly picks

function currentQuery() {
  const params = new URLSearchParams();
  const asof = $('sel-asof').value;
  if (asof) params.set('as_of', asof);
  filterSel.account.forEach((id) => params.append('account', id));
  filterSel.asset_class.forEach((id) => params.append('asset_class', id));
  filterSel.broker.forEach((id) => params.append('broker', id));
  const hist = currentHistoryStart();
  if (hist !== 'all') params.set('history_start', hist);
  if (_topN) params.set('top_n', _topN);
  return params;
}

/* ============ AI NARRATION BOX (AI S1) ============
   Server-generated model text renders via textContent ONLY — it is not
   trusted markup and must never reach el(..., {html:...}). Off-state
   (enabled:false — no key/package server-side) removes the panel; transport
   errors show a quiet unavailable line instead. */
const _aiBoxes = {};   // section -> box element; reloadAiBox() re-fetches it

/* S1-v2: highlight numbers (percent/pp/bps family + bare decimals) and
   ticker-shaped uppercase tokens inside MODEL text. Builds DOM nodes only —
   model text stays textContent-safe, never markup. */
const _AI_ACRO = new Set(['TWR', 'IRR', 'VAR', 'CVAR', 'ES', 'DD', 'YTD',
                          'EWMA', 'CAPM', 'ERP', 'AI', 'FACTS', 'JSON',
                          'IRA', 'N']);
const _AI_HL_RE = /([+-]?\d[\d.,]*\s?(?:%|pp|bps)|\b\d+\.\d+\b|\b[A-Z][A-Z0-9.]{1,6}\b)/g;
function aiRichText(s) {
  const frag = document.createDocumentFragment();
  const str = String(s == null ? '' : s);
  let last = 0, m;
  _AI_HL_RE.lastIndex = 0;
  while ((m = _AI_HL_RE.exec(str)) !== null) {
    const tok = m[0];
    const isWord = /^[A-Z]/.test(tok);
    if (isWord && _AI_ACRO.has(tok)) continue;   // acronym — stays plain
    if (m.index > last) frag.appendChild(document.createTextNode(str.slice(last, m.index)));
    frag.appendChild(el('span', { class: isWord ? 'ai-hl-tk' : 'ai-hl-num',
                                  text: tok }));
    last = m.index + tok.length;
  }
  frag.appendChild(document.createTextNode(str.slice(last)));
  return frag;
}

function aiBulletList(items) {
  const ul = el('ul', { class: 'ai-box-bullets' });
  (items || []).forEach((b) => {
    const li = document.createElement('li');
    li.appendChild(aiRichText(b));
    ul.appendChild(li);
  });
  return ul;
}

function aiStructuredNodes(sd) {
  const nodes = [];
  const h = el('div', { class: 'ai-box-headline' });
  h.appendChild(aiRichText(sd.headline));
  nodes.push(h, aiBulletList(sd.bullets));
  if (sd.caveat) {
    const c = el('div', { class: 'ai-box-caveat' });
    c.appendChild(aiRichText(sd.caveat));
    nodes.push(c);
  }
  if (sd.watch && sd.watch.length) {
    nodes.push(el('div', { class: 'ai-box-watch-label' }, 'Watch'));
    nodes.push(aiBulletList(sd.watch));
  }
  return nodes;
}

/* B1a: poll an AI narration endpoint until it stops reporting kind:"generating".
   doFetch() -> Promise<data>; isCurrent() guards against stale scope/tab flips
   (the caller's gen counter); onData(data) renders a terminal state; onFail()
   handles transport errors. pollMs cadence (2.5s narration / 1s chat), 180s ceiling. */
const AI_POLL_MS = 2500, AI_POLL_MAX_MS = 180000;  // Fable 5 turns run longer
const AI_CHAT_POLL_MS = 1000;    // chat turns are short (effort=medium): poll tighter
function pollAiNarrative(doFetch, isCurrent, onData, onFail, pollMs = AI_POLL_MS) {
  const started = Date.now();
  const tick = () => {
    doFetch()
      .then((d) => {
        if (!isCurrent()) return;                 // superseded — drop silently
        if (d && d.kind === 'generating') {
          if (Date.now() - started > AI_POLL_MAX_MS) {
            onData({ kind: 'error', text: null,
                     error: 'taking longer than usual — try Regenerate' });
            return;
          }
          setTimeout(tick, pollMs);
          return;
        }
        onData(d);
      })
      .catch(() => { if (isCurrent()) onFail(); });
  };
  tick();
}

function aiBoxEl(section, getDims) {
  const withDims = (target) => {
    const d = getDims ? getDims() : null;
    if (d) Object.keys(d).forEach((k) => {
      if (d[k] != null) {
        if (target instanceof URLSearchParams) target.set(k, String(d[k]));
        else target[k] = String(d[k]);
      }
    });
    return target;
  };
  let gen = 0;   // request generation — a stale response must never paint (S3 review)
  const stamp = el('span', { class: 'section-note ai-box-stamp' });
  const qline = el('div', { class: 'ai-box-q' });
  qline.style.display = 'none';
  const body = el('div', { class: 'ai-box-body', text: 'Generating analysis…' });
  const btn = el('button', { class: 'act-inline act-btn', type: 'button' },
                 '⟳ Regenerate');
  const box = el('div', { class: 'ai-box' }, [
    el('div', { class: 'ai-box-head' }, [
      el('span', { class: 'ai-box-glyph' }, '✦'),
      el('span', { class: 'ai-box-title' }, 'AI Analysis'),
      stamp]),
    qline, body, el('div', { class: 'ai-box-foot' }, btn)]);
  const render = (d) => {
    if (!d || d.enabled === false) { box.remove(); return; }
    if (d.question) { qline.textContent = d.question; qline.style.display = ''; }
    if (d.structured) {
      body.textContent = '';
      aiStructuredNodes(d.structured).forEach((n) => body.appendChild(n));
    } else {
      body.textContent = d.text ||
        (d.error ? 'Narration unavailable: ' + d.error : '—');
    }
    const bits = [];
    if (d.generated_at) bits.push('as of ' + d.generated_at);
    if (d.model) bits.push(d.model);
    if (d.stale) bits.push('STALE');
    else if (d.cached) bits.push('cached');
    if (d.kind === 'error' && d.error && d.text)
      bits.push('regenerate failed — showing last text');
    stamp.textContent = bits.join(' · ');
  };
  const fail = () => { body.textContent = 'Narration unavailable.'; };
  const load = () => {
    const g = ++gen;
    pollAiNarrative(
      () => {
        const q = withDims(currentQuery()); q.set('section', section);
        return fetch('/api/ai/explain?' + q).then(
          (r) => (r.ok ? r.json() : Promise.reject(new Error(String(r.status)))));
      },
      () => g === gen, render, fail);
  };
  btn.addEventListener('click', () => {
    const g = ++gen;
    body.textContent = 'Regenerating…'; stamp.textContent = '';
    const q = currentQuery();
    const brokers = q.getAll('broker');
    const accounts = q.getAll('account');
    const classes = q.getAll('asset_class');
    fetch('/api/ai/regenerate', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(withDims({
        section,
        broker: brokers.length ? brokers : ['all'],
        history_start: q.get('history_start') || 'all',
        account: accounts.length ? accounts : ['all'],
        asset_class: classes.length ? classes : ['all'],
      })),
    }).then((r) => (r.ok
        ? pollAiNarrative(
            () => {
              const q2 = withDims(currentQuery()); q2.set('section', section);
              return fetch('/api/ai/explain?' + q2).then(
                (r2) => (r2.ok ? r2.json() : Promise.reject(new Error(String(r2.status)))));
            },
            () => g === gen, render, fail)
        : Promise.reject(new Error(String(r.status)))))
      .catch(() => { if (g === gen) fail(); });
  });
  box.reload = load;   // seg clicks re-fetch just the box (S3)
  load();
  return box;
}

function mountAiBox(containerId, section, getDims) {
  const host = $(containerId);
  if (!host) return;
  host.innerHTML = '';                    // idempotent across re-renders
  const box = aiBoxEl(section, getDims);
  _aiBoxes[section] = box;
  host.appendChild(box);
}

function reloadAiBox(section) {
  const box = _aiBoxes[section];
  if (box && box.isConnected) box.reload();   // removed boxes (off-state) stay quiet
}

/* ---- AI ANALYSIS tab (S2). Narrative fields render via textContent ONLY
   (model output is not trusted markup); facts tables come server-formatted
   in the renderFactorTable spec shape. */
const _AI_NAR_FIELDS = [['verdict', 'Verdict'], ['why', 'Why'],
                        ['changes', 'What changed'], ['watch', 'Watch']];

/* B1b: render the AI-Analysis 4-field narrative (or a placeholder) into `fields`,
   and set the stamp. Shared by the initial render, the poll completion, and the
   Regenerate completion. `d` is the /api/ai/portfolio payload OR an
   /api/ai/explain poll payload — both carry narrative + generated_at/model/
   stale/cached (portfolio-GET nests those under narrative_meta; the explain
   poll puts them top-level). */
let _aiNarGen = 0;   // staleness counter — a new renderAiAnalysis/regenerate cancels an in-flight poll
function _aiNarStamp(stampEl, m) {
  const bits = [];
  if (m.generated_at) bits.push('as of ' + m.generated_at);
  if (m.model) bits.push(m.model);
  if (m.stale) bits.push('STALE'); else if (m.cached) bits.push('cached');
  stampEl.textContent = bits.join(' · ');
}
function _aiNarFields(fields, narrative, kind, error, questions) {
  fields.innerHTML = '';
  if (narrative) {
    _AI_NAR_FIELDS.forEach(([key, label]) => {
      const wrap = el('div', { class: 'ai-nar-field' }, [
        el('div', { class: 'ai-nar-label' },
           (questions && questions[key]) || label)]);
      const v = narrative[key];
      if (Array.isArray(v)) {
        wrap.appendChild(aiBulletList(v));
      } else {
        const t = el('div', { class: 'ai-nar-text' });
        t.appendChild(aiRichText(v || '—'));
        wrap.appendChild(t);
      }
      fields.appendChild(wrap);
    });
  } else {
    fields.appendChild(el('div', { class: 'ai-nar-text' },
      kind === 'error' ? 'Narration unavailable: ' + (error || 'generation failed')
                       : 'Generating analysis…'));
  }
}
function _aiNarPollFetch(bm) {   // the LIGHT poll: /api/ai/explain?section=portfolio
  const src = currentQuery();
  const q = new URLSearchParams();
  src.getAll('broker').forEach((b) => q.append('broker', b));
  if (src.get('history_start')) q.set('history_start', src.get('history_start'));
  q.set('benchmark', bm.id);
  q.set('section', 'portfolio');
  return fetch('/api/ai/explain?' + q).then(
    (r) => (r.ok ? r.json() : Promise.reject(new Error(String(r.status)))));
}

function renderAiAnalysis(data) {
  ensureFilterSelects(data.meta);
  const bm = (data.meta && data.meta.benchmark) ||
             { id: 'spy', short: 'SPY', label: 'SPY (S&P 500 TR)' };
  const qs = data.questions || null;
  const bsel = $('ai-benchmark-select');
  if (bsel && !bsel._wired) {
    bsel.addEventListener('change', function () {
      _benchState.bench = bsel.value;
      fetchTab('ai', currentQuery());
    });
    bsel._wired = true;
  }
  if (bsel) bsel.value = _benchState.bench;
  const wtitle = $('ai-window-title');
  if (wtitle) wtitle.textContent = 'Portfolio vs ' + bm.short;
  const bcap = $('ai-benchmark-cap');
  if (bcap) {
    if (_benchState.bench === 'auto' && bm.id === '60_40') {
      bcap.textContent = 'Auto → 60/40 · this scope is majority fixed income';
    } else if (bm.unavailable_fallback) {
      bcap.textContent = '60/40 unavailable (AGG data not loaded) — showing SPY';
    } else {
      bcap.textContent = bm.label;
    }
  }
  const empty = $('ai-empty');
  const host = $('ai-narrative');
  host.innerHTML = '';
  if (!data.facts || data.facts.available === false) {
    empty.hidden = false;
    empty.textContent = 'Not enough history to analyze yet — ingest statements first.';
    $('ai-window-table').innerHTML = '';
    $('ai-tiles').innerHTML = '';
    return;
  }
  empty.hidden = true;

  renderFactorTable($('ai-window-table'), data.display.window_table);
  const tiles = $('ai-tiles');
  tiles.innerHTML = '';
  const grid = el('div', { class: 'snapshot-grid' });
  // Redesign v3.1: "Top risk" closes the posture row (display order only —
  // the payload's order, and the portfolio golden, stay untouched).
  const tilesIn = data.display.tiles || [];
  const tileList = tilesIn.filter((t) => t.label !== 'Top risk')
    .concat(tilesIn.filter((t) => t.label === 'Top risk'));
  tileList.forEach((t) => {
    grid.appendChild(el('div', { class: 'kpi' }, [
      el('div', { class: 'kpi-label' }, t.label),
      el('div', { class: 'kpi-value' }, t.value)]));
  });
  tiles.appendChild(grid);

  if (data.enabled === false) {          // facts-only mode, no panel
    const bh = $('brief-ai');
    if (bh) bh.innerHTML = '';           // drop a previously-mounted brief box
    const w = $('ai-chat-wrap');
    if (w) w.hidden = true;              // hide chat if a prior render showed it
    return;
  }

  mountAiBox('brief-ai', 'brief');

  const stamp = el('span', { class: 'section-note ai-box-stamp' });
  const btn = el('button', { class: 'act-inline act-btn', type: 'button' },
                 '⟳ Regenerate');
  const fields = el('div', { class: 'ai-nar-grid' });
  const box = el('div', { class: 'ai-box' }, [
    el('div', { class: 'ai-box-head' }, [
      el('span', { class: 'ai-box-glyph' }, '✦'),
      el('span', { class: 'ai-box-title' }, 'AI Analysis'),
      stamp]),
    fields, el('div', { class: 'ai-box-foot' }, btn)]);
  const g = ++_aiNarGen;
  _aiNarStamp(stamp, data.narrative_meta || {});
  _aiNarFields(fields, data.narrative, data.kind, data.error, qs);

  const startPoll = () => pollAiNarrative(
    () => _aiNarPollFetch(bm),
    () => g === _aiNarGen,
    (d) => { _aiNarStamp(stamp, d); _aiNarFields(fields, d.narrative, d.kind, d.error, qs); },
    () => _aiNarFields(fields, null, 'error', 'unavailable', qs));

  if (!data.narrative && data.kind === 'generating') startPoll();   // cold GET -> poll to fill

  btn.addEventListener('click', () => {
    const gg = ++_aiNarGen;                     // supersede any in-flight poll
    fields.innerHTML = '';
    fields.appendChild(el('div', { class: 'ai-nar-text' }, 'Regenerating…'));
    stamp.textContent = '';
    const q = currentQuery();
    const brokers = q.getAll('broker');
    fetch('/api/ai/regenerate', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ section: 'portfolio',
        broker: brokers.length ? brokers : ['all'],
        history_start: q.get('history_start') || 'all',
        benchmark: bm.id }),
    }).then((r) => (r.ok
        ? pollAiNarrative(() => _aiNarPollFetch(bm), () => gg === _aiNarGen,
            (d) => { _aiNarStamp(stamp, d); _aiNarFields(fields, d.narrative, d.kind, d.error, qs); },
            () => _aiNarFields(fields, null, 'error', 'unavailable', qs))
        : Promise.reject(new Error(String(r.status)))))
      .catch(() => { if (gg === _aiNarGen) _aiNarFields(fields, null, 'error', 'regenerate failed', qs); });
  });
  host.appendChild(box);

  const chatWrap = $('ai-chat-wrap');
  if (chatWrap) {
    chatWrap.hidden = false;
    _aiChatWarm();     // pre-build this scope's pack while the reader is on the brief
    const form = $('ai-chat-form');
    if (form && !form._wired) {
      form.addEventListener('submit', (e) => { e.preventDefault(); _aiChatSend(); });
      form._wired = true;
    }
    // Redesign v3 (REDESIGN_NOTES §6.2): suggested-question chips — a click
    // asks that question outright; the row hides after the first turn.
    const sug = $('ai-chat-suggest');
    if (sug && !sug._wired) {
      sug.querySelectorAll('button').forEach((b) => b.addEventListener('click', () => {
        const inp = $('ai-chat-input');
        if (!inp || _aiChat.busy) return;
        inp.value = b.textContent.trim();
        _aiChatSend();
      }));
      sug._wired = true;
    }
    const copyBtn = $('ai-chat-copy');
    if (copyBtn && !copyBtn._wired) {
      copyBtn.addEventListener('click', _aiChatCopy);
      $('ai-chat-clear').addEventListener('click', _aiChatClear);
      copyBtn._wired = true;
    }
    _aiChatRender();   // re-paint existing history on tab re-entry
  }
}

/* ---- AI-tab chat (v2 S2). History lives in JS memory only (survives tab
   switches, clears on reload); model text renders via aiRichText — DOM
   text nodes only, never markup. The server pops each answer on read, so
   the FE history is the ONLY record of the conversation. */
const _aiChat = { messages: [], busy: false, error: null, gen: 0,
                  clearArmed: false, clearTimer: null };   // two-click Clear (2026-08-22)

function _aiChatBubble(text, cls) {
  const b = el('div', { class: 'ai-chat-msg ' + cls });
  b.appendChild(aiRichText(text));
  return b;
}

function _aiChatRender() {
  const log = $('ai-chat-log');
  if (!log) return;
  log.innerHTML = '';
  _aiChat.messages.forEach((m) => log.appendChild(
    _aiChatBubble(m.content, m.role === 'user' ? 'ai-chat-user' : 'ai-chat-ai')));
  if (_aiChat.busy)
    log.appendChild(_aiChatBubble('Analyzing…', 'ai-chat-ai ai-chat-busy'));
  if (_aiChat.error)
    log.appendChild(_aiChatBubble('Unavailable: ' + _aiChat.error,
                                  'ai-chat-ai ai-chat-err'));
  const btn = $('ai-chat-send');
  if (btn) btn.disabled = _aiChat.busy;
  const sug = $('ai-chat-suggest');
  if (sug) sug.hidden = _aiChat.messages.length > 0;
  const copyBtn = $('ai-chat-copy'), clearBtn = $('ai-chat-clear');
  if (copyBtn) copyBtn.disabled = _aiChat.messages.length === 0;
  if (clearBtn) {                          // never clear mid-turn (no leaked job)
    clearBtn.disabled = _aiChat.busy || _aiChat.messages.length === 0;
    clearBtn.textContent = _aiChat.clearArmed ? 'Sure?' : 'Clear';
    clearBtn.classList.toggle('armed', _aiChat.clearArmed);
  }
  log.scrollTop = log.scrollHeight;
}

/* Dock header tools (TK feedback 2026-08-22). Copy = plain-text transcript
   on the clipboard; Clear = two-click confirm (history is memory-only). */
function _aiChatScope() {
  // The ONE derivation of the chat scope: warm, send and the transcript
  // header must agree byte-for-byte or the warm builds a pack the send
  // never hits (server memo key = broker list + history_start).
  const q = currentQuery();
  const brokers = q.getAll('broker');
  return { broker: brokers.length ? brokers : ['all'],
           history_start: q.get('history_start') || 'all' };
}

function _aiChatScopeLine() {
  const s = _aiChatScope();
  return 'brokers: ' + s.broker.join(', ') + ' · history: ' + s.history_start;
}

function _aiChatTranscript() {
  const d = new Date();
  const ymd = d.getFullYear() + '-' + String(d.getMonth() + 1).padStart(2, '0')
            + '-' + String(d.getDate()).padStart(2, '0');
  const head = 'MERIDIAN · Ask AI Analyst · ' + ymd + ' · ' + _aiChatScopeLine();
  const turns = _aiChat.messages.map((m) =>
    (m.role === 'user' ? 'You: ' : 'Analyst: ') + m.content);
  return [head].concat(turns).join('\n\n') + '\n';
}

function _aiChatFlash(btn, label, restore) {
  btn.textContent = label;
  setTimeout(() => { btn.textContent = restore; }, 1500);
}

function _aiChatCopy() {
  const btn = $('ai-chat-copy');
  if (!btn || !_aiChat.messages.length) return;
  const text = _aiChatTranscript();
  const legacy = () => {                    // non-secure-context fallback
    const ta = document.createElement('textarea');
    ta.value = text; ta.setAttribute('readonly', '');
    ta.style.position = 'fixed'; ta.style.left = '-9999px';
    document.body.appendChild(ta); ta.select();
    let ok = false;
    try { ok = document.execCommand('copy'); } catch (e) { ok = false; }
    ta.remove();
    return ok;
  };
  const done = (ok) => _aiChatFlash(btn, ok ? 'Copied ✓' : 'Copy failed', 'Copy');
  if (navigator.clipboard && navigator.clipboard.writeText)
    navigator.clipboard.writeText(text).then(() => done(true), () => done(legacy()));
  else done(legacy());
}

function _aiChatClear() {
  if (_aiChat.busy || !_aiChat.messages.length) return;
  if (!_aiChat.clearArmed) {                // first click: arm for 3 s
    _aiChat.clearArmed = true;
    clearTimeout(_aiChat.clearTimer);
    _aiChat.clearTimer = setTimeout(() => {
      _aiChat.clearArmed = false; _aiChatRender();
    }, 3000);
    _aiChatRender();
    return;
  }
  clearTimeout(_aiChat.clearTimer);         // second click: wipe
  _aiChat.clearArmed = false;
  _aiChat.messages = []; _aiChat.error = null; _aiChat.gen++;
  _aiChatRender();
  const inp = $('ai-chat-input');
  if (inp) inp.focus();
}

function _aiChatWarm() {
  // Fire-and-forget: the server pre-builds this scope's facts pack (~1 min on
  // the real book) while the reader is on the brief, so the first turn costs
  // only model time. A miss at send time still builds (server-side lock).
  fetch('/api/ai/chat/warm', {
    method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(_aiChatScope()),
  }).catch(() => {});
}

function _aiChatSend() {
  const input = $('ai-chat-input');
  const qtext = input.value.trim();
  if (!qtext || _aiChat.busy) return;
  input.value = '';
  _aiChat.messages.push({ role: 'user', content: qtext });
  _aiChat.busy = true; _aiChat.error = null;
  const g = ++_aiChat.gen;
  _aiChatRender();
  const done = (err, answer) => {
    if (g !== _aiChat.gen) return;
    _aiChat.busy = false;
    if (answer) _aiChat.messages.push({ role: 'assistant', content: answer });
    else _aiChat.error = err || 'generation failed';
    _aiChatRender();
  };
  fetch('/api/ai/chat', {
    method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ messages: _aiChat.messages, ..._aiChatScope() }),
  }).then((r) => (r.ok ? r.json() : Promise.reject(new Error(String(r.status)))))
    .then((d) => {
      if (d.enabled === false) { done('AI is off (no key)'); return; }
      pollAiNarrative(
        () => fetch('/api/ai/chat?id=' + encodeURIComponent(d.chat_id)).then(
          (r) => (r.ok ? r.json() : Promise.reject(new Error(String(r.status))))),
        () => g === _aiChat.gen,
        (dd) => done(
          dd.error === 'taking longer than usual — try Regenerate'
            ? 'taking longer than usual — ask again'
            : dd.error,
          dd.kind === 'ok' ? dd.text : null),
        () => done('network error'),
        AI_CHAT_POLL_MS);
    })
    .catch(() => done('request failed'));
}

/* ============ GLOBAL CHROME (QA-polish S7) ============ */
let _chromeRetried = false;
function fetchChrome() {
  const q = new URLSearchParams();
  filterSel.broker.forEach((id) => q.append('broker', id));
  const hist = currentHistoryStart();
  if (hist !== 'all') q.set('history_start', hist);
  fetch('/api/chrome?' + q)
    .then((r) => { if (!r.ok) throw new Error(String(r.status)); return r.json(); })
    .then((d) => { _chromeRetried = false; renderGlobalChrome(d); })
    .catch(() => {                       // one warm-up retry, never a loop
      if (!_chromeRetried) { _chromeRetried = true; setTimeout(fetchChrome, 4000); }
    });
}

/* NOT renderChrome — that name is the Holdings meta renderer (As-of picker,
   synthetic badge); S7 shipped a duplicate declaration that silently shadowed
   it (later function definition wins), emptying the As-of picker. */
function renderGlobalChrome(d) {
  // footer
  const foot = $('app-footer');
  if (foot) foot.textContent = d.footer || '';
  // persistent KPI tape — broker/history-scoped like the rest of the chrome.
  // Shipped here (not just in the Holdings payload) so a filter change made
  // from ANY tab repaints it; before this it stayed stale until Holdings
  // refetched.
  if (d.tape) renderTape(d.tape);
  // data-status pill + popover
  const pill = $('pill-status');
  if (!pill) return;
  pill.hidden = false;
  const n = (d.warnings || []).length;
  const dot = $('status-dot');
  dot.className = 'status-dot ' + (n ? 'status-warn' : 'status-ok');
  $('status-n').textContent = n ? String(n) : '';
  const pop = $('status-popover');
  pop.innerHTML = '';
  pop.appendChild(el('div', { class: 'status-cap' }, d.prices_caption || ''));
  (d.warnings || []).forEach((w) => {
    pop.appendChild(el('div', { class: 'status-warning' }, w.icon + ' ' + w.text));
  });
  const src = d.data_sources || {};
  pop.appendChild(el('div', { class: 'status-cap status-sources' }, src.caption || ''));
  if ((src.stale_rows || []).length) {
    const tbl = el('table', { class: 'tbl status-stale-tbl' });
    const thead = el('thead'), htr = el('tr');
    ['Symbol', 'Last bar', 'Last close', 'Days stale'].forEach((h, i) =>
      htr.appendChild(el('th', { class: i === 0 ? 'l' : 'r' }, h)));
    thead.appendChild(htr); tbl.appendChild(thead);
    const tb = el('tbody');
    src.stale_rows.forEach((r) => {
      const tr = el('tr');
      tr.appendChild(el('td', { class: 'l sym' }, r.symbol));
      tr.appendChild(el('td', { class: 'muted' }, r.last_bar));
      tr.appendChild(el('td', { class: 'num' }, r.last_close));
      tr.appendChild(el('td', { class: 'num' }, String(r.days)));
      tb.appendChild(tr);
    });
    tbl.appendChild(tb);
    // The pill/dot/warnings above are chrome, not data -- but this nested
    // table (only built when stale rows exist) is a genuine one, same shape
    // as every other table in this pass, so it gets the same treatment.
    makeSortable(tbl);
    pop.appendChild(tbl);
  }
}

/* ============ TAB ROUTER ============ */
const TABS = {
  holdings:     { api: '/api/holdings',     render: renderHoldings,     body: 'tab-holdings' },
  tax:          { api: '/api/tax',          render: renderTax,          body: 'tab-tax' },
  performance:  { api: '/api/performance',  render: renderPerformance,  body: 'tab-performance' },
  benchmark:    { api: '/api/benchmark',    render: renderBenchmark,    body: 'tab-benchmark' },
  income:       { api: '/api/income',       render: renderIncome,       body: 'tab-income' },
  factor:       { api: '/api/factor',       render: renderFactor,       body: 'tab-factor' },
  risk:         { api: '/api/risk',         render: renderRisk,         body: 'tab-risk' },
  riskcontrib:  { api: '/api/riskcontrib',  render: renderRiskContrib,  body: 'tab-riskcontrib' },
  risksim:      { api: '/api/risksim',      render: renderRiskSim,      body: 'tab-risksim' },
  options:      { api: '/api/options',      render: renderOptions,      body: 'tab-options' },
  health:       { api: '/api/health',       render: renderDataHealth,   body: 'tab-health' },
  dip:          { api: '/api/dip',          render: renderDip,          body: 'tab-dip' },
  ai:           { api: '/api/ai/portfolio', render: renderAiAnalysis,   body: 'tab-ai' },
};
let activeTab = 'holdings';
let dipWatch = [];  // rendered-card symbols (the client-side 'already' guard set)
// Bumped when a data-ops job finishes (the CSVs on disk changed) — invalidates
// every tab's render-cache signature at once.
let dataEpoch = 0;

/* The query string a tab's fetch would use right now (also the tab's
   render-cache identity — same string ⇒ same server response). */
function tabQuery(id, params) {
  if (id === 'health' || id === 'income' || id === 'factor' || id === 'dip'
      || id === 'tax' || id === 'ai') {
    // whole-book: account/asset_class don't apply, but broker + history_start
    // are GLOBAL (tax: the global account picker lists IRAs, which a tax
    // view excludes by design — its account/term/evidence filters are
    // client-side over the shipped lots)
    const p = new URLSearchParams();
    filterSel.broker.forEach((x) => p.append('broker', x));
    const hist = currentHistoryStart();
    if (hist !== 'all') p.set('history_start', hist);
    if (id === 'ai' && _benchState.bench !== 'auto') p.set('benchmark', _benchState.bench);
    return p.toString() ? '?' + p.toString() : '';
  }
  if (id === 'performance' || id === 'benchmark' || id === 'risk' || id === 'riskcontrib' || id === 'risksim') {
    const p = new URLSearchParams();
    filterSel.account.forEach((x) => p.append('account', x));
    filterSel.asset_class.forEach((x) => p.append('asset_class', x));
    filterSel.broker.forEach((x) => p.append('broker', x));
    const hist = currentHistoryStart();
    if (hist !== 'all') p.set('history_start', hist);
    if (id === 'benchmark' && _benchState.bench !== 'auto') p.set('benchmark', _benchState.bench);
    return p.toString() ? '?' + p.toString() : '';
  }
  const qp = (params || currentQuery()).toString();
  return qp ? '?' + qp : '';
}

function switchTab(id) {
  if (!TABS[id]) return;
  activeTab = id;
  // nav active state
  document.querySelectorAll('.nav-item[data-tab]').forEach((b) => {
    b.classList.toggle('active', b.dataset.tab === id);
  });
  // body visibility
  Object.values(TABS).forEach((t) => { $(t.body).hidden = true; });
  $(TABS[id].body).hidden = false;
  // As-of is a Holdings-only control; account/class apply to both
  $('sel-asof').disabled = (id !== 'holdings');
  $('sel-asof').closest('.pill').classList.toggle('dim', id !== 'holdings');
  // `tab` rides only the ADDRESS BAR (deep-link identity) — never the fetch
  // query, so the cache signature is identical whichever path fetched last.
  const params = currentQuery();
  const urlParams = new URLSearchParams(params);
  urlParams.set('tab', id);
  history.replaceState(null, '', '?' + urlParams.toString());
  // Render cache (TK 2026-07-19): coming BACK to a tab must not reload it.
  // Skip the fetch when the tab has rendered content and neither its query
  // string nor the data epoch moved — filter changes and finished data-ops
  // jobs change the signature, so those still refetch. Client-side state on
  // the skipped tab (risksim runs, dip lookups) survives untouched.
  const t = TABS[id];
  if (t._sig != null && t._sig === dataEpoch + '|' + tabQuery(id, params)) {
    // Cached re-entry: no refetch, but aspect-compensated marks (scatter
    // dots, tradeoff markers/stars) may be sized for a stale width — a
    // window resize while this tab was hidden skips _rescaleMarks (a hidden
    // svg measures 0), and a render that finished off-screen drew against
    // the fallback width. Re-scale from the now-visible geometry.
    _rescaleMarks();
    // The chat pack memo is in-process: a server relaunch empties it while
    // this tab's signature still matches, so a cached re-entry must still
    // warm (review catch 2026-08-22) — the server answers 'ready' cheaply.
    if (id === 'ai' && !$('ai-chat-wrap').hidden) _aiChatWarm();
    return;
  }
  fetchTab(id, params);
}

async function fetchTab(id, params) {
  const t = TABS[id];
  // Monotonic token: a late response from a superseded fetch (rapid filter
  // flips) must never paint over the newest one.
  t._tok = (t._tok || 0) + 1;
  const tok = t._tok;
  const qs = tabQuery(id, params);
  const body = $(t.body);
  body.classList.add('loading');
  try {
    const res = await fetch(t.api + qs);
    if (!res.ok) {
      let detail = '';
      try { detail = String((await res.json()).detail || ''); } catch (_) { /* non-JSON body */ }
      // A filter id the server no longer knows — e.g. a Alpine account still
      // selected after the broker pill narrowed to Harbor — 422s by contract.
      // Swallowing that into console.error left the PREVIOUS render on screen
      // (whole-book value under a "Harbor" pill) and, with the signature nulled,
      // re-failed on every re-entry (TK 2026-08-22). Reset that filter to
      // "all", say so, and refetch once; anything else surfaces in the tab.
      const reset = (res.status === 422) ? _resetUnknownFilter(detail) : null;
      if (reset && tok === t._tok) { t._pendingNotice = reset; fetchTab(id); return; }
      throw new Error('HTTP ' + res.status + (detail ? ' — ' + detail : ''));
    }
    const data = await res.json();
    if (tok !== t._tok) return;   // superseded while in flight
    t.render(data);
    tabNotice(body, t._pendingNotice || null);
    t._pendingNotice = null;
    // Cache identity recorded from POST-render state (tabQuery(id, null) reads
    // the selects fresh): the first holdings render populates sel-asof, so the
    // pre-render query string would never match the next visit's check.
    t._sig = dataEpoch + '|' + tabQuery(id, null);
  } catch (err) {
    console.error('tab load failed', id, err);
    if (tok === t._tok) {
      tabNotice(body, 'Could not load this tab (' + err.message
        + '). The content below may be stale — adjust the filters or reload.');
    }
    t._sig = null;                   // failed/partial render -> next visit retries
  } finally {
    if (tok === t._tok) body.classList.remove('loading');
  }
}

/* One amber notice pinned to the top of a tab body (null clears it). The tab
   renderers rewrite their own sections, never the body, so it survives a
   render and is cleared explicitly on the next clean load. */
function tabNotice(body, text) {
  let n = body.querySelector(':scope > .tab-notice');
  if (!text) { if (n) n.remove(); return; }
  if (!n) {
    n = el('div', { class: 'callout callout-warn tab-notice' });
    body.insertBefore(n, body.firstChild);
  }
  n.innerHTML = '';
  n.appendChild(el('span', { class: 'callout-icon' }, '⚠'));
  n.appendChild(el('span', { class: 'callout-text' }, text));
}

/* Opt-in: warm every tab (compute + its AI box) into its hidden body WITHOUT
   switching the visible tab, so clicking through afterward is instant. Skips
   already-warm tabs (the switchTab render-cache signature) and paces the
   compute burst through a small pool; the AI-narration pollers each render
   starts run async in the background (the "warm everything" heavier path). */
async function loadAllTabs() {
  const btn = $('load-all-tabs');
  if (btn && btn.disabled) return;                 // already warming
  const warm = (id) => {
    const t = TABS[id];
    return t._sig != null && t._sig === dataEpoch + '|' + tabQuery(id, null);
  };
  const ids = Object.keys(TABS).filter((id) => !warm(id));
  const orig = btn ? btn.innerHTML : '';
  if (btn) btn.disabled = true;
  const total = ids.length;
  let done = 0;
  const tick = () => { if (btn) btn.textContent = total ? ('Loading ' + done + '/' + total + '…') : 'All loaded'; };
  tick();
  const POOL = 3;
  let idx = 0;
  const worker = async () => {
    while (idx < ids.length) {
      const id = ids[idx++];
      await fetchTab(id);                          // fetchTab catches its own errors
      done += 1; tick();
    }
  };
  await Promise.all(Array.from({ length: Math.min(POOL, ids.length) }, worker));
  if (btn) {
    // Keep the button disabled for the whole "All loaded" display window so a
    // double-click can't re-enter and re-capture "All loaded" as the label.
    if (total === 0) { setTimeout(() => { btn.innerHTML = orig; btn.disabled = false; }, 1200); }
    else { btn.innerHTML = orig; btn.disabled = false; }
  }
}

function onFilterChange() {
  fetchTab(activeTab);
  fetchChrome();       // broker/history feed the global chrome too
}

/* ============ DATA-OPS ACTIONS (QA-polish S6) ============ */
/* One job at a time (server-enforced single-flight). Click -> POST -> 1s
   status polling into the rail panel; on completion the active tab refetches
   (frames reload per request, so refreshed CSVs appear without a restart). */
let _actTimer = null;

function _actIcon(s) {
  return s === 'running' ? '▶' : s === 'pending' ? '·' : (s || '·');
}

function renderActPanel(st) {
  const panel = $('act-panel');
  if (!panel) return;
  const show = st && (st.running || st.ok != null);
  panel.hidden = !show;
  document.querySelectorAll('.act-btn').forEach((b) => { b.disabled = !!(st && st.running); });
  if (!show) return;
  panel.innerHTML = '';
  const head = st.running
    ? '▶ ' + (st.label || '') + '…'
    : (st.ok ? '✅ ' : '⚠️ ') + (st.label || '') + (st.ok ? ' — done' : ' — finished with issues');
  panel.appendChild(el('div', { class: 'act-head' }, head));
  (st.steps || []).forEach((s) => {
    panel.appendChild(el('div', { class: 'act-step' }, _actIcon(s.status) + ' ' + s.label));
  });
  if (st.tail) {
    const det = el('details', { class: 'act-tail' });
    det.appendChild(el('summary', {}, 'last output'));
    det.appendChild(el('pre', {}, st.tail));
    panel.appendChild(det);
  }
}

function _actWatch() {
  if (_actTimer) return;
  const tick = () => {
    fetch('/api/actions/status').then((r) => r.json()).then((st) => {
      renderActPanel(st);
      if (st.running) {
        _actTimer = setTimeout(tick, 1000);
      } else {
        _actTimer = null;
        dataEpoch += 1;               // CSVs changed on disk — every cached tab is stale
        fetchTab(activeTab);          // pick up the refreshed CSVs
        fetchChrome();                // staleness warnings may have cleared
      }
    }).catch(() => { _actTimer = null; });
  };
  _actTimer = setTimeout(tick, 400);
}

function startAction(id, label) {
  fetch('/api/actions/' + id, { method: 'POST' })
    .then((r) => {
      if (r.ok && label) {
        // optimistic panel — the first status poll lands ~1s later
        renderActPanel({ running: true, label: label.replace(/^⟳\s*/, ''), steps: [] });
      }
      if (r.ok || r.status === 409) _actWatch();
    })
    .catch(() => {});
}

/* Small per-tab action button (Income / Factor / Options). Created inside
   hosts the renderers innerHTML='' every pass, so no duplicate stacking. */
function actButton(id, label) {
  const b = el('button', { class: 'act-inline act-btn', type: 'button' }, label);
  b.addEventListener('click', () => startAction(id, label));
  return b;
}

/* History's own change handler: clamp sel-asof's OPTIONS to the new cutoff
   BEFORE the shared refetch reads sel-asof.value. Without this, picking a
   History cutoff that excludes the currently-selected As-of date would send
   that now-out-of-range as_of on the very next request — the server doesn't
   422 it (validated against the load-time-cached, never-narrowed
   available_dates field) but positions ARE sliced by the cutoff, so it would
   render a spurious empty Holdings snapshot for one round trip. Clamping
   first lets the browser's native "remove the selected option" behavior
   settle sel-asof.value onto a valid date first. */
function onHistoryChange() {
  clampAsOfToHistory(currentHistoryStart());
  onFilterChange();
}

/* Re-scale aspect-compensated marks after a viewport resize. Deliberately NOT a
   full tab re-render (that would wipe in-progress risk-sim grid edits and
   re-fire the Options live chain fetch) — only rx / star geometry is touched.
   %-positioned overlays (axes, flags, gauge value) survive resize by design. */
function _rescaleMarks() {
  document.querySelectorAll('svg[data-vbw]').forEach((svg) => {
    const w = svg.getBoundingClientRect().width;
    const vbw = parseFloat(svg.getAttribute('data-vbw'));
    if (!w || !vbw) return;                        // hidden tab: rescaled on next render
    const sx = w / vbw;
    svg.querySelectorAll('ellipse[data-rpx]').forEach((e) =>
      e.setAttribute('rx', (parseFloat(e.getAttribute('data-rpx')) / sx).toFixed(2)));
    svg.querySelectorAll('path[data-star]').forEach((p) =>
      p.setAttribute('d', _starPath(parseFloat(p.getAttribute('data-cx')),
        parseFloat(p.getAttribute('data-cy')), parseFloat(p.getAttribute('data-r')),
        parseFloat(p.getAttribute('data-spikes')), sx)));
  });
}
let _rescaleT = null;

function init() {
  $('sel-asof').addEventListener('change', onFilterChange);
  if ($('sel-history')) $('sel-history').addEventListener('change', onHistoryChange);
  window.addEventListener('resize', () => {
    clearTimeout(_rescaleT); _rescaleT = setTimeout(_rescaleMarks, 150);
  });
  // The whole pill opens the picker — clicking the "As of"/"History" key text
  // only focuses the wrapped <select> natively, which reads as a dead zone.
  document.querySelectorAll('label.pill-select').forEach((pill) => {
    const sel = pill.querySelector('select');
    if (!sel) return;
    pill.addEventListener('click', (e) => {
      if (e.target === sel || sel.disabled) return;
      e.preventDefault();                       // stop the label's re-dispatched click
      try { sel.showPicker(); } catch (_) { sel.focus(); }
    });
  });
  document.querySelectorAll('.nav-item[data-tab]').forEach((b) => {
    b.addEventListener('click', () => switchTab(b.dataset.tab));
  });
  const _dipForm = $('dip-adhoc-form');
  if (_dipForm) _dipForm.addEventListener('submit', onDipLookup);
  // data-ops buttons (rail group + the ctl "Refresh all" chip)
  document.querySelectorAll('.act-btn[data-action]').forEach((b) => {
    b.addEventListener('click', () => startAction(b.dataset.action, b.textContent.trim()));
  });
  const _loadAll = $('load-all-tabs');
  if (_loadAll) _loadAll.addEventListener('click', loadAllTabs);
  // a job may already be running (page reload mid-run): resume the poll
  fetch('/api/actions/status').then((r) => r.json()).then((st) => {
    if (st.running) _actWatch(); else renderActPanel(st);
  }).catch(() => {});
  // global chrome (status pill / footer) + popover toggle
  const statusPill = $('pill-status');
  if (statusPill) {
    statusPill.addEventListener('click', (e) => {
      const pop = $('status-popover');
      if (!pop.contains(e.target)) pop.hidden = !pop.hidden;
    });
    document.addEventListener('click', (e) => {
      if (!statusPill.contains(e.target)) $('status-popover').hidden = true;
    });
  }
  fetchChrome();
  const startTab = new URLSearchParams(location.search).get('tab') || 'holdings';
  switchTab(startTab);
}

if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', init);
} else {
  init();
}

/* ============ PERFORMANCE RENDER ============ */
const PERF_EXPLAINER =
  "<b>Time-weighted return (TWR)</b> strips out deposits, withdrawals, and " +
  "cross-account transfers, isolating investment performance from new-money " +
  "effects. Each month's return is chained: (1+R₁)(1+R₂)…(1+Rₙ) − 1.<br><br>" +
  "<b>Money-weighted return (IRR)</b> is the annualized rate that NPVs the " +
  "actual cashflow stream to zero. TWR asks “how well did the strategy " +
  "perform?”, IRR asks “what rate did you actually earn given when " +
  "you put money in?”.";

/* scale helpers for hand-built SVG line/area charts */
function _bounds(points, key) {
  let lo = Infinity, hi = -Infinity;
  points.forEach((p) => { lo = Math.min(lo, p[key]); hi = Math.max(hi, p[key]); });
  if (!isFinite(lo)) { lo = 0; hi = 1; }
  if (lo === hi) { hi = lo + 1; }
  return [lo, hi];
}

/* build an area+line SVG into `host`. opts: {color, key, baseline} */
function drawAreaChart(host, points, opts) {
  host.innerHTML = '';
  if (!points || !points.length) {
    host.appendChild(el('div', { class: 'empty-state' }, 'No series.')); return;
  }
  const W = 900, H = opts.height || 240, padL = 4, padR = 4, padT = 12, padB = 12;
  const [lo0, hi0] = _bounds(points, opts.key);
  const lo = Math.min(lo0, opts.baseline != null ? opts.baseline : lo0);
  const hi = Math.max(hi0, opts.baseline != null ? opts.baseline : hi0);
  const x = (i) => padL + (i / (points.length - 1 || 1)) * (W - padL - padR);
  const y = (v) => padT + (1 - (v - lo) / (hi - lo || 1)) * (H - padT - padB);
  const svg = svgEl('svg', { viewBox: `0 0 ${W} ${H}`, width: '100%',
    height: String(H), preserveAspectRatio: 'none', style: 'display:block' });
  // gridlines (quartiles)
  [0.25, 0.5, 0.75].forEach((f) => svg.appendChild(svgEl('line',
    { x1: 0, x2: W, y1: padT + f * (H - padT - padB), y2: padT + f * (H - padT - padB),
      stroke: '#303C4B' })));
  if (opts.baseline != null) svg.appendChild(svgEl('line',
    { x1: 0, x2: W, y1: y(opts.baseline), y2: y(opts.baseline),
      stroke: '#4C5866', 'stroke-dasharray': '2 3' }));
  const xy = points.map((p, i) => `${x(i).toFixed(1)},${y(p[opts.key]).toFixed(1)}`);
  const line = xy.join(' ');
  const base = y(opts.baseline != null ? opts.baseline : lo);
  svg.appendChild(svgEl('path', { d: `M${x(0)},${base} L${line.replace(/ /g, ' L')} L${x(points.length - 1)},${base} Z`,
    fill: opts.fill || 'rgba(77,163,245,.18)' }));
  // Provisional tail (interim stub, spec 2026-08-22): the solid line stops at
  // the last statement point; the final segment is dashed amber with a
  // hollow marker. Points carry `provisional: true` only on the stub row.
  const provIdx = points.findIndex((p) => p.provisional);
  const solid = (provIdx > 0 ? xy.slice(0, provIdx) : xy).join(' ');
  svg.appendChild(svgEl('polyline', { points: solid, fill: 'none',
    stroke: opts.color, 'stroke-width': '2' }));
  if (provIdx > 0) {
    svg.appendChild(svgEl('polyline', { points: xy.slice(provIdx - 1).join(' '), fill: 'none',
      stroke: '#E0A030', 'stroke-width': '2', 'stroke-dasharray': '5 4' }));
    const lp = points[points.length - 1];
    svg.appendChild(svgEl('circle', { cx: x(points.length - 1).toFixed(1),
      cy: y(lp[opts.key]).toFixed(1), r: '4', fill: '#0F141D',
      stroke: '#E0A030', 'stroke-width': '2' }));
  }
  // onboarding markers: dashed vline in-svg; the label as an HTML overlay chip
  // (svg <text> stretches with viewport width under preserveAspectRatio:none)
  let flagLayer = null;
  let flagRow = 0;
  (opts.markers || []).forEach((m) => {
    const idx = points.findIndex((p) => p.x === m.x);
    if (idx < 0) return;
    const mx = x(idx);
    svg.appendChild(svgEl('line', { x1: mx, x2: mx, y1: 0, y2: H - padB,
      stroke: '#4DA3F5', 'stroke-dasharray': '4 4', opacity: '.65' }));
    if (!flagLayer) flagLayer = el('div', { class: 'chart-axes' });
    const flag = el('div', { class: 'ax-flag' }, m.label);
    flag.style.left = ((mx + 4) / W * 100).toFixed(3) + '%';
    // stagger rows so neighboring flags don't mash at narrow widths; the
    // full (possibly "·"-joined) label rides the title tooltip
    flag.style.top = (2 + (flagRow % 3) * 13) + 'px';
    flag.title = m.label;
    flagLayer.appendChild(flag);
    flagRow++;
  });
  host.appendChild(svg);
  if (flagLayer) {
    host.style.position = 'relative';
    host.appendChild(flagLayer);
    _pinOverlay(flagLayer, host, H);
  }
  attachAxes(host, {
    W, H, padL, padR, padT, padB, yLo: lo, yHi: hi,
    yfmt: opts.yfmt || axNum,
    x: opts.xdates === false ? null : { kind: 'date', points },
  });
  attachCrosshair(host, {
    W, H, padL, padR, padT, padB, yLo: lo, yHi: hi, n: points.length, points,
    series: [{ label: opts.crosshairLabel || '', color: opts.color,
               values: points.map((p) => p[opts.key]), fmt: opts.yfmt || axNum }],
  });
  // drag-zoom: re-render with the sliced window (bounds rescale); opts._full
  // remembers the original for double-click / ⟲ reset.
  const zoomFull = opts._full || points;
  attachZoom(host, { W, H, padL, padR, n: points.length }, {
    zoomed: !!opts._full,
    onZoom: (i0, i1) => drawAreaChart(host, points.slice(i0, i1 + 1),
      Object.assign({}, opts, { _full: zoomFull })),
    onReset: opts._full
      ? () => drawAreaChart(host, zoomFull, Object.assign({}, opts, { _full: null }))
      : null,
  });
}

function chartCard(host, head, drawFn) {
  host.innerHTML = '';
  host.appendChild(el('div', { class: 'ch' }, [
    el('div', { class: 'ctitle' }, head.title),
    el('div', { class: 'csub' }, head.sub),
  ]));
  const slot = el('div');
  host.appendChild(slot);
  drawFn(slot);
}

function renderPerfHeadline(cards) {
  const grid = el('div', { class: 'snapshot-grid' });
  (cards || []).forEach((c) => {
    grid.appendChild(el('div', { class: 'kpi' }, [
      el('div', { class: 'kpi-label' }, c.label),
      el('div', { class: 'kpi-value ' + (c.color || '') }, c.value),
      el('div', { class: 'kpi-sub' }, c.sub || ''),
    ]));
  });
  const root = $('perf-headline'); root.innerHTML = ''; root.appendChild(grid);
}

function renderPerfCashflows(cf) {
  const root = $('perf-cashflows'); root.innerHTML = '';
  if (!cf) { root.appendChild(el('div', { class: 'footnote' },
    'Cash-flow tiles are hidden under an asset-class filter — transactions aren’t tagged by class.')); return; }
  const grid = el('div', { class: 'flow-grid' });
  const tile = (label, val, sub, cls) => el('div', { class: 'kpi' }, [
    el('div', { class: 'kpi-label' }, label),
    el('div', { class: 'kpi-value ' + (cls || ''), style: 'font-size:21px' }, val),
    el('div', { class: 'kpi-sub' }, sub),
  ]);
  grid.appendChild(tile('Deposits in', cf.deposits.value, cf.deposits.n + ' transfers', 'gain'));
  grid.appendChild(tile('Withdrawals out', cf.withdrawals.value, cf.withdrawals.n + ' transfers', 'loss'));
  grid.appendChild(tile('Net new money', cf.net.value, 'Real outside capital', ''));
  root.appendChild(grid);
  if (cf.synthetic_note) root.appendChild(el('div', { class: 'cap' }, cf.synthetic_note));
}

function renderPerfPeriodic(per) {
  const root = $('perf-periodic'); root.innerHTML = '';
  const head = el('div', { class: 'ch' }, [
    el('div', { class: 'ctitle' }, 'TWR by period'),
    el('div', { class: 'csub' }, 'green positive · coral negative'),
  ]);
  const seg = el('span', { class: 'seg', style: 'margin-left:auto' });
  ['monthly', 'quarterly', 'yearly'].forEach((g, i) => {
    const b = el('button', { class: i === 0 ? 'on' : '' }, g[0].toUpperCase() + g.slice(1));
    b.addEventListener('click', () => {
      seg.querySelectorAll('button').forEach((x) => x.classList.remove('on'));
      b.classList.add('on');
      drawBars(per[g]);
    });
    seg.appendChild(b);
  });
  head.appendChild(seg);
  root.appendChild(head);
  const barsHost = el('div');
  const cap = el('div', { class: 'cap' });
  root.appendChild(barsHost); root.appendChild(cap);
  // drawSpreadBars: the same single-series diverging green/coral bars this
  // chart wants (the old bespoke .vbars encoded |v| only — sign was color-only,
  // unlike Streamlit's signed bars) + y-axis + x date labels for free.
  function drawBars(gran) {
    drawSpreadBars(barsHost, gran.bars || []);
    cap.textContent = 'Win-rate ' + gran.winrate + '.';
  }
  drawBars(per.monthly);
}

function renderPerfPerAccount(pa) {
  const table = $('perf-peraccount'); table.innerHTML = '';
  const cols = [['Account', 'l'], ['First', 'l'], ['Last', 'l'], ['Months', 'r'],
    ['Start NAV', 'r'], ['End NAV', 'r'], ['Net flow', 'r'],
    ['Cum TWR %', 'r'], ['Ann TWR %', 'r'], ['IRR %', 'r']];
  const thead = el('thead'), htr = el('tr');
  cols.forEach(([l, a]) => htr.appendChild(el('th', { class: a }, l)));
  thead.appendChild(htr); table.appendChild(thead);
  const tb = el('tbody');
  (pa.rows || []).forEach((r) => {
    const tr = el('tr');
    tr.appendChild(el('td', { class: 'l sym' }, r.account));
    tr.appendChild(el('td', { class: 'l' }, r.first));
    tr.appendChild(el('td', { class: 'l' }, r.last));
    tr.appendChild(el('td', { class: 'num' }, String(r.months)));
    tr.appendChild(el('td', { class: 'muted' }, r.start_nav));
    tr.appendChild(el('td', { class: 'muted' }, r.end_nav));
    tr.appendChild(el('td', { class: 'muted' }, r.net_flow));
    tr.appendChild(el('td', {}, r.cum_twr === '—' ? document.createTextNode('—') : buildChip(r.cum_twr, r.cum_dir, false)));
    tr.appendChild(el('td', {}, r.ann_twr === '—' ? document.createTextNode('—') : buildChip(r.ann_twr, r.ann_dir, false)));
    tr.appendChild(el('td', {}, r.irr === '—' ? document.createTextNode('—') : buildChip(r.irr, r.irr_dir, false)));
    tb.appendChild(tr);
  });
  table.appendChild(tb);
  $('perf-peraccount-foot').textContent = pa.footnote || '';
  makeSortable(table);
}

function renderPerfNav(nav) {
  const trio = $('perf-nav-trio'); trio.innerHTML = '';
  if (nav.trio) {
    const g = el('div', { class: 'flow-grid' });
    [['Current NAV', nav.trio.current], ['All-time-peak NAV', nav.trio.peak],
     ['Months tracked', nav.trio.months]].forEach(([label, t]) => {
      g.appendChild(el('div', { class: 'kpi' }, [
        el('div', { class: 'kpi-label' }, label),
        el('div', { class: 'kpi-value', style: 'font-size:21px' }, t.value),
        el('div', { class: 'kpi-sub' }, t.sub),
      ]));
    });
    trio.appendChild(g);
  }
  chartCard($('perf-nav'), nav.head, (slot) => drawAreaChart(slot, nav.points,
    { color: '#2DD4BF', key: 'v', fill: 'rgba(45,212,191,.18)', markers: nav.markers, height: 240, yfmt: axUsd }));
  $('perf-nav-note').textContent = nav.reconcile_note || '';
}

/* Build the Account/Class/Broker pills once from a tab's meta if they haven't
   been built yet (e.g. landing on ?tab=performance before Holdings has
   rendered); selection state lives in filterSel. Also (re)populates the
   History select — cheap to refresh unconditionally since it's a plain
   <select>, not a stateful popover like the multi-selects. */
function ensureFilterSelects(meta) {
  if (!$('ms-account').children.length) {
    buildMultiSelect('ms-account', 'account', 'All accounts', meta.accounts || []);
  }
  if (!$('ms-class').children.length) {
    buildMultiSelect('ms-class', 'asset_class', 'All asset classes', meta.classes || []);
  }
  if (!$('ms-broker').children.length) {
    const bopts = _brokerOpts(meta);
    buildMultiSelect('ms-broker', 'broker', _brokerAllLabel(bopts), bopts);
  }
  populateHistorySelect(meta);
}

function renderPerformance(data) {
  ensureFilterSelects(data.meta);
  $('perf-lede').textContent =
    'Time-weighted return (TWR), drawdowns, and per-account breakdown — deposits and transfers stripped out so only investment performance shows.';
  $('perf-explainer').innerHTML = PERF_EXPLAINER;
  // disclosures
  const comb = $('perf-combined');
  if (data.disclosures && data.disclosures.combined_statement) {
    comb.hidden = false; comb.innerHTML = '';
    comb.appendChild(el('span', { class: 'callout-icon' }, 'ℹ'));
    comb.appendChild(el('span', { class: 'callout-text' }, data.disclosures.combined_statement.text));
  } else { comb.hidden = true; }
  const filt = $('perf-filter');
  if (data.disclosures && data.disclosures.holdings_filter) {
    filt.hidden = false; filt.innerHTML = '';
    filt.appendChild(el('span', { class: 'callout-icon' }, 'ℹ'));
    filt.appendChild(el('span', { class: 'callout-text' }, data.disclosures.holdings_filter.text));
  } else { filt.hidden = true; }
  $('perf-headline-note').textContent = data.headline.length
    ? ('since ' + (data.headline[0].sub || '').replace('Since ', '')) : '';

  if (data.meta.empty) {
    renderPerfHeadline([]);
    $('perf-cashflows').innerHTML = '';
    $('perf-ai').innerHTML = '';
    ['perf-cumtwr', 'perf-drawdown', 'perf-periodic', 'perf-nav'].forEach((id) => { $(id).innerHTML = ''; });
    const pcap0 = $('perf-stub-caption');
    if (pcap0) { pcap0.hidden = true; pcap0.textContent = ''; }
    $('perf-peraccount').innerHTML = '';
    $('perf-peraccount-foot').textContent = '';
    $('perf-nav-trio').innerHTML = '';
    $('perf-nav-note').textContent = '';
    $('perf-headline').appendChild(el('div', { class: 'empty-state' },
      'No return series for the current filter — the slice has no priceable holdings.'));
    return;
  }

  mountAiBox('perf-ai', 'performance');   // B3: narrates the filtered slice (B2 pattern)
  renderPerfHeadline(data.headline);
  renderPerfCashflows(data.cashflows);
  chartCard($('perf-cumtwr'), data.cum_twr.head, (slot) => drawAreaChart(slot, data.cum_twr.points,
    { color: '#4DA3F5', key: 'v', baseline: 0, markers: data.cum_twr.markers, height: 240, yfmt: axPct }));
  const pcap = $('perf-stub-caption');
  if (pcap) { pcap.textContent = data.stub ? data.stub.caption : ''; pcap.hidden = !data.stub; }
  chartCard($('perf-drawdown'), data.drawdown.head, (slot) => drawAreaChart(slot, data.drawdown.points,
    { color: '#FB6F63', key: 'dd', baseline: 0, fill: 'rgba(251,111,99,.22)', height: 160, yfmt: axPct }));
  renderPerfPeriodic(data.periodic);
  renderPerfPerAccount(data.per_account);
  renderPerfNav(data.nav);
}

/* ============ BENCHMARK (Performance vs SPY) ============ */

/* Bounds folded across every series in `series` for the field `key`,
   optionally including a baseline value in the range. */
function _boundsMulti(series, key, baseline) {
  let lo = Infinity, hi = -Infinity;
  (series || []).forEach((s) => {
    (s.points || []).forEach((p) => {
      const v = p[key];
      if (v == null || !isFinite(v)) return;
      lo = Math.min(lo, v); hi = Math.max(hi, v);
    });
  });
  if (baseline != null) { lo = Math.min(lo, baseline); hi = Math.max(hi, baseline); }
  if (!isFinite(lo)) { lo = 0; hi = 1; }
  if (lo === hi) { hi = lo + 1; }
  return [lo, hi];
}

/* Light tint of a hex/rgb color for area fills (low-alpha rgba). */
function _tint(color, alpha) {
  if (color && color[0] === '#') {
    const h = color.slice(1);
    const r = parseInt(h.slice(0, 2), 16);
    const g = parseInt(h.slice(2, 4), 16);
    const b = parseInt(h.slice(4, 6), 16);
    return `rgba(${r},${g},${b},${alpha})`;
  }
  return color;
}

/* "Nice" axis tick values spanning [lo, hi] with ~target steps, using
   1/2/2.5/5 ×10^n increments (Heckbert). Ascending, clamped within [lo, hi].
   Non-finite input -> []; zero-width -> [lo]. Pure; no DOM. */
function niceTicks(lo, hi, target) {
  target = target || 5;
  if (!isFinite(lo) || !isFinite(hi)) return [];
  if (hi < lo) { const t = lo; lo = hi; hi = t; }
  if (hi === lo) return [lo];
  const raw = (hi - lo) / target;
  const mag = Math.pow(10, Math.floor(Math.log10(raw)));
  const res = raw / mag;
  let step = res <= 1 ? 1 : res <= 2 ? 2 : res <= 2.5 ? 2.5 : res <= 5 ? 5 : 10;
  step *= mag;
  const start = Math.ceil(lo / step) * step;
  const ticks = [];
  for (let i = 0; start + i * step <= hi + 1e-9 * step; i++) {
    ticks.push(Number((start + i * step).toFixed(10)));
  }
  return ticks;
}

/* Shared axis value formatters (callers pick one for yfmt/xfmt). */
const axPct = (v) => v.toFixed(Math.abs(v) >= 100 ? 0 : 1) + '%';
const axPctFrac = (v) => (v * 100).toFixed(Math.abs(v * 100) >= 100 ? 0 : 1) + '%'; // decimal-fraction values (e.g. opt_curve vol 0.169), unlike axPct's already-*100 inputs
const axNum = (v) => Math.abs(v) >= 100 ? v.toFixed(0) : v.toFixed(Math.abs(v) < 1 ? 2 : 1);
const axUsd = _fmtUsd;                            // _fmtUsd already exists (grep it)
const _shortDate = (s) => { s = String(s); return /^\d{4}-\d{2}-\d{2}/.test(s) ? s.slice(0, 7) : s; }; // ISO -> 'YYYY-MM'; other label strings pass through

/* Overlay HTML axis ticks on a chart host, reusing the drawer's own linear
   scale. spec = {
     W, H, padL, padR, padT, padB,       // viewBox dims + pads (the drawer's)
     yLo, yHi, yfmt, yCount=5,           // y value range + formatter
     x: null                             // categorical: no x ticks
        | {kind:'date', points, count=6} // sampled index -> points[i].x date
        | {kind:'num', lo, hi, xfmt, count=5}
   }
   Positions labels by % of the host box; never emits SVG text. */
function attachAxes(host, spec) {
  const { W, H, padL, padR, padT, padB, yLo, yHi } = spec;
  host.style.position = 'relative';
  const layer = el('div', { class: 'chart-axes' });
  // Pin the overlay to the svg's OWN box (not the full host): some hosts are
  // taller than the svg (legend/marker rows) or padded (.card) — an inset:0
  // layer would otherwise stretch/offset and drift the % ticks.
  _pinOverlay(layer, host, H);
  const yPix = (v) => padT + (1 - (v - yLo) / ((yHi - yLo) || 1)) * (H - padT - padB);
  const xPixI = (i, n) => padL + (i / ((n - 1) || 1)) * (W - padL - padR);
  const xPixV = (v, lo, hi) => padL + ((v - lo) / ((hi - lo) || 1)) * (W - padL - padR);
  // y ticks (spec.yTicks overrides — e.g. log-space decades on the histogram)
  (spec.yTicks || niceTicks(yLo, yHi, spec.yCount || 5)).forEach((t) => {
    const lab = el('div', { class: 'ax-y' }, (spec.yfmt || axNum)(t));
    lab.style.top = (yPix(t) / H * 100).toFixed(3) + '%';
    layer.appendChild(lab);
  });
  // x ticks. Labels are translateX(-50%)-centered; a label whose tick sits on a
  // plot edge would overhang it (and collide with the corner y-label), so:
  //  - clamp: within ~1.5% of an edge, left/right-align instead (classes below);
  //  - date mode samples MIDPOINTS ((k+0.5)/cnt of the index range, no endpoint
  //    ticks) — exact start/end dates stay available via the crosshair.
  const edgeClamp = (lab, leftPct) => {
    if (leftPct <= (padL / W * 100) + 1.5) lab.classList.add('ax-x-first');
    else if (leftPct >= 100 - (padR / W * 100) - 1.5) lab.classList.add('ax-x-last');
  };
  const xs = spec.x;
  if (xs && xs.kind === 'date') {
    const pts = xs.points || [];
    const n = pts.length;
    const cnt = Math.min(xs.count || 6, n);
    const xPixBar = (i) => padL + ((i + 0.5) / n) * (W - padL - padR);
    for (let k = 0; k < cnt; k++) {
      const i = (n <= 1 || cnt <= 1) ? Math.floor(n / 2)
        : xs.bar ? Math.min(n - 1, Math.max(0, Math.round((k + 0.5) * n / cnt - 0.5)))
                 : Math.round((k + 0.5) * (n - 1) / cnt);
      const lab = el('div', { class: 'ax-x' }, _shortDate(pts[i].x != null ? pts[i].x : pts[i].t));
      const leftPct = (xs.bar ? xPixBar(i) : xPixI(i, n)) / W * 100;
      lab.style.left = leftPct.toFixed(3) + '%';
      edgeClamp(lab, leftPct);
      layer.appendChild(lab);
    }
  } else if (xs && xs.kind === 'num') {
    niceTicks(xs.lo, xs.hi, xs.count || 5).forEach((t) => {
      const lab = el('div', { class: 'ax-x' }, (xs.xfmt || axNum)(t));
      const leftPct = xPixV(t, xs.lo, xs.hi) / W * 100;
      lab.style.left = leftPct.toFixed(3) + '%';
      edgeClamp(lab, leftPct);
      layer.appendChild(lab);
    });
  }
  host.appendChild(layer);
}

/* Pin an overlay layer (axis ticks / crosshair) to the host's CONTENT box, so
   host padding (e.g. a .card = 18px) doesn't offset it and a legend row appended
   below the svg doesn't stretch it. Reads padding via getComputedStyle — a static
   value, correct even before layout / on a hidden tab (unlike offsetWidth, which
   is 0 until the element is laid out). The svg is the host's first child (drawers
   innerHTML='' then append it), so it sits at the content-box top and is H px tall. */
function _pinOverlay(layer, host, H) {
  const cs = getComputedStyle(host);
  layer.style.top = (parseFloat(cs.paddingTop) || 0) + 'px';
  layer.style.left = (parseFloat(cs.paddingLeft) || 0) + 'px';
  layer.style.right = (parseFloat(cs.paddingRight) || 0) + 'px';
  layer.style.height = H + 'px';
  layer.style.bottom = 'auto';
}

/* Labels-only y-axis ticks for the hand-built HTML <div> bar charts. Unlike the
   8 SVG cartesian charts (attachAxes, viewBox %), these have no viewBox — their
   scale is direct CSS %, so this is a purpose-built sibling. Reuses niceTicks +
   the .chart-axes/.ax-y overlay CSS (labels only, no gridlines). Two geometries:
     kind:'diverging' — symmetric [-max,+max] about the CSS mid-line; the plot
       area is the host's content box (static padding → no measurement).
     kind:'baseline'  — [0,max] up from a baseline; the plot area is the
       .topbar-track band, which sits ABOVE a label row and must be measured.
   host = the .diverge-host card (diverging) or the .topbars wrap (baseline).
   spec = { kind, max, yfmt, yCount=5 }. No-op on a missing host / non-finite/≤0
   max (callers already early-return on an empty series). */
const AX_GUTTER = 54;                     // left gutter (px) for y-labels; clears the widest ($NN,NNN / -NN.N%) chip
function attachDivYAxis(host, spec) {
  const { kind, max } = spec;
  const fmt = spec.yfmt || axNum;
  if (!host || !isFinite(max) || max <= 0) return;
  host.style.position = 'relative';
  host.style.paddingLeft = AX_GUTTER + 'px';   // inline: overrides the class shorthand's left only
  const layer = el('div', { class: 'chart-axes' });
  const addTicks = (lo, hi, toPct) => {
    niceTicks(lo, hi, spec.yCount || 5).forEach((t) => {
      const lab = el('div', { class: 'ax-y' }, fmt(t));
      lab.style.top = toPct(t).toFixed(3) + '%';
      layer.appendChild(lab);
    });
  };
  if (kind === 'diverging') {
    // Plot area = content box; +max at top (0%), 0 at mid (50%), -max at bottom.
    const cs = getComputedStyle(host);
    layer.style.top = (parseFloat(cs.paddingTop) || 0) + 'px';
    layer.style.bottom = (parseFloat(cs.paddingBottom) || 0) + 'px';
    layer.style.left = '0';
    layer.style.right = '0';
    layer.style.height = 'auto';
    addTicks(-max, max, (t) => 50 * (1 - t / max));
    host.appendChild(layer);
  } else if (kind === 'baseline') {
    // Plot area = the .topbar-track band (excludes the ticker-label row).
    const track = host.querySelector('.topbar-track');
    if (!track) return;
    const hostR = host.getBoundingClientRect();
    const trR = track.getBoundingClientRect();
    if (trR.height <= 0) return;   // not laid out (hidden tab); income renders on the active tab — smoke-verified
    layer.style.top = (trR.top - hostR.top) + 'px';
    layer.style.left = '0';
    layer.style.right = '0';
    layer.style.height = trR.height + 'px';
    layer.style.bottom = 'auto';
    addTicks(0, max, (t) => (1 - t / max) * 100);
    host.appendChild(layer);
  }
}

/* Sampled x-category labels for the HTML <div> bar charts (sibling of
   attachDivYAxis — same overlay idiom). Bottom-pinned chips at (i+0.5)/n of the
   host CONTENT box: reading paddingLeft clears the AX_GUTTER when attachDivYAxis
   ran first (it sets the inline padding), and is a no-op 0 otherwise. Labels
   every bar when n <= 16 (e.g. risk-sim tickers), else ~8 sampled midpoints. */
function attachDivXLabels(host, labels) {
  const n = (labels || []).length;
  if (!host || !n) return;
  host.style.position = 'relative';
  const cs = getComputedStyle(host);
  const layer = el('div', { class: 'chart-axes' });
  layer.style.top = 'auto';
  layer.style.bottom = (parseFloat(cs.paddingBottom) || 0) + 'px';
  layer.style.left = (parseFloat(cs.paddingLeft) || 0) + 'px';
  layer.style.right = (parseFloat(cs.paddingRight) || 0) + 'px';
  layer.style.height = '14px';
  const cnt = n <= 16 ? n : Math.min(8, n);
  for (let k = 0; k < cnt; k++) {
    const i = (cnt === n) ? k : Math.min(n - 1, Math.max(0, Math.round((k + 0.5) * n / cnt - 0.5)));
    const lab = el('div', { class: 'ax-x' }, _shortDate(labels[i]));
    lab.style.left = (((i + 0.5) / n) * 100).toFixed(3) + '%';
    layer.appendChild(lab);
  }
  host.appendChild(layer);
}

/* Drag-to-zoom an x-range on an index-spaced line chart (Plotly-style), with
   double-click or the "⟲ reset" chip to restore. The drag maps svg px → plot
   fraction → [i0, i1]; the CALLER re-renders itself with the sliced points
   (its own innerHTML='' wipes this layer + svg listeners — no teardown).
   spec = {W,H,padL,padR,n}; handlers = {zoomed, onZoom(i0,i1), onReset|null}.
   A <8px drag counts as a click; a <3-point range is ignored. */
function attachZoom(host, spec, handlers) {
  const { W, H, padL, padR, n } = spec;
  if (!n || n < 3) return;
  const svg = host.querySelector('svg');
  if (!svg) return;
  const layer = el('div', { class: 'chart-zoom' });
  host.appendChild(layer);
  _pinOverlay(layer, host, H);
  const sel = el('div', { class: 'zoom-sel' });
  sel.hidden = true;
  layer.appendChild(sel);
  if (handlers.zoomed) {
    const chip = el('span', { class: 'zoom-reset' }, '⟲ reset');
    chip.addEventListener('click', () => handlers.onReset && handlers.onReset());
    layer.appendChild(chip);
  }
  const fracOf = (clientX) => {
    const r = svg.getBoundingClientRect();
    const vx = ((clientX - r.left) / (r.width || 1)) * W;
    return Math.min(1, Math.max(0, (vx - padL) / ((W - padL - padR) || 1)));
  };
  let dragFrom = null;
  svg.addEventListener('mousedown', (e) => {
    dragFrom = { x: e.clientX, f: fracOf(e.clientX) };
    e.preventDefault();
  });
  svg.addEventListener('mousemove', (e) => {
    if (dragFrom == null) return;
    const f0 = Math.min(dragFrom.f, fracOf(e.clientX));
    const f1 = Math.max(dragFrom.f, fracOf(e.clientX));
    const plotL = padL / W * 100, plotW = (W - padL - padR) / W * 100;
    sel.hidden = false;
    sel.style.left = (plotL + f0 * plotW).toFixed(3) + '%';
    sel.style.width = ((f1 - f0) * plotW).toFixed(3) + '%';
  });
  const endDrag = (e) => {
    if (dragFrom == null) return;
    const moved = Math.abs(e.clientX - dragFrom.x);
    const f0 = Math.min(dragFrom.f, fracOf(e.clientX));
    const f1 = Math.max(dragFrom.f, fracOf(e.clientX));
    dragFrom = null;
    sel.hidden = true;
    if (moved < 8) return;
    const i0 = Math.round(f0 * (n - 1)), i1 = Math.round(f1 * (n - 1));
    if (i1 - i0 >= 2) handlers.onZoom(i0, i1);
  };
  svg.addEventListener('mouseup', endDrag);
  svg.addEventListener('mouseleave', (e) => { if (dragFrom) endDrag(e); });
  if (handlers.onReset) svg.addEventListener('dblclick', () => handlers.onReset());
}

/* Hover crosshair + tooltip for index-spaced line charts. Reuses the S1
   overlay idiom (a layer pinned to the svg's H-px box). spec = {
     W,H,padL,padR,padT,padB, yLo,yHi,   // same geometry/scale the drawer plotted
     n,                                   // index-domain size
     points,                              // reference points -> date at index (p.x ?? p.t)
     series: [{label, color, values:[per-index], fmt}]  // one per line
   }. Snaps to nearest index; null/non-finite value -> "—", no dot. */
function attachCrosshair(host, spec) {
  const { W, H, padL, padR, padT, padB, yLo, yHi, n, points, series } = spec;
  if (!n || n < 1 || !series || !series.length) return;
  host.style.position = 'relative';
  const svg = host.querySelector('svg');
  if (!svg) return;
  const layer = el('div', { class: 'chart-crosshair' });
  _pinOverlay(layer, host, H);
  const rule = el('div', { class: 'cx-rule' });
  const tip = el('div', { class: 'cx-tip' });
  rule.style.display = 'none';
  tip.style.display = 'none';
  layer.appendChild(rule);
  layer.appendChild(tip);
  const dots = series.map((s) => {
    const d = el('div', { class: 'cx-dot' });
    d.style.background = s.color;
    d.style.display = 'none';
    layer.appendChild(d);
    return d;
  });
  host.appendChild(layer);
  const yPix = (v) => padT + (1 - (v - yLo) / ((yHi - yLo) || 1)) * (H - padT - padB);
  const xPixI = (i) => padL + (i / ((n - 1) || 1)) * (W - padL - padR);
  const hide = () => {
    rule.style.display = 'none';
    tip.style.display = 'none';
    dots.forEach((d) => { d.style.display = 'none'; });
  };
  svg.addEventListener('mouseleave', hide);
  svg.addEventListener('mousemove', (e) => {
    const r = svg.getBoundingClientRect();
    if (r.width <= 0) return;
    const vbx = (e.clientX - r.left) / r.width * W;
    let idx = Math.round((vbx - padL) / ((W - padL - padR) || 1) * (n - 1));
    idx = Math.max(0, Math.min(n - 1, idx));
    const leftPct = xPixI(idx) / W * 100;
    rule.style.left = leftPct.toFixed(2) + '%';
    rule.style.display = 'block';
    const pt = points[idx] || {};
    const date = pt.x != null ? pt.x : (pt.t != null ? pt.t : '');
    tip.innerHTML = '';
    tip.appendChild(el('div', { class: 'cx-date' }, String(date)));
    series.forEach((s) => {
      const v = s.values[idx];
      const txt = (v == null || !isFinite(v)) ? '—' : s.fmt(v);
      const row = el('div', { class: 'cx-row' }, [
        el('span', { class: 'cx-sw', style: 'background:' + s.color }),
      ]);
      if (s.label) row.appendChild(el('span', { class: 'cx-lab' }, s.label));
      row.appendChild(el('span', { class: 'cx-val' }, txt));
      tip.appendChild(row);
    });
    tip.style.display = 'block';
    if (leftPct > 60) { tip.style.right = (100 - leftPct).toFixed(2) + '%'; tip.style.left = 'auto'; }
    else { tip.style.left = leftPct.toFixed(2) + '%'; tip.style.right = 'auto'; }
    const th = tip.offsetHeight || 40;   // clamp by the tooltip's own height so it can't spill below the plot
    tip.style.top = Math.max(0, Math.min(H - th - 2, (e.clientY - r.top) - 10)) + 'px';
    dots.forEach((d, k) => {
      const v = series[k].values[idx];
      if (v == null || !isFinite(v)) { d.style.display = 'none'; return; }
      d.style.left = leftPct.toFixed(2) + '%';
      d.style.top = (yPix(v) / H * 100).toFixed(2) + '%';
      d.style.display = 'block';
    });
  });
}

/* ---- Shared per-mark hover tooltip (bars / points / segments / cells) ---- */
let _mkTipEl = null;
function _markTipEl() {
  if (!_mkTipEl) {
    _mkTipEl = el('div', { class: 'mk-tip' });
    _mkTipEl.style.display = 'none';
    document.body.appendChild(_mkTipEl);
  }
  return _mkTipEl;
}
function _positionMarkTip(t, e) {
  const pad = 12;
  let x = e.clientX + pad, y = e.clientY + pad;
  if (x + t.offsetWidth > window.innerWidth - 4) x = e.clientX - t.offsetWidth - pad;
  if (y + t.offsetHeight > window.innerHeight - 4) y = e.clientY - t.offsetHeight - pad;
  t.style.left = x + 'px';
  t.style.top = y + 'px';
}
/* Styled hover tip (at the cursor) for one discrete chart mark. content = plain string. */
function attachMarkTip(markEl, content) {
  if (!markEl || content == null) return;
  markEl.addEventListener('mouseenter', (e) => {
    const t = _markTipEl();
    t.textContent = content;
    t.style.display = 'block';
    _positionMarkTip(t, e);
  });
  markEl.addEventListener('mousemove', (e) => { if (_mkTipEl && _mkTipEl.style.display !== 'none') _positionMarkTip(_mkTipEl, e); });
  markEl.addEventListener('mouseleave', () => { if (_mkTipEl) _mkTipEl.style.display = 'none'; });
}
/* Nearest-point tip for a dense cloud — one delegated listener, not one-per-point.
   points = [{bx,py}]; spec = {W,H,padL,padR,padT,padB, xmin,xmax,ymin,ymax, labelFn}. */
function attachScatterTip(svg, points, spec) {
  if (!svg || !points || !points.length) return;
  const { W, H, padL, padR, padT, padB, xmin, xmax, ymin, ymax, labelFn } = spec;
  const X = (v) => padL + ((v - xmin) / ((xmax - xmin) || 1)) * (W - padL - padR);
  const Y = (v) => padT + (1 - (v - ymin) / ((ymax - ymin) || 1)) * (H - padT - padB);
  const px = points.map((p) => ({ p, x: X(p.bx), y: Y(p.py) }));
  svg.addEventListener('mousemove', (e) => {
    const r = svg.getBoundingClientRect();
    if (r.width <= 0) return;
    const vbx = (e.clientX - r.left) / r.width * W;
    const vby = (e.clientY - r.top) / r.height * H;
    let best = null, bestD = 225;   // ~15 viewBox-unit snap radius²
    for (let i = 0; i < px.length; i++) {
      const dx = px[i].x - vbx, dy = px[i].y - vby, d = dx * dx + dy * dy;
      if (d < bestD) { bestD = d; best = px[i].p; }
    }
    const t = _markTipEl();
    if (best) { t.textContent = labelFn(best); t.style.display = 'block'; _positionMarkTip(t, e); }
    else t.style.display = 'none';
  });
  svg.addEventListener('mouseleave', () => { if (_mkTipEl) _mkTipEl.style.display = 'none'; });
}

/* Overlay N line series sharing one y-scale (generalizes drawAreaChart).
   series = [{name,key?,color,dash?,points:[{x,<key>:v}]}]
   opts   = {key, baseline, height, fillFirst?, fillEach?} */
function drawOverlayChart(host, series, opts) {
  host.innerHTML = '';
  if (!series || !series.length || !series.some((s) => (s.points || []).length)) {
    host.appendChild(el('div', { class: 'empty-state' }, 'No series.')); return;
  }
  const W = 900, H = opts.height || 240, padL = 4, padR = 4, padT = 12, padB = 12;
  const n = Math.max(...series.map((s) => (s.points || []).length));
  const [lo, hi] = _boundsMulti(series, opts.key, opts.baseline);
  const x = (i) => padL + (i / (n - 1 || 1)) * (W - padL - padR);
  const y = (v) => padT + (1 - (v - lo) / (hi - lo || 1)) * (H - padT - padB);
  const svg = svgEl('svg', { viewBox: `0 0 ${W} ${H}`, width: '100%',
    height: String(H), preserveAspectRatio: 'none', style: 'display:block' });
  // gridlines (quartiles)
  [0.25, 0.5, 0.75].forEach((f) => svg.appendChild(svgEl('line',
    { x1: 0, x2: W, y1: padT + f * (H - padT - padB), y2: padT + f * (H - padT - padB),
      stroke: '#303C4B' })));
  // baseline (dashed) once
  if (opts.baseline != null) svg.appendChild(svgEl('line',
    { x1: 0, x2: W, y1: y(opts.baseline), y2: y(opts.baseline),
      stroke: '#4C5866', 'stroke-dasharray': '2 3' }));
  const base = y(opts.baseline != null ? opts.baseline : lo);
  series.forEach((s, si) => {
    const pts = s.points || [];
    if (!pts.length) return;
    // Null / non-finite points are GAPS (rolling betas skip structurally-empty
    // windows): break the polyline there so the line doesn't dive to a fake 0,
    // mirroring Plotly. Series with no gaps render exactly as before (one
    // segment, fill intact).
    const finite = pts.map((p) => { const v = p[opts.key]; return v != null && isFinite(v); });
    const hasGap = finite.some((f) => !f);
    const doFill = !hasGap && (opts.fillEach || (opts.fillFirst && si === 0));
    if (doFill) {
      const line = pts.map((p, i) => `${x(i).toFixed(1)},${y(p[opts.key]).toFixed(1)}`).join(' ');
      svg.appendChild(svgEl('path', {
        d: `M${x(0)},${base} L${line.replace(/ /g, ' L')} L${x(pts.length - 1)},${base} Z`,
        fill: _tint(s.color, 0.16),
      }));
    }
    let seg = [];
    const flush = () => {
      if (seg.length >= 2) svg.appendChild(svgEl('polyline', {
        points: seg.join(' '), fill: 'none', stroke: s.color,
        'stroke-width': String(s.width || 2), 'stroke-dasharray': s.dash ? '4 3' : null,
      }));
      seg = [];
    };
    pts.forEach((p, i) => {
      if (!finite[i]) { flush(); return; }
      const xy = `${x(i).toFixed(1)},${y(p[opts.key]).toFixed(1)}`;
      if (p.provisional && seg.length) {
        // Provisional tail (interim stub, spec 2026-08-22): the solid line stops
        // at the last statement point; this segment is dashed amber with a
        // hollow marker. Index-aligned with the series, so the x-axis holds.
        const prev = seg[seg.length - 1];
        flush();
        svg.appendChild(svgEl('polyline', { points: prev + ' ' + xy, fill: 'none',
          stroke: '#E0A030', 'stroke-width': '2', 'stroke-dasharray': '5 4' }));
        svg.appendChild(svgEl('circle', { cx: x(i).toFixed(1),
          cy: y(p[opts.key]).toFixed(1), r: '4', fill: '#0F141D',
          stroke: '#E0A030', 'stroke-width': '2' }));
        return;
      }
      seg.push(xy);
    });
    flush();
  });
  host.appendChild(svg);
  attachAxes(host, {
    W, H, padL, padR, padT, padB, yLo: lo, yHi: hi,
    yfmt: opts.yfmt || axNum,
    x: opts.xdates === false ? null
       : { kind: 'date', points: (series.find((s) => (s.points || []).length) || {}).points || [] },
  });
  attachCrosshair(host, {
    W, H, padL, padR, padT, padB, yLo: lo, yHi: hi, n,
    points: (series.find((s) => (s.points || []).length) || {}).points || [],
    series: series.map((s) => ({ label: s.name || '', color: s.color,
      values: (s.points || []).map((p) => p[opts.key]), fmt: opts.yfmt || axNum })),
  });
  // drag-zoom: slice every series by the shared index (they're date-aligned).
  const zoomFull = opts._full || series;
  attachZoom(host, { W, H, padL, padR, n }, {
    zoomed: !!opts._full,
    onZoom: (i0, i1) => drawOverlayChart(host,
      series.map((s) => Object.assign({}, s, { points: (s.points || []).slice(i0, i1 + 1) })),
      Object.assign({}, opts, { _full: zoomFull })),
    onReset: opts._full
      ? () => drawOverlayChart(host, zoomFull, Object.assign({}, opts, { _full: null }))
      : null,
  });
}

/* Diverging grouped bars about a horizontal zero mid-line.
   periods = {port:[{x,v}], bench:[{x,v}]} — two bars per period index. */
function drawGroupedBars(host, periods, opts) {
  host.innerHTML = '';
  const port = (periods && periods.port) || [];
  const bench = (periods && periods.bench) || [];
  if (!port.length && !bench.length) {
    host.appendChild(el('div', { class: 'empty-state' }, 'No periods.')); return;
  }
  const maxAbs = Math.max(1e-9,
    ...port.map((b) => Math.abs(b.v)), ...bench.map((b) => Math.abs(b.v)));
  const wrap = el('div', { class: 'bars-diverge' });
  const n = Math.max(port.length, bench.length);
  for (let i = 0; i < n; i++) {
    const grp = el('div', { class: 'bar-grp' });
    [[port[i], '#4DA3F5'], [bench[i], '#6B7786']].forEach(([b, color]) => {
      if (!b) { grp.appendChild(el('div', { class: 'vb-cell' })); return; }
      const h = Math.max(2, (Math.abs(b.v) / maxAbs) * 50); // half-height each side
      const up = b.v >= 0;
      const bar = el('div', {
        class: 'vb-diverge ' + (up ? 'up' : 'down'),
        style: `height:${h}%;background:${color}`,
      });
      attachMarkTip(bar, b.x + ': ' + (b.v >= 0 ? '+' : '') + b.v.toFixed(1) + '%');
      grp.appendChild(el('div', { class: 'vb-cell' }, bar));
    });
    wrap.appendChild(grp);
  }
  const card = el('div', { class: 'diverge-host' }, [
    el('div', { class: 'diverge-mid' }), wrap,
  ]);
  host.appendChild(card);
  if (opts && opts.axis) {
    attachDivYAxis(card, { kind: 'diverging', max: maxAbs, yfmt: opts.axis.yfmt || axPct });
  }
  attachDivXLabels(card, (port.length ? port : bench).map((b) => b.x));
}

/* Single-series diverging bars about a zero mid-line; green = beat SPY,
   red = trailed. spread = [{x,v}]. */
function drawSpreadBars(host, spread) {
  host.innerHTML = '';
  const rows = spread || [];
  if (!rows.length) {
    host.appendChild(el('div', { class: 'empty-state' }, 'No spread.')); return;
  }
  const maxAbs = Math.max(1e-9, ...rows.map((b) => Math.abs(b.v)));
  const wrap = el('div', { class: 'bars-diverge' });
  rows.forEach((b) => {
    const h = Math.max(2, (Math.abs(b.v) / maxAbs) * 50);
    const up = b.v >= 0;
    const grp = el('div', { class: 'bar-grp' });
    const bar = el('div', {
      class: 'vb-diverge ' + (up ? 'up' : 'down'),
      style: `height:${h}%;background:${up ? 'var(--gain)' : 'var(--loss)'}`,
    });
    attachMarkTip(bar, b.x + ': ' + (b.v >= 0 ? '+' : '') + b.v.toFixed(1) + '%');
    grp.appendChild(el('div', { class: 'vb-cell' }, bar));
    wrap.appendChild(grp);
  });
  const card = el('div', { class: 'diverge-host' }, [
    el('div', { class: 'diverge-mid' }), wrap,
  ]);
  host.appendChild(card);
  attachDivYAxis(card, { kind: 'diverging', max: maxAbs, yfmt: axPct });
  attachDivXLabels(card, rows.map((b) => b.x));
}

function renderBenchPeriodic(per, short) {
  const benchShort = short || 'SPY';
  const root = $('bench-periodic'); root.innerHTML = '';
  const head = el('div', { class: 'ch' }, [
    el('div', { class: 'ctitle' }, 'TWR by period — portfolio vs ' + benchShort),
    el('div', { class: 'csub' }, 'azure portfolio · grey ' + benchShort),
  ]);
  const seg = el('span', { class: 'seg', style: 'margin-left:auto' });
  const grans = ['monthly', 'quarterly', 'yearly'];
  grans.forEach((g, i) => {
    const b = el('button', { class: i === 0 ? 'on' : '' }, g[0].toUpperCase() + g.slice(1));
    b.addEventListener('click', () => {
      seg.querySelectorAll('button').forEach((x) => x.classList.remove('on'));
      b.classList.add('on');
      draw(g);
    });
    seg.appendChild(b);
  });
  head.appendChild(seg);
  root.appendChild(head);

  const grouped = el('div');
  const spreadHost = el('div', { style: 'margin-top:14px' });
  const cap = el('div', { class: 'cap' });
  root.appendChild(grouped);
  root.appendChild(spreadHost);
  root.appendChild(cap);

  function draw(g) {
    const d = (per && per[g]) || {};
    drawGroupedBars(grouped, { port: d.port || [], bench: d.bench || [] }, { axis: { yfmt: axPct } });
    drawSpreadBars(spreadHost, d.spread || []);
    // Mirror app.py's benchmark periodic caption (no win-rate here — win-rate is
    // a headline KPI only); the granularity adjective matches "{period}ly TWR".
    cap.textContent = 'Top: side-by-side ' + g + ' TWR. Bottom: spread ' +
      '(portfolio − ' + benchShort + '); green = beat, red = trailed.';
  }
  draw('monthly');
}

function renderBenchHeadline(cards) {
  const grid = el('div', { class: 'snapshot-grid' });
  (cards || []).forEach((c) => {
    const kids = [
      el('div', { class: 'kpi-label' }, c.label),
      el('div', { class: 'kpi-value ' + (c.color || '') }, c.value),
    ];
    if (c.delta) {
      kids.push(el('div', { class: 'kpi-chip-row' }, [
        buildChip(c.delta, c.delta_dir, false, c.delta_color || c.delta_dir),
      ]));
    }
    kids.push(el('div', { class: 'kpi-sub' }, c.sub || ''));
    grid.appendChild(el('div', { class: 'kpi' }, kids));
  });
  const root = $('bench-headline'); root.innerHTML = ''; root.appendChild(grid);
}

function renderBenchDdTrio(trio, short) {
  const root = $('bench-dd-trio'); root.innerHTML = '';
  if (!trio) return;
  const benchShort = short || 'SPY';
  const grid = el('div', { class: 'flow-grid' });
  [['Portfolio worst drawdown', trio.port], [benchShort + ' worst drawdown', trio.bench],
   ['Spread', trio.spread]].forEach(([label, t]) => {
    if (!t) return;
    grid.appendChild(el('div', { class: 'kpi' }, [
      el('div', { class: 'kpi-label' }, label),
      el('div', { class: 'kpi-value', style: 'font-size:21px' }, t.value),
      el('div', { class: 'kpi-sub' }, t.sub),
    ]));
  });
  root.appendChild(grid);
}

function renderBenchReturns(rt) {
  const host = $('bench-returns');
  host.innerHTML = '';
  if (!rt || !rt.rows) { host.hidden = true; return; }
  host.hidden = false;
  const pct = (v) => (v == null ? '—' : (v >= 0 ? '+' : '') + (v * 100).toFixed(2) + '%');
  const vpct = (v) => (v == null ? '—' : (v * 100).toFixed(2) + '%');   // vol: unsigned
  const head = el('div', { class: 'tr-row tr-head' }, [
    el('span', { class: 'tr-cell tr-period' }, 'Period'),
    el('span', { class: 'tr-cell' }, 'Portfolio'),
    el('span', { class: 'tr-cell' }, 'Vol'),
    el('span', { class: 'tr-cell' }, rt.bench_label),
    el('span', { class: 'tr-cell' }, rt.bench_label + ' vol'),
    el('span', { class: 'tr-cell' }, 'Vol spread'),
    el('span', { class: 'tr-cell' }, 'Return spread'),
  ]);
  const bcap = $('bench-stub-caption');
  if (bcap) { bcap.textContent = rt.caption || ''; bcap.hidden = !rt.caption; }
  const rows = rt.rows.map((r) => {
    const lbl = r.label + (r.annualized ? ' (ann.)' : '')
      + (r.provisional ? ' · to ' + String(r.to_date).slice(5) + ' · prov.' : '');
    if (!r.available) {
      return el('div', { class: 'tr-row tr-na' }, [
        el('span', { class: 'tr-cell tr-period' }, lbl),
        el('span', { class: 'tr-cell', title: 'window not fully covered' }, '—'),
        el('span', { class: 'tr-cell' }, '—'),
        el('span', { class: 'tr-cell' }, '—'),
        el('span', { class: 'tr-cell' }, '—'),
        el('span', { class: 'tr-cell' }, '—'),
        el('span', { class: 'tr-cell' }, '—'),
      ]);
    }
    // With a provisional stub the return columns show the to-date figures
    // (statement values stay in the payload); vol columns are statement-only.
    const port = r.provisional ? r.port_to_date : r.port;
    const bench = r.provisional ? r.bench_to_date : r.bench;
    const spread = r.provisional ? r.spread_to_date : r.spread;
    const sc = spread >= 0 ? 'tint-good' : 'tint-bad';
    const vspread = (r.port_vol == null || r.bench_vol == null)
      ? null : r.port_vol - r.bench_vol;
    const vsc = vspread == null ? '' : (vspread <= 0 ? 'tint-good' : 'tint-bad');
    return el('div', { class: 'tr-row' + (r.provisional ? ' tr-prov' : '') }, [
      el('span', { class: 'tr-cell tr-period' }, lbl),
      el('span', { class: 'tr-cell' }, pct(port)),
      el('span', { class: 'tr-cell' }, vpct(r.port_vol)),
      el('span', { class: 'tr-cell' }, pct(bench)),
      el('span', { class: 'tr-cell' }, vpct(r.bench_vol)),
      el('span', { class: 'tr-cell ' + vsc }, pct(vspread)),
      el('span', { class: 'tr-cell ' + sc }, pct(spread)),
    ]);
  });
  host.appendChild(el('div', { class: 'tr-table' }, [head, ...rows]));
}

function renderBenchmark(data) {
  ensureFilterSelects(data.meta);

  const bm = (data.meta && data.meta.benchmark) || { id: 'spy', short: 'SPY', label: 'SPY (S&P 500 TR)' };
  const bsel = $('bench-benchmark-select');
  if (bsel && !bsel._wired) {
    bsel.addEventListener('change', function () {
      _benchState.bench = bsel.value;
      fetchTab('benchmark', currentQuery());
    });
    bsel._wired = true;
  }
  if (bsel) bsel.value = _benchState.bench;
  $('bench-title').textContent = 'Performance vs ' + bm.short;
  $('bench-growth-note').textContent = 'portfolio vs ' + bm.short + ' total return';
  $('bench-dd-note').textContent = 'peak-to-trough — portfolio vs ' + bm.short;
  $('bench-periodic-note').textContent = 'TWR by period — portfolio vs ' + bm.short;
  const cap = $('bench-benchmark-cap');
  if (cap) {
    if (_benchState.bench === 'auto' && bm.id === '60_40') {
      cap.textContent = 'Auto → 60/40 · this scope is majority fixed income';
    } else if (bm.unavailable_fallback) {
      cap.textContent = '60/40 unavailable (AGG data not loaded) — showing SPY';
    } else {
      cap.textContent = bm.label;
    }
  }

  $('bench-lede').textContent = data.meta.filter_caption || '';
  $('bench-explainer').innerHTML =
    (data.disclosures && data.disclosures.methodology) || '';

  // subset / holdings-filter callout
  const filt = $('bench-filter');
  if (data.meta.holdings_filter_active) {
    filt.hidden = false; filt.innerHTML = '';
    filt.appendChild(el('span', { class: 'callout-icon' }, 'ℹ'));
    filt.appendChild(el('span', { class: 'callout-text' }, data.meta.subset_label || ''));
  } else { filt.hidden = true; }

  // non-ok states: show a callout, clear every body section, bail out
  const empty = $('bench-empty');
  if (data.meta.state !== 'ok') {
    const isError = data.meta.message_level === 'error';
    empty.className = 'callout ' + (isError ? 'callout-warn' : 'callout-blue');
    empty.hidden = false; empty.innerHTML = '';
    empty.appendChild(el('span', { class: 'callout-icon' }, isError ? '⚠' : 'ℹ'));
    empty.appendChild(el('span', { class: 'callout-text' }, data.message || ''));
    const bcap0 = $('bench-stub-caption');
    if (bcap0) { bcap0.hidden = true; bcap0.textContent = ''; }
    ['bench-headline', 'bench-growth', 'bench-drawdown', 'bench-periodic', 'bench-dd-trio', 'bench-returns', 'bench-ai']
      .forEach((id) => { $(id).innerHTML = ''; });
    $('bench-window-note').textContent = '';
    return;
  }
  empty.hidden = true;

  const w = data.meta.window || {};
  $('bench-window-note').textContent =
    w.n_months + ' months (' + Number(w.years).toFixed(2) + 'y) · ' + w.start + ' → ' + w.end;

  renderBenchHeadline(data.headline);
  renderBenchReturns(data.returns_table);
  chartCard($('bench-growth'), data.growth.head, (slot) =>
    overlayWithLegend(slot, data.growth.series,
      { key: 'v', baseline: data.growth.base, height: 240, fillFirst: true, yfmt: axUsd }));
  renderBenchDdTrio(data.drawdown.trio, bm.short);
  chartCard($('bench-drawdown'), data.drawdown.head, (slot) =>
    overlayWithLegend(slot, data.drawdown.series,
      { key: 'dd', baseline: 0, height: 160, fillEach: true, yfmt: axPct }));
  renderBenchPeriodic(data.periodic, bm.short);
  mountAiBox('bench-ai', 'benchmark', () => ({ benchmark: _benchState.bench }));
}

/* ============ DATA HEALTH RENDER ============ */
const HEALTH_CALLOUT = {
  success: 'callout-health', warning: 'callout-warn',
  error: 'callout-error', muted: 'callout-muted', info: 'callout-blue',
};
const HEALTH_ICON = {
  success: '✓', warning: '⚠', error: '✗', muted: '⌖', info: 'ℹ',
};
const HEALTH_STATE_PILL = {
  'Verified': 'hpill-ok', 'Missing': 'hpill-error',
  'Carried forward': 'hpill-carried',
};
const HEALTH_VERDICT_PILL = {
  'ok': 'hpill-ok', 'known': 'hpill-known', 'watch': 'hpill-watch',
  'error': 'hpill-error', 'carried': 'hpill-carried',
  'carried (lagging)': 'hpill-carried',
};

function healthPill(text, cls) {
  return el('span', { class: 'hpill ' + (cls || 'hpill-carried') }, text);
}

function renderHealthSummary(summary, reconAvailable) {
  const wrap = $('health-summary');
  wrap.innerHTML = '';
  if (!reconAvailable) { wrap.hidden = true; return; }
  wrap.hidden = false;
  const chips = [['n_ok', 'verified', 'hpill-ok', true],
                 ['n_known', 'known', 'hpill-known', false],
                 ['n_watch', 'watch', 'hpill-watch', false],
                 ['n_error', 'off-band', 'hpill-error', false],
                 ['n_carried', 'carried', 'hpill-carried', false]];
  chips.forEach(([key, word, cls, always]) => {
    const n = summary[key] || 0;
    if (!always && !n) return;
    wrap.appendChild(healthPill(n + ' ' + word, cls));
  });
}

function renderHealthTable(rows) {
  const table = $('health-table');
  table.innerHTML = '';
  const cols = [['Account', 'l'], ['Broker', 'l'], ['State', 'l'],
                ['Last verified', 'l'], ['Extracted', 'r'], ['Reported', 'r'],
                ['Δ$', 'r'], ['Δ%', 'r'], ['Verdict', 'l']];
  const thead = el('thead'), htr = el('tr');
  cols.forEach(([l, a]) => htr.appendChild(el('th', { class: a }, l)));
  thead.appendChild(htr); table.appendChild(thead);
  const tb = el('tbody');
  (rows || []).forEach((r) => {
    const tr = el('tr');
    tr.appendChild(el('td', { class: 'l sym' }, r['Account']));
    tr.appendChild(el('td', { class: 'l' }, r['Broker']));
    tr.appendChild(el('td', { class: 'l' },
      healthPill(r['State'], HEALTH_STATE_PILL[r['State']])));
    tr.appendChild(el('td', { class: 'l muted' }, r['Last verified']));
    tr.appendChild(el('td', { class: 'num' }, r['Extracted']));
    tr.appendChild(el('td', { class: 'num' }, r['Reported']));
    tr.appendChild(el('td', { class: 'num' }, r['Δ$']));
    tr.appendChild(el('td', { class: 'num' }, r['Δ%']));
    tr.appendChild(el('td', { class: 'l' },
      healthPill(r['Verdict'], HEALTH_VERDICT_PILL[r['Verdict']])));
    tb.appendChild(tr);
  });
  table.appendChild(tb);
  // "Last verified" renders "Jun 2026" (_pretty_month) -- lossy for a plain
  // text sort (month abbreviations aren't alphabetical-order == calendar-
  // order); _monYearKey recovers the real month. Every other column already
  // parses cleanly as number or text.
  makeSortable(table, {
    key: (td, colIndex) => (colIndex === 3 ? _monYearKey(td.textContent) : undefined),
  });
}

function renderDataHealth(data) {
  ensureFilterSelects(data.meta);

  // verdict callout
  const cw = $('health-verdict');
  const level = (data.headline && data.headline.level) || 'muted';
  cw.className = 'callout ' + (HEALTH_CALLOUT[level] || 'callout-muted');
  cw.innerHTML = '';
  cw.appendChild(el('span', { class: 'callout-icon' }, HEALTH_ICON[level] || '⌖'));
  // strip a leading glyph the engine text already carries (avoid doubling)
  const text = String((data.headline && data.headline.text) || '')
    .replace(/^[✓⚠⚑⌖✗ℹ]\s*/, '');
  cw.appendChild(el('span', { class: 'callout-text' }, text));

  // count strip
  renderHealthSummary(data.summary || {}, data.meta.recon_available);

  // recon note
  $('health-recon-note').textContent = data.meta.as_of_month
    ? ('latest statement month · ' + data.meta.as_of_month) : '';

  // table or empty-state message
  const empty = $('health-empty');
  if (data.table && data.table.length) {
    empty.hidden = true;
    renderHealthTable(data.table);
  } else {
    $('health-table').innerHTML = '';
    if (data.message) {
      empty.hidden = false;
      empty.className = 'callout ' + (HEALTH_CALLOUT[data.message.level] || 'callout-blue');
      empty.innerHTML = '';
      empty.appendChild(el('span', { class: 'callout-icon' },
        HEALTH_ICON[data.message.level] || 'ℹ'));
      empty.appendChild(el('span', { class: 'callout-text' }, data.message.text));
    } else {
      empty.hidden = true;
    }
  }
}

/* ============ INCOME RENDER ============ */

/* "$1,234" / "-$8" from a number (used by chart tooltips). */
function _fmtUsd(v) {
  const n = Math.round(Math.abs(Number(v) || 0));
  return (v < 0 ? '-$' : '$') + n.toLocaleString('en-US');
}

/* horizontal swatch+name legend (reuses the alloc-account footer styling) */
/* Interactive legend (Plotly-style): click an item to hide/show its series;
   redrawVisible(visibleSubset) re-renders the chart, so the drawer's own
   bounds rescale to what's shown. Hidden items dim. The legend host must sit
   OUTSIDE the node the drawer wipes. Hidden state is per-render — a seg/view
   switch rebuilds everything visible, like a fresh Plotly figure. */
function toggleLegend(legendHost, series, redrawVisible) {
  legendHost.innerHTML = '';
  const hidden = new Set();
  const foot = el('div', { class: 'legend-foot' });
  const redraw = () => redrawVisible((series || []).filter((_, i) => !hidden.has(i)));
  (series || []).forEach((s, i) => {
    const item = el('span', { class: 'legend-foot-item legend-toggle' }, [
      el('span', { class: 'legend-swatch', style: 'background:' + s.color }),
      s.name,
    ]);
    item.addEventListener('click', () => {
      if (hidden.has(i)) { hidden.delete(i); item.classList.remove('legend-off'); }
      else { hidden.add(i); item.classList.add('legend-off'); }
      redraw();
    });
    foot.appendChild(item);
  });
  legendHost.appendChild(foot);
  redraw();
}

/* overlay chart + clickable series legend in ONE host — the toggleLegend
   idiom for call sites that draw straight into a container div. Every
   multi-series overlay gets the same click-to-isolate UX (TK 2026-07-19). */
function overlayWithLegend(host, series, opts) {
  host.innerHTML = '';
  const box = el('div');
  const legend = el('div');
  host.appendChild(box); host.appendChild(legend);
  toggleLegend(legend, series, (vis) => drawOverlayChart(box, vis, opts));
}

/* generic KPI-card grid (label / value / sub) */
function renderKpiCards(host, cards) {
  const grid = el('div', { class: 'snapshot-grid' });
  (cards || []).forEach((c) => {
    grid.appendChild(el('div', { class: 'kpi' }, [
      el('div', { class: 'kpi-label' }, c.label),
      el('div', { class: 'kpi-value ' + (c.color || '') }, c.value),
      el('div', { class: 'kpi-sub' }, c.sub || ''),
    ]));
  });
  host.innerHTML = ''; host.appendChild(grid);
}

/* Signed vertical stacked bars about a zero baseline. view = {x, series:
   [{name,color,values:[...]}]}. Positive segments stack up from zero, negative
   stack down — withholding (negative) reads below the line, mirroring the
   Streamlit relative-barmode chart. Axis ticks are wired via attachAxes and
   per-segment hover via attachMarkTip (see the tail of this function). */
function drawStackedBars(host, view) {
  host.innerHTML = '';
  const x = view.x || [];
  const series = view.series || [];
  const fmt = view.fmt || _fmtUsd;   // attribution passes a pp formatter
  if (!x.length || !series.some((s) => (s.values || []).some((v) => v))) {
    host.appendChild(el('div', { class: 'empty-state' }, view.emptyMsg || 'No income in range.')); return;
  }
  const n = x.length;
  let hi = 0, lo = 0;
  for (let i = 0; i < n; i++) {
    let pos = 0, neg = 0;
    series.forEach((s) => { const v = (s.values[i] || 0); if (v >= 0) pos += v; else neg += v; });
    hi = Math.max(hi, pos); lo = Math.min(lo, neg);
  }
  if (hi === lo) { hi = (hi || 1); lo = Math.min(0, lo); }
  const W = 900, H = 260, padL = 4, padR = 4, padT = 12, padB = 12;
  const y = (v) => padT + (1 - (v - lo) / (hi - lo || 1)) * (H - padT - padB);
  const svg = svgEl('svg', { viewBox: `0 0 ${W} ${H}`, width: '100%',
    height: String(H), preserveAspectRatio: 'none', style: 'display:block' });
  [0.25, 0.5, 0.75].forEach((f) => svg.appendChild(svgEl('line',
    { x1: 0, x2: W, y1: padT + f * (H - padT - padB), y2: padT + f * (H - padT - padB),
      stroke: '#303C4B' })));
  svg.appendChild(svgEl('line', { x1: 0, x2: W, y1: y(0), y2: y(0),
    stroke: '#4C5866', 'stroke-dasharray': '2 3' }));
  const slot = (W - padL - padR) / n;
  const bw = Math.max(2, Math.min(46, slot * 0.62));
  for (let i = 0; i < n; i++) {
    const cx = padL + slot * (i + 0.5);
    let posOff = 0, negOff = 0;
    series.forEach((s) => {
      const v = s.values[i] || 0;
      if (!v) return;
      let y0, y1;
      if (v > 0) { y0 = y(posOff + v); y1 = y(posOff); posOff += v; }
      else { y0 = y(negOff); y1 = y(negOff + v); negOff += v; }
      const top = Math.min(y0, y1), h = Math.max(1, Math.abs(y1 - y0));
      const rect = svgEl('rect', { x: (cx - bw / 2).toFixed(1), y: top.toFixed(1),
        width: bw.toFixed(1), height: h.toFixed(1), fill: s.color });
      attachMarkTip(rect, x[i] + ' · ' + s.name + ': ' + fmt(v));
      svg.appendChild(rect);
    });
  }
  host.appendChild(svg);
  attachAxes(host, {
    W, H, padL, padR, padT, padB, yLo: lo, yHi: hi,
    yfmt: fmt,                                    // per-caller: income $ / attribution pp
    x: { kind: 'date', points: x.map((s) => ({ x: s })), bar: true },
  });
}

/* Single-series labeled vertical bars for the top-payers chart. */
function drawIncomeTopBars(host, bars) {
  host.innerHTML = '';
  if (!bars || !bars.length) {
    host.appendChild(el('div', { class: 'empty-state' }, 'No payers.')); return;
  }
  const maxV = Math.max(1e-9, ...bars.map((b) => b.v));
  const wrap = el('div', { class: 'topbars' });
  bars.forEach((b) => {
    const h = Math.max(2, (b.v / maxV) * 100);
    const col = el('div', { class: 'topbar-col' }, [
      el('div', { class: 'topbar-track' }, [
        el('div', { class: 'topbar-fill', style: `height:${h}%` }),
      ]),
      el('div', { class: 'topbar-lab' }, b.x),
    ]);
    attachMarkTip(col, b.x + ' · ' + _fmtUsd(b.v));
    wrap.appendChild(col);
  });
  host.appendChild(wrap);
  attachDivYAxis(wrap, { kind: 'baseline', max: maxV, yfmt: axUsd });
}

/* Income-received chart card: two segmented controls (View × Split) swap the
   precomputed series client-side (stacked bars or — for cumulative — lines). */
function segOn(seg, btn) {
  seg.querySelectorAll('button').forEach((x) => x.classList.remove('on'));
  btn.classList.add('on');
}

function renderIncomeReceivedChart(chart) {
  const card = $('income-received-chart');
  card.hidden = false; card.innerHTML = '';
  const head = el('div', { class: 'ch' }, [
    el('div', { class: 'ctitle' }, 'Income over time'),
    el('div', { class: 'csub' }, 'gross dividends + interest, net of withholding'),
  ]);
  let view = 'monthly', split = 'components';
  const viewSeg = el('span', { class: 'seg', style: 'margin-left:auto' });
  ['monthly', 'yearly', 'cumulative'].forEach((g, i) => {
    const b = el('button', { class: i === 0 ? 'on' : '' }, g[0].toUpperCase() + g.slice(1));
    b.addEventListener('click', () => { segOn(viewSeg, b); view = g; draw(); });
    viewSeg.appendChild(b);
  });
  const splitSeg = el('span', { class: 'seg', style: 'margin-left:8px' });
  [['components', 'Components'], ['by_account', 'By account']].forEach(([k, lab], i) => {
    const b = el('button', { class: i === 0 ? 'on' : '' }, lab);
    b.addEventListener('click', () => { segOn(splitSeg, b); split = k; draw(); });
    splitSeg.appendChild(b);
  });
  head.appendChild(viewSeg); head.appendChild(splitSeg);
  card.appendChild(head);
  const legend = el('div'); card.appendChild(legend);
  const slot = el('div'); card.appendChild(slot);

  function draw() {
    const v = ((chart || {})[split] || {})[view] || { x: [], series: [], mode: 'stacked' };
    if (v.mode === 'line') {
      const conv = (v.series || []).map((s) => ({
        name: s.name, color: s.color, width: s.width,
        points: (v.x || []).map((xi, i) => ({ x: xi, v: s.values[i] })),
      }));
      toggleLegend(legend, conv, (vis) =>
        drawOverlayChart(slot, vis, { key: 'v', baseline: 0, height: 260, yfmt: axUsd }));
    } else {
      toggleLegend(legend, v.series, (vis) =>
        drawStackedBars(slot, { x: v.x, series: vis, fmt: v.fmt, emptyMsg: v.emptyMsg }));
    }
  }
  draw();
}

function renderIncomeDetail(table, detail) {
  table.innerHTML = '';
  if (!detail) return;
  const cols = detail.columns || [];
  const thead = el('thead'), htr = el('tr');
  cols.forEach((c, i) => htr.appendChild(el('th', { class: i === 0 ? 'l' : 'r' }, c)));
  thead.appendChild(htr); table.appendChild(thead);
  const tb = el('tbody');
  (detail.rows || []).forEach((row) => {
    const tr = el('tr');
    cols.forEach((c, i) => {
      tr.appendChild(el('td', { class: i === 0 ? 'l sym' : 'num' }, row[c]));
    });
    tb.appendChild(tr);
  });
  table.appendChild(tb);
  makeSortable(table);
}

function _incomeCallout(node, message) {
  node.hidden = false; node.innerHTML = '';
  node.appendChild(el('span', { class: 'callout-icon' }, 'ℹ'));
  node.appendChild(el('span', { class: 'callout-text' }, message || ''));
}

function renderIncome(data) {
  ensureFilterSelects(data.meta);
  $('income-caption').textContent = data.caption || '';
  $('income-methodology').innerHTML = data.methodology || '';
  mountAiBox('income-ai', 'income');   // B3: whole-book income posture

  // ---- Section A: income received ----
  const r = data.received || {};
  const rEmpty = $('income-received-empty');
  if (r.empty) {
    _incomeCallout(rEmpty, r.empty_message);
    $('income-received-kpis').innerHTML = '';
    $('income-received-chart').hidden = true;
    $('income-received-chart').innerHTML = '';
  } else {
    rEmpty.hidden = true;
    renderKpiCards($('income-received-kpis'), r.kpis);
    renderIncomeReceivedChart(r.chart);
  }

  // ---- Section B: forward income ----
  const f = data.forward || {};
  const fUnavail = $('income-forward-unavailable');
  const topCard = $('income-top');
  const detail = $('income-detail');
  if (!f.available) {
    _incomeCallout(fUnavail, f.unavailable_message);
    $('income-forward-kpis').innerHTML = '';
    $('income-nav-caption').textContent = '';
    $('income-payers-empty').hidden = true;
    topCard.hidden = true; topCard.innerHTML = '';
    detail.innerHTML = '';
    $('income-history-caption').textContent = '';
    return;
  }
  fUnavail.hidden = true;
  renderKpiCards($('income-forward-kpis'), f.kpis);
  $('income-nav-caption').textContent = f.nav_caption || '';

  const payersEmpty = $('income-payers-empty');
  if (f.payers_empty) {
    _incomeCallout(payersEmpty, f.payers_message);
    topCard.hidden = true; topCard.innerHTML = '';
    detail.innerHTML = '';
  } else {
    payersEmpty.hidden = true;
    topCard.hidden = false; topCard.innerHTML = '';
    topCard.appendChild(el('div', { class: 'ch' }, [
      el('div', { class: 'ctitle' }, 'Top projected payers'),
      el('div', { class: 'csub' }, 'largest projected 12-month income'),
    ]));
    const slot = el('div'); topCard.appendChild(slot);
    drawIncomeTopBars(slot, (f.top_chart || {}).bars);
    renderIncomeDetail(detail, f.detail);
  }
  $('income-history-caption').textContent = f.history_through_caption || '';
  const acts = $('income-actions');
  if (acts) { acts.innerHTML = ''; acts.appendChild(actButton('dividends', '⟳ Refresh dividend history')); }
}

/* ============ TAX RENDER ============ */

/* Client-side filter state for the open-lots table (the global account
   picker lists IRAs, which a tax view excludes by design — so tax carries
   its own in-tab account/type/term/evidence filters over the shipped
   lots). type opens on individual stocks — TK's default view
   (2026-07-31); 'All types' is one click away. */
let taxData = null;
let taxEstimate = null;       // last /api/tax/estimate payload
let taxOverrides = null;      // session-only profile overrides, or null
let taxSimSel = new Map();    // lot_id -> qty (Task 6 populates)
let _taxEstTimer = null;
const taxSel = { account: 'all', type: 'stock', term: 'all',
                 evidence: 'all' };
/* expanded (account, instrument) rollups — survives redraws/filter flips
   so an opened instrument stays open while its lots remain in view */
const taxExpanded = new Set();
/* 'lots' | 'harvest' | 'realized' — one fetch, client toggle (all views
   ride on the same /api/tax payload). */
let taxView = 'lots';

/* Whole-dollar money for the KPI tiles (TK round 3: no cents in the
   headline numbers). Table cells and note lines keep cents — lots are
   reconciled to the cent and the tiles are read at a glance. */
function _taxUsd0(v) {
  if (v == null) return '—';
  const sign = v < 0 ? '-' : '';
  return sign + '$' + Math.round(Math.abs(v)).toLocaleString('en-US');
}

/* One predicate for the Type filter in BOTH draw paths. 'tlh' selects
   the tax-loss-harvest account's lots (an ACCOUNT fact, served as the
   is_tlh boolean); the three instrument buckets exclude them, so
   "Individual stocks" never mixes in the TLH sleeve (TK round 3,
   filter-only — the type field itself stays instrument truth). */
function _taxTypeMatch(r) {
  if (taxSel.type === 'all') return true;
  if (taxSel.type === 'tlh') return !!r.is_tlh;
  return r.type === taxSel.type && !r.is_tlh;
}

function _taxUsd(v) {
  if (v == null) return '—';
  const sign = v < 0 ? '-' : '';
  return sign + '$' + Math.abs(v).toLocaleString('en-US',
    { minimumFractionDigits: 2, maximumFractionDigits: 2 });
}

function _taxSelect(host, key, label, options) {
  const wrap = el('label', { class: 'fctl-group' }, [
    el('span', { class: 'fctl-label' }, label),
  ]);
  const sel = el('select');
  options.forEach(([id, lab]) => {
    const o = el('option', { value: id }, lab);
    if (taxSel[key] === id) o.selected = true;
    sel.appendChild(o);
  });
  sel.addEventListener('change', () => {
    taxSel[key] = sel.value;
    drawTaxTable();
  });
  wrap.appendChild(sel);
  host.appendChild(wrap);
}

function _taxViewToggle(host) {
  const wrap = el('label', { class: 'fctl-group' }, [
    el('span', { class: 'fctl-label' }, 'View'),
  ]);
  const seg = el('span', { class: 'seg' });
  [['lots', 'Open lots'], ['harvest', 'Harvest'],
   ['realized', 'Realized YTD']].forEach(([id, lab]) => {
    const b = el('button', { class: taxView === id ? 'on' : '' }, lab);
    b.addEventListener('click', () => {
      if (taxView === id) return;
      taxView = id;
      // the whole control row changes with the view (Harvest hides the
      // evidence filter — it is always reconstructed), so rebuild it too
      buildTaxControls();
      drawTaxTable();
    });
    seg.appendChild(b);
  });
  wrap.appendChild(seg);
  host.appendChild(wrap);
}

function buildTaxControls() {
  const host = $('tax-controls');
  host.innerHTML = '';
  _taxViewToggle(host);
  // Harvest is scoped to its own candidate rows, so its account list must
  // come from THOSE rows — offering an account with no candidates would
  // silently empty the table.
  const accts = [['all', 'All accounts']].concat(
    taxView === 'harvest'
      ? [...new Map((taxHarvest().candidates || [])
          .map((c) => [c.account_id, c.account_label])).entries()]
      : taxView === 'realized'
        ? [...new Map((taxRealized().by_account || [])
            .map((x) => [x.account_id, x.account_label])).entries()]
        : (taxData.summary.accounts || []).map((a) => [a.id, a.label]));
  // A refetch (e.g. global broker narrowing) or a view switch can drop the
  // account the in-tab filter pointed at; a stale id would silently empty
  // the table while the select shows "All accounts" — reset instead.
  if (!accts.some(([id]) => id === taxSel.account)) taxSel.account = 'all';
  _taxSelect(host, 'account', 'Account', accts);
  // instrument type is server-classified from positions asset_class —
  // shown for lots and harvest (harvest candidates carry type too);
  // realized rows have no instrument dimension, so it's skipped there.
  if (taxView !== 'realized') {
    _taxSelect(host, 'type', 'Type',
               [['all', 'All types'], ['stock', 'Individual stocks'],
                ['etf', 'ETFs'], ['tlh', 'Tax loss harvesting'],
                ['other', 'Other']]);
  }
  _taxSelect(host, 'term', 'Term', [['all', 'All terms'], ['long', 'Long'],
                                    ['short', 'Short'],
                                    ['unknown', 'Unknown']]);
  // evidence filter is meaningless in Harvest: candidates are
  // reconstructed-only by construction (printed-evidence lots are excluded
  // and counted in the strip)
  if (taxView === 'lots') {
    _taxSelect(host, 'evidence', 'Basis evidence',
               [['all', 'All evidence'], ['reconstructed', 'Reconstructed'],
                ['printed', 'Printed']]);
  }
}

/* One row per (account, instrument) — TK 2026-07-30: 691 flat lot rows
   were unnavigable. Expanding a rollup shows its lots inline. */
const TAX_GROUP_COLS = [
  ['symbol', 'Symbol', 'l'], ['account_label', 'Account', 'l'],
  ['lots', 'Lots', 'r'], ['quantity', 'Qty', 'r'],
  ['basis', 'Basis', 'r'], ['price', 'Price', 'r'],
  ['market_value', 'Market value', 'r'],
  ['unrealized_gl', 'Unrealized G/L', 'r'],
  ['days_to_long_term', 'Days to LT', 'r'],
  ['basis_evidence', 'Evidence', 'l'], ['band', 'Band', 'l'],
];

const TAX_BAND_RANK = { error: 0, qty_mismatch: 1, watch: 2,
                        reported_unknown: 3 };

const TAX_HARVEST_COLS = [
  ['account_label', 'Account', 'l'], ['symbol', 'Symbol', 'l'],
  ['term', 'Term', 'l'], ['quantity_remaining', 'Qty', 'r'],
  ['basis_remaining', 'Basis', 'r'], ['price', 'Price', 'r'],
  ['market_value', 'Value', 'r'], ['unrealized_gl', 'Unrl G/L', 'r'],
  ['wash_status', 'Wash', 'l'], ['window_ends', 'Window ends', 'l'],
];

function taxHarvest() {
  return (taxData && taxData.harvest) || {};
}

function taxRealized() {
  return ((taxData || {}).summary || {}).realized_ytd || {};
}

/* The realized view's named unavailable reason, or null. `by_account`
   is absent only on the degrade path OF A MATCHED SERVICE — a payload
   missing it with no reason set means this page is newer than the
   service it is talking to (static assets reload per request, the
   Python module does not). Saying "no realized closes" there would
   assert something the build never computed. */
function _realizedUnavailable() {
  const rz = taxRealized();
  if (rz.unavailable) return rz.unavailable;
  if (!Array.isArray(rz.by_account)) {
    return 'this page is newer than the server it is talking to — the '
      + 'payload carries no per-account realized detail. Restart the '
      + 'terminal so the service matches the page.';
  }
  return null;
}

/* One blocking buy, as the row's own evidence: which account, when, how
   much, and what kind. The quantity matters — a fractional-share dividend
   reinvestment vetoing a large harvest is a very different fact from a
   real second purchase, and without the number they look identical. */
function _washWhy(b) {
  const qty = b.quantity == null ? '?'
    : Number(b.quantity).toLocaleString('en-US', { maximumFractionDigits: 4 });
  return (b.is_ira ? '⛔ IRA ' : '') + b.date + ' · ' + b.account_id
    + ' · ' + qty + ' ' + (b.transaction_type || '');
}

function _washCell(r) {
  const td = el('td', { class: 'l' });
  const blocked = r.wash_status === 'blocked';
  const cls = !blocked ? 'wash wash-clear'
    : (r.is_ira_blocked ? 'wash wash-ira' : 'wash wash-blocked');
  td.appendChild(el('span', { class: cls },
    blocked ? (r.is_ira_blocked ? 'BLOCKED · IRA' : 'BLOCKED') : 'CLEAR'));
  (r.blocking_buys || []).forEach((b) => {
    td.appendChild(el('span', {
      class: 'wash-why' + (b.is_ira ? ' wash-why-ira' : ''),
    }, _washWhy(b)));
  });
  if (r.is_ira_blocked) {
    td.appendChild(el('span', { class: 'wash-why wash-why-ira' },
      'IRA replacement — loss is destroyed, not deferred'));
  }
  return td;
}

function drawHarvestTable() {
  const table = $('tax-lots');
  table.innerHTML = '';
  const h = taxHarvest();
  // an empty table under "harvest unavailable" would read as "nothing to
  // harvest", which is the opposite of what happened — show no table
  if (h.unavailable) { $('tax-lots-note').textContent = ''; return; }
  const all = h.candidates || [];
  const rows = all.filter((r) =>
    (taxSel.account === 'all' || r.account_id === taxSel.account)
    && _taxTypeMatch(r)
    && (taxSel.term === 'all' || r.term === taxSel.term));

  const thead = el('thead'), htr = el('tr');
  TAX_HARVEST_COLS.forEach(([, lab, side]) =>
    htr.appendChild(el('th', { class: side }, lab)));
  thead.appendChild(htr);
  table.appendChild(thead);

  const tb = el('tbody');
  rows.forEach((r) => {
    const tr = el('tr');
    TAX_HARVEST_COLS.forEach(([key, , side]) => {
      const v = r[key];
      if (key === 'wash_status') {
        tr.appendChild(_washCell(r));
      } else if (key === 'unrealized_gl') {
        // every candidate is a loss by construction, so this column is
        // always tint-bad — the tint marks the column's meaning, not a
        // sign test that could never go the other way
        tr.appendChild(el('td', { class: 'num tint-bad' }, _taxUsd(v)));
      } else if (key === 'basis_remaining' || key === 'price'
                 || key === 'market_value') {
        tr.appendChild(el('td', { class: 'num' }, _taxUsd(v)));
      } else if (key === 'quantity_remaining') {
        tr.appendChild(el('td', { class: 'num' },
          v == null ? '—' : Number(v).toLocaleString('en-US',
            { maximumFractionDigits: 6 })));
      } else if (key === 'symbol') {
        tr.appendChild(el('td', { class: 'l sym' },
          v || r.instrument_key || '—'));
      } else {
        tr.appendChild(el('td', { class: side === 'r' ? 'num' : 'l' },
          v == null ? '—' : String(v)));
      }
    });
    tb.appendChild(tr);
  });

  const tr = el('tr', { class: 'total-row' });
  const totLoss = rows.reduce((s, r) => s + (r.unrealized_gl || 0), 0);
  TAX_HARVEST_COLS.forEach(([key], i) => {
    if (i === 0) tr.appendChild(el('td', { class: 'l' }, 'Total'));
    else if (key === 'unrealized_gl') {
      tr.appendChild(el('td', { class: 'num tint-bad' }, _taxUsd(totLoss)));
    } else tr.appendChild(el('td', {}, ''));
  });
  tb.appendChild(tr);
  table.appendChild(tb);

  const blocked = rows.filter((r) => r.wash_status === 'blocked').length;
  $('tax-lots-note').textContent = rows.length + ' of ' + all.length
    + ' candidate lot(s)' + (blocked ? ' · ' + blocked + ' blocked' : '')
    + ' · deepest loss first';
  // Wash is a status chip plus a variable-length list of blocking-buy
  // evidence lines, not one comparable value -- no sort affordance.
  // Natural order (deepest loss first, the server's own sort) is left
  // alone: makeSortable only acts once a header is clicked.
  makeSortable(table, {
    skip: [TAX_HARVEST_COLS.findIndex((c) => c[0] === 'wash_status')],
  });
}

/* The Harvest strip. Every number here is the service's own count — the
   point is that the reader can see what the scan is SILENT about (rows it
   never looked at) and how little of the wash window it could observe. */
function _harvestHonesty() {
  const h = taxHarvest(), s = h.summary || {}, sem = h.semantics || {};
  const skipped = [];
  if (s.excluded_printed_evidence) {
    skipped.push(s.excluded_printed_evidence + ' printed-evidence');
  }
  if (s.excluded_unpriced) skipped.push(s.excluded_unpriced + ' unpriced');
  if (s.excluded_ira_accounts) {
    skipped.push(s.excluded_ira_accounts + ' IRA-held');
  }
  if (s.excluded_unknown_accounts) {
    skipped.push(s.excluded_unknown_accounts + ' of unprovable account type');
  }
  if (s.excluded_gain_or_flat) {
    skipped.push(s.excluded_gain_or_flat + ' at or above cost');
  }
  const bits = ['Candidates are drawn from the reconstructed slice of the '
    + 'ledger only.'];
  if (skipped.length) {
    bits.push('Of ' + s.lots_seen + ' open lots, ' + skipped.join(', ')
      + ' were not scanned.');
  }
  if (sem.window_observed_pct != null) {
    bits.push('The wash guard sees history through ' + sem.tx_frontier
      + ' — ' + sem.window_days_observed + ' of the '
      + sem.window_days_total + ' backward-window days ('
      + sem.window_observed_pct + '%) are observable, so CLEAR means "'
      + sem.clear_means + '".');
  } else if (sem.clear_means) {
    bits.push('CLEAR means "' + sem.clear_means + '".');
  }
  if (sem.stale_note) bits.push('⚠ ' + sem.stale_note + '.');
  return bits.join(' ');
}

function _syncTaxChrome() {
  const harvest = taxView === 'harvest';
  const realized = taxView === 'realized';
  const rz = taxRealized();
  $('tax-view-title').textContent = harvest ? 'Harvest candidates'
    : realized ? ('Realized YTD' + (rz.year != null ? ' ' + rz.year : ''))
      : 'Open lots';
  $('tax-harvest-exp').hidden = !harvest;
  const strip = $('tax-harvest-honesty');
  // element id predates the third view: it is the ACTIVE view's named
  // unavailable callout (harvest or realized), never both at once
  const unavailable = harvest ? taxHarvest().unavailable
    : realized ? _realizedUnavailable() : null;
  const bad = $('tax-harvest-unavailable');
  if ((harvest || realized) && unavailable) {
    bad.hidden = false;
    bad.innerHTML = '';
    bad.appendChild(el('span', { class: 'callout-icon' }, '⚠'));
    bad.appendChild(el('span', { class: 'callout-text' }, unavailable));
  } else bad.hidden = true;
  if (harvest && !taxHarvest().unavailable) {
    strip.hidden = false;
    strip.textContent = _harvestHonesty();
  } else strip.hidden = true;
}

function _taxUglCell(v, extra) {
  // tint-good/tint-bad are the styled table-cell tints in app.css;
  // they are declared after table.tbl td.num, so they actually win
  // (bare 'gain'/'loss' on a td matches no table rule at all)
  const cls = v == null ? 'num'
    : 'num ' + (v > 0 ? 'tint-good' : (v < 0 ? 'tint-bad' : ''));
  return el('td', { class: cls + (extra || '') }, _taxUsd(v));
}

function _taxQty(v) {
  return v == null ? '—' : Number(v).toLocaleString('en-US',
    { maximumFractionDigits: 6 });
}

function _taxPriceCell(price, source, extra) {
  // a dagger marks a statement mark (month-end price, not live) — the
  // note line under the table defines it with its as-of month
  const txt = price == null ? '—'
    : _taxUsd(price) + (source === 'statement' ? '†' : '');
  return el('td', { class: 'num' + (extra || '') }, txt);
}

const RIPENING_DAYS = 60;  // ST->LT: highlight a short-term GAIN lot within
                           // this many days of long-term (TK 2026-08-04)
function _taxRipening(r) {
  return r.term === 'short'
    && r.days_to_long_term != null && r.days_to_long_term <= RIPENING_DAYS
    && r.unrealized_gl != null && r.unrealized_gl > 0;
}

function drawTaxTable() {
  _syncTaxChrome();
  if (taxView === 'harvest') { drawHarvestTable(); return; }
  if (taxView === 'realized') { drawRealizedTable(); return; }
  const table = $('tax-lots');
  table.innerHTML = '';
  const all = (taxData && taxData.lots) || [];
  const rows = all.filter((r) =>
    (taxSel.account === 'all' || r.account_id === taxSel.account)
    && _taxTypeMatch(r)
    && (taxSel.term === 'all' || r.term === taxSel.term)
    && (taxSel.evidence === 'all' || r.basis_evidence === taxSel.evidence));

  const thead = el('thead'), htr = el('tr');
  htr.appendChild(el('th', { class: 'c sel-col', 'data-sort-label': '' },
    ''));
  TAX_GROUP_COLS.forEach(([, lab, side]) =>
    htr.appendChild(el('th', { class: side }, lab)));
  thead.appendChild(htr);
  table.appendChild(thead);

  // instrument rollups over the FILTERED lots (the aggregates must sum
  // exactly what the filters show, so grouping is client-side)
  const groups = new Map();
  rows.forEach((r) => {
    const k = r.account_id + '\u0000' + r.instrument_key;
    if (!groups.has(k)) groups.set(k, []);
    groups.get(k).push(r);
  });

  const tb = el('tbody');
  const sorted = [...groups.entries()].sort((a, b) => {
    const ra = a[1][0], rb = b[1][0];
    const ka = ra.account_label + '\u0000' + (ra.symbol || ra.instrument_key);
    const kb = rb.account_label + '\u0000' + (rb.symbol || rb.instrument_key);
    return ka.localeCompare(kb);
  });
  sorted.forEach(([k, mine]) => {
    const priced = mine.filter((r) => r.market_value != null);
    const ev = new Set(mine.map((r) => r.basis_evidence));
    const band = mine.map((r) => r.band).sort((a, b) =>
      (TAX_BAND_RANK[a] ?? 9) - (TAX_BAND_RANK[b] ?? 9))[0] || '—';
    const first = mine[0];
    const open = taxExpanded.has(k);
    const g = {
      symbol: first.symbol || first.instrument_key || '—',
      account_label: first.account_label,
      lots: mine.length,
      quantity: mine.reduce((s, r) => s + (r.quantity_remaining || 0), 0),
      basis: mine.reduce((s, r) => s + (r.basis_remaining || 0), 0),
      price: first.price,
      price_source: first.price_source,
      market_value: priced.length
        ? priced.reduce((s, r) => s + r.market_value, 0) : null,
      unrealized_gl: priced.length
        ? priced.reduce((s, r) => s + (r.unrealized_gl || 0), 0) : null,
      days_to_long_term: (() => {
        // non-null days_to_long_term <=> term === 'short' (the
        // lot_engine.days_to_long_term invariant); filter on term
        // explicitly so this rollup min never leans on that coupling.
        const ds = mine.filter((r) => r.term === 'short')
                       .map((r) => r.days_to_long_term)
                       .filter((d) => d != null);
        return ds.length ? Math.min(...ds) : null;
      })(),
    };
    const tr = el('tr', { class: 'tax-grp' + (open ? ' open' : '')
      + (mine.some(_taxRipening) ? ' tax-ripening' : '') });
    tr.addEventListener('click', () => {
      if (taxExpanded.has(k)) taxExpanded.delete(k);
      else taxExpanded.add(k);
      drawTaxTable();
    });
    const selTd = el('td', { class: 'c sel-col' });
    const gcb = el('input', { type: 'checkbox' });
    const simmable = mine.filter((r) => r.market_value != null
      && (r.term === 'short' || r.term === 'long'));
    if (!simmable.length) {
      gcb.disabled = true;
      gcb.title = 'no simmable lots in this group — unpriced or '
        + 'unknown-term';
    } else {
      const on = simmable.filter((r) => taxSimSel.has(r.lot_id));
      gcb.checked = on.length === simmable.length;
      gcb.indeterminate = on.length > 0 && on.length < simmable.length;
    }
    gcb.addEventListener('click', (e) => {
      e.stopPropagation();
      const allSelected = simmable.every((r) => taxSimSel.has(r.lot_id));
      simmable.forEach((r) => {
        if (allSelected) taxSimSel.delete(r.lot_id);
        else taxSimSel.set(r.lot_id, r.quantity_remaining);
      });
      drawTaxTable();
      fetchTaxEstimate();
    });
    selTd.appendChild(gcb);
    tr.appendChild(selTd);
    TAX_GROUP_COLS.forEach(([key, , side]) => {
      if (key === 'symbol') {
        const td = el('td', { class: 'l sym' });
        td.appendChild(el('span', { class: 'chev' }, '›'));
        td.appendChild(document.createTextNode(g.symbol));
        tr.appendChild(td);
      } else if (open && key !== 'account_label' && key !== 'lots') {
        // replace-in-place (TK 2026-07-31): an open rollup keeps only its
        // identity cells (symbol + account + lots count); every aggregate
        // cell goes blank so the per-lot rows below don't visually blend
        // into a second row of numbers. Cell classes stay so column
        // alignment holds; collapse restores the aggregates. The value is
        // NOT unknown, only unprinted -- carry it via data-sort so a sort
        // mid-expand doesn't read the row as missing and sink it (FIX 1,
        // slice-1 review). evidence/band aren't on `g`; match what the
        // COLLAPSED cell would show for those two.
        const sortVal = key === 'basis_evidence'
          ? (ev.size === 1 ? [...ev][0] : 'mixed')
          : key === 'band' ? band : g[key];
        tr.appendChild(el('td', { class: side === 'r' ? 'num' : 'l',
          'data-sort': String(sortVal ?? '') }, ''));
      } else if (key === 'unrealized_gl') {
        tr.appendChild(_taxUglCell(g.unrealized_gl));
      } else if (key === 'price') {
        tr.appendChild(_taxPriceCell(g.price, g.price_source));
      } else if (key === 'basis' || key === 'market_value') {
        tr.appendChild(el('td', { class: 'num' }, _taxUsd(g[key])));
      } else if (key === 'quantity') {
        tr.appendChild(el('td', { class: 'num' }, _taxQty(g.quantity)));
      } else if (key === 'lots') {
        tr.appendChild(el('td', { class: 'num' }, String(g.lots)));
      } else if (key === 'basis_evidence') {
        tr.appendChild(el('td', { class: 'l' },
          ev.size === 1 ? [...ev][0] : 'mixed'));
      } else if (key === 'band') {
        tr.appendChild(el('td', { class: 'l' }, band));
      } else if (key === 'days_to_long_term') {
        tr.appendChild(el('td', { class: 'num' },
          g.days_to_long_term != null ? String(g.days_to_long_term) : ''));
      } else {
        tr.appendChild(el('td', { class: 'l' }, String(g[key] ?? '—')));
      }
    });
    tb.appendChild(tr);

    if (open) {
      mine.forEach((r) => {
        const dtr = el('tr', { class: 'tax-det'
          + (_taxRipening(r) ? ' tax-ripening' : '') });
        dtr.appendChild(el('td', { class: 'l' },
          '· ' + (r.acquired_date || 'no date')));
        dtr.appendChild(el('td', { class: 'l' },
          r.term + (r.origin && r.origin !== 'purchase'
                    ? ' · ' + r.origin : '')));
        dtr.appendChild(el('td', { class: 'num' }, ''));
        dtr.appendChild(el('td', { class: 'num' },
          _taxQty(r.quantity_remaining)));
        dtr.appendChild(el('td', { class: 'num' },
          _taxUsd(r.basis_remaining)));
        dtr.appendChild(_taxPriceCell(r.price, r.price_source));
        dtr.appendChild(el('td', { class: 'num' },
          _taxUsd(r.market_value)));
        dtr.appendChild(_taxUglCell(r.unrealized_gl));
        dtr.appendChild(el('td', { class: 'num' },
          r.days_to_long_term != null ? String(r.days_to_long_term) : ''));
        dtr.appendChild(el('td', { class: 'l' }, r.basis_evidence || '—'));
        dtr.appendChild(el('td', { class: 'l' }, r.band || '—'));
        const dtd = el('td', { class: 'c sel-col' });
        const canSim = r.market_value != null
          && (r.term === 'short' || r.term === 'long');
        const cb = el('input', { type: 'checkbox' });
        cb.checked = taxSimSel.has(r.lot_id);
        cb.disabled = !canSim;
        cb.title = canSim ? '' : 'unpriced or unknown-term — cannot simulate';
        cb.addEventListener('click', (e) => {
          e.stopPropagation();
          if (taxSimSel.has(r.lot_id)) taxSimSel.delete(r.lot_id);
          else taxSimSel.set(r.lot_id, r.quantity_remaining);
          drawTaxTable();
          fetchTaxEstimate();
        });
        dtd.appendChild(cb);
        if (taxSimSel.has(r.lot_id)
            && taxSimSel.get(r.lot_id) !== r.quantity_remaining) {
          dtd.appendChild(el('span', { class: 'sel-part' }, '½'));
        }
        if (taxSimSel.has(r.lot_id)) {
          const qin = el('input', { type: 'number', class: 'sel-qty',
            value: String(taxSimSel.get(r.lot_id)), min: '0',
            max: String(r.quantity_remaining), step: 'any' });
          qin.addEventListener('click', (e) => e.stopPropagation());
          qin.addEventListener('change', (e) => {
            e.stopPropagation();
            const v = Number(qin.value);
            if (Number.isFinite(v) && v > 0
                && v <= r.quantity_remaining) {
              taxSimSel.set(r.lot_id, v);
              drawTaxTable();
            } else {
              qin.value = String(taxSimSel.get(r.lot_id));
            }
            fetchTaxEstimate();
          });
          dtd.appendChild(qin);
        }
        dtr.insertBefore(dtd, dtr.firstChild);
        tb.appendChild(dtr);
      });
    }
  });

  // totals over the FILTERED rows, at display precision; unmarked lots
  // contribute basis but never a fabricated market value — the note
  // names them so the columns cannot silently disagree
  const priced = rows.filter((r) => r.market_value != null);
  const tr = el('tr', { class: 'total-row' });
  tr.appendChild(el('td', { class: 'c sel-col' }, ''));
  const tot = {
    basis: rows.reduce((s, r) => s + (r.basis_remaining || 0), 0),
    market_value: priced.reduce((s, r) => s + r.market_value, 0),
    unrealized_gl: priced.reduce((s, r) => s + (r.unrealized_gl || 0), 0),
  };
  TAX_GROUP_COLS.forEach(([key], i) => {
    if (i === 0) tr.appendChild(el('td', { class: 'l' }, 'Total'));
    else if (key in tot) {
      const cls = key === 'unrealized_gl'
        ? 'num ' + (tot[key] > 0 ? 'tint-good'
                    : (tot[key] < 0 ? 'tint-bad' : ''))
        : 'num';
      tr.appendChild(el('td', { class: cls }, _taxUsd(tot[key])));
    } else tr.appendChild(el('td', {}, ''));
  });
  tb.appendChild(tr);
  table.appendChild(tb);

  const unpriced = rows.length - priced.length;
  const stmt = rows.filter((r) => r.price_source === 'statement');
  const stmtAsof = stmt.length ? (stmt[0].price_asof || '') : '';
  $('tax-lots-note').textContent = groups.size + ' instrument(s) · '
    + rows.length + ' of ' + all.length + ' lots'
    + (stmt.length ? ' · † ' + stmt.length + ' lot(s) at the '
       + stmtAsof + ' statement mark (no live feed)' : '')
    + (unpriced ? ' · ' + unpriced + ' unmarked lot(s) — no live or '
       + 'statement price; value left blank (likely stale open lots)'
       : '');
  // column 0 is the selection checkbox (Task 6) -- no header label and no
  // sort affordance (skip: [0]). Column 1 (shifted from 0 by that leading
  // column) renders a '›' chevron (the expand affordance) glued onto the
  // symbol with no separator; strip it so a Symbol sort compares the
  // ticker, not the decoration, and so a genuinely symbol-less row ('›—')
  // still reads as missing instead of a stray one-character "text" value.
  // Defaults cover pinned/detail: this view has BOTH .total-row and the
  // expanded rollups' .tax-det lot rows.
  makeSortable(table, {
    skip: [0],
    key: (td, i) => (i === 1 ? td.textContent.replace(/^›/, '').trim() : undefined),
  });
}

const TAX_REALIZED_COLS = [
  ['account_label', 'Account', 'l'], ['term', 'Term', 'l'],
  ['closes', 'Closes', 'r'], ['gains', 'Gains', 'r'],
  ['losses', 'Losses', 'r'], ['net', 'Net', 'r'],
];

function _taxTermLabel(t) {
  return t === 'short' ? 'Short' : t === 'long' ? 'Long'
    : t === 'unknown' ? 'Unknown' : (t || '—');
}

function drawRealizedTable() {
  const table = $('tax-lots');
  table.innerHTML = '';
  const rz = taxRealized();
  // chrome already showed the named unavailable callout — an empty table
  // under it would read as "nothing realized", the claim the build never
  // made (S3b rule)
  if (_realizedUnavailable()) { $('tax-lots-note').textContent = ''; return; }
  const all = rz.by_account;
  const rows = all.filter((x) =>
    (taxSel.account === 'all' || x.account_id === taxSel.account)
    && (taxSel.term === 'all' || x.term === taxSel.term));

  const thead = el('thead'), htr = el('tr');
  TAX_REALIZED_COLS.forEach(([, lab, side]) =>
    htr.appendChild(el('th', { class: side }, lab)));
  thead.appendChild(htr);
  table.appendChild(thead);

  const tb = el('tbody');
  if (!all.length) {
    // a true statement about the year, distinct from the degrade path
    tb.appendChild(el('tr', {}, el('td',
      { class: 'l', colspan: String(TAX_REALIZED_COLS.length) },
      'No realized closes in ' + (rz.year != null ? rz.year : 'this year')
      + ' for taxable accounts in scope.')));
    table.appendChild(tb);
    $('tax-lots-note').textContent = '';
    return;
  }
  rows.forEach((x) => {
    const tr = el('tr');
    TAX_REALIZED_COLS.forEach(([key, , side]) => {
      if (key === 'gains' || key === 'losses' || key === 'net') {
        tr.appendChild(_taxUglCell(x[key]));
      } else if (key === 'closes') {
        tr.appendChild(el('td', { class: 'num' },
          x[key] == null ? '—' : String(x[key])));
      } else if (key === 'term') {
        tr.appendChild(el('td', { class: 'l' }, _taxTermLabel(x.term)));
      } else {
        tr.appendChild(el('td', { class: side === 'r' ? 'num' : 'l' },
          x[key] == null ? '—' : String(x[key])));
      }
    });
    tb.appendChild(tr);
  });

  // totals over the FILTERED rows (Open-lots idiom: aggregates sum what
  // the filters show); at all/all this equals the tiles by construction
  const tot = el('tr', { class: 'total-row' });
  const sum = (k) => rows.reduce((s, x) => s + (x[k] || 0), 0);
  TAX_REALIZED_COLS.forEach(([key], i) => {
    if (i === 0) tot.appendChild(el('td', { class: 'l' }, 'Total'));
    else if (key === 'closes') {
      tot.appendChild(el('td', { class: 'num' }, String(sum('closes'))));
    } else if (key === 'gains' || key === 'losses' || key === 'net') {
      tot.appendChild(_taxUglCell(sum(key)));
    } else tot.appendChild(el('td', {}, ''));
  });
  tb.appendChild(tot);
  table.appendChild(tb);

  $('tax-lots-note').textContent = rows.length + ' of ' + all.length
    + ' account-term row(s)';
  // Reached only past both early returns above (unavailable / no closes
  // this year) -- there is a real row set here, so the sort affordance is
  // never offered over an empty table. Natural order is the server's own
  // sort; defaults cover pinned (.total-row), and this view has no detail
  // rows to group.
  makeSortable(table);
}

function renderTax(data) {
  ensureFilterSelects(data.meta);
  taxData = data;
  const err = $('tax-error'), body = $('tax-body');
  if (data.kind === 'error') {
    err.hidden = false;
    err.innerHTML = '';
    err.appendChild(el('span', { class: 'callout-icon' }, 'ℹ'));
    err.appendChild(el('span', { class: 'callout-text' }, data.reason || ''));
    body.hidden = true;
    $('tax-caption').textContent = '';
    clearTimeout(_taxEstTimer);
    $('tax-estimate').innerHTML = '';
    return;
  }
  err.hidden = true;
  body.hidden = false;
  mountAiBox('tax-ai', 'tax');
  const m = data.meta || {}, s = data.summary || {}, t = s.totals || {};
  $('tax-caption').textContent = 'Open lots from the gated ledger — evidence '
    + 'carried per lot, consumers decide. Taxable accounts only; IRAs are '
    + 'excluded by design.';

  const stale = $('tax-stale');
  if (m.stale) {
    stale.hidden = false;
    stale.innerHTML = '';
    stale.appendChild(el('span', { class: 'callout-icon' }, '⚠'));
    stale.appendChild(el('span', { class: 'callout-text' },
      'Ledger stale: ' + (m.stale_reason || '')));
  } else stale.hidden = true;

  const unrl = t.unrealized_gl;
  // Harvestable tile reuses the numbers the Harvest strip already ships
  const h = taxHarvest(), hs = h.summary || {}, hsem = h.semantics || {};
  const hv = hs.total_unrealized_loss;
  const hOk = typeof hv === 'number' && hv !== 0;
  // Realized YTD is build-time materialized (ledger closes + Harbor option
  // confirms); an older lots_meta.json without the block is the documented
  // degrade path — em-dash tiles with the reason, never zeros
  const r = s.realized_ytd || { unavailable: 'no data' };
  const ru = !!r.unavailable;
  const bt = r.by_term || {};
  const term = (tm, f) => (bt[tm] ? bt[tm][f] : 0);
  const stlt = (f) => 'ST ' + _taxUsd0(term('short', f)) + ' · LT '
    + _taxUsd0(term('long', f));
  // Basis / Market value / Unrealized are one consistent universe (the
  // MARKED lots: mv - priced_basis == unrealized_gl); the unmarked
  // remainder is stated on the Basis card instead of silently deflating
  // market value against an all-lots basis (TK feedback 2026-07-30).
  // All KPIs are broker-picker-scoped and deliberately ignore the in-tab
  // filters (unchanged semantics from the pre-rework tiles).
  renderKpiCards($('tax-kpis'), [
    { label: 'Basis (marked lots)', value: _taxUsd0(t.priced_basis),
      sub: t.unpriced_lots
        ? '+ ' + _taxUsd0(t.unpriced_basis) + ' in ' + t.unpriced_lots
          + ' unmarked lot(s)'
        : 'all lots marked' },
    { label: 'Market value', value: _taxUsd0(t.market_value),
      sub: t.stmt_priced_lots
        ? t.stmt_priced_lots + ' lot(s) at statement marks'
        : '' },
    { label: 'Unrealized G/L', value: _taxUsd0(unrl),
      color: unrl == null ? '' : (unrl > 0 ? 'gain' : (unrl < 0 ? 'loss' : '')),
      sub: 'marked lots only' },
    { label: 'Harvestable losses',
      value: hOk ? _taxUsd0(Math.abs(hv))
        : (h.unavailable ? '—' : _taxUsd0(0)),
      color: hOk ? 'loss' : '',
      sub: h.unavailable ? 'harvest scan unavailable'
        : (hs.candidates || 0) + ' candidate lot(s)'
          + (hsem.window_observed_pct != null
             ? ' · ' + hsem.window_observed_pct + '% of window observed'
             : '') },
    { label: 'Realized YTD gains', value: ru ? '—' : _taxUsd0(r.gains),
      color: !ru && r.gains > 0 ? 'gain' : '',
      // term-unknown appended per tile when nonzero — the tile VALUE
      // includes that bucket, so an ST·LT-only sub would not cross-foot
      sub: ru ? '' : stlt('gains') + (bt.unknown && bt.unknown.gains
        ? ' · term-unknown ' + _taxUsd0(bt.unknown.gains) : '') },
    { label: 'Realized YTD losses', value: ru ? '—' : _taxUsd0(r.losses),
      color: !ru && r.losses < 0 ? 'loss' : '',
      sub: ru ? '' : stlt('losses') + (bt.unknown && bt.unknown.losses
        ? ' · term-unknown ' + _taxUsd0(bt.unknown.losses) : '') },
    { label: 'Realized YTD net', value: ru ? '—' : _taxUsd0(r.net),
      color: ru ? '' : (r.net > 0 ? 'gain' : (r.net < 0 ? 'loss' : '')),
      sub: ru ? r.unavailable
        : stlt('net') + (bt.unknown && bt.unknown.net
            ? ' · term-unknown ' + _taxUsd0(bt.unknown.net) : '') },
  ]);
  fetchTaxEstimate();
  // provenance stated ONCE in the fine print, not per-tile: what the
  // realized figures are made of and what they exclude (the S4b bound)
  $('tax-silent-note').textContent = (s.silent_share_note || '')
    + (ru ? '' : ' Realized YTD ' + r.year + ' at build: ledger closes'
       + ' + Harbor option confirms; excludes Alpine options'
       + (r.options_uncovered ? '; ' + r.options_uncovered
          + ' option close(s) unpriced by confirms' : '')
       + (r.broker_unresolved ? '; ' + r.broker_unresolved
          + ' ledger row(s) dropped as broker-unresolvable at build' : '')
       + (r.unrecognized_slots ? '; ' + r.unrecognized_slots
          + ' unrecognized realized bucket(s) ignored' : '') + '.');

  buildTaxControls();
  drawTaxTable();

  const ex = s.excluded || {};
  const bits = [];
  if (ex.ira_accounts) bits.push(ex.ira_accounts + ' IRA account(s)');
  if (ex.unknown_accounts) {
    bits.push(ex.unknown_accounts + ' account(s) of unprovable type');
  }
  $('tax-excluded-note').textContent = bits.length
    ? 'Excluded from this tax view: ' + bits.join(' and ')
      + ' (their lots stay in the ledger).' : '';
}

/* ============ YEAR TAX ESTIMATE PANEL ============ */

function _taxSimBody() {
  return [...taxSimSel.entries()].map(([lot_id, qty]) =>
    ({ lot_id, qty }));
}

function fetchTaxEstimate() {
  clearTimeout(_taxEstTimer);
  _taxEstTimer = setTimeout(async () => {
    const body = {};
    if (taxOverrides) body.overrides = taxOverrides;
    const sim = _taxSimBody();
    if (sim.length) body.sim = sim;
    try {
      const res = await fetch('/api/tax/estimate', {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
      });
      if (!res.ok) throw new Error('HTTP ' + res.status);
      taxEstimate = await res.json();
    } catch (e) {
      taxEstimate = { kind: 'error',
        reason: 'estimate request failed (' + e.message + ')' };
    }
    renderTaxEstimate(taxEstimate);
  }, 250);
}

function _estRow(label, value, cls) {
  const tr = el('tr', cls ? { class: cls } : {});
  tr.appendChild(el('td', { class: 'l' }, label));
  tr.appendChild(el('td', { class: 'r' }, value));
  return tr;
}

function _estUsd(v) {
  // whole dollars, signed; the engine ships ints
  if (v == null || !Number.isFinite(v)) return '—';
  const sign = v < 0 ? '−' : '';
  return sign + '$' + Math.abs(v).toLocaleString('en-US');
}

function renderTaxEstimate(out) {
  const host = $('tax-estimate');
  host.innerHTML = '';
  if (!out) return;
  const box = el('div', { class: 'tax-est-panel' });
  const yr = out.year || (out.baseline && out.baseline.year) || '';
  box.appendChild(el('h2', { class: 'section-title' },
    'Year tax estimate' + (yr ? ' (' + yr + ')' : '')));
  if (out.kind === 'error') {
    const co = el('div', { class: 'callout callout-blue' });
    co.appendChild(el('span', { class: 'callout-icon' }, 'ℹ'));
    co.appendChild(el('span', { class: 'callout-text' },
      out.reason || ''));
    box.appendChild(co);
    box.appendChild(_estOverrideForm(null));
    host.appendChild(box);
    return;
  }
  const b = out.baseline;
  const tbl = el('table', { class: 'tbl tax-est-tbl' });
  const tb = el('tbody');
  tb.appendChild(_estRow('Federal (ordinary + LT/qualified)',
    _estUsd(b.federal)));
  tb.appendChild(_estRow('California', _estUsd(b.state)));
  tb.appendChild(_estRow('NIIT (3.8%)', _estUsd(b.niit)));
  if (b.ftc) tb.appendChild(_estRow('Foreign withholding credit (approx)',
    '−' + _estUsd(b.ftc).replace('−', '')));
  tb.appendChild(_estRow('Estimated tax attributable to the portfolio',
    _estUsd(b.total), 'total-row'));
  tbl.appendChild(tb);
  // Redesign v3.2b: estimate table left, the simulated-sale card right
  // (.tax-est-cols > .tax-est-main | .tax-est-sim; stacks ≤1100px).
  const cols = el('div', { class: 'tax-est-cols' });
  const main = el('div', { class: 'tax-est-main' });
  cols.appendChild(main);
  main.appendChild(tbl);

  const nt = b.netting;
  main.appendChild(el('div', { class: 'footnote' },
    'Netting: ST ' + _estUsd(Math.round(nt.st_net)) + ' · LT '
    + _estUsd(Math.round(nt.lt_net))
    + (nt.ordinary_offset ? ' · ordinary offset '
       + _estUsd(Math.round(nt.ordinary_offset)) : '')
    + (nt.carryforward_out ? ' · carryforward out '
       + _estUsd(Math.round(nt.carryforward_out)) : '')));
  const ut = b.unknown_term;
  if (ut.amount) {
    main.appendChild(el('div', { class: 'footnote' },
      'Unknown-term gains ' + _estUsd(Math.round(ut.amount))
      + ' taxed as ' + ut.assumption + '-term; if '
      + (ut.assumption === 'long' ? 'short' : 'long') + '-term instead: '
      + (ut.swing_if_other >= 0 ? '+' : '') + _estUsd(ut.swing_if_other)));
  }
  main.appendChild(el('div', { class: 'footnote' },
    'Marginal rates in effect — fed ordinary '
    + (b.marginal.fed_ordinary * 100).toFixed(1) + '% · fed LT '
    + (b.marginal.fed_preferential * 100).toFixed(1) + '% · CA '
    + (b.marginal.ca * 100).toFixed(1) + '%'));

  // Task 6 extends here with the with-sim block (right column of the row).
  if (typeof _estSimBlock === 'function') _estSimBlock(cols, out);
  box.appendChild(cols);

  const pv = out.provenance || {};
  box.appendChild(el('div', { class: 'footnote' },
    'Income through ' + (pv.income_through || 'n/a') + ' · '
    + (pv.table_note || '')
    + (pv.lots_stale ? ' · ⚠ ledger stale: '
       + (pv.stale_reason || '') : '')));
  const det = el('details', { class: 'exp' });
  const sum = el('summary');
  sum.appendChild(el('span', { class: 'chev' }, '›'));
  sum.appendChild(document.createTextNode(
    ' Assumptions — estimate only, not tax advice, '
    + 'not a filing document'));
  det.appendChild(sum);
  const ul = el('ul', { class: 'tax-est-assumptions' });
  (out.assumptions || []).forEach((a) =>
    ul.appendChild(el('li', {}, a)));
  det.appendChild(ul);
  box.appendChild(det);
  box.appendChild(_estOverrideForm(out.profile_used || null));
  host.appendChild(box);
}

/* Sell-simulator with-sim block: selection chip + rejection callouts +
   the added-ST/added-LT/added-tax table + a per-leg footnote. Called from
   renderTaxEstimate (guarded by a typeof check there). The only client
   arithmetic here is display sugar over server ints: summing leg `gl` by
   term for the two "Added ..." rows, and out.with_sim.total minus
   out.baseline.total for the "Added tax" delta -- never a recomputation
   of a server number. */
function _estSimBlock(box, out) {
  const n = taxSimSel.size;
  if (!n && !(out.sim_rejected || []).length) return;
  const wrap = el('div', { class: 'tax-est-sim' });
  const chip = el('div', { class: 'tax-sim-chip' });
  chip.appendChild(el('span', {},
    'Simulated sale — ' + n + ' lot(s)'));
  const clear = el('button', { class: 'btn btn-small' }, 'Clear');
  clear.addEventListener('click', () => {
    taxSimSel.clear();
    drawTaxTable();
    fetchTaxEstimate();
  });
  chip.appendChild(clear);
  wrap.appendChild(chip);
  (out.sim_rejected || []).forEach((rj) => {
    const co = el('div', { class: 'callout callout-warn' });
    co.appendChild(el('span', { class: 'callout-icon' }, '⚠'));
    co.appendChild(el('span', { class: 'callout-text' }, rj));
    wrap.appendChild(co);
  });
  const ws = out.with_sim;
  if (ws) {
    const legs = out.sim_legs || [];
    const st = legs.filter((l) => l.term === 'short')
      .reduce((s, l) => s + l.gl, 0);
    const lt = legs.filter((l) => l.term === 'long')
      .reduce((s, l) => s + l.gl, 0);
    const delta = ws.total - out.baseline.total;
    const tbl = el('table', { class: 'tbl tax-est-tbl' });
    const tb = el('tbody');
    tb.appendChild(_estRow('Added ST realized',
      _estUsd(Math.round(st))));
    tb.appendChild(_estRow('Added LT realized',
      _estUsd(Math.round(lt))));
    tb.appendChild(_estRow('Added tax',
      (delta >= 0 ? '+' : '') + _estUsd(delta)));
    tb.appendChild(_estRow('Year total with simulated sale',
      _estUsd(ws.total), 'total-row'));
    tbl.appendChild(tb);
    wrap.appendChild(tbl);
    legs.forEach((l) => {
      wrap.appendChild(el('div', { class: 'footnote' },
        l.symbol + ' · ' + l.account_label + ' · qty '
        + l.qty + ' · ' + l.term + ' · G/L '
        + _estUsd(Math.round(l.gl))
        + (l.wash_observed
           ? ' · ⚠ recent buy observed — loss may defer'
           : '')));
    });
  }
  box.appendChild(wrap);
}

const _EST_FIELDS = [
  ['filing_status', 'Filing status', 'select',
   ['single', 'married_joint']],
  ['w2_income', 'W-2 income ($)', 'number', null],
  ['state', 'State', 'select', ['CA']],
  ['deduction', 'Deduction ("standard" or $)', 'text', null],
  ['carryforward_loss', 'Prior-year loss carryforward ($)', 'number',
   null],
  ['qualified_dividend_pct', 'Qualified dividend share (0–1)',
   'number', null],
  ['unknown_term_assumption', 'Unknown-term treated as', 'select',
   ['long', 'short']],
];

function _estOverrideForm(profile) {
  const det = el('details', { class: 'exp tax-est-form' });
  const sum = el('summary');
  sum.appendChild(el('span', { class: 'chev' }, '›'));
  sum.appendChild(document.createTextNode(
    ' Profile' + (taxOverrides ? ' (session overrides active)' : '')));
  det.appendChild(sum);
  const grid = el('div', { class: 'tax-est-grid' });
  const inputs = {};
  _EST_FIELDS.forEach(([key, label, kind, opts]) => {
    grid.appendChild(el('label', { for: 'est-' + key }, label));
    let inp;
    const cur = (taxOverrides && taxOverrides[key] != null)
      ? taxOverrides[key]
      : (profile ? profile[key] : '');
    if (kind === 'select') {
      inp = el('select', { id: 'est-' + key });
      opts.forEach((o) => {
        const op = el('option', { value: o }, o);
        if (String(cur) === o) op.selected = true;
        inp.appendChild(op);
      });
    } else {
      inp = el('input', { id: 'est-' + key, type: kind });
      if (kind === 'number') inp.step = 'any';
      inp.value = cur == null ? '' : String(cur);
    }
    inputs[key] = inp;
    grid.appendChild(inp);
  });
  det.appendChild(grid);
  const row = el('div', { class: 'tax-est-actions' });
  const apply = el('button', { class: 'btn' }, 'Apply');
  apply.addEventListener('click', () => {
    const ov = {};
    _EST_FIELDS.forEach(([key, , kind]) => {
      const raw = inputs[key].value;
      if (raw === '' || raw == null) return;
      if (kind === 'number') {
        const n = Number(raw);
        if (Number.isFinite(n)) ov[key] = n;
      } else if (key === 'deduction') {
        const n = Number(raw);
        ov[key] = Number.isFinite(n) && raw.trim() !== '' && raw.trim()
          .toLowerCase() !== 'standard' ? n : 'standard';
      } else ov[key] = raw;
    });
    taxOverrides = Object.keys(ov).length ? ov : null;
    fetchTaxEstimate();
  });
  const reset = el('button', { class: 'btn' }, 'Reset');
  reset.addEventListener('click', () => {
    taxOverrides = null;
    fetchTaxEstimate();
  });
  row.appendChild(apply);
  row.appendChild(reset);
  det.appendChild(row);
  return det;
}

/* ============ FACTOR ANALYSIS RENDER ============ */

/* "+1.2 pp" / "-0.3 pp" tooltip formatter for attribution stacked bars. */
function _fmtPp(v) {
  const n = Number(v) || 0;
  return (n >= 0 ? '+' : '') + n.toFixed(1) + ' pp';
}

/* Vertical attribution waterfall: relative steps (green up / coral down from the
   running cumulative) + a final absolute total bar. Value + factor-name labels
   sit in an HTML row below the width-stretched SVG so the text isn't distorted.
   Y-axis ticks are wired via attachAxes (x is categorical — that label row IS
   the x axis) and per-bar hover via attachMarkTip. */
function drawWaterfall(host, data) {
  host.innerHTML = '';
  const items = (data.items || []).filter((it) => it.value_pp != null);
  if (!items.length || data.total_pp == null) {
    host.appendChild(el('div', { class: 'empty-state' }, 'No attribution.')); return;
  }
  let run = 0;
  const steps = items.map((it) => {
    const start = run, end = run + it.value_pp; run = end;
    return { label: it.label, text: it.text, start, end, up: it.value_pp >= 0 };
  });
  steps.push({ label: data.total_label, text: data.total_text,
    start: 0, end: data.total_pp, total: true });
  let lo = 0, hi = 0;
  steps.forEach((s) => { lo = Math.min(lo, s.start, s.end); hi = Math.max(hi, s.start, s.end); });
  if (lo === hi) hi = lo + 1;
  const W = 900, H = 300, padL = 4, padR = 4, padT = 16, padB = 12;
  const y = (v) => padT + (1 - (v - lo) / (hi - lo || 1)) * (H - padT - padB);
  const svg = svgEl('svg', { viewBox: `0 0 ${W} ${H}`, width: '100%',
    height: String(H), preserveAspectRatio: 'none', style: 'display:block' });
  svg.appendChild(svgEl('line', { x1: 0, x2: W, y1: y(0), y2: y(0),
    stroke: '#4C5866', 'stroke-dasharray': '2 3' }));
  const slot = (W - padL - padR) / steps.length;
  const bw = Math.max(2, Math.min(64, slot * 0.6));
  steps.forEach((s, i) => {
    const cx = padL + slot * (i + 0.5);
    const y0 = y(s.start), y1 = y(s.end);
    const top = Math.min(y0, y1), h = Math.max(1, Math.abs(y1 - y0));
    const color = s.total ? 'var(--accent)' : (s.up ? 'var(--gain)' : 'var(--loss)');
    const rect = svgEl('rect', { x: (cx - bw / 2).toFixed(1), y: top.toFixed(1),
      width: bw.toFixed(1), height: h.toFixed(1), fill: color });
    attachMarkTip(rect, s.label + ': ' + s.text + ' pp');
    svg.appendChild(rect);
  });
  host.appendChild(svg);
  attachAxes(host, {
    W, H, padL, padR, padT, padB, yLo: lo, yHi: hi,
    yfmt: axNum,                                  // percentage-point contributions
    x: null,
  });
  const cols = el('div', { class: 'wf-cols' });
  steps.forEach((s) => {
    cols.appendChild(el('div', { class: 'wf-col' }, [
      el('div', { class: 'wf-val ' + (s.total ? '' : (s.up ? 'gain' : 'loss')) }, s.text),
      el('div', { class: 'wf-name' }, s.label),
    ]));
  });
  host.appendChild(cols);
}

/* generic 1:1 table — first col + any Significant/Metric/Factor/Symbol col is
   left-aligned, the rest right/num. Beta / per-holding / cross-check tables. */
function renderFactorTable(tableEl, spec) {
  tableEl.innerHTML = '';
  if (!spec || !spec.columns) return;
  const cols = spec.columns;
  const isText = (c, i) => i === 0 || /^Significant/.test(c) ||
    c === 'Metric' || c === 'Factor' || c === 'Symbol';
  const thead = el('thead'), htr = el('tr');
  cols.forEach((c, i) => htr.appendChild(el('th', { class: isText(c, i) ? 'l' : 'r' }, c)));
  thead.appendChild(htr); tableEl.appendChild(thead);
  const tb = el('tbody');
  (spec.rows || []).forEach((row) => {
    const tr = el('tr');
    cols.forEach((c, i) => tr.appendChild(
      el('td', { class: isText(c, i) ? (i === 0 ? 'l sym' : 'l') : 'num' }, row[c])));
    tb.appendChild(tr);
  });
  tableEl.appendChild(tb);
  // Shared by all three factor tables (beta / per-holding / cross-check) --
  // one call here covers factor-beta, factor-ph-table, and factor-cc-table.
  // Every column already parses cleanly (plain +/-.2f numbers, or a single
  // "✓"/joined-label/"—" text value with no multi-line structure).
  makeSortable(tableEl);
}

/* alpha-by-model strip: one KPI card per model, the t/R²/CI help on hover. */
function renderFactorStrip(host, strip) {
  const grid = el('div', { class: 'snapshot-grid' });
  (strip || []).forEach((e) => {
    grid.appendChild(el('div', { class: 'kpi', title: e.help || '' }, [
      el('div', { class: 'kpi-label' }, e.model),
      el('div', { class: 'kpi-value' }, e.value),
      el('div', { class: 'kpi-sub' }, e.delta || ''),
    ]));
  });
  host.innerHTML = ''; host.appendChild(grid);
}

/* a labeled segmented-control group for the factor controls bar */
function factorSeg(label, options, current, onPick) {
  const seg = el('span', { class: 'seg' });
  options.forEach((opt) => {
    const val = Array.isArray(opt) ? opt[0] : opt;
    const lab = Array.isArray(opt) ? opt[1] : opt;
    const b = el('button', { class: String(val) === String(current) ? 'on' : '' }, String(lab));
    b.addEventListener('click', () => { segOn(seg, b); onPick(val); });
    seg.appendChild(b);
  });
  return el('div', { class: 'fctl-group' }, [
    el('span', { class: 'fctl-label' }, label), seg,
  ]);
}

let _factorData = null;
const _factorState = { window: null, model: null, attrView: null, roll: null };

function renderFactor(data) {
  ensureFilterSelects(data.meta);
  _factorData = data;
  $('factor-caption').textContent = data.caption || '';
  const st = data.state || {};

  const empty = $('factor-empty');
  const body = $('factor-body');
  if (!st.available) {
    empty.hidden = false; empty.innerHTML = '';
    empty.appendChild(el('span', { class: 'callout-icon' }, 'ℹ'));
    const txt = el('span', { class: 'callout-text' }, st.unavailable_message || '');
    txt.appendChild(el('div', {}, actButton('factor_data', '⟳ Fetch factor data')));
    empty.appendChild(txt);
    $('factor-ai').innerHTML = '';        // no data -> no narration panel
    body.hidden = true;
    return;
  }
  empty.hidden = true; body.hidden = false;
  $('factor-methodology').innerHTML = data.methodology || '';

  // seed control state from defaults (persist a valid prior choice across re-fetch)
  if (!st.windows.includes(_factorState.window)) _factorState.window = st.default_window;
  if (!st.models.includes(_factorState.model)) _factorState.model = st.default_model;
  if (!st.roll_windows.map(String).includes(String(_factorState.roll)))
    _factorState.roll = st.default_roll;
  if (!st.attr_views.includes(_factorState.attrView))
    _factorState.attrView = st.default_attr_view;

  const ctl = $('factor-controls'); ctl.innerHTML = '';
  ctl.appendChild(factorSeg('Window', st.windows, _factorState.window, (w) => {
    _factorState.window = w; renderFactorWindow(); reloadAiBox('factors');
  }));
  ctl.appendChild(factorSeg('Model', st.models, _factorState.model, (m) => {
    _factorState.model = m; renderFactorModel(); reloadAiBox('factors');
  }));
  renderFactorWindow();
  mountAiBox('factor-ai', 'factors',
    () => ({ window: _factorState.window, model: _factorState.model }));
}

function renderFactorWindow() {
  const wb = (_factorData.by_window || {})[_factorState.window] || {};
  const alignedEmpty = $('factor-aligned-empty');
  const sections = $('factor-sections');
  if (wb.aligned_empty) {
    alignedEmpty.hidden = false; alignedEmpty.innerHTML = '';
    alignedEmpty.appendChild(el('span', { class: 'callout-icon' }, '⚠'));
    alignedEmpty.appendChild(el('span', { class: 'callout-text' }, wb.aligned_message || ''));
    sections.hidden = true;
    return;
  }
  alignedEmpty.hidden = true; sections.hidden = false;
  renderFactorStrip($('factor-strip'), wb.strip);
  renderFactorModel();
}

function renderFactorModel() {
  const wb = (_factorData.by_window || {})[_factorState.window] || {};
  const mb = (wb.models || {})[_factorState.model] || {};
  $('factor-model-note').textContent = _factorState.model;

  const toofew = $('factor-toofew');
  const detail = $('factor-detail');
  if (!mb.available) {
    toofew.hidden = false; toofew.innerHTML = '';
    toofew.appendChild(el('span', { class: 'callout-icon' }, '⚠'));
    toofew.appendChild(el('span', { class: 'callout-text' }, mb.too_few_message || ''));
    detail.hidden = true;
  } else {
    toofew.hidden = true; detail.hidden = false;
    renderKpiCards($('factor-metrics'), [
      { label: 'Months (n)', value: mb.metrics.n },
      { label: 'R²', value: mb.metrics.r2 },
      { label: 'Adj. R²', value: mb.metrics.adj_r2 },
    ]);
    $('factor-window-caption').textContent = mb.window_caption || '';
    const lowobs = $('factor-lowobs');
    if (mb.low_obs_warning) {
      lowobs.hidden = false; lowobs.innerHTML = '';
      lowobs.appendChild(el('span', { class: 'callout-icon' }, '⚠'));
      lowobs.appendChild(el('span', { class: 'callout-text' }, mb.low_obs_warning));
    } else { lowobs.hidden = true; }
    renderFactorTable($('factor-beta'), mb.beta_table);
    renderFactorWaterfall(mb.waterfall);
    renderFactorAttr(mb.attribution);
    renderFactorRolling(mb.rolling);
  }
  renderFactorPerHolding(mb.per_holding);
  renderFactorCrossCheck(mb.cross_check);
}

function renderFactorWaterfall(wf) {
  const card = $('factor-waterfall'); card.innerHTML = '';
  card.appendChild(el('div', { class: 'ch' }, [
    el('div', { class: 'ctitle' }, 'Where the return came from'),
    el('div', { class: 'csub' }, 'annualized contribution (pp)'),
  ]));
  const slot = el('div'); card.appendChild(slot);
  drawWaterfall(slot, wf || {});
  card.appendChild(el('div', { class: 'cap' }, (wf && wf.caption) || ''));
}

function renderFactorAttr(attr) {
  const card = $('factor-attr'); card.innerHTML = '';
  const head = el('div', { class: 'ch' }, [
    el('div', { class: 'ctitle' }, 'Attribution over time'),
    el('div', { class: 'csub' }, 'RF + factors + unexplained = return'),
  ]);
  const seg = el('span', { class: 'seg', style: 'margin-left:auto' });
  ['Cumulative', 'Monthly'].forEach((v) => {
    const b = el('button', { class: v === _factorState.attrView ? 'on' : '' }, v);
    b.addEventListener('click', () => { segOn(seg, b); _factorState.attrView = v; draw(); });
    seg.appendChild(b);
  });
  head.appendChild(seg); card.appendChild(head);
  const legend = el('div'); card.appendChild(legend);
  const slot = el('div'); card.appendChild(slot);
  card.appendChild(el('div', { class: 'cap' }, (attr && attr.caption) || ''));

  function draw() {
    if (!attr || !attr.series) { slot.innerHTML = ''; return; }
    const sers = attr.series;
    if (_factorState.attrView === 'Monthly') {
      const mapped = sers.map((s) => ({ name: s.name, color: s.color,
        values: (s.values || []).map((v) => (v == null ? 0 : v * 100)) }));
      toggleLegend(legend, mapped, (vis) => drawStackedBars(slot, {
        x: attr.x, fmt: _fmtPp, emptyMsg: 'No attribution.', series: vis,
      }));
    } else {
      const conv = sers.map((s) => {
        let run = 0;
        return { name: s.name, color: s.color, dash: s.dash,
          points: (attr.x || []).map((xi, i) => {
            run += (s.values[i] == null ? 0 : s.values[i] * 100); return { x: xi, v: run };
          }) };
      });
      let runT = 0;
      const totalPts = (attr.x || []).map((xi, i) => {
        let p = 0; sers.forEach((s) => { p += (s.values[i] == null ? 0 : s.values[i]); });
        runT += p * 100; return { x: xi, v: runT };
      });
      conv.push({ name: attr.total_label, color: attr.total_color, width: 4, points: totalPts });
      toggleLegend(legend, conv, (vis) =>
        drawOverlayChart(slot, vis, { key: 'v', baseline: 0, height: 300, yfmt: axPct }));
    }
  }
  draw();
}

function renderFactorRolling(roll) {
  const card = $('factor-rolling'); card.innerHTML = '';
  const head = el('div', { class: 'ch' }, [
    el('div', { class: 'ctitle' }, 'Rolling factor betas'),
    el('div', { class: 'csub' }, 'β solid · α dashed'),
  ]);
  const seg = el('span', { class: 'seg', style: 'margin-left:auto' });
  ((_factorData.state || {}).roll_windows || []).forEach((w) => {
    const b = el('button', { class: String(w) === String(_factorState.roll) ? 'on' : '' }, w + 'm');
    b.addEventListener('click', () => { segOn(seg, b); _factorState.roll = w; draw(); });
    seg.appendChild(b);
  });
  head.appendChild(seg); card.appendChild(head);
  const legend = el('div'); card.appendChild(legend);
  const slot = el('div'); card.appendChild(slot);
  const note = el('div', { class: 'cap' }); card.appendChild(note);
  card.appendChild(el('div', { class: 'cap' }, (roll && roll.caption) || ''));

  function draw() {
    const r = ((roll || {}).by_roll || {})[String(_factorState.roll)] || {};
    legend.innerHTML = '';
    if (!r.available) {
      note.textContent = '';
      slot.innerHTML = '';
      slot.appendChild(el('div', { class: 'empty-state' }, r.message || 'Unavailable.'));
      return;
    }
    note.textContent = r.low_obs_warning || '';
    const conv = (r.series || []).map((s) => ({ name: s.name, color: s.color, dash: s.dash,
      points: (r.x || []).map((xi, i) => ({ x: xi, v: s.values[i] })) }));
    toggleLegend(legend, conv, (vis) =>
      drawOverlayChart(slot, vis, { key: 'v', baseline: 0, height: 300, yfmt: axNum }));
  }
  draw();
}

function renderFactorPerHolding(ph) {
  ph = ph || {};
  const emptyEl = $('factor-ph-empty');
  const wrap = $('factor-ph');
  if (!ph.available) {
    emptyEl.hidden = false; emptyEl.innerHTML = '';
    emptyEl.appendChild(el('span', { class: 'callout-icon' }, 'ℹ'));
    emptyEl.appendChild(el('span', { class: 'callout-text' }, ph.message || ''));
    wrap.hidden = true;
    return;
  }
  emptyEl.hidden = true; wrap.hidden = false;
  renderFactorTable($('factor-ph-table'), ph.table);
  $('factor-ph-skipped').textContent = ph.skipped_caption || '';
  $('factor-ph-caption').textContent = ph.caption || '';
}

function renderFactorCrossCheck(cc) {
  cc = cc || {};
  const emptyEl = $('factor-cc-empty');
  const wrap = $('factor-cc');
  if (!cc.available) {
    emptyEl.hidden = false; emptyEl.innerHTML = '';
    emptyEl.appendChild(el('span', { class: 'callout-icon' }, 'ℹ'));
    emptyEl.appendChild(el('span', { class: 'callout-text' }, cc.message || ''));
    wrap.hidden = true;
    return;
  }
  emptyEl.hidden = true; wrap.hidden = false;
  renderFactorTable($('factor-cc-table'), cc.table);
  $('factor-cc-caption').textContent = cc.caption || '';
}

/* ============ BUY THE DIP RENDER ============ */

/* verdict callout level (success/info/warning) -> the existing callout class */
const DIP_VERDICT_CALLOUT = {
  success: 'callout-health', info: 'callout-blue', warning: 'callout-warn',
};

function renderDip(data) {
  dipWatch = (data.meta && data.meta.symbols) || [];
  const _ar = $('dip-adhoc-result'); if (_ar) _ar.innerHTML = '';
  const _ai = $('dip-adhoc-input'); if (_ai) _ai.value = '';
  ensureFilterSelects(data.meta || {});
  $('dip-caption').textContent = data.caption || '';

  // Data-vintage chip + refresh action (TK 2026-07-19 — the dip CSVs update
  // ONLY via the fetcher and once sat silently stale for a month; say the
  // as-of out loud and warn past 7 calendar days). Computed client-side from
  // meta.vintage so the fixture golden stays deterministic.
  const staleHost = $('dip-staleness');
  staleHost.innerHTML = '';
  const vin = data.meta && data.meta.vintage;
  if (vin) {
    const days = Math.floor((Date.now() - new Date(vin + 'T00:00:00')) / 86400000);
    const old = days > 7;
    staleHost.appendChild(el('div', { class: 'section-note' + (old ? ' stale-warn' : '') },
      (old ? '⚠ ' : '') + 'price data through ' + vin
      + (old ? ' — ' + days + ' days old, refresh before acting' : '')));
  }
  staleHost.appendChild(actButton('dip_history', '⟳ Refresh dip history'));

  // turbulence banner (mirrors app.py's "Market turbulence: ..." markdown). The
  // engine's label already carries its own colored glyph, so the callout uses a
  // neutral ℹ icon and the label inline.
  const tb = $('dip-turbulence');
  if (data.turbulence) {
    const t = data.turbulence;
    const pct = (t.percentile != null)
      ? ` (${t.percentile.toFixed(0)}th pct of the last ${t.n} days)` : '';
    tb.hidden = false;
    tb.innerHTML = '';
    tb.appendChild(el('span', { class: 'callout-icon' }, 'ℹ'));
    tb.appendChild(el('span', { class: 'callout-text' },
      `Market turbulence: ${t.label}${pct}. Dip stats below are conditioned on `
      + "each asset's own volatility regime (calm vs stressed); parens show the "
      + 'all-regime reference.'));
  } else { tb.hidden = true; }

  // legend (the "how to read this tab" expander); title is plain text, body is
  // server-authored HTML (trusted, no user input) — same innerHTML posture as the
  // factor/income methodology blocks. textContent would leak the raw markdown.
  $('dip-legend-title').textContent = data.legend.title;
  $('dip-legend-body').innerHTML = data.legend.body;

  // empty state + cards
  const empty = $('dip-empty');
  const host = $('dip-cards');
  host.innerHTML = '';
  if (data.empty) {
    $('dip-ai').innerHTML = '';
    empty.hidden = false; empty.innerHTML = '';
    empty.appendChild(el('span', { class: 'callout-icon' }, 'ℹ'));
    empty.appendChild(el('span', { class: 'callout-text' }, data.empty.message));
    return;
  }
  empty.hidden = true;
  mountAiBox('dip-ai', 'dip');   // B3: whole-market dip verdicts
  (data.cards || []).forEach((c) => host.appendChild(dipCard(c)));
}

// build a .tbl table from a {columns, rows, tints?} spec — shared by the dip
// forward-return table and the registered referee table.
// dip_service's _dual() prints "value" or "value (all-condition value)" --
// the leading value is today's-regime (the number the card is framed
// around), the parenthetical a reference shown only when it differs. The
// paren isn't part of the shared decoration set, so left alone it degrades
// the WHOLE column to text (verified live: every row using the dual format
// broke numeric order on "Typical Median Forward Return"). "History (Days)"
// has the same "count (count)" shape. Strip a trailing " (...)" before the
// normal cleanup -- a cell with no parens (the "History (Days)" empty
// case) is unaffected, and a genuinely non-numeric cell (horizon label,
// verdict band) still safely falls through to undefined.
function _dipLeadKey(text) {
  const t = (text || '').trim();
  if (t === '' || t === '—') return undefined;
  // Referee's "Omega 12m" prints "∞" for a bucket with zero down-days --
  // unboundedly GOOD, not unknown. Left to the default text parser, "∞"
  // degrades the whole column to locale collation (verified live: it sorted
  // FIRST ascending / LAST descending -- backwards, since the worst ratio
  // should lead ascending). 1e9 stands in as a finite "larger than any real
  // Omega" value -- opts.key must return finite or the helper reads it as
  // missing.
  if (t === '∞') return 1e9;
  const n = Number(_sortClean(t.replace(/\s*\([^)]*\)\s*$/, '')));
  return Number.isFinite(n) ? n : undefined;
}

// The referee's two trailing aggregate rows (dip_service._referee_block,
// terminal/dip_service.py ~393-399) aren't a verdict band -- match by their
// exact label (the payload's only row-identity signal; no separate
// is-aggregate flag is sent) rather than position, so a future change to
// the band count can't silently stop pinning them.
const DIP_REFEREE_PINNED_LABELS = new Set([
  '— All days', '★ Deeper than 85% + edge-claimed',
]);

function dipTbl(ft) {
  const table = el('table', { class: 'tbl' });
  const thead = el('thead'), htr = el('tr');
  ft.columns.forEach((col, i) => htr.appendChild(el('th', { class: i === 0 ? 'l' : 'r' }, col)));
  thead.appendChild(htr); table.appendChild(thead);
  const tb = el('tbody');
  ft.rows.forEach((row, i) => {
    // FIX 1 (slice-2 review): tag the referee's aggregate rows .total-row so
    // makeSortable's default pinned selector sinks them below the real bands
    // on every sort instead of letting them sort into the middle as if they
    // were two more bands. Never matches the forward-return table -- its
    // column 0 is a horizon label, not a verdict band.
    const pin = DIP_REFEREE_PINNED_LABELS.has(row[ft.columns[0]]);
    const tr = el('tr', pin ? { class: 'total-row' } : null);
    ft.columns.forEach((col, j) => {
      const td = el('td', { class: j === 0 ? 'l' : 'num' }, row[col]);
      const tint = ((ft.tints || [])[i] || {})[col];
      if (tint) td.setAttribute('style', tint);
      // FIX 2 (slice-2 review): the dual-value columns ship an exact
      // pre-rounding number alongside the display text
      // (dip_service._forward_table's `numeric` block, aligned by row index
      // same as `tints` above). Stash it as a data-sort override so the sort
      // reads the authoritative value instead of re-parsing rounded
      // "12.6% (13.7%)" text -- which both mishandles a missing lead
      // ("— (-4.6%)") and adds tie-breaking noise where raw values differ
      // but round the same. Absent for columns/tables with no numeric
      // side-table (referee; "If You Hold"; "History (Days)") -- those keep
      // falling through to _dipLeadKey below, unchanged.
      const num = ((ft.numeric || [])[i] || {})[col];
      if (num !== undefined) td.dataset.sort = num == null ? '—' : String(num);
      tr.appendChild(td);
    });
    tb.appendChild(tr);
  });
  table.appendChild(tb);
  // Shared by both the per-card forward-return table and the (SPY-only)
  // registered referee table -- one call covers each instance this builds.
  // Column 0 (horizon label / verdict band, a small closed vocabulary) and
  // the referee's "Omega 12m" ("∞"/"—"/number) fall through _dipLeadKey to
  // plain text sort, same treatment as Tax's Term column -- not a lossy
  // encoding of a richer value the way the dual-value columns are. A cell
  // carrying the data-sort override set above must win over the text parse:
  // _sortCellValue tries opts.key first, so return undefined here and let
  // its own td.dataset.sort fallback (already part of the frozen helper's
  // contract) pick it up.
  makeSortable(table, {
    key: (td) => (td.dataset.sort !== undefined ? undefined : _dipLeadKey(td.textContent)),
  });
  return table;
}

function dipCard(c) {
  const card = el('div', { class: 'card dip-card' });
  card.appendChild(el('h2', { class: 'section-title' }, c.symbol));
  card.appendChild(el('div', { class: 'cap' },
    `Conditioned on ${c.symbol}'s own volatility regime today: ${c.regime_chip} `
    + '· parens below = all-regime reference.'));

  // verdict callout — the text already begins with a colored glyph, so it is the
  // icon (no separate .callout-icon to avoid doubling the emoji).
  const v = el('div', {
    class: 'callout ' + (DIP_VERDICT_CALLOUT[c.verdict.level] || 'callout-blue'),
  });
  v.appendChild(el('span', { class: 'callout-text' }, c.verdict.text));
  card.appendChild(v);

  // 3-up KPIs — reuse the snapshot KPI grid; help text on hover (factor-strip idiom)
  const grid = el('div', { class: 'snapshot-grid', style: 'margin-top:14px' });
  [['Current drawdown', c.kpis.current_dd], ['Deeper than', c.kpis.deeper_than],
   ['Locked yield', c.kpis.locked_yield]].forEach(([label, k]) => {
    const cell = [
      el('div', { class: 'kpi-label' }, label),
      el('div', { class: 'kpi-value' }, k.value),
    ];
    // TR-basis drawdown sub-line (present only when it differs from the
    // price-basis headline — TK 2026-07-19).
    if (k.sub) cell.push(el('div', { class: 'kpi-sub' }, k.sub));
    grid.appendChild(el('div', { class: 'kpi', title: k.help || '' }, cell));
  });
  card.appendChild(grid);

  card.appendChild(el('div', { class: 'cap' }, c.bridge_text));

  // forward-return table with per-cell tints (reuse .table-wrap / table.tbl)
  const ft = c.forward_table;
  card.appendChild(el('div', { class: 'table-wrap', style: 'margin-top:14px' }, dipTbl(ft)));
  card.appendChild(el('div', { class: 'footnote' }, ft.caption));

  // narrative captions. time_underwater carries server-authored <strong> markup
  // (trusted engine prose), so render it via innerHTML; the rest are plain text.
  card.appendChild(el('div', { class: 'cap' }, c.further_fall.text));
  card.appendChild(el('div', { class: 'cap', html: c.time_underwater }));
  card.appendChild(el('div', { class: 'cap' }, c.track_record));

  // registered walk-forward referee table (SPY only — server attaches it
  // solely to the artifact's ticker; absent key = no block, by design)
  if (c.referee) {
    card.appendChild(el('div', { class: 'table-wrap', style: 'margin-top:14px' },
      dipTbl(c.referee)));
    card.appendChild(el('div', { class: 'footnote' }, c.referee.caption));
  }

  // underwater sparkline (existing single-series area chart)
  const chartHost = el('div', { style: 'margin-top:14px' });
  card.appendChild(chartHost);
  drawAreaChart(chartHost, (c.underwater || []).map((p) => ({ x: p.x, v: p.v })),
    { key: 'v', color: '#FB7185', height: 120, fill: 'rgba(251,113,133,.15)', yfmt: axPct });

  return card;
}

function adhocCallout(cls, icon, text) {
  return el('div', { class: 'callout ' + cls }, [
    el('span', { class: 'callout-icon' }, icon),
    el('span', { class: 'callout-text' }, text),
  ]);
}

async function onDipLookup(ev) {
  ev.preventDefault();
  const input = $('dip-adhoc-input');
  const btn = $('dip-adhoc-btn');
  const host = $('dip-adhoc-result');
  const sym = (input.value || '').trim().toUpperCase();  // mirror normalize_ticker
  host.innerHTML = '';
  if (!sym) return;
  if (dipWatch.includes(sym)) {  // best-effort no-network guard; server is authoritative
    host.appendChild(adhocCallout('callout-blue', 'ℹ', sym + ' is already shown below.'));
    return;
  }
  const label = btn.textContent;
  btn.disabled = true;
  btn.textContent = 'Checking…';
  try {
    const res = await fetch('/api/dip/lookup?ticker=' + encodeURIComponent(sym));
    if (res.status === 422) {
      host.appendChild(adhocCallout('callout-blue', 'ℹ',
        "Check the symbol — letters, numbers, '.' and '-' only."));
      return;
    }
    if (!res.ok) {
      host.appendChild(adhocCallout('callout-error', '⚠',
        "Couldn't reach the data source. Try again."));
      return;
    }
    const body = await res.json();
    if (body.status === 'ok' || body.status === 'stale') {
      host.appendChild(el('div', { class: 'cap' }, body.note));
      host.appendChild(dipCard(body.card));
    } else if (body.status === 'short') {
      host.appendChild(adhocCallout('callout-warn', '⚠', body.note));
    } else if (body.status === 'error') {
      host.appendChild(adhocCallout('callout-error', '⚠', body.note));
    } else {  // empty / already
      host.appendChild(adhocCallout('callout-blue', 'ℹ', body.note));
    }
  } catch (err) {
    console.error('dip lookup failed', err);
    host.appendChild(adhocCallout('callout-error', '⚠',
      "Couldn't reach the data source. Try again."));
  } finally {
    btn.disabled = false;
    btn.textContent = label;
  }
}

/* ============ RISK OVERVIEW RENDER ============ */

/* compare-tile grid: port value (green when beating SPY / red when behind) +
   a 'SPY: X' line + sub-caption. Also handles the plain (no-SPY) beta /
   concentration tiles — t.spy/t.dir simply absent. */
function renderRiskTiles(host, tiles) {
  const grid = el('div', { class: 'snapshot-grid' });
  (tiles || []).forEach((t) => {
    const valCls = t.dir === 'up' ? 'gain' : (t.dir === 'down' ? 'loss' : '');
    const kids = [
      el('div', { class: 'kpi-label' }, t.label),
      el('div', { class: 'kpi-value ' + valCls }, t.value),
    ];
    if (t.spy != null) kids.push(el('div', { class: 'kpi-spy' }, 'SPY: ' + t.spy));
    if (t.sub) kids.push(el('div', { class: 'kpi-sub' }, t.sub));
    grid.appendChild(el('div', { class: 'kpi' }, kids));
  });
  host.innerHTML = ''; host.appendChild(grid);
}

// Recovery can read "Ongoing" -- the episode hasn't closed, which is KNOWN
// information, not an unknown value, so it must not sink to the bottom in
// BOTH directions the way a real missing cell does. Months-underwater
// carries a live count even mid-episode ("14+ ongoing"), or (SPY-side, no
// running count kept) the bare word alone. 1e9 stands in for "latest/most"
// -- opts.key must return a finite number, or the helper reads it as
// missing (Number.isFinite(Infinity) is false).
const _SORT_ONGOING = 1e9;
function _episodeMonthsKey(text) {
  const t = (text || '').trim();
  if (t === 'ongoing') return _SORT_ONGOING;
  const m = /^(\d+)/.exec(t);
  return m ? Number(m[1]) : undefined;
}

function renderRiskEpisodes(block) {
  const tableEl = $('risk-dd-episodes');
  const empty = $('risk-dd-episodes-empty');
  tableEl.innerHTML = '';
  if (!block || !block.available) {
    tableEl.hidden = true; empty.hidden = false;
    empty.innerHTML = '';
    empty.appendChild(el('span', { class: 'callout-icon' }, 'ℹ'));
    empty.appendChild(el('span', { class: 'callout-text' },
      (block && block.message) || 'No drawdown episode deeper than -2% in history.'));
    return;
  }
  empty.hidden = true; tableEl.hidden = false;
  const head = ['Peak', 'Trough', 'Recovery', 'Portfolio decline',
    'SPY decline (same window)', 'Months underwater (port)', 'Months underwater (SPY)'];
  tableEl.appendChild(el('thead', {}, el('tr', {}, head.map((h) => el('th', {}, h)))));
  const tintCls = (c) => c === 'good' ? 'tint-good' : (c === 'bad' ? 'tint-bad' : null);
  const tb = el('tbody');
  block.rows.forEach((r) => {
    tb.appendChild(el('tr', {}, [
      el('td', {}, r.peak), el('td', {}, r.trough), el('td', {}, r.recovery),
      el('td', { class: tintCls(r.port_decline_cls) }, r.port_decline),
      el('td', { class: tintCls(r.spy_decline_cls) }, r.spy_decline),
      el('td', { class: tintCls(r.port_uw_cls) }, r.port_uw),
      el('td', { class: tintCls(r.spy_uw_cls) }, r.spy_uw),
    ]));
  });
  tableEl.appendChild(tb);
  // Peak/Trough/Recovery render "Jun 2026" (%b %Y) -- lossy for plain text
  // sort; Recovery adds "Ongoing" on top. Months-underwater carries a
  // "+ ongoing" suffix (port) or is bare "ongoing" (SPY, see
  // _episodeMonthsKey). Portfolio/SPY decline are plain tinted percentages
  // -- already clean, no override needed.
  makeSortable(tableEl, {
    key: (td, colIndex) => {
      if (colIndex <= 2) {
        const t = td.textContent.trim();
        return t === 'Ongoing' ? _SORT_ONGOING : _monYearKey(t);
      }
      if (colIndex === 5 || colIndex === 6) return _episodeMonthsKey(td.textContent);
      return undefined;
    },
  });
}

/* per-ticker concentration donut (adapts renderAllocClass's ring). */
function drawRiskDonut(host, donut) {
  host.innerHTML = '';
  if (!donut) { host.appendChild(el('div', { class: 'empty-state' }, 'No non-cash single-name positions.')); return; }
  host.appendChild(el('div', { class: 'ch' }, [
    el('div', { class: 'ctitle' }, donut.head), el('div', { class: 'csub' }, ''),
  ]));
  const R = 46, C = 2 * Math.PI * R;
  const svg = svgEl('svg', { viewBox: '0 0 120 120', width: '170', height: '170', class: 'donut' });
  svg.appendChild(svgEl('circle', { cx: 60, cy: 60, r: R, fill: 'none', stroke: '#232E3C', 'stroke-width': 15 }));
  let offset = 0;
  (donut.slices || []).forEach((s) => {
    const len = (s.pct / 100) * C;
    const arc = svgEl('circle', { cx: 60, cy: 60, r: R, fill: 'none', stroke: s.color, 'stroke-width': 15,
      'stroke-dasharray': len.toFixed(1) + ' ' + (C - len).toFixed(1),
      'stroke-dashoffset': offset === 0 ? null : (-offset).toFixed(1), transform: 'rotate(-90 60 60)' });
    svg.appendChild(arc);
    attachMarkTip(arc, s.label + ' · ' + s.pct.toFixed(1) + '% · ' + s.value);
    offset += len;
  });
  svg.appendChild(svgEl('text', {
    x: 60, y: 57, 'text-anchor': 'middle', fill: '#EAF0F8', 'font-size': 13,
    'font-family': "'IBM Plex Mono',monospace", 'font-weight': 600,
  })).textContent = donut.total_label || '';
  const sub = svgEl('text', {
    x: 60, y: 70, 'text-anchor': 'middle', fill: '#6B7786', 'font-size': 7,
    'font-family': "'IBM Plex Mono',monospace", 'letter-spacing': 1,
  });
  sub.textContent = donut.n_names + ' NAMES';
  svg.appendChild(sub);
  const legend = el('div', { class: 'legend' });
  (donut.slices || []).forEach((s) => {
    legend.appendChild(el('div', { class: 'legend-row' }, [
      el('span', { class: 'legend-name' }, [
        el('span', { class: 'legend-swatch', style: 'background:' + s.color }), s.label]),
      el('span', { class: 'legend-pct' }, s.pct.toFixed(1) + '%'),
      el('span', { class: 'legend-val' }, s.value),
    ]));
  });
  host.appendChild(el('div', { class: 'donut-row' }, [svg, legend]));
}

/* daily-return distribution: overlaid count bars (port azure / SPY grey) + a
   dashed normal-fit curve + dashed mean/VaR/CVaR marker lines. Log-count y with
   decade ticks via attachAxes; per-bar hover via attachMarkTip. */
function drawHistogram(host, dist, opts) {
  opts = opts || {};
  const hide = opts.hide || {};
  host.innerHTML = '';
  if (!dist || !dist.available) {
    host.appendChild(el('div', { class: 'empty-state' }, 'Need ≥30 daily returns for the distribution.')); return;
  }
  // Clickable legend: toggle a series off to read the other at full scale
  // (TK 2026-07-17 — the overlaid distributions couldn't be inspected solo).
  const legend = el('div', { class: 'hist-legend' });
  const chip = (key, label, color) => {
    const off = !!hide[key];
    const c = el('button', { class: 'hist-chip' + (off ? ' off' : ''), type: 'button' }, [
      el('span', { class: 'hist-swatch', style: 'background:' + color }), label,
    ]);
    c.addEventListener('click', () => {
      const h2 = Object.assign({}, hide); h2[key] = !off;
      drawHistogram(host, dist, Object.assign({}, opts, { hide: h2 }));
    });
    return c;
  };
  legend.appendChild(chip('port', 'Portfolio', dist.port_color));
  if (dist.spy_counts && dist.spy_counts.length) legend.appendChild(chip('spy', 'SPY', dist.spy_color));
  host.appendChild(legend);
  const W = 900, H = 320, padL = 4, padR = 4, padT = 16, padB = 14;
  const xmin = dist.x_min, xmax = dist.x_max, nb = dist.port_counts.length;
  let ymax = 1;
  if (!hide.port) {
    dist.port_counts.forEach((c) => { ymax = Math.max(ymax, c); });
    (dist.fit || []).forEach((p) => { ymax = Math.max(ymax, p.y); });
  }
  if (!hide.spy) (dist.spy_counts || []).forEach((c) => { ymax = Math.max(ymax, c); });
  // Log-count y (Streamlit parity: yaxis type="log"). On a linear axis the huge
  // central bins flatten the tail bins — where VaR/CVaR live — to ~1px. Floor
  // 0.7 keeps count-1 bars visibly above the baseline (plotly-style).
  const L = (v) => Math.log10(Math.max(v, 0.7));
  const yLo = L(0.7), yHi = L(ymax);
  const X = (v) => padL + ((v - xmin) / ((xmax - xmin) || 1)) * (W - padL - padR);
  const Y = (v) => padT + (1 - (L(v) - yLo) / ((yHi - yLo) || 1)) * (H - padT - padB);
  const svg = svgEl('svg', { viewBox: `0 0 ${W} ${H}`, width: '100%', height: String(H), preserveAspectRatio: 'none', style: 'display:block' });
  [0.25, 0.5, 0.75].forEach((f) => svg.appendChild(svgEl('line',
    { x1: 0, x2: W, y1: padT + f * (H - padT - padB), y2: padT + f * (H - padT - padB), stroke: '#303C4B' })));
  const bw = (W - padL - padR) / nb;
  for (let i = 0; i < nb; i++) {
    const x0 = padL + i * bw;
    [[hide.port ? 0 : dist.port_counts[i], dist.port_color, 0.6, 'Port'],
     [hide.spy ? 0 : (dist.spy_counts || [])[i], dist.spy_color, 0.5, 'SPY']].forEach(([c, color, op, nm]) => {
      if (!c) return;
      const y0 = Y(c);
      const rect = svgEl('rect', { x: (x0 + 1).toFixed(1), y: y0.toFixed(1),
        width: Math.max(1, bw - 2).toFixed(1), height: (H - padB - y0).toFixed(1), fill: color, opacity: String(op) });
      svg.appendChild(rect);
      const binLo = xmin + i * (xmax - xmin) / nb, binHi = xmin + (i + 1) * (xmax - xmin) / nb;
      // Sparse bins (<= 3 obs) carry their dates — "when was that tail day"
      // (TK 2026-07-19). Dense bins stay count-only.
      const dts = (nm === 'Port' ? dist.port_dates : dist.spy_dates) || {};
      const when = dts[String(i)];
      attachMarkTip(rect, axPct(binLo) + '…' + axPct(binHi) + ' · ' + nm + ': ' + c
        + (when && when.length ? ' · ' + when.join(', ') : ''));
    });
  }
  if (dist.fit && dist.fit.length && !hide.port) {   // the fit is the PORT normal
    const line = dist.fit.map((p) => `${X(p.x).toFixed(1)},${Y(p.y).toFixed(1)}`).join(' ');
    svg.appendChild(svgEl('polyline', { points: line, fill: 'none', stroke: '#9AA6B6', 'stroke-width': 2, 'stroke-dasharray': '4 3' }));
  }
  (dist.markers || []).forEach((m) => {
    const mx = X(m.x);
    svg.appendChild(svgEl('line', { x1: mx, x2: mx, y1: padT, y2: H - padB, stroke: m.color, 'stroke-width': 1.5, 'stroke-dasharray': '4 3' }));
  });
  host.appendChild(svg);
  // decade ticks (1, 10, 100, …) in log space
  const decades = [];
  for (let d = 0; Math.pow(10, d) <= ymax; d++) decades.push(d);
  attachAxes(host, {
    W, H, padL, padR, padT, padB, yLo, yHi,
    yTicks: decades,
    yfmt: (t) => { const v = Math.round(Math.pow(10, t)); return v >= 1000 ? (v / 1000) + 'k' : String(v); },
    x: { kind: 'num', lo: xmin, hi: xmax, xfmt: axPct },
  });
  // marker labels ON the lines (Streamlit-style annotations), replacing the old
  // detached text row; rows staggered — VaR/CVaR cluster on the left.
  const flags = el('div', { class: 'chart-axes' });
  (dist.markers || []).forEach((m, i) => {
    const flag = el('div', { class: 'ax-flag' }, m.label);
    flag.style.left = (X(m.x) / W * 100).toFixed(3) + '%';
    flag.style.top = (2 + (i % 2) * 13) + 'px';
    flag.style.color = m.color;
    flags.appendChild(flag);
  });
  const scaleNote = el('div', { class: 'ax-flag' }, 'count · log');
  scaleNote.style.right = '2px';
  scaleNote.style.top = '2px';
  flags.appendChild(scaleNote);
  host.style.position = 'relative';
  host.appendChild(flags);
  _pinOverlay(flags, host, H);
  // (no footer legend — the clickable chips above ARE the legend; the static
  // twin was redundant, TK 2026-07-19)
}

/* portfolio-vs-SPY daily-return scatter + up/down OLS lines + β=1 diagonal. */
function drawScatter(host, sc) {
  host.innerHTML = '';
  if (!sc || !sc.available) {
    host.appendChild(el('div', { class: 'empty-state' }, 'Need ≥30 aligned days for the scatter.')); return;
  }
  const W = 900, H = 360, padL = 4, padR = 4, padT = 12, padB = 12;
  let xmin = Infinity, xmax = -Infinity, ymin = Infinity, ymax = -Infinity;
  sc.points.forEach((p) => { xmin = Math.min(xmin, p.bx); xmax = Math.max(xmax, p.bx); ymin = Math.min(ymin, p.py); ymax = Math.max(ymax, p.py); });
  [sc.up_line, sc.dn_line, sc.diag].forEach((L) => { if (L) { xmin = Math.min(xmin, L.x0, L.x1); xmax = Math.max(xmax, L.x0, L.x1); ymin = Math.min(ymin, L.y0, L.y1); ymax = Math.max(ymax, L.y0, L.y1); } });
  if (!isFinite(xmin)) { xmin = -1; xmax = 1; ymin = -1; ymax = 1; }
  if (xmin === xmax) xmax = xmin + 1;
  if (ymin === ymax) ymax = ymin + 1;
  const X = (v) => padL + ((v - xmin) / ((xmax - xmin) || 1)) * (W - padL - padR);
  const Y = (v) => padT + (1 - (v - ymin) / ((ymax - ymin) || 1)) * (H - padT - padB);
  const svg = svgEl('svg', { viewBox: `0 0 ${W} ${H}`, width: '100%', height: String(H), preserveAspectRatio: 'none', style: 'display:block', 'data-vbw': W });
  if (xmin < 0 && xmax > 0) svg.appendChild(svgEl('line', { x1: X(0), x2: X(0), y1: padT, y2: H - padB, stroke: 'rgba(234,240,248,0.22)', 'stroke-width': 1 }));
  if (ymin < 0 && ymax > 0) svg.appendChild(svgEl('line', { x1: 0, x2: W, y1: Y(0), y2: Y(0), stroke: 'rgba(234,240,248,0.22)', 'stroke-width': 1 }));
  if (sc.diag) svg.appendChild(svgEl('line', { x1: X(sc.diag.x0), y1: Y(sc.diag.y0), x2: X(sc.diag.x1), y2: Y(sc.diag.y1), stroke: 'rgba(234,240,248,0.30)', 'stroke-width': 1, 'stroke-dasharray': '2 3' }));
  // Screen-round marks: the svg is preserveAspectRatio:none, so a <circle>
  // renders as an ellipse stretched by hostWidth/W. Compensate rx at draw time
  // (ry needs none — the rendered height equals the viewBox height); the
  // debounced resize handler (_rescaleMarks) keeps them round afterwards.
  const dotSx = (host.getBoundingClientRect().width || 951) / W;
  const DOT_RPX = 2.6;
  sc.points.forEach((p) => svg.appendChild(svgEl('ellipse', { cx: X(p.bx).toFixed(1), cy: Y(p.py).toFixed(1), rx: (DOT_RPX / dotSx).toFixed(2), ry: DOT_RPX, fill: sc.port_color, opacity: '0.7', 'data-rpx': DOT_RPX })));
  if (sc.up_line) svg.appendChild(svgEl('line', { x1: X(sc.up_line.x0), y1: Y(sc.up_line.y0), x2: X(sc.up_line.x1), y2: Y(sc.up_line.y1), stroke: sc.up_color, 'stroke-width': 2.2 }));
  if (sc.dn_line) svg.appendChild(svgEl('line', { x1: X(sc.dn_line.x0), y1: Y(sc.dn_line.y0), x2: X(sc.dn_line.x1), y2: Y(sc.dn_line.y1), stroke: sc.dn_color, 'stroke-width': 2.2 }));
  host.appendChild(svg);
  attachScatterTip(svg, sc.points, {
    W, H, padL, padR, padT, padB, xmin, xmax, ymin, ymax,
    labelFn: (p) => 'SPY ' + axPct(p.bx) + ' · Port ' + axPct(p.py),
  });
  attachAxes(host, {
    W, H, padL, padR, padT, padB, yLo: ymin, yHi: ymax,
    yfmt: axPct,
    x: { kind: 'num', lo: xmin, hi: xmax, xfmt: axPct },
  });
  const leg = el('div', { class: 'legend-foot' });
  if (sc.up_label) leg.appendChild(el('span', { class: 'legend-foot-item' }, [el('span', { class: 'legend-swatch', style: 'background:' + sc.up_color }), sc.up_label]));
  if (sc.dn_label) leg.appendChild(el('span', { class: 'legend-foot-item' }, [el('span', { class: 'legend-swatch', style: 'background:' + sc.dn_color }), sc.dn_label]));
  host.appendChild(leg);
}

/* Vol-vs-concentration tradeoff curve (a sibling to drawScatter, NOT an
   extension — the beta-scatter golden stays untouched). N line+marker series
   (each a cap-sweep of one optimizer, cap-sorted) + a highlighted current-book
   star. Per-point hover is wired via attachMarkTip and x/y axis ticks via
   attachAxes; the axes are also described in the chart-card sub. Continuous X
   (vol) / Y (Effective N, or another metric via opts) scaling, same idiom as
   drawScatter. Also doubles as the efficient-frontier chart (opts.yKey =
   'exp_return') — TRADEOFF_COLORS carries a dedicated `frontier` key so that
   series never falls back to the min-variance blue; the ◆ markers below are
   absent from the trace payload, so the trace chart's rendered output is
   unaffected by their addition.
   data: { series:[{key,label,points:[{vol,effective_n,...}]}],
           current:{vol,effective_n,...}|null, markers?:[{key,label,vol,<yKey>}] }
   opts: { yKey='effective_n', yFmt=axNum, yLabel='EffN', xLabel='vol',
           xClampHi=null, onPointClick=null } */
const TRADEOFF_COLORS = { min_variance: '#4DA3F5', risk_parity: '#2DD4BF', frontier: '#F5A45D' };
const TRADEOFF_STAR = '#EAF0F8';

function drawTradeoffCurve(host, data, opts) {
  const o = opts || {};
  const yKey = o.yKey || 'effective_n';
  const yFmt = o.yFmt || axNum;
  const yLabel = o.yLabel || 'EffN';
  const xLabel = o.xLabel || 'vol';
  const xClampHi = Number.isFinite(o.xClampHi) ? o.xClampHi : null;
  const onPointClick = typeof o.onPointClick === 'function' ? o.onPointClick : null;
  const frontierGap = !!o.frontierGap;
  const hoverReadout = !!o.hoverReadout;
  host.innerHTML = '';
  if (_mkTipEl) _mkTipEl.style.display = 'none';   // clear any lingering mark tip on async re-Trace redraw
  const series = (data && data.series) || [];
  const visOf = (s) => (s.points || []).filter((p) => xClampHi == null || p.vol <= xClampHi);
  const allPts = series.flatMap(visOf);
  const cur = (data && data.current) || null;
  if (!allPts.length && !cur && !(data.markers || []).length) {
    host.appendChild(el('div', { class: 'empty-state' }, 'No feasible cap points to plot.'));
    return;
  }
  const W = 900, H = 360, padL = 8, padR = 8, padT = 14, padB = 14;
  let xmin = Infinity, xmax = -Infinity, ymin = Infinity, ymax = -Infinity;
  const consider = (x, y) => {
    xmin = Math.min(xmin, x); xmax = Math.max(xmax, x);
    ymin = Math.min(ymin, y); ymax = Math.max(ymax, y);
  };
  allPts.forEach((p) => consider(p.vol, p[yKey]));
  if (cur) consider(cur.vol, cur[yKey]);
  (data.markers || []).forEach((m) => {
    if (Number.isFinite(m.vol) && Number.isFinite(m[yKey])) consider(m.vol, m[yKey]);
  });
  if (xmin === xmax) xmax = xmin + 1;
  if (ymin === ymax) ymax = ymin + 1;
  const X = (v) => padL + ((v - xmin) / ((xmax - xmin) || 1)) * (W - padL - padR);
  const Y = (v) => padT + (1 - (v - ymin) / ((ymax - ymin) || 1)) * (H - padT - padB);
  const svg = svgEl('svg', { viewBox: `0 0 ${W} ${H}`, width: '100%', height: String(H), preserveAspectRatio: 'none', style: 'display:block', 'data-vbw': W });
  // Screen-round markers under preserveAspectRatio:none (same treatment as
  // drawScatter): compensate rx by the host/viewBox width ratio.
  const dotSx = (host.getBoundingClientRect().width || 951) / W;
  // Efficient-frontier interpolation (opt-gated; the trace chart sets neither
  // flag, so this is inert there). Linear-interpolate the frontier series'
  // y (exp_return) at an arbitrary vol; null outside the curve's vol range.
  const _fSeries = series.find((s) => s.key === 'frontier') || null;
  const _fpts = _fSeries ? (_fSeries.points || [])
    .filter((p) => Number.isFinite(p.vol) && Number.isFinite(p[yKey]))
    .slice().sort((a, b) => a.vol - b.vol) : [];
  const _frontierYAt = (vol) => {
    if (_fpts.length < 2 || vol < _fpts[0].vol || vol > _fpts[_fpts.length - 1].vol) return null;
    for (let i = 1; i < _fpts.length; i++) {
      if (vol <= _fpts[i].vol) {
        const a = _fpts[i - 1], b = _fpts[i];
        const t = (vol - a.vol) / ((b.vol - a.vol) || 1);
        return a[yKey] + t * (b[yKey] - a[yKey]);
      }
    }
    return null;
  };
  series.forEach((s) => {
    const color = TRADEOFF_COLORS[s.key] || '#4DA3F5';
    const pts = visOf(s);
    if (pts.length > 1) {
      const d = pts.map((p, i) => (i ? 'L' : 'M') + X(p.vol).toFixed(1) + ' ' + Y(p[yKey]).toFixed(1)).join(' ');
      svg.appendChild(svgEl('path', { d: d, fill: 'none', stroke: color, 'stroke-width': 2.2, opacity: '0.9' }));
    }
    pts.forEach((p) => {
      const c = svgEl('ellipse', { cx: X(p.vol).toFixed(1), cy: Y(p[yKey]).toFixed(1), rx: (3.2 / dotSx).toFixed(2), ry: 3.2, fill: color, 'data-rpx': 3.2 });
      svg.appendChild(c);
      attachMarkTip(c, s.label + (p.cap != null ? ' · cap ' + axPctFrac(p.cap) : '')
        + (p.lam != null ? ' · λ ' + axNum(p.lam) : '')
        + ' · vol ' + axPctFrac(p.vol) + ' · ' + yLabel + ' ' + yFmt(p[yKey])
        + (yKey === 'exp_return' && p.effective_n != null
           ? ' · EffN ' + axNum(p.effective_n) : ''));
      if (onPointClick) {
        c.style.cursor = 'pointer';
        c.addEventListener('click', () => onPointClick(p, s.key));
      }
    });
  });
  if (cur) {
    const sx = X(cur.vol), syStar = Y(cur[yKey]);
    let gapTip = '';
    if (frontierGap) {
      const fEr = _frontierYAt(cur.vol);
      if (fEr != null && fEr > cur[yKey]) {
        const syF = Y(fEr);
        svg.appendChild(svgEl('line', { x1: sx.toFixed(1), y1: syStar.toFixed(1),
          x2: sx.toFixed(1), y2: syF.toFixed(1), stroke: TRADEOFF_STAR,
          'stroke-width': 1, 'stroke-dasharray': '3 3', opacity: '0.7',
          'pointer-events': 'none' }));
        const gapPp = (fEr - cur[yKey]) * 100;
        const lbl = svgEl('text', { x: (sx + 5).toFixed(1),
          y: ((syStar + syF) / 2).toFixed(1), fill: TRADEOFF_STAR, 'font-size': '11',
          'dominant-baseline': 'middle', 'pointer-events': 'none',
          style: 'paint-order:stroke;stroke:#0B0E13;stroke-width:3px' });
        lbl.textContent = '−' + gapPp.toFixed(1) + 'pp E[r] below frontier';
        svg.appendChild(lbl);
        gapTip = ' · frontier E[r] here ' + yFmt(fEr)
          + ' (−' + gapPp.toFixed(1) + 'pp)';
      }
    }
    const star = svgEl('path', { d: _starPath(sx, syStar, 8.5, 5, dotSx),
      fill: TRADEOFF_STAR, stroke: '#0B0E13', 'stroke-width': 0.8,
      'data-star': '1', 'data-cx': sx.toFixed(1), 'data-cy': syStar.toFixed(1),
      'data-r': 8.5, 'data-spikes': 5 });
    svg.appendChild(star);
    attachMarkTip(star, 'Current book · vol ' + axPctFrac(cur.vol)
      + ' · ' + yLabel + ' ' + yFmt(cur[yKey]) + gapTip);
  }
  (data.markers || []).forEach((m) => {
    if (!Number.isFinite(m.vol) || !Number.isFinite(m[yKey])) return;
    const cx = X(m.vol), cy = Y(m[yKey]);
    const d = svgEl('path', {
      d: `M${(cx).toFixed(1)} ${(cy - 6).toFixed(1)} L${(cx + 6).toFixed(1)} ${(cy).toFixed(1)} `
         + `L${(cx).toFixed(1)} ${(cy + 6).toFixed(1)} L${(cx - 6).toFixed(1)} ${(cy).toFixed(1)} Z`,
      fill: TRADEOFF_COLORS[m.key] || '#F5A45D', stroke: '#0B0E13', 'stroke-width': 0.8,
    });
    svg.appendChild(d);
    attachMarkTip(d, m.label + ' · ' + xLabel + ' ' + axPctFrac(m.vol)
                  + ' · ' + yLabel + ' ' + yFmt(m[yKey]));
  });
  host.appendChild(svg);
  if (hoverReadout && _fpts.length >= 2) {
    // Continuous read-off along the frontier. The rule+dot carry pointer-events:
    // none and the handler only fires on the bare svg (e.target === svg), so a
    // hovered data dot / star / marker keeps its own richer attachMarkTip.
    const rule = svgEl('line', { y1: padT, y2: H - padB, stroke: TRADEOFF_STAR,
      'stroke-width': 0.8, 'stroke-dasharray': '2 3', opacity: '0.55',
      'pointer-events': 'none', style: 'display:none' });
    const dot = svgEl('circle', { r: (3.5 / dotSx).toFixed(2), fill: 'none',
      stroke: TRADEOFF_STAR, 'stroke-width': 1.4, 'pointer-events': 'none',
      style: 'display:none' });
    svg.appendChild(rule); svg.appendChild(dot);
    const hideRule = () => { rule.style.display = 'none'; dot.style.display = 'none'; };
    const hideAll = () => { hideRule(); if (_mkTipEl) _mkTipEl.style.display = 'none'; };
    svg.addEventListener('mousemove', (e) => {
      // Over a data dot / ★ / ◆ (e.target !== svg): hide only the crosshair marks
      // and leave the hovered mark's own attachMarkTip tip standing — mousemove
      // bubbles, so this fires AFTER the mark's handler set the shared tip.
      if (e.target !== svg) { hideRule(); return; }
      const r = svg.getBoundingClientRect();
      if (r.width <= 0) return;
      const vbx = (e.clientX - r.left) / r.width * W;
      const vol = xmin + ((vbx - padL) / ((W - padL - padR) || 1)) * (xmax - xmin);
      const fEr = _frontierYAt(vol);
      if (fEr == null) { hideAll(); return; }   // empty area off the curve's vol range
      const px = X(vol), py = Y(fEr);
      rule.setAttribute('x1', px.toFixed(1)); rule.setAttribute('x2', px.toFixed(1));
      rule.style.display = '';
      dot.setAttribute('cx', px.toFixed(1)); dot.setAttribute('cy', py.toFixed(1));
      dot.style.display = '';
      const t = _markTipEl();
      t.textContent = 'vol ' + axPctFrac(vol) + ' → frontier E[r] ' + yFmt(fEr);
      t.style.display = 'block'; _positionMarkTip(t, e);
    });
    svg.addEventListener('mouseleave', hideAll);
  }
  attachAxes(host, {
    W, H, padL, padR, padT, padB, yLo: ymin, yHi: ymax,
    yfmt: yFmt,                        // Effective N (unitless) by default
    x: { kind: 'num', lo: xmin, hi: xmax, xfmt: axPctFrac },
  });
  const leg = el('div', { class: 'legend-foot' });
  series.forEach((s) => leg.appendChild(el('span', { class: 'legend-foot-item' },
    [el('span', { class: 'legend-swatch', style: 'background:' + (TRADEOFF_COLORS[s.key] || '#4DA3F5') }), s.label])));
  (data.markers || []).forEach((m) => {
    if (!Number.isFinite(m.vol) || !Number.isFinite(m[yKey])) return;
    leg.appendChild(el('span', { class: 'legend-foot-item' },
      [el('span', { class: 'legend-swatch',
                    style: 'background:' + (TRADEOFF_COLORS[m.key] || '#F5A45D') }),
       m.label + ' ◆']));
  });
  if (cur) leg.appendChild(el('span', { class: 'legend-foot-item' },
    [el('span', { class: 'legend-swatch', style: 'background:' + TRADEOFF_STAR }), 'Current book']));
  host.appendChild(leg);
}

/* SVG path for a `spikes`-point star centered (cx,cy), outer radius r. */
function _starPath(cx, cy, r, spikes, sx) {
  const step = Math.PI / spikes, inner = r * 0.45, xs = 1 / (sx || 1);
  let d = '', rot = -Math.PI / 2;
  for (let i = 0; i < spikes * 2; i++) {
    const rad = (i % 2 === 0) ? r : inner;
    d += (i ? 'L' : 'M') + (cx + Math.cos(rot) * rad * xs).toFixed(1) + ' ' + (cy + Math.sin(rot) * rad).toFixed(1);
    rot += step;
  }
  return d + 'Z';
}

/* overlay-line chart card; renders an empty/insufficient message when the
   series aren't available (rolling windows need enough daily history). */
function riskOverlayCard(hostId, title, sub, chart, opts) {
  chartCard($(hostId), { title: title, sub: sub }, (slot) => {
    if (!chart || !chart.available || !(chart.series || []).some((s) => (s.points || []).length)) {
      slot.appendChild(el('div', { class: 'empty-state' },
        (chart && chart.message) || 'Not enough daily history yet.'));
      return;
    }
    const box = el('div');
    const legend = el('div');
    slot.appendChild(box); slot.appendChild(legend);
    toggleLegend(legend, chart.series, (vis) => drawOverlayChart(box, vis, opts));
  });
}

function _riskCallout(node, text, on) {
  if (!on) { node.hidden = true; return; }
  node.hidden = false; node.innerHTML = '';
  node.appendChild(el('span', { class: 'callout-icon' }, 'ℹ'));
  node.appendChild(el('span', { class: 'callout-text' }, text));
}

function renderRisk(data) {
  ensureFilterSelects(data.meta);
  $('risk-caption').textContent = data.caption || '';
  $('risk-asof-note').hidden = true;

  const body = $('risk-body');
  if (!data.state || !data.state.available) {
    body.hidden = true;
    const e = $('risk-empty'); e.hidden = false; e.innerHTML = '';
    e.appendChild(el('span', { class: 'callout-icon' }, '⚠'));
    e.appendChild(el('span', { class: 'callout-text' },
      (data.state && data.state.unavailable_message) || 'Risk metrics unavailable.'));
    return;
  }
  $('risk-empty').hidden = true;
  body.hidden = false;
  mountAiBox('risk-ai', 'risk');   // B2: box narrates the filtered slice

  _riskCallout($('risk-filter-note'), data.filter_note, !!data.filter_note);

  // Coverage-gap expander.
  const cov = $('risk-coverage');
  if (data.coverage_gaps) {
    cov.hidden = false;
    const g = data.coverage_gaps;
    $('risk-coverage-title').textContent =
      `⚠ Coverage gaps — ${g.n} symbol(s), ${g.weight_total.toFixed(1)}% of risk-tab weight has > 5% missing daily prices`;
    const bodyEl = $('risk-coverage-body'); bodyEl.innerHTML = '';
    const tbl = el('table', { class: 'tbl' });
    tbl.appendChild(el('thead', {}, el('tr', {}, ['Symbol', 'Weight %', 'Days in window', 'Days missing', 'Missing %']
      .map((h) => el('th', {}, h)))));
    const tb = el('tbody');
    g.rows.forEach((r) => tb.appendChild(el('tr', {}, [
      el('td', {}, r.symbol), el('td', {}, r.weight_pct), el('td', {}, String(r.n_days_total)),
      el('td', {}, String(r.n_days_no_price)), el('td', {}, r.pct_no_price)])));
    tbl.appendChild(tb);
    // The only real <table> renderRisk builds -- the tab's correlation
    // views (drawHeatmap, below) are SVG heatmaps, not tables, so there is
    // no correlation matrix here for makeSortable to attach to.
    makeSortable(tbl);
    bodyEl.appendChild(tbl);
  } else { cov.hidden = true; }

  // Section 1 — Risk-adjusted return.
  const ra = data.risk_adjusted;
  renderRiskTiles($('risk-ra-tiles'), ra.tiles);
  riskOverlayCard('risk-ra-chart', 'Rolling 1Y Sharpe (252d window)', 'azure portfolio · grey SPY',
    ra.rolling_sharpe, { key: 'v', baseline: 0, height: 260, yfmt: axNum });

  // Section 2 — Drawdown.
  const dd = data.drawdown;
  renderRiskTiles($('risk-dd-tiles'), dd.tiles);
  renderRiskEpisodes(dd.episodes);
  chartCard($('risk-dd-chart'), { title: 'Underwater drawdown — portfolio vs SPY', sub: 'coral portfolio area · grey SPY' }, (slot) => {
    if (!dd.underwater || !dd.underwater.available) { slot.appendChild(el('div', { class: 'empty-state' }, 'No drawdown history.')); return; }
    const box = el('div');
    const legend = el('div');
    slot.appendChild(box); slot.appendChild(legend);
    toggleLegend(legend, dd.underwater.series, (vis) =>
      drawOverlayChart(box, vis, { key: 'v', baseline: 0, height: 280, fillFirst: true, yfmt: axPct }));
  });

  // Section 3 — Concentration.
  const conc = data.concentration;
  $('risk-conc-note').textContent = conc.donut ? (conc.donut.total_label + ' · ' + conc.donut.n_names + ' names') : '';
  renderRiskTiles($('risk-conc-tiles'), conc.tiles);
  drawRiskDonut($('risk-conc-donut'), conc.donut);
  $('risk-conc-caption').textContent = data.conc_caption || '';

  // Section 4 — Volatility & tail (daily).
  const dailyEmpty = $('risk-daily-empty');
  const dailyBody = $('risk-daily-body');
  if (!data.daily_available || !data.daily) {
    _riskCallout(dailyEmpty, data.daily_unavailable_message || 'Daily metrics unavailable.', true);
    dailyBody.hidden = true;
  } else {
    dailyEmpty.hidden = true; dailyBody.hidden = false;
    renderRiskTiles($('risk-vol-tiles'), data.daily.tiles);
    chartCard($('risk-dist-chart'), { title: `Daily return distribution${data.daily.distribution.available ? ' (' + data.daily.distribution.n_days + ' days)' : ''}`, sub: 'azure portfolio · grey SPY · dashed normal fit' },
      (slot) => drawHistogram(slot, data.daily.distribution));
    riskOverlayCard('risk-vol-chart', 'Rolling 60d annualized volatility', 'azure portfolio · grey SPY',
      data.daily.rolling_vol, { key: 'v', baseline: null, height: 260, yfmt: axPct });
  }

  // Section 5 — Beta to SPY.
  const betaEmpty = $('risk-beta-empty');
  const betaBody = $('risk-beta-body');
  if (!data.beta || !data.beta.available) {
    _riskCallout(betaEmpty, (data.beta && data.beta.message) || 'Daily-resolution beta needs daily prices.', true);
    betaBody.hidden = true;
  } else {
    betaEmpty.hidden = true; betaBody.hidden = false;
    renderRiskTiles($('risk-beta-tiles'), data.beta.tiles);
    chartCard($('risk-scatter'), { title: `Portfolio vs SPY daily returns${data.beta.scatter.available ? ' (' + data.beta.scatter.n + ' days, ~1Y)' : ''}`, sub: 'each dot = one trading day' },
      (slot) => drawScatter(slot, data.beta.scatter));
    riskOverlayCard('risk-beta-chart', 'Rolling β to SPY — 252d / 60d / Up-β / Down-β', 'dashed line at β = 1',
      data.beta.rolling_beta, { key: 'v', baseline: (data.beta.rolling_beta && data.beta.rolling_beta.baseline) != null ? data.beta.rolling_beta.baseline : 1, height: 260, yfmt: axNum });
    riskOverlayCard('risk-alpha-chart', 'Rolling 252d α (OLS intercept vs SPY) — annualized', 'raw returns, not CAPM',
      data.beta.rolling_alpha, { key: 'v', baseline: 0, height: 260, yfmt: axPct });
    $('risk-quadrant').innerHTML = data.quadrant_html || '';
  }
}

/* ============ RISK CONTRIBUTION RENDER ============ */

/* Two positive bars per category (weight vs PCTR).
   data: {symbols:[…], weight:[…], pctr:[…]} */
function drawPairedBars(host, data) {
  host.innerHTML = '';
  const syms = (data && data.symbols) || [];
  if (!syms.length) { host.appendChild(el('div', { class: 'empty-state' }, 'No positions.')); return; }
  const W = '#9AA6B6', P = '#4DA3F5';  // weight (neutral reference grey), PCTR (accent)
  const maxV = Math.max(1e-9, ...data.weight.map((v) => v || 0), ...data.pctr.map((v) => v || 0));
  host.appendChild(el('div', { class: 'pairbar-legend' }, [
    el('span', {}, [el('span', { class: 'pairbar-swatch', style: 'background:' + W }), 'Weight %']),
    el('span', {}, [el('span', { class: 'pairbar-swatch', style: 'background:' + P }), 'PCTR (risk) %']),
  ]));
  syms.forEach(function(s, i) {
    const w = data.weight[i] || 0, p = data.pctr[i] || 0;
    // One flex line per series: bar track + value at the bar's end (inside the
    // bar when it nearly fills the track, so the label never overflows).
    const line = function(v, color) {
      const pct = (Math.abs(v) / maxV) * 100;
      const val = el('span', { class: 'pairbar-val' + (pct > 82 ? ' pairbar-val-in' : '') },
                     v.toFixed(1) + '%');
      if (pct <= 82) val.style.left = 'calc(' + pct.toFixed(1) + '% + 6px)';
      return el('div', { class: 'pairbar-line' }, [
        el('div', { class: 'pairbar-bar', style: 'width:' + pct.toFixed(1) + '%;background:' + color }),
        val,
      ]);
    };
    const row = el('div', { class: 'pairbar-row' }, [
      el('div', { class: 'pairbar-sym' }, s),
      el('div', { class: 'pairbar-track' }, [line(w, W), line(p, P)]),
    ]);
    attachMarkTip(row, s + ' · Weight ' + w.toFixed(1) + '% · PCTR ' + p.toFixed(1) + '%');
    host.appendChild(row);
  });
}

/* Horizontal regime gauge: three colored zones [lo,stress]/[stress,calm]/[calm,hi]
   + a needle at value + the numeric value. g = {lo,hi,value,stress_thr,calm_thr,...}.
   Zone boundaries ARE the threshold positions (no separate axis labels — the
   caption states the numbers, per the deferred axis-label decision). */
function drawGauge(host, g) {
  host.innerHTML = '';
  if (!g || g.value == null || g.lo == null || g.hi == null) {
    host.appendChild(el('div', { class: 'empty-state' }, 'No ratio.')); return;
  }
  const lo = g.lo, hi = g.hi, span = (hi - lo) || 1;
  const W = 900, H = 64, padL = 8, padR = 8, top = 24, barH = 30;
  const x = (v) => padL + ((v - lo) / span) * (W - padL - padR);
  const svg = svgEl('svg', { viewBox: `0 0 ${W} ${H}`, width: '100%',
    height: String(H), preserveAspectRatio: 'none', style: 'display:block' });
  const zone = (a, b, cls) => svg.appendChild(svgEl('rect', { x: x(a).toFixed(1),
    y: top, width: Math.max(0, x(b) - x(a)).toFixed(1), height: barH, class: cls }));
  zone(lo, g.stress_thr, 'rc-gauge-loss');
  zone(g.stress_thr, g.calm_thr, 'rc-gauge-mid');
  zone(g.calm_thr, hi, 'rc-gauge-gain');
  const nx = x(g.value);
  svg.appendChild(svgEl('line', { x1: nx.toFixed(1), x2: nx.toFixed(1),
    y1: top - 4, y2: top + barH + 4, class: 'rc-gauge-needle' }));
  host.appendChild(svg);
  // Value as an HTML overlay, not svg <text> — this svg is
  // preserveAspectRatio:none, so in-svg digits stretch with viewport width.
  host.style.position = 'relative';
  const layer = el('div', { class: 'chart-axes' });
  const val = el('span', { class: 'rc-gauge-val' }, g.value.toFixed(3));
  val.style.left = Math.min(96, Math.max(4, (nx / W) * 100)).toFixed(1) + '%';
  // Above the bar (the zone fills swallowed the mid-bar label at high values).
  val.style.top = '11px';
  layer.appendChild(val);
  host.appendChild(layer);
  _pinOverlay(layer, host, H);
}

/* Ratio time series with three shaded horizontal regime bands + two dashed
   threshold lines. series = [{x, v}] (null v = gap); bands = {y_lo,y_hi,
   stress_thr,calm_thr}. Bespoke (drawOverlayChart has no band support). */
function drawBandedLine(host, series, bands) {
  host.innerHTML = '';
  if (!series || !series.length || !bands) {
    host.appendChild(el('div', { class: 'empty-state' }, 'No ratio series.')); return;
  }
  const lo = bands.y_lo, hi = bands.y_hi, span = (hi - lo) || 1;
  const W = 900, H = 280, padL = 4, padR = 4, padT = 10, padB = 10;
  const n = series.length;
  const x = (i) => padL + (i / (n - 1 || 1)) * (W - padL - padR);
  const y = (v) => padT + (1 - (v - lo) / span) * (H - padT - padB);
  const svg = svgEl('svg', { viewBox: `0 0 ${W} ${H}`, width: '100%',
    height: String(H), preserveAspectRatio: 'none', style: 'display:block' });
  const band = (a, b, cls) => svg.appendChild(svgEl('rect', { x: 0,
    y: y(b).toFixed(1), width: W, height: Math.max(0, y(a) - y(b)).toFixed(1), class: cls }));
  const bc = bands.classes || {};
  band(lo, bands.stress_thr, bc.lo || 'rc-band-loss');
  band(bands.stress_thr, bands.calm_thr, bc.mid || 'rc-band-mid');
  band(bands.calm_thr, hi, bc.hi || 'rc-band-gain');
  [[bc.lineLo || 'rc-band-line-loss', bands.stress_thr],
   [bc.lineHi || 'rc-band-line-gain', bands.calm_thr]]
    .forEach(([cls, v]) => svg.appendChild(svgEl('line', { x1: 0, x2: W,
      y1: y(v).toFixed(1), y2: y(v).toFixed(1), class: cls })));
  let seg = [];
  const flush = () => {
    if (seg.length >= 2) svg.appendChild(svgEl('polyline',
      { points: seg.join(' '), class: (bands.classes && bands.classes.series) || 'rc-band-series' }));
    seg = [];
  };
  series.forEach((p, i) => {
    if (p.v != null && isFinite(p.v)) seg.push(`${x(i).toFixed(1)},${y(p.v).toFixed(1)}`);
    else flush();
  });
  flush();
  host.appendChild(svg);
  attachAxes(host, {
    W, H, padL, padR, padT, padB, yLo: lo, yHi: hi,
    yfmt: axNum,                                  // diversification ratio (unitless)
    x: { kind: 'date', points: series },
  });
  attachCrosshair(host, {
    W, H, padL, padR, padT, padB, yLo: lo, yHi: hi, n, points: series,
    series: [{ label: '', color: '#EAF0F8',
               values: series.map((p) => p.v), fmt: axNum }],
  });
  // drag-zoom: bands (thresholds/regime rects) redraw against the sliced
  // window; bands._full remembers the original series for reset.
  const zoomFull = bands._full || series;
  attachZoom(host, { W, H, padL, padR, n }, {
    zoomed: !!bands._full,
    onZoom: (i0, i1) => drawBandedLine(host, series.slice(i0, i1 + 1),
      Object.assign({}, bands, { _full: zoomFull })),
    onReset: bands._full
      ? () => drawBandedLine(host, zoomFull, Object.assign({}, bands, { _full: null }))
      : null,
  });
}

/* 3x3 regime heatmap as a CSS grid: row labels (SPY state) x col labels (VIX
   state) + colored cells (server-computed fill) with mean/Delta/N text, plus a
   divergent legend. hm = {rows, cols, cells[3][3], legend{lo,hi,title}}. */
function drawHeatmap(host, hm, opts) {
  opts = opts || {};
  host.innerHTML = '';
  if (!hm || !hm.cells) { host.appendChild(el('div', { class: 'empty-state' }, 'No data.')); return; }
  const ncols = hm.cols.length;
  const grid = el('div', { class: 'hm-grid' + (opts.compact ? ' hm-compact' : '') });
  grid.style.gridTemplateColumns = (opts.compact ? '96px' : '84px') + ' repeat(' + ncols + ', 1fr)';
  grid.appendChild(el('div', { class: 'hm-corner' }, ''));
  hm.cols.forEach((c) => grid.appendChild(el('div', { class: 'hm-col-label' }, c)));
  hm.rows.forEach((rlabel, i) => {
    grid.appendChild(el('div', { class: 'hm-row-label' }, rlabel));
    (hm.cells[i] || []).forEach((cell, j) => {
      const d = el('div', { class: 'hm-cell' + (cell.present ? '' : ' hm-empty') });
      if (cell.color) d.style.background = cell.color;
      d.innerHTML = cell.text_html;
      if (cell.present) attachMarkTip(d, rlabel + ' · ' + hm.cols[j]);
      grid.appendChild(d);
    });
  });
  host.appendChild(grid);
  const lg = hm.legend || {};
  const bar = el('span', { class: 'hm-legend-bar' });
  if (lg.gradient) bar.style.background = lg.gradient;
  host.appendChild(el('div', { class: 'hm-legend' }, [
    el('span', {}, lg.title || 'Δ vs baseline'),
    el('span', {}, (lg.lo != null ? lg.lo.toFixed(1) : '')),
    bar,
    el('span', {}, (lg.hi != null ? '+' + lg.hi.toFixed(1) : '')),
  ]));
}

let _rcData = null;
const _rcState = { est: 'ewma_lw', alpha: '0.05', thr: '0.0', bench: 'SPY', thr_method: 'fixed', b3_roll_window: 'All',
                   b3_roll_hidden: new Set() };   // pair-series toggled off on the Big-3 rolling chart

function renderRiskContrib(data) {
  ensureFilterSelects(data.meta);
  _rcData = data;
  // .callout is a flex row (icon + text); raw inline HTML would shatter into
  // one flex item per <b>/text/<i> fragment — wrap it like every other callout.
  const rcInfo = $('rc-info');
  rcInfo.innerHTML = '';
  rcInfo.appendChild(el('span', { class: 'callout-icon' }, '🧭'));
  const rcInfoText = el('span', { class: 'callout-text' });
  rcInfoText.innerHTML = data.info_html || '';
  rcInfo.appendChild(rcInfoText);
  $('rc-caption').innerHTML = data.caption_html || '';
  const fn = $('rc-filter-note');
  fn.hidden = !data.filter_note;
  if (data.filter_note) fn.textContent = data.filter_note;
  const st = data.state || {};
  const empty = $('rc-empty'), body = $('rc-body');
  if (!st.available) {
    body.hidden = true; empty.hidden = false;
    empty.innerHTML = '';
    empty.appendChild(el('span', { class: 'callout-icon' }, 'ℹ'));
    empty.appendChild(el('span', { class: 'callout-text' }, st.unavailable_message || 'Unavailable.'));
    return;
  }
  empty.hidden = true; body.hidden = false;
  // reset control state to defaults on each (re)fetch
  _rcState.est = data.controls.estimators[0].id;
  _rcState.alpha = data.controls.es_levels[0].id;
  _rcState.thr = data.controls.thresholds[0].id;
  _rcState.bench = data.controls.benchmarks[0].id;
  _rcState.b3_roll_window = 'All';
  renderRcControls(data.controls);
  mountAiBox('rc-ai', 'riskcontrib',            // B2: box narrates the filtered slice
    () => ({ estimator: _rcState.est, benchmark: _rcState.bench }));
  const lad = $('rc-ladder');
  lad.hidden = !data.treasury_ladder.present;
  if (data.treasury_ladder.present) lad.textContent = data.treasury_ladder.caption;
  renderRcBody();
  const dr = data.dr_in_context;
  _rcState.thr_method = (dr && dr.control && dr.control.default) || 'fixed';
  renderRcDr();
  renderRcRegime();
  renderRcCorr();
}

function renderRcControls(c) {
  const host = $('rc-controls'); host.innerHTML = '';
  const seg = function(label, opts, cur, key, hasCap) {
    const group = factorSeg(label, opts.map(function(o) { return [o.id, o.label]; }), cur, function(v) {
      _rcState[key] = v;
      if (hasCap) {
        const capEl = group.querySelector('.rc-ctl-cap');
        const found = opts.find(function(o) { return o.id === v; });
        if (capEl) {
          capEl.textContent = (found && found.caption) || '';
          capEl.title = (found && found.caption) || '';   // full text on hover (caps are one-line ellipsized)
        }
      }
      renderRcBody();
      if (key === 'est') reloadAiBox('riskcontrib');
    });
    if (hasCap) {
      const cur0 = opts.find(function(o) { return o.id === cur; });
      group.appendChild(el('div', { class: 'rc-ctl-cap', title: (cur0 && cur0.caption) || '' },
        (cur0 && cur0.caption) || ''));
    }
    return group;
  };
  host.appendChild(seg('Covariance estimator', c.estimators, _rcState.est, 'est', false));
  host.appendChild(seg('Expected Shortfall confidence', c.es_levels, _rcState.alpha, 'alpha', true));
  host.appendChild(seg('Downside threshold', c.thresholds, _rcState.thr, 'thr', true));
  // benchmark: native select (universe can be large)
  const bsel = el('select', { class: 'pill-native' },
    c.benchmarks.map(function(o) { return el('option', { value: o.id }, o.label); }));
  bsel.value = _rcState.bench;
  bsel.addEventListener('change', function() {
    _rcState.bench = bsel.value; renderRcBody(); reloadAiBox('riskcontrib');
  });
  host.appendChild(el('div', { class: 'fctl-group' }, [
    el('span', { class: 'fctl-label' }, 'Benchmark'), bsel,
  ]));
}

function renderRcBody() {
  const d = _rcData;
  const key = _rcState.est + '|' + _rcState.alpha + '|' + _rcState.thr;
  const combo = d.combos[key];
  if (!combo) return;
  $('rc-estimator-strip').innerHTML = combo.estimator_strip;
  // sample warnings
  const wh = $('rc-warnings'); wh.innerHTML = '';
  (combo.sample_warnings || []).forEach(function(w) {
    wh.appendChild(el('div', { class: 'callout callout-warn' }, [
      el('span', { class: 'callout-icon' }, '⚠'),
      el('span', { class: 'callout-text' }, w.message),
    ]));
  });
  // portfolio panel (two tiles + benchmark line)
  renderRcPortfolio(combo.portfolio, d.benchmarks[_rcState.bench]);
  // top contributors
  renderRiskTiles($('rc-top-tiles'), combo.top_tiles);
  drawPairedBars($('rc-pairbar'), combo.weight_vs_pctr);
  // table
  renderRcTable(combo.table);
  $('rc-row-cue').innerHTML = combo.table.row_cue_caption;
  $('rc-window-caption').innerHTML = combo.table.window_caption;
  renderRcCorrTop15();
}

function renderRcDr() {
  const dr = (_rcData && _rcData.dr_in_context) || null;
  const wrap = $('rc-dr');
  if (!dr) { if (wrap) wrap.hidden = true; return; }
  wrap.hidden = false;
  $('rc-dr-caption').innerHTML = dr.lede_html || '';
  const empty = $('rc-dr-empty'), analytics = $('rc-dr-analytics');
  if (!dr.available) {
    analytics.hidden = true; empty.hidden = false; empty.innerHTML = '';
    empty.appendChild(el('span', { class: 'callout-icon' }, 'ℹ'));
    empty.appendChild(el('span', { class: 'callout-text' }, dr.message || 'Unavailable.'));
    return;
  }
  empty.hidden = true; analytics.hidden = false;
  $('rc-dr-howto').innerHTML = dr.howto_html || '';
  renderRcDrControl(dr.control);
  renderRiskTiles($('rc-dr-tiles'), dr.tiles);
  $('rc-dr-chart-title').textContent = dr.dr_chart.title || '';
  overlayWithLegend($('rc-dr-chart'), dr.dr_chart.series,
    { key: 'v', baseline: dr.dr_chart.baseline, height: 300, yfmt: axNum });
  $('rc-dr-chart-cap').innerHTML = dr.caption_html || '';
  renderRcDrMethod();
}

function renderRcDrControl(c) {
  const host = $('rc-dr-control'); host.innerHTML = '';
  const grp = factorSeg(c.label, c.options.map((o) => [o.id, o.label]),
    _rcState.thr_method, (v) => { _rcState.thr_method = v; renderRcDrMethod(); });
  // Column wrapper so the help caption stacks UNDER the buttons — both the
  // .rc-controls row and .fctl-group are horizontal flex, so appending the
  // caption to either lays it out BESIDE the control.
  host.appendChild(el('div', { class: 'rc-ctl-col' }, [
    grp,
    // One line, ellipsized on narrow panels (title = full text on hover);
    // the long-form explanation lives under "How this works".
    el('div', { class: 'rc-ctl-cap', title: c.help || '' }, c.help || ''),
  ]));
}

function renderRcDrMethod() {
  const dr = _rcData.dr_in_context;
  const tb = (dr.thresholds && (dr.thresholds[_rcState.thr_method] || dr.thresholds.fixed));
  if (!tb) return;
  const fb = $('rc-dr-fallback');
  fb.hidden = !tb.fallback;
  if (tb.fallback) {
    fb.innerHTML = '';
    fb.appendChild(el('span', { class: 'callout-icon' }, '⚠'));
    fb.appendChild(el('span', { class: 'callout-text' },
      tb.fallback + ' — falling back to fixed defaults for this run.'));
  }
  $('rc-dr-gauge-title').innerHTML = tb.gauge.title_html || '';
  drawGauge($('rc-dr-gauge'), tb.gauge);
  $('rc-dr-gauge-cap').innerHTML = tb.gauge.caption_html || '';
  $('rc-dr-bands-title').textContent = tb.bands.title || '';
  drawBandedLine($('rc-dr-bands'), dr.ratio_series, tb.bands);
}

const _RC_REGIME_LEVEL = { ok: 'callout-health', info: 'callout-blue',
  warn: 'callout-warn', error: 'callout-error' };

function renderRcRegime() {
  const rg = (_rcData && _rcData.dr_regime) || null;
  const wrap = $('rc-regime');
  if (!rg || rg.reason === 'dr_unavailable') { if (wrap) wrap.hidden = true; return; }
  wrap.hidden = false;
  $('rc-regime-caption').innerHTML = rg.caption_html || '';
  $('rc-regime-howto').innerHTML = rg.howto_html || '';
  const empty = $('rc-regime-empty'), analytics = $('rc-regime-analytics');
  if (!rg.available) {
    analytics.hidden = true; empty.hidden = false; empty.innerHTML = '';
    empty.appendChild(el('span', { class: 'callout-icon' }, 'ℹ'));
    empty.appendChild(el('span', { class: 'callout-text' }, rg.message || 'Unavailable.'));
    return;
  }
  empty.hidden = true; analytics.hidden = false;
  $('rc-regime-window').innerHTML = rg.window_caption_html || '';
  const ch = rg.character || {};
  const cEl = $('rc-regime-character');
  cEl.className = 'callout ' + (_RC_REGIME_LEVEL[ch.level] || 'callout-blue');
  cEl.innerHTML = '';
  cEl.appendChild(el('span', { class: 'callout-icon' }, 'ℹ'));
  cEl.appendChild(el('span', { class: 'callout-text' }, ch.headline || ''));
  const asy = $('rc-regime-asymmetry');
  asy.hidden = !ch.asymmetry_note;
  if (ch.asymmetry_note) asy.textContent = ch.asymmetry_note;
  drawHeatmap($('rc-regime-heatmap'), rg.heatmap);
  renderRiskTiles($('rc-regime-tiles'), rg.tiles);
  renderRcRegimeDetail(rg.detail);
}

function renderRcRegimeDetail(detail) {
  const t = $('rc-regime-detail'); t.innerHTML = '';
  if (!detail || !detail.columns || !detail.columns.length) return;
  t.appendChild(el('thead', {}, el('tr', {}, detail.columns.map((c) => el('th', {}, c)))));
  const tb = el('tbody');
  (detail.rows || []).forEach((row) => {
    tb.appendChild(el('tr', {}, row.map((v) => el('td', {}, String(v)))));
  });
  t.appendChild(tb);
  makeSortable(t);
}

let _rcRollData = null;

function _rcHeatBlock(block, msgId, capId, heatId, opts) {
  block = block || {};
  const msg = $(msgId);
  msg.hidden = !block.message;
  if (block.message) {
    msg.innerHTML = '';
    msg.appendChild(el('span', { class: 'callout-icon' }, 'ℹ'));
    msg.appendChild(el('span', { class: 'callout-text' }, block.message));
  }
  $(capId).innerHTML = block.caption_html || '';
  drawHeatmap($(heatId), block.heatmap, opts || {});
}

function renderRcCorr() {
  const c = (_rcData && _rcData.correlations && _rcData.correlations.major) || null;
  const wrap = $('rc-corr');
  if (!c) { if (wrap) wrap.hidden = true; return; }
  wrap.hidden = false;
  $('rc-corr-caption').innerHTML = c.caption_html || '';

  // ----- Big-3 -----
  const b3 = c.big3 || {};
  const msg = $('rc-corr-b3-msg');
  msg.hidden = !b3.message;
  if (b3.message) {
    msg.innerHTML = '';
    msg.appendChild(el('span', { class: 'callout-icon' }, 'ℹ'));
    msg.appendChild(el('span', { class: 'callout-text' }, b3.message));
  }
  $('rc-corr-b3-cap').innerHTML = b3.caption_html || '';
  drawHeatmap($('rc-corr-b3-heatmap'), b3.heatmap, { compact: false });

  const rTitle = $('rc-corr-b3-roll-title'), rLeg = $('rc-corr-b3-roll-legend');
  if (b3.rolling) {
    _rcRollData = b3.rolling;
    rTitle.hidden = false; rTitle.textContent = b3.rolling.title;
    renderRcCorrRollCtl(b3.rolling);
    // Clickable pair legend (TK 2026-07-19 — isolate e.g. SPY–SGOV alone).
    // Hidden-state lives in _rcState so the window seg-control's redraws and
    // the legend's redraws compose instead of resetting each other.
    rLeg.innerHTML = '';
    const foot = el('div', { class: 'legend-foot' });
    b3.rolling.series.forEach((s) => {
      const off = _rcState.b3_roll_hidden.has(s.name);
      const item = el('span', { class: 'legend-foot-item legend-toggle' + (off ? ' legend-off' : '') }, [
        el('span', { class: 'legend-swatch', style: 'background:' + (s.color || '#888') }),
        s.name,
      ]);
      item.addEventListener('click', () => {
        if (_rcState.b3_roll_hidden.has(s.name)) { _rcState.b3_roll_hidden.delete(s.name); item.classList.remove('legend-off'); }
        else { _rcState.b3_roll_hidden.add(s.name); item.classList.add('legend-off'); }
        drawRcCorrRoll();
      });
      foot.appendChild(item);
    });
    rLeg.appendChild(foot);
    drawRcCorrRoll();
  } else {
    _rcRollData = null;
    rTitle.hidden = true;
    $('rc-corr-b3-roll-ctl').innerHTML = '';
    rLeg.innerHTML = '';
    $('rc-corr-b3-roll').innerHTML = '';
  }

  // ----- Stress section (Big-3 once; Top-15 is per-estimator in renderRcCorrTop15) -----
  const stress = (_rcData.correlations && _rcData.correlations.stress) || null;
  const sWrap = $('rc-corr-stress');
  if (stress) {
    sWrap.hidden = false;
    $('rc-corr-stress-caption').innerHTML = stress.caption_html || '';
    $('rc-corr-stress-why').innerHTML = stress.why_spearman_html || '';
    _rcHeatBlock(stress.big3, 'rc-corr-sb3-msg', 'rc-corr-sb3-cap', 'rc-corr-sb3-heatmap', { compact: false });
  } else {
    sWrap.hidden = true;
  }

  renderRcCorrTop15();
}

function renderRcCorrRollCtl(roll) {
  const host = $('rc-corr-b3-roll-ctl'); host.innerHTML = '';
  const opts = roll.window_options.map((o) => [o.id, o.label]);
  host.appendChild(factorSeg('View window', opts, _rcState.b3_roll_window, (v) => {
    _rcState.b3_roll_window = v; drawRcCorrRoll();
  }));
}

function drawRcCorrRoll() {
  const roll = _rcRollData; if (!roll) return;
  const opt = roll.window_options.find((o) => o.id === _rcState.b3_roll_window)
              || roll.window_options[0];
  const start = opt ? opt.start : null;
  let series = roll.series.map((s) => ({
    name: s.name, color: s.color, width: 1.6,
    points: (start == null ? s.points : s.points.filter((p) => p.t >= start)),
  }));
  if (!series.some((s) => s.points.length)) {           // clipped to empty -> full
    series = roll.series.map((s) => ({ name: s.name, color: s.color, width: 1.6, points: s.points }));
  }
  const vis = series.filter((s) => !_rcState.b3_roll_hidden.has(s.name));
  drawOverlayChart($('rc-corr-b3-roll'), vis, { key: 'v', baseline: 0, height: 360, yfmt: axNum });
}

function renderRcCorrTop15() {
  const c = (_rcData && _rcData.correlations && _rcData.correlations.major) || null;
  if (!c) return;
  const blk = (c.top15 && c.top15[_rcState.est]) || {};
  const msg = $('rc-corr-t15-msg');
  msg.hidden = !blk.message;
  if (blk.message) {
    msg.innerHTML = '';
    msg.appendChild(el('span', { class: 'callout-icon' }, 'ℹ'));
    msg.appendChild(el('span', { class: 'callout-text' }, blk.message));
  }
  $('rc-corr-t15-cap').innerHTML = blk.caption_html || '';
  drawHeatmap($('rc-corr-t15-heatmap'), blk.heatmap, { compact: true });
  $('rc-corr-t15-legend').innerHTML = blk.legend_caption_html || '';

  const aTitle = $('rc-corr-t15-avg-title');
  if (blk.avg_roll) {
    aTitle.hidden = false; aTitle.textContent = blk.avg_roll.title;
    drawOverlayChart($('rc-corr-t15-avg'),
      blk.avg_roll.series.map((s) => ({ color: s.color, width: 1.8, points: s.points })),
      { key: 'v', baseline: blk.avg_roll.baseline, height: 340, yfmt: axNum });
    $('rc-corr-t15-avg-howto').innerHTML = blk.avg_roll.howto_html || '';
  } else {
    aTitle.hidden = true;
    $('rc-corr-t15-avg').innerHTML = '';
    $('rc-corr-t15-avg-howto').innerHTML = '';
  }

  // stress Top-15 for the current estimator
  const stress = (_rcData.correlations && _rcData.correlations.stress) || null;
  const sblk = (stress && stress.top15 && stress.top15[_rcState.est]) || {};
  _rcHeatBlock(sblk, 'rc-corr-st15-msg', 'rc-corr-st15-cap', 'rc-corr-st15-heatmap', { compact: true });
  const sHowto = $('rc-corr-st15-howto-det');
  if (sblk.heatmap) {           // how-to only in the populated branch (app.py enough-branch)
    sHowto.hidden = false;
    $('rc-corr-st15-howto').innerHTML = sblk.howto_html || '';
  } else {
    sHowto.hidden = true;
    $('rc-corr-st15-howto').innerHTML = '';
  }
}

function renderRcPortfolio(p, bench) {
  const host = $('rc-portfolio'); host.innerHTML = '';
  const grid = el('div', { class: 'snapshot-grid' });
  const benchLine = function(cmp, label) {
    if (!cmp || cmp.value == null) return null;
    const dirCls = cmp.dir === 'up' ? 'gain' : (cmp.dir === 'down' ? 'loss' : '');
    const arrow = cmp.dir === 'up' ? ' ▲' : (cmp.dir === 'down' ? ' ▼' : '');
    return el('div', { class: 'kpi-spy' }, [
      el('b', {}, label), ': ' + cmp.value + ' · ',
      el('span', { class: dirCls }, cmp.delta + arrow), ' vs ' + label,
    ]);
  };
  const label = (_rcData.controls.benchmarks.find(function(b) { return b.id === _rcState.bench; }) || {}).label || _rcState.bench;
  // vol tile
  const volKids = [
    el('div', { class: 'kpi-label' }, 'Portfolio volatility (annualized)'),
    el('div', { class: 'kpi-value' }, p.vol.value),
    el('div', { class: 'kpi-sub' }, p.vol.caption),
  ];
  const vl = bench && benchLine(bench.vol[_rcState.est], label);
  if (vl) volKids.push(vl);
  // ES tile
  const esKids = [
    el('div', { class: 'kpi-label' }, 'Expected Shortfall (daily)'),
    el('div', { class: 'kpi-value' }, p.es.value),
    el('div', { class: 'kpi-sub' }, p.es.caption),
  ];
  // bench ES is keyed [est][alpha] — the faithful shape (see service Task 3)
  const esCmp = bench && bench.es && bench.es[_rcState.est] && bench.es[_rcState.est][_rcState.alpha];
  const el2 = bench && benchLine(esCmp, label);
  if (el2) esKids.push(el2);
  grid.appendChild(el('div', { class: 'kpi' }, volKids));
  grid.appendChild(el('div', { class: 'kpi' }, esKids));
  host.appendChild(grid);
}

function renderRcTable(table) {
  const tEl = $('rc-table'); tEl.innerHTML = '';
  const cols = ['Symbol', 'Weight', 'PCTR', 'Risk Δ (pp)', 'Downside PCTR',
    'Δ down (pp)', 'ES PCTR', 'ES Δ (pp)', 'Standalone vol'];
  const thead = el('thead'), htr = el('tr');
  cols.forEach(function(c, i) { htr.appendChild(el('th', { class: i === 0 ? 'l' : 'r' }, c)); });
  thead.appendChild(htr); tEl.appendChild(thead);
  const tb = el('tbody');
  // obj may be a plain string/"—" or {text, cls, bold?}
  const cell = function(obj) {
    if (obj == null) return el('td', { class: 'num' }, '—');
    if (typeof obj === 'string') return el('td', { class: 'num' }, obj);
    const cls = 'num' + (obj.cls ? ' ' + obj.cls : '') + (obj.bold ? ' es-bold' : '');
    return el('td', { class: cls }, obj.text);
  };
  (table.rows || []).forEach(function(r) {
    tb.appendChild(el('tr', {}, [
      el('td', { class: 'l sym' }, r.symbol),
      cell(r.weight), cell(r.pctr), cell(r.risk_delta),
      cell(r.downside_pctr), cell(r.delta_down),
      cell(r.es_pctr), cell(r.es_delta), cell(r.standalone_vol),
    ]));
  });
  tEl.appendChild(tb);
  makeSortable(tEl);
}

function renderRiskSim(data) {
  ensureFilterSelects(data.meta);
  $('rss-caption').innerHTML = data.caption_html || '';
  const empty = $('rss-empty'), body = $('rss-body');
  if (!data.state.available) {
    body.hidden = true;
    _rssCalloutInto(empty, 'info', data.state.unavailable_message || 'Unavailable.', true);
    return;
  }
  empty.hidden = true; body.hidden = false;
  $('rss-grid-caption').innerHTML = data.grid.caption_html || '';
  $('rss-result').hidden = true;
  $('rss-run-error').hidden = true;
  _rssBuildGrid(data.grid.rows);
  _rssBindCandidate();
  _rssRenderOptimizer(data.optimizer);
  _rssSweepInit();
  $('rss-normalize').onclick = _rssNormalize;
  $('rss-reset').onclick = _rssReset;
  $('rss-run').onclick = _rssRun;
  _rssRecompute();
}

let _rssStaticBound = false;   // rss-opt-cap / rss-opt-erp are static markup —
                               // bind their staleness listeners exactly once.

function _rssRenderOptimizer(opt) {
  const box = $('rss-opt');
  if (!opt) { box.hidden = true; return; }
  box.hidden = false;
  $('rss-opt-caption').innerHTML = opt.caption_html || '';
  $('rss-opt-cap').value = String(opt.cap_default_pct);
  const t = $('rss-console-table'); t.innerHTML = '';
  t.appendChild(el('thead', {}, el('tr', {},
    ['Class', 'Floor %', 'Cap %', 'Budget %'].map((h) => el('th', {}, h)))));
  const tb = el('tbody');
  (opt.buckets || []).forEach((b) => {
    const floor = el('input', { type: 'number', min: '0', max: '100', step: '1',
      class: 'rss-opt-floor', 'data-bucket': b.key,
      value: String(b.floor_default_pct) });
    const cap = el('input', { type: 'number', min: '0', max: '100', step: '1',
      class: 'rss-opt-cap-in', 'data-bucket': b.key, value: '100' });
    const bud = el('input', { type: 'number', min: '0', max: '100', step: '1',
      class: 'rss-opt-budget', 'data-bucket': b.key, placeholder: '—' });
    [floor, cap, bud].forEach((inp) => inp.addEventListener('input', _rssFlagStale));
    tb.appendChild(el('tr', {}, [el('td', {}, b.label),
      el('td', {}, floor), el('td', {}, cap), el('td', {}, bud)]));
  });
  t.appendChild(tb);
  const msg = $('rss-opt-msg'); msg.hidden = true; msg.textContent = '';
  $('rss-opt-minvar').onclick = () => _rssOptimize('min_variance');
  $('rss-opt-riskparity').onclick = () => _rssOptimize('risk_parity');
  if (!_rssStaticBound) {
    $('rss-opt-cap').addEventListener('input', _rssFlagStale);
    $('rss-opt-erp').addEventListener('input', _rssFlagStale);
    _rssStaticBound = true;
  }
}

function _rssFlagStale() { _rssSweepStale(); }

// Render the optimizer banner via the shared callout builder, matching _rssRun.
function _rssOptMsg(kind, text) {
  _rssCalloutInto($('rss-opt-msg'), kind === 'error' ? 'error' : 'info', text, kind === 'error');
}

async function _rssOptimize(optimizer) {
  const cap = parseFloat($('rss-opt-cap').value);
  const floors = _rssReadFloors();
  const caps = _rssReadCaps();
  const budgets = _rssReadBudgets();
  const b1 = $('rss-opt-minvar'), b2 = $('rss-opt-riskparity');
  const active = optimizer === 'min_variance' ? b1 : b2;
  const label = active.textContent;
  _rssLock(true); active.textContent = 'Optimizing…';
  try {
    const acct = filterSel.account.length ? filterSel.account : ['all'];
    const cls = filterSel.asset_class.length ? filterSel.asset_class : ['all'];
    const brk = filterSel.broker.length ? filterSel.broker : ['all'];
    const hist = currentHistoryStart();
    const res = await fetch('/api/risksim/optimize', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ account: acct, asset_class: cls, broker: brk,
        history_start: hist, optimizer, cap_pct: cap, floors, caps, budgets,
        candidates: _rssCandidates() }),
    });
    if (!res.ok) throw new Error('HTTP ' + res.status);
    const out = await res.json();
    _rssOptMsg(out.kind, out.message);
    const warn = $('rss-opt-warn'), warnMsgs = out.warnings || [];
    if (warnMsgs.length) { warn.hidden = false; warn.textContent = warnMsgs.join(' · '); }
    else warn.hidden = true;
    _rssAnnounce(label + (out.kind === 'success' ? ' complete.' : ' failed.'));
    if (out.kind === 'success' && out.new_pct) {
      const rows = _rssRows();
      const vals = rows.map((tr) =>
        Number(out.new_pct[tr.querySelector('.rss-new').dataset.ticker]) || 0);
      const app = _rssApportion(vals);
      rows.forEach((tr, i) => { tr.querySelector('.rss-new').value = app[i].toFixed(2); });
      $('rss-result').hidden = true;   // suggested weights supersede any prior Run
      _rssRecompute();
    }
  } catch (e) {
    console.error('optimize failed', e);
    _rssOptMsg('error', 'Optimize request failed: ' + e.message);
    _rssAnnounce(label + ' failed.');
  } finally {
    _rssLock(false); active.textContent = label;
    _rssHidePoint(); // a point previewed during a Suggest is stale once weights change
  }
}

function _rssReadFloors() {
  const floors = {};
  $('rss-console-table').querySelectorAll('.rss-opt-floor').forEach((inp) => {
    floors[inp.dataset.bucket] = parseFloat(inp.value) || 0;
  });
  return floors;
}

// Class-cap inputs: 100 (the default) = off. parseFloat||0 would turn an
// explicit 0% cap (legal: exclude the class) into "off" - use isFinite.
function _rssReadCaps() {
  const caps = {};
  $('rss-console-table').querySelectorAll('.rss-opt-cap-in').forEach((inp) => {
    const v = parseFloat(inp.value);
    caps[inp.dataset.bucket] = Number.isFinite(v) ? v : 100;
  });
  return caps;
}

// Risk budgets: empty/0 = UNSET (omitted from the payload) - the CORRECT
// semantic here, deliberately opposite the caps isFinite rule where an
// explicit 0 must survive. Only entered, positive values are sent.
function _rssReadBudgets() {
  const budgets = {};
  $('rss-console-table').querySelectorAll('.rss-opt-budget').forEach((inp) => {
    const v = parseFloat(inp.value);
    if (Number.isFinite(v) && v > 0) budgets[inp.dataset.bucket] = v;
  });
  return budgets;
}

// One callout builder for the whole tab. kind: error|warn|blue|health|info
// ('info' = bare .callout). Icon only where the old paths showed one.
function _rssCalloutNode(kind, text, icon) {
  const cls = kind === 'info' ? 'callout' : 'callout callout-' + kind;
  const node = el('div', { class: cls });
  if (icon) node.appendChild(el('span', { class: 'callout-icon' },
    kind === 'error' ? '⛔' : kind === 'warn' ? '⚠' : 'ℹ'));
  node.appendChild(el('span', { class: 'callout-text' }, text || ''));
  return node;
}

// Apply the same content to a FIXED host element (class + children in place).
function _rssCalloutInto(hostEl, kind, text, icon) {
  const built = _rssCalloutNode(kind, text, icon);
  hostEl.className = built.className;
  hostEl.innerHTML = '';
  while (built.firstChild) hostEl.appendChild(built.firstChild);
  hostEl.hidden = false;
}

// One sweep engine, two configs. Payload state is retained per kind so
// sub-tab switches re-render instantly without a re-fetch; renderRiskSim
// clears it (a fresh fetch means the filters/data changed).
let _rssSweepState = null;
let _rssFullArc = false;
let _rssApplySel = null;   // {kind, point, seriesKey} for the open panel

const _RSS_SWEEPS = {
  trace: {
    url: '/api/risksim/trace',
    runLabel: 'Run trace',
    title: 'Vol vs concentration',
    sub: 'x: portfolio vol (ann.) · y: Effective N (1/Σw²) · ★ current book',
    chartOpts: null,
    failText: 'Cap sweep failed: ',
    emptyText: 'No feasible cap points. ',
    inputs: () => ({ cap: parseFloat($('rss-opt-cap').value),
      floors: _rssReadFloors(), caps: _rssReadCaps(),
      budgets: _rssReadBudgets() }),
    body: (inp) => ({ cap_pct: inp.cap, floors: inp.floors, caps: inp.caps,
      budgets: inp.budgets }),
  },
  frontier: {
    url: '/api/risksim/frontier',
    runLabel: 'Run frontier',
    title: 'Efficient frontier',
    sub: 'x: portfolio vol (ann.) · y: expected return (CAPM, ann.) · ★ current book · ◆ min-var / ERC',
    chartOpts: { yKey: 'exp_return', yFmt: axPctFrac, yLabel: 'E[r]', frontierGap: true, hoverReadout: true },
    failText: 'Frontier failed: ',
    emptyText: 'No feasible frontier points. ',
    inputs: () => ({ cap: parseFloat($('rss-opt-cap').value),
      floors: _rssReadFloors(), caps: _rssReadCaps(),
      erp: parseFloat($('rss-opt-erp').value) }),
    body: (inp) => ({ cap_pct: inp.cap, floors: inp.floors, caps: inp.caps,
      erp_pct: inp.erp }),
  },
};

function _rssSweepInit() {
  _rssSweepState = { active: 'trace',
    trace: { payload: null, inputs: null },
    frontier: { payload: null, inputs: null } };
  _rssFullArc = false;
  $('rss-tab-trace').onclick = () => _rssSweepSelect('trace');
  $('rss-tab-frontier').onclick = () => _rssSweepSelect('frontier');
  $('rss-sweep-run').onclick = _rssRunSweep;
  $('rss-sweep-fullarc').onclick = () => { _rssFullArc = !_rssFullArc; $('rss-sweep-fullarc').textContent = _rssFullArc ? 'Decision view' : 'Full arc'; _rssRenderSweep('frontier'); };
  $('rss-apply-btn').onclick = _rssApplyTraced; $('rss-apply-dismiss').onclick = _rssHidePoint; _rssHidePoint();
  _rssSweepSelect('trace');
}

function _rssSweepSelect(kind) {
  _rssSweepState.active = kind;
  const isTrace = kind === 'trace';
  $('rss-tab-trace').classList.toggle('active', isTrace);
  $('rss-tab-trace').setAttribute('aria-selected', String(isTrace));
  $('rss-tab-frontier').classList.toggle('active', !isTrace);
  $('rss-tab-frontier').setAttribute('aria-selected', String(!isTrace));
  $('rss-sweep-run').textContent = _RSS_SWEEPS[kind].runLabel;
  $('rss-sweep-fullarc').hidden = kind !== 'frontier';
  $('rss-sweep-fullarc').textContent = _rssFullArc ? 'Decision view' : 'Full arc';
  _rssHidePoint();
  _rssRenderSweep(kind);
}

// #2: mount the frontier AI summary box on a valid frontier render; clear it
// otherwise (trace / error / empty). Re-mount only on a NEW sig or if the box
// was cleared, so a Full-arc toggle or tab round-trip doesn't needlessly
// re-fetch. The box narrates the MEMOIZED frontier keyed by payload.sig.
let _lastFrontierSig = null;
function _frontierAiSync(kind, payload) {
  const fhost = $('frontier-ai');
  if (!fhost) return;
  const ok = kind === 'frontier' && payload && !payload.error
    && payload.series && payload.series.length && payload.sig;
  if (!ok) { _lastFrontierSig = null; fhost.innerHTML = ''; return; }
  if (payload.sig !== _lastFrontierSig || !fhost.firstChild) {
    _lastFrontierSig = payload.sig;
    mountAiBox('frontier-ai', 'frontier', () => ({ sig: _lastFrontierSig }));
  }
}

// Mount the risk-sim AI box on a successful simulation result; clear it on
// Reset or an error run. Re-mount only on a NEW sig or if the box was cleared,
// so an unrelated re-render doesn't needlessly re-fetch. Mirrors
// _frontierAiSync. The box narrates the MEMOIZED simulation keyed by sig.
let _lastRisksimSig = null;
function _risksimAiSync(sig) {
  const host = $('risksim-ai');
  if (!host) return;
  if (!sig) { _lastRisksimSig = null; host.innerHTML = ''; return; }
  if (sig !== _lastRisksimSig || !host.firstChild) {
    _lastRisksimSig = sig;
    mountAiBox('risksim-ai', 'risksim', () => ({ sig: _lastRisksimSig }));
  }
}

function _rssRenderSweep(kind) {
  const st = _rssSweepState[kind];
  const host = $('rss-sweep-result');
  const chart = $('rss-sweep-chart');
  const capEl = $('rss-sweep-caption');
  const warnEl = $('rss-sweep-warn');
  const cfg = _RSS_SWEEPS[kind];
  if (!st.payload) { host.hidden = true; warnEl.hidden = true; _frontierAiSync(kind, null); _rssSweepStale(); return; }
  host.hidden = false;
  const out = st.payload;
  const sweepWarnMsgs = out.warnings || [];
  if (sweepWarnMsgs.length) { warnEl.hidden = false; warnEl.textContent = sweepWarnMsgs.join(' · '); }
  else warnEl.hidden = true;
  if (out.error) {
    chart.innerHTML = '';
    chart.appendChild(_rssCalloutNode('error', cfg.failText + out.error, true));
    capEl.textContent = '';
  } else if (!out.series || !out.series.length) {
    chart.innerHTML = '';
    chart.appendChild(_rssCalloutNode('warn',
      cfg.emptyText + (out.empty_message || ''), false));
    capEl.textContent = '';
  } else {
    let chartOpts = cfg.chartOpts ? Object.assign({}, cfg.chartOpts) : null;
    let clampNote = '';
    if (kind === 'frontier' && !_rssFullArc) {
      const anchors = [out.current && out.current.vol]
        .concat((out.markers || []).map((m) => m.vol))
        .filter(Number.isFinite);
      const pts = (out.series[0] && out.series[0].points) || [];
      const dataMax = Math.max.apply(null, pts.map((p) => p.vol));
      if (anchors.length && pts.length) {
        const hi = Math.min(dataMax, 1.35 * Math.max.apply(null, anchors));
        const dropped = pts.filter((p) => p.vol > hi).length;
        if (dropped > 0) {
          chartOpts = Object.assign(chartOpts || {}, { xClampHi: hi });
          clampNote = ' <span>(' + dropped + ' higher-vol point(s) beyond view'
            + ' — Full arc shows them.)</span>';
        }
      }
    }
    chartOpts = Object.assign(chartOpts || {}, { onPointClick: (p, sk) => _rssShowPoint(kind, p, sk) });
    chartCard(chart, { title: cfg.title, sub: cfg.sub },
      (slot) => (chartOpts ? drawTradeoffCurve(slot, out, chartOpts)
                           : drawTradeoffCurve(slot, out)));
    capEl.innerHTML = (out.caption_html || '') + clampNote;
  }
  _frontierAiSync(kind, st.payload);
  _rssSweepStale();
}

// One structured staleness compare for whichever sweep is showing. Writes the
// dedicated note element — never string-appended into the caption.
function _rssSweepStale() {
  const kind = _rssSweepState ? _rssSweepState.active : 'trace';
  const st = _rssSweepState && _rssSweepState[kind];
  const note = $('rss-sweep-note');
  if (!st || !st.inputs || $('rss-sweep-result').hidden) {
    note.textContent = ''; return;
  }
  const live = _RSS_SWEEPS[kind].inputs();
  note.textContent = (JSON.stringify(live) === JSON.stringify(st.inputs))
    ? '' : 'Inputs changed since this sweep — re-run to refresh.';
}

// Everything that can mutate optimizer/sweep state locks during a sweep —
// the server's single-flight slot is the guarantee, this is the UX mirror.
// Grid New % edits stay enabled (they never fire a request and don't feed
// sweep staleness).
// _rssLocked also pins the Run gate through _rssRecompute during a sweep.
let _rssLocked = false;
function _rssLock(on) {
  _rssLocked = on;
  ['rss-opt-minvar', 'rss-opt-riskparity', 'rss-sweep-run', 'rss-sweep-fullarc',
   'rss-tab-trace', 'rss-tab-frontier', 'rss-opt-cap', 'rss-opt-erp', 'rss-apply-btn']
    .forEach((id) => { const n = $(id); if (n) n.disabled = on; });
  $('rss-console-table').querySelectorAll('input')
    .forEach((inp) => { inp.disabled = on; });
  // rss-run's ENABLE is owned by _rssRecompute's validity gate (Σ≈100 and
  // changed>0) — locking may force-disable it, unlocking must re-derive it.
  if (on) $('rss-run').disabled = true;
  else _rssRecompute();
}

function _rssAnnounce(text) { $('rss-live').textContent = text; }

async function _rssRunSweep() {
  _rssHidePoint(); // a re-run's fresh points supersede any open preview panel
  const kind = _rssSweepState.active;
  const cfg = _RSS_SWEEPS[kind];
  const inp = cfg.inputs();
  const btn = $('rss-sweep-run');
  _rssLock(true);
  btn.textContent = 'Running…';
  $('rss-sweep-chart').classList.add('rss-dim');
  $('rss-sweep-chart').setAttribute('aria-busy', 'true');
  _rssAnnounce(cfg.title + ' sweep started.');
  const bar = $('rss-sweep-progress');
  const fill = $('rss-sweep-progress-fill');
  const ptext = $('rss-sweep-progress-text');
  bar.hidden = false; fill.style.width = '0%';
  let secs = 0;
  ptext.textContent = 'starting… 0s';
  // Poll the O4b-1 progress slot ~1s. COMPLETION IS THE POST RESOLVING,
  // never running:false — the frontier's μ fit and marker solves run
  // outside the slot, so running flips false before the response lands.
  const runToken = {};
  bar._rssToken = runToken;
  const poll = setInterval(async () => {
    secs += 1;
    if (bar._rssToken !== runToken) return;
    try {
      const r = await fetch('/api/risksim/progress');
      const p = r.ok ? await r.json() : null;
      if (bar._rssToken !== runToken) return;
      if (p && p.running) {
        fill.style.width = Math.round(100 * p.done / Math.max(1, p.total)) + '%';
        ptext.textContent = 'point ' + p.done + '/' + p.total + ' · ' + secs + 's';
      } else {
        ptext.textContent = 'finishing… ' + secs + 's';
      }
    } catch (_e) { /* progress is best-effort; the POST is the truth */ }
  }, 1000);
  try {
    const acct = filterSel.account.length ? filterSel.account : ['all'];
    const cls = filterSel.asset_class.length ? filterSel.asset_class : ['all'];
    const brk = filterSel.broker.length ? filterSel.broker : ['all'];
    const hist = currentHistoryStart();
    const res = await fetch(cfg.url, {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(Object.assign({ account: acct, asset_class: cls,
        broker: brk, history_start: hist, candidates: _rssCandidates() }, cfg.body(inp))),
    });
    if (!res.ok) throw new Error('HTTP ' + res.status);
    _rssSweepState[kind] = { payload: await res.json(), inputs: inp };
    _rssAnnounce(cfg.title + ' sweep complete.');
  } catch (e) {
    console.error(kind + ' sweep failed', e);
    _rssSweepState[kind] = {
      payload: { error: 'Sweep request failed: ' + e.message }, inputs: inp };
    _rssAnnounce(cfg.title + ' sweep failed.');
  } finally {
    clearInterval(poll);
    bar.hidden = true;
    _rssLock(false);
    _rssHidePoint(); // a point previewed mid-run is stale against the fresh chart
    btn.textContent = _RSS_SWEEPS[_rssSweepState.active].runLabel;
    $('rss-sweep-chart').classList.remove('rss-dim');
    $('rss-sweep-chart').removeAttribute('aria-busy');
    if (_rssSweepState.active === kind) _rssRenderSweep(kind);
    // #3: on a frontier Run, auto-open the compare panel on the computed point
    // nearest the current book's vol (pairs with the "gap at your vol" readout).
    // Run-completion only (not _rssRenderSweep), so a tab round-trip or a
    // dismissed panel is never force-reopened. Display-only — the grid is
    // untouched until the explicit Apply.
    if (kind === 'frontier' && _rssSweepState.active === 'frontier') {
      const out = _rssSweepState.frontier.payload;
      const pts = out && !out.error && out.series && out.series[0]
        && out.series[0].points || [];
      const cur = out && out.current;
      if (cur && Number.isFinite(cur.vol) && pts.length) {
        let best = null, bestD = Infinity;
        pts.forEach((p) => {
          const d = Math.abs(p.vol - cur.vol);
          if (Number.isFinite(p.vol) && d < bestD) { bestD = d; best = p; }
        });
        if (best) _rssShowPoint('frontier', best, 'frontier',
          'Frontier at your vol · nearest point (' + axPctFrac(best.vol) + ')');
      }
    }
  }
}

// Click-a-point preview. Misclick-safe by construction: the grid changes
// ONLY via the explicit Apply button (spec §7.3).
function _rssShowPoint(kind, point, seriesKey, note) {
  _rssApplySel = { kind: kind, point: point, seriesKey: seriesKey };
  $('rss-apply-subhead').textContent = note || 'Traced point';
  const panel = $('rss-apply-panel');
  const facts = $('rss-apply-facts'); facts.innerHTML = '';
  const label = seriesKey === 'frontier' ? 'Efficient frontier'
    : seriesKey === 'min_variance' ? 'Min-variance' : 'Risk-parity';
  const rows = [['Source', label]];
  if (point.cap != null) rows.push(['Per-name cap', axPctFrac(point.cap)]);
  if (point.lam != null) rows.push(['Risk aversion λ', axNum(point.lam)]);
  rows.push(['Vol (ann.)', axPctFrac(point.vol)]);
  if (point.exp_return != null) rows.push(['E[r] (ann.)', axPctFrac(point.exp_return)]);
  if (point.effective_n != null) rows.push(['Effective N', axNum(point.effective_n)]);
  if (point.max_weight != null) rows.push(['Max weight', axPctFrac(point.max_weight)]);
  if (point.converged === false) rows.push(['Converged', 'no — approx']);
  const tb = el('tbody');
  rows.forEach((r) => tb.appendChild(el('tr', {},
    [el('td', {}, r[0]), el('td', { class: 'num' }, r[1])])));
  facts.appendChild(tb);
  const wt = $('rss-apply-weights'); wt.innerHTML = '';
  const w = point.weights_pct || {};
  // Current book from the grid seed (data-now), keyed by ticker. NOT the
  // editable New % — "Now" always means the current book.
  const nowByTicker = {};
  _rssRows().forEach((tr) => {
    const tk = tr.querySelector('.rss-new').dataset.ticker;
    nowByTicker[tk] = Number(tr.dataset.now) || 0;
  });
  const tickers = Array.from(new Set(
    Object.keys(nowByTicker).concat(Object.keys(w))));
  const rowsW = tickers.map((tk) => {
    const now = nowByTicker[tk] || 0;
    const tgt = Number(w[tk]) || 0;
    return { tk, now, tgt, delta: tgt - now };
  });
  const shown = rowsW.filter((r) => r.now > 0.005 || r.tgt > 0.005)
    .sort((a, b) => b.tgt - a.tgt);
  const zeroed = rowsW.length - shown.length;
  wt.appendChild(el('thead', {}, el('tr', {},
    ['Ticker', 'Now %', 'Target %', 'Δ'].map((h) => el('th', {}, h)))));
  const wb = el('tbody');
  const fmtPp = (v) => (v >= 0 ? '+' : '') + v.toFixed(2);
  shown.forEach((r) => wb.appendChild(el('tr', {}, [
    el('td', {}, r.tk),
    el('td', { class: 'num' }, r.now.toFixed(2) + '%'),
    el('td', { class: 'num' }, r.tgt.toFixed(2) + '%'),
    el('td', { class: 'num' }, fmtPp(r.delta))])));
  if (zeroed > 0) wb.appendChild(el('tr', {}, [
    el('td', {}, zeroed + ' holding(s)'),
    el('td', { class: 'num' }, '0%'),
    el('td', { class: 'num' }, '0%'),
    el('td', { class: 'num' }, '0.00')]));
  wt.appendChild(wb);
  $('rss-apply-btn').disabled = _rssLocked;
  panel.hidden = false;
  $('rss-sweep-result').classList.add('has-panel');
}

function _rssHidePoint() {
  _rssApplySel = null;
  $('rss-apply-panel').hidden = true;
  $('rss-sweep-result').classList.remove('has-panel');
}

// The Suggest fill idiom exactly (spec Update: same _rssApportion path) —
// zero re-solve, Σ lands at 100.00 by construction, Run stays manual.
function _rssApplyTraced() {
  if (!_rssApplySel || _rssLocked) return;
  const w = _rssApplySel.point.weights_pct || {};
  const rows = _rssRows();
  const vals = rows.map((tr) =>
    Number(w[tr.querySelector('.rss-new').dataset.ticker]) || 0);
  const app = _rssApportion(vals);
  rows.forEach((tr, i) => { tr.querySelector('.rss-new').value = app[i].toFixed(2); });
  $('rss-result').hidden = true;   // applied weights supersede any prior Run
  _rssRecompute();
  _rssAnnounce('Traced point applied to the grid.');
  const g = $('rss-grid');
  g.classList.add('rss-flash');
  setTimeout(() => g.classList.remove('rss-flash'), 900);
}

// The real book's full-precision weights sum to 100, but each shown at 2dp does
// NOT (a ~12-name book lands at ~99.98). The grid reads back the 2dp cell
// strings (Streamlit keeps full precision), so drop the rounding residual onto
// the largest cell → the DISPLAYED grid sums to exactly 100.00. Used for the
// seed and Normalize so Σ / Unallocated and the Run gate aren't fooled by 2dp
// drift (which otherwise leaves the untouched real book at "Unallocated 0.02%").
function _rssApportion(vals) {
  const r = vals.map((v) => Math.round((Number(v) || 0) * 100) / 100);
  const resid = Math.round((100 - r.reduce((a, v) => a + v, 0)) * 100) / 100;
  if (Math.abs(resid) >= 0.005 && r.length) {
    let mi = 0;
    for (let i = 1; i < r.length; i++) if (r[i] > r[mi]) mi = i;
    r[mi] = Math.round((r[mi] + resid) * 100) / 100;
  }
  return r;
}

function _rssBuildGrid(rows) {
  const t = $('rss-grid'); t.innerHTML = '';
  t.appendChild(el('thead', {}, el('tr', {},
    ['Ticker', 'Now %', 'New %', 'Δ %'].map((h) => el('th', {}, h)))));
  const tb = el('tbody');
  const now2 = _rssApportion(rows.map((r) => r.now_pct));
  rows.forEach((r, i) => {
    const nv = now2[i].toFixed(2);   // Now == New == same 2dp value → Δ 0, Σ 100.00 at seed
    const inp = el('input', {
      type: 'number', step: '0.10', min: '0', max: '100',
      class: 'rss-new', 'data-ticker': r.ticker, value: nv,
    });
    inp.addEventListener('input', _rssRecompute);
    tb.appendChild(el('tr', { 'data-now': nv }, [
      el('td', {}, r.ticker),
      el('td', { class: 'num' }, nv),
      el('td', {}, inp),
      el('td', { class: 'num rss-delta' }, '0.00'),
    ]));
  });
  t.appendChild(tb);
}

const _RSS_CAND_CLASSES = [['equity', 'Equity'], ['fixed_income', 'Fixed Income'],
                           ['gold', 'Gold'], ['other', 'Other']];
const _RSS_CAND_SLOTS = [1, 2, 3];

function _rssCandCell(i) {                       // one slot's raw inputs
  return { ticker: ($('rss-cand-' + i).value || '').trim().toUpperCase(),
           asset_class: $('rss-class-' + i).value || 'equity',
           proxy: ($('rss-proxy-' + i).value || '').trim().toUpperCase() };
}
function _rssCandidates() {                       // POST payload: non-blank tickers only
  return _RSS_CAND_SLOTS.map(_rssCandCell).filter((c) => c.ticker);
}

function _rssBindCandidate() {
  const onChange = () => { _rssSyncCandidateRows(); _rssCandValidate(); };
  _RSS_CAND_SLOTS.forEach((i) => {
    const cand = $('rss-cand-' + i), cls = $('rss-class-' + i), proxy = $('rss-proxy-' + i);
    cls.innerHTML = '';
    _RSS_CAND_CLASSES.forEach(([id, lab]) => cls.appendChild(el('option', { value: id }, lab)));
    cand.value = ''; cls.value = 'equity'; proxy.value = '';
    cand.oninput = onChange;
    cls.onchange = onChange;
    proxy.oninput = onChange;
  });
  $('rss-cand-note').hidden = true;
}

// Add / update / remove each candidate's editable 0% grid row as the user types
// (no fetch — the fetch is on Run). Mirrors Streamlit seeding the candidate at 0%.
function _rssSyncCandidateRows() {
  const tbody = $('rss-grid').querySelector('tbody');
  _RSS_CAND_SLOTS.forEach((i) => {
    const existing = tbody.querySelector('tr[data-candidate="' + i + '"]');
    const cand = _rssCandCell(i).ticker;
    const held = _rssRows().some((tr) => !tr.dataset.candidate
      && tr.querySelector('.rss-new').dataset.ticker === cand);
    if (!cand || held) { if (existing) existing.remove(); return; }
    if (existing) {
      if (existing.querySelector('.rss-new').dataset.ticker === cand) return;
      existing.remove();
    }
    const inp = el('input', { type: 'number', step: '0.10', min: '0', max: '100',
      class: 'rss-new', 'data-ticker': cand, value: '0.00' });
    inp.addEventListener('input', _rssRecompute);
    tbody.appendChild(el('tr', { 'data-now': '0.00', 'data-candidate': String(i) }, [
      el('td', {}, cand + ' ✨'), el('td', { class: 'num' }, '0.00'),
      el('td', {}, inp), el('td', { class: 'num rss-delta' }, '0.00')]));
  });
  _rssRecompute();
}

// Client mirror of app.py's inline candidate validations (instant feedback;
// the server re-checks authoritatively).
function _rssCandValidate() {
  const note = $('rss-cand-note');
  const cells = _RSS_CAND_SLOTS.map(_rssCandCell);
  let msg = '', kind = 'warn';
  for (const c of cells) {
    if (!c.ticker || msg) continue;
    const held = _rssRows().some((tr) => !tr.dataset.candidate
      && tr.querySelector('.rss-new').dataset.ticker === c.ticker);
    if (held) { msg = c.ticker + ' is already held — reweight it in the grid instead.'; kind = 'error'; }
    else if (c.proxy && c.proxy === c.ticker) { msg = 'Proxy ticker must differ from candidate.'; kind = 'error'; }
  }
  if (!msg) {
    const seen = new Set();
    for (const c of cells) {
      if (!c.ticker || msg) continue;
      if (seen.has(c.ticker)) { msg = c.ticker + ' is entered more than once — use one row per candidate.'; kind = 'error'; }
      seen.add(c.ticker);
    }
  }
  if (!msg) { note.hidden = true; return; }
  _rssCalloutInto(note, kind, msg, kind === 'error');
}

function _rssRows() { return Array.from($('rss-grid').querySelectorAll('tbody tr')); }

function _rssRecompute() {
  let sum = 0, changed = 0;
  _rssRows().forEach((tr) => {
    const now = parseFloat(tr.dataset.now) || 0;
    const inp = tr.querySelector('.rss-new');
    const nv = parseFloat(inp.value);
    const val = isFinite(nv) ? nv : 0;
    sum += val;
    const d = val - now;
    if (Math.abs(d) > 1e-4) changed++;
    const dcell = tr.querySelector('.rss-delta');
    dcell.textContent = (d >= 0 ? '+' : '') + d.toFixed(2);
    dcell.className = 'num rss-delta ' + (Math.abs(d) < 1e-4 ? '' : (d > 0 ? 'gain' : 'loss'));
  });
  const unalloc = Math.round((100 - sum) * 1e4) / 1e4;
  $('rss-sum').textContent = sum.toFixed(2) + '%';
  $('rss-unalloc').textContent = unalloc.toFixed(2) + '%';
  const st = $('rss-status');
  let kind = 'blue', txt = '';
  if (Math.abs(unalloc) < 0.01 && changed > 0) {
    kind = 'health';
    txt = '✅ Simulated weights total 100% · ' + changed + ' holding' + (changed !== 1 ? 's' : '') + ' changed.';
  } else if (Math.abs(unalloc) < 0.01) {
    kind = 'blue'; txt = 'No changes yet — edit a New % cell to simulate.';
  } else if (unalloc > 0) {
    kind = 'warn'; txt = '⚠ Unallocated ' + unalloc.toFixed(2) + '% — raise some New % cells (or Normalize).';
  } else {
    kind = 'error'; txt = '⛔ Over-allocated by ' + Math.abs(unalloc).toFixed(2) + '% — reduce some New % cells (or Normalize).';
  }
  _rssCalloutInto(st, kind, txt, false);
  $('rss-run').disabled = _rssLocked || !(Math.abs(unalloc) < 0.01 && changed > 0);
}

function _rssNormalize() {
  const rows = _rssRows();
  const sum = rows.reduce((a, tr) => a + (parseFloat(tr.querySelector('.rss-new').value) || 0), 0);
  if (sum <= 0) return;
  const scaled = _rssApportion(rows.map(
    (tr) => (parseFloat(tr.querySelector('.rss-new').value) || 0) / sum * 100));
  rows.forEach((tr, i) => { tr.querySelector('.rss-new').value = scaled[i].toFixed(2); });
  _rssRecompute();
}

function _rssReset() {
  _rssRows().forEach((tr) => {
    tr.querySelector('.rss-new').value = (parseFloat(tr.dataset.now) || 0).toFixed(2);
  });
  _RSS_CAND_SLOTS.forEach((i) => {
    $('rss-cand-' + i).value = ''; $('rss-proxy-' + i).value = ''; $('rss-class-' + i).value = 'equity';
  });
  $('rss-grid').querySelectorAll('tbody tr[data-candidate]').forEach((tr) => tr.remove());
  $('rss-cand-note').hidden = true;
  $('rss-opt-warn').hidden = true;
  $('rss-sweep-warn').hidden = true;
  $('rss-result').hidden = true;
  _risksimAiSync(null);
  _rssRecompute();
}

async function _rssRun() {
  const weights = {};
  _rssRows().forEach((tr) => {
    weights[tr.querySelector('.rss-new').dataset.ticker] =
      parseFloat(tr.querySelector('.rss-new').value) || 0;
  });
  const btn = $('rss-run'), err = $('rss-run-error');
  btn.disabled = true; btn.textContent = 'Running…'; err.hidden = true;
  try {
    const acct = filterSel.account.length ? filterSel.account : ['all'];
    const cls = filterSel.asset_class.length ? filterSel.asset_class : ['all'];
    const brk = filterSel.broker.length ? filterSel.broker : ['all'];
    const hist = currentHistoryStart();
    const res = await fetch('/api/risksim/simulate', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ account: acct, asset_class: cls, broker: brk, weights,
        history_start: hist, candidates: _rssCandidates() }),
    });
    if (!res.ok) throw new Error('HTTP ' + res.status);
    const view = await res.json();
    if (view.error) {
      _rssCalloutInto(err, 'error', view.error, true);
      $('rss-result').hidden = true;
      _risksimAiSync(null);
    } else {
      _rssRenderResult(view);
    }
  } catch (e) {
    console.error('risksim run failed', e);
    _rssCalloutInto(err, 'error', 'Simulation request failed: ' + e.message, true);
  } finally {
    btn.textContent = 'Run simulation'; _rssRecompute();
  }
}

function _rssRenderResult(view) {
  $('rss-result').hidden = false;
  $('rss-coverage').innerHTML = view.coverage_html || '';
  const noteEl = $('rss-run-note');
  if (view.note) {
    noteEl.hidden = false; noteEl.innerHTML = '';
    noteEl.appendChild(el('span', { class: 'callout-text' }, view.note));
  } else { noteEl.hidden = true; }
  drawDeltaBars($('rss-weightbars'), view.weight_bars);
  const h = view.headline;
  renderRiskTiles($('rss-h-risk'), h.risk);
  renderRiskTiles($('rss-h-div'), h.diversification);
  renderRiskTiles($('rss-h-conc'), h.concentration);
  $('rss-headline-caption').innerHTML = h.caption_html || '';
  const d = view.detail;
  _rssTable($('rss-vol-table'), ['Metric', 'Before', 'After', 'Δ'], d.vol_table.rows,
    (r) => [r.metric, r.before, r.after, r.delta], _RSS_MASTER_OPTS);
  $('rss-dr-note').innerHTML = d.diversification.dr_note_html || '';
  _rssHeat($('rss-corr-before'), d.diversification.corr_before);
  _rssHeat($('rss-corr-after'), d.diversification.corr_after);
  const rc = d.diversification.risk_contrib;
  _rssTable($('rss-riskcontrib'),
    ['Symbol', 'Weight %', 'Standalone vol', 'MCTR', 'CCTR', 'PCTR %'],
    (rc && rc.rows) || [],
    (r) => [r.symbol, r.weight_pct, r.standalone_vol_ann, r.mctr_ann, r.cctr_ann, r.pctr_pct]);
  $('rss-mcr-note').innerHTML = d.diversification.mcr_html || '';
  if (d.tail.drawdown) overlayWithLegend($('rss-drawdown'), d.tail.drawdown.series,
    { key: 'v', baseline: 0, height: 240, yfmt: axPct });
  else $('rss-drawdown').innerHTML = '';
  _rssTable($('rss-tail-table'), ['Metric', 'Before', 'After', 'Δ'], d.tail.tail_table.rows,
    (r) => [r.metric, r.before, r.after, r.delta], _RSS_MASTER_OPTS);
  $('rss-stress-caption').innerHTML = d.stress.caption_html || '';
  _rssHeat($('rss-scorr-before'), d.stress.scorr_before);
  _rssHeat($('rss-scorr-after'), d.stress.scorr_after);
  _rssTable($('rss-stress-table'), ['Metric', 'Before', 'After', 'Δ'], d.stress.stress_table.rows,
    (r) => [r.metric, r.before, r.after, r.delta], _RSS_MASTER_OPTS);
  _risksimAiSync(view.sig);
}

// vol_table/tail_table's "Δ" cells render e.g. "+1.23pp" (risksim_service's
// _d_pct) -- "pp" isn't part of the shared decoration set (_sortClean only
// knows $ , % + and the U+2212 minus), so left alone even ONE such cell
// degrades the whole column to text (the stress_table's Δ uses _d_num, no
// suffix, and riskcontrib has no Δ column at all -- both no-ops below).
function _rssDeltaKey(text) {
  const t = (text || '').trim();
  if (t === '') return null;
  const n = Number(_sortClean(t).replace(/pp$/, ''));
  return Number.isFinite(n) ? n : undefined;
}

// Redesign v3 (REDESIGN_NOTES §6.1 / §6.3): the three before/after tables
// share one "master" card whose 2nd/3rd theads are CSS-hidden, so they are
// NOT sortable (only the first would show sort headers) and their Δ cell
// carries gain/loss by the headline tiles' improvement semantics
// (risksim_service._dir: higher_better per metric — mirrored here by label).
const _RSS_HIGHER_BETTER = {
  'Portfolio vol (ann.)': false, 'Sharpe': true, 'Sortino': true,
  'Max drawdown': true, 'VaR 95% (daily)': true, 'CVaR 95% (daily)': true,
  'Conditional avg corr': false, 'Down-β vs SPY': false, 'Stressed DR': true,
};
function _rssDeltaDir(row) {
  const hb = _RSS_HIGHER_BETTER[row.metric];
  const n = _rssDeltaKey(row.delta);
  if (hb == null || !Number.isFinite(n) || n === 0) return '';
  return (hb ? n > 0 : n < 0) ? 'gain' : 'loss';
}
const _RSS_MASTER_OPTS = { sortable: false, deltaDir: _rssDeltaDir };

// Shared by all four risk-sim detail tables (vol / diversification's risk-
// contrib / tail / stress) -- one call here covers rss-vol-table,
// rss-riskcontrib, rss-tail-table, and rss-stress-table.
// opts: {sortable: false} skips makeSortable; {deltaDir(row)} returns
// 'gain' | 'loss' | '' for the Δ cell's class.
function _rssTable(tableEl, headers, rows, cols, opts) {
  tableEl.innerHTML = '';
  tableEl.appendChild(el('thead', {}, el('tr', {}, headers.map((h) => el('th', {}, h)))));
  const tb = el('tbody');
  const deltaCol = headers.indexOf('Δ');
  const dirOf = opts && opts.deltaDir;
  (rows || []).forEach((r) => {
    const cells = cols(r).map((c) => el('td', {}, String(c)));
    if (dirOf && deltaCol >= 0 && cells[deltaCol]) {
      const dir = dirOf(r);
      if (dir) cells[deltaCol].className = dir;
    }
    tb.appendChild(el('tr', {}, cells));
  });
  tableEl.appendChild(tb);
  if (opts && opts.sortable === false) return;
  makeSortable(tableEl, deltaCol < 0 ? undefined : {
    key: (td, colIndex) => (colIndex === deltaCol ? _rssDeltaKey(td.textContent) : undefined),
  });
}

/* Redesign v3.1 (REDESIGN_NOTES §8): "Current vs simulated weights" as
   horizontal DELTA bars — one row per holding, diverging from a centre zero
   line, sorted Δ desc, ± pp label. Reads the UNCHANGED weight_bars payload
   (port = current %, bench = simulated %). Scale: max |Δ| rounded up → the
   half-track; sub-pixel moves keep a .25% sliver so every row shows a side. */
function drawDeltaBars(host, wb) {
  host.innerHTML = '';
  const port = (wb && wb.port) || [];
  const bench = (wb && wb.bench) || [];
  if (!port.length && !bench.length) {
    host.appendChild(el('div', { class: 'empty-state' }, 'No weights.')); return;
  }
  const cur = {}, sim = {};
  port.forEach((b) => { cur[b.x] = Number(b.v) || 0; });
  bench.forEach((b) => { sim[b.x] = Number(b.v) || 0; });
  const rows = Array.from(new Set(Object.keys(cur).concat(Object.keys(sim))))
    .map((x) => ({ x, d: (sim[x] || 0) - (cur[x] || 0) }))
    .sort((a, b) => b.d - a.d);
  const half = Math.max(1, Math.ceil(Math.max(...rows.map((r) => Math.abs(r.d)))));
  host.appendChild(el('div', { class: 'deltabar-legend' }, [
    el('span', {}, 'Δ weight per holding — simulated − current, pp'),
    el('span', { class: 'swatches' }, [
      el('span', {}, [el('span', { class: 'pairbar-swatch', style: 'background:var(--gain)' }), 'added']),
      el('span', {}, [el('span', { class: 'pairbar-swatch', style: 'background:var(--loss)' }), 'trimmed']),
    ])]));
  rows.forEach((r) => {
    const dir = r.d > 1e-9 ? 'gain' : (r.d < -1e-9 ? 'loss' : '');
    const w = Math.max(0.25, Math.abs(r.d) / half * 50);
    const bar = el('div', { class: 'deltabar-bar ' + (dir === 'loss' ? 'neg' : 'pos'),
      style: 'width:' + w.toFixed(2) + '%' });
    const label = (dir === 'gain' ? '+' : (dir === 'loss' ? '−' : ''))
      + Math.abs(r.d).toFixed(2) + 'pp';
    host.appendChild(el('div', { class: 'deltabar-row' }, [
      el('div', { class: 'deltabar-sym' }, r.x),
      el('div', { class: 'deltabar-track' }, [el('div', { class: 'deltabar-mid' }), bar]),
      el('div', { class: 'deltabar-val' + (dir ? ' ' + dir : '') }, label)]));
  });
}

function _rssHeat(host, hm) {
  if (hm) drawHeatmap(host, hm, { compact: true });
  else host.innerHTML = '';
}

/* ============ OPTIONS HEDGING (read half) ============ */
function renderOptions(data) {
  // Every other tab renderer builds the filter pills; without this a direct
  // ?tab=options landing leaves broker/history unbuilt, so the recommend
  // fetch (which now forwards them) could never be narrowed.
  ensureFilterSelects(data.meta);
  $('opt-caption').textContent = (data.meta && data.meta.caption) || '';

  // Staleness chips always render (even in the empty state).
  const stale = $('opt-staleness');
  stale.innerHTML = '';
  ['snapshot', 'atm_iv'].forEach((k) => {
    const s = data.staleness && data.staleness[k];
    if (s) stale.appendChild(el('div', { class: 'section-note' }, s.chip));
  });
  stale.appendChild(actButton('option_iv', '⟳ Refresh option IV'));
  stale.appendChild(actButton('atm_iv', '⟳ Refresh ATM IV history'));

  const tiles = $('opt-tiles');
  const ivCap = $('opt-iv-caption');
  const spark = $('opt-iv-spark');
  const footer = $('opt-footer');
  const emptyBox = $('opt-empty');

  if (data.empty) {
    emptyBox.hidden = false;
    emptyBox.textContent = data.empty_message || 'No option positions.';
    tiles.innerHTML = ''; ivCap.textContent = ''; spark.innerHTML = '';
    footer.textContent = '';
    // C2: clear the recommendation region too — no read-half means no fetch.
    ['opt-composition', 'opt-signals-headline', 'opt-signals-table', 'opt-signals-caption']
      .forEach((id) => { const n = $(id); if (n) n.innerHTML = ''; });
    $('opt-controls').hidden = true;
    $('opt-scenarios-section').hidden = true;
    $('opt-coverage-section').hidden = true;
    return;
  }
  emptyBox.hidden = true;

  // 8 aggregate tiles via the shared KPI-card grid (renderRiskTiles).
  const a = data.aggregates;
  const money = (v) => (v == null ? '—' : '$' + Math.round(v).toLocaleString('en-US'));
  const signedMoney = (v) => (v == null ? '—'
    : (v >= 0 ? '+' : '−') + '$' + Math.abs(Math.round(v)).toLocaleString('en-US'));
  const signedInt = (v) => (v == null ? '—'
    : (v >= 0 ? '+' : '−') + Math.abs(Math.round(v)).toLocaleString('en-US'));
  const pct1 = (v) => (v == null ? '—' : v.toFixed(1) + '%');
  const gm = a.greeks_missing;
  renderRiskTiles(tiles, [
    { label: 'Notional protected (puts)', value: money(a.notional_protected),
      sub: a.notional_pct_nav == null ? null : pct1(a.notional_pct_nav) + ' of NAV' },
    { label: 'Premium at risk (mark)', value: money(a.premium_at_risk),
      sub: 'vs cost basis ' + money(a.cost_basis)
           + (a.n_excluded > 0 ? ' · excl. ' + a.n_excluded + ' expired/closed' : '') },
    { label: 'Unrealized P&L (vs cost)', value: signedMoney(a.unrealized_pnl),
      dir: a.unrealized_pnl == null ? 'flat' : (a.unrealized_pnl >= 0 ? 'up' : 'down'),
      sub: a.pnl_pct_cost == null ? null
        : (a.pnl_pct_cost >= 0 ? '+' : '') + a.pnl_pct_cost.toFixed(1) + '% vs cost' },
    { label: 'Weighted DTE',
      value: a.weighted_dte == null ? '—' : a.weighted_dte.toFixed(0) + ' days' },
    { label: 'Γ ($, per 1% spot)', value: gm ? '—' : signedInt(a.gamma_dollar) },
    { label: 'Vega ($, per 1 vol pt)', value: gm ? '—' : signedInt(a.vega_dollar) },
    { label: 'Θ ($/day)', value: gm ? '—' : signedInt(a.theta_dollar) },
    { label: 'Weighted IV',
      value: a.weighted_iv == null ? '—' : (a.weighted_iv * 100).toFixed(1) + '%' },
  ]);

  // IV-percentile caption + sparkline. The engine caption carries markdown bold
  // (**NN%**); render it as <strong> so it doesn't show literal asterisks (the
  // Streamlit st.caption renders the same markdown). Caption is engine-generated,
  // so escape HTML first, then convert the bold spans.
  const iv = data.iv_percentile || {};
  ivCap.innerHTML = iv.caption
    ? iv.caption.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
        .replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>')
    : '';
  if (iv.series && iv.series.length >= 2) {
    drawBandedLine(spark, iv.series, {
      y_lo: iv.bands.y_lo, y_hi: iv.bands.y_hi,
      stress_thr: iv.bands.cheap_thr, calm_thr: iv.bands.rich_thr,
      // cheap (0-20) green at the bottom, rich (80-100) red at the top: reuse the
      // existing rc-band classes swapped, so no new CSS is needed.
      classes: { lo: 'rc-band-gain', mid: 'rc-band-mid', hi: 'rc-band-loss',
                 lineLo: 'rc-band-line-gain', lineHi: 'rc-band-line-loss',
                 series: 'rc-band-series' },
    });
  } else {
    spark.innerHTML = '';
  }

  const f = data.footer || {};
  footer.textContent = (f.r_rate_pct == null ? ''
    : 'Spot anchor: latest Polygon snapshot. Risk-free rate r = '
      + f.r_rate_pct.toFixed(2) + '%. Dividend q: ' + (f.q_note || '') + '.');

  // C2: kick the recommendation fetch (composition + controls + hedge signals).
  $('opt-controls').hidden = false;
  _optRec = { mode: 'A', target: 0.10 };
  fetchRecommend();
}

// C2 recommendation state (module-level so toggles re-fetch).
let _optRec = { mode: 'A', target: 0.10 };

// Segmented control: `.seg` + plain <button> children, active = class 'on'
// (matches the existing risksim/factor/income .seg idiom — no '.seg-btn').
function _seg(hostId, options, current, onPick) {
  const host = $(hostId); host.innerHTML = '';
  options.forEach((o) => {
    const b = el('button', { class: o.value === current ? 'on' : '' }, o.label);
    b.addEventListener('click', () => onPick(o.value));
    host.appendChild(b);
  });
}

const OPT_TARGETS = {
  A: [['5%', 0.05], ['10%', 0.10], ['15%', 0.15], ['20%', 0.20], ['25%', 0.25]],
  B: [['0.5%', 0.005], ['1.0%', 0.010], ['1.5%', 0.015]] };

async function fetchRecommend() {
  $('opt-signals-headline').textContent = 'Loading recommendation…';
  try {
    // broker + history_start are GLOBAL filters — the recommend route applies
    // them server-side (same contract as GET /api/options); account/class stay
    // whole-book by design. renderOptions re-kicks this fetch on every filter
    // change, so the composition row tracks the pickers.
    const p = new URLSearchParams();
    p.set('mode', _optRec.mode);
    p.set('target', _optRec.target);
    filterSel.broker.forEach((x) => p.append('broker', x));
    const hist = currentHistoryStart();
    if (hist !== 'all') p.set('history_start', hist);
    const r = await fetch('/api/options/recommend?' + p.toString());
    if (!r.ok) { $('opt-signals-headline').textContent =
      'Recommendation unavailable (' + r.status + ').';
      $('opt-scenarios-section').hidden = true;
      $('opt-coverage-section').hidden = true; return; }
    renderRecommend(await r.json());
  } catch (e) {
    $('opt-signals-headline').textContent = 'Recommendation fetch failed.';
    $('opt-scenarios-section').hidden = true;
    $('opt-coverage-section').hidden = true;
  }
}

// Render both segmented controls from _optRec. Called on each toggle click BEFORE
// the (possibly slow, live) fetch so the highlight updates instantly — then the
// resolved fetch re-renders via renderRecommend. Matches Streamlit's instant radio.
function _optRenderControls() {
  _seg('opt-mode', [{ label: 'A — Cap drawdown', value: 'A' },
                    { label: 'B — Tail hedge', value: 'B' }], _optRec.mode, (m) => {
    _optRec.mode = m; _optRec.target = (m === 'A') ? 0.10 : 0.010;
    _optRenderControls(); fetchRecommend();
  });
  _seg('opt-target', OPT_TARGETS[_optRec.mode].map(([label, value]) => ({ label, value })),
       _optRec.target, (t) => { _optRec.target = t; _optRenderControls(); fetchRecommend(); });
}

function renderRecommend(v) {
  const c = v.composition;
  const money = (x) => (x == null ? '—' : '$' + Math.round(x).toLocaleString('en-US'));
  const pct0 = (x) => (x == null ? '' : Math.round(x) + '%');
  renderRiskTiles($('opt-composition'), [
    { label: 'Portfolio value', value: money(c.portfolio_value) },
    { label: 'Equity slice', value: money(c.equity_mv), sub: pct0(c.equity_pct) },
    { label: 'Cash-equivalent slice', value: money(c.cash_mv), sub: pct0(c.cash_pct) },
    { label: 'Existing options', value: money(c.options_mv),
      sub: c.options_pct == null ? '' : c.options_pct.toFixed(1) + '%' },
  ]);
  // Controls (segmented) — highlight updates instantly on click (optimistic),
  // then the resolved fetch re-renders everything via renderRecommend.
  _optRenderControls();

  const warn = $('opt-rec-warn');
  const msgs = (v.warnings || []).concat(v.chain_error ? ['Live chain: ' + v.chain_error] : []);
  if (msgs.length) { warn.hidden = false; warn.textContent = msgs.join(' · '); }
  else warn.hidden = true;

  const sig = v.hedge_signals || { level: 'grey', headline: '', rows: [] };
  const hl = $('opt-signals-headline');
  hl.textContent = sig.headline || '—';
  hl.className = 'section-note ' + (sig.level === 'green' ? 'sig-green'
    : sig.level === 'amber' ? 'sig-amber' : '');
  const tbl = $('opt-signals-table'); tbl.innerHTML = '';
  if (sig.rows && sig.rows.length) {
    const cols = Object.keys(sig.rows[0]);
    tbl.appendChild(el('thead', {}, el('tr', {}, cols.map((h) => el('th', {}, h)))));
    const tb = el('tbody');
    sig.rows.forEach((row) => tb.appendChild(
      el('tr', {}, cols.map((k) => el('td', {}, String(row[k]))))));
    tbl.appendChild(tb);
    makeSortable(tbl);
  }
  $('opt-signals-caption').textContent =
    'Cheap = trailing-year IV percentile below 25 (bottom quartile); '
    + 'concentrated = excess MCR > 1.5× the name’s natural SPY weight.';
  // C3: scenarios table + headline (same recommendation payload).
  renderScenarios(v.recommendation);
  // C4: existing/new puts + combined coverage (same payload; as_of for Roll-now).
  renderCoverage(v.recommendation, v.meta && v.meta.as_of);
}

// C3: scenarios table columns (10 cols mirroring app.py 8803-8816). `grp` drives
// the color group: exist=blue, new=gold, comb=green. `capital_saved` is derived.
const SC_COLS = [
  { key: 'portfolio_drawdown', label: 'Portfolio drawdown', fmt: 'dd' },
  { key: 'implied_spy_drop', label: 'Implied SPY drop', fmt: 'pct' },
  { key: 'unhedged_pnl', label: 'Unhedged P&L', fmt: 'money' },
  { key: 'existing_payoff', label: 'Existing-hedge payoff', fmt: 'money', grp: 'exist' },
  { key: 'existing_pnl', label: 'Existing-hedge P&L', fmt: 'money', grp: 'exist' },
  { key: 'existing_pnl_pct', label: 'Existing-hedge P&L (%)', fmt: 'pct', grp: 'exist' },
  { key: 'new_payoff', label: 'New-hedge payoff', fmt: 'money', grp: 'new' },
  { key: 'combined_pnl', label: 'Combined hedged P&L', fmt: 'money', grp: 'comb' },
  { key: 'combined_pnl_pct', label: 'Combined hedged P&L (%)', fmt: 'pct', grp: 'comb' },
  { key: 'capital_saved', label: 'Capital saved by hedges', fmt: 'money', grp: 'comb' },
];
// Static legend (app.py 8836-8841); has markup, so inserted via innerHTML.
const SC_LEGEND_HTML =
  '🟦 <strong>Existing-hedge</strong> columns = the result with only your current '
  + 'puts (<em>Existing-hedge P&L</em> = Unhedged P&amp;L + Existing-hedge payoff). '
  + '🟩 <strong>Combined</strong> columns = after adding the recommended new puts. '
  + '<em>Existing-hedge P&L (%)</em> is where the existing-puts headline number '
  + 'below appears as a row; a <code>+</code> value is a net gain.';
const HEADLINE_LEVELS = { success: 'callout-health', warn: 'callout-warn',
                          note: 'callout-muted', info: 'callout-blue' };
// Per-level glyph, matching the HEALTH_ICON convention (every terminal callout
// carries a leading .callout-icon). `note` reuses the neutral muted glyph.
const HEADLINE_ICON = { success: '✓', warn: '⚠', note: '⌖', info: 'ℹ' };

// Render the scenarios table + data-driven notes + the headline callout from the
// SAME recommendation payload C2 fetched. `rec` is v.recommendation (may be null).
function renderScenarios(rec) {
  const sect = $('opt-scenarios-section');
  if (!rec) { sect.hidden = true; return; }
  sect.hidden = false;
  const scMoney = (v) => (v == null ? '—' : '$' + Math.round(v).toLocaleString('en-US'));
  const scPct = (v) => (v == null ? '—' : (v >= 0 ? '+' : '') + (v * 100).toFixed(1) + '%');
  const scDd = (v) => (v == null ? '—' : (v >= 0 ? '+' : '') + Math.round(v * 100) + '%');
  const fmt = { money: scMoney, pct: scPct, dd: scDd };
  const tbl = $('opt-scenarios'); tbl.innerHTML = '';
  tbl.appendChild(el('thead', {}, el('tr', {}, SC_COLS.map((c) =>
    el('th', { class: c.grp ? 'opt-sc-' + c.grp : null }, c.label)))));
  const tb = el('tbody');
  (rec.scenarios || []).forEach((s) => {
    const capSaved = (s.existing_payoff == null || s.new_payoff == null)
      ? null : s.existing_payoff + s.new_payoff;
    const row = Object.assign({ capital_saved: capSaved }, s);
    tb.appendChild(el('tr', {}, SC_COLS.map((c) =>
      el('td', { class: c.grp ? 'opt-sc-' + c.grp : null }, fmt[c.fmt](row[c.key])))));
  });
  tbl.appendChild(tb);
  makeSortable(tbl);
  // Notes: static legend (innerHTML) + data-driven notes (textContent, safe plain text).
  const notesHost = $('opt-scenario-notes'); notesHost.innerHTML = '';
  notesHost.appendChild(el('div', { class: 'section-note', html: SC_LEGEND_HTML }));
  (rec.scenario_notes || []).forEach((n) =>
    notesHost.appendChild(el('div', { class: 'section-note', text: n })));
  // Headline callout: level -> .callout-* class + .callout-icon glyph; html into
  // .callout-text (matches the house-style icon+text callout structure).
  const head = rec.headline;
  const hb = $('opt-headline');
  if (head && head.html) {
    hb.hidden = false;
    hb.className = 'callout ' + (HEADLINE_LEVELS[head.level] || 'callout-muted');
    hb.innerHTML = '';
    hb.appendChild(el('span', { class: 'callout-icon' }, HEADLINE_ICON[head.level] || '⌖'));
    hb.appendChild(el('div', { class: 'callout-text', html: head.html }));
  } else {
    hb.hidden = true;
  }
}

// C4: existing-puts column caption (app.py 8975-8987), shown under the table when
// there ARE held puts. Has markup, so inserted via innerHTML.
const EXISTING_PUTS_CAPTION_HTML =
  '<strong>Sell at (3&times;)</strong> = the profit-take exit — close each put once its '
  + 'market value reaches 3&times; the premium you paid (compare with <strong>Current value</strong>). '
  + '<strong>Roll by</strong> = 90 days before expiry; <strong>Roll into &asymp;</strong> = a fresh '
  + '~180-day put dated around then (nearest listed monthly). Also roll if SPY drifts &plusmn;10% from '
  + 'the strike anchor — whichever comes first; don’t sell into a drawdown. '
  + '<strong>Worst-case payoff</strong> = each put’s intrinsic value in the worst modeled scenario '
  + '(a &minus;25% portfolio drawdown — implied per-name drops are larger via crash beta), '
  + '<strong>not</strong> the underlying going to $0.';

// Render §6 existing puts + §7 new puts + §8 coverage tiles from the SAME
// recommendation payload. `rec` = v.recommendation (may be null); `asOf` = v.meta.as_of.
function renderCoverage(rec, asOf) {
  const sect = $('opt-coverage-section');
  if (!rec) { sect.hidden = true; return; }
  sect.hidden = false;
  const money0 = (v) => (v == null ? '—' : '$' + Math.round(v).toLocaleString('en-US'));
  const money2 = (v) => (v == null ? '—' : '$' + v.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 }));
  const pct1 = (v) => (v == null ? '—' : (v * 100).toFixed(1) + '%');
  const pct2 = (v) => (v == null ? '—' : (v * 100).toFixed(2) + '%');
  const rollBy = (iso) => (iso == null ? '—' : (iso <= asOf ? 'Roll now' : iso));      // ISO dates compare lexically
  const rollInto = (iso) => {
    if (iso == null) return '—';
    const d = new Date(iso + 'T00:00:00');
    return '≈ ' + d.toLocaleString('en-US', { month: 'short' }) + ' ' + d.getFullYear();
  };
  // cols entries are [label, displayFn] or [label, displayFn, rawSortFn].
  // rawSortFn, when given, is stashed as data-sort -- "Roll by"/"Roll into"
  // display 'Roll now' / '≈ Nov 2026' (see rollBy/rollInto below), neither
  // of which sorts correctly as text; the real ISO date carries through
  // instead. Every other column's display text is already the full value.
  const buildTable = (tblId, cols, rows) => {
    const tbl = $(tblId); tbl.innerHTML = '';
    if (!rows.length) return;
    tbl.appendChild(el('thead', {}, el('tr', {}, cols.map((c) => el('th', {}, c[0])))));
    const tb = el('tbody');
    rows.forEach((r) => tb.appendChild(el('tr', {}, cols.map((c) => {
      const attrs = c[2] ? { 'data-sort': String(c[2](r) ?? '') } : null;
      return el('td', attrs, String(c[1](r)));
    }))));
    tbl.appendChild(tb);
    makeSortable(tbl);
  };
  // §6 existing puts (app.py 8960-8972)
  const exCols = [
    ['Ticker', (p) => p.ticker], ['Strike', (p) => money2(p.strike)], ['Expiry', (p) => p.expiry],
    ['Roll by', (p) => rollBy(p.roll_by), (p) => p.roll_by],
    ['Roll into ≈', (p) => rollInto(p.roll_into), (p) => p.roll_into],
    ['Contracts', (p) => p.contracts], ['Cost basis', (p) => money0(p.cost_basis)],
    ['Current value', (p) => money0(p.current_value)], ['Sell at (3×)', (p) => money0(p.sell_at)],
    ['Worst-case payoff', (p) => money0(p.worst_case_payoff)],
  ];
  const exPuts = rec.existing_puts || [];
  buildTable('opt-existing-puts', exCols, exPuts);
  const exNote = $('opt-existing-puts-note');
  if (exPuts.length) { exNote.innerHTML = EXISTING_PUTS_CAPTION_HTML; }
  else { exNote.textContent = 'No existing put positions.'; }
  // §7 new puts (app.py 9003-9015)
  const newCols = [
    ['Ticker', (p) => p.ticker], ['Role', (p) => p.role], ['Strike', (p) => money2(p.strike)],
    ['Strike % OTM', (p) => pct1(p.strike_pct_otm)], ['Expiry', (p) => p.expiry],
    ['Contracts', (p) => p.contracts], ['Premium / share', (p) => money2(p.premium_per_share)],
    ['Position cost', (p) => money0(p.position_cost)], ['Sell at (3×)', (p) => money0(p.sell_at)],
    ['Annualized drag', (p) => pct2(p.annualized_drag_pct)],
  ];
  const newPuts = rec.new_puts || [];
  buildTable('opt-new-puts', newCols, newPuts);
  const newNote = $('opt-new-puts-note');
  if (newPuts.length) { newNote.textContent = ''; newNote.hidden = true; }
  else {
    newNote.hidden = false;
    newNote.textContent = rec.mode === 'A'
      ? 'Already covered — nothing to buy.'
      : 'Budget too small to buy a whole 20%-OTM contract — raise the budget or refresh the chain.';
  }
  // §8 combined coverage tiles (app.py 9022-9027)
  renderRiskTiles($('opt-coverage-tiles'), [
    { label: 'Total new premium', value: money0(rec.total_new_premium) },
    { label: 'New drag (annualized)', value: pct2(rec.total_new_drag_pct) },
    { label: 'Combined drag (existing + new)', value: pct2(rec.total_combined_drag_pct) },
  ]);
}
