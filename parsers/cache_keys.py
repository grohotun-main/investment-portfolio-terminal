"""File-state signatures for @st.cache_data keys.

Streamlit caches by argument value, so a loader keyed only on a directory
``Path`` never re-reads after an out-of-band re-ingest rewrites a data file
— the Factor / Performance tabs then show stale numbers until a manual
cache clear or a server restart (audit WSC-1). Pass
``file_signature(<files>)`` as an extra loader argument: the ``(mtime, size)``
tuple is stable while the files are untouched (so the cache still hits every
rerun) but changes the moment any file is rewritten, added, or removed,
busting the cache automatically.
"""
from __future__ import annotations

from pathlib import Path


def file_signature(*paths) -> tuple:
    """Return a ``((mtime, size), ...)`` tuple, one entry per path in order.

    A missing path contributes the ``(0.0, 0)`` sentinel rather than raising,
    so loaders for not-yet-fetched artifacts (e.g. the Ken French files
    before the first fetch) still cache cleanly.
    """
    sig = []
    for p in paths:
        try:
            stt = Path(p).stat()
            sig.append((stt.st_mtime, stt.st_size))
        except OSError:
            sig.append((0.0, 0))
    return tuple(sig)
