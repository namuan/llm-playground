#!/usr/bin/env -S uv run --quiet --script
# /// script
# dependencies = [
#   "pandas",
#   "matplotlib",
#   "pyarrow",
#   "persistent-cache@git+https://github.com/namuan/persistent-cache",
# ]
# ///
"""
Analyze historical opencode sessions from the local opencode.db SQLite store.

The opencode DB is at ~/.local/share/opencode/opencode.db (XDG path on macOS too)
and has three "core" tables plus a handful of supporting ones:

    session  742 rows  - one row per chat session
    message  27.5k     - one row per message; the interesting fields live in JSON `data`
    part     117k      - one row per message-part (text, tool call, tool result, ...);
                         `data` is also a JSON blob

Gotchas baked in:
  * All `time_*` columns are Unix MILLISECONDS, not seconds.
  * `session.model` and the `data` columns are JSON strings; flatten on load.
  * The DB can be >1 GB and the WAL is live while opencode is running, so cache
    parsed DataFrames keyed on (path, size, mtime_ns) and quit opencode first
    if you intend to back up / copy the file.

Usage:
  ./analyze_opencode_sessions.py -h
  ./analyze_opencode_sessions.py -vv                # debug logging
  ./analyze_opencode_sessions.py --db /custom/path.db
  ./analyze_opencode_sessions.py --csv-dir ./out    # dump parquet/CSV
  ./analyze_opencode_sessions.py --open             # open the HTML report
  ./analyze_opencode_sessions.py --no-report        # skip HTML report

By default an HTML report is written to
    $TMPDIR/opencode-analytics/report.html
(use --report PATH to override, --no-report to skip, --open to launch in the
default browser). All output paths are explicit so the cwd stays clean.
"""
import base64
import io
import json
import logging
import os
import sqlite3
import subprocess
import sys
import tempfile
from argparse import ArgumentParser, RawDescriptionHelpFormatter
from datetime import datetime, timezone
from html import escape
from pathlib import Path

import pandas as pd
from persistent_cache import PersistentCache

# opencode uses XDG-style paths even on macOS
DEFAULT_DB = Path.home() / ".local/share/opencode/opencode.db"
CACHE_DIR = Path.home() / ".cache/opencode-analytics"
REPORT_DIR = Path(tempfile.gettempdir()) / "opencode-analytics"
DEFAULT_REPORT = REPORT_DIR / "report.html"

log = logging.getLogger("opencode_analytics")

# persistent-cache is decorator-only; key is auto-derived from (func, args, kwargs).
# We pass `db_path` explicitly so the signature (size + mtime) lands in the key,
# giving us free cache-invalidation when opencode writes to the DB.
_cache = PersistentCache(cache_dir=CACHE_DIR)


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def setup_logging(verbosity: int) -> None:
    level = logging.WARNING
    if verbosity == 1:
        level = logging.INFO
    elif verbosity >= 2:
        level = logging.DEBUG
    logging.basicConfig(
        handlers=[logging.StreamHandler()],
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=level,
    )
    logging.captureWarnings(capture=True)


def parse_args():
    parser = ArgumentParser(
        description=__doc__, formatter_class=RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "-v", "--verbose", action="count", default=0, help="increase logging verbosity"
    )
    parser.add_argument(
        "--db",
        default=str(DEFAULT_DB),
        help=f"path to opencode.db (default: {DEFAULT_DB})",
    )
    parser.add_argument(
        "--csv-dir",
        default=None,
        help="if set, write sessions/messages/parts parquet (with CSV fallback) into this dir",
    )
    parser.add_argument(
        "--plot",
        default=None,
        help="if set, write a sessions-per-day PNG to this path (also embedded in the HTML report)",
    )
    parser.add_argument(
        "--report",
        default=None,
        metavar="PATH",
        help=f"path for the HTML report (default: {DEFAULT_REPORT})",
    )
    parser.add_argument(
        "--no-report",
        action="store_true",
        help="skip HTML report generation",
    )
    parser.add_argument(
        "--open",
        action="store_true",
        help="open the HTML report in the default app after generation (implies report generation)",
    )
    parser.add_argument(
        "--no-summary",
        action="store_true",
        help="skip the textual summary at the end (useful with --csv-dir/--plot)",
    )
    return parser.parse_args()


def _db_signature(db_path: Path) -> tuple:
    """Cache-key component: (path, size, mtime_ns).

    Including mtime means the cache auto-invalidates whenever opencode
    appends to the DB — no manual `cache clear` needed.
    """
    st = db_path.stat()
    return (str(db_path.resolve()), st.st_size, st.st_mtime_ns)


def _connect(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def _parse_json(value, default=None):
    if value is None or value == "":
        return default
    try:
        return json.loads(value)
    except (json.JSONDecodeError, TypeError):
        return default


def _to_ms_datetime(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, unit="ms", utc=True, errors="coerce")


# --------------------------------------------------------------------------- #
# loaders — each is a thin wrapper; @_cache memoises by (func, args, kwargs).
# The (db_path, db_sig) signature is the cache key. `db_sig` carries size +
# mtime so the cache auto-invalidates when opencode writes to the DB.
# --------------------------------------------------------------------------- #
@_cache
def load_projects(db_path: str, db_sig: tuple) -> pd.DataFrame:
    log.info("loading projects from %s (sig=%s)", db_path, db_sig)
    conn = _connect(Path(db_path))
    try:
        df = pd.read_sql_query("SELECT * FROM project", conn)
    finally:
        conn.close()
    for col in ("time_created", "time_updated", "time_initialized"):
        if col in df.columns:
            df[col] = _to_ms_datetime(df[col])
    return df


@_cache
def load_sessions(db_path: str, db_sig: tuple) -> pd.DataFrame:
    log.info("loading sessions from %s (sig=%s)", db_path, db_sig)
    conn = _connect(Path(db_path))
    try:
        df = pd.read_sql_query(
            """
            SELECT id, project_id, parent_id, slug, directory, title, version,
                   share_url, summary_additions, summary_deletions, summary_files,
                   revert, permission,
                   time_created, time_updated, time_compacting, time_archived,
                   workspace_id, path, agent, model, cost,
                   tokens_input, tokens_output, tokens_reasoning,
                   tokens_cache_read, tokens_cache_write, metadata
            FROM session
            """,
            conn,
        )
    finally:
        conn.close()

    # JSON columns
    df["model"] = df["model"].apply(lambda v: _parse_json(v, {}))
    df["metadata"] = df["metadata"].apply(lambda v: _parse_json(v, {}))
    df["model_id"] = df["model"].apply(
        lambda m: m.get("id") if isinstance(m, dict) else None
    )
    df["model_provider"] = df["model"].apply(
        lambda m: m.get("providerID") if isinstance(m, dict) else None
    )
    for col in ("time_created", "time_updated", "time_compacting", "time_archived"):
        df[col] = _to_ms_datetime(df[col])
    return df


@_cache
def load_messages(db_path: str, db_sig: tuple) -> pd.DataFrame:
    log.info("loading messages from %s (sig=%s)", db_path, db_sig)
    conn = _connect(Path(db_path))
    try:
        df = pd.read_sql_query(
            "SELECT id, session_id, time_created, time_updated, data FROM message",
            conn,
        )
    finally:
        conn.close()

    # Flatten the JSON `data` column. Common nested fields:
    #   role, mode, agent, variant, parentID, modelID, providerID,
    #   path.cwd, path.root,
    #   cost, tokens.input/output/reasoning/cache.read/cache.write,
    #   time.created
    parsed = df["data"].apply(_parse_json)
    flat = pd.json_normalize(parsed).add_prefix("data_")
    df = pd.concat([df.drop(columns=["data"]), flat], axis=1)

    df["time_created"] = _to_ms_datetime(df["time_created"])
    df["time_updated"] = _to_ms_datetime(df["time_updated"])
    if "data_time_created" in df.columns:
        df["data_time_created"] = _to_ms_datetime(df["data_time_created"])

    for c in df.columns:
        if c.startswith("data_tokens") or c in ("data_cost",):
            df[c] = pd.to_numeric(df[c], errors="coerce")

    return df


@_cache
def load_parts(db_path: str, db_sig: tuple) -> pd.DataFrame:
    """Load the `part` table; ~117k rows in the default install.

    Parts are typed (text, tool, ...) and the `data` blob varies by type.
    Useful flattened columns you can rely on:
        data_type           e.g. 'text', 'tool', 'step-start', 'step-finish'
        data_text           text content (for type='text')
        data_tool           tool name (for type='tool')  e.g. 'bash', 'edit', 'read'
        data_callID         tool call id
        data_state.status   'running' | 'completed' | 'error'
        data_state.input    tool input dict (json_normalize flattens)
        data_state.output   tool output string
        data_state.metadata.output   captured stdout (for bash) — THIS is the
                                     >1GB driver; consider .str.len() instead of
                                     materializing the full string
    """
    log.info("loading parts from %s (sig=%s)", db_path, db_sig)
    conn = _connect(Path(db_path))
    try:
        df = pd.read_sql_query(
            "SELECT id, message_id, session_id, time_created, time_updated, data FROM part",
            conn,
        )
    finally:
        conn.close()

    parsed = df["data"].apply(_parse_json)
    flat = pd.json_normalize(parsed).add_prefix("data_")
    df = pd.concat([df.drop(columns=["data"]), flat], axis=1)

    df["time_created"] = _to_ms_datetime(df["time_created"])
    df["time_updated"] = _to_ms_datetime(df["time_updated"])
    for c in ("data_time_start", "data_time_end"):
        if c in df.columns:
            df[c] = _to_ms_datetime(df[c])

    # convenience length columns (cheaper than carrying huge strings around)
    if "data_text" in df.columns:
        df["data_text_len"] = df["data_text"].fillna("").str.len()
    for c in ("data_state.output", "data_state.metadata.output"):
        if c in df.columns:
            df[c.replace(".", "_") + "_len"] = (
                df[c].fillna("").astype(str).str.len()
            )

    return df


@_cache
def load_todos(db_path: str, db_sig: tuple) -> pd.DataFrame:
    log.info("loading todos from %s (sig=%s)", db_path, db_sig)
    conn = _connect(Path(db_path))
    try:
        df = pd.read_sql_query("SELECT * FROM todo", conn)
    finally:
        conn.close()
    df["time_created"] = _to_ms_datetime(df["time_created"])
    df["time_updated"] = _to_ms_datetime(df["time_updated"])
    return df


# --------------------------------------------------------------------------- #
# analysis
# --------------------------------------------------------------------------- #
def print_summary(
    sessions: pd.DataFrame, messages: pd.DataFrame, parts: pd.DataFrame
) -> None:
    bar = "=" * 70
    print(bar)
    print(f"opencode session summary  ·  db: {DEFAULT_DB}")
    print(bar)
    print(f"sessions : {len(sessions):>6,}")
    print(f"messages : {len(messages):>6,}")
    print(f"parts    : {len(parts):>6,}")

    if not sessions.empty:
        first = sessions["time_created"].min()
        last = sessions["time_created"].max()
        span = last - first
        print(f"range    : {first}  →  {last}   (Δ {span})")

    if "model_id" in sessions.columns and not sessions["model_id"].isna().all():
        print("\nmodels used:")
        print(sessions["model_id"].value_counts().to_string(header=False))

    if "directory" in sessions.columns:
        print("\ntop project directories:")
        print(
            sessions["directory"]
            .value_counts()
            .head(5)
            .to_string(header=False)
        )

    if "data_role" in messages.columns and not messages["data_role"].isna().all():
        print("\nmessage roles:")
        print(messages["data_role"].value_counts().to_string(header=False))

    if "data_type" in parts.columns and not parts["data_type"].isna().all():
        print("\npart types:")
        print(parts["data_type"].value_counts().to_string(header=False))

    if "data_tool" in parts.columns and not parts["data_tool"].isna().all():
        print("\ntop tools called:")
        print(
            parts["data_tool"]
            .value_counts()
            .head(10)
            .to_string(header=False)
        )

    if not sessions.empty:
        total_cost = sessions["cost"].sum()
        total_in = sessions["tokens_input"].sum()
        total_out = sessions["tokens_output"].sum()
        total_cache_read = sessions["tokens_cache_read"].sum()
        print(
            f"\ntotals   : cost=${total_cost:,.4f}  "
            f"in_tok={int(total_in):,}  out_tok={int(total_out):,}  "
            f"cache_read={int(total_cache_read):,}"
        )


def _mpl():
    """Lazy import of matplotlib in headless mode."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.style.use("dark_background")
    return plt


def _fig_to_b64(fig) -> str:
    """Render a matplotlib figure to a base64-encoded PNG string."""
    plt = _mpl()
    buf = io.BytesIO()
    fig.savefig(
        buf,
        format="png",
        dpi=100,
        bbox_inches="tight",
        facecolor=fig.get_facecolor(),
    )
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _empty_fig(message: str = "no data"):
    plt = _mpl()
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.text(0.5, 0.5, message, ha="center", va="center", color="#8b949e")
    ax.set_axis_off()
    return fig


# --- per-chart helpers, each returns a base64 PNG ----------------------------
def chart_sessions_per_day(sessions: pd.DataFrame) -> str:
    plt = _mpl()
    if sessions.empty:
        return _fig_to_b64(_empty_fig("no sessions"))
    per_day = sessions.set_index("time_created").resample("D").size()
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.plot(per_day.index, per_day.values, color="#58a6ff", linewidth=1.5)
    ax.fill_between(per_day.index, per_day.values, alpha=0.15, color="#58a6ff")
    ax.set_title("Sessions per day", fontsize=11, pad=8)
    ax.set_ylabel("sessions")
    ax.set_xlabel("")
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    return _fig_to_b64(fig)


def chart_tokens_per_day(sessions: pd.DataFrame) -> str:
    plt = _mpl()
    if sessions.empty or "tokens_input" not in sessions.columns:
        return _fig_to_b64(_empty_fig("no token data"))
    df = sessions.set_index("time_created")[
        ["tokens_input", "tokens_output", "tokens_cache_read"]
    ].resample("D").sum()
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.stackplot(
        df.index,
        df["tokens_input"],
        df["tokens_output"],
        df["tokens_cache_read"],
        labels=["input", "output", "cache read"],
        colors=["#58a6ff", "#3fb950", "#d29922"],
        alpha=0.85,
    )
    ax.set_title("Tokens per day (stacked)", fontsize=11, pad=8)
    ax.set_ylabel("tokens")
    ax.set_xlabel("")
    ax.legend(loc="upper left", fontsize=8, frameon=False)
    ax.grid(True, alpha=0.2)
    ax.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))
    fig.tight_layout()
    return _fig_to_b64(fig)


def chart_cost_per_day(sessions: pd.DataFrame) -> str:
    plt = _mpl()
    if sessions.empty or "cost" not in sessions.columns:
        return _fig_to_b64(_empty_fig("no cost data"))
    per_day = sessions.set_index("time_created")["cost"].resample("D").sum()
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.fill_between(per_day.index, per_day.values, alpha=0.2, color="#3fb950")
    ax.plot(per_day.index, per_day.values, color="#3fb950", linewidth=1.5)
    ax.set_title("Cost per day (USD)", fontsize=11, pad=8)
    ax.set_ylabel("$")
    ax.set_xlabel("")
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    return _fig_to_b64(fig)


def chart_top_models(sessions: pd.DataFrame, n: int = 10) -> str:
    plt = _mpl()
    if "model_id" not in sessions.columns or sessions["model_id"].isna().all():
        return _fig_to_b64(_empty_fig("no model data"))
    counts = sessions["model_id"].value_counts().head(n).sort_values()
    fig, ax = plt.subplots(figsize=(8, max(2, 0.3 * len(counts))))
    ax.barh(counts.index, counts.values, color="#58a6ff")
    ax.set_title(f"Top {n} models", fontsize=11, pad=8)
    ax.set_xlabel("sessions")
    ax.grid(True, alpha=0.2, axis="x")
    fig.tight_layout()
    return _fig_to_b64(fig)


def chart_top_tools(parts: pd.DataFrame, n: int = 10) -> str:
    plt = _mpl()
    if "data_tool" not in parts.columns or parts["data_tool"].isna().all():
        return _fig_to_b64(_empty_fig("no tool data"))
    counts = parts["data_tool"].value_counts().head(n).sort_values()
    fig, ax = plt.subplots(figsize=(8, max(2, 0.3 * len(counts))))
    ax.barh(counts.index, counts.values, color="#a371f7")
    ax.set_title(f"Top {n} tools called", fontsize=11, pad=8)
    ax.set_xlabel("invocations")
    ax.grid(True, alpha=0.2, axis="x")
    fig.tight_layout()
    return _fig_to_b64(fig)


def chart_top_directories(sessions: pd.DataFrame, n: int = 10) -> str:
    plt = _mpl()
    if "directory" not in sessions.columns or sessions["directory"].isna().all():
        return _fig_to_b64(_empty_fig("no directory data"))
    counts = sessions["directory"].value_counts().head(n).sort_values()
    # truncate long paths for display
    labels = [p if len(p) <= 50 else "…" + p[-49:] for p in counts.index]
    fig, ax = plt.subplots(figsize=(8, max(2, 0.3 * len(counts))))
    ax.barh(labels, counts.values, color="#d29922")
    ax.set_title(f"Top {n} project directories", fontsize=11, pad=8)
    ax.set_xlabel("sessions")
    ax.grid(True, alpha=0.2, axis="x")
    fig.tight_layout()
    return _fig_to_b64(fig)


def chart_activity_heatmap(sessions: pd.DataFrame) -> str:
    """Day-of-week × hour-of-day heatmap of session *start* times (UTC)."""
    import numpy as np

    plt = _mpl()
    if sessions.empty:
        return _fig_to_b64(_empty_fig("no session data"))
    starts = sessions["time_created"].dt.tz_convert("UTC")
    dow = starts.dt.dayofweek  # 0=Mon
    hour = starts.dt.hour
    grid = np.zeros((7, 24), dtype=int)
    for d, h in zip(dow, hour):
        grid[d, h] += 1
    fig, ax = plt.subplots(figsize=(10, 3))
    im = ax.imshow(grid, aspect="auto", cmap="viridis", interpolation="nearest")
    ax.set_yticks(range(7), ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"])
    ax.set_xticks(range(0, 24, 2), [f"{h:02d}" for h in range(0, 24, 2)])
    ax.set_xlabel("hour (UTC)")
    ax.set_title("Activity heatmap — session starts", fontsize=11, pad=8)
    fig.colorbar(im, ax=ax, label="sessions", shrink=0.8)
    fig.tight_layout()
    return _fig_to_b64(fig)


# --------------------------------------------------------------------------- #
# HTML report
# --------------------------------------------------------------------------- #
HTML_CSS = """
  :root {
    --bg: #0d1117; --fg: #c9d1d9; --muted: #8b949e;
    --border: #30363d; --accent: #58a6ff; --accent-2: #3fb950;
    --card: #161b22; --code-bg: #21262d;
  }
  * { box-sizing: border-box; }
  body {
    font: 14px/1.5 -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif;
    background: var(--bg); color: var(--fg);
    margin: 0; padding: 2rem;
  }
  .container { max-width: 1200px; margin: 0 auto; }
  h1 { font-size: 1.75rem; margin: 0 0 0.25rem; font-weight: 600; }
  h2 {
    font-size: 0.8rem; margin: 0 0 1rem;
    color: var(--muted); font-weight: 600; text-transform: uppercase; letter-spacing: 0.08em;
  }
  h3 { font-size: 0.95rem; margin: 0 0 0.75rem; font-weight: 600; }
  .meta { color: var(--muted); font-size: 0.85rem; margin-bottom: 1.5rem; }
  .meta span { margin-right: 1.25rem; }
  .meta code { background: var(--code-bg); padding: 0.1em 0.4em; border-radius: 3px; font-size: 0.85em; }
  .cards {
    display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
    gap: 0.75rem; margin: 1rem 0 1.5rem;
  }
  .card {
    background: var(--card); border: 1px solid var(--border);
    border-radius: 6px; padding: 0.85rem 1rem;
    min-width: 0;       /* let the grid track shrink below content width */
  }
  .card .v {
    font-size: clamp(1.2rem, 1.3rem + 0.3vw, 1.6rem);
    font-weight: 600; color: var(--accent);
    font-variant-numeric: tabular-nums;
    line-height: 1.15;
  }
  .card .l { color: var(--muted); font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.05em; margin-top: 0.25rem; }
  section {
    background: var(--card); border: 1px solid var(--border);
    border-radius: 6px; padding: 1.5rem; margin-bottom: 1rem;
  }
  section img { max-width: 100%; height: auto; display: block; margin: 0 auto; }
  table { width: 100%; border-collapse: collapse; font-size: 0.85rem; }
  th, td { padding: 0.45rem 0.6rem; text-align: left; border-bottom: 1px solid var(--border); }
  th { color: var(--muted); font-weight: 600; font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.05em; }
  tr:last-child td { border-bottom: 0; }
  .num { text-align: right; font-variant-numeric: tabular-nums; font-family: ui-monospace, SFMono-Regular, Consolas, monospace; font-size: 0.8rem; }
  .grid-2 { display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; }
  @media (max-width: 800px) { .grid-2 { grid-template-columns: 1fr; } }
  .path { font-family: ui-monospace, SFMono-Regular, Consolas, monospace; font-size: 0.78rem; color: var(--muted); }
  footer { color: var(--muted); font-size: 0.8rem; text-align: center; margin-top: 2rem; }
"""


def _humanize_int(n: int) -> str:
    """Format a count compactly for stat-card display.

        742           -> '742'
        27,635        -> '27.6K'
        117,771       -> '118K'
        10,762,574    -> '10.8M'
        142,980,724   -> '143M'
        1,798,558,997 -> '1.80B'

    Numbers < 10,000 keep full precision so '742' doesn't become '0.7K'.
    Trailing zeros after the decimal are stripped: 1.00B -> '1B'.
    """
    if abs(n) < 10_000:
        return f"{n:,}"
    for threshold, letter in ((10**9, "B"), (10**6, "M"), (10**3, "K")):
        if abs(n) >= threshold:
            v = n / threshold
            if abs(v) < 10:
                num = f"{v:.2f}"
            elif abs(v) < 100:
                num = f"{v:.1f}"
            else:
                num = f"{v:.0f}"
            num = num.rstrip("0").rstrip(".")
            return f"{num}{letter}"
    return f"{n:,}"


def _top_sessions_table(sessions: pd.DataFrame, messages: pd.DataFrame) -> str:
    """Top 10 sessions by message count, joined with cost/tokens."""
    if sessions.empty:
        return "<p class='meta'>no sessions</p>"
    msg_counts = (
        messages.groupby("session_id").size().rename("messages") if not messages.empty else None
    )
    cols = ["title", "directory", "time_created", "cost", "tokens_input", "tokens_output"]
    cols = [c for c in cols if c in sessions.columns]
    df = sessions[cols].copy()
    if msg_counts is not None:
        df = df.join(msg_counts, how="left").fillna({"messages": 0})
        df = df.sort_values("messages", ascending=False).head(10)
    else:
        df = df.sort_values("cost", ascending=False).head(10)

    df["time_created"] = df["time_created"].dt.strftime("%Y-%m-%d %H:%M")
    rows = []
    for _, r in df.iterrows():
        cells = [
            escape(str(r.get("title", ""))[:60]),
            f"<span class='path'>{escape(str(r.get('directory', ''))[-50:])}</span>",
            escape(str(r.get("time_created", ""))),
            f"<span class='num'>${r.get('cost', 0):.4f}</span>",
            f"<span class='num'>{int(r.get('tokens_input', 0)):,}</span>",
            f"<span class='num'>{int(r.get('tokens_output', 0)):,}</span>",
        ]
        if "messages" in df.columns:
            cells.append(f"<span class='num'>{int(r['messages']):,}</span>")
        rows.append("<tr>" + "".join(f"<td>{c}</td>" for c in cells) + "</tr>")

    headers = ["title", "directory", "started", "cost", "in tok", "out tok"]
    if "messages" in df.columns:
        headers.append("msgs")
    header_html = "".join(f"<th>{h}</th>" for h in headers)
    return f"<table><thead><tr>{header_html}</tr></thead><tbody>{''.join(rows)}</tbody></table>"


def generate_html_report(
    sessions: pd.DataFrame,
    messages: pd.DataFrame,
    parts: pd.DataFrame,
    db_path: Path,
    out_path: Path,
) -> Path:
    """Render an HTML report to `out_path` and return that path."""
    out_path.parent.mkdir(parents=True, exist_ok=True)

    log.info("rendering charts")
    charts = {
        "sessions_per_day": chart_sessions_per_day(sessions),
        "tokens_per_day": chart_tokens_per_day(sessions),
        "cost_per_day": chart_cost_per_day(sessions),
        "top_models": chart_top_models(sessions),
        "top_tools": chart_top_tools(parts),
        "top_directories": chart_top_directories(sessions),
        "activity_heatmap": chart_activity_heatmap(sessions),
    }

    # summary numbers
    n_sessions = len(sessions)
    n_messages = len(messages)
    n_parts = len(parts)
    total_cost = float(sessions["cost"].sum()) if not sessions.empty else 0.0
    total_in = int(sessions["tokens_input"].sum()) if not sessions.empty else 0
    total_out = int(sessions["tokens_output"].sum()) if not sessions.empty else 0
    total_cache = int(sessions["tokens_cache_read"].sum()) if not sessions.empty else 0
    if not sessions.empty:
        first = sessions["time_created"].min().strftime("%Y-%m-%d")
        last = sessions["time_created"].max().strftime("%Y-%m-%d")
        date_range = f"{first} → {last}"
    else:
        date_range = "—"

    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    html = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>opencode analytics — {escape(generated_at)}</title>
<style>{HTML_CSS}</style>
</head>
<body>
<div class="container">
  <header>
    <h1>opencode analytics</h1>
    <div class="meta">
      <span>db <code>{escape(str(db_path))}</code></span>
      <span>generated {escape(generated_at)}</span>
      <span>range {escape(date_range)}</span>
    </div>
  </header>

  <div class="cards">
    <div class="card"><div class="v">{_humanize_int(n_sessions)}</div><div class="l">sessions</div></div>
    <div class="card"><div class="v">{_humanize_int(n_messages)}</div><div class="l">messages</div></div>
    <div class="card"><div class="v">{_humanize_int(n_parts)}</div><div class="l">parts</div></div>
    <div class="card"><div class="v">${total_cost:,.2f}</div><div class="l">total cost</div></div>
    <div class="card"><div class="v">{_humanize_int(total_in)}</div><div class="l">input tokens</div></div>
    <div class="card"><div class="v">{_humanize_int(total_out)}</div><div class="l">output tokens</div></div>
    <div class="card"><div class="v">{_humanize_int(total_cache)}</div><div class="l">cache read</div></div>
  </div>

  <section>
    <h2>Activity over time</h2>
    <div class="grid-2">
      <div><h3>Sessions per day</h3><img alt="" src="data:image/png;base64,{charts['sessions_per_day']}"></div>
      <div><h3>Tokens per day</h3><img alt="" src="data:image/png;base64,{charts['tokens_per_day']}"></div>
    </div>
  </section>

  <section>
    <h2>Cost</h2>
    <img alt="" src="data:image/png;base64,{charts['cost_per_day']}">
  </section>

  <section>
    <h2>Tools &amp; models</h2>
    <div class="grid-2">
      <div><h3>Top models</h3><img alt="" src="data:image/png;base64,{charts['top_models']}"></div>
      <div><h3>Top tools</h3><img alt="" src="data:image/png;base64,{charts['top_tools']}"></div>
    </div>
  </section>

  <section>
    <h2>When do you use opencode?</h2>
    <p class="meta">Session starts by day-of-week × hour-of-day (UTC)</p>
    <img alt="" src="data:image/png;base64,{charts['activity_heatmap']}">
  </section>

  <section>
    <h2>Top project directories</h2>
    <img alt="" src="data:image/png;base64,{charts['top_directories']}">
  </section>

  <section>
    <h2>Top sessions</h2>
    {_top_sessions_table(sessions, messages)}
  </section>

  <footer>Generated by analyze_opencode_sessions.py</footer>
</div>
</body>
</html>
"""
    out_path.write_text(html, encoding="utf-8")
    log.info("wrote HTML report → %s (%d KB)", out_path, len(html) // 1024)
    return out_path


def open_in_default_app(path: Path) -> None:
    """Open `path` in the OS default app (browser for HTML)."""
    path = path.resolve()
    try:
        if sys.platform == "darwin":
            subprocess.Popen(["open", str(path)])
        elif sys.platform == "win32":
            os.startfile(str(path))  # type: ignore[attr-defined]
        else:
            subprocess.Popen(["xdg-open", str(path)])
    except Exception as e:
        log.error("failed to open %s: %s", path, e)


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main(args) -> None:
    db = Path(args.db).expanduser()
    if not db.exists():
        raise SystemExit(f"opencode.db not found at {db}")

    log.info("loading from %s", db)
    db_sig = _db_signature(db)
    sessions = load_sessions(str(db), db_sig)
    messages = load_messages(str(db), db_sig)
    parts = load_parts(str(db), db_sig)
    log.info(
        "loaded sessions=%d messages=%d parts=%d",
        len(sessions),
        len(messages),
        len(parts),
    )

    if args.csv_dir:
        out = Path(args.csv_dir).expanduser()
        out.mkdir(parents=True, exist_ok=True)
        # parquet is much smaller + faster than CSV for the parts table
        try:
            sessions.to_parquet(out / "sessions.parquet", index=False)
            messages.to_parquet(out / "messages.parquet", index=False)
            parts.to_parquet(out / "parts.parquet", index=False)
            log.info("wrote parquet → %s", out)
        except Exception as e:  # pyarrow missing or schema issue
            log.warning("parquet export failed (%s); falling back to CSV", e)
            sessions.to_csv(out / "sessions.csv", index=False)
            messages.to_csv(out / "messages.csv", index=False)
            parts.to_csv(out / "parts.csv", index=False)

    if args.plot:
        out = Path(args.plot).expanduser()
        out.parent.mkdir(parents=True, exist_ok=True)
        # also reuse the same chart-helper used by the HTML report
        from matplotlib import pyplot as plt  # noqa: F401  (already imported via _mpl)

        fig_b64 = chart_sessions_per_day(sessions)
        out.write_bytes(base64.b64decode(fig_b64))
        log.info("wrote plot → %s", out)

    # --open implies report generation, but explicit --no-report still wins
    want_report = (not args.no_report) or args.open
    if want_report:
        if args.no_report and args.open:
            log.warning("--open passed with --no-report; skipping report (and open)")
        else:
            report_path = (
                Path(args.report).expanduser() if args.report else DEFAULT_REPORT
            )
            generate_html_report(sessions, messages, parts, db, report_path)
            if args.open:
                open_in_default_app(report_path)

    if not args.no_summary:
        print_summary(sessions, messages, parts)


if __name__ == "__main__":
    args = parse_args()
    setup_logging(args.verbose)
    main(args)
