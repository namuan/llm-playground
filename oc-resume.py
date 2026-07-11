#!/usr/bin/env -S uv run --quiet --script
# /// script
# dependencies = []
# ///
"""
Resume the most recent OpenCode session for a given directory.

Queries the OpenCode SQLite database to find all non-archived sessions
whose working directory matches the current (or specified) directory,
then opens the most recently updated one.

Usage:
  ./oc-resume.py                  # Resume latest session for cwd
  ./oc-resume.py -d /some/path    # Resume latest session for a specific dir
  ./oc-resume.py --list           # List all matching sessions (no open)
  ./oc-resume.py --dry-run        # Show which session would be opened
  ./oc-resume.py -v               # Verbose logging
"""
import logging
import os
import sqlite3
import subprocess
import sys
from argparse import ArgumentParser, RawDescriptionHelpFormatter
from pathlib import Path

DB_PATH = Path.home() / ".local" / "share" / "opencode" / "opencode.db"


def setup_logging(verbosity):
    level = logging.WARNING
    if verbosity == 1:
        level = logging.INFO
    elif verbosity >= 2:
        level = logging.DEBUG

    logging.basicConfig(
        handlers=[logging.StreamHandler()],
        format="%(asctime)s - %(filename)s:%(lineno)d - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=level,
    )
    logging.captureWarnings(capture=True)


def parse_args():
    parser = ArgumentParser(
        description=__doc__, formatter_class=RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "-d",
        "--directory",
        default=None,
        help="Directory to look up sessions for (default: current directory)",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List all matching sessions instead of opening one",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show which session would be opened, but do not launch it",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=0,
        dest="verbose",
        help="Increase verbosity of logging output",
    )
    return parser.parse_args()


def find_sessions(directory: str, limit: int | None = None) -> list[dict]:
    """Return non-archived sessions for the given directory, newest first."""
    if not DB_PATH.exists():
        logging.error("OpenCode database not found at %s", DB_PATH)
        return []

    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    query = """
        SELECT id, slug, title, directory, time_created, time_updated
        FROM session
        WHERE directory = ?
          AND time_archived IS NULL
        ORDER BY time_updated DESC
    """
    if limit is not None:
        query += " LIMIT ?"
        cursor.execute(query, (directory, limit))
    else:
        cursor.execute(query, (directory,))

    rows = [dict(row) for row in cursor.fetchall()]
    conn.close()

    logging.debug("Found %d session(s) for directory %s", len(rows), directory)
    return rows


def open_session(session_id: str, fork: bool = False):
    """Launch opencode with the given session ID."""
    cmd = ["opencode", "-s", session_id]
    if fork:
        cmd.append("--fork")

    logging.info("Launching: %s", " ".join(cmd))
    subprocess.run(cmd)


def main(args):
    directory = os.path.abspath(args.directory) if args.directory else os.getcwd()
    directory = os.path.realpath(directory)  # Resolve symlinks
    logging.info("Looking up sessions for: %s", directory)

    if not DB_PATH.exists():
        print(f"OpenCode database not found at: {DB_PATH}", file=sys.stderr)
        print("Is OpenCode installed? Run 'opencode' at least once to create the database.", file=sys.stderr)
        sys.exit(1)

    sessions = find_sessions(directory, limit=None if args.list else 10)

    if not sessions:
        print(f"No existing non-archived OpenCode sessions found for: {directory}")
        sys.exit(0)

    if args.list:
        print(f"Sessions for {directory}:\n")
        for s in sessions:
            print(f"  {s['id']}  {s['slug']:20s}  {s['title']}")
        return

    # Pick the most recent
    latest = sessions[0]

    if args.dry_run:
        print(f"Would resume: {latest['id']} ({latest['slug']})")
        print(f"  Title:     {latest['title']}")
        print(f"  Directory: {latest['directory']}")
        print(f"  Updated:   {latest['time_updated']}")
        if len(sessions) > 1:
            print(f"\n  ({len(sessions)} total matching sessions — this is the newest)")
        return

    print(f"Resuming session: {latest['slug']} ({latest['title']})")
    if len(sessions) > 1:
        logging.info(
            "(%d other matching sessions exist — using the most recent)",
            len(sessions) - 1,
        )
    open_session(latest["id"])


if __name__ == "__main__":
    args = parse_args()
    setup_logging(args.verbose)
    main(args)
