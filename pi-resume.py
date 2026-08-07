#!/usr/bin/env -S uv run --quiet --script
# /// script
# dependencies = []
# ///
"""
Resume the most recent Pi session for a given directory.

Pi stores sessions as JSONL files under ~/.pi/agent/sessions/, organized
by working directory (the directory name is the absolute path with '/'
replaced by '-', wrapped in --...--). Each session is one .jsonl file
named <timestamp>_<session-uuid>.jsonl. There is no database -- we just
scan the directory and pick the newest file.

Usage:
  ./pi-resume.py                  # Resume latest session for cwd
  ./pi-resume.py -d /some/path    # Resume latest session for a specific dir
  ./pi-resume.py --list           # List all matching sessions (no open)
  ./pi-resume.py --dry-run        # Show which session would be opened
  ./pi-resume.py --fork           # Fork the latest session instead of resuming
  ./pi-resume.py -v               # Verbose logging
"""
import json
import logging
import os
import subprocess
import sys
from argparse import ArgumentParser, RawDescriptionHelpFormatter
from pathlib import Path

DEFAULT_SESSION_DIR = Path.home() / ".pi" / "agent" / "sessions"


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
        "-f",
        "--fork",
        action="store_true",
        help="Fork the latest session into a new session (pi --fork)",
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


def session_dir() -> Path:
    """Resolve the sessions root, honoring the pi environment override."""
    override = os.environ.get("PI_CODING_AGENT_SESSION_DIR")
    return Path(override) if override else DEFAULT_SESSION_DIR


def encode_dir(path: str) -> str:
    """Encode an absolute path into pi's per-directory folder name.

    Pi strips the leading '/' from the absolute path, replaces '/' with '-',
    then wraps the result in --...--. E.g. /Users/me/proj -> --Users-me-proj--.
    """
    return "--" + path.lstrip("/").replace("/", "-") + "--"


def extract_text(content) -> str | None:
    """Pull the first text block out of a message content field."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                return block.get("text")
    return None


def parse_session(file: Path) -> dict:
    """Extract id, title, and created timestamp from a session JSONL file."""
    session_id = None
    name = None
    first_user_text = None

    with open(file, "r", encoding="utf-8") as fh:
        for line in fh:
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            etype = entry.get("type")
            if etype == "session":
                session_id = entry.get("id") or file.stem.split("_")[-1]
            elif etype == "session_info" and entry.get("name"):
                name = entry["name"]  # /name display name overrides first message
            elif etype == "message" and first_user_text is None:
                msg = entry.get("message", {})
                if msg.get("role") == "user":
                    first_user_text = extract_text(msg.get("content"))

    title = name or first_user_text or ""
    # Keep titles on one line for clean listing output
    title = " ".join(title.split())[:100]
    return {"id": session_id, "title": title}


def find_sessions(directory: str, session_root: Path) -> list[dict]:
    """Return sessions for the given directory, newest first (by mtime)."""
    proj_dir = session_root / encode_dir(os.path.realpath(directory))
    if not proj_dir.is_dir():
        logging.debug("No session folder: %s", proj_dir)
        return []

    sessions = []
    for file in sorted(proj_dir.glob("*.jsonl"), key=lambda p: p.stat().st_mtime, reverse=True):
        try:
            info = parse_session(file)
        except OSError as exc:
            logging.warning("Skipping %s: %s", file.name, exc)
            continue
        info["file"] = file
        info["updated"] = file.stat().st_mtime
        sessions.append(info)

    logging.debug("Found %d session(s) for directory %s", len(sessions), directory)
    return sessions


def open_session(file: Path, fork: bool = False):
    """Launch pi with the given session file."""
    flag = "--fork" if fork else "--session"
    cmd = ["pi", flag, str(file)]

    logging.info("Launching: %s", " ".join(cmd))
    subprocess.run(cmd)


def main(args):
    session_root = session_dir()
    directory = os.path.abspath(args.directory) if args.directory else os.getcwd()
    directory = os.path.realpath(directory)  # Resolve symlinks
    logging.info("Looking up sessions for: %s", directory)

    if not session_root.is_dir():
        print(f"Pi session directory not found at: {session_root}", file=sys.stderr)
        print("Is pi installed? Run 'pi' at least once to create sessions.", file=sys.stderr)
        sys.exit(1)

    sessions = find_sessions(directory, session_root)

    if not sessions:
        print(f"No existing Pi sessions found for: {directory}")
        sys.exit(0)

    if args.list:
        print(f"Sessions for {directory}:\n")
        for s in sessions:
            stamp = __import__("datetime").datetime.fromtimestamp(s["updated"]).strftime("%Y-%m-%d %H:%M")
            print(f"  {s['id']}  {stamp}  {s['title']}")
        return

    latest = sessions[0]

    if args.dry_run:
        verb = "Fork" if args.fork else "Resume"
        print(f"Would {verb.lower()}: {latest['id']}")
        print(f"  Title:   {latest['title']}")
        print(f"  File:    {latest['file']}")
        print(f"  Updated: {__import__('datetime').datetime.fromtimestamp(latest['updated'])}")
        if len(sessions) > 1:
            print(f"\n  ({len(sessions)} total matching sessions — this is the newest)")
        return

    verb = "Forking" if args.fork else "Resuming"
    print(f"{verb} session: {latest['id']} ({latest['title']})")
    if len(sessions) > 1:
        logging.info(
            "(%d other matching sessions exist — using the most recent)",
            len(sessions) - 1,
        )
    open_session(latest["file"], fork=args.fork)


if __name__ == "__main__":
    args = parse_args()
    setup_logging(args.verbose)
    main(args)
