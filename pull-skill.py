#!/usr/bin/env -S uv run --quiet --script
# /// script
# dependencies = [
#   "requests",
# ]
# ///
"""
Pull a skill folder from a GitHub repository.

Usage:
./pull-skill.py https://github.com/dbosk/claude-skills/tree/main/update-project-docs

This will copy all files and folders recursively from the specified GitHub
directory to /Users/nnn/workspace/namuan/agents/skills/<folder-name>
"""

import logging
import os
import re
import sys
from argparse import ArgumentParser, RawDescriptionHelpFormatter
from pathlib import Path

import requests

DEFAULT_OUTPUT_DIR = Path.home() / "workspace/namuan/agents/skills"


def setup_logging(verbosity):
    logging_level = logging.WARNING
    if verbosity == 1:
        logging_level = logging.INFO
    elif verbosity >= 2:
        logging_level = logging.DEBUG

    logging.basicConfig(
        handlers=[
            logging.StreamHandler(),
        ],
        format="%(asctime)s - %(filename)s:%(lineno)d - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging_level,
    )
    logging.captureWarnings(capture=True)


def parse_github_url(url):
    """
    Parse a GitHub URL and extract owner, repo, branch, and path.

    Supports formats:
    - https://github.com/owner/repo/tree/branch/path/to/folder
    - https://github.com/owner/repo/blob/branch/path/to/file

    Returns dict with keys: owner, repo, branch, path, folder_name
    """
    # Pattern for GitHub tree/blob URLs
    pattern = r"https?://github\.com/(?P<owner>[^/]+)/(?P<repo>[^/]+)/(?:tree|blob)/(?P<branch>[^/]+)(?:/(?P<path>.+))?"
    match = re.match(pattern, url)

    if not match:
        raise ValueError(
            f"Invalid GitHub URL format. Expected: https://github.com/owner/repo/tree/branch/path"
        )

    owner = match.group("owner")
    repo = match.group("repo")
    branch = match.group("branch")
    path = match.group("path") or ""

    # The folder name is the first component of the path, or the repo name if no path
    if path:
        folder_name = path.split("/")[-1]
    else:
        folder_name = repo

    return {
        "owner": owner,
        "repo": repo,
        "branch": branch,
        "path": path,
        "folder_name": folder_name,
    }


def fetch_directory_contents(owner, repo, branch, path, session):
    """
    Fetch the contents of a directory from GitHub API.

    Returns a list of items (files and directories).
    """
    api_url = f"https://api.github.com/repos/{owner}/{repo}/contents/{path}"
    params = {"ref": branch}

    logging.debug(f"Fetching: {api_url} with ref={branch}")
    response = session.get(api_url, params=params)
    response.raise_for_status()

    contents = response.json()

    if not isinstance(contents, list):
        # Single file or error
        return [contents] if isinstance(contents, dict) else []

    return contents


def download_file(download_url, local_path, session):
    """Download a file from a URL and save it locally."""
    logging.info(f"Downloading: {local_path}")

    response = session.get(download_url)
    response.raise_for_status()

    local_path.parent.mkdir(parents=True, exist_ok=True)
    local_path.write_bytes(response.content)


def recursively_download(
    owner, repo, branch, path, local_base_path, session, original_path=None
):
    """
    Recursively download all files from a GitHub directory.

    Args:
        owner: GitHub repository owner
        repo: GitHub repository name
        branch: Git branch name
        path: Path in the repository (e.g., "folder/subfolder")
        local_base_path: Local directory to save files
        session: requests.Session for HTTP calls
        original_path: The original path from which relative paths are calculated
    """
    if original_path is None:
        original_path = path

    contents = fetch_directory_contents(owner, repo, branch, path, session)

    for item in contents:
        item_path = item["path"]
        item_type = item["type"]

        if item_type == "file":
            download_url = item.get("download_url")
            if download_url:
                # Calculate relative path from the original path
                relative_path = os.path.relpath(item_path, original_path)
                local_item_path = local_base_path / relative_path
                download_file(download_url, local_item_path, session)
            else:
                logging.warning(f"No download URL for: {item_path}")

        elif item_type == "dir":
            # Calculate the relative directory path and create local subdirectory
            relative_dir = os.path.relpath(item_path, original_path)
            local_subdir = local_base_path / relative_dir
            local_subdir.mkdir(parents=True, exist_ok=True)
            logging.info(f"Entering directory: {item_path}")
            recursively_download(
                owner, repo, branch, item_path, local_base_path, session, original_path
            )

        elif item_type == "symlink":
            logging.warning(f"Skipping symlink: {item_path}")


def parse_args():
    parser = ArgumentParser(
        description=__doc__, formatter_class=RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "url",
        help="GitHub URL to pull from (e.g., https://github.com/owner/repo/tree/branch/path)",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory (default: {DEFAULT_OUTPUT_DIR})",
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


def main(args):
    logging.debug(f"Parsing URL: {args.url}")

    try:
        parsed = parse_github_url(args.url)
    except ValueError as e:
        logging.error(str(e))
        sys.exit(1)

    logging.info(f"Owner: {parsed['owner']}")
    logging.info(f"Repo: {parsed['repo']}")
    logging.info(f"Branch: {parsed['branch']}")
    logging.info(f"Path: {parsed['path']}")
    logging.info(f"Folder name: {parsed['folder_name']}")

    # Create output directory
    output_dir = args.output_dir / parsed["folder_name"]
    output_dir.mkdir(parents=True, exist_ok=True)
    logging.info(f"Output directory: {output_dir}")

    # Create session with headers for GitHub API
    session = requests.Session()
    session.headers.update(
        {
            "Accept": "application/vnd.github.v3+json",
            "User-Agent": "pull-skill-script",
        }
    )

    # Check for GitHub token in environment
    github_token = os.environ.get("GITHUB_TOKEN")
    if github_token:
        session.headers["Authorization"] = f"token {github_token}"
        logging.debug("Using GitHub token for authentication")

    try:
        recursively_download(
            parsed["owner"],
            parsed["repo"],
            parsed["branch"],
            parsed["path"],
            output_dir,
            session,
        )
        logging.info(f"Successfully downloaded skill to: {output_dir}")
    except requests.exceptions.HTTPError as e:
        logging.error(f"HTTP error: {e}")
        sys.exit(1)
    except Exception as e:
        logging.error(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    args = parse_args()
    setup_logging(args.verbose)
    main(args)
