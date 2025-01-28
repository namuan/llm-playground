#!/usr/bin/env python3
"""
This script can search for files locally or in a GitHub repository.
It can filter by file extensions, ignore specified directories, and optionally print file contents.

Usage:
python3 llm-context-builder.py --extensions .json .py --ignored_dirs build dist --print_contents > context.py
python3 llm-context-builder.py --github_url https://github.com/motion-canvas/motion-canvas/tree/main/packages/docs/docs --extensions .md .mdx  --print_contents > motion-canvas.md
"""

import logging
import os
import shutil
import tempfile
import urllib.request
from argparse import ArgumentParser
from argparse import RawDescriptionHelpFormatter
from urllib.parse import urljoin
from urllib.parse import urlparse
from zipfile import ZipFile

from logger import setup_logging


def parse_github_url(url):
    parsed_url = urlparse(url)
    if parsed_url.netloc != "github.com":
        raise ValueError("Not a valid GitHub URL")

    path_parts = parsed_url.path.strip("/").split("/")
    if len(path_parts) < 2:
        raise ValueError("URL doesn't contain a valid repository path")

    repo_url = urljoin(
        f"{parsed_url.scheme}://{parsed_url.netloc}", "/".join(path_parts[:2])
    )

    if len(path_parts) < 4 or path_parts[2] != "tree":
        return repo_url, None, None

    branch_name = path_parts[3]
    folder_path = "/" + "/".join(path_parts[4:]) if len(path_parts) > 4 else None

    return repo_url, branch_name, folder_path


def build_zip_url(repo_url, branch):
    return f"{repo_url}/archive/{branch}.zip"


def download_and_extract_repo(zip_url, target_folder):
    with tempfile.TemporaryDirectory() as temp_dir:
        zip_path = os.path.join(temp_dir, "repo.zip")

        with urllib.request.urlopen(zip_url) as response:
            with open(zip_path, "wb") as f:
                f.write(response.read())

        with ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(temp_dir)

        extracted_folder = next(os.walk(temp_dir))[1][0]
        extracted_path = os.path.join(temp_dir, extracted_folder)

        def copy_and_overwrite(src, dst):
            if os.path.isdir(dst):
                shutil.rmtree(dst)
            elif os.path.exists(dst):
                os.remove(dst)
            shutil.copy2(src, dst)

        shutil.copytree(
            extracted_path,
            target_folder,
            dirs_exist_ok=True,
            copy_function=copy_and_overwrite,
        )

    return target_folder


def find_files(directory, extensions, ignored_dirs, print_contents):
    for root, dirs, files in os.walk(directory):
        dirs[:] = [d for d in dirs if d not in ignored_dirs]

        for file in files:
            if any(file.endswith(ext) for ext in extensions):
                file_path = os.path.join(root, file)
                logging.info(f"Found file: {file_path}")

                if print_contents:
                    try:
                        with open(file_path, encoding="utf-8") as f:
                            print(f"# File: {file_path}")
                            print(f.read())
                            print("# " + ("-" * 50))  # Separator
                    except Exception as e:
                        logging.error(f"Error reading file {file_path}: {e}")


def main():
    parser = ArgumentParser(
        description=__doc__, formatter_class=RawDescriptionHelpFormatter
    )
    parser.add_argument("-g", "--github_url", help="GitHub URL to download and search")
    parser.add_argument(
        "-e",
        "--extensions",
        nargs="+",
        required=True,
        help="List of file extensions to search for",
    )
    parser.add_argument(
        "-i",
        "--ignored_dirs",
        nargs="*",
        default=[],
        help="List of directories to ignore",
    )
    parser.add_argument(
        "-p",
        "--print_contents",
        action="store_true",
        help="Flag to print file contents",
    )
    parser.add_argument(
        "-v", "--verbose", action="count", default=0, help="Increase output verbosity"
    )

    args = parser.parse_args()
    setup_logging(args.verbose)

    if args.github_url:
        try:
            repo_url, branch_name, folder_path = parse_github_url(args.github_url)
            if not branch_name:
                branch_name = "main"  # Default branch

            zip_url = build_zip_url(repo_url, branch_name)
            logging.info(f"Downloading repository from: {zip_url}")

            target_folder = os.path.join(os.getcwd(), "downloaded_repo")
            extracted_path = download_and_extract_repo(zip_url, target_folder)
            logging.info(f"Repository downloaded and extracted to: {extracted_path}")

            if folder_path:
                search_path = os.path.join(extracted_path, folder_path.strip("/"))
                if not os.path.exists(search_path):
                    logging.warning(
                        f"Specified folder '{folder_path}' not found in the repository."
                    )
                    return
            else:
                search_path = extracted_path
        except Exception as e:
            logging.error(f"An error occurred while processing GitHub URL: {e}")
            return
    else:
        search_path = "."

    find_files(search_path, args.extensions, args.ignored_dirs, args.print_contents)


if __name__ == "__main__":
    main()
