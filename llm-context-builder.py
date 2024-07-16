#!/usr/bin/env python3
"""
Usage:
python3 llm-context-builder.py -e .txt .py -i venv node_modules -p

This example will search for .txt and .py files,
ignore the venv and node_modules directories,
and print the contents of the found files.
"""
import os
from argparse import ArgumentParser, RawDescriptionHelpFormatter


def find_files(extensions, ignored_dirs, print_contents):
    for root, dirs, files in os.walk("."):
        # Remove ignored directories
        dirs[:] = [d for d in dirs if d not in ignored_dirs]

        for file in files:
            if any(file.endswith(ext) for ext in extensions):
                file_path = os.path.join(root, file)
                print(f"Found file: {file_path}")

                if print_contents:
                    try:
                        with open(file_path) as f:
                            print("File contents:")
                            print(f.read())
                            print("-" * 50)  # Separator
                    except Exception as e:
                        print(f"Error reading file: {e}")


def main():
    parser = ArgumentParser(
        description=__doc__, formatter_class=RawDescriptionHelpFormatter
    )
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

    args = parser.parse_args()

    find_files(args.extensions, args.ignored_dirs, args.print_contents)


if __name__ == "__main__":
    main()
