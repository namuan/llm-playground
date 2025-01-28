#!/usr/bin/env -S uv run --quiet --script
# /// script
# dependencies = [
#   "requests",
#   "beautifulsoup4",
# ]
# ///
"""
This script fetches and parses model tags from Ollama's library page and generates a markdown table.
The table shows model names, sizes, and SHA1 IDs sorted by model base size.

Example:
    python ollama-tags-table.py https://ollama.com/library/deepseek-r1/tags

Output format:
| Models | Size | SHA1 ID |
|--------|------|---------|
| model1 👉 model2 | 1.2GB | abc123... |
"""

import argparse
import re
from collections import defaultdict

import requests
from bs4 import BeautifulSoup


def parse_arguments():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("url", type=str, help="URL to fetch model tags from")
    return parser.parse_args()


def fetch_and_parse(url):
    print(f"Fetching URL: {url}")

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }

    try:
        response = requests.get(url, headers=headers)

        soup = BeautifulSoup(response.text, "html.parser")

        model_info = defaultdict(list)

        tag_entries = soup.find_all("div", class_="flex px-4 py-3")

        for entry in tag_entries:
            model_name_element = entry.find("a", class_="group")
            if not model_name_element:
                continue
            model_name = model_name_element.text.strip()

            info_text = entry.find("span", class_="font-mono")
            if not info_text:
                continue

            sha = info_text.text.strip()
            size = info_text.parent.text.strip().split("•")[1].strip()

            model_info[sha].append((model_name, size))

        print(f"Found {len(model_info)} unique SHA entries")
        return model_info

    except Exception as e:
        print(f"Error occurred: {str(e)}")
        return defaultdict(list)


def convert_size_to_gb(size_str):
    number = float(size_str[:-2])
    if size_str.endswith("TB"):
        return number * 1024
    return number


def sort_by_model_size(models):
    def get_base_size(model_name):
        match = re.match(r"(\d+\.?\d*)b", model_name)
        return float(match.group(1)) if match else float("inf")

    first_model = models[0][0] if models else ""
    return get_base_size(first_model)


def generate_markdown_table(url):
    model_info = fetch_and_parse(url)

    if not model_info:
        print("No model information found!")
        return "No data available"

    sorted_shas = sorted(
        model_info.keys(), key=lambda sha: sort_by_model_size(model_info[sha])
    )

    table = "\n\n| Models | Size | SHA1 ID |\n|----------|---------|-------|\n"

    for sha in sorted_shas:
        models = model_info[sha]
        model_names = " 👉 ".join(model[0] for model in models)
        size = models[0][1]
        table += f"| {model_names} | {size} | {sha} |\n"

    return table


if __name__ == "__main__":
    args = parse_arguments()
    table = generate_markdown_table(args.url)

    print(table)
