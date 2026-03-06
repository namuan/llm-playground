#!/usr/bin/env -S uv run --quiet --script
# /// script
# dependencies = [
#     "openai",
# ]
# ///

"""
AI CLI helper that answers natural-language questions using any command's --help text.

Works with ANY OpenAI-compatible server (local or remote).

Default: http://localhost:8080/v1
You can pass ANY API key and ANY model name.

Usage examples:

    fd --help | ./some-cli.py "How can I delete files using fd" --model Qwen3-0.6B

    # With environment variables (super convenient)
    export OPENAI_BASE_URL=http://localhost:8080/v1
    export OPENAI_API_KEY=sk-dummy
    export MODEL=dummy-model

    fd --help | ./some-cli.py "How can I delete files using fd"

Setup symlink
$ ln -s $(pwd)/llm-help.py ~/.local/bin/llm-help
"""

import logging
import os
import sys
from argparse import ArgumentParser, RawDescriptionHelpFormatter

from openai import OpenAI


def setup_logging(verbosity: int):
    logging_level = logging.WARNING
    if verbosity == 1:
        logging_level = logging.INFO
    elif verbosity >= 2:
        logging_level = logging.DEBUG
    logging.basicConfig(
        handlers=[logging.StreamHandler()],
        format="%(asctime)s - %(filename)s:%(lineno)d - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging_level,
    )


def get_llm_answer(
    query: str, help_text: str, model: str, base_url: str, api_key: str
) -> str:
    """Call any OpenAI-compatible endpoint (no caching)."""
    client = OpenAI(api_key=api_key, base_url=base_url)

    prompt = f"""You are an expert CLI assistant. The user piped the full `--help` output of a command.

Here is the complete help text:
{help_text}

Question: {query}

Answer concisely:
- Show the exact command + flags needed
- Include one ready-to-copy example
- Only use options that actually appear in the help text
- If the question cannot be answered from the help, say so politely

Answer:"""

    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=1024,
    )
    return response.choices[0].message.content.strip()


def parse_args():
    parser = ArgumentParser(
        description=__doc__, formatter_class=RawDescriptionHelpFormatter
    )

    parser.add_argument(
        "query",
        help='Natural language question, e.g. "How can I delete files using fd"',
    )

    parser.add_argument(
        "--base-url",
        default=None,
        help="OpenAI-compatible base URL (default: http://localhost:8080/v1)",
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help="API key (most local servers accept sk-dummy)",
    )
    parser.add_argument(
        "-m",
        "--model",
        default=None,
        help="Model name (e.g. Qwen3-0.6B). Can also be set via $MODEL env var.",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=0,
        dest="verbose",
        help="Increase verbosity",
    )
    return parser.parse_args()


def main(args):
    help_text = sys.stdin.read().strip()
    if not help_text:
        print(
            "Error: No help text received.\n"
            'Usage:   command --help | ./some-cli.py "your question"',
            file=sys.stderr,
        )
        sys.exit(1)

    # Environment variable fallbacks
    base_url = (
        args.base_url or os.getenv("OPENAI_BASE_URL") or "http://localhost:8080/v1"
    )
    api_key = args.api_key or os.getenv("OPENAI_API_KEY") or "sk-dummy"
    model = args.model or os.getenv("MODEL") or "dummy-model"

    if not model:
        print(
            "Error: No model specified.\n"
            "Use --model Qwen3-0.6B or set:\n"
            "    export MODEL=Qwen3-0.6B",
            file=sys.stderr,
        )
        sys.exit(1)

    logging.info(f"Query: {args.query}")
    logging.info(f"Model: {model} @ {base_url}")

    try:
        answer = get_llm_answer(args.query, help_text, model, base_url, api_key)
        print(answer)
    except Exception as e:
        logging.error(f"Failed to get answer: {e}")
        sys.exit(1)


if __name__ == "__main__":
    parsed_args = parse_args()
    setup_logging(parsed_args.verbose)
    main(parsed_args)
