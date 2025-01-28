#!/usr/bin/env python3
"""
Evaluate all Ollama models

Usage:
./ollama-model-eval.py -h

$ python3 ollama-model-eval.py --question "Show me the git command to revert the changes from the last two git commits"
"""

import json
from argparse import ArgumentParser
from argparse import RawDescriptionHelpFormatter

import requests

from logger import setup_logging
from providers.ollama import OllamaProvider


def parse_args():
    parser = ArgumentParser(
        description=__doc__, formatter_class=RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=0,
        dest="verbose",
        help="Increase verbosity of logging output",
    )
    parser.add_argument(
        "-q",
        "--question",
        type=str,
        required=True,
        help="The question to ask",
    )

    return parser.parse_args()


def main(args):
    question = args.question
    ollama = OllamaProvider()
    models = ollama.get_available_models()
    ignored_models = ["bakllava:latest", "llava:34b"]
    for model in models:
        if model in ignored_models:
            continue
        print("")
        print("-" * 100)
        print("Evaluating model: " + model)
        print("-" * 100)

        url = "http://localhost:11434/api/generate"
        payload = {
            "model": model,
            "prompt": "Answer the following question: " + question,
            "stream": True,
        }
        headers = {"Content-Type": "application/json"}
        try:
            with requests.post(
                url, json=payload, headers=headers, stream=True
            ) as response:
                response.raise_for_status()
                for line in response.iter_lines():
                    if line:
                        decoded_line = line.decode("utf-8")
                        data = json.loads(decoded_line)["response"]
                        print(data, end="")
        except requests.exceptions.RequestException as e:
            print(f"API request failed: {e}")


if __name__ == "__main__":
    args = parse_args()
    setup_logging(args.verbose)
    main(args)
