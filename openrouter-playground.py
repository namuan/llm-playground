#!/usr/bin/env -S uv run --quiet --script
# /// script
# dependencies = [
#   "litellm",
# ]
# ///
import os
from argparse import ArgumentParser
from argparse import RawDescriptionHelpFormatter

import litellm

from logger import setup_logging


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
    return parser.parse_args()


models_to_try = [
    "google/gemini-2.0-flash-thinking-exp:free",
    "google/gemini-2.0-flash-exp:free",
    "google/gemini-exp-1206:free",
    "google/gemini-exp-1121:free",
    "google/learnlm-1.5-pro-experimental:free",
    "google/gemini-exp-1114:free",
    "meta-llama/llama-3.2-3b-instruct:free",
    "meta-llama/llama-3.2-1b-instruct:free",
    "google/gemini-flash-1.5-exp",
    "google/gemini-flash-1.5-8b-exp",
    "google/gemini-pro-1.5-exp",
    "meta-llama/llama-3.1-405b-instruct:free",
    "meta-llama/llama-3.1-8b-instruct:free",
    "meta-llama/llama-3.1-70b-instruct:free",
    "qwen/qwen-2-7b-instruct:free",
    "google/gemma-2-9b-it:free",
    "mistralai/mistral-7b-instruct:free",
    "microsoft/phi-3-mini-128k-instruct:free",
    "microsoft/phi-3-medium-128k-instruct:free",
    "meta-llama/llama-3-8b-instruct:free",
    "openchat/openchat-7b:free",
    "undi95/toppy-m-7b:free",
    "huggingfaceh4/zephyr-7b-beta:free",
    "gryphe/mythomax-l2-13b:free",
]


def main(args):
    ork = os.getenv("OPENROUTER_API_KEY")

    for model in models_to_try:
        print(f"Using {model=}")
        response = litellm.completion(
            model=f"openrouter/{model}",
            messages=[{"role": "user", "content": "What is the meaning of life?"}],
            api_base="https://openrouter.ai/api/v1",
            api_key=ork,
        )
        if hasattr(response, "choices"):
            print(response)
            break  # Exit the loop if successful

        if hasattr(response, "error") and response.error.get("code") == 429:
            print(f"Error with {model=}")
            continue  # Try the next model


if __name__ == "__main__":
    args = parse_args()
    setup_logging(args.verbose)
    main(args)
