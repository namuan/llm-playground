#!/usr/bin/env -S uv run --quiet --script
# /// script
# dependencies = [
#   "openai",
# ]
# ///
"""
A simple script

Usage:
./template_py_scripts.py -h

./template_py_scripts.py -v # To log INFO messages
./template_py_scripts.py -vv # To log DEBUG messages
"""
import logging
import os
from argparse import ArgumentParser, RawDescriptionHelpFormatter


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


from openai import OpenAI

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
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=ork,
    )

    for model in models_to_try:
        print(f"Using {model=}")
        completion = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "What is the meaning of life?"}],
        )
        if hasattr(completion, 'choices'):
            print(completion)
            break  # Exit the loop if successful

        if hasattr(completion, 'error') and completion.error.get('code') == 429:
            print(f"Error with {model=}")
            continue  # Try the next model



if __name__ == "__main__":
    args = parse_args()
    setup_logging(args.verbose)
    main(args)
