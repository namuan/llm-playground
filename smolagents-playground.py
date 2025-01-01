#!/usr/bin/env -S uv run --quiet --script
# /// script
# dependencies = [
#   "smolagents",
# ]
# ///
from argparse import ArgumentParser, RawDescriptionHelpFormatter

from smolagents import CodeAgent, DuckDuckGoSearchTool, LiteLLMModel

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


def main(args):
    agent = CodeAgent(
        tools=[DuckDuckGoSearchTool()],
        model=LiteLLMModel(
            model_id="ollama_chat/qwen2.5-coder:14b",
            api_base="http://localhost:11434", # replace with remote open-ai compatible server if necessary
            api_key="your-api-key" # replace with API key if necessary
        ),
    )
    agent.run(
        "How many seconds would it take for a leopard at full speed to run through Pont des Arts?"
    )


if __name__ == "__main__":
    args = parse_args()
    setup_logging(args.verbose)
    main(args)