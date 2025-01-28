#!/usr/bin/env -S uv run --quiet --script
# /// script
# dependencies = [
#   "browser-use==0.1.29",
#   "langchain",
#   "langchain-community",
#   "langchain-ollama",
# ]
# ///
"""
A script to search for flights using browser-use agent

Usage:
./agent-browser-use.py -h

./agent-browser-use.py -v # To log INFO messages
./agent-browser-use.py -vv # To log DEBUG messages
./agent-browser-use.py --model mistral # To specify a different model
"""

import asyncio
import logging
from argparse import ArgumentParser
from argparse import RawDescriptionHelpFormatter
from datetime import datetime
from datetime import timedelta

from browser_use import Agent
from langchain_ollama import ChatOllama


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
    parser.add_argument(
        "--model",
        default="llama3.1:latest",
        help="Name of the Ollama model to use",
    )
    return parser.parse_args()


async def run_agent(model_name):
    llm = ChatOllama(model=model_name, num_ctx=128000)
    # llm = ChatOpenAI(
    #     base_url="http://localhost:11434/v1",
    #     model="qwen2.5:latest",
    #     api_key=SecretStr("foo"),
    #     temperature=0.0,
    # )
    # llm = ChatOpenAI(model="gpt-4o", temperature=0.0)
    tomorrow = (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")
    agent = Agent(
        task=f"""
        Start with Google Search.
        Accept all cookies.
        Find a one-way flight from London to Dubai flying {tomorrow} on Google Flights.
        Return me the cheapest option.
        Show me all the details including flight time, duration and ticket price.
        """,
        llm=llm,
        use_vision=True,
    )
    logging.info(f"Starting agent task with model: {model_name}")
    result = await agent.run()
    logging.info("Agent task completed")
    print(result)


def main(args):
    logging.debug(f"Verbosity level: {args.verbose}")
    logging.debug(f"Using model: {args.model}")
    asyncio.run(run_agent(args.model))


if __name__ == "__main__":
    args = parse_args()
    setup_logging(args.verbose)
    main(args)
