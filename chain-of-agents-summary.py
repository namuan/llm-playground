#!/usr/bin/env -S uv run --quiet --script
# /// script
# dependencies = [
#   "pandas",
# ]
# ///
"""
A Chain of Agents Summary Generator
"""
import argparse
from pathlib import Path


# Input Layer: Handles chunking and task identification
class InputChunker:
    def __init__(self, context_window_size: int):
        self.context_window_size = context_window_size

    def chunk_input(self, text: str) -> list:
        """Splits the input text into chunks based on the context window size and paragraph boundaries."""


# Control Layer: Manages the task and flow between agents
class TaskScheduler:
    def __init__(self, workers: list, manager):
        self.workers = workers
        self.manager = manager

    def schedule_task(self, text_chunks: list):
        """Assigns chunks to worker agents in sequence and manages communication between them."""


# Worker Agent: Processes individual chunks and generates Communication Units (CU)
class WorkerAgent:
    def __init__(self, model):
        self.model = model

    def process_chunk(self, chunk: str, previous_cu: dict = None) -> dict:
        """Processes the given chunk using the LLM and returns a communication unit (CU)."""


# Manager Agent: Collects CUs and synthesizes the final output
class ManagerAgent:
    def __init__(self, model):
        self.model = model

    def synthesize_output(self, communication_units: list) -> str:
        """Synthesizes the communication units from worker agents into a final output."""


# Error Handling: Manages retries and skipping in case of failures
class ErrorHandler:
    def __init__(self, retry_limit: int):
        self.retry_limit = retry_limit

    def handle_error(self, worker: WorkerAgent, chunk: str):
        """Retries the worker agent processing and skips the chunk after reaching the retry limit."""


class CLI:
    def __init__(self, input_text: str, output_file: Path):
        self.input_text = input_text  # This will be set in main() based on input file
        self.output_file = output_file  # This will be set in main() if provided

    def get_input(self) -> str:
        """Returns the input text for the system."""
        return self.input_text

    def display_output(self, output: str):
        """Displays the final summarized output, or saves it to a file if an output file is specified."""
        if self.output_file:
            with open(self.output_file, "w") as file:
                file.write(output)
            print(f"Summarized output saved to {self.output_file}")
        else:
            print("Summarized Output:")
            print(output)


# Logging Layer: Handles logging at different levels
class Logger:
    def __init__(self, log_level: str):
        self.log_level = log_level

    def log(self, message: str, level: str = "INFO"):
        """Logs messages with specified logging levels (DEBUG, INFO, ERROR)."""


# Main System Class to orchestrate the entire process
class ChainOfAgentsSummarizationSystem:
    def __init__(self, input_chunker: InputChunker, scheduler: TaskScheduler, cli: CLI):
        self.input_chunker = input_chunker
        self.scheduler = scheduler
        self.cli = cli

    def start(self):
        """Starts the chain-of-agents system for summarization."""

        # Step 1: Get the input text from the user via the CLI
        input_text = self.cli.get_input()

        # Step 2: Chunk the input text using the InputChunker
        text_chunks = self.input_chunker.chunk_input(input_text)

        # Step 3: Schedule the task by passing the chunks to the TaskScheduler
        self.scheduler.schedule_task(text_chunks)

        # Step 4: After processing, get the final synthesized output from the manager agent
        final_output = self.scheduler.manager.synthesize_output(text_chunks)

        # Step 5: Display the final output to the user via the CLI
        self.cli.display_output(final_output)


def main():
    # Argument parsing
    parser = argparse.ArgumentParser(
        description="Summarize a text file using Chain-of-Agents system."
    )

    # Required argument: Path to the input file
    parser.add_argument(
        "input_file", type=str, help="Path to the input text file to be summarized."
    )

    # Optional argument: Path to the output file
    parser.add_argument(
        "-o",
        "--output_file",
        type=str,
        help="Optional path to save the summarized output.",
    )

    args = parser.parse_args()

    # Read input file
    with open(args.input_file) as file:
        input_text = file.read()

    # Initialize InputChunker
    input_chunker = InputChunker(context_window_size=8000)

    # Initialize Worker Agents (example models, these can be real LLM instances)
    workers = [WorkerAgent(model="LLM_1"), WorkerAgent(model="LLM_2")]

    # Initialize Manager Agent
    manager = ManagerAgent(model="LLM_Manager")

    # Initialize Task Scheduler with workers and manager
    scheduler = TaskScheduler(workers=workers, manager=manager)

    # Initialize CLI (Pass input text directly to CLI for simplicity in this example)
    cli = CLI(input_text=input_text, output_file=args.output_file)

    # Initialize the Chain-of-Agents system
    system = ChainOfAgentsSummarizationSystem(
        input_chunker=input_chunker, scheduler=scheduler, cli=cli
    )

    # Start the summarization process
    system.start()


if __name__ == "__main__":
    main()
