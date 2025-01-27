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
import logging
from pathlib import Path


# Input Layer: Handles chunking and task identification
class InputChunker:
    def __init__(self, context_window_size: int):
        self.context_window_size = context_window_size

    def chunk_input(self, text: str) -> list:
        """Splits input text into chunks based on context window size and paragraph boundaries."""
        chunks = []
        paragraphs = text.split("\n\n")
        current_chunk = []
        current_size = 0

        for paragraph in paragraphs:
            # Rough estimation of tokens (words + punctuation)
            paragraph_size = len(paragraph.split())

            # If adding this paragraph exceeds the window size
            if current_size + paragraph_size > self.context_window_size:
                # Save current chunk if not empty
                if current_chunk:
                    chunks.append("\n\n".join(current_chunk))
                    current_chunk = []
                    current_size = 0

            current_chunk.append(paragraph)
            current_size += paragraph_size

        # Add the last chunk if there's anything left
        if current_chunk:
            chunks.append("\n\n".join(current_chunk))

        return chunks


# Control Layer: Manages the task and flow between agents
class TaskScheduler:
    def __init__(self, workers: list, manager):
        self.workers = workers
        self.manager = manager

    def schedule_task(self, text_chunks: list):
        """Schedules and manages the processing of chunks by worker agents."""
        communication_units = []
        previous_cu = None

        for i, chunk in enumerate(text_chunks):
            # Round-robin worker selection
            worker = self.workers[i % len(self.workers)]

            # Process chunk with retry logic
            for retry in range(3):  # Maximum 3 retries
                try:
                    cu = worker.process_chunk(chunk, previous_cu)
                    if cu["status"] == "success":
                        communication_units.append(cu)
                        previous_cu = cu
                        break
                except Exception as e:
                    if retry == 2:  # Last retry
                        # Log error and continue with next chunk
                        print(f"Failed to process chunk after 3 retries: {str(e)}")

        return communication_units


# Worker Agent: Processes individual chunks and generates Communication Units (CU)
class WorkerAgent:
    def __init__(self, model):
        self.model = model

    def process_chunk(self, chunk: str, previous_cu: dict = None) -> dict:
        """Process chunk and return a Communication Unit."""
        try:
            # Here you would typically call your LLM with the chunk
            # For now, we'll create a mock summary
            summary = f"Summary of chunk: {chunk[:100]}..."

            communication_unit = {
                "summary": summary,
                "chunk_length": len(chunk),
                "previous_context": previous_cu["summary"] if previous_cu else None,
                "model_used": self.model,
                "status": "success",
            }

            return communication_unit

        except Exception as e:
            return {
                "status": "error",
                "error_message": str(e),
                "chunk_length": len(chunk),
            }


# Manager Agent: Collects CUs and synthesizes the final output
class ManagerAgent:
    def __init__(self, model):
        self.model = model

    def synthesize_output(self, communication_units: list) -> str:
        """Synthesize final output from communication units."""
        if not communication_units:
            return "No content to summarize."

        # Combine all summaries
        combined_summary = []
        for cu in communication_units:
            if cu.get("status") == "success":
                combined_summary.append(cu["summary"])

        # Here you would typically use the manager's LLM to create a cohesive summary
        # For now, we'll just join them
        final_summary = "\n\n".join(combined_summary)

        return final_summary


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
        """Log messages with specified level."""
        if not hasattr(self, "_logger"):
            # Configure logger
            logging.basicConfig(
                format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                level=getattr(logging, self.log_level.upper()),
            )
            self._logger = logging.getLogger("ChainOfAgents")

        log_method = getattr(self._logger, level.lower())
        log_method(message)


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
        communication_units = self.scheduler.schedule_task(
            text_chunks
        )  # Store the return value

        # Step 4: After processing, get the final synthesized output from the manager agent
        final_output = self.scheduler.manager.synthesize_output(
            communication_units
        )  # Pass communication_units instead of text_chunks

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
