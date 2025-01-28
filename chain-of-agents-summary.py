#!/usr/bin/env -S uv run --quiet --script
# /// script
# dependencies = [
#   "pandas",
#   "litellm",
# ]
# ///
"""
A Chain of Agents Summary Generator
"""

import argparse
from pathlib import Path

from litellm import completion

LITELLM_MODEL = "ollama/llama3.1:latest"
LITELLM_BASE_URL = "http://localhost:11434"


def generate_text_summary(model, chapter_text):
    prompt = f"""
    Summarize the text below, synthesizing it with the provided context.

    Text:
    {chapter_text}

    Summarize the text with clarity and critical analysis, keeping it relevant to the overall text.

    Do not add any headings in the response.

    Do not use any markdown formatting in your response.

    Write the summary from the author as the first person perspective.

    Consider linking any new evidence or data to earlier arguments or unresolved questions.

    Identify any key contributions, such as new theories, frameworks, or shifts in perspective, and mention practical applications or real-world implications, if relevant.

    Finally, address any unanswered questions or gaps this section might highlight.

    Point out any weaknesses or areas where further analysis could be valuable.

    Capture the gist of any stories told by the author along with any minor by important details.
    """

    response = completion(
        model=model,
        messages=[{"content": prompt, "role": "user"}],
        api_base=LITELLM_BASE_URL,
        temperature=0.6,
        stream=False,
    )

    return response["choices"][0]["message"]["content"]


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
    def __init__(self, model, worker_id=None):
        self.model = model
        self.worker_id = worker_id

    def process_chunk(self, chunk: str, previous_cu: dict = None) -> dict:
        """Process chunk and return a Communication Unit."""
        try:
            print(
                f"Worker {self.worker_id=} processing {chunk[:100]} using {self.model=}"
            )
            summary = generate_text_summary(self.model, chunk)

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


# Main System Class to orchestrate the entire process
class ChainOfAgentsSummarizationSystem:
    def __init__(
        self,
        input_chunker: InputChunker,
        scheduler: TaskScheduler,
        cli: CLI,
        num_passes: int,
    ):
        self.input_chunker = input_chunker
        self.scheduler = scheduler
        self.cli = cli
        self.num_passes = num_passes

    def start(self):
        """Starts the chain-of-agents system for summarization."""
        input_text = self.cli.get_input()
        text_chunks = self.input_chunker.chunk_input(input_text)
        selected_chunks = text_chunks[:10]

        final_chunks = selected_chunks
        final_output = None
        for pass_num in range(self.num_passes):
            print(
                f"🏁 Going through pass {pass_num}. Total lines to process {len(final_chunks)}"
            )
            communication_units = self.scheduler.schedule_task(final_chunks)
            final_output = self.scheduler.manager.synthesize_output(communication_units)
            if pass_num < self.num_passes - 1:
                final_chunks = self.input_chunker.chunk_input(final_output)

        self.cli.display_output(final_output)


def main(args):
    # Initialize InputChunker
    input_chunker = InputChunker(context_window_size=8000)

    # Initialize workers with unique IDs
    workers = [
        WorkerAgent(model=LITELLM_MODEL, worker_id=f"worker_{i}") for i in range(2)
    ]

    manager = ManagerAgent(model="LLM_Manager")
    scheduler = TaskScheduler(workers=workers, manager=manager)
    cli = CLI(input_text=args.input_file.read_text(), output_file=args.output_file)
    system = ChainOfAgentsSummarizationSystem(
        input_chunker=input_chunker,
        scheduler=scheduler,
        cli=cli,
        num_passes=args.passes,
    )

    # Start the summarization process
    system.start()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Summarize a text file using Chain-of-Agents system."
    )
    parser.add_argument(
        "input_file", type=Path, help="Path to the input text file to be summarized."
    )
    parser.add_argument(
        "-o",
        "--output_file",
        type=str,
        help="Optional path to save the summarized output.",
    )
    parser.add_argument(
        "-p",
        "--passes",
        type=int,
        default=3,
        help="Number of passes to run the summarization. Default is 3.",
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)
