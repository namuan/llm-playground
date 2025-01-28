#!/usr/bin/env -S uv run --quiet --script
# /// script
# dependencies = [
#   "litellm",
# ]
# ///
"""
Recursive text summarizer that processes text in multiple passes to generate a concise summary.

Required packages:
pip install openai

Usage:
    ./recursive_summarizer.py --input-file input.txt [--output-file output.txt] [--max-chunk-size 8000]
"""

import argparse
import os
import sys
from pathlib import Path
from typing import List
from typing import Optional

from litellm import completion

LITELLM_MODEL = "ollama/llama3.1:latest"
# LITELLM_MODEL = "groq/llama-3.1-8b-instant"


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Recursive text summarizer")
    parser.add_argument(
        "--input-file", type=Path, required=True, help="Input file path"
    )
    parser.add_argument("--output-file", type=Path, help="Output file path (optional)")
    parser.add_argument(
        "--max-chunk-size", type=int, default=32000, help="Maximum chunk size"
    )
    return parser.parse_args()


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
        # api_base=LITELLM_BASE_URL,
        temperature=0.6,
        stream=False,
        max_tokens=32000,
    )

    return response["choices"][0]["message"]["content"]


def ensure_directory_exists(directory: str) -> None:
    if not os.path.exists(directory):
        os.makedirs(directory)


def save_progress(content: str, filename: str, directory: str) -> None:
    ensure_directory_exists(directory)
    filepath = os.path.join(directory, filename)
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(content)


def save_chunks(chunks: List[str], pass_num: int, directory: str) -> None:
    for i, chunk in enumerate(chunks, 1):
        filename = f"pass_{pass_num}_chunk_{i}.txt"
        save_progress(chunk, filename, directory)


def validate_paths(input_path: str, output_path: Optional[str] = None) -> bool:
    if not os.path.isfile(input_path):
        print(f"Error: Input file {input_path} does not exist")
        return False

    if output_path:
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            try:
                os.makedirs(output_dir)
            except OSError as e:
                print(f"Error: Cannot create output directory: {e}")
                return False
    return True


def read_file(file_path: str) -> str:
    try:
        with open(file_path, encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        print(f"Error reading file: {e}")
        sys.exit(1)


def chunk_text(text: str, max_size: int) -> List[str]:
    chunks = []
    current_chunk = []
    current_size = 0

    for paragraph in text.split("\n\n"):
        paragraph = paragraph.strip()
        if not paragraph:
            continue

        paragraph_size = len(paragraph)

        if paragraph_size > max_size:
            if current_chunk:
                chunks.append("\n\n".join(current_chunk))
                current_chunk = []
                current_size = 0
            chunks.append(paragraph[:max_size])
            continue

        if current_size + paragraph_size > max_size:
            chunks.append("\n\n".join(current_chunk))
            current_chunk = [paragraph]
            current_size = paragraph_size
        else:
            current_chunk.append(paragraph)
            current_size += paragraph_size

    if current_chunk:
        chunks.append("\n\n".join(current_chunk))

    return chunks


def summarize_chunk(text: str) -> str:
    return generate_text_summary(LITELLM_MODEL, text)


def process_chunks(chunks: List[str], pass_num: int, output_dir: str) -> str:
    summaries = []
    total_chunks = len(chunks)

    # Save the input chunks
    save_chunks(chunks, pass_num, output_dir)

    for i, chunk in enumerate(chunks, 1):
        print(f"Processing chunk {i}/{total_chunks}")
        summary = summarize_chunk(chunk)
        summaries.append(summary)

        # Save individual chunk summary
        filename = f"pass_{pass_num}_chunk_{i}_summary.txt"
        save_progress(summary, filename, output_dir)

    combined_summary = "\n\n".join(summaries)

    # Save the combined summary for this pass
    filename = f"pass_{pass_num}_combined_summary.txt"
    save_progress(combined_summary, filename, output_dir)

    return combined_summary


def write_output(text: str, output_path: Optional[Path]) -> None:
    if output_path:
        output_path.write_text(text)
    else:
        print("\nFinal Summary:")
        print("-" * 80)
        print(text)
        print("-" * 80)


def main() -> None:
    args = parse_arguments()

    if not validate_paths(args.input_file, args.output_file):
        sys.exit(1)

    # Create output directory
    base_name = args.input_file.stem
    output_dir = os.path.join("target", "recursive-summarizer", base_name)
    ensure_directory_exists(output_dir)

    # Save the original input
    input_text = args.input_file.read_text(encoding="utf-8")
    save_progress(input_text, "original_input.txt", output_dir)

    text = input_text
    for pass_num in range(3):
        print(f"\nPass {pass_num + 1}/3")
        chunks = chunk_text(text, args.max_chunk_size)
        text = process_chunks(chunks, pass_num + 1, output_dir)

    print("\nGenerating final summary...")
    final_summary = summarize_chunk(text)

    # Save the final summary
    save_progress(final_summary, "final_summary.txt", output_dir)

    # Write to output file if specified
    write_output(final_summary, args.output_file)

    print(f"\nAll progress has been saved in: {output_dir}")


if __name__ == "__main__":
    main()
