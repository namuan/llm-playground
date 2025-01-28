#!/usr/bin/env -S uv run --quiet --script
# /// script
# dependencies = [
#   "litellm",
# ]
# ///
import argparse
import json
import logging
from pathlib import Path

from litellm import completion

LITELLM_MODEL = "ollama/llama3.1:latest"


def load_text(file_path):
    with file_path.open("r", encoding="utf-8") as file:
        return file.read()


def split_into_chapters(text, separator):
    return text.split(separator)


def openai_summarize_chapter(context, chapter_text):
    prompt = f"""
    Summarize the text below, synthesizing it with the provided context.

    Context:
    {context}

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
        model=LITELLM_MODEL,
        messages=[{"content": prompt, "role": "user"}],
        temperature=0.6,
        stream=False,
    )

    return response["choices"][0]["message"]["content"]


class ProgressTracker:
    def __init__(self, checkpoint_file):
        self.checkpoint_file = checkpoint_file
        self.progress = self.load_progress()

    def load_progress(self):
        if self.checkpoint_file.exists():
            with self.checkpoint_file.open("r") as f:
                return json.load(f)
        return {
            "current_chapter": 0,
            "current_line": 0,
            "context_summary": "",
            "chapter_summaries": [],
        }

    def save_progress(self):
        with self.checkpoint_file.open("w") as f:
            json.dump(self.progress, f, indent=2)

    def update_progress(
        self, chapter_idx, line_idx, context_summary, chapter_summaries
    ):
        self.progress["current_chapter"] = chapter_idx
        self.progress["current_line"] = line_idx
        self.progress["context_summary"] = context_summary
        self.progress["chapter_summaries"] = chapter_summaries
        self.save_progress()


def iterative_refinement(chapters, book_dir, tracker):
    context_summary = tracker.progress["context_summary"]
    chapter_summaries = tracker.progress["chapter_summaries"]
    start_chapter = tracker.progress["current_chapter"]

    # Create a file for incremental chapter summaries
    incremental_file = book_dir / "incremental_summaries.txt"

    def append_to_summary(text):
        with incremental_file.open("a", encoding="utf-8") as f:
            f.write(text)

    for i in range(start_chapter, len(chapters)):
        print(f"Summarizing chapter {i+1}...")
        append_to_summary(f"\nChapter {i+1} Summary:\n")
        lines = [line.strip() for line in chapters[i].split("\n") if line.strip()]
        line_summaries = []
        context_so_far = ""

        start_line = tracker.progress["current_line"] if i == start_chapter else 0

        number_of_lines_to_group = 3
        for j in range(start_line, len(lines), number_of_lines_to_group):
            line_group = " ".join(lines[j : j + number_of_lines_to_group])
            if not line_group:
                continue

            line_summary = openai_summarize_chapter(context_so_far, line_group)
            line_summaries.append(line_summary)
            context_so_far = refine_summary(context_so_far, line_summary)

            print(
                f"Processed lines {j+1}-{min(j+number_of_lines_to_group, len(lines))} of chapter {i+1}"
            )
            logging.debug(f" > {line_summary}")

            # Update progress after each line group
            append_to_summary(line_summary + "\n")
            tracker.update_progress(
                i, j + number_of_lines_to_group, context_summary, chapter_summaries
            )

        chapter_summary = "\n".join(line_summaries)
        chapter_summaries.append(chapter_summary)
        context_summary = refine_summary(context_summary, chapter_summary)

        # Update progress after each chapter
        tracker.update_progress(i + 1, 0, context_summary, chapter_summaries)

    return chapter_summaries


def refine_summary(current_summary, new_summary):
    return current_summary + "\n" + new_summary


def summarize_book(file_path, chapter_separator, output_dir):
    text = load_text(file_path)
    chapters = split_into_chapters(text, chapter_separator)

    # Set up the book directory and checkpoint file
    base_name = file_path.stem
    book_dir = output_dir / base_name
    book_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_file = book_dir / f"{base_name}_checkpoint.json"
    tracker = ProgressTracker(checkpoint_file)

    # Generate summaries with progress tracking
    chapter_summaries = iterative_refinement(chapters, book_dir, tracker)

    # Save final outputs
    summary_file = book_dir / f"{base_name}_summary.txt"
    chapter_summaries_file = book_dir / f"{base_name}_chapter_summaries.txt"

    save_summaries(chapter_summaries, chapter_summaries_file)
    final_summary = "\n".join(chapter_summaries)
    save_summary(final_summary, summary_file)

    # Clean up checkpoint file after successful completion
    checkpoint_file.unlink(missing_ok=True)


def save_summary(summary, file_path):
    with file_path.open("w", encoding="utf-8") as file:
        file.write(summary)


def save_summaries(summaries, file_path):
    with file_path.open("w", encoding="utf-8") as file:
        for i, chapter_summary in enumerate(summaries, start=1):
            file.write(f"Chapter {i} Summary:\n")
            file.write(chapter_summary + "\n\n")


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summarize_book(args.file_path, args.separator, args.output_dir)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Summarize a book using iterative refinement."
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=0,
        dest="verbose",
        help="Increase verbosity of logging output",
    )
    parser.add_argument("file_path", type=Path, help="The path to the book text file.")
    parser.add_argument(
        "--separator",
        type=str,
        default="\n\n--CHAPTER-BREAK--\n\n",
        help="The string used to separate chapters in the book (default: '\n\n--CHAPTER-BREAK--\n\n').",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path.cwd(),
        help="The output directory to save the summary files (default: current directory).",
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    main()
