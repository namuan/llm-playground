#!/usr/bin/env -S uv run --quiet --script
# /// script
# dependencies = [
#   "litellm",
# ]
# ///
import argparse
from pathlib import Path

from litellm import completion

LITELLM_MODEL = "ollama/deepseek-r1:1.5b"
LITELLM_BASE_URL = "http://localhost:11434"


# Function to load the entire text file
def load_text(file_path):
    with file_path.open("r", encoding="utf-8") as file:
        return file.read()


# Function to split the text into chapters based on a separator
def split_into_chapters(text, separator):
    return text.split(separator)


# Function to call OpenAI and summarize a chapter with context using the new prompt
def openai_summarize_chapter(context, chapter_text):
    prompt = f"""
    Summarize the text below, synthesizing it with the provided context. Prioritize clarity, critical analysis, and relevance to the book's purpose.

    Context:
    {context}

    Text:
    {chapter_text}

    Summarize the text with clarity and critical analysis, keeping it relevant to the book's purpose. Do not use any markdown formatting in your response.

    Start by providing a concise and neutral overview of the section's main ideas. Next, explain how this section contributes to or challenges the book's thesis. Consider linking any new evidence or data to earlier arguments or unresolved questions.

    Identify any key contributions, such as new theories, frameworks, or shifts in perspective, and mention practical applications or real-world implications, if relevant.

    Finally, address any unanswered questions or gaps this section might highlight. Point out any weaknesses or areas where further analysis could be valuable.
    """

    # Use litellm to call the OpenAI API
    response = completion(
        model=LITELLM_MODEL,
        messages=[{"content": prompt, "role": "user"}],
        api_base=LITELLM_BASE_URL,
        temperature=0.6,
        stream=False,
    )

    return response["choices"][0]["message"]["content"]


def iterative_refinement(chapters):
    context_summary = ""
    chapter_summaries = []

    for i, chapter in enumerate(chapters):
        print(f"Summarizing chapter {i+1}...")

        lines = [line.strip() for line in chapter.split("\n") if line.strip()]
        line_summaries = []
        line_context = ""

        for j in range(0, len(lines), 3):
            line_group = " ".join(lines[j : j + 3])
            if not line_group:
                continue

            line_summary = openai_summarize_chapter(line_context, line_group)
            line_summaries.append(line_summary)
            line_context = refine_summary(line_context, line_summary)

            print(f"Processed lines {j+1}-{min(j+3, len(lines))} of chapter {i+1}")
            print(f" > {line_summary}")

        chapter_summary = "\n".join(line_summaries)
        chapter_summaries.append(chapter_summary)
        context_summary = refine_summary(context_summary, chapter_summary)

    return chapter_summaries


# Function to refine the overall summary with the current chapter's summary (used for building context)
def refine_summary(current_summary, new_summary):
    # Concatenate the current summary with the new chapter's summary for future context
    refined_summary = current_summary + "\n" + new_summary
    return refined_summary


def summarize_book(file_path, chapter_separator, output_dir):
    # Load the book content
    text = load_text(file_path)

    # Split the text into chapters
    chapters = split_into_chapters(text, chapter_separator)

    # Perform iterative refinement to generate a summary for each chapter
    chapter_summaries = iterative_refinement(chapters)

    # Generate output file paths using the input file name and output directory
    base_name = file_path.stem  # Get the base name without extension
    book_dir = output_dir / base_name
    book_dir.mkdir(parents=True, exist_ok=True)

    summary_file = book_dir / f"{base_name}_summary.txt"
    chapter_summaries_file = book_dir / f"{base_name}_chapter_summaries.txt"

    # Save the chapter summaries to a file
    save_summaries(chapter_summaries, chapter_summaries_file)

    # Save the final summary to the specified output file
    final_summary = "\n".join(chapter_summaries)
    save_summary(final_summary, summary_file)


# Function to save the final summary to a file
def save_summary(summary, file_path):
    with file_path.open("w", encoding="utf-8") as file:
        file.write(summary)


# Function to save individual chapter summaries to a file
def save_summaries(summaries, file_path):
    with file_path.open("w", encoding="utf-8") as file:
        for i, chapter_summary in enumerate(summaries, start=1):
            file.write(f"Chapter {i} Summary:\n")
            file.write(chapter_summary + "\n\n")


def main():
    args = parse_args()

    # Ensure the output directory exists
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Call the main summarization function
    summarize_book(args.file_path, args.separator, args.output_dir)


def parse_args():
    # Set up command-line argument parsing
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
        default=Path.cwd(),  # Default to the current working directory
        help="The output directory to save the summary files (default: current directory).",
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    main()
