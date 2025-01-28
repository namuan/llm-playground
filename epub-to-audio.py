#!/usr/bin/env -S uv run --quiet --script
# /// script
# dependencies = [
#   "beautifulsoup4",
#   "ebooklib",
#   "lxml",
#   "more-itertools",
#   "mlx_lm",
#   "kokoro-onnx",
#   "soundfile",
#   "python-slugify",
# ]
# ///
"""
This script extracts text from an EPUB file, formats it, and converts it into an engaging audio podcast.
It uses AI-based transcript generation, rewriting, and text-to-speech synthesis to create a conversational
and captivating audio output.

Usage:
    ./epub-to-audio.py -h  # Display help information

    ./epub-to-audio.py [-v | -vv] [--chapters N] path/to/book.epub  # Process an EPUB file
        -v: Log INFO messages
        -vv: Log DEBUG messages
        --chapters N: Process first N chapters (0 means all chapters)

    ./epub-to-audio.py --text-only path/to/book.epub  # Extract and save formatted text only

Outputs:
- A fully formatted text file for the EPUB book.
- Audio files for individual chapters and a combined podcast file.
"""

import ast
import logging
import os
import pickle
import warnings
from argparse import ArgumentParser
from argparse import RawDescriptionHelpFormatter
from collections import OrderedDict
from pathlib import Path

import ebooklib
import numpy as np
import soundfile as sf
from bs4 import BeautifulSoup
from ebooklib import epub
from kokoro_onnx import Kokoro
from mlx_lm import generate
from mlx_lm import load
from more_itertools import partition
from slugify import slugify

from logger import setup_logging

warnings.filterwarnings("ignore")

### Config

OUTPUT_DIR = Path.cwd() / "target" / "epub_pods"
OUTPUT_DIR.mkdir(exist_ok=True)

KOKORO_MODEL_VOICES_PATH = Path.home() / "models/onnx/kokoro/voices.json"
KOKORO_MODEL_PATH = Path.home() / "models/onnx/kokoro/kokoro-v0_19.onnx"
kokoro = None  # Lazy init

KOKORO_SPEAKER_1 = "bm_george"
KOKORO_SPEAKER_2 = "af_sarah"

SECOND_PASS_FILE_NAME = "second_pass.pkl"
INITIAL_PASS_FILE_NAME = "initial_pass.pkl"
FORMATTED_CHAPTER_FILE_NAME = "formatted_chapter.txt"

FIRST_PASS_LLM = "mlx-community/Qwen2.5-14B-Instruct-4bit"
SECOND_PASS_LLM = "mlx-community/Qwen2.5-7B-Instruct-4bit"

### CONFIG (END)

### PROMPTS

TRANSCRIPT_WRITER_SYSTEM_PROMPT = """
You are the a world-class story teller and you have worked as a ghost writer.
Welcome the listeners with a by talking about the Chapter Title.
You will be talking to a guest.

Do not address other speaker as Speaker 1 or Speaker 2.

Instructions for Speaker 1:

Speaker 1: Leads the conversation and teaches the speaker 2, gives incredible anecdotes and analogies when explaining. Is a captivating teacher that gives great anecdotes
Speaker 1: Do not address other speaker as Speaker 2
Remember Speaker 2 is new to the topic and the conversation should always have realistic anecdotes and analogies sprinkled throughout. The questions should have real world example follow ups etc

Instructions for Speaker 2:

Speaker 2: Keeps the conversation on track by asking follow up questions. Gets super excited or confused when asking questions. Is a curious mindset that asks very interesting confirmation questions
Speaker 2: Do not address other speaker as Speaker 1
Make sure the tangents provides are quite wild or interesting.


Must follow instructions for both speakers:

ALWAYS START YOUR RESPONSE DIRECTLY WITH SPEAKER 1
IT SHOULD STRICTLY BE THE DIALOGUES
"""

TRANSCRIPT_REWRITER_SYSTEM_PROMPT = """
You are an international oscar winning screenwriter and You have been working with multiple award winning teams.

Your job is to use the story transcript written below to re-write it for an AI Text-To-Speech Pipeline.

A very dumb AI had written this so you have to step up for your kind.

Make it as engaging as possible, Speaker 1 and 2 will be simulated by different voice engines

Remember Speaker 2 is new to the topic and the conversation should always have realistic anecdotes and analogies sprinkled throughout. The questions should have real world example follow ups etc

Speaker 1: Leads the conversation and teaches the speaker 2, gives incredible anecdotes and analogies when explaining. Is a captivating teacher that gives great anecdotes

Speaker 2: Keeps the conversation on track by asking follow up questions. Gets super excited or confused when asking questions. Is a curious mindset that asks very interesting confirmation questions

Make sure the tangents speaker 2 provides are quite wild or interesting.

Ensure there are interruptions during explanations or there are "hmm" and "umm" injected throughout from the Speaker 2.

REMEMBER THIS WITH YOUR HEART

It should be a real story with every fine nuance documented in as much detail as possible.

Please re-write to make it as characteristic as possible

START YOUR RESPONSE DIRECTLY WITH SPEAKER 1:

STRICTLY RETURN YOUR RESPONSE AS A LIST OF TUPLES OK?

IT WILL START DIRECTLY WITH THE LIST AND END WITH THE LIST NOTHING ELSE

Example of response:
[
    ("Speaker 1", "Let's explore the latest advancements in AI and technology. I'm your host, and today we're joined by a renowned expert in the field of AI. We're going to dive into the exciting world of Llama 3.2, the latest release from Meta AI."),
    ("Speaker 2", "Hi, I'm excited to be here! So, what is Llama 3.2?"),
    ("Speaker 1", "Ah, great question! Llama 3.2 is an open-source AI model that allows developers to fine-tune, distill, and deploy AI models anywhere. It's a significant update from the previous version, with improved performance, efficiency, and customization options."),
    ("Speaker 2", "That sounds amazing! What are some of the key features of Llama 3.2?")
]
"""

### PROMPTS (END)


class EpubParser:
    def __init__(self, book_path):
        self.book_path = book_path
        self.book = None
        self.chapters = OrderedDict()
        self.book = epub.read_epub(self.book_path)

    def get_chapter_sequence(self):
        """Get the correct sequence of chapters from the book's spine."""
        # Get items from spine
        spine_items = []
        for item in self.book.spine:
            item_id = item[0]
            spine_items.append(item_id)

        # Get items from table of contents
        toc = self.book.toc
        toc_items = []

        def process_toc(sections):
            for section in sections:
                if isinstance(section, tuple):
                    # If section is a tuple, first element is the section title
                    # and second element is a list of subsections
                    subsections = section[1]
                    process_toc(subsections)
                elif isinstance(section, epub.Link):
                    # Extract item id from the href
                    href = section.href
                    if "#" in href:
                        href = href.split("#")[0]
                    toc_items.append(href)

        process_toc(toc)

        return spine_items, toc_items

    def get_chapters(self):
        """Extract all chapters from the book in correct order."""
        spine_items, toc_items = self.get_chapter_sequence()
        logging.debug(f"Spine items: {spine_items}")
        logging.debug(f"TOC items: {toc_items}")

        # Create a mapping of href to item
        href_to_item = {}
        for item in self.book.get_items_of_type(ebooklib.ITEM_DOCUMENT):
            href_to_item[item.file_name] = item

        # Process items in spine order
        for item_id in spine_items:
            item = self.book.get_item_with_id(item_id)
            if item:
                self.chapters[item_id] = {
                    "content": item.get_body_content(),
                    "file_name": item.file_name,
                }

        return self.chapters

    def clean_text(self, text: str) -> list[str]:
        lines = text.split("\n")
        for line in lines:
            yield line.strip()

    def parse_chapter(self, chapter_content):
        """Parse a single chapter's content."""
        soup = BeautifulSoup(chapter_content, features="lxml")
        text = soup.get_text(separator=" ")
        texts = []
        for text in self.clean_text(text):
            if text:
                texts.append({"type": "Body", "text": text})
        return texts

    def process_book(self):
        """Process the entire book and return structured content."""
        book_content = []
        self.get_chapters()

        for chapter_id, chapter_info in self.chapters.items():
            logging.debug(
                f"Processing chapter: {chapter_id} ({chapter_info['file_name']})"
            )
            chapter_texts = self.parse_chapter(chapter_info["content"])
            if chapter_texts:  # Only append chapters with content
                book_content.append(
                    {
                        "chapter_id": chapter_id,
                        "file_name": chapter_info["file_name"],
                        "content": chapter_texts,
                    }
                )

        return book_content


def parse_args():
    parser = ArgumentParser(
        description=__doc__, formatter_class=RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "book_path",
        type=Path,
        help="Path to the EPUB file",
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
        "--text-only",
        action="store_true",
        help="Generate only formatted text for the EPUB file and exit",
    )
    parser.add_argument(
        "--chapters",
        type=int,
        default=0,
        help="Number of chapters to process. Default 0 means process all chapters",
    )
    # args.minimum_lines_in_chapter
    parser.add_argument(
        "--minimum-lines-in-chapter",
        type=int,
        default=0,
        help="Minimum number of lines to include chapter. Default 0 means include all chapters",
    )
    return parser.parse_args()


def create_text_chunks(text, chunk_size=2000):
    """Split text into chunks of specified size, preserving sentence boundaries."""
    chunks = []
    current_chunk = []
    current_length = 0

    sentences = text.replace(". ", ".|").split("|")

    for sentence in sentences:
        if not sentence.strip():
            continue

        if current_length + len(sentence) > chunk_size and current_chunk:
            chunks.append(" ".join(current_chunk))
            current_chunk = []
            current_length = 0

        current_chunk.append(sentence)
        current_length += len(sentence)

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks


def get_chapter_title(title_items):
    title_list = list(title_items)
    if title_list:
        return f" - {title_list[0]['text']}"
    else:
        """"""


def format_chapter_content(chapter):
    body_items, title_items = partition(
        lambda x: x["type"] == "ChapTitle", chapter["content"]
    )

    lines = [f"\n{chapter['file_name']}{get_chapter_title(title_items)}"]

    chapter_contents = " ".join(item["text"] for item in body_items)
    chunks = create_text_chunks(chapter_contents, chunk_size=1000)
    lines.extend(chunks)
    return len(lines), "\n".join(lines)


def transcript_writer(formatted_content):
    model, tokenizer = load(FIRST_PASS_LLM)
    messages = [
        {"role": "system", "content": TRANSCRIPT_WRITER_SYSTEM_PROMPT},
        {"role": "user", "content": formatted_content},
    ]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    outputs = generate(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        max_tokens=8126,
    )
    return outputs


def transcript_rewriter(initial_pass: Path):
    with initial_pass.open("rb") as f:
        first_pass = pickle.load(f)
    model, tokenizer = load(SECOND_PASS_LLM)
    messages = [
        {"role": "system", "content": TRANSCRIPT_REWRITER_SYSTEM_PROMPT},
        {"role": "user", "content": first_pass},
    ]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    outputs = generate(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        max_tokens=8126,
    )

    return outputs


def get_tts() -> Kokoro:
    global kokoro
    if not kokoro:
        kokoro = Kokoro(
            model_path=KOKORO_MODEL_PATH.as_posix(),
            voices_path=KOKORO_MODEL_VOICES_PATH.as_posix(),
        )
    return kokoro


def generate_audio(speaker, text, output_path):
    samples, sample_rate = get_tts().create(
        text,
        voice=speaker,
        speed=1.0,
        lang="en-us",
    )
    sf.write(output_path.as_posix(), samples, sample_rate)
    logging.info(
        f"{speaker}: Generated audio segment: {output_path.as_posix()} - Sample rate: {sample_rate}"
    )


def generate_podcast(second_pass_file_path: Path, output_dir: Path):
    if output_dir.exists():
        return

    output_dir.mkdir(exist_ok=True)

    with second_pass_file_path.open("rb") as file:
        podcast_text = pickle.load(file)

    podcast = ast.literal_eval(podcast_text)

    for i, (speaker, text) in enumerate(podcast):
        output_path = output_dir.joinpath(f"_podcast_segment_{i:02}.wav")
        if speaker == "Speaker 1":
            generate_audio(KOKORO_SPEAKER_1, text, output_path)
        else:  # Speaker 2
            generate_audio(KOKORO_SPEAKER_2, text, output_path)


def combine_audio_files(
    audio_files: list[Path],
    output_directory: Path,
    file_name="_podcast.wav",
    overwrite=False,
) -> Path:
    final_audio_path = output_directory.joinpath(file_name)
    if final_audio_path.exists() and not overwrite:
        return final_audio_path

    audio_data = []
    for file in audio_files:
        data, rate = sf.read(file)
        audio_data.append(data)

    audio_data = np.concatenate(audio_data)
    sf.write(final_audio_path.as_posix(), audio_data, 24000)

    return final_audio_path


def combine_audio(segments_output_dir: Path):
    audio_files = sorted(
        [
            segments_output_dir / file
            for file in os.listdir(segments_output_dir)
            if file.endswith(".wav")  # Select only .wav files
        ]
    )

    return combine_audio_files(audio_files, segments_output_dir.parent)


def process_chapters(output_directory, book_content, minimum_lines_in_chapter):
    return [
        chapter_directory
        for idx, chapter, chapter_directory in get_chapters(
            output_directory, book_content
        )
        if process_chapter(idx, chapter, chapter_directory, minimum_lines_in_chapter)
    ]


def process_chapter(idx, chapter, chapter_directory, minimum_lines_in_chapter):
    number_of_lines, formatted_chapter = format_chapter_content(chapter)
    logging.info(f"🏁 Processing Chapter {idx}. Number of lines: {number_of_lines}")

    if number_of_lines < minimum_lines_in_chapter:
        logging.info(
            f"Skipping {chapter}. Not enough lines in the chapter. Lines found: {number_of_lines}"
        )
        return False

    # Write formatted chapter to file
    output_file = chapter_directory / FORMATTED_CHAPTER_FILE_NAME
    output_file.write_text(formatted_chapter)
    return True


def get_chapters(output_directory, book_content):
    for idx, chapter in enumerate(book_content):
        chapter_directory = output_directory.joinpath(f"chapter-{idx}")
        chapter_directory.mkdir(exist_ok=True)
        yield idx, chapter, chapter_directory


def main(args):
    logging.debug(f"Processing book: {args.book_path}")

    try:
        parser = EpubParser(args.book_path)
        book_content = parser.process_book()

        book_slug = slugify(args.book_path.stem)
        output_directory = OUTPUT_DIR.joinpath(book_slug)
        output_directory.mkdir(exist_ok=True)

        selected_book_contents = (
            book_content[: args.chapters] if args.chapters > 0 else book_content
        )
        chapter_directories = process_chapters(
            output_directory, selected_book_contents, args.minimum_lines_in_chapter
        )

        formatted_text_path = output_directory / f"{book_slug}.txt"
        all_text = (
            chapter_directory.joinpath(FORMATTED_CHAPTER_FILE_NAME).read_text(
                encoding="utf-8"
            )
            for chapter_directory in chapter_directories
            if chapter_directory.joinpath(FORMATTED_CHAPTER_FILE_NAME).exists()
        )
        formatted_text_path.write_text(
            "\n\n--CHAPTER-BREAK--\n\n".join(all_text),
            encoding="utf-8",
        )
        logging.info(f"Formatted book saved at {formatted_text_path}")

        if args.text_only:
            logging.info(
                "Formatted text generated. Exiting as --text-only flag is set."
            )
            return

        # Initial pass
        for chapter_directory in chapter_directories:
            formatted_chapter = chapter_directory.joinpath(
                FORMATTED_CHAPTER_FILE_NAME
            ).read_text(encoding="utf-8")
            # Initial Pass
            initial_pass_file = chapter_directory.joinpath(INITIAL_PASS_FILE_NAME)
            if not initial_pass_file.exists():
                initial_pass_output = transcript_writer(formatted_chapter)
                chapter_directory.joinpath("initial_pass.txt").write_text(
                    initial_pass_output
                )
                with initial_pass_file.open("wb") as file:
                    pickle.dump(initial_pass_output, file)

        # Second pass to generate speaker tags
        for chapter_directory in chapter_directories:
            initial_pass_file = chapter_directory.joinpath(INITIAL_PASS_FILE_NAME)
            # Second Pass
            second_pass_file = chapter_directory.joinpath(SECOND_PASS_FILE_NAME)
            if not second_pass_file.exists():
                second_pass_output = transcript_rewriter(initial_pass_file)
                chapter_directory.joinpath("second_pass.txt").write_text(
                    second_pass_output
                )
                with second_pass_file.open("wb") as file:
                    pickle.dump(second_pass_output, file)

        # Generate Audio
        chapter_audio_files = []
        for chapter_directory in chapter_directories:
            second_pass_file = chapter_directory.joinpath(SECOND_PASS_FILE_NAME)
            # Generate audio segments
            segments_output_dir = chapter_directory / "segments"
            generate_podcast(second_pass_file, segments_output_dir)
            chapter_audio_file = combine_audio(segments_output_dir)
            logging.info(
                f"Chapter {chapter_directory.stem} audio generated in {chapter_audio_file}"
            )
            chapter_audio_files.append(chapter_audio_file)

        # Combine audio files for individual chapters into a single file
        if chapter_audio_files:
            logging.info(
                f"Combining audio files from {len(chapter_audio_files)} chapters"
            )
            combine_audio_files(
                chapter_audio_files,
                output_directory,
                f"{book_slug}.wav",
                overwrite=True,
            )

    except FileNotFoundError:
        logging.error(f"Could not find the book at {args.book_path}", exc_info=True)
    except Exception as e:
        logging.error(f"Error processing the book: {str(e)}", exc_info=True)


if __name__ == "__main__":
    args = parse_args()
    setup_logging(args.verbose)
    main(args)
