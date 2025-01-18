#!/usr/bin/env -S uv run --quiet --script
# /// script
# dependencies = [
#   "beautifulsoup4",
#   "ebooklib",
#   "lxml",
#   "more-itertools",
#   "mlx_lm",
#   "tqdm",
#   "kokoro-onnx",
#   "soundfile",
# ]
# ///
"""
EPUB text extraction script

Usage:
./epub-to-audio.py -h

./epub-to-audio.py -v path/to/book.epub # To log INFO messages
./epub-to-audio.py -vv path/to/book.epub # To log DEBUG messages
"""
import ast
import logging
import os
import pickle
import warnings
from argparse import ArgumentParser, RawDescriptionHelpFormatter
from collections import OrderedDict
from pathlib import Path

import ebooklib
import numpy as np
import soundfile as sf
from bs4 import BeautifulSoup
from ebooklib import epub
from kokoro_onnx import Kokoro
from mlx_lm import generate, load
from more_itertools import partition

from logger import setup_logging

SECOND_PASS_FILE_NAME = "second_pass.pkl"
INITIAL_PASS_FILE_NAME = "initial_pass.pkl"
FORMATTED_CHAPTER_FILE_NAME = "formatted_chapter.txt"

warnings.filterwarnings("ignore")

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

OUTPUT_DIR = Path.cwd() / "target" / "epub_pods"
OUTPUT_DIR.mkdir(exist_ok=True)

MODEL_VOICES_PATH = Path.home() / "models/onnx/kokoro/voices.json"
MODEL_PATH = Path.home() / "models/onnx/kokoro/kokoro-v0_19.onnx"
kokoro = None  # Lazy init

KOKORO_SPEAKER_1 = "bm_george"
KOKORO_SPEAKER_2 = "af_sarah"


class EpubParser:
    def __init__(self, book_path):
        self.book_path = book_path
        self.book = None
        self.chapters = OrderedDict()
        self.target_elements = {
            "ChapTitle": {"name": "span", "class_": "ChapTitle"},
            "Body": {"name": "p", "class_": "Body"},
            "LevelA": {"name": "h3", "class_": "LevelA"},
        }
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

    def parse_chapter(self, chapter_content):
        """Parse a single chapter's content."""
        texts = []
        soup = BeautifulSoup(chapter_content, features="lxml")

        for element in soup.body.descendants:
            if element.name is None:
                continue

            for key, target in self.target_elements.items():
                if element.name == target["name"] and target["class_"] in element.get(
                    "class", []
                ):
                    text = element.get_text(strip=True)
                    if text:  # Only append non-empty text
                        texts.append({"type": key, "text": text})
                    break

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


def format_chapter_content(chapter):
    lines = [f"\nChapter Metadata: {chapter['chapter_id']} ({chapter['file_name']})"]

    body_items, title_items = partition(
        lambda x: x["type"] == "ChapTitle", chapter["content"]
    )

    title_list = list(title_items)
    if title_list:
        lines.append(f"Chapter Title: {title_list[0]['text']}")

    chapter_contents = " ".join(item["text"] for item in body_items)
    chunks = create_text_chunks(chapter_contents, chunk_size=5000)
    lines.extend(chunks)
    return len(lines), "\n".join(lines)


def transcript_writer(formatted_content):
    model, tokenizer = load("mlx-community/Qwen2.5-14B-Instruct-4bit")
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
    model, tokenizer = load("mlx-community/Qwen2.5-7B-Instruct-4bit")
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
            model_path=MODEL_PATH.as_posix(), voices_path=MODEL_VOICES_PATH.as_posix()
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
    audio_files: list[Path], output_directory: Path, overwrite=False
) -> Path:
    final_audio_path = output_directory.joinpath("_podcast.wav")
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


def process_chapters(book_content):
    return [
        chapter_directory
        for idx, chapter, chapter_directory in get_chapters(book_content)
        if process_chapter(idx, chapter, chapter_directory)
    ]


def process_chapter(idx, chapter, chapter_directory):
    number_of_lines, formatted_chapter = format_chapter_content(chapter)
    logging.info(f"🏁 Processing Chapter {idx}. Number of lines: {number_of_lines}")

    if number_of_lines < 3:
        logging.info(
            f"Skipping {chapter}. Not enough lines in the chapter. Lines found: {number_of_lines}"
        )
        return False

    # Write formatted chapter to file
    output_file = chapter_directory / FORMATTED_CHAPTER_FILE_NAME
    output_file.write_text(formatted_chapter)
    return True


def get_chapters(book_content):
    for idx, chapter in enumerate(book_content):
        chapter_directory = OUTPUT_DIR.joinpath(f"chapter-{idx}")
        chapter_directory.mkdir(exist_ok=True)
        yield idx, chapter, chapter_directory


def main(args):
    logging.debug(f"Processing book: {args.book_path}")

    try:
        parser = EpubParser(args.book_path)
        book_content = parser.process_book()

        # Generate formatted text for all the chapters
        chapter_directories = process_chapters(book_content)

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
            combine_audio_files(chapter_audio_files, OUTPUT_DIR, overwrite=True)

    except FileNotFoundError:
        logging.error(f"Could not find the book at {args.book_path}", exc_info=True)
    except Exception as e:
        logging.error(f"Error processing the book: {str(e)}", exc_info=True)


if __name__ == "__main__":
    args = parse_args()
    setup_logging(args.verbose)
    main(args)
