#!/usr/bin/env -S uv run --quiet --script
# /// script
# dependencies = [
#     "beautifulsoup4==4.12.3",
#     "ebooklib==0.18",
#     "marimo==0.10.13",
#     "lxml==5.1.0"
# ]
# ///
"""
EPUB text extraction script

Usage:
./epub-to-audio.py -h

./epub-to-audio.py -v path/to/book.epub # To log INFO messages
./epub-to-audio.py -vv path/to/book.epub # To log DEBUG messages
"""
import logging
from argparse import ArgumentParser, RawDescriptionHelpFormatter
from collections import OrderedDict
from pathlib import Path

import ebooklib
from bs4 import BeautifulSoup
from ebooklib import epub

from logger import setup_logging


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
    return parser.parse_args()


def format_chapter_content(chapter):
    """Format a chapter's content into a string."""
    lines = [f"\nChapter: {chapter['chapter_id']} ({chapter['file_name']})"]
    for item in chapter["content"]:
        lines.append(f"{item['type']}: {item['text']}")
    return "\n".join(lines)


def main(args):
    logging.debug(f"Processing book: {args.book_path}")

    try:
        parser = EpubParser(args.book_path)
        book_content = parser.process_book()
        formatted_content = "\n".join(
            format_chapter_content(chapter) for chapter in book_content
        )
        print(formatted_content)
    except FileNotFoundError:
        logging.error(f"Could not find the book at {args.book_path}")
    except Exception as e:
        logging.error(f"Error processing the book: {str(e)}")


if __name__ == "__main__":
    args = parse_args()
    setup_logging(args.verbose)
    main(args)
