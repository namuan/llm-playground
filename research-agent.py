#!/usr/bin/env -S uv run --quiet --script
# /// script
# dependencies = [
#   "litellm",
#   "beautifulsoup4",
#   "duckduckgo_search",
#   "python-slugify",
# ]
# ///
"""
A personal research assistant

Usage:
./research-agent.py --help

./research-agent.py -q QUESTION -f MARKDOWN_FILE
./research-agent.py -q "What is the best way to learn programming?" -f output.md
"""

import argparse
from argparse import ArgumentParser
from argparse import RawDescriptionHelpFormatter
from pathlib import Path

import litellm
import requests
from bs4 import BeautifulSoup
from duckduckgo_search import DDGS
from slugify import slugify

from logger import setup_logging

LITELLM_MODEL = "ollama/llama3.1:latest"
LITELLM_BASE_URL = "http://localhost:11434"


def parse_args() -> argparse.Namespace:
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
        "-q",
        "--question",
        type=str,
        required=True,
        help="Question to be asked",
    )
    parser.add_argument(
        "-f",
        "--target-folder",
        type=str,
        required=True,
        help="Target folder for output files",
    )
    return parser.parse_args()


def generate_slug(text: str) -> str:
    """Generate a clean slug from input text."""
    return slugify(text, max_length=100, lowercase=True)


class Question:
    def __init__(self, question_text: str):
        """Initializes the question object."""
        self.question_text = question_text

    def receive_question(self) -> str:
        """Receives a question from the user and returns it as a string."""
        return self.question_text


class SearchEngine:
    def __init__(self):
        """Initializes the search engine object."""

    def search_for_question(self, question_text: str) -> list:
        results = DDGS().text(
            question_text,
            region="wt-wt",
            safesearch="Off",
            timelimit="y",
            max_results=10,
        )
        return [
            Website(result["href"], result["title"], result["body"])
            for result in results
        ]


class Website:
    """A class to represent a website."""

    def __init__(self, url: str, text: str, description: str):
        """Initializes the website object."""
        self.url = url
        self.text = text
        self.description = description

    def scrape_website(self):
        """Scrapes the website and returns the article."""
        print(f"⛏️ Scraping website...{self.url}")
        response = requests.get(self.url)
        soup = BeautifulSoup(response.text, "html.parser")
        article_text = " ".join([p.text for p in soup.find_all("p")])
        return article_text

    def get_summary(self) -> str:
        """Returns the summary of the website."""
        return f"Brief: {self.description}\nText: {self.scrape_website()}"


class Summary:
    """A class to represent a summary."""

    def __init__(self, summary_text: str, website_title: str, link: str):
        """Initializes the summary object."""
        self.summary_text = summary_text
        self.website_title = website_title
        self.link = link

    def __str__(self) -> str:
        """Returns the summary as a string."""
        return f"### {self.website_title}:\n{self.summary_text}"


class SummaryWriter:
    summary_system_prompt: str = """
    Your goal is to generate a high-quality summary

    1. Highlight the most relevant information from each source
    2. Provide a concise overview of the key points related to the report topic
    3. Emphasize significant findings or insights
    4. Ensure a coherent flow of information

    CRITICAL REQUIREMENTS:
    - Start IMMEDIATELY with the summary content - no introductions or meta-commentary
    - DO NOT include ANY of the following:
      * Phrases about your thought process ("Let me start by...", "I should...", "I'll...")
      * Explanations of what you're going to do
      * Statements about understanding or analyzing the sources
      * Mentions of summary extension or integration
    - Focus ONLY on factual, objective information
    - Maintain a consistent technical depth
    - Avoid redundancy and repetition
    - DO NOT use phrases like "based on the new results" or "according to additional sources"
    - DO NOT add a References or Works Cited section
    - DO NOT use any XML-style tags like <think> or <answer>
    - Begin directly with the summary text without any tags, prefixes, or meta-commentary
    """

    def write_report(self, webpage_text: str) -> str:
        messages = [
            {
                "role": "system",
                "content": self.summary_system_prompt,
            },
            {"role": "user", "content": webpage_text},
        ]
        response = litellm.completion(
            model=LITELLM_MODEL,
            messages=messages,
            api_base=LITELLM_BASE_URL,
            stream=False,
        )
        return response["choices"][0]["message"]["content"]


def main(args: argparse.Namespace) -> None:
    question_slug = generate_slug(args.question)
    target_folder = Path(args.target_folder) / question_slug
    target_folder.mkdir(parents=True, exist_ok=True)

    raw_data_folder = target_folder / "raw_data"
    raw_data_folder.mkdir(exist_ok=True)

    question = Question(args.question)
    search_engine = SearchEngine()
    websites = search_engine.search_for_question(question.receive_question())
    writer = SummaryWriter()

    output_file = target_folder / f"{question_slug}.md"
    summaries = []

    for website in websites:
        scraped_text = website.get_summary()
        website_slug = generate_slug(website.url)
        raw_data_file = raw_data_folder / f"{website_slug}.txt"
        with open(raw_data_file, "w", encoding="utf-8") as f:
            f.write(scraped_text)

        generated_summary = writer.write_report(scraped_text)
        summary = Summary(generated_summary, website.text, website.url)
        summaries.append(summary)

    with open(output_file, "w", encoding="utf-8") as f:
        for summary in summaries:
            f.write(f"{summary}")

        collective_summaries = " ".join([summary.summary_text for summary in summaries])
        final_report = writer.write_report(collective_summaries)
        f.write("# Final Report\n")
        f.write(f"{final_report}\n\n")
        f.write("# References: \n")
        for summary in summaries:
            f.write(f"- {summary.link}\n")


if __name__ == "__main__":
    args = parse_args()
    setup_logging(args.verbose)
    main(args)
