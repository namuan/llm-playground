#!/usr/bin/env python3
"""
A personal research assistant

Usage:
./research-agent.py --help

./research-agent.py -q QUESTION -f MARKDOWN_FILE
./research-agent.py -q "What is the best way to learn programming?" -f output.md
"""
import logging
from argparse import ArgumentParser, RawDescriptionHelpFormatter
from itertools import islice
from pathlib import Path
from typing import List

from openai import OpenAI

client = OpenAI()
import requests
from bs4 import BeautifulSoup
from duckduckgo_search import DDGS


def setup_logging(verbosity: int) -> None:
    logging_level = logging.WARNING
    if verbosity == 1:
        logging_level = logging.INFO
    elif verbosity >= 2:
        logging_level = logging.DEBUG

    logging.basicConfig(
        handlers=[
            logging.StreamHandler(),
        ],
        format="%(asctime)s - %(filename)s:%(lineno)d - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging_level,
    )
    logging.captureWarnings(capture=True)


def parse_args() -> argparse.Namespace:
    parser = ArgumentParser(description=__doc__, formatter_class=RawDescriptionHelpFormatter)
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
        "--file",
        type=str,
        required=True,
        help="Target markdown file path",
    )
    return parser.parse_args()


class Question:
    """A class to represent a question."""

    def __init__(self, question_text: str):
        """Initializes the question object."""
        self.question_text = question_text

    def receive_question(self) -> str:
        """Receives a question from the user and returns it as a string."""
        return self.question_text


class SearchEngine:
    """A class to represent a search engine."""

    def __init__(self):
        """Initializes the search engine object."""

    def search_for_question(self, question_text: str) -> list:
        """Searches for the question and returns a list of search results."""
        results = DDGS().text(question_text, region="wt-wt", safesearch="Off", timelimit="y")
        return [Website(result["href"], result["title"], result["body"]) for result in islice(results, 10)]


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

    def __init__(self, summary_text: str, link: list):
        """Initializes the summary object."""
        self.summary_text = summary_text
        self.link = link

    def __str__(self) -> str:
        """Returns the summary as a string."""
        return f"* {self.summary_text}\n{self.link}"


class OpenAIWriter:
    def write_report(self, webpage_text: str) -> str:
        response = client.chat.completions.create(model="gpt-3.5-turbo",
        messages=[
            {
                "role": "system",
                "content": "Summarize content you are provided.Ignore any whitespace and irrelevant information.",
            },
            {"role": "user", "content": webpage_text},
        ],
        temperature=0,
        max_tokens=1024)

        return response.choices[0].message.content

    def embeddings(self, input: List[str]) -> List[List[str]]:
        response = client.embeddings.create(model="text-embedding-ada-002", input=input)
        return [data.embedding for data in response.data]


class SummaryGenerator:
    def __init__(self):
        self.oai = OpenAIWriter()

    def generate_summary(self, summaries):
        return " ".join([summary.summary_text for summary in summaries])


def main(args: argparse.Namespace) -> None:
    question = Question(args.question)
    search_engine = SearchEngine()
    websites = search_engine.search_for_question(question.receive_question())
    print(f"🌐 Found {len(websites)} Websites")
    summaries = []
    oai = OpenAIWriter()
    output_file = Path(args.file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(args.file, "w") as f:
        for website in websites:
            generated_summary = oai.write_report(website.get_summary())
            summary = Summary(generated_summary, website.url)
            summaries.append(summary)
            print(f"📝 {summary.summary_text}")
            f.write(f"# {website.text}\n")
            f.write(f"{summary.summary_text}\n\n")

        summary_generator = SummaryGenerator()
        final_report = summary_generator.generate_summary(summaries)
        f.write("# Final Report\n")
        f.write(f"{final_report}\n\n")


if __name__ == "__main__":
    args = parse_args()
    setup_logging(args.verbose)
    main(args)
