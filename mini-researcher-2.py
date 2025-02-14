#!/usr/bin/env -S uv run --quiet --script
# /// script
# dependencies = [
#   "requests",
#   "beautifulsoup4",
#   "html2text",
#   "networkx",
#   "pyvis",
# ]
# ///
"""
Search Graph Generator Script

This script takes a question as input and generates a knowledge graph by:
1. Understanding the question and setting up assumptions
2. Generating search queries
3. Running searches and processing results
4. Building and displaying a graph structure

Usage:
./search_graph.py -h
./search_graph.py -q "What is machine learning?" -v  # For INFO logging
./search_graph.py -q "What is machine learning?" -vv  # For DEBUG logging
"""

import logging
import time
from argparse import ArgumentParser, RawDescriptionHelpFormatter
from concurrent.futures import ThreadPoolExecutor
from threading import Lock
from urllib.parse import quote_plus

import networkx as nx
import requests
from bs4 import BeautifulSoup
from html2text import HTML2Text
from pyvis.network import Network


class SearchGraph:
    def __init__(self, question):
        self.question = question
        self.graph = nx.Graph(title=question)
        self.search_queries = []
        self.graph_lock = Lock()

    def generate_search_queries(self):
        """Generate 10 search queries based on the input question"""
        logging.debug(f"Generating search queries for question: {self.question}")
        base_queries = [
            self.question,
            f"how to {self.question}",
            f"what is {self.question}",
            f"explain {self.question}",
            f"{self.question} tutorial",
            f"{self.question} guide",
            f"{self.question} examples",
            f"{self.question} best practices",
            f"{self.question} overview",
            f"{self.question} detailed explanation",
        ]
        self.search_queries = base_queries
        return base_queries

    def visualize_graph(self, output_file="search_graph.html"):
        """
        Create an interactive visualization of the graph
        """
        logging.info(f"Generating interactive visualization: {output_file}")

        net = Network(
            height="750px",
            width="100%",
            bgcolor="#ffffff",
            font_color="#000000",
            notebook=False,
        )

        net.force_atlas_2based(
            gravity=-50,
            central_gravity=0.01,
            spring_length=100,
            spring_strength=0.08,
            damping=0.4,
            overlap=0,
        )

        color_map = {"question": "#ff7675", "query": "#74b9ff", "result": "#55efc4"}

        net.add_node(
            self.question,
            label=self.question[:30] + "..."
            if len(self.question) > 30
            else self.question,
            color=color_map["question"],
            size=20,
            title=self.question,
        )

        for node, data in self.graph.nodes(data=True):
            if node == self.question:
                continue

            if node in self.search_queries:
                node_color = color_map["query"]
                node_size = 15
            else:
                node_color = color_map["result"]
                node_size = 10

            label = node[:30] + "..." if len(node) > 30 else node
            title = data.get("content", node)
            if title:
                title = title[:200] + "..." if len(title) > 200 else title

            net.add_node(
                node, label=label, color=node_color, size=node_size, title=title
            )

        for edge in self.graph.edges():
            net.add_edge(edge[0], edge[1], color="#2d3436")

        try:
            net.write_html(output_file)
            logging.info(f"Successfully generated visualization at: {output_file}")
        except Exception as e:
            logging.error(f"Failed to generate visualization: {e}")

    def fetch_webpage(self, url):
        """Fetch webpage content"""
        logging.debug(f"Fetching webpage: {url}")
        try:
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
            }
            response = requests.get(url, timeout=10, headers=headers)
            return response.text
        except Exception as e:
            logging.error(f"Error fetching {url}: {e}")
            return None

    def html_to_markdown(self, html_content):
        """Convert HTML to Markdown"""
        if not html_content:
            return None

        h2t = HTML2Text()
        h2t.ignore_links = False
        h2t.ignore_images = True
        return h2t.handle(html_content)

    def google_search(self, query, max_results=10):
        """Simulate Google Search"""
        logging.debug(f"Performing search for query: {query}")
        search_url = f"https://www.google.com/search?q={quote_plus(query)}"

        try:
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
            }
            response = requests.get(search_url, headers=headers)
            BeautifulSoup(response.text, "html.parser")
            results = []
            # Simulate results (replace with actual Google Search API in production)
            for i in range(max_results):
                results.append(f"https://example.com/result_{i}")
            return results
        except Exception as e:
            logging.error(f"Error in search: {e}")
            return []

    def process_search_result(self, url, query):
        """Process a single search result"""
        logging.debug(f"Processing result URL: {url}")
        html_content = self.fetch_webpage(url)
        markdown_content = self.html_to_markdown(html_content)

        if markdown_content:
            with self.graph_lock:
                self.graph.add_node(url, content=markdown_content[:500])
                self.graph.add_edge(query, url, weight=1)

        return {
            "url": url,
            "content": markdown_content[:500] if markdown_content else None,
        }

    def process_search_query(self, query):
        """Process a single search query"""
        logging.info(f"Processing query: {query}")
        results = self.google_search(query)

        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [
                executor.submit(self.process_search_result, url, query)
                for url in results
            ]
            results = [future.result() for future in futures]

        return {"query": query, "results": results}

    def process_question(self):
        """Main processing method"""
        logging.info("Step 1: Understanding question")

        logging.info("Step 2: Generating search queries")
        queries = self.generate_search_queries()

        logging.info("Step 3: Processing search queries in parallel")
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = [
                executor.submit(self.process_search_query, query) for query in queries
            ]
            all_results = [future.result() for future in futures]

        logging.info("Step 4: Collecting all outputs")
        logging.info("Step 5: Printing graph structure")
        self.print_graph()

        return all_results

    def print_graph(self):
        """Print the graph structure"""
        logging.info("\nGraph Summary:")
        print(f"Number of nodes: {self.graph.number_of_nodes()}")
        print(f"Number of edges: {self.graph.number_of_edges()}")

        logging.debug("\nNodes:")
        for node in self.graph.nodes(data=True):
            logging.debug(f"Node: {node[0]}")
            if "content" in node[1]:
                logging.debug(f"Content preview: {node[1]['content'][:100]}...")

        logging.debug("\nEdges:")
        for edge in self.graph.edges(data=True):
            logging.debug(
                f"Edge: {edge[0]} -> {edge[1]} (weight: {edge[2].get('weight', 1)})"
            )


def setup_logging(verbosity):
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


def parse_args():
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
        help="The question to process",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default="search_graph.html",
        help="Output HTML file for visualization (default: search_graph.html)",
    )
    return parser.parse_args()


def main(args):
    start_time = time.time()

    search_graph = SearchGraph(args.question)
    search_graph.process_question()

    search_graph.visualize_graph(args.output)

    end_time = time.time()
    logging.info(f"Total processing time: {end_time - start_time:.2f} seconds")
    logging.info(f"Visualization saved to: {args.output}")


if __name__ == "__main__":
    args = parse_args()
    setup_logging(args.verbose)
    main(args)
