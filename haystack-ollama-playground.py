#!/usr/bin/env -S uv run --quiet --script
# /// script
# dependencies = [
#   "haystack-ai",
#   "ollama-haystack",
#   "chroma-haystack",
# ]
# ///
"""
Haystack Ollama playground script for various NLP tasks:
- Text generation using BM25 retrieval
- Chat interaction
- Semantic search using embeddings

Usage:
./haystack_ollama_playground.py -h

./haystack_ollama_playground.py -v # To log INFO messages
./haystack_ollama_playground.py -vv # To log DEBUG messages
"""

import logging
from argparse import ArgumentParser
from argparse import RawDescriptionHelpFormatter
from pathlib import Path

from haystack import Document
from haystack import Pipeline
from haystack.components.builders.prompt_builder import PromptBuilder
from haystack.components.retrievers.in_memory import InMemoryBM25Retriever
from haystack.components.retrievers.in_memory import InMemoryEmbeddingRetriever
from haystack.dataclasses import ChatMessage
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack_integrations.components.embedders.ollama.document_embedder import (
    OllamaDocumentEmbedder,
)
from haystack_integrations.components.embedders.ollama.text_embedder import (
    OllamaTextEmbedder,
)
from haystack_integrations.components.generators.ollama import OllamaChatGenerator
from haystack_integrations.components.generators.ollama import OllamaGenerator
from haystack_integrations.document_stores.chroma import ChromaDocumentStore

from logger import setup_logging


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
        "--model",
        default="llama3.2:latest",
        dest="model",
        help="Ollama model to use",
    )
    parser.add_argument(
        "--url",
        default="http://localhost:11434",
        dest="url",
        help="Ollama server URL",
    )
    return parser.parse_args()


def run_generation_example():
    print("------ Running generation example")
    # document_store = InMemoryDocumentStore()
    document_store = ChromaDocumentStore(
        persist_path=Path.cwd().joinpath("target").as_posix()
    )
    document_store.write_documents(
        [
            Document(content="Super Mario was an important politician"),
            Document(
                content="Mario owns several castles and uses them to conduct important political business"
            ),
            Document(
                content="Super Mario was a successful military leader who fought off several invasion attempts by "
                "his arch rival - Bowser"
            ),
        ]
    )

    template = """
    Given only the following information, answer the question.
    Ignore your own knowledge.

    Context:
    {% for document in documents %}
        {{ document.content }}
    {% endfor %}

    Question: {{ query }}?
    """

    pipe = Pipeline()
    pipe.add_component(
        "retriever", InMemoryBM25Retriever(document_store=document_store)
    )
    pipe.add_component("prompt_builder", PromptBuilder(template=template))
    pipe.add_component("llm", OllamaGenerator(model=OLLAMA_MODEL, url=OLLAMA_URL))
    pipe.connect("retriever", "prompt_builder.documents")
    pipe.connect("prompt_builder", "llm")

    query = "Who is Super Mario?"
    response = pipe.run(
        {"prompt_builder": {"query": query}, "retriever": {"query": query}}
    )
    print(f"Generation response: {response['llm']['replies']}")


def run_chat_example():
    print("------ Running chat example")

    messages = [
        ChatMessage.from_user("What's Natural Language Processing?"),
        ChatMessage.from_system(
            "Natural Language Processing (NLP) is a field of computer science and artificial "
            "intelligence concerned with the interaction between computers and human language"
        ),
        ChatMessage.from_user("How do I get started?"),
    ]
    client = OllamaChatGenerator(model=OLLAMA_MODEL, timeout=45, url=OLLAMA_URL)
    response = client.run(messages, generation_kwargs={"temperature": 0.2})
    print(f"Chat response: {response['replies'][0].text}")


def run_embedding_example():
    print("------ Running embedding example")
    document_store = InMemoryDocumentStore(embedding_similarity_function="cosine")
    documents = [
        Document(content="I saw a black horse running"),
        Document(content="Germany has many big cities"),
        Document(content="My name is Wolfgang and I live in Berlin"),
    ]

    document_embedder = OllamaDocumentEmbedder()
    documents_with_embeddings = document_embedder.run(documents)["documents"]
    document_store.write_documents(documents_with_embeddings)

    query_pipeline = Pipeline()
    query_pipeline.add_component("text_embedder", OllamaTextEmbedder())
    query_pipeline.add_component(
        "retriever", InMemoryEmbeddingRetriever(document_store=document_store)
    )
    query_pipeline.connect("text_embedder.embedding", "retriever.query_embedding")

    query = "Who lives in Berlin?"
    result = query_pipeline.run({"text_embedder": {"text": query}})
    print(f"Embedding search result: {result['retriever']['documents'][0]}")


def main(args):
    logging.debug(f"Starting Haystack Ollama playground with verbosity: {args.verbose}")

    global OLLAMA_MODEL
    global OLLAMA_URL
    OLLAMA_MODEL = args.model
    OLLAMA_URL = args.url

    try:
        run_generation_example()
        run_chat_example()
        run_embedding_example()
    except Exception as e:
        logging.error(f"Error occurred: {str(e)}", exc_info=True)


if __name__ == "__main__":
    args = parse_args()
    setup_logging(args.verbose)
    main(args)
