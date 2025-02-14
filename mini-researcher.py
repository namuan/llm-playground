#!/usr/bin/env -S uv run --quiet --script
# /// script
# dependencies = [
#   "ollama",
#   "litellm",
#   "googlesearch-python",
# ]
# ///
import logging
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime, UTC
from enum import Enum
from typing import Any, Dict, List, Optional

import ollama
from googlesearch import search
from pydantic import BaseModel

LITELLM_MODEL = "llama3.1:latest"
LITELLM_BASE_URL = "http://localhost:11434"


class SearchQueries(BaseModel):
    queries: List[str]


def llm_call(model, prompt, format, system_prompt="You are a helpful assistant"):
    response = ollama.generate(
        model=model, system=system_prompt, prompt=prompt, format=format
    ).response
    return response


# -------------------------------
# Error Classes
# -------------------------------
class AgentError(Exception):
    pass


class ToolError(Exception):
    pass


class PipelineError(Exception):
    pass


# -------------------------------
# Enums and Dataclasses
# -------------------------------
class DocumentStatus(Enum):
    CREATED = "created"
    RESEARCH = "research"
    REVIEW = "review"
    REVISION = "revision"
    WRITING = "writing"
    PUBLISHING = "publishing"
    FINAL = "final"


@dataclass
class Document:
    title: str
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    status: DocumentStatus = DocumentStatus.CREATED
    version: int = 1
    history: List[Dict[str, Any]] = field(default_factory=list)

    @classmethod
    def create_new(cls, title: str = "", content: str = "") -> "Document":
        now = datetime.now(UTC).isoformat()
        initial_metadata = {
            "created": now,
            "last_modified": now,
        }
        initial_history = [
            {
                "event": "created",
                "timestamp": now,
                "status": DocumentStatus.CREATED.value,
            }
        ]
        return cls(
            title=title,
            content=content,
            metadata=initial_metadata,
            status=DocumentStatus.CREATED,
            version=1,
            history=initial_history,
        )

    def update(
        self,
        content: Optional[str] = None,
        title: Optional[str] = None,
        status: Optional[DocumentStatus] = None,
    ):
        if title:
            self.title = title
        if content:
            self.content = content
        self.version += 1
        now = datetime.now(UTC).isoformat()
        self.metadata["last_modified"] = now
        if status:
            self.status = status
        self.history.append(
            {"event": "updated", "timestamp": now, "status": self.status.value}
        )


# -------------------------------
# Tool Abstract Class and Implementations
# -------------------------------


class Tool(ABC):
    @abstractmethod
    def execute(self, *args, **kwargs) -> Any:
        pass


class TextEditorTool(Tool):
    def execute(self, document: Document) -> Document:
        logging.info("TextEditorTool: Editing document...")
        new_content = document.content + "\n[Edited by Editor Agent]"
        document.update(content=new_content, status=DocumentStatus.RESEARCH)
        return document


class WebSearchTool(Tool):
    def execute(self, document: Document) -> Document:
        logging.info("WebSearchTool: Researching document...")
        query = document.metadata.get("query", document.title)
        search_results = search(query, num_results=10)
        search_content = "\n".join([f"- {result}" for result in search_results])
        new_content = (
            f"{document.content}\n\nSearch Results for '{query}':\n{search_content}"
        )
        document.update(content=new_content, status=DocumentStatus.REVIEW)
        return document


class GrammarCheckTool(Tool):
    def execute(self, document: Document) -> Document:
        logging.info("GrammarCheckTool: Reviewing document...")
        new_content = document.content + "\n[Document reviewed and approved]"
        document.update(content=new_content, status=DocumentStatus.REVISION)
        return document


class ContentGenerationTool(Tool):
    def execute(self, document: Document) -> Document:
        logging.info("ContentGenerationTool: Refining document content...")
        new_content = document.content + "\n[Content refined by Writer Agent]"
        document.update(content=new_content, status=DocumentStatus.PUBLISHING)
        return document


class SearchQueryGeneratorTool(Tool):
    def execute(self, document: Document) -> Document:
        # Extract keywords from title and content
        text = f"{document.title} {document.content}"
        words = text.lower().split()
        stopwords = {"and", "the", "is", "in", "it", "of", "to", "by", "[", "]"}
        keywords = [word for word in words if word not in stopwords and len(word) > 2]

        prompt = f"""
Given the following title and keywords, generate 5-7 focused search queries for research:
Title: {document.title}
Keywords: {', '.join(keywords)}

Generate specific, research-oriented queries that would help gather comprehensive information on this topic.
"""
        result = llm_call(
            model=LITELLM_MODEL,
            prompt=prompt,
            format=SearchQueries.model_json_schema(),
            system_prompt="You are a research assistant specialized in generating effective search queries",
        )

        search_queries = SearchQueries.model_validate_json(result)
        queries = search_queries.queries
        logging.info(f"Search queries: {queries}")

        document.metadata["search_queries"] = queries
        document.update(
            content=document.content + f"\n[Generated {len(queries)} search queries]"
        )
        return document


# -------------------------------
# Agent Abstract Base Class and Implementations
# -------------------------------
class Agent(ABC):
    @abstractmethod
    def process(self, document: Document) -> Document:
        pass

    @abstractmethod
    def validate(self, document: Document) -> bool:
        pass

    @abstractmethod
    def log_action(self, action: str):
        pass

    @abstractmethod
    def handle_error(self, error: Exception):
        pass


class EditorAgent(Agent):
    def __init__(self):
        self.text_editor = TextEditorTool()

    def process(self, document: Document) -> Document:
        try:
            self.log_action("Starting editing process")
            if not self.validate(document):
                raise AgentError("Validation failed in EditorAgent")
            document = self.text_editor.execute(document)
            self.log_action("Finished editing process")
            return document
        except Exception as e:
            self.handle_error(e)
            raise

    def validate(self, document: Document) -> bool:
        return True

    def log_action(self, action: str):
        logging.info(f"[EditorAgent] {action}")

    def handle_error(self, error: Exception):
        logging.error(f"[EditorAgent] Error: {str(error)}")


class ResearchAgent(Agent):
    def __init__(self):
        self.research_tool = WebSearchTool()

    def process(self, document: Document) -> Document:
        try:
            self.log_action("Starting research process")
            if not self.validate(document):
                raise AgentError("Validation failed in ResearchAgent")
            document = self.research_tool.execute(document)
            self.log_action("Finished research process")
            return document
        except Exception as e:
            self.handle_error(e)
            raise

    def validate(self, document: Document) -> bool:
        return True

    def log_action(self, action: str):
        logging.info(f"[ResearchAgent] {action}")

    def handle_error(self, error: Exception):
        logging.error(f"[ResearchAgent] Error: {str(error)}")


class ReviewerAgent(Agent):
    def __init__(self):
        self.review_tool = GrammarCheckTool()

    def process(self, document: Document) -> Document:
        try:
            self.log_action("Starting review process")
            if not self.validate(document):
                raise AgentError("Validation failed in ReviewerAgent")
            document = self.review_tool.execute(document)
            self.log_action("Finished review process")
            return document
        except Exception as e:
            self.handle_error(e)
            raise

    def validate(self, document: Document) -> bool:
        return True

    def log_action(self, action: str):
        logging.info(f"[ReviewerAgent] {action}")

    def handle_error(self, error: Exception):
        logging.error(f"[ReviewerAgent] Error: {str(error)}")


class WriterAgent(Agent):
    def __init__(self):
        self.writer_tool = ContentGenerationTool()

    def process(self, document: Document) -> Document:
        try:
            self.log_action("Starting writing process")
            if not self.validate(document):
                raise AgentError("Validation failed in WriterAgent")
            document = self.writer_tool.execute(document)
            document.update(status=DocumentStatus.FINAL)
            self.log_action("Finished writing process")
            return document
        except Exception as e:
            self.handle_error(e)
            raise

    def validate(self, document: Document) -> bool:
        return True

    def log_action(self, action: str):
        logging.info(f"[WriterAgent] {action}")

    def handle_error(self, error: Exception):
        logging.error(f"[WriterAgent] Error: {str(error)}")


class SearchQueryAgent(Agent):
    def __init__(self):
        self.query_generator = SearchQueryGeneratorTool()

    def process(self, document: Document) -> Document:
        try:
            self.log_action("Starting search query generation")
            if not self.validate(document):
                raise AgentError("Validation failed in SearchQueryAgent")
            document = self.query_generator.execute(document)
            self.log_action("Finished search query generation")
            return document
        except Exception as e:
            self.handle_error(e)
            raise

    def validate(self, document: Document) -> bool:
        return bool(document.title and document.content)

    def log_action(self, action: str):
        logging.info(f"[SearchQueryAgent] {action}")

    def handle_error(self, error: Exception):
        logging.error(f"[SearchQueryAgent] Error: {str(error)}")


# -------------------------------
# Parallel Processing Function
# -------------------------------
def process_query(query: str, base_document: Document) -> str:
    """
    Process a single search query by running the ResearchAgent and ReviewerAgent sequentially.
    Returns the additional content generated for this query.
    """

    # Create a deepcopy to avoid modifying the base document concurrently.
    document_copy = deepcopy(base_document)
    document_copy.metadata["query"] = query
    # Process with ResearchAgent.
    research_agent = ResearchAgent()
    document_copy = research_agent.process(document_copy)
    # Process with ReviewerAgent.
    reviewer_agent = ReviewerAgent()
    document_copy = reviewer_agent.process(document_copy)
    # The function returns the newly appended content.
    # For simplicity, assume that appended content is what differs after processing.
    return document_copy.content


# -------------------------------
# Pipeline Coordinator
# -------------------------------
class DocumentPipeline:
    def __init__(self):
        self.agents: List[Agent] = [
            EditorAgent(),
            SearchQueryAgent(),  # Generates the search queries.
        ]
        self.history: List[Dict[str, Any]] = []
        self.writer_agent = WriterAgent()

    def process_document(self, document: Document) -> Document:
        try:
            # Run initial agents sequentially.
            for agent in self.agents:
                agent_name = type(agent).__name__
                logging.info(f"Pipeline: Processing document with {agent_name}...")
                document = agent.process(document)
                self.history.append(
                    {
                        "agent": agent_name,
                        "timestamp": datetime.now(UTC).isoformat(),
                        "status": document.status.value,
                        "version": document.version,
                    }
                )
            # Retrieve the search queries generated.
            queries = document.metadata.get("search_queries", [])
            logging.info(
                f"Pipeline: Running parallel processing for {len(queries)} search queries..."
            )
            # Use ThreadPoolExecutor to process each query in parallel.
            aggregated_content = []
            with ThreadPoolExecutor(max_workers=len(queries)) as executor:
                # Using list comprehension to map each query to the worker function.
                results = executor.map(lambda q: process_query(q, document), queries)
                for res in results:
                    aggregated_content.append(res)
            # Aggregate the results into the main document.
            document.content += "\n" + "\n".join(aggregated_content)
            document.update(content=document.content, status=DocumentStatus.WRITING)
            # Now hand over to WriterAgent.
            logging.info(
                "Pipeline: Handing document to WriterAgent for final processing..."
            )
            document = self.writer_agent.process(document)
            return document
        except Exception as e:
            logging.error(f"Pipeline error: {str(e)}")
            raise PipelineError(e)


# -------------------------------
# Main Entry Point
# -------------------------------
def main():
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    pipeline = DocumentPipeline()
    document = Document.create_new(
        title="Artificial Intelligence in Healthcare",
        content="AI applications in medical diagnosis and treatment planning. Machine learning models for patient care optimization.",
    )

    try:
        result = pipeline.process_document(document)
        print("Processing complete:")
        print(f"Status: {result.status.value}")
        print(f"Title: {result.title}")
        print(f"Content:\n{result.content}")
        print(f"Version: {result.version}")
        print("History:")
        for entry in result.history:
            print(f"  - {entry}")
    except PipelineError as e:
        print(f"Processing failed: {str(e)}")


if __name__ == "__main__":
    main()
