#!/usr/bin/env -S uv run --quiet --script
# /// script
# dependencies = [
#   "pandas",
# ]
# ///
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional


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
        now = datetime.utcnow().isoformat()
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
        now = datetime.utcnow().isoformat()
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
        # Simulate editing: update content and move to next status.
        logging.info("TextEditorTool: Editing document...")
        new_content = document.content + "\n[Edited by Editor Agent]"
        document.update(content=new_content, status=DocumentStatus.RESEARCH)
        return document


class WebSearchTool(Tool):
    def execute(self, document: Document) -> Document:
        # Simulate researching: append research content and update status.
        logging.info("WebSearchTool: Researching document...")
        new_content = document.content + "\n[Research data appended]"
        document.update(content=new_content, status=DocumentStatus.REVIEW)
        return document


class GrammarCheckTool(Tool):
    def execute(self, document: Document) -> Document:
        # Simulate reviewing: append review note and update status.
        logging.info("GrammarCheckTool: Reviewing document...")
        new_content = document.content + "\n[Document reviewed and approved]"
        document.update(content=new_content, status=DocumentStatus.REVISION)
        return document


class ContentGenerationTool(Tool):
    def execute(self, document: Document) -> Document:
        # Simulate content generation/refinement.
        logging.info("ContentGenerationTool: Refining document content...")
        new_content = document.content + "\n[Content refined by Writer Agent]"
        document.update(content=new_content, status=DocumentStatus.PUBLISHING)
        return document


# -------------------------------
# Agent Abstract Class and Implementations
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
        # For demo purposes assume validation is successful.
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
            # Set final status.
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


# -------------------------------
# Pipeline Coordinator
# -------------------------------
class DocumentPipeline:
    def __init__(self):
        # Pipeline order as per processing flow:
        self.agents: List[Agent] = [
            EditorAgent(),
            ResearchAgent(),
            ReviewerAgent(),
            WriterAgent(),
        ]
        self.history: List[Dict[str, Any]] = []

    def process_document(self, document: Document) -> Document:
        try:
            for agent in self.agents:
                agent_name = type(agent).__name__
                logging.info(f"Pipeline: Processing document with {agent_name}...")
                document = agent.process(document)
                self.history.append(
                    {
                        "agent": agent_name,
                        "timestamp": datetime.utcnow().isoformat(),
                        "status": document.status.value,
                        "version": document.version,
                    }
                )
            return document
        except Exception as e:
            logging.error(f"Pipeline error: {str(e)}")
            raise PipelineError(e)


# -------------------------------
# Main Entry Point
# -------------------------------
def main():
    # Configure logging
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    # Initialize pipeline and create a new document.
    pipeline = DocumentPipeline()
    document = Document.create_new(title="Example Document", content="Initial content")

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
