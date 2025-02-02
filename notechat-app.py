#!/usr/bin/env -S uv run --quiet --script
# /// script
# dependencies = [
#   "PyQt6",
#   "sentence_transformers",
#   "trafilatura",
#   "ollama",
#   "sqlite-vec",
# ]
# ///
import logging
import re
import secrets
import sqlite3
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List
from typing import Optional

import ollama
import sqlite_vec
import trafilatura
from PyQt6.QtCore import QPropertyAnimation, QEasingCurve, QObject
from PyQt6.QtCore import QThread, pyqtSignal
from PyQt6.QtCore import QTimer
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QApplication
from PyQt6.QtWidgets import QHBoxLayout
from PyQt6.QtWidgets import QLabel
from PyQt6.QtWidgets import QLineEdit
from PyQt6.QtWidgets import QMainWindow
from PyQt6.QtWidgets import (
    QMessageBox,
)
from PyQt6.QtWidgets import QPushButton
from PyQt6.QtWidgets import QScrollArea
from PyQt6.QtWidgets import QVBoxLayout
from PyQt6.QtWidgets import QWidget
from sentence_transformers import SentenceTransformer

EMBEDDINGS_PATH = Path.home() / ".cache" / "notechat" / "notes.db"
OLLAMA_MODEL = "qwen2.5:latest"
model = SentenceTransformer("all-MiniLM-L6-v2")

EXTRACT_SCRIPT = """
tell application "Notes"
   repeat with eachNote in every note
      set noteId to the id of eachNote
      set noteTitle to the name of eachNote
      set noteBody to the body of eachNote
      set noteCreatedDate to the creation date of eachNote
      set noteCreated to (noteCreatedDate as «class isot» as string)
      set noteUpdatedDate to the modification date of eachNote
      set noteUpdated to (noteUpdatedDate as «class isot» as string)
      set noteContainer to container of eachNote
      set noteFolderId to the id of noteContainer
      log "{split}-id: " & noteId & "\n"
      log "{split}-created: " & noteCreated & "\n"
      log "{split}-updated: " & noteUpdated & "\n"
      log "{split}-folder: " & noteFolderId & "\n"
      log "{split}-title: " & noteTitle & "\n\n"
      log noteBody & "\n"
      log "{split}{split}" & "\n"
   end repeat
end tell
""".strip()


class EmbeddingUtils:
    @staticmethod
    def get_embeddings(text: str) -> bytes:
        return model.encode(text).tobytes()


class DatabaseConnection:
    def __init__(self, db_path: Path):
        if not db_path.parent.exists():
            db_path.parent.mkdir(exist_ok=True)
        self.db_path = db_path.as_posix()
        self.conn: Optional[sqlite3.Connection] = None

    def __enter__(self) -> sqlite3.Cursor:
        try:
            self.conn = sqlite3.connect(self.db_path)
            # Enable and load the vector search extension
            self.conn.enable_load_extension(True)
            sqlite_vec.load(self.conn)
            self.conn.enable_load_extension(False)
            return self.conn.cursor()
        except sqlite3.Error as e:
            logging.error(f"Error connecting to database: {e}")
            raise

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.conn:
            if exc_type:
                self.conn.rollback()
            else:
                self.conn.commit()
            self.conn.close()


class DatabaseManager:
    def __init__(self, db_path: Path):
        self.db_path = db_path

    def initialize_database(self) -> None:
        with DatabaseConnection(self.db_path) as cursor:
            try:
                # Create the main notes table
                cursor.execute(
                    """
                    CREATE TABLE IF NOT EXISTS notes (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        title TEXT NOT NULL,
                        created DATETIME NOT NULL,
                        updated DATETIME NOT NULL,
                        folder TEXT,
                        content TEXT NOT NULL,
                        content_embeddings BLOB
                    )
                """
                )

                # Create the VSS virtual table
                cursor.execute(
                    """
                    CREATE VIRTUAL TABLE IF NOT EXISTS vss_notes USING vec0(
                        content_embedding float[384]
                    )
                    """
                )

                # Create the FTS virtual table
                cursor.execute(
                    """
                    CREATE VIRTUAL TABLE IF NOT EXISTS fts_notes using fts5(
                      content,
                      content='notes', content_rowid='id'
                    );
                    """
                )
            except sqlite3.Error as e:
                print(f"Error creating tables: {e}")
                raise

    def find_similar_notes(self, query: str, limit: int = 5) -> list:
        with DatabaseConnection(self.db_path) as cursor:
            try:
                cursor.execute(
                    """
                with vec_matches as (
                  select
                    rowid,
                    row_number() over (order by distance) as rank_number,
                    distance
                  from vss_notes
                  where
                    content_embedding match :embedding
                    and k = :limit
                ),
                -- the FTS5 search results
                fts_matches as (
                  select
                    rowid,
                    row_number() over (order by rank) as rank_number,
                    rank as score
                  from fts_notes
                  where content match :query
                  limit :limit
                ),
                -- combine FTS5 + vector search results with RRF
                final as (
                  select
                    notes.id as id,
                    notes.title as title,
                    notes.content as content,
                    notes.created as created,
                    notes.folder as folder,
                    vec_matches.rank_number as vec_rank,
                    fts_matches.rank_number as fts_rank,
                    (
                      coalesce(1.0 / (:rrf_k + fts_matches.rank_number), 0.0) * :weight_fts +
                      coalesce(1.0 / (:rrf_k + vec_matches.rank_number), 0.0) * :weight_vec
                    ) as combined_rank,
                    vec_matches.distance as vec_distance,
                    fts_matches.score as fts_score
                  from fts_matches
                  full outer join vec_matches on vec_matches.rowid = fts_matches.rowid
                  join notes on notes.id = coalesce(fts_matches.rowid, vec_matches.rowid)
                  order by combined_rank desc
                )
                select title, folder, content, created from final;
                    """,
                    {
                        "query": query,
                        "embedding": EmbeddingUtils.get_embeddings(query),
                        "limit": limit,
                        "rrf_k": 60,
                        "weight_fts": 1.0,
                        "weight_vec": 1.0,
                    },
                )
                return cursor.fetchall()
            except sqlite3.Error as e:
                logging.error(f"Error searching notes: {e}")
                raise


class SummarizationAssistant:
    def summary_prompt_for(self, matching_notes: str) -> str:
        prompt = f"""
        You are a summarization assistant. Below is a list of notes.
        Your task is to generate an accurate and concise summary that captures the key points from these notes.
        Identify key supporting ideas
        Highlight important facts or evidence
        Reveal the author's purpose or perspective
        Explore any significant implications or conclusions.

        Please provide your answer strictly in valid HTML.
        Do not include any markdown formatting (such as markdown quotes or code block formatting), explanations, or any text outside of the HTML.
        The HTML should include appropriate tags (e.g., <html>, <head>, <body>, <h1>, <p>, <ul>, <li>) for a complete HTML document if applicable.

        List of Notes:
        {matching_notes}

        Summary (in HTML):
        """
        return prompt

    def generate_summary(self, matching_notes: str) -> str:
        prompt = self.summary_prompt_for(matching_notes)
        response = ollama.generate(
            model=OLLAMA_MODEL,
            system="You are a helpful summary generator for selected notes.",
            prompt=prompt,
        ).response
        return response


class StreamingLabel(QLabel):
    finished = pyqtSignal()

    def __init__(self):
        super().__init__()
        self.setWordWrap(True)
        self.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.setTextFormat(Qt.TextFormat.RichText)
        self.full_text = ""
        self.current_position = 0
        self.words = []
        self.timer = QTimer()
        self.timer.timeout.connect(self.add_word)

    def stream_text(self, text, interval=50):
        self.full_text = text
        self.current_position = 0
        self.words = text.split()
        self.setText("")
        self.timer.start(interval)

    def add_word(self):
        if self.current_position < len(self.words):
            current_text = self.text()
            new_text = (
                current_text
                + (" " if current_text else "")
                + self.words[self.current_position]
            )
            self.setText(new_text)
            self.current_position += 1
        else:
            self.timer.stop()
            self.finished.emit()


class EmbeddingsWorker(QThread):
    progress_signal = pyqtSignal(str)

    def __init__(self, notes):
        super().__init__()
        self.notes = notes

    def chunk_content(self, content: str, note_id: str) -> List[dict]:
        chunks = []
        sentences = re.split("[.!?]+", content)
        current_chunk = []
        current_size = 0

        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue

            sentence_size = len(sentence)

            if current_size + sentence_size > 1000:
                if current_chunk:  # Save current chunk
                    chunk_content = ". ".join(current_chunk) + "."
                    chunks.append(
                        {"id": f"{note_id}_{len(chunks)}", "content": chunk_content}
                    )
                    current_chunk = []
                    current_size = 0

            current_chunk.append(sentence)
            current_size += sentence_size

        # Handle remaining chunk
        if current_chunk:
            chunk_content = ". ".join(current_chunk) + "."
            chunks.append({"id": f"{note_id}_{len(chunks)}", "content": chunk_content})

        return chunks

    def run(self):
        try:
            self.progress_signal.emit("Creating embeddings...")
            filtered_notes_with_content = []

            for note in self.notes:
                content = trafilatura.extract(note["body"])
                if content:
                    chunks = self.chunk_content(content, note["id"])
                    for chunk in chunks:
                        note_with_content = note.copy()
                        note_with_content["id"] = chunk["id"]
                        note_with_content["extracted_content"] = chunk["content"]
                        filtered_notes_with_content.append(note_with_content)

            # Prepare data for adding to database
            with DatabaseConnection(EMBEDDINGS_PATH) as cursor:
                for note in filtered_notes_with_content:
                    embeddings = EmbeddingUtils.get_embeddings(
                        note["extracted_content"]
                    )
                    created = datetime.fromisoformat(note["created"])
                    updated = datetime.fromisoformat(note["updated"])
                    # Insert into main notes table
                    cursor.execute(
                        """
                        INSERT INTO notes (title, created, updated, folder, content, content_embeddings)
                        VALUES (?, ?, ?, ?, ?, ?)
                        """,
                        (
                            note["title"],
                            created,
                            updated,
                            note["folder"],
                            note["extracted_content"],
                            embeddings,
                        ),
                    )

                    # Get the rowid of the inserted note
                    note_id = cursor.lastrowid

                    # Insert into VSS virtual table
                    cursor.execute(
                        """
                        INSERT INTO vss_notes(rowid, content_embedding)
                        VALUES (?, ?)
                        """,
                        (note_id, embeddings),
                    )

            self.progress_signal.emit("Embeddings created successfully!")

            with DatabaseConnection(EMBEDDINGS_PATH) as cursor:
                cursor.execute(
                    """
                    insert into fts_notes(rowid, content)
                    select rowid, content from notes;
                    """,
                )

            self.progress_signal.emit("Search Index created successfully!")

        except Exception as e:
            self.progress_signal.emit(f"Error creating embeddings: {str(e)}")


class NotesExtractorWorker(QThread):
    progress_signal = pyqtSignal(str)
    note_extracted = pyqtSignal(dict)
    error = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.split = secrets.token_hex(8)

    def run(self):
        try:
            # Get total number of notes first
            total_notes = int(
                subprocess.check_output(
                    [
                        "osascript",
                        "-e",
                        'tell application "Notes" to get count of notes',
                    ],
                ).strip()
            )
            self.progress_signal.emit(f"Processing {total_notes} notes ...")

            # Start extraction process
            process = subprocess.Popen(
                ["osascript", "-e", EXTRACT_SCRIPT.format(split=self.split)],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
            )

            note: Dict[str, str] = {}
            body: List[str] = []

            for line in process.stdout:
                if self.isInterruptionRequested():
                    process.terminate()
                    return

                line = line.decode("mac_roman").strip()

                if line == f"{self.split}{self.split}":
                    if note.get("id"):
                        note["body"] = "\n".join(body).strip()
                        self.note_extracted.emit(note)
                    note, body = {}, []
                    continue

                found_key = False
                for key in ("id", "title", "folder", "created", "updated"):
                    if line.startswith(f"{self.split}-{key}: "):
                        note[key] = line[len(f"{self.split}-{key}: ") :]
                        found_key = True
                        break
                if not found_key:
                    body.append(line)

            process.stdout.close()
            process.wait()

        except Exception as e:
            self.error.emit(str(e))


class OllamaStatusWorker(QThread):
    status_signal = pyqtSignal(bool)

    def run(self):
        try:
            ollama.ps()
            self.status_signal.emit(True)
        except Exception:
            self.status_signal.emit(False)


class NotesExtractor(QObject):
    progress_signal = pyqtSignal(str)
    note_extracted = pyqtSignal(dict)
    finished = pyqtSignal()
    error = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.worker = NotesExtractorWorker()

        # Connect worker signals
        self.worker.progress_signal.connect(self.progress_signal)
        self.worker.note_extracted.connect(self.note_extracted)
        self.worker.error.connect(self.error)
        self.worker.finished.connect(self.finished)
        self.worker.finished.connect(self.worker.deleteLater)

    def start_extraction(self):
        self.worker.start()

    def stop_extraction(self):
        self.worker.requestInterruption()


class Notechat(QMainWindow):
    def __init__(self):
        super().__init__()
        self.chat_layout = None
        self.text_input = None
        self.extracted_notes = []
        self.extractor = self.setup_extractor()
        self.db_manager = DatabaseManager(EMBEDDINGS_PATH)
        self.summarization_assistant = SummarizationAssistant()
        self.init_ui()
        self.db_manager.initialize_database()
        self.check_ollama_status()

    def create_title_bar(self):
        title_bar = QWidget()
        title_bar_layout = QHBoxLayout(title_bar)
        title_bar_layout.setContentsMargins(10, 10, 10, 10)

        title_widget = QWidget()
        title_layout = QHBoxLayout(title_widget)
        title_layout.setContentsMargins(0, 0, 0, 0)

        self.icon_label = QLabel()
        self.icon_label.setFixedSize(20, 20)
        self.icon_label.setStyleSheet("background-color: #FFD700; border-radius: 5px;")

        self.progress_label = QLabel()
        self.progress_label.setStyleSheet(
            """
            padding: 2px 8px;
            color: #666;
            font-size: 12px;
        """
        )
        self.progress_label.hide()

        title_layout.addWidget(self.icon_label)
        title_layout.addWidget(self.progress_label)
        title_layout.addStretch()

        refresh_notes_button = QPushButton("Refresh Notes")
        refresh_notes_button.setStyleSheet(
            """
            QPushButton {
                padding: 5px 10px;
                border: none;
                border-radius: 5px;
                background-color: #4CAF50;
                color: white;
            }
        """
        )
        refresh_notes_button.clicked.connect(self.handle_extraction)

        title_bar_layout.addWidget(title_widget)
        title_bar_layout.addWidget(refresh_notes_button)

        return title_bar

    def check_ollama_status(self):
        self.ollama_worker = OllamaStatusWorker()
        self.ollama_worker.status_signal.connect(self.update_ollama_status)
        self.ollama_worker.start()

    def update_ollama_status(self, is_connected: bool):
        if is_connected:
            self.icon_label.setStyleSheet(
                "background-color: #4CAF50; border-radius: 5px;"
            )
            self.icon_label.setToolTip("Ollama is connected and running")
        else:
            self.icon_label.setStyleSheet(
                "background-color: #F44336; border-radius: 5px;"
            )
            self.icon_label.setToolTip("Ollama is not connected")

    def setup_extractor(self):
        self.extractor = NotesExtractor()

        # Connect signals
        self.extractor.progress_signal.connect(self.update_progress_message)
        self.extractor.note_extracted.connect(self.handle_note)
        self.extractor.error.connect(self.handle_error)
        self.extractor.finished.connect(self.extraction_finished)

        return self.extractor

    def handle_extraction(self):
        # Start extraction
        self.extractor.start_extraction()

    def handle_note(self, note: dict):
        self.extracted_notes.append(note)

    def update_progress_message(self, message: str):
        self.progress_label.show()
        self.progress_label.setText(message)

    def extraction_finished(self):
        self.progress_label.hide()
        self.handle_extracted_notes(self.extracted_notes)

    def handle_error(self, error_msg: str):
        QMessageBox.critical(self, "Error", f"Error extracting notes: {error_msg}")

    def handle_extracted_notes(self, notes: List[Dict[str, str]]):
        message = f"Extracted {len(notes)} notes successfully!"
        # Add system message to chat
        system_widget = QLabel(message)
        system_widget.setStyleSheet(
            """
            background-color: #E8F5E9;
            padding: 10px;
            border-radius: 5px;
        """
        )
        self.chat_layout.insertWidget(self.chat_layout.count() - 1, system_widget)
        self.prepare_embeddings(notes)

    def create_chat_area(self):
        chat_widget = QWidget()
        self.chat_layout = QVBoxLayout(chat_widget)
        self.chat_layout.setContentsMargins(20, 20, 20, 20)
        self.chat_layout.setSpacing(20)
        self.chat_layout.addStretch()

        scroll_area = QScrollArea()
        scroll_area.setWidget(chat_widget)
        scroll_area.setWidgetResizable(True)
        scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll_area.setStyleSheet(
            """
                QScrollArea {
                    border: none;
                    background-color: transparent;
                }
                QScrollBar:vertical {
                    width: 10px;
                    background: transparent;
                }
                QScrollBar::handle:vertical {
                    background: #CCCCCC;
                    border-radius: 5px;
                }
            """
        )
        return scroll_area

    def create_input_area(self):
        input_widget = QWidget()
        input_layout = QHBoxLayout(input_widget)
        input_layout.setContentsMargins(20, 10, 20, 10)

        self.text_input = QLineEdit()
        self.text_input.setPlaceholderText("Type your message...")
        self.text_input.setStyleSheet(
            """
                QLineEdit {
                    padding: 10px;
                    border: none;
                    border-radius: 5px;
                    background-color: white;
                }
            """
        )

        send_button = QPushButton("Send")
        send_button.setStyleSheet(
            """
                QPushButton {
                    padding: 10px 20px;
                    border: none;
                    border-radius: 5px;
                    background-color: #1E2A38;
                    color: white;
                }
            """
        )

        input_layout.addWidget(self.text_input)
        input_layout.addWidget(send_button)

        # Connect signals
        send_button.clicked.connect(self.handle_send)
        self.text_input.returnPressed.connect(self.handle_send)

        return input_widget

    def init_ui(self):
        self.setWindowTitle("Notechat")
        self.setMinimumSize(1000, 600)
        self.setStyleSheet("QMainWindow { background-color: #F5F5F1; }")

        # Create main layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Add components
        layout.addWidget(self.create_title_bar())
        layout.addWidget(self.create_chat_area())
        layout.addWidget(self.create_input_area())

        # Focus on input
        self.text_input.setFocus()

    def build_assistant_widget(self, response_data):
        assistant_widget = QWidget()
        assistant_widget.setStyleSheet("background-color: white; border-radius: 10px;")
        assistant_layout = QVBoxLayout(assistant_widget)

        header = QWidget()
        header_layout = QHBoxLayout(header)
        icon = QLabel()
        icon.setFixedSize(20, 20)
        icon.setStyleSheet("background-color: #FFD700; border-radius: 5px;")
        title = QLabel("Apple Notes")
        header_layout.addWidget(icon)
        header_layout.addWidget(title)
        header_layout.addStretch()

        # Use StreamingLabel instead of QLabel
        response_text = StreamingLabel()
        response_text.finished.connect(self.on_streaming_finished)

        # Note collections
        collections = QWidget()
        collections_layout = QHBoxLayout(collections)
        collections_layout.setSpacing(10)
        collections.hide()  # Hide initially

        for note_data in response_data["collections"]:
            note_button = QPushButton(f"📝 {note_data['title']}  {note_data['date']}")
            note_button.setStyleSheet(
                """
                QPushButton {
                    background-color: #F5F5F5;
                    border: 1px solid #E0E0E0;
                    border-radius: 5px;
                    padding: 5px 10px;
                    color: #666;
                }
                QPushButton:hover {
                    background-color: #EEEEEE;
                }
            """
            )
            note_button.setCursor(Qt.CursorShape.PointingHandCursor)
            note_button.clicked.connect(
                lambda checked, title=note_data["title"]: self.open_note(title)
            )
            collections_layout.addWidget(note_button)
            collections_layout.addStretch()

        assistant_layout.addWidget(header)
        assistant_layout.addWidget(response_text)
        assistant_layout.addWidget(collections)

        # Start streaming
        response_text.stream_text(response_data["response"])

        # Store collections widget reference
        response_text.collections_widget = collections

        return assistant_widget

    def build_user_widget(self, message=None):
        user_message = QLabel("You")
        user_message.setStyleSheet("font-weight: bold;")
        user_question = QLabel(
            message
            if message
            else "Can you find my notes about high-protein breakfast recipes and meal prep ideas?"
        )
        user_question.setWordWrap(True)

        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(user_message)
        layout.addWidget(user_question)

        return container

    def handle_send(self):
        message = self.text_input.text()
        if message.strip():
            user_widget = self.build_user_widget(message)
            self.chat_layout.insertWidget(self.chat_layout.count() - 1, user_widget)
            matching_documents = self.search_embeddings(message)
            matching_documents["response"] = self.generate_summary_from(
                matching_documents["response"]
            )
            assistant_widget = self.build_assistant_widget(matching_documents)
            self.chat_layout.insertWidget(
                self.chat_layout.count() - 1, assistant_widget
            )
            QTimer.singleShot(200, self.scroll_to_bottom)

    def scroll_to_bottom(self):
        scroll_area = self.findChild(QScrollArea)
        if scroll_area:
            QApplication.processEvents()
            vertical_bar = scroll_area.verticalScrollBar()
            current = vertical_bar.value()
            maximum = vertical_bar.maximum()
            animation = QPropertyAnimation(vertical_bar, b"value")
            animation.setDuration(400)
            animation.setStartValue(current)
            animation.setEndValue(maximum)
            animation.setEasingCurve(QEasingCurve.Type.OutCubic)
            animation.start()
            # Keep reference to prevent garbage collection
            self._animation = animation

    def on_streaming_finished(self):
        # Show collections after streaming is complete
        streaming_label = self.sender()
        streaming_label.collections_widget.show()
        # Schedule scroll to bottom
        QTimer.singleShot(200, self.scroll_to_bottom)
        # Clear input field
        self.text_input.clear()

    def prepare_embeddings(self, notes: List[Dict[str, str]]):
        self.embeddings_worker = EmbeddingsWorker(notes)
        self.embeddings_worker.progress_signal.connect(self.update_progress_message)
        self.embeddings_worker.start()

    def search_embeddings(self, message):
        try:
            results = self.db_manager.find_similar_notes(message, limit=2)
            response_data = {"response": "", "collections": []}

            if results:
                # Build natural response from results
                response_text = ""
                seen_titles = set()

                for title, folder, content, created in results:
                    # Add a summary sentence from each document
                    summary = content[:500] + "..." if len(content) > 500 else content
                    response_text += f"{summary} "

                    # Only add to collections if title hasn't been seen yet
                    title_date = (title, created)
                    if title_date not in seen_titles:
                        seen_titles.add(title_date)
                        response_data["collections"].append(
                            {
                                "title": title,
                                "date": created.split(" ")[0],
                            }
                        )

                response_data["response"] = response_text.strip()
            else:
                response_data["response"] = (
                    "I couldn't find any relevant notes matching your query."
                )

            return response_data

        except Exception as e:
            return {"response": f"Error searching notes: {str(e)}", "collections": []}

    def generate_summary_from(self, matching_notes: str):
        return self.summarization_assistant.generate_summary(matching_notes)

    def open_note(self, title):
        script = f"""
        tell application "Notes"
            show note "{title}"
        end tell
        """
        try:
            subprocess.run(["osascript", "-e", script], check=True)
        except subprocess.CalledProcessError as e:
            QMessageBox.warning(self, "Error", f"Could not open note: {str(e)}")


class NotechatApp:
    @staticmethod
    def run():
        app = QApplication(sys.argv)
        window = Notechat()
        window.show()
        sys.exit(app.exec())


if __name__ == "__main__":
    NotechatApp.run()
