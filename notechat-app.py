#!/usr/bin/env -S uv run --quiet --script
# /// script
# dependencies = [
#   "PyQt6",
# ]
# ///

import secrets
import subprocess
import sys
from typing import Dict, List

from PyQt6.QtCore import QPropertyAnimation, QEasingCurve, QObject
from PyQt6.QtCore import QThread
from PyQt6.QtCore import QTimer
from PyQt6.QtCore import Qt
from PyQt6.QtCore import pyqtSignal
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


# Add this new class after the imports
class StreamingLabel(QLabel):
    finished = pyqtSignal()

    def __init__(self):
        super().__init__()
        self.setWordWrap(True)
        self.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
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


def get_response_data():
    # Mock data - will be replaced with database query
    return {
        "response": (
            "I found several breakfast and meal prep ideas in your notes. For high-protein breakfasts, you've saved "
            "a recipe for overnight protein oats (32g protein) with whey, chia seeds, and Greek yogurt. There's also "
            "a savory breakfast bowl with scrambled tofu, black beans, and quinoa (28g protein). For meal prep, "
            "you noted a weekly prep routine: hard-boiled eggs, turkey-veggie egg white muffins, and protein "
            "pancakes made with cottage cheese. You also bookmarked a protein smoothie recipe with frozen "
            "banana, spinach, protein powder, and almond butter that you rated 9/10 for taste."
        ),
        "collections": [
            {"title": "Breakfast Recipe Collection", "date": "15/01/2024"},
            {"title": "Weekly Meal Prep Guide", "date": "28/02/2024"},
            {"title": "Protein-Rich Recipes", "date": "10/03/2024"},
        ],
    }


class NotesExtractorWorker(QThread):
    progress_signal = pyqtSignal(int)  # total notes count
    note_extracted = pyqtSignal(dict)  # extracted note data
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
            self.progress_signal.emit(total_notes)

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


class NotesExtractor(QObject):
    progress_signal = pyqtSignal(int)
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
        self.init_ui()

    def create_title_bar(self):
        title_bar = QWidget()
        title_bar_layout = QHBoxLayout(title_bar)
        title_bar_layout.setContentsMargins(10, 10, 10, 10)

        # Create title widget with icon
        title_widget = QWidget()
        title_layout = QHBoxLayout(title_widget)
        title_layout.setContentsMargins(0, 0, 0, 0)

        icon_label = QLabel()
        icon_label.setFixedSize(20, 20)
        icon_label.setStyleSheet("background-color: #FFD700; border-radius: 5px;")

        title_layout.addWidget(icon_label)
        title_layout.addStretch()

        extract_button = QPushButton("Extract Notes")
        extract_button.setStyleSheet("""
            QPushButton {
                padding: 5px 10px;
                border: none;
                border-radius: 5px;
                background-color: #4CAF50;
                color: white;
            }
        """)
        extract_button.clicked.connect(self.handle_extraction)

        # Create status widget
        status_widget = QWidget()
        status_layout = QHBoxLayout(status_widget)
        status_layout.setContentsMargins(0, 0, 0, 0)

        active_label = QLabel("Active")
        active_label.setStyleSheet(
            "background-color: #2ECC71; color: white; padding: 5px 10px; border-radius: 15px;"
        )

        status_layout.addWidget(active_label)

        title_bar_layout.addWidget(title_widget)
        title_bar_layout.addWidget(extract_button)
        title_bar_layout.addWidget(status_widget, alignment=Qt.AlignmentFlag.AlignRight)

        return title_bar

    def setup_extractor(self):
        self.extractor = NotesExtractor()

        # Connect signals
        self.extractor.progress_signal.connect(self.update_progress)
        self.extractor.note_extracted.connect(self.handle_note)
        self.extractor.error.connect(self.handle_error)
        self.extractor.finished.connect(self.extraction_finished)

        return self.extractor

    def handle_extraction(self):
        # Start extraction
        self.extractor.start_extraction()

    def handle_note(self, note: dict):
        self.extracted_notes.append(note)

    def update_progress(self, total: int): ...

    def extraction_finished(self):
        self.handle_extracted_notes(self.extracted_notes)

    def handle_error(self, error_msg: str):
        QMessageBox.critical(self, "Error", f"Error extracting notes: {error_msg}")

    def handle_extracted_notes(self, notes: List[Dict[str, str]]):
        message = f"Extracted {len(notes)} notes successfully!"

        # Add system message to chat
        system_widget = QLabel(message)
        system_widget.setStyleSheet("""
            background-color: #E8F5E9;
            padding: 10px;
            border-radius: 5px;
        """)
        self.chat_layout.insertWidget(self.chat_layout.count() - 1, system_widget)

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

    def build_assistant_widget(self):
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

        # Get response data
        response_data = get_response_data()

        # Use StreamingLabel instead of QLabel
        response_text = StreamingLabel()
        response_text.finished.connect(self.on_streaming_finished)

        # Note collections
        collections = QWidget()
        collections_layout = QHBoxLayout(collections)
        collections_layout.setSpacing(10)
        collections.hide()  # Hide initially

        for note_data in response_data["collections"]:
            note = QLabel(f"📝 {note_data['title']}  {note_data['date']}")
            note.setStyleSheet("color: #666;")
            collections_layout.addWidget(note)
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
            # Add user message
            user_widget = self.build_user_widget(message)
            self.chat_layout.insertWidget(self.chat_layout.count() - 1, user_widget)

            # Add assistant response
            assistant_widget = self.build_assistant_widget()
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


def main():
    app = QApplication(sys.argv)
    window = Notechat()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
