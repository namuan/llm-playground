import sys
import json
import requests  # Dependency: pip install requests
import pyperclip  # Dependency: pip install pyperclip
from PyQt6.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QTextEdit,
    QComboBox,
)
from PyQt6.QtCore import Qt, QObject, QThread, pyqtSignal

# --- Ollama API Configuration ---
OLLAMA_ENDPOINT = "http://localhost:11434/api/generate"
OLLAMA_SELECTED_MODEL = "llama3.2:latest"


class ApiWorker(QObject):
    """
    A worker object that runs the Ollama API call in a separate thread
    to avoid blocking the main GUI thread.
    """

    chunk_received = pyqtSignal(str)
    finished = pyqtSignal()
    error_occurred = pyqtSignal(str)

    def __init__(self, prompt):
        super().__init__()
        self.prompt = prompt

    def run(self):
        """
        Makes the streaming API call to Ollama.
        Emits signals for each response chunk, on error, and on completion.
        It is designed to be interruptible.
        """
        url = OLLAMA_ENDPOINT
        payload = {
            "model": OLLAMA_SELECTED_MODEL,
            "prompt": self.prompt,
            "stream": True,
        }
        headers = {"Content-Type": "application/json"}

        try:
            # Check if the thread has been told to stop before starting the request
            if QThread.currentThread().isInterruptionRequested():
                self.finished.emit()
                return

            with requests.post(
                    url, json=payload, headers=headers, stream=True
            ) as response:
                response.raise_for_status()
                for line in response.iter_lines():
                    # Check for interruption during the streaming loop
                    if QThread.currentThread().isInterruptionRequested():
                        break
                    if line:
                        decoded_line = line.decode("utf-8")
                        data = json.loads(decoded_line)
                        response_chunk = data.get("response", "")
                        self.chunk_received.emit(response_chunk)
                        if data.get("done", False):
                            break
        except requests.exceptions.RequestException as e:
            # Avoid emitting an error if the thread was just interrupted.
            if not QThread.currentThread().isInterruptionRequested():
                error_message = f"API request failed: {e}"
                self.error_occurred.emit(error_message)
        finally:
            # Use try-except to prevent crash if worker is deleted before signal emission
            try:
                self.finished.emit()
            except RuntimeError:
                pass


class AppWindow(QMainWindow):
    """
    A desktop application that processes clipboard text using Ollama.
    The user can add additional manual instructions. The "Copy Text" button
    copies the result and closes the app. The window is centered on screen.
    """

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Popup")
        self.setWindowFlags(Qt.WindowType.FramelessWindowHint)
        self.setGeometry(100, 100, 650, 550)
        self.center_window()

        # References to the current running thread and worker
        self.api_thread = None
        self.api_worker = None
        # Store the main text from the clipboard at launch
        self.main_clipboard_text = ""

        self.central_widget = QWidget()
        self.central_widget.setObjectName("centralWidget")
        self.setCentralWidget(self.central_widget)

        self.main_layout = QVBoxLayout(self.central_widget)
        self.main_layout.setContentsMargins(12, 12, 12, 12)
        self.main_layout.setSpacing(10)

        # --- Top Bar ---
        top_bar_layout = QHBoxLayout()
        top_bar_layout.setSpacing(8)

        self.dropdown = QComboBox()
        chat_modes = self.get_chat_modes()
        self.dropdown.addItems(chat_modes.keys())
        self.dropdown.currentTextChanged.connect(self.on_mode_change)

        self.copy_text_button = QPushButton("Copy Text")
        self.copy_text_button.clicked.connect(self.copy_and_close)

        self.close_button = QPushButton("Close")
        self.close_button.setObjectName("closeButton")
        self.close_button.clicked.connect(self.close)

        top_bar_layout.addWidget(self.dropdown, 1)
        top_bar_layout.addWidget(self.copy_text_button)
        top_bar_layout.addWidget(self.close_button)

        # --- Text Input Area ---
        self.text_input = QTextEdit()
        self.text_input.setPlaceholderText(
            "Add additional instructions here (optional)..."
        )
        self.text_input.setFixedHeight(70)

        # --- Read-Only Text Area ---
        self.read_only_text_area = QTextEdit()
        self.read_only_text_area.setReadOnly(True)

        # --- Assemble Main Layout ---
        self.main_layout.addLayout(top_bar_layout)
        self.main_layout.addWidget(self.text_input)
        self.main_layout.addWidget(self.read_only_text_area, 1)

        self.apply_styles()

        self.process_clipboard_on_launch()

    def center_window(self):
        """Centers the application window on the screen."""
        # Get the screen geometry
        screen_geometry = QApplication.primaryScreen().geometry()
        window_geometry = self.frameGeometry()

        # Calculate the center position
        x = (screen_geometry.width() - window_geometry.width()) // 2
        y = (screen_geometry.height() - window_geometry.height()) // 2

        # Move the window to the calculated position
        self.move(x, y)

    def on_mode_change(self):
        """
        Triggered when the dropdown selection changes.
        Reruns the API call with the new mode using the original clipboard text
        and any new manual instructions.
        """
        self.run_generation()

    def run_generation(self):
        """
        Central function to handle the Ollama API call. It safely stops any
        existing call and starts a new one based on the stored clipboard text
        and current manual input.
        """
        if not self.main_clipboard_text.strip():
            self.read_only_text_area.setText("Main text from clipboard is empty.")
            return

        # Safely stop any previous thread that might be running.
        try:
            if self.api_thread and self.api_thread.isRunning():
                self.api_thread.requestInterruption()
                self.api_thread.quit()
        except RuntimeError:
            pass  # Object was already deleted, safe to ignore.

        self.read_only_text_area.clear()

        # Build the prompt for the new request
        mode_key = self.dropdown.currentText()
        prompt_template = self.get_chat_modes().get(mode_key, "")

        additional_instructions = self.text_input.toPlainText().strip()
        prompt_body = self.main_clipboard_text
        if additional_instructions:
            prompt_body += f"\n\nAdditional Instructions: {additional_instructions}"

        full_prompt = f"{prompt_template}\n\n{prompt_body}"

        # Create and configure a new thread and worker
        self.api_thread = QThread()
        self.api_worker = ApiWorker(full_prompt)
        self.api_worker.moveToThread(self.api_thread)

        # Set up connections for the new worker and thread lifecycle management.
        self.api_thread.started.connect(self.api_worker.run)
        self.api_worker.finished.connect(self.api_thread.quit)
        # Ensure both worker and thread are deleted after they finish.
        self.api_worker.finished.connect(self.api_worker.deleteLater)
        self.api_thread.finished.connect(self.api_thread.deleteLater)

        # Connect worker signals to UI slots
        self.api_worker.chunk_received.connect(self.update_output_text)
        self.api_worker.error_occurred.connect(self.show_api_error)

        self.api_thread.start()

    def process_clipboard_on_launch(self):
        """Grabs clipboard text and starts the initial API call."""
        self.main_clipboard_text = pyperclip.paste()
        self.text_input.clear()  # Ensure the manual input is empty

        if not self.main_clipboard_text:
            self.read_only_text_area.setText(
                "Clipboard is empty. Copy some text and restart the application."
            )
            return

        self.run_generation()

    def update_output_text(self, chunk: str):
        """Slot to append new text chunks to the read-only text area."""
        self.read_only_text_area.insertPlainText(chunk)

    def show_api_error(self, error_message: str):
        """Slot to display API errors in the read-only text area."""
        self.read_only_text_area.setText(error_message)

    def copy_and_close(self):
        """Copies the text from the read-only text area and closes the app."""
        text_to_copy = self.read_only_text_area.toPlainText()
        if text_to_copy:
            pyperclip.copy(text_to_copy)
        self.close()

    def get_chat_modes(self):
        """Returns a dictionary of chat modes."""
        return {
            "Proofread": "You are a grammar proofreading assistant. Output ONLY the corrected text without any additional comments. Maintain the original text structure and writing style. Respond in the same language as the input (e.g., English US, French):",
            "Summarise": "Provide summary in bullet points for the following text:",
            "Explain": "Can you explain the following:",
            "Rewrite": "You are a writing assistant. Rewrite the text provided by the user to improve phrasing. Output ONLY the rewritten text without additional comments. Respond in the same language as the input (e.g., English US, French):",
            "Professional": "You are a writing assistant. Rewrite the text provided by the user to sound more professional. Output ONLY the professional text without additional comments. Respond in the same language as the input (e.g., English US, French):",
            "Friendly": "You are a writing assistant. Rewrite the text provided by the user to be more friendly. Output ONLY the friendly text without additional comments. Respond in the same language as the input (e.g., English US, French):",
            "Concise": "You are a writing assistant. Rewrite the text provided by the user to be slightly more concise in tone, thus making it just a bit shorter. Do not change the text too much or be too reductive. Output ONLY the concise version without additional comments. Respond in the same language as the input (e.g., English US, French):",
            "Fallacy Finder": "I want you to act as a fallacy finder. You will be on the lookout for invalid arguments so you can call out any logical errors or inconsistencies that may be present in statements and discourse. Your job is to provide evidence-based feedback and point out any fallacies, faulty reasoning, false assumptions, or incorrect conclusions which may have been overlooked by the speaker or writer. Text:",
            "Answer It": "You are an intelligent assistant. Help user with the query. Query:",
        }

    def apply_styles(self):
        """Applies QSS to style the application widgets."""
        style_sheet = """
            #centralWidget {
                background-color: #FFFFFF;
                border: 2px solid #005fa3;
                border-radius: 6px;
            }

            QComboBox, QPushButton, QTextEdit {
                font-family: "Fantasque Sans Mono";
                font-size: 18px;
            }

            QComboBox, QPushButton {
                min-height: 28px;
                border-radius: 4px;
            }

            QComboBox {
                background-color: #F9F9F9;
                border: 1px solid #C6C6C6;
                padding: 1px 10px 1px 10px;
            }

            QComboBox::drop-down { border: none;}

            QComboBox:hover {
                background-color: #0078d7;
                color: black;
            }

            QComboBox:selected {
                background-color: #005fa3;
                color: white;
            }

            QPushButton {
                background-color: #F0F0F0;
                border: 1px solid #C6C6C6;
                padding: 0 14px;
            }

            QPushButton:hover { background-color: #E6E6E6; }

            QPushButton#closeButton {
                background-color: #005fa3;
                color: white;
                border: none;
                font-weight: 500;
                padding: 0 22px;
            }

            QPushButton#closeButton:hover { background-color: #007BD6; }

            QTextEdit {
                border: 1px solid #C6C6C6;
                border-radius: 2px;
                padding: 4px;
                background-color: #FFFFFF;
                color: #333333;
            }
        """
        self.setStyleSheet(style_sheet)

    def closeEvent(self, event):
        """Ensure any running thread is stopped cleanly on window close."""
        try:
            if self.api_thread and self.api_thread.isRunning():
                self.api_thread.requestInterruption()
                self.api_thread.quit()
                # Using wait() is crucial here to ensure the thread finishes
                # its cleanup before the main application exits.
                self.api_thread.wait()
        except RuntimeError:
            pass  # Object was already deleted, safe to ignore.
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = AppWindow()
    window.show()
    sys.exit(app.exec())