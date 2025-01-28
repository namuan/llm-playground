#!/usr/bin/env -S uv run --quiet --script
# /// script
# dependencies = [
#   "PyQt6",
# ]
# ///
import sys

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QApplication
from PyQt6.QtWidgets import QHBoxLayout
from PyQt6.QtWidgets import QLabel
from PyQt6.QtWidgets import QLineEdit
from PyQt6.QtWidgets import QMainWindow
from PyQt6.QtWidgets import QPushButton
from PyQt6.QtWidgets import QScrollArea
from PyQt6.QtWidgets import QVBoxLayout
from PyQt6.QtWidgets import QWidget
from PyQt6.QtCore import QTimer
from PyQt6.QtCore import QPropertyAnimation, QEasingCurve


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


class Notechat(QMainWindow):
    def __init__(self):
        super().__init__()
        self.chat_layout = None
        self.text_input = None
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

        # Create status widget
        status_widget = QWidget()
        status_layout = QHBoxLayout(status_widget)
        status_layout.setContentsMargins(0, 0, 0, 0)

        local_label = QLabel("Local")
        local_label.setStyleSheet(
            "background-color: white; padding: 5px 10px; border-radius: 15px;"
        )

        active_label = QLabel("Active")
        active_label.setStyleSheet(
            "background-color: #2ECC71; color: white; padding: 5px 10px; border-radius: 15px;"
        )

        status_layout.addWidget(local_label)
        status_layout.addWidget(active_label)

        title_bar_layout.addWidget(title_widget)
        title_bar_layout.addWidget(status_widget, alignment=Qt.AlignmentFlag.AlignRight)

        return title_bar

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

        response_text = QLabel(response_data["response"])
        response_text.setWordWrap(True)
        response_text.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )

        # Note collections
        collections = QWidget()
        collections_layout = QHBoxLayout(collections)
        collections_layout.setSpacing(10)

        for note_data in response_data["collections"]:
            note = QLabel(f"📝 {note_data['title']}  {note_data['date']}")
            note.setStyleSheet("color: #666;")
            collections_layout.addWidget(note)
            collections_layout.addStretch()

        assistant_layout.addWidget(header)
        assistant_layout.addWidget(response_text)
        assistant_layout.addWidget(collections)

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

            # Clear input field
            self.text_input.clear()

            # Schedule scroll to bottom
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


def main():
    app = QApplication(sys.argv)
    window = Notechat()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
