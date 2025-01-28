#!uv run
# /// script
# dependencies = [
#   "timm",
#   "transformers",
#   "einops",
#   "PyAutoGUI",
#   "PyQt5",
#   "PyQtWebEngine",
# ]
# ///
import sys

import pyautogui
import torch
from PIL import ImageGrab
from PyQt5.QtCore import QPoint
from PyQt5.QtCore import QUrl
from PyQt5.QtGui import QColor
from PyQt5.QtGui import QPalette
from PyQt5.QtWebEngineWidgets import QWebEngineView
from PyQt5.QtWidgets import QApplication
from PyQt5.QtWidgets import QFrame
from PyQt5.QtWidgets import QHBoxLayout
from PyQt5.QtWidgets import QLineEdit
from PyQt5.QtWidgets import QMainWindow
from PyQt5.QtWidgets import QVBoxLayout
from PyQt5.QtWidgets import QWidget
from tinyclick_utils import postprocess
from tinyclick_utils import prepare_inputs
from transformers import AutoModelForCausalLM
from transformers import AutoProcessor


class BrowserWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = AutoProcessor.from_pretrained(
            "Samsung/TinyClick", trust_remote_code=True
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            "Samsung/TinyClick", trust_remote_code=True
        ).to(self.device)
        self.setup_ui()
        self.setup_browser()

    def setup_ui(self):
        self.setWindowTitle("Chrome Browser")
        self.setFixedSize(1024, 1084)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        self.main_layout = QVBoxLayout(central_widget)
        self.main_layout.setContentsMargins(0, 0, 0, 0)
        self.main_layout.setSpacing(0)

        # Browser area
        self.browser_frame = QFrame()
        self.browser_frame.setFixedSize(1024, 1024)
        self.main_layout.addWidget(self.browser_frame)

        # Input area
        self.input_frame = QFrame()
        self.input_frame.setFixedHeight(60)
        self.input_layout = QHBoxLayout(self.input_frame)
        self.input_layout.setContentsMargins(10, 10, 10, 10)

        self.instruction_entry = QLineEdit()
        self.instruction_entry.setPlaceholderText("Enter click instructions...")
        self.instruction_entry.returnPressed.connect(self.analyze_and_move)
        self.input_layout.addWidget(self.instruction_entry)

        self.main_layout.addWidget(self.input_frame)

    def setup_browser(self):
        browser_layout = QVBoxLayout(self.browser_frame)
        browser_layout.setContentsMargins(0, 0, 0, 0)

        self.web_view = QWebEngineView()
        browser_layout.addWidget(self.web_view)

        self.web_view.setUrl(QUrl("https://reddit.com"))

    def analyze_and_move(self):
        instruction = self.instruction_entry.text()
        if not instruction:
            return

        # Get window position and capture screenshot
        pos = self.web_view.mapToGlobal(QPoint(0, 0))
        screenshot = ImageGrab.grab(
            bbox=(pos.x(), pos.y(), pos.x() + 1024, pos.y() + 1024)
        )

        # Save and analyze
        temp_path = "temp_screenshot.png"
        screenshot.save(temp_path)

        inputs = prepare_inputs(temp_path, instruction, self.processor)
        img_size = inputs.pop("image_size")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        outputs = self.model.generate(**inputs)
        generated_texts = self.processor.batch_decode(
            outputs, skip_special_tokens=False
        )
        result = postprocess(generated_texts[0], img_size)

        if "click_point" in result:
            click_x, click_y = result["click_point"]
            screen_x = click_x + pos.x()
            screen_y = click_y + pos.y()

            # Smoothly move the mouse to the predicted location over 1 second
            pyautogui.moveTo(
                screen_x, screen_y, duration=1, tween=pyautogui.easeInOutQuad
            )

        self.instruction_entry.clear()


def main():
    app = QApplication(sys.argv)

    # Set application-wide dark theme
    app.setStyle("Fusion")
    palette = QPalette()
    palette.setColor(QPalette.Window, QColor(32, 33, 36))
    palette.setColor(QPalette.WindowText, QColor(255, 255, 255))
    palette.setColor(QPalette.Base, QColor(48, 49, 52))
    palette.setColor(QPalette.AlternateBase, QColor(48, 49, 52))
    palette.setColor(QPalette.ToolTipBase, QColor(255, 255, 255))
    palette.setColor(QPalette.ToolTipText, QColor(255, 255, 255))
    palette.setColor(QPalette.Text, QColor(255, 255, 255))
    palette.setColor(QPalette.Button, QColor(48, 49, 52))
    palette.setColor(QPalette.ButtonText, QColor(255, 255, 255))
    palette.setColor(QPalette.BrightText, QColor(255, 255, 255))
    app.setPalette(palette)

    window = BrowserWindow()
    window.show()

    try:
        sys.exit(app.exec_())
    except SystemExit:
        print("Closing browser window...")


if __name__ == "__main__":
    main()
