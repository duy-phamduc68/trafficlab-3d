import os
import sys
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QWidget, QHBoxLayout, QVBoxLayout, QTextBrowser


class WelcomeTab(QWidget):
    def __init__(self):
        super().__init__()

        # Root horizontal layout (3-column illusion)
        root = QHBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)

        # Left empty column
        root.addStretch(1)

        # Center content column
        content_container = QWidget()
        content_container.setMinimumWidth(900)
        content_layout = QVBoxLayout(content_container)
        content_layout.setContentsMargins(0, 32, 0, 32)

        viewer = QTextBrowser()
        viewer.setOpenExternalLinks(True)
        viewer.setReadOnly(True)
        viewer.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        viewer.setStyleSheet("""
            QTextBrowser {
                background: transparent;
                color: #e6e6e6;
                border: none;
                font-family:
                    "VT323",
                    "PxPlus IBM VGA9",
                    "Courier New",
                    monospace;
                font-size: 15px;
                line-height: 2.1;
            }

            h1 {
                font-size: 50px;
                margin-top: 12px;
                margin-bottom: 18px;
            }

            h2 {
                font-size: 40px;
                margin-top: 12px;
                margin-bottom: 18px;
            }

            h3 {
                font-size: 30px;
                margin-top: 28px;
                margin-bottom: 14px;
            }

            h4 {
                font-size: 22px;
                margin-top: 20px;
                margin-bottom: 10px;
            }

            p, div {
                margin-top: 24px;
                margin-bottom: 24px;
            }

            a {
                color: #4da3ff;
                text-decoration: none;
            }

            a:hover {
                text-decoration: underline;
            }
        """)

        # Resolve root relative to main.py
        app_root = os.path.dirname(os.path.abspath(sys.argv[0]))
        md_path = os.path.join(app_root, "media", "welcome-trafficlab.md")

        if os.path.exists(md_path):
            with open(md_path, "r", encoding="utf-8") as f:
                viewer.setMarkdown(f.read())
        else:
            viewer.setPlainText(f"Missing markdown:\n{md_path}")

        content_layout.addWidget(viewer)
        root.addWidget(content_container)

        # Right empty column
        root.addStretch(1)
