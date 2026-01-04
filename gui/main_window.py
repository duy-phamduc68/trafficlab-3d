import os
import sys
from PyQt5.QtCore import Qt, QEvent
from PyQt5.QtWidgets import QMainWindow, QTabWidget, QApplication, QWidget, QVBoxLayout, QLabel, QTextBrowser, QHBoxLayout
from .tabs.tab_welcome import WelcomeTab
from .tabs.tab_calibration import CalibrationTab
from .tabs.tab_inference import InferenceTab
from .tabs.tab_visualization import VisualizationTab
from .tabs.tab_location import LocationTab

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("TrafficLab 3D v1.0")

        # Restore normal window decorations
        self.setWindowFlags(Qt.Window)

        # Create tabs and set as central widget
        tabs = QTabWidget()
        
        tabs.addTab(WelcomeTab(), "Welcome")
        tabs.addTab(LocationTab(), "Location")
        tabs.addTab(CalibrationTab(), "Calibration")
        tabs.addTab(InferenceTab(), "Inference")
        tabs.addTab(VisualizationTab(), "Visualization")
        self.setCentralWidget(tabs)

        # Overlay that appears when the window is NOT maximized
        # Parent to the main window so it survives central widget changes
        self.overlay = QWidget(self)
        # Use a solid background color (no transparency)
        self.overlay.setStyleSheet("background-color: #000000;")
        ol_layout = QVBoxLayout(self.overlay)
        ol_label = QLabel("Window is not maximized. Click anywhere to maximize.")
        ol_label.setStyleSheet("color: white; font-size: 18px;")
        ol_label.setAlignment(Qt.AlignCenter)
        ol_layout.addWidget(ol_label)
        self.overlay.hide()
        # Make overlay clickable
        self.overlay.mousePressEvent = self._overlay_clicked

        # Start maximized
        self.showMaximized()

    def _overlay_clicked(self, event):
        self.showMaximized()
        self.overlay.hide()

    def changeEvent(self, event):
        # Show overlay when window state becomes Normal (not maximized/minimized)
        if event.type() == QEvent.WindowStateChange:
            if self.windowState() & Qt.WindowMaximized:
                self.overlay.hide()
            elif self.windowState() & Qt.WindowMinimized:
                self.overlay.hide()
            else:
                # Normal/restored state -> show overlay covering central widget
                self.overlay.setGeometry(self.centralWidget().geometry())
                self.overlay.show()
                self.overlay.raise_()
        super().changeEvent(event)

    def resizeEvent(self, event):
        # Keep overlay sized to central widget when visible
        if hasattr(self, 'overlay') and self.overlay.isVisible():
            self.overlay.setGeometry(self.centralWidget().geometry())
        super().resizeEvent(event)
