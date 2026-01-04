import sys
from PyQt5.QtWidgets import QApplication
from PyQt5.QtGui import QIcon
import qdarktheme
from gui.main_window import MainWindow


def main():
    app = QApplication(sys.argv)

    app.setWindowIcon(QIcon("./media/icon.png"))

    qdarktheme.setup_theme("dark")

    win = MainWindow()
    win.show()

    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
