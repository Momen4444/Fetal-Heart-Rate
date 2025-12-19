import sys
from PyQt6.QtWidgets import QApplication
from gui import HeartMonitorApp

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = HeartMonitorApp()
    window.show()
    sys.exit(app.exec())
