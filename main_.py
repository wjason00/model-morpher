"""
Docstring for main_
"""

import sys
import os 

from PyQt5 import QtWidgets 
from ui.load_window import LoadWindow 

# Prevent OS crashes
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

def main():
    """
    Docstring for main
    """

    app = QtWidgets.QApplication(sys.argv)
    window = LoadWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
