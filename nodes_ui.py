# Show the UI
import importlib
from ui import regresion_win
from ui import pca_win
importlib.reload(regresion_win)
importlib.reload(pca_win)

def regrssion_ui():
    ui = regresion_win.RegressionUI()
    ui.show()
    ui.load_needed_nodes()

def pca_ui():
    ui = pca_win.PCAUI()
    ui.show()