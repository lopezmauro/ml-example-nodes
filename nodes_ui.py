# Show the UI
import imp
from ui import regresion_win
from ui import pca_win
imp.reload(regresion_win)
imp.reload(pca_win)

def regrssion_ui():
    ui = regresion_win.RegressionUI()
    ui.show()

def pca_ui():
    ui = pca_win.PCAUI()
    ui.show()