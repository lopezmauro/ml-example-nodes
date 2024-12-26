# Show the UI
import imp
from ui import regresion_win
imp.reload(regresion_win)
def show():
    ui = regresion_win.RegressionUI()
    ui.show()