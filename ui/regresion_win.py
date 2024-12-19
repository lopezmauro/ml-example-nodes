import maya.cmds as cmds
import maya.OpenMayaUI as omui
from PySide2 import QtWidgets, QtCore
from shiboken2 import wrapInstance
import numpy as np
from sklearn.linear_model import LinearRegression

# Helper to get Maya's main window
def getMayaMainWindow():
    mainWindowPtr = omui.MQtUtil.mainWindow()
    return wrapInstance(int(mainWindowPtr), QtWidgets.QWidget)

class RegressionUI(QtWidgets.QDialog):
    def __init__(self, parent=getMayaMainWindow()):
        """
        Initialize the RegressionUI dialog.
        Args:
            parent (QWidget): The parent widget.
        """
        super(RegressionUI, self).__init__(parent)
        self.setWindowTitle("Linear Regression & PCA Nodes")
        self.setMinimumSize(500, 400)
        self.initUI()

    def initUI(self):
        """
        Initialize the UI components.
        """
        layout = QtWidgets.QVBoxLayout(self)

        # Inputs Section
        inputLayout = QtWidgets.QVBoxLayout()
        self.inputAttributesList = QtWidgets.QListWidget()
        self.inputAttributesList.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)
        inputLayout.addWidget(QtWidgets.QLabel("Select Input Attributes:"))
        inputLayout.addWidget(self.inputAttributesList)

        buttonLayout = QtWidgets.QHBoxLayout()
        self.addButton = QtWidgets.QPushButton("+")
        self.addButton.clicked.connect(self.addSelectedAttribute)
        self.removeButton = QtWidgets.QPushButton("-")
        self.removeButton.clicked.connect(self.removeSelectedAttribute)
        self.clearButton = QtWidgets.QPushButton("Clear")
        self.clearButton.clicked.connect(self.clearAttributesList)
        buttonLayout.addWidget(self.addButton)
        buttonLayout.addWidget(self.removeButton)
        buttonLayout.addWidget(self.clearButton)
        inputLayout.addLayout(buttonLayout)

        layout.addLayout(inputLayout)

        # Outputs Section
        outputLayout = QtWidgets.QHBoxLayout()
        self.outputAttributeLineEdit = QtWidgets.QLineEdit()
        self.outputAttributeLineEdit.setReadOnly(True)
        self.outputAddButton = QtWidgets.QPushButton("+")
        self.outputAddButton.clicked.connect(self.setOutputAttribute)
        outputLayout.addWidget(QtWidgets.QLabel("Select Output Attribute:"))
        outputLayout.addWidget(self.outputAttributeLineEdit)
        outputLayout.addWidget(self.outputAddButton)
        layout.addLayout(outputLayout)

        # Normalize Options
        self.normalizeInput = QtWidgets.QCheckBox("Normalize Inputs")
        layout.addWidget(self.normalizeInput)

        # Execute Button
        self.executeButton = QtWidgets.QPushButton("Create Nodes")
        self.executeButton.clicked.connect(self.train_and_create_node)
        layout.addWidget(self.executeButton)

    def addSelectedAttribute(self):
        """
        Add the currently selected attribute in Maya to the input attributes list.
        """
        selected_attrs = cmds.channelBox('mainChannelBox', q=True, selectedMainAttributes=True)
        if selected_attrs:
            for attr in selected_attrs:
                self.inputAttributesList.addItem(attr)

    def removeSelectedAttribute(self):
        """
        Remove the currently selected attribute from the input attributes list.
        """
        for item in self.inputAttributesList.selectedItems():
            self.inputAttributesList.takeItem(self.inputAttributesList.row(item))

    def clearAttributesList(self):
        """
        Clear all attributes from the input attributes list.
        """
        self.inputAttributesList.clear()

    def setOutputAttribute(self):
        """
        Set the selected attribute in Maya to the output attribute line edit.
        """
        selected_attrs = cmds.channelBox('mainChannelBox', q=True, selectedMainAttributes=True)
        if selected_attrs:
            self.outputAttributeLineEdit.setText(selected_attrs[0])

    def gatherFrameData(self, attributes):
        """
        Gather attribute values for all frames in the animation.
        Args:
            attributes (list): List of attribute names.
        Returns:
            np.array: Array of gathered attribute values.
        """
        start_frame = cmds.playbackOptions(q=True, min=True)
        end_frame = cmds.playbackOptions(q=True, max=True)

        data = []
        for frame in range(int(start_frame), int(end_frame) + 1):
            cmds.currentTime(frame)
            frame_values = []
            for attr in attributes:
                frame_values.append(cmds.getAttr(attr))
            data.append(frame_values)
        return np.array(data)

    def gatherAndNormalizeData(self, input_attrs, output_attr):
        """
        Gather and normalize input data.
        Args:
            input_attrs (list): List of input attribute names.
            output_attr (list): List of output attribute names.
        Returns:
            tuple: Normalized input data, output data, mean, and standard deviation.
        """
        # Gather data
        X = self.gatherFrameData(input_attrs)
        y = self.gatherFrameData(output_attr).flatten()

        # Normalize inputs if selected
        normalize = self.normalizeInput.isChecked()
        if normalize:
            X_mean, X_std = X.mean(axis=0), X.std(axis=0)
            X = (X - X_mean) / X_std
        else:
            X_mean, X_std = np.zeros(X.shape[1]), np.ones(X.shape[1])

        return X, y, X_mean, X_std

    def trainLinearRegression(self, X, y):
        """
        Train Linear Regression model.
        Args:
            X (np.array): Input data.
            y (np.array): Output data.
        Returns:
            tuple: Weights and bias of the trained model.
        """
        # Linear Regression is a simple machine learning model that assumes a linear relationship
        # between the input variables (features) and the output variable (target).
        # It tries to find the best-fitting straight line through the data points.
        model = LinearRegression()
        model.fit(X, y)
        weights = model.coef_
        bias = model.intercept_
        return weights, bias

    def train_and_create_node(self):
        """
        Train Linear Regression model and create the regression node in Maya.
        """
        # Get selected attributes
        input_attrs = [item.text() for item in self.inputAttributesList.selectedItems()]
        output_attr = self.outputAttributeLineEdit.text()
        if not input_attrs or not output_attr:
            QtWidgets.QMessageBox.warning(self, "Error", "Please select input and output attributes.")
            return

        # Step 1: Data Gathering and Normalization
        X, y, X_mean, X_std = self.gatherAndNormalizeData(input_attrs, [output_attr])

        # Step 2: Train Linear Regression
        weights, bias = self.trainLinearRegression(X, y)

        # Step 3: Node Creation
        node = cmds.createNode("linearRegressionNode", name="linearRegressionNode1")
        cmds.setAttr(f"{node}.features", len(input_attrs), *[0] * len(input_attrs), type="floatArray")
        cmds.setAttr(f"{node}.weights", len(weights), *weights, type="doubleArray")
        cmds.setAttr(f"{node}.bias", bias)
        if self.normalizeInput.isChecked():
            cmds.setAttr(f"{node}.meanVector", len(X_mean), *X_mean, type="doubleArray")
            cmds.setAttr(f"{node}.stdVector", len(X_std), *X_std, type="doubleArray")

        QtWidgets.QMessageBox.information(self, "Success", "Linear Regression Node created and initialized.")
    
# Run UI
def show():
    """
    Show the RegressionUI dialog.
    """
    ui = RegressionUI()
    ui.show()

show()
