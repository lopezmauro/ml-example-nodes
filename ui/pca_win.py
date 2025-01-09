import maya.cmds as cmds
import maya.OpenMayaUI as omui
from PySide2 import QtWidgets, QtCore
from shiboken2 import wrapInstance
import numpy as np
from sklearn.decomposition import PCA

# Helper to get Maya's main window
def getMayaMainWindow():
    mainWindowPtr = omui.MQtUtil.mainWindow()
    return wrapInstance(int(mainWindowPtr), QtWidgets.QWidget)

class PCAUI(QtWidgets.QDialog):
    def __init__(self, parent=getMayaMainWindow()):
        """
        Initialize the PCAUI dialog.
        Args:
            parent (QWidget): The parent widget.
        """
        super(PCAUI, self).__init__(parent)
        self.setWindowTitle("PCA Node")
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

        # Number of Components
        self.nComponentsSpinBox = QtWidgets.QSpinBox()
        self.nComponentsSpinBox.setMinimum(1)
        self.nComponentsSpinBox.setMaximum(100)
        layout.addWidget(QtWidgets.QLabel("Number of Components:"))
        layout.addWidget(self.nComponentsSpinBox)

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

    def gatherAndNormalizeData(self, input_attrs):
        """
        Gather and normalize input data.
        Args:
            input_attrs (list): List of input attribute names.
        Returns:
            tuple: Normalized data, mean, and standard deviation.
        """
        # Gather data
        X = self.gatherFrameData(input_attrs)

        # Normalize inputs if selected
        normalize = self.normalizeInput.isChecked()
        if normalize:
            X_mean, X_std = X.mean(axis=0), X.std(axis=0)
            X = (X - X_mean) / X_std
        else:
            X_mean, X_std = np.zeros(X.shape[1]), np.ones(X.shape[1])

        return X, X_mean, X_std

    def trainPCA(self, X, n_components):
        """
        Train PCA model.
        Args:
            X (np.array): Input data.
            n_components (int): Number of components.
        Returns:
            tuple: PCA components and mean.
        """
        # PCA (Principal Component Analysis) is a technique used to reduce the dimensionality of data.
        # It transforms the data into a new coordinate system such that the greatest variance by any projection
        # of the data comes to lie on the first coordinate (called the first principal component),
        # the second greatest variance on the second coordinate, and so on.
        pca = PCA(n_components=n_components)
        pca.fit(X)
        components = pca.components_  # Principal axes in feature space, representing the directions of maximum variance in the data.
        mean = pca.mean_  # Per-feature empirical mean, estimated from the training set.
        return components, mean

    def train_and_create_node(self):
        """
        Train PCA model and create the PCA node in Maya.
        """
        # Get selected attributes
        input_attrs = [item.text() for item in self.inputAttributesList.selectedItems()]
        if not input_attrs:
            QtWidgets.QMessageBox.warning(self, "Error", "Please select input attributes.")
            return

        # Get number of components
        n_components = self.nComponentsSpinBox.value()

        # Step 1: Data Gathering and Normalization
        X, X_mean, X_std = self.gatherAndNormalizeData(input_attrs)

        # Step 2: Train PCA
        components, mean = self.trainPCA(X, n_components)

        # Step 3: Node Creation
        node = cmds.createNode("pcaNode", name="pcaNode1")
        cmds.setAttr(f"{node}.features", len(input_attrs), *[0] * len(input_attrs), type="floatArray")
        cmds.setAttr(f"{node}.components", len(components.flatten()), *components.flatten(), type="doubleArray")
        cmds.setAttr(f"{node}.mean", len(mean), *mean, type="doubleArray")
        cmds.setAttr(f"{node}.nComponents", n_components)
        if self.normalizeInput.isChecked():
            cmds.setAttr(f"{node}.inputMean", len(X_mean), *X_mean, type="doubleArray")
            cmds.setAttr(f"{node}.inputStd", len(X_std), *X_std, type="doubleArray")

        QtWidgets.QMessageBox.information(self, "Success", "PCA Node created and initialized.")
    
# Run UI
def show():
    """
    Show the PCAUI dialog.
    """
    ui = PCAUI()
    ui.show()

show()
