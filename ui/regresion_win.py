import imp
import numpy as np
from functools import partial, wraps
from PySide2 import QtWidgets, QtCore, QtGui
from shiboken2 import wrapInstance
from maya import OpenMayaUI as omui
from maya import cmds as cmds
from . import maya_utils
from sklearn.linear_model import ElasticNet
imp.reload(maya_utils)

def maya_main_window():
    main_window_ptr = omui.MQtUtil.mainWindow()
    return wrapInstance(int(main_window_ptr), QtWidgets.QWidget)

def wait_cursor(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
        try:
            return func(*args, **kwargs)
        finally:
            QtWidgets.QApplication.restoreOverrideCursor()
    return wrapper

class MyDoubleSpinBox(QtWidgets.QDoubleSpinBox):
    def textFromValue(self, value):
        return "{:g}".format(value)

class RegressionUI(QtWidgets.QDialog):
    update_progress = QtCore.Signal(int)
    update_status = QtCore.Signal(str)  # Define the signal

    def __init__(self, parent=maya_main_window()):
        super(RegressionUI, self).__init__(parent)

        self.setWindowTitle("Train Regression Model")
        self.setWindowFlags(self.windowFlags() ^ QtCore.Qt.WindowContextHelpButtonHint)

        self.layout = QtWidgets.QVBoxLayout(self)
        self.splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal, self)

        # Input attributes layout
        self.input_layout = QtWidgets.QVBoxLayout()
        self.input_layout.addWidget(QtWidgets.QLabel("Input Attributes"))
        self.inputs_attributes = QtWidgets.QListWidget()
        self.input_layout.addWidget(self.inputs_attributes)

        self.button_layout = QtWidgets.QHBoxLayout()
        self.add_button = QtWidgets.QPushButton("Add")
        self.remove_button = QtWidgets.QPushButton("Remove")
        self.clear_button = QtWidgets.QPushButton("Clear")
        self.button_layout.addWidget(self.add_button)
        self.button_layout.addWidget(self.remove_button)
        self.button_layout.addWidget(self.clear_button)
        self.input_layout.addLayout(self.button_layout)

        self.input_widget = QtWidgets.QWidget()
        self.input_widget.setLayout(self.input_layout)
        self.splitter.addWidget(self.input_widget)
    
        # Output attributes layout
        self.outputs_attributes_widget = QtWidgets.QWidget()
        self.outputs_attributes = QtWidgets.QListWidget()
        self.output_attr_layout = QtWidgets.QVBoxLayout()
        self.output_attr_layout.addWidget(QtWidgets.QLabel("Target Attributes"))
        self.output_attr_layout.addWidget(self.outputs_attributes)
        self.button_layout2 = QtWidgets.QHBoxLayout()
        self.add_button2 = QtWidgets.QPushButton("Add")
        self.remove_button2 = QtWidgets.QPushButton("Remove")
        self.clear_button2 = QtWidgets.QPushButton("Clear")
        self.button_layout2.addWidget(self.add_button2)
        self.button_layout2.addWidget(self.remove_button2)
        self.button_layout2.addWidget(self.clear_button2)
        self.output_attr_layout.addLayout(self.button_layout2)
        self.outputs_attributes_widget.setLayout(self.output_attr_layout)

        self.output_widget = QtWidgets.QWidget()
        self.output_layout = QtWidgets.QVBoxLayout()
        self.output_widget.setLayout(self.output_layout)
        self.output_layout.addWidget(self.outputs_attributes_widget)
        self.splitter.addWidget(self.output_widget)
        # Separator
        self.separator1 = QtWidgets.QFrame(frameShape=QtWidgets.QFrame.HLine)

        # Animation layout
        self.animation_layout = QtWidgets.QVBoxLayout()

        # Radio buttons for animation type
        self.animation_radio_layout = QtWidgets.QHBoxLayout()
        self.use_current_animation = QtWidgets.QRadioButton("Use current animation")
        self.generate_random_animation = QtWidgets.QRadioButton("Generate random animation")
        # Button group for radio buttons
        self.animation_button_group = QtWidgets.QButtonGroup(self)
        self.animation_button_group.addButton(self.use_current_animation)
        self.animation_button_group.addButton(self.generate_random_animation)
        self.animation_radio_layout.addWidget(self.use_current_animation)
        self.animation_radio_layout.addWidget(self.generate_random_animation)
        self.animation_layout.addLayout(self.animation_radio_layout)
        self.use_current_animation.setChecked(True)

        # Random animation widget
        self.random_animation = QtWidgets.QWidget()
        self.random_animation_layout = QtWidgets.QHBoxLayout(self.random_animation)
        self.amount_of_frames = QtWidgets.QSpinBox()
        self.amount_of_frames.setRange(0, 10000)
        self.amount_of_frames.setValue(1000)
        self.amount_of_frames.setPrefix("Amount of Frames: ")
        self.min_value = QtWidgets.QDoubleSpinBox()
        self.min_value.setRange(-1000, 1000)
        self.min_value.setValue(-50)
        self.min_value.setPrefix("Min: ")
        self.max_value = QtWidgets.QDoubleSpinBox()
        self.max_value.setRange(-1000, 1000)
        self.max_value.setValue(50)
        self.max_value.setPrefix("Max: ")
        #self.adv_config_button = QtWidgets.QPushButton("Adv Config")
        self.random_animation_layout.addWidget(self.amount_of_frames)
        self.random_animation_layout.addWidget(self.min_value)
        self.random_animation_layout.addWidget(self.max_value)
        #self.random_animation_layout.addWidget(self.adv_config_button)
        self.animation_layout.addWidget(self.random_animation)
        self.random_animation.setEnabled(False)

        # Separator
        self.separator2 = QtWidgets.QFrame(frameShape=QtWidgets.QFrame.HLine)

        # Section to select the model hyperparameters
        self.alpha = MyDoubleSpinBox()
        self.alpha.setPrefix("Alpha: ")
        self.alpha.setMinimum(0)
        self.alpha.setDecimals(4)
        self.alpha.setValue(1)

        self.l1_ratio = MyDoubleSpinBox()
        self.l1_ratio.setPrefix("L1 ratio: ")
        self.l1_ratio.setMinimum(0)
        self.l1_ratio.setDecimals(4)
        self.l1_ratio.setValue(0.5)

        self.tol = MyDoubleSpinBox()
        self.tol.setPrefix("tolerance: ")
        self.tol.setMinimum(0)
        self.tol.setDecimals(6)
        self.tol.setValue(0.001)


        # Create the frames
        self.train_parameters_frame = QtWidgets.QGroupBox("Train Parameters")

        # Create the layouts
        self.train_parameters_layout = QtWidgets.QHBoxLayout()
        
        # Add the widgets to the train parameters layout
        self.train_parameters_layout.addWidget(self.alpha)
        self.train_parameters_layout.addWidget(self.l1_ratio)
        self.train_parameters_layout.addWidget(self.tol)

        # Set the layouts for the frames
        self.train_parameters_frame.setLayout(self.train_parameters_layout)

        # Separator
        self.separator3 = QtWidgets.QFrame(frameShape=QtWidgets.QFrame.HLine)

        # Train button
        self.max_iter = QtWidgets.QSpinBox()
        self.max_iter.setPrefix("Max iter: ")
        self.max_iter.setMinimum(0)
        self.max_iter.setMaximum(100000)
        self.max_iter.setValue(1000)
        self.train_button = QtWidgets.QPushButton("Train")

        self.duplicate_checkbox = QtWidgets.QCheckBox("Duplicate output node (for debugging purposes)")
   
        # max_iter and train button layout
        self.max_iter_train_layout = QtWidgets.QHBoxLayout()
        self.max_iter_train_layout.addWidget(self.max_iter)
        self.max_iter_train_layout.addWidget(self.train_button)
        self.max_iter_train_layout.setStretch(1, 1)


        self.status_bar = QtWidgets.QStatusBar(self)
        # Add elements to the main layout in the specified order
        self.layout.addWidget(self.splitter)
        self.layout.addWidget(self.separator1)
        self.layout.addLayout(self.animation_layout)
        self.layout.addWidget(self.separator2)
        self.layout.addWidget(self.train_parameters_frame)
        self.layout.addWidget(self.separator3)
        self.layout.addWidget(self.duplicate_checkbox)
        self.layout.addLayout(self.max_iter_train_layout)
        self.layout.addWidget(self.status_bar)

        # set button sizes
        self.add_button.setFixedSize(50, 15)
        self.add_button.setIcon(QtGui.QIcon(":/addClip.png"))
        self.add_button.setIconSize(QtCore.QSize(15,15))
        self.remove_button.setFixedSize(50, 15)
        self.clear_button.setFixedSize(50, 15)
        self.clear_button.setIcon(QtGui.QIcon(":/smallTrash.png"))
        self.clear_button.setIconSize(QtCore.QSize(15,15))
        self.add_button2.setFixedSize(50, 15)
        self.add_button2.setIcon(QtGui.QIcon(":/addClip.png"))
        self.add_button2.setIconSize(QtCore.QSize(15,15))
        self.remove_button2.setFixedSize(50, 15)
        self.clear_button2.setFixedSize(50, 15)
        self.clear_button2.setIcon(QtGui.QIcon(":/smallTrash.png"))
        self.clear_button2.setIconSize(QtCore.QSize(15,15))
        #self.adv_config_button.setFixedSize(80, 15)
        #self.adv_config_button.setIcon(QtGui.QIcon(":/advancedSettings.png"))
        #self.adv_config_button.setIconSize(QtCore.QSize(15,15))

        # Connect radio button to toggle enable state of random animation widget
        self.generate_random_animation.toggled.connect(self.random_animation.setEnabled)

        # Connect radio buttons to toggle visibility of list widgets
        self.add_button.clicked.connect(partial(self.add_attributes, self.inputs_attributes))
        self.add_button2.clicked.connect(partial(self.add_attributes, self.outputs_attributes))
        self.train_button.clicked.connect(self.train)
        self.update_status.connect(self.status_bar.showMessage)
        self.clear_button.clicked.connect(partial(self.clear_list, self.inputs_attributes))
        self.remove_button.clicked.connect(partial(self.remove_selected_item, self.inputs_attributes))

        self.clear_button2.clicked.connect(partial(self.clear_list, self.outputs_attributes))
        self.remove_button2.clicked.connect(partial(self.remove_selected_item, self.outputs_attributes))


    def add_attributes(self, list_widget):
        slected_nodes = cmds.ls(sl=1)
        selected_attr = cmds.channelBox('mainChannelBox', query=True, selectedMainAttributes=True)
        attributes_list = list()
        for node in slected_nodes:
            attributes_list.extend(cmds.ls([f'{node}.{a}' for a in selected_attr]))
        if attributes_list:
            listed_attributes = [list_widget.item(i).text() for i in range(list_widget.count())]
            missing_attributes = sorted(set(attributes_list) - set(listed_attributes))
            list_widget.addItems(missing_attributes)


    def clear_list(self, list_widget):
        list_widget.clear()

    def remove_selected_item(self, list_widget):
        list_items = list_widget.selectedItems()
        if not list_items: return
        for item in list_items:
            list_widget.takeItem(list_widget.row(item))
            
    def trainLinearRegression(self, X, y, **kwargs):
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
        model = ElasticNet(**kwargs)
        model.fit(X, y)
        weights = model.coef_
        bias = model.intercept_
        return weights, bias
    
    @wait_cursor
    def train(self):
        if not cmds.pluginInfo('regression_node.py', q=True, l=True):
             maya_utils.load_node('regression_node')
        # Get UI values
        input_attributes = [self.inputs_attributes.item(i).text() for i in range(self.inputs_attributes.count())]
        anim_curves = list()
        if self.generate_random_animation.isChecked():
            start_frame = 0
            end_frame = self.amount_of_frames.value()
            min_value = self.min_value.value()
            max_value = self.max_value.value()
            for attr in input_attributes:
                anim_curves.append(maya_utils.create_random_animation(attr, start_frame, end_frame, min_value, max_value))
        frame_range = maya_utils.get_animation_range(input_attributes)
        if not frame_range:
            cmds.error("No keyframes found in the selected attributes")
            return
        target_attributes = [self.outputs_attributes.item(i).text() for i in range(self.outputs_attributes.count())]
        frames = range(frame_range[0], frame_range[1] + 1)
        max_iter = self.max_iter.value()
        alpha = self.alpha.value()
        l1_ratio = self.l1_ratio.value()
        tol = self.tol.value()
        duplicate = self.duplicate_checkbox.isChecked()
        # Step 1: Data Gathering and Normalization
        self.status_bar.showMessage("Gathering scene data...")
        input_anim = maya_utils.get_values_at_frames(input_attributes, frames)
        target_anim = maya_utils.get_values_at_frames(target_attributes, frames)
        X = np.array([input_anim[a] for a in input_attributes]).T
        y = np.array([target_anim[a] for a in target_attributes]).T
        X_mean, X_std = np.zeros(X.shape[1]), np.ones(X.shape[1])
        #normalize = self.normalizeInput.isChecked()
        #if normalize:
        #    X, X_mean, X_std = maya_utils.normalize_features(X)
        self.status_bar.showMessage("Training Linear Regression model...")

        # Step 2: Train Linear Regression
        weights, bias = self.trainLinearRegression(X, y, 
                                                   max_iter=max_iter,
                                                   alpha=alpha, 
                                                   l1_ratio=l1_ratio, 
                                                   tol=tol)

        # Step 3: Node Creation
        node = cmds.createNode("RegressionNode")
        for i, attr in enumerate(input_attributes):
            cmds.connectAttr(attr, f"{node}.features[{i}]", force=True)
        for i, attr in enumerate(weights):
            cmds.setAttr(f"{node}.weights[{i}]", weights[i], type="doubleArray")
        cmds.setAttr(f"{node}.bias", bias, type="doubleArray")
        if duplicate:
            target_nodes = dict([(a, a.split('.')[0]) for a in target_attributes])
            duplicates = dict()
            for n in set(target_nodes.values()):
                dupli = cmds.duplicate(n)[0]
                duplicates[n] = dupli
            for i, attr in enumerate(target_attributes):
                new_attr = attr.replace(target_nodes[attr], duplicates[target_nodes[attr]])
                cmds.connectAttr(f"{node}.prediction[{i}]", new_attr, force=True)
        else:
            for i, attr in enumerate(target_attributes):
                cmds.connectAttr(f"{node}.prediction[{i}]", attr, force=True)
            
        #if self.normalizeInput.isChecked():
        #    cmds.setAttr(f"{node}.inputMean", len(X_mean), *X_mean, type="doubleArray")
        #    cmds.setAttr(f"{node}.inputStd", len(X_std), *X_std, type="doubleArray")
        self.status_bar.showMessage("Linear Regression Node created and initialized.")

    