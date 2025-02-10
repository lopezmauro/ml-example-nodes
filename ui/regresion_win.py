import importlib
import numpy as np
import json
from functools import partial, wraps
from PySide2 import QtWidgets, QtCore, QtGui
from maya import cmds as cmds
from . import maya_utils
from sklearn.linear_model import ElasticNet
importlib.reload(maya_utils)

ROT_MATRIX_INDICES = [0, 1, 2, 4, 5, 6]

def wait_cursor(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
        try:
            return func(*args, **kwargs)
        finally:
            QtWidgets.QApplication.restoreOverrideCursor()
    return wrapper

def show_message(title, message):
    msg_box = QtWidgets.QMessageBox()
    msg_box.setWindowTitle(title)
    msg_box.setText(message)
    msg_box.exec_()

class MyDoubleSpinBox(QtWidgets.QDoubleSpinBox):
    def textFromValue(self, value):
        return "{:g}".format(value)


class RegressionUI(QtWidgets.QDialog):
    update_progress = QtCore.Signal(int)
    update_status = QtCore.Signal(str)

    def __init__(self, parent=maya_utils.maya_main_window()):
        super(RegressionUI, self).__init__(parent)

        self.setWindowTitle("Train Regression Model")
        self.setWindowFlags(self.windowFlags() ^ QtCore.Qt.WindowContextHelpButtonHint)

        self.layout = QtWidgets.QVBoxLayout(self)

        # Menu bar
        self.menu_bar = QtWidgets.QMenuBar(self)
        self.layout.setMenuBar(self.menu_bar)

        # File menu
        style = QtWidgets.QApplication.style()

        self.file_menu = self.menu_bar.addMenu("File")
        self.import_action = QtWidgets.QAction(style.standardIcon(QtWidgets.QStyle.SP_DialogOpenButton), "Import Trained Data", self)
        self.export_action = QtWidgets.QAction(style.standardIcon(QtWidgets.QStyle.SP_DialogSaveButton), "Export Trained Data", self)

        self.file_menu.addAction(self.import_action)
        self.file_menu.addAction(self.export_action)
        self.import_action.triggered.connect(self.import_trained_data)
        self.export_action.triggered.connect(self.export_trained_data)

        # About menu
        self.about_menu = self.menu_bar.addMenu("Help")
        # Create an action for the 'Help' menu with a clickable link
        open_link_action = QtWidgets.QAction(style.standardIcon(QtWidgets.QStyle.SP_MessageBoxQuestion), 'Visit tool wiki page', self)
        self.about_menu.addAction(open_link_action)

        # Connect the action to a function to open the URL
        open_link_action.triggered.connect(self.open_link)

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
        self.splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal, self)
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
        self.random_animation_layout.addWidget(self.amount_of_frames)
        self.random_animation_layout.addWidget(self.min_value)
        self.random_animation_layout.addWidget(self.max_value)
        self.animation_layout.addWidget(self.random_animation)
        self.random_animation.setVisible(False)

        # Separator
        self.separator2 = QtWidgets.QFrame(frameShape=QtWidgets.QFrame.HLine)


        # Animation range widget
        self.animation_range_widget = QtWidgets.QWidget()
        self.animation_range_layout = QtWidgets.QHBoxLayout(self.animation_range_widget)
        #self.use_input_range_checkbox = QtWidgets.QCheckBox("Use input animation range")
        #self.use_input_range_checkbox.setChecked(True)
        self.start_frame = QtWidgets.QSpinBox()
        self.start_frame.setPrefix("Start Frame: ")
        self.end_frame = QtWidgets.QSpinBox()
        self.end_frame.setPrefix("End Frame: ")
        #self.animation_range_layout.addWidget(self.use_input_range_checkbox)
        self.animation_range_layout.addWidget(self.start_frame)
        self.animation_range_layout.addWidget(self.end_frame)
        self.animation_layout.addWidget(self.animation_range_widget)

        # Hyperparameters
        self.alpha = MyDoubleSpinBox()
        self.alpha.setPrefix("Alpha: ")
        self.alpha.setMinimum(0)
        self.alpha.setDecimals(4)
        self.alpha.setValue(.01)
        self.alpha.setToolTip("Regularization strength for ElasticNet.")

        self.l1_ratio = MyDoubleSpinBox()
        self.l1_ratio.setPrefix("L1 ratio: ")
        self.l1_ratio.setMinimum(0)
        self.l1_ratio.setDecimals(4)
        self.l1_ratio.setValue(0.5)
        self.l1_ratio.setToolTip("Mixing parameter for L1/L2 regularization (0 = L2, 1 = L1).")

        self.tol = MyDoubleSpinBox()
        self.tol.setPrefix("Tolerance: ")
        self.tol.setMinimum(0)
        self.tol.setDecimals(6)
        self.tol.setValue(0.001)
        self.tol.setToolTip("Optimization tolerance for stopping criteria.")

        self.train_parameters_frame = QtWidgets.QGroupBox("Train Parameters")
        self.train_parameters_layout = QtWidgets.QHBoxLayout()
        self.train_parameters_layout.addWidget(self.alpha)
        self.train_parameters_layout.addWidget(self.l1_ratio)
        self.train_parameters_layout.addWidget(self.tol)
        self.train_parameters_frame.setLayout(self.train_parameters_layout)

        # Separator
        self.separator3 = QtWidgets.QFrame(frameShape=QtWidgets.QFrame.HLine)

        # Train button
        self.max_iter = QtWidgets.QSpinBox()
        self.max_iter.setPrefix("Max Iter: ")
        self.max_iter.setMinimum(0)
        self.max_iter.setMaximum(100000)
        self.max_iter.setValue(1000)
        self.max_iter.setToolTip("Maximum number of iterations for training.")

        self.train_button = QtWidgets.QPushButton("Train")
        self.duplicate_checkbox = QtWidgets.QCheckBox("Duplicate output node (for debugging purposes)")

        self.max_iter_train_layout = QtWidgets.QHBoxLayout()
        self.max_iter_train_layout.addWidget(self.max_iter)
        self.max_iter_train_layout.addWidget(self.train_button)
        self.max_iter_train_layout.setStretch(1, 1)

        self.status_bar = QtWidgets.QStatusBar(self)

        # Add elements to the main layout
        self.layout.addWidget(self.splitter)
        self.layout.addWidget(self.separator1)
        self.layout.addLayout(self.animation_layout)
        self.layout.addWidget(self.separator2)
        self.layout.addWidget(self.train_parameters_frame)
        self.layout.addWidget(self.separator3)
        self.layout.addWidget(self.duplicate_checkbox)
        self.layout.addLayout(self.max_iter_train_layout)
        self.layout.addWidget(self.status_bar)

        # Set button sizes and tooltips
        self.add_button.setFixedSize(50, 15)
        self.add_button.setIcon(QtGui.QIcon(":/addClip.png"))
        self.add_button.setIconSize(QtCore.QSize(15, 15))
        self.add_button.setToolTip("Add selected attributes to the list.")

        self.remove_button.setFixedSize(50, 15)
        self.remove_button.setToolTip("Remove selected attributes from the list.")

        self.clear_button.setFixedSize(50, 15)
        self.clear_button.setIcon(QtGui.QIcon(":/smallTrash.png"))
        self.clear_button.setIconSize(QtCore.QSize(15, 15))
        self.clear_button.setToolTip("Clear all attributes from the list.")

        self.add_button2.setFixedSize(50, 15)
        self.add_button2.setIcon(QtGui.QIcon(":/addClip.png"))
        self.add_button2.setIconSize(QtCore.QSize(15, 15))
        self.add_button2.setToolTip("Add selected attributes to the list.")

        self.remove_button2.setFixedSize(50, 15)
        self.remove_button2.setToolTip("Remove selected attributes from the list.")

        self.clear_button2.setFixedSize(50, 15)
        self.clear_button2.setIcon(QtGui.QIcon(":/smallTrash.png"))
        self.clear_button2.setIconSize(QtCore.QSize(15, 15))
        self.clear_button2.setToolTip("Clear all attributes from the list.")

        # Connect signals
        self.generate_random_animation.toggled.connect(self.random_animation.setVisible)
        self.use_current_animation.toggled.connect(self.animation_range_widget.setVisible)
        #self.use_input_range_checkbox.toggled.connect(self.toggle_animation_range_controls)
        #self.toggle_animation_range_controls(True)
        self.add_button.clicked.connect(partial(self.add_attributes, self.inputs_attributes, True))
        self.add_button2.clicked.connect(partial(self.add_attributes, self.outputs_attributes, False))
        self.train_button.clicked.connect(self.train)
        self.update_status.connect(self.status_bar.showMessage)
        self.clear_button.clicked.connect(partial(self.clear_list, self.inputs_attributes))
        self.remove_button.clicked.connect(partial(self.remove_selected_item, self.inputs_attributes))
        self.clear_button2.clicked.connect(partial(self.clear_list, self.outputs_attributes))
        self.remove_button2.clicked.connect(partial(self.remove_selected_item, self.outputs_attributes))

    def toggle_animation_range_controls(self, checked):
        self.start_frame.setEnabled(not checked)
        self.end_frame.setEnabled(not checked)

    def open_link(self):
        link = "https://github.com/lopezmauro/ml-example-nodes/wiki/Linear-Regression-Tool-for-Autodesk-Maya"
        QtGui.QDesktopServices.openUrl(QtCore.QUrl(link))


    def add_attributes(self, list_widget, update_range=True):
        selected_nodes = cmds.ls(sl=1)
        selected_attr = cmds.channelBox('mainChannelBox', query=True, selectedMainAttributes=True)
        attributes_list = []
        rot_attributes = list()
        for node in selected_nodes:
            for attr in selected_attr:
                long_attr = maya_utils.get_long_attr_name(node, attr)
                full_attr = f'{node}.{long_attr}'
                if long_attr in ['rotateX', 'rotateY', 'rotateZ']:
                    rot_attributes.append(full_attr)
                else:
                    attributes_list.append(full_attr)

        if rot_attributes:
            # Ask the user if they want the rotation matrix instead
            msg_box = QtWidgets.QMessageBox()
            msg_box.setIcon(QtWidgets.QMessageBox.Question)
            msg_box.setWindowTitle("Rotation Attribute Detected")
            msg_box.setText("You have selected a rotation attribute.\n"
                            "Linear Regression often struggles with Euler rotations due to gimbal lock and non-linearities.\n"
                            "Would you like to use the rotation matrix instead, which may provide better results?")
            msg_box.setStandardButtons(QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No)
            response = msg_box.exec_()

            if response == QtWidgets.QMessageBox.Yes:
                # Replace rotation attributes with rotation matrix attributes
                attributes_list.extend([f"{node}.rotMatrix" for node in selected_nodes])
            else:
                attributes_list.extend(rot_attributes)

        if attributes_list:
            listed_attributes = {list_widget.item(i).text() for i in range(list_widget.count())}
            missing_attributes = sorted(set(attributes_list) - listed_attributes)
            list_widget.addItems(missing_attributes)
        if update_range:
            self.get_animation_range()

    def get_animation_range(self):
        input_attributes = [self.inputs_attributes.item(i).text() for i in range(self.inputs_attributes.count())]
        if not input_attributes:
            return
        rot_matrices = [a for a in input_attributes if a.endswith('.rotMatrix')]
        for rot_matrix in rot_matrices:
            node = rot_matrix.split('.')[0]
            input_attributes.remove(rot_matrix)
            input_attributes.extend([f'{node}.{a}' for a in ['rotateX', 'rotateY', 'rotateZ']])
        frame_range = maya_utils.get_animation_range(input_attributes)
        if not frame_range:
            cmds.error("No keyframes found in the selected attributes.")
            return
        self.start_frame.setValue(frame_range[0])
        self.end_frame.setValue(frame_range[1])


    def clear_list(self, list_widget):
        list_widget.clear()

    def remove_selected_item(self, list_widget):
        list_items = list_widget.selectedItems()
        if not list_items:
            return
        for item in list_items:
            list_widget.takeItem(list_widget.row(item))

    def trainLinearRegression(self, X, y, **kwargs):
        model = ElasticNet(**kwargs)
        model.fit(X, y)
        weights = model.coef_
        bias = model.intercept_
        return weights, bias

    @wait_cursor
    def train(self):
        
        duplicates = dict()
        input_attributes = [self.inputs_attributes.item(i).text() for i in range(self.inputs_attributes.count())]
        target_attributes = [self.outputs_attributes.item(i).text() for i in range(self.outputs_attributes.count())]
        if not input_attributes or not target_attributes:
            cmds.warning("Please select input and target attributes.")
            return
        if self.use_current_animation.isChecked():
            frame_range = (self.start_frame.value(), self.end_frame.value())
        else:
            frame_range = (0, self.amount_of_frames.value())
        if self.duplicate_checkbox.isChecked():
            target_attributes, duplicates = self.duplicate_target_nodes(target_attributes)

        # handle if any attribute is a rotation matrix
        input_attributes = self.handle_input_attributes(input_attributes)
        target_attributes, target_attr_mapping, floats_to_transforms, temp_matrix_to_floats = self.handle_target_attributes(target_attributes)
        print(input_attributes)
        print(target_attributes)
        # get animation data
        frames = range(frame_range[0], frame_range[1] + 1)
        input_anim = maya_utils.get_values_at_frames(input_attributes, frames)
        target_anim = maya_utils.get_values_at_frames(target_attributes, frames)
        # train model and create regression node
        # return
        try:
            X = np.array([input_anim[a] for a in input_attributes]).T
            y = np.array([target_anim[a] for a in target_attributes]).T
            #X_mean, X_std = None, None
            #normalize = self.normalizeInput.isChecked()
            #if normalize:
            X, X_mean, X_std = maya_utils.normalize_features(X)
            self.status_bar.showMessage("Training Linear Regression model...")
            weights, bias = self.trainLinearRegression(X, y,
                                                       max_iter=self.max_iter.value(),
                                                       alpha=self.alpha.value(),
                                                       l1_ratio=self.l1_ratio.value(),
                                                       tol=self.tol.value())
           
            # if there are any rot matrix as target, we need to connect the output of the floatsToTransform node to the target node
            attr_to_connect = self.get_target_attributes_to_connect(target_attributes, target_attr_mapping)
            for node, floats_to_transform_node in floats_to_transforms.items():
                cmds.connectAttr(f"{floats_to_transform_node}.outputRotate", f"{node}.rotate", f=1)
                cmds.connectAttr(f"{floats_to_transform_node}.outputRotateX", f"{node}.rotateX", f=1)
                cmds.connectAttr(f"{floats_to_transform_node}.outputRotateY", f"{node}.rotateY", f=1)
                cmds.connectAttr(f"{floats_to_transform_node}.outputRotateZ", f"{node}.rotateZ", f=1)
            print(attr_to_connect)
            maya_utils.create_regression_node('regressionNode', 
                                weights, 
                                bias, 
                                input_attributes, 
                                attr_to_connect,
                                X_mean.tolist(), X_std.tolist())

            self.status_bar.showMessage("Linear Regression Node created and initialized.")
            #if temp_matrix_to_floats:
            #    cmds.delete(temp_matrix_to_floats)
            show_message("Success", "Linear Regression Node created and initialized.")

        except Exception as e:
            for node in duplicates.values():
                cmds.delete(node)
            cmds.error(f"An error occurred during training: {str(e)}")
            self.status_bar.showMessage("Training failed.")
            show_message("Error", "Training Failed.")

    def handle_input_attributes(self, input_attributes):
        input_rot_matrices = [a for a in input_attributes if a.endswith('.rotMatrix')]
        for rot_matrix in input_rot_matrices:
            node = rot_matrix.split('.')[0]
            input_attributes.remove(rot_matrix)
            matrix_to_floats = maya_utils.create_matrix_to_floats_node(node)
            input_attributes.extend([f'{matrix_to_floats}.outputFloat{i}' for i in ROT_MATRIX_INDICES])
        return input_attributes

    def handle_target_attributes(self, target_attributes):
        target_rot_matrices = [a for a in target_attributes if a.endswith('.rotMatrix')]
        target_attr_mapping = dict()
        floats_to_transforms = dict()
        temp_matrix_to_floats = list()
        for rot_matrix in target_rot_matrices:
            node = rot_matrix.split('.')[0]
            target_attributes.remove(rot_matrix)
            matrix_to_floats = maya_utils.create_matrix_to_floats_node(node)
            temp_matrix_to_floats.append(matrix_to_floats)
            floats_to_transform_node = cmds.createNode("floatsToTransform", name=f"{node}_floats_to_transform")
            for i in ROT_MATRIX_INDICES:
                mtx_to_float = f"{matrix_to_floats}.outputFloat{i}"
                target_attr_mapping[mtx_to_float] = f"{floats_to_transform_node}.inputFloat{i}"
                target_attributes.append(mtx_to_float)
            floats_to_transforms[node] = floats_to_transform_node
        return target_attributes, target_attr_mapping, floats_to_transforms, temp_matrix_to_floats

    def get_target_attributes_to_connect(self, target_attributes, target_attr_mapping):
        attr_to_connect = list()
        for attr in target_attributes:
            if attr in target_attr_mapping:
                attr_to_connect.append(target_attr_mapping[attr])
            else:
                attr_to_connect.append(attr)
        return attr_to_connect
        
    def duplicate_target_nodes(self, target_attributes):
        new_target_attributes = list()
        target_nodes = {a: a.split('.')[0] for a in target_attributes}
        duplicates = dict()
        for node in set(target_nodes.values()):
            duplicates[node] = cmds.duplicate(node)[0]
            src_connections = cmds.listConnections(node, d=False, s=True, p=True, c=True) or list()
            for src, dst in zip(src_connections[1::2], src_connections[0::2]):
                cmds.connectAttr(src, dst.replace(node, duplicates[node]))
        for attr in target_attributes:
            new_attr = attr.replace(target_nodes[attr], duplicates[target_nodes[attr]])
            new_target_attributes.append(new_attr)
        target_attributes = new_target_attributes[:]
        return target_attributes, duplicates
           
    def export_trained_data(self):
        # Open file dialog to exort trained data
        export_iu = RegressionNodeExporterUI()
        export_iu.exec_()

    def import_trained_data(self):
        # Open file dialog to import trained data
        file_path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Import Trained Data", "", "JSON Files (*.json)")
        if file_path:
            created_node = maya_utils.import_regression_node(file_path)
            show_message("Success", f"{created_node} created and initialized.")

    def load_needed_nodes(self):
        maya_utils.load_node('regression_node')
        maya_utils.load_node('matrix_to_floats')
        maya_utils.load_node('floats_to_transform')
        

class RegressionNodeExporterUI(QtWidgets.QDialog):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Regression Node Exporter")
        self.setLayout(QtWidgets.QVBoxLayout())

        self.node_list = QtWidgets.QListWidget()
        self.refresh_nodes()
        self.layout().addWidget(self.node_list)

        file_layout = QtWidgets.QHBoxLayout()
        self.file_path_edit = QtWidgets.QLineEdit()
        self.browse_btn = QtWidgets.QPushButton()
        self.browse_btn.setIcon(QtGui.QIcon(QtWidgets.QApplication.style().standardIcon(QtWidgets.QStyle.SP_DirOpenIcon)))
        self.browse_btn.clicked.connect(self.browse_file)
        file_layout.addWidget(self.file_path_edit)
        file_layout.addWidget(self.browse_btn)
        self.layout().addLayout(file_layout)

        btn_layout = QtWidgets.QHBoxLayout()
        self.export_btn = QtWidgets.QPushButton("Export Selected")
        self.export_btn.clicked.connect(self.export_selected_node)
        self.close_btn = QtWidgets.QPushButton("Close")
        self.close_btn.clicked.connect(self.close)

        btn_layout.addWidget(self.export_btn)
        btn_layout.addWidget(self.close_btn)
        self.layout().addLayout(btn_layout)

    def refresh_nodes(self):
        self.node_list.clear()
        nodes = cmds.ls(type="RegressionNode")
        if nodes:
            self.node_list.addItems(nodes)
        else:
            self.node_list.addItem("No RegressionNodes found")

    def browse_file(self):
        file_path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Export Trained Data", "", "JSON Files (*.json)")
        if file_path:
            self.file_path_edit.setText(file_path)

    def export_selected_node(self):
        selected_item = self.node_list.currentItem()
        if not selected_item or selected_item.text() == "No RegressionNodes found":
            show_message("Warning", "Please select a valid RegressionNode to export.")
            return

        file_path = self.file_path_edit.text()
        if not file_path:
            show_message("Warning", "Please select a file path for export.")
            return

        maya_utils.export_regression_node(selected_item.text(), file_path)
        show_message("Success", f"Exported trained data to {file_path}")
        self.accept()  # Close and delete the UI after exporting