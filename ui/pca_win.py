import imp
import maya.cmds as cmds
from PySide2 import QtWidgets
import numpy as np
from sklearn.decomposition import PCA
from ui import maya_utils
imp.reload(maya_utils)

class PCAUI(QtWidgets.QDialog):
    def __init__(self, parent=maya_utils.maya_main_window()):
        super(PCAUI, self).__init__(parent)
        self.setWindowTitle("PCA Blendshape UI")
        self.setFixedSize(450, 200)

        self.init_ui()

    def init_ui(self):
        # Main layout
        main_layout = QtWidgets.QVBoxLayout(self)

        # Source Blendshape Layout
        source_layout = QtWidgets.QHBoxLayout()
        source_label = QtWidgets.QLabel("Source Blendshape:")
        self.source_text_field = QtWidgets.QLineEdit()
        self.source_button = QtWidgets.QPushButton("Get Blendshape Node")
        self.source_button.clicked.connect(self.populate_source)

        source_layout.addWidget(source_label)
        source_layout.addWidget(self.source_text_field)
        source_layout.addWidget(self.source_button)

        # Target Mesh Layout
        target_layout = QtWidgets.QHBoxLayout()
        target_label = QtWidgets.QLabel("Target Mesh:")
        self.target_text_field = QtWidgets.QLineEdit()
        self.target_button = QtWidgets.QPushButton("Get Target Mesh")
        self.target_button.clicked.connect(self.populate_target)

        target_layout.addWidget(target_label)
        target_layout.addWidget(self.target_text_field)
        target_layout.addWidget(self.target_button)

        # Create PCA Blendshape Button
        self.create_button = QtWidgets.QPushButton("Create PCA Blendshape")
        self.create_button.clicked.connect(self.do_it)

        # Add layouts to the main layout
        main_layout.addLayout(source_layout)
        main_layout.addLayout(target_layout)
        main_layout.addWidget(self.create_button)

    def populate_source(self):
        """Populate the source blendshape field."""
        selected = cmds.ls(selection=True)
        if not selected:
            cmds.warning("No object selected.")
            return

        node = selected[0]
        blendshape_node = None
        if cmds.nodeType(node) == "blendShape":
            blendshape_node = node
        else:
            bs_nodes = [a for a in cmds.listHistory(node, pdo=1) if cmds.nodeType(a) == "blendShape"]
            if not bs_nodes:
                cmds.warning(f"No blendshape node found for {node}.")
                return
            blendshape_node = bs_nodes[0]
        self.source_text_field.setText(blendshape_node)


    def populate_target(self):
        """Populate the target mesh field."""
        selected = cmds.ls(selection=True, type="transform")
        if not selected:
            cmds.warning("No object selected.")
            return

        shapes = cmds.listRelatives(selected[0], shapes=True, fullPath=True) or []
        if shapes and cmds.nodeType(shapes[0]) == "mesh":
            self.target_text_field.setText(selected[0])
        else:
            cmds.warning("The selected object is not a valid mesh.")

    def get_blendshape_data(self, mesh, blendshape_node):
        """
        Retrieves blendshape data from the source mesh and blendshape node.

        Args:
            source_mesh (str): The name of the source mesh.
            source_blendshape (str): The name of the source blendshape node.

        Returns:
            tuple: A tuple containing shapes deltas and aliases dictionary.
        """
        shapes_deltas = maya_utils.get_blenshape_points(mesh, blendshape_node)
        aliases = cmds.aliasAttr(blendshape_node, q=1)
        aliases_dict = dict(zip(aliases[1::2], aliases[::2]))
        return shapes_deltas, aliases_dict

    def get_pca_data(self, shapes_deltas):
        """
        This method takes a list of shape deltas, flattens the data, and applies Principal Component Analysis (PCA) 
        to extract the principal components, the mean of the data, the weights of the data in the PCA space, 
        and the cumulative variance ratio explained by the components.
            
        Args:
            shapes_deltas (list): A list of shape deltas. Each element in the list represents the delta of a shape.

        Returns:
            tuple: A tuple containing the following elements:
                - pca.mean_ (numpy.ndarray): The mean of the shape deltas.
                - pca.components_ (numpy.ndarray): The principal components of the shape deltas.
                - pcaWeights (numpy.ndarray): The weights of the shape deltas in the PCA space.
                - cumulative_variance_ratio (numpy.ndarray): The cumulative variance ratio explained by the principal components.
        """

        ### get PCA data
        # Flatten the data
        delta_data_flat = shapes_deltas.reshape(shapes_deltas.shape[0], -1)
        # Fit PCA
        pca = PCA()
        pca.fit(delta_data_flat)
        # Transform data to get the PCA weights
        pcaWeights = pca.transform(delta_data_flat)

        # Get explained variance in order to compress the data in the node
        explained_variance = pca.explained_variance_
        total_variance = np.sum(explained_variance)
        cumulative_variance_ratio = np.cumsum(explained_variance) / total_variance
        return pca.mean_, pca.components_, pcaWeights, cumulative_variance_ratio
    
    def create_pca_blendshape(self, mesh, pca_mean, pca_components, pca_weights, cumulative_variance_ratio, aliases_dict):
        """
        Creates a PCA blendshape node on the target mesh using the provided PCA data.

        Args:
            target_mesh (str): The name of the target mesh.
            pca_mean (np.array): The mean of the PCA.
            pca_components (np.array): The components of the PCA.
            pca_weights (np.array): The weights of the PCA.
            cumulative_variance_ratio (np.array): The cumulative variance ratio of the PCA.
            aliases_dict (dict): A dictionary of aliases.

        Returns:
            str: The name of the created PCA blendshape node.
        """
        node = cmds.deformer(mesh, type='pcaBlendshape')[0]
        # set main PCA data
        cmds.setAttr(f'{node}.pcaMean', pca_mean, type="doubleArray")
        for i, each in enumerate(pca_components):
            cmds.setAttr(f'{node}.pcaComponents[{i}]', each, type="doubleArray")    
        # get the cached weight to recontruct the shapes
        for i, each in enumerate(pca_weights):
            cmds.setAttr(f'{node}.pcaWeights[{i}]', each, type="doubleArray")    
        # set the explained variance in order to compress on the fly
        cmds.setAttr(f'{node}.cumulativeVarianceRatio', cumulative_variance_ratio, type="doubleArray")

        # create shape weights and rename them with the blendshape weights aliases
        for i in range(pca_weights.shape[0]):
            cmds.setAttr(f'{node}.shapeWeights[{i}]', 0)
            cmds.aliasAttr(aliases_dict[f'weight[{i}]'], f'{node}.shapeWeights[{i}]')
        return node
    
    def do_it(self):
        """
        This method performs a series of checks and operations to create a PCA blendshape node.
        It verifies the target mesh, source mesh, and their vertex counts, then loads the PCA blendshape node,
        retrieves blendshape data, performs PCA, and creates the PCA blendshape node.

        Returns:
            None
        """
        source_blendshape = self.source_text_field.text()
        target_mesh = self.target_text_field.text()

        if not source_blendshape:
            cmds.warning("Source blendshape node is not specified.")
            return
        if not target_mesh:
            cmds.warning("Target mesh is not specified.")
            return

        # Verify vertex count
        source_mesh = cmds.listConnections(source_blendshape, type="mesh") or []
        if not source_mesh:
            cmds.warning("Cannot find source mesh for the blendshape node.")
            return

        source_vertex_count = cmds.polyEvaluate(source_mesh[0], vertex=True)
        target_vertex_count = cmds.polyEvaluate(target_mesh, vertex=True)

        if source_vertex_count != target_vertex_count:
            cmds.warning("Source and target meshes do not have the same number of vertices.")
            return
        maya_utils.load_node('pca_blendshape')
        shapes_deltas, aliases_dict = self.get_blendshape_data(source_mesh[0], source_blendshape)
        pca_mean, pca_components, pca_weights, cumulative_variance_ratio = self.get_pca_data(shapes_deltas)
        # Create the deformer node
        node = self.create_pca_blendshape(target_mesh, pca_mean, pca_components, pca_weights, cumulative_variance_ratio, aliases_dict)
        cmds.select(node)
        QtWidgets.QMessageBox.information(self, "Success", f"PCA Node {node} created and initialized.")
 
    
