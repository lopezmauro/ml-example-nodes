import pathlib
import json
import random
import numpy as np
from maya import cmds
from maya.api import OpenMaya as om
from maya.api import OpenMayaAnim as oma
from maya import OpenMayaUI as omui
from shiboken2 import wrapInstance
from PySide2 import QtWidgets

_file_path = pathlib.Path(__file__)

def maya_main_window():
    main_window_ptr = omui.MQtUtil.mainWindow()
    return wrapInstance(int(main_window_ptr), QtWidgets.QWidget)

def load_node(node_name):
    """
    Load a node in Maya.

    Parameters:
    node_name (str): The name of the node to load.

    """
    plugin_name = f"{node_name}.py"
    if cmds.pluginInfo(plugin_name, query=True, loaded=True):
        return
    file_dir = _file_path.parent.parent
    file_found = list(file_dir.joinpath('nodes').glob(plugin_name))
    if not file_found:
        raise ValueError(f"Node {node_name} not found.")
    plugin_path = str(file_found[0].resolve())
    cmds.loadPlugin(plugin_path)

def get_long_attr_name(node, short_attr):
    """Convert short attribute name to long name."""
    try:
        return cmds.attributeQuery(short_attr, node=node, longName=True)
    except:
        return short_attr  # If the query fails, return the short name

def get_attribute_mplug(attribute_string):
    """
    Get the MPlug of an attribute in Maya.

    Parameters:
    attribute_string (str): A string in the format "node_name.attribute".

    Returns:
    MPlug: The MPlug of the attribute.
    """
    node_name, attribute_name = attribute_string.split('.')
    m_obj = om.MSelectionList().add(node_name).getDependNode(0)
    m_fn = om.MFnDependencyNode(m_obj)
    m_plug = m_fn.findPlug(attribute_name, False)
    return m_plug

def get_values_at_frames(attributes, frames):
    """
    Query all values of given attributes in a frame range in Maya.

    Parameters:
    attributes (list): A list of attribute names to query.
    frames (list): A list of frame numbers to query.

    Returns:
    dict: A dictionary with the queried frame numbers and their corresponding attribute values.
    """
    result = dict([(a, []) for a in attributes])
    for frame in frames:
        for attribute in attributes:
            value = cmds.getAttr(attribute, time=frame)
            result[attribute].append(value)
    return result

def get_node_dag_path(node_name):
    """
    Get the DAG path of a node in Maya.

    Parameters:
    node_name (str): The name of the node to query.

    Returns:
    MDagPath: The DAG path of the node.
    """
    selection_list = om.MSelectionList()
    selection_list.add(node_name)
    return selection_list.getDagPath(0)

def get_mesh_fn(object_name):
    """
    Get the MFnMesh object of the selected mesh in Maya.

    Returns:
    MFnMesh: The MFnMesh object of the selected mesh.
    """
    node_dag_path = get_node_dag_path(object_name)
    # Create an MFnMesh object
    fn_mesh = om.MFnMesh(node_dag_path)
    return fn_mesh

def get_original_shape(mesh_name):
    """
    Get the original shape of a mesh.

    Parameters:
    mesh_name (str): The name of the mesh.

    Returns:
    str: The name of the original shape of the mesh.
    """
    # Get the shapes of the mesh
    shapes = cmds.listRelatives(mesh_name, shapes=True, noIntermediate=False)

    # Filter out the original shape
    original_shape = [shape for shape in shapes if cmds.getAttr(shape + ".intermediateObject")]

    return original_shape[0] if original_shape else None

def get_orig_mesh_points(object_name):
    """
    Query all mesh points of a given object in Maya.

    Parameters:
    object_name (str): The name of the object to query.

    Returns:
    list: A list of mesh points.
    """
    # Create an MFnMesh object
    orig_mesh = get_original_shape(object_name)
    fn_mesh = get_mesh_fn(orig_mesh)
    # Get the vertices
    return fn_mesh.getPoints(om.MSpace.kObject)

def get_mesh_points_at_frames(object_name, frames):
    """
    Query all mesh points of a given object in a frame range in Maya.

    Parameters:
    object_name (str): The name of the object to query.
    frames (list): A list of frame numbers to query.

    Returns:
    dict: A dictionary with the queried frame numbers and their corresponding mesh points.
    """
    result = {}
    # Create an MFnMesh object
    fn_mesh = get_mesh_fn(object_name)

    for frame in frames:
        oma.MAnimControl.setCurrentTime(om.MTime(frame))
        # Get the vertices
        result[frame] = fn_mesh.getPoints(om.MSpace.kObject)
    return result

def _create_random_animation_curve(curve_type, start_frame, end_frame, min_value, max_value):
    """
    Create a random animation curve in a frame range in Maya.

    Parameters:
    curve_type (str): The type of the animation curve. It can be 'animCurveTL', 'animCurveTA', 'animCurveTT', or 'animCurveTU'.
    start_frame (int): The start frame of the animation.
    end_frame (int): The end frame of the animation.
    min_value (float): The minimum value for the animation.
    max_value (float): The maximum value for the animation.
    """
    # Create an animCurve node
    curve = cmds.createNode(curve_type)

    # Set random keyframes
    for frame in range(start_frame, end_frame + 1):
        value = random.uniform(min_value, max_value)
        cmds.setKeyframe(curve, time=frame, value=value)

    return curve

def _get_anim_curve_type(attribute_name):
    """
    Get the correct animation curve type based on an attribute name.

    Parameters:
    attribute_name (str): The name of the attribute.

    Returns:
    str: The type of the animation curve. It can be 'animCurveTL', 'animCurveTA', 'animCurveTT', or 'animCurveTU'.
    """
    # Map attribute types to animation curve types
    attribute_type_to_curve_type = {
        'doubleLinear': 'animCurveTL',
        'doubleAngle': 'animCurveTA',
        'time': 'animCurveTT',
        'double': 'animCurveTU',
    }

    # Get the attribute type
    attribute_type = cmds.getAttr(attribute_name, type=True)

    # Get the animation curve type
    curve_type = attribute_type_to_curve_type.get(attribute_type)

    if curve_type is None:
        raise ValueError(f"Unsupported attribute type: {attribute_type}")

    return curve_type

def create_random_animation(attribute_name, start_frame, end_frame, min_value, max_value):
    """
    Create a random animation for a given attribute in a frame range in Maya.

    Parameters:
    - attribute_name (str): The name of the attribute to animate.
    - start_frame (int): The start frame of the animation.
    - end_frame (int): The end frame of the animation.
    - min_value (float): The minimum value for the random animation.
    - max_value (float): The maximum value for the random animation.

    Returns:
    None
    """
    curve_type = _get_anim_curve_type(attribute_name)
    anim_curve = _create_random_animation_curve(curve_type, start_frame, end_frame, min_value, max_value)
    cmds.connectAttr(f"{anim_curve}.output", attribute_name)
    return anim_curve

def get_animation_range(attributes):
    """
    Get the animation range of a list of attributes in Maya.

    Parameters:
    attributes (list): A list of attribute names.

    Returns:
    tuple: A tuple with the start frame and end frame of the animation.
    """
    frames = set()

    for attribute in attributes:
        keyframes = cmds.keyframe(attribute, query=True)
        if keyframes:
            frames.update(keyframes)

    return int(min(frames)), int(max(frames))

def normalize_features(inputs):
    """
    Normalize the input features by subtracting the mean and dividing by the standard deviation.

    Args:
        inputs (torch.Tensor): The input features to be normalized.

    Returns:
        tuple: A tuple containing the normalized inputs, mean, and standard deviation.

    """
    mean = np.mean(inputs, axis=0)
    std = np.std(inputs, axis=0) + np.finfo(float).eps
    normalized_inputs = (inputs - mean) / std
    return normalized_inputs, mean, std

def get_blendshape_data(blendshape_node):
    """
    Retrieve deltas from a blendshape node using the `inputPointsTarget` and `inputComponentsTarget` attributes.

    :param blendshape_node: The name of the blendshape node.
    :return: A dictionary containing deltas for each target.
    """
    def getDigits(s):
        return int(''.join([i for i in s if i.isdigit()]))
    # Dictionary to store deltas for each target
    deltas = {}

    # Check the inputTarget attribute
    input_target_attr = f"{blendshape_node}.inputTarget"
    if not cmds.attributeQuery("inputTarget", node=blendshape_node, exists=True):
        print(f"Blendshape node {blendshape_node} has no inputTarget attribute.")
        return deltas

    # Iterate over target indices
    target_indices = cmds.getAttr(input_target_attr, multiIndices=True)
    if not target_indices:
        print(f"No targets found on blendshape node {blendshape_node}.")
        return deltas

    for target_index in target_indices:
        # Access inputTargetGroup for the target
        input_target_group_attr = f"{input_target_attr}[{target_index}].inputTargetGroup"
        weight_indices = cmds.getAttr(input_target_group_attr, multiIndices=True)
        if not weight_indices:
            continue

        for weight_index in weight_indices:
            # Access inputPointsTarget
            input_points_attr = f"{input_target_group_attr}[{weight_index}].inputPointsTarget"
            if cmds.getAttr(input_points_attr, size=True) > 0:
                points = cmds.getAttr(input_points_attr)
            
            # Access inputComponentsTarget
            input_components_attr = f"{input_target_group_attr}[{weight_index}].inputComponentsTarget"
            if cmds.getAttr(input_components_attr, size=True) > 0:
                components = cmds.getAttr(input_components_attr)
                indices = list()
                for vtx in components:
                    if ':' in vtx:
                        start, end = [getDigits(a) for a in vtx.split(':')]
                        indices.extend(range(start, end + 1))
                    else:
                        indices.append(getDigits(vtx)) 
            # Combine points and components into a delta dictionary
            deltas[weight_index] = {
                "points": points,
                "indices": indices,
            }

    return deltas

def get_blenshape_points(mesh, blendshape_node):
    """
    Retrieve blendshape points for a given mesh and blendshape node.
    """
    fn = get_mesh_fn(mesh)
    bshape_data = get_blendshape_data(blendshape_node)
    shapes_deltas = list()
    for _, delta_data in bshape_data.items():
        deltas = np.zeros((fn.numVertices,3))
        deltas[delta_data['indices']] = np.asarray(delta_data['points'])[:, :3]
        shapes_deltas.append(deltas)
    return np.asarray(shapes_deltas)

def export_regression_node(regression_node, file_path):
    """
    Export the trained regression node's attributes and connections to a JSON file.
    """
    indices = cmds.getAttr(f"{regression_node}.weights", mi=1)
    weights = list()
    for i in indices:
        weights.append(cmds.getAttr(f"{regression_node}.weights[{i}]"))
    matrix_to_floats = set(cmds.listConnections(regression_node, s=1, d=0, type='matrixToFloats') or list())
    matrix_to_floats_data = dict()
    for node in matrix_to_floats:
        src_connections = cmds.listConnections(node, s=1, d=0, p=1, c=1) or list()
        matrix_to_floats_data[node] = list(zip(src_connections[1::2], src_connections[::2]))
        
    floats_to_transforms = set(cmds.listConnections(regression_node, s=0, d=1, type='floatsToTransform') or list())
    floats_to_transforms_data = dict()
    for node in floats_to_transforms:
        dest_connections = cmds.listConnections(node, s=0, d=1, p=1, c=1) or list()
        floats_to_transforms_data[node] = list(zip(dest_connections[::2], dest_connections[1::2]))
    node_data = {"node_name": regression_node,
                "weights": weights,
                "bias": cmds.getAttr(f"{regression_node}.bias"),
                "input_connections": cmds.listConnections(f"{regression_node}.features", source=True, destination=False, plugs=True) or [],
                "output_connections": cmds.listConnections(f"{regression_node}.prediction", source=False, destination=True, plugs=True) or [],
                "matrix_to_floats": matrix_to_floats_data,
                "floats_to_transforms": floats_to_transforms_data
            }
    with open(file_path, "w") as f:
        json.dump(node_data, f, indent=4)
    
def import_regression_node(file_path):
    """
    Import trained regression node data from a JSON file and recreate the node with connections.
    """
    load_node('regression_node')
    load_node('matrix_to_floats')
    load_node('floats_to_transform')
    with open(file_path, "r") as f:
        node_data = json.load(f)
    # create auxilary nodes if they are needed
    matrix_to_floats_data = node_data.get('matrix_to_floats', dict())
    for node, connections in matrix_to_floats_data.items():
        if not cmds.ls(node):
            cmds.createNode("matrixToFloats", name=node)
        for src, dest in connections:
            cmds.connectAttr(src, dest, force=1)
    floats_to_transforms = node_data.get('floats_to_transforms', dict())
    for node, connections in floats_to_transforms.items():
        if not cmds.ls(node):
            cmds.createNode("floatsToTransform", name=node)
        for src, dest in connections:
            cmds.connectAttr(src, dest, force=1)
    node = create_regression_node(node_data["node_name"], 
                           node_data["weights"],
                           node_data["bias"],
                           node_data["input_connections"],
                           node_data["output_connections"])
    return node

def create_regression_node(name, weights, bias, input_attributes, target_attributes, 
                           input_mean=None, input_std=None):
    node = cmds.createNode("RegressionNode", name=name)
    for i, attr in enumerate(input_attributes):
        cmds.connectAttr(attr, f"{node}.features[{i}]", force=True)
    for i, weight in enumerate(weights):
        cmds.setAttr(f"{node}.weights[{i}]", weight, type="doubleArray")
    cmds.setAttr(f"{node}.bias", bias, type="doubleArray")
    if input_mean:
        cmds.setAttr(f"{node}.inputMean", input_mean, type="doubleArray")
    if input_std:    
        cmds.setAttr(f"{node}.inputStd", input_std, type="doubleArray")
    for i, attr in enumerate(target_attributes):
        cmds.connectAttr(f"{node}.prediction[{i}]", attr, force=True)
    return node

def create_matrix_to_floats_node(node):
    
    """
    Create a MatrixToFloats node and connect it to the given node.

    Parameters:
    node (str): The name of the node to connect the MatrixToFloats node to.

    Returns:
    str: The name of the created MatrixToFloats node.
    """
    matrix_to_floats_node = cmds.createNode("matrixToFloats", name=f"{node}_matrix_to_floats")
    cmds.connectAttr(f"{node}.matrix", f"{matrix_to_floats_node}.inputMatrix")
    return matrix_to_floats_node

def create_floats_to_transform_node(node):
    """
    Create a FloatsToTransform node and connect it to the given node.

    Parameters:
    node (str): The name of the node to connect the FloatsToTransform node to.

    Returns:
    str: The name of the created FloatsToTransform node.
    """
    floats_to_transform_node = cmds.createNode("floatsToTransform", name=f"{node}_floats_to_transform")
    cmds.connectAttr(f"{node}.outputFloats", f"{floats_to_transform_node}.inputFloats")
    return floats_to_transform_node