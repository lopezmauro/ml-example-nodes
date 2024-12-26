import pathlib
import random
import numpy as np
from maya import cmds
from maya.api import OpenMaya as om
from maya.api import OpenMayaAnim as oma

_file_path = pathlib.Path(__file__)

def load_node(node_name):
    """
    Load a node in Maya.

    Parameters:
    node_name (str): The name of the node to load.

    """
    file_dir = _file_path.parent.parent
    file_found = list(file_dir.joinpath('nodes').glob(f"{node_name}.py"))
    if not file_found:
        raise ValueError(f"Node {node_name} not found.")
    plugin_path = str(file_found[0].resolve())
    cmds.loadPlugin(plugin_path)
    
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
    std = np.std(inputs, axis=0)
    normalized_inputs = (inputs - mean) / std
    return normalized_inputs, mean, std

def get_blendshape_deltas(blendshape_node):
    """
    Retrieve deltas from a blendshape node using the `inputPointsTarget` and `inputComponentsTarget` attributes.

    :param blendshape_node: The name of the blendshape node.
    :return: A dictionary containing deltas for each target.
    """
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
        group_indices = cmds.getAttr(input_target_group_attr, multiIndices=True)
        if not group_indices:
            continue

        for group_index in group_indices:
            # Access inputPointsTarget
            input_points_attr = f"{input_target_group_attr}[{group_index}].inputPointsTarget"
            if cmds.getAttr(input_points_attr, size=True) > 0:
                points = cmds.getAttr(input_points_attr)
            
            # Access inputComponentsTarget
            input_components_attr = f"{input_target_group_attr}[{group_index}].inputComponentsTarget"
            if cmds.getAttr(input_components_attr, size=True) > 0:
                components = cmds.getAttr(input_components_attr)
            
            # Combine points and components into a delta dictionary
            deltas[(target_index, group_index)] = {
                "points": points,
                "components": components,
            }

    return deltas

