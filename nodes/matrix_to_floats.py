import maya.api.OpenMaya as om
from maya import mel

def maya_useNewAPI():
    """
    This function is required by Maya to indicate that this plugin uses the Maya Python API 2.0.
    """
    pass

class MatrixToFloats(om.MPxNode):
    """
    A custom Maya node that converts a 4x4 matrix into 16 float values, 
    representing the individual elements of the matrix.
    """
    
    # Node name and ID
    kNodeName = "matrixToFloats"
    kNodeId = om.MTypeId(0x0005C1B8)

    # Input attributes
    aInputMatrix = None  # Attribute for the input matrix

    # Output attributes
    aOutputFloats = None  # Compound attribute to store the 16 float output attributes
    floatAttributes = []  # List to store the individual float attributes

    def __init__(self):
        """
        Constructor for the MatrixToFloats node.
        """
        om.MPxNode.__init__(self)

    def compute(self, plug, data):
        """
        Compute method that is called whenever the node needs to be evaluated.
        
        Args:
            plug (MPlug): The plug that needs to be computed.
            data (MDataBlock): The data block containing the node's data.
        """
        # Check if the plug is the output compound attribute or one of its children
        if plug == self.aOutputFloats or plug.isChild:
            # Get the input matrix value
            inputMatrixHandle = data.inputValue(self.aInputMatrix)
            inputMatrix = inputMatrixHandle.asMatrix()

            # Get the output compound attribute handle
            outputFloatsHandle = data.outputValue(self.aOutputFloats)

            # Set the output float values from the matrix elements
            for i, attr in enumerate(MatrixToFloats.floatAttributes):
                childHandle = outputFloatsHandle.child(attr)
                childHandle.setFloat(inputMatrix[i])
                childHandle.setClean()

            # Mark the plug as clean to indicate it has been computed
            data.setClean(plug)

def creator():
    """
    Creates an instance of the MatrixToFloats node.
    
    Returns:
        MatrixToFloats: A new instance of the MatrixToFloats node.
    """
    return MatrixToFloats()

def initialize():
    """
    Initializes the MatrixToFloats node by creating its attributes and setting up dependencies.
    """
    nAttr = om.MFnNumericAttribute()
    mAttr = om.MFnMatrixAttribute()
    cAttr = om.MFnCompoundAttribute()

    # Input matrix attribute
    MatrixToFloats.aInputMatrix = mAttr.create("inputMatrix", "im", om.MFnMatrixAttribute.kDouble)
    mAttr.storable = True  # Allow the attribute to be stored in the Maya file
    mAttr.writable = True  # Allow the attribute to be written to
    mAttr.readable = False  # The attribute is not directly readable (used for computation)
    mAttr.keyable = True  # Allow the attribute to be keyable in the timeline

    # Output compound attribute
    MatrixToFloats.aOutputFloats = cAttr.create("outputFloats", "of")
    cAttr.storable = False  # Output attributes are not stored in the Maya file
    cAttr.writable = False  # Output attributes are not writable
    cAttr.readable = True  # Output attributes are readable

    # Create and add child float attributes
    for i in range(16):
        attr = nAttr.create(f"outputFloat{i}", f"of{i}", om.MFnNumericData.kFloat, 0.0)
        nAttr.storable = False  # Output attributes are not stored in the Maya file
        nAttr.writable = False  # Output attributes are not writable
        nAttr.readable = True  # Output attributes are readable
        MatrixToFloats.floatAttributes.append(attr)
        cAttr.addChild(attr)

    # Add attributes to the node
    MatrixToFloats.addAttribute(MatrixToFloats.aInputMatrix)
    MatrixToFloats.addAttribute(MatrixToFloats.aOutputFloats)

    # Set attribute dependencies
    MatrixToFloats.attributeAffects(MatrixToFloats.aInputMatrix, MatrixToFloats.aOutputFloats)

def initializePlugin(mobject):
    """
    Initializes the plugin by registering the node.
    
    Args:
        mobject (MObject): The plugin object.
    """
    vendor = "Mauro Lopez"
    version = "1.0"
    plugin = om.MFnPlugin(mobject, vendor, version, "Any")
    try:
        # Register the node with Maya
        plugin.registerNode(MatrixToFloats.kNodeName, MatrixToFloats.kNodeId, creator, initialize)
        # Evaluate the MEL template for the node's attribute editor
        mel.eval(aetemplate)
    except:
        raise RuntimeError("Failed to register nodes")

def uninitializePlugin(mobject):
    """
    Uninitializes the plugin by deregistering the node.
    
    Args:
        mobject (MObject): The plugin object.
    """
    plugin = om.MFnPlugin(mobject)
    try:
        # Deregister the node from Maya
        plugin.deregisterNode(MatrixToFloats.kNodeId)
    except:
        raise RuntimeError("Failed to deregister nodes")

# Define the MEL AE template as a string
aetemplate = """
global proc AEmatrixToFloatsTemplate(string $nodeName) {
    editorTemplate -beginScrollLayout;

    // Input Section
    editorTemplate -beginLayout "Input" -collapse 0;
        editorTemplate -addControl "inputMatrix";
    editorTemplate -endLayout;

    // Output Section
    editorTemplate -beginLayout "Output Floats" -collapse 0;
        editorTemplate -addControl "outputFloats";
    editorTemplate -endLayout;

    // Include default Maya attributes
    AEdependNodeTemplate $nodeName;

    editorTemplate -addExtraControls;
    editorTemplate -endScrollLayout;
}"""