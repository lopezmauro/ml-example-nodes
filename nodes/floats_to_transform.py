from maya import mel
import maya.api.OpenMaya as om
import math

def maya_useNewAPI():
    """
    This function is required by Maya to indicate that this plugin uses the Maya Python API 2.0.
    """
    pass

class FloatsToTransform(om.MPxNode):
    """
    A custom Maya node that converts a set of 16 float values (representing a 4x4 matrix) 
    into a transform node's translation, rotation, and scale attributes.
    """
    
    # Node name and ID
    kNodeName = "floatsToTransform"
    kNodeId = om.MTypeId(0x0003A7F2)

    # Input attributes
    aInputFloats = None  # Compound attribute to store the 16 float input attributes
    floatAttributes = []  # List to store the individual float attributes
    aOrthoNormalize = None  # Attribute to orthogonalize the matrix

    # Output attributes
    aOutputTranslate = None  # Output attribute for translation
    aOutputRotateX = None  # Output attribute for rotation around X-axis
    aOutputRotateY = None  # Output attribute for rotation around Y-axis
    aOutputRotateZ = None  # Output attribute for rotation around Z-axis
    aOutputScale = None  # Output attribute for scale

    def __init__(self):
        """
        Constructor for the FloatsToTransform node.
        """
        om.MPxNode.__init__(self)

    def compute(self, plug, data):
        """
        Compute method that is called whenever the node needs to be evaluated.
        
        Args:
            plug (MPlug): The plug that needs to be computed.
            data (MDataBlock): The data block containing the node's data.
        """
        # Check if the plug is one of the output attributes
        compute = False
        for attr in [FloatsToTransform.aOutputTranslate, FloatsToTransform.aOutputRotate, 
                     FloatsToTransform.aOutputScale]:
            if plug == attr:
                compute = True
                break
        if not compute:
            return 

        # Get input float values from the compound attribute
        inputFloatsHandle = data.inputValue(self.aInputFloats)
        inputFloats = []
        for i, attr in enumerate(FloatsToTransform.floatAttributes):
            childHandle = inputFloatsHandle.child(attr)
            inputFloats.append(childHandle.asFloat())

        # Create matrix from input floats
        matrix = om.MMatrix(inputFloats)
       
        # Orthogonalize if checked
        orthoNormalize = data.inputValue(self.aOrthoNormalize).asBool()
        if orthoNormalize:
            matrix = self.orthoNormalizeMatrix(matrix)


        # Decompose matrix into translation, rotation, and scale
        transformMatrix = om.MTransformationMatrix(matrix)
        translation = transformMatrix.translation(om.MSpace.kWorld)
        rotation = transformMatrix.rotation()  # Returns MEulerRotation in radians
        scale = transformMatrix.scale(om.MSpace.kWorld)

        # Set output values
        outputTranslateHandle = data.outputValue(self.aOutputTranslate)
        outputTranslateHandle.set3Float(translation.x, translation.y, translation.z)
        outputTranslateHandle.setClean()

        # Set rotation values (in degrees)
        outputRotateXHandle = data.outputValue(self.aOutputRotateX)
        outputRotateXHandle.setMAngle(om.MAngle(math.degrees(rotation.x), om.MAngle.kDegrees))
        outputRotateXHandle.setClean()

        outputRotateYHandle = data.outputValue(self.aOutputRotateY)
        outputRotateYHandle.setMAngle(om.MAngle(math.degrees(rotation.y), om.MAngle.kDegrees))
        outputRotateYHandle.setClean()

        outputRotateZHandle = data.outputValue(self.aOutputRotateZ)
        outputRotateZHandle.setMAngle(om.MAngle(math.degrees(rotation.z), om.MAngle.kDegrees))
        outputRotateZHandle.setClean()

        # Set scale values
        outputScaleHandle = data.outputValue(self.aOutputScale)
        outputScaleHandle.set3Float(scale[0], scale[1], scale[2])
        outputScaleHandle.setClean()

        data.setClean(plug)

    def orthoNormalizeMatrix(self, matrix):
        """
        Orthogonalizes the vectors of the given matrix.
        
        Args:
            matrix (MMatrix): The matrix to orthogonalize.
        
        Returns:
            MMatrix: The orthogonalized matrix.
        """
        xAxis = om.MVector(matrix[0], matrix[1], matrix[2]).normal()
        yAxis = om.MVector(matrix[4], matrix[5], matrix[6]).normal()
        zAxis = (xAxis ^ yAxis).normal() 
        yAxis = (zAxis ^ xAxis).normal()

        # Create a new orthogonalized matrix
        orthogonalMatrix = matrix
        orthogonalMatrix[0] = xAxis.x
        orthogonalMatrix[1] = xAxis.y
        orthogonalMatrix[2] = xAxis.z
        orthogonalMatrix[4] = yAxis.x
        orthogonalMatrix[5] = yAxis.y
        orthogonalMatrix[6] = yAxis.z
        orthogonalMatrix[8] = zAxis.x
        orthogonalMatrix[9] = zAxis.y
        orthogonalMatrix[10] = zAxis.z

        return orthogonalMatrix

def creator():
    """
    Creates an instance of the FloatsToTransform node.
    
    Returns:
        FloatsToTransform: A new instance of the FloatsToTransform node.
    """
    return FloatsToTransform()

def initialize():
    """
    Initializes the FloatsToTransform node by creating its attributes and setting up dependencies.
    """
    nAttr = om.MFnNumericAttribute()
    cAttr = om.MFnCompoundAttribute()

    # Input compound attribute
    FloatsToTransform.aInputFloats = cAttr.create("inputFloats", "if")
    cAttr.storable = True
    cAttr.writable = True
    cAttr.readable = False

    # Create and add child float attributes
    default_values = [1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1]
    for i, value in enumerate(default_values):
        attr = nAttr.create(f"inputFloat{i}", f"if{i}", om.MFnNumericData.kFloat, value)
        FloatsToTransform.floatAttributes.append(attr)
        cAttr.addChild(attr)

    # Orthogonalize checkbox
    nAttr = om.MFnNumericAttribute()
    FloatsToTransform.aOrthoNormalize = nAttr.create("orthoNormalize", "on", om.MFnNumericData.kBoolean, True)
    nAttr.storable = True
    nAttr.writable = True
    nAttr.readable = True

    # Output translate attribute
    nAttr = om.MFnNumericAttribute()
    FloatsToTransform.aOutputTranslate = nAttr.createPoint("outputTranslate", "ot")
    nAttr.storable = False
    nAttr.writable = False
    nAttr.readable = True

    # Output rotate attributes (as angles)
    cmpAttr = om.MFnCompoundAttribute()
    FloatsToTransform.aOutputRotate = cmpAttr.create("outputRotate", "or")
    uAttr = om.MFnUnitAttribute()
    FloatsToTransform.aOutputRotateX = uAttr.create("outputRotateX", "orx", om.MFnUnitAttribute.kAngle, 0.0)
    uAttr.storable = False
    uAttr.writable = False
    uAttr.readable = True
    cmpAttr.addChild(FloatsToTransform.aOutputRotateX)

    FloatsToTransform.aOutputRotateY = uAttr.create("outputRotateY", "ory", om.MFnUnitAttribute.kAngle, 0.0)
    uAttr.storable = False
    uAttr.writable = False
    uAttr.readable = True
    cmpAttr.addChild(FloatsToTransform.aOutputRotateY)

    FloatsToTransform.aOutputRotateZ = uAttr.create("outputRotateZ", "orz", om.MFnUnitAttribute.kAngle, 0.0)
    uAttr.storable = False
    uAttr.writable = False
    uAttr.readable = True
    cmpAttr.addChild(FloatsToTransform.aOutputRotateZ)

    # Output scale attribute
    nAttr = om.MFnNumericAttribute()
    FloatsToTransform.aOutputScale = nAttr.createPoint("outputScale", "os")
    nAttr.storable = False
    nAttr.writable = False
    nAttr.readable = True

    # Add attributes
    FloatsToTransform.addAttribute(FloatsToTransform.aInputFloats)
    FloatsToTransform.addAttribute(FloatsToTransform.aOrthoNormalize)
    FloatsToTransform.addAttribute(FloatsToTransform.aOutputTranslate)
    FloatsToTransform.addAttribute(FloatsToTransform.aOutputRotate)
    FloatsToTransform.addAttribute(FloatsToTransform.aOutputScale)

    # Set attribute dependencies
    FloatsToTransform.attributeAffects(FloatsToTransform.aInputFloats, FloatsToTransform.aOutputTranslate)
    FloatsToTransform.attributeAffects(FloatsToTransform.aInputFloats, FloatsToTransform.aOutputRotate)
    FloatsToTransform.attributeAffects(FloatsToTransform.aInputFloats, FloatsToTransform.aOutputRotateY)
    FloatsToTransform.attributeAffects(FloatsToTransform.aInputFloats, FloatsToTransform.aOutputRotateZ)
    FloatsToTransform.attributeAffects(FloatsToTransform.aInputFloats, FloatsToTransform.aOutputScale)
    FloatsToTransform.attributeAffects(FloatsToTransform.aOrthoNormalize, FloatsToTransform.aOutputTranslate)
    FloatsToTransform.attributeAffects(FloatsToTransform.aOrthoNormalize, FloatsToTransform.aOutputRotate)
    FloatsToTransform.attributeAffects(FloatsToTransform.aOrthoNormalize, FloatsToTransform.aOutputRotateY)
    FloatsToTransform.attributeAffects(FloatsToTransform.aOrthoNormalize, FloatsToTransform.aOutputRotateZ)
    FloatsToTransform.attributeAffects(FloatsToTransform.aOrthoNormalize, FloatsToTransform.aOutputScale)

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
        plugin.registerNode(FloatsToTransform.kNodeName, FloatsToTransform.kNodeId, creator, initialize)
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
        plugin.deregisterNode(FloatsToTransform.kNodeId)
    except:
        raise RuntimeError("Failed to deregister nodes")

# Define the MEL AE template as a string
aetemplate = """
global proc AEfloatsToTransformTemplate(string $nodeName) {
    editorTemplate -beginScrollLayout;

    // Input Section
    editorTemplate -beginLayout "Input Floats" -collapse 1;
        editorTemplate -addControl "inputFloats";
    editorTemplate -endLayout;

    // Options Section
    editorTemplate -beginLayout "Options" -collapse 0;
        editorTemplate -addControl "orthoNormalize";
    editorTemplate -endLayout;

    // Output Section
    editorTemplate -beginLayout "Output" -collapse 0;
        editorTemplate -addControl "outputTranslate";
        editorTemplate -addControl "outputRotate";
        editorTemplate -addControl "outputScale";
    editorTemplate -endLayout;

    // Include default Maya attributes
    AEdependNodeTemplate $nodeName;

    editorTemplate -addExtraControls;
    editorTemplate -endScrollLayout;
}"""