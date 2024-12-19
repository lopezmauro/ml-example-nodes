import maya.api.OpenMaya as om
import numpy as np

class PCANode(om.MPxNode):
    kNodeName = "pcaNode"
    kNodeId = om.MTypeId(0x0007F7FA)  # Replace with your own unique ID

    # Attributes
    featuresAttr = None
    componentsAttr = None
    meanAttr = None
    nComponentsAttr = None
    projectionAttr = None

    normalizeInputAttr = None
    inputMeanAttr = None
    inputStdAttr = None

    denormalizeOutputAttr = None
    outputMeanAttr = None
    outputStdAttr = None

    def __init__(self):
        super(PCANode, self).__init__()

    def compute(self, plug, dataBlock):
        if plug != PCANode.projectionAttr:
            return om.kUnknownParameter

        # Retrieve input features
        featuresHandle = dataBlock.inputArrayValue(self.featuresAttr)
        features = []
        while not featuresHandle.isDone():
            features.append(featuresHandle.inputValue().asFloat())
            featuresHandle.next()
        features = np.array(features)

        # Retrieve PCA-related inputs
        components = np.array(dataBlock.inputValue(self.componentsAttr).asDoubleArray()).reshape(-1, len(features))
        mean = np.array(dataBlock.inputValue(self.meanAttr).asDoubleArray())
        n_components = dataBlock.inputValue(self.nComponentsAttr).asInt()

        # Retrieve normalization settings
        normalizeInput = dataBlock.inputValue(self.normalizeInputAttr).asBool()
        inputMean = np.array(dataBlock.inputValue(self.inputMeanAttr).asDoubleArray())
        inputStd = np.array(dataBlock.inputValue(self.inputStdAttr).asDoubleArray())

        denormalizeOutput = dataBlock.inputValue(self.denormalizeOutputAttr).asBool()
        outputMean = np.array(dataBlock.inputValue(self.outputMeanAttr).asDoubleArray())
        outputStd = np.array(dataBlock.inputValue(self.outputStdAttr).asDoubleArray())

        # Validation
        if len(features) != components.shape[1]:
            om.MGlobal.displayError("Mismatch: The number of features must match the number of columns in components.")
            return
        if len(mean) != len(features):
            om.MGlobal.displayError("Mismatch: The mean vector size must match the number of features.")
            return
        if normalizeInput and (len(inputMean) != len(features) or len(inputStd) != len(features)):
            om.MGlobal.displayError("Mismatch: Input mean and std must match the number of features.")
            return
        if denormalizeOutput and (len(outputMean) != n_components or len(outputStd) != n_components):
            om.MGlobal.displayError("Mismatch: Output mean and std must match the number of components.")
            return

        # Input normalization
        if normalizeInput:
            features = (features - inputMean) / inputStd

        # Use all components if n_components is 0 or greater than available components
        if n_components == 0 or n_components > components.shape[0]:
            n_components = components.shape[0]

        # PCA inference
        centered_features = features - mean
        projection = np.dot(centered_features, components[:n_components].T)

        # Output denormalization
        if denormalizeOutput:
            projection = projection * outputStd + outputMean

        # Set output projection array
        projectionHandle = dataBlock.outputArrayValue(self.projectionAttr)
        builder = projectionHandle.builder()
        for i, value in enumerate(projection):
            outputHandle = builder.addElement(i)
            outputHandle.setFloat(value)
        projectionHandle.set(builder)
        dataBlock.setClean(plug)

# Plugin Initialization
def nodeCreator():
    return PCANode()

def nodeInitializer():
    # Create attribute functions
    numericAttr = om.MFnNumericAttribute()
    typedAttr = om.MFnTypedAttribute()

    # Features (array of floats)
    PCANode.featuresAttr = numericAttr.create("features", "feat", om.MFnNumericData.kFloat)
    numericAttr.array = True
    numericAttr.usesArrayDataBuilder = True
    numericAttr.readable = True
    numericAttr.writable = True
    numericAttr.storable = True
    numericAttr.keyable = True
    PCANode.addAttribute(PCANode.featuresAttr)

    # Components (double array)
    PCANode.componentsAttr = typedAttr.create("components", "comp", om.MFnData.kDoubleArray)
    typedAttr.readable = True
    typedAttr.writable = True
    typedAttr.storable = True
    typedAttr.keyable = True
    PCANode.addAttribute(PCANode.componentsAttr)

    # Mean (double array)
    PCANode.meanAttr = typedAttr.create("mean", "mean", om.MFnData.kDoubleArray)
    typedAttr.readable = True
    typedAttr.writable = True
    typedAttr.storable = True
    typedAttr.keyable = True
    PCANode.addAttribute(PCANode.meanAttr)

    # n_components (integer)
    PCANode.nComponentsAttr = numericAttr.create("nComponents", "nc", om.MFnNumericData.kInt, 0)
    numericAttr.readable = True
    numericAttr.writable = True
    numericAttr.storable = True
    numericAttr.keyable = True
    PCANode.addAttribute(PCANode.nComponentsAttr)

    # Normalize input attributes
    PCANode.normalizeInputAttr = numericAttr.create("normalizeInput", "normIn", om.MFnNumericData.kBoolean, False)
    numericAttr.readable = True
    numericAttr.writable = True
    numericAttr.storable = True
    numericAttr.keyable = True
    PCANode.addAttribute(PCANode.normalizeInputAttr)

    PCANode.inputMeanAttr = typedAttr.create("inputMean", "inMean", om.MFnData.kDoubleArray)
    numericAttr.readable = True
    numericAttr.writable = True
    numericAttr.storable = True
    numericAttr.keyable = False
    PCANode.addAttribute(PCANode.inputMeanAttr)

    PCANode.inputStdAttr = typedAttr.create("inputStd", "inStd", om.MFnData.kDoubleArray)
    numericAttr.readable = True
    numericAttr.writable = True
    numericAttr.storable = True
    numericAttr.keyable = False
    PCANode.addAttribute(PCANode.inputStdAttr)

    # Denormalize output attributes
    PCANode.denormalizeOutputAttr = numericAttr.create("denormalizeOutput", "denormOut", om.MFnNumericData.kBoolean, False)
    numericAttr.readable = True
    numericAttr.writable = True
    numericAttr.storable = True
    numericAttr.keyable = True
    PCANode.addAttribute(PCANode.denormalizeOutputAttr)

    PCANode.outputMeanAttr = typedAttr.create("outputMean", "outMean", om.MFnData.kDoubleArray)
    numericAttr.readable = True
    numericAttr.writable = True
    numericAttr.storable = True
    numericAttr.keyable = False
    PCANode.addAttribute(PCANode.outputMeanAttr)

    PCANode.outputStdAttr = typedAttr.create("outputStd", "outStd", om.MFnData.kDoubleArray)
    numericAttr.readable = True
    numericAttr.writable = True
    numericAttr.storable = True
    numericAttr.keyable = False
    PCANode.addAttribute(PCANode.outputStdAttr)

    # Projection (array of floats, output)
    PCANode.projectionAttr = numericAttr.create("projection", "proj", om.MFnNumericData.kFloat)
    numericAttr.array = True
    numericAttr.usesArrayDataBuilder = True
    numericAttr.readable = True
    numericAttr.writable = False
    numericAttr.storable = False
    PCANode.addAttribute(PCANode.projectionAttr)


    # Set attribute dependencies
    PCANode.attributeAffects(PCANode.featuresAttr, PCANode.projectionAttr)
    PCANode.attributeAffects(PCANode.componentsAttr, PCANode.projectionAttr)
    PCANode.attributeAffects(PCANode.meanAttr, PCANode.projectionAttr)
    PCANode.attributeAffects(PCANode.nComponentsAttr, PCANode.projectionAttr)
    PCANode.attributeAffects(PCANode.normalizeInputAttr, PCANode.projectionAttr)
    PCANode.attributeAffects(PCANode.inputMeanAttr, PCANode.projectionAttr)
    PCANode.attributeAffects(PCANode.inputStdAttr, PCANode.projectionAttr)
    PCANode.attributeAffects(PCANode.denormalizeOutputAttr, PCANode.projectionAttr)
    PCANode.attributeAffects(PCANode.outputMeanAttr, PCANode.projectionAttr)
    PCANode.attributeAffects(PCANode.outputStdAttr, PCANode.projectionAttr)

# Plugin Registration
def initializePlugin(mobject):
    """
    Initialize the plugin when Maya loads it.
    
    Args:
        mobject: The MObject representing the plugin.
    """
    plugin = om.MFnPlugin(mobject)
    try:
        plugin.registerNode(
            PCANode.kNodeName,
            PCANode.kNodeId,
            nodeCreator,
            nodeInitializer
        )
        om.MGlobal.displayInfo(f"Registered node: {PCANode.kNodeName}")
    except Exception as e:
        om.MGlobal.displayError(f"Failed to register node: {PCANode.kNodeName}. Error: {e}")

def uninitializePlugin(mobject):
    plugin = om.MFnPlugin(mobject)
    try:
        plugin.deregisterNode(PCANode.kNodeId)
    except:
        om.MGlobal.displayError("Failed to deregister node: " + PCANode.kNodeName)
