import maya.api.OpenMaya as om
import numpy as np

class RegressionNode(om.MPxNode):
    """
    RegressionNode is a custom Maya node that performs linear regression inference.
    Attributes:
        kNodeName (str): The name of the node.
        kNodeId (om.MTypeId): The unique Autodesk ID of the node.
        featuresAttr (om.MObject): Attribute for input features.
        weightsAttr (om.MObject): Attribute for regression weights.
        biasAttr (om.MObject): Attribute for regression bias.
        predictionAttr (om.MObject): Attribute for output prediction.
        normalizeInputAttr (om.MObject): Attribute to enable input normalization.
        inputMeanAttr (om.MObject): Attribute for input mean values.
        inputStdAttr (om.MObject): Attribute for input standard deviation values.
        denormalizeOutputAttr (om.MObject): Attribute to enable output denormalization.
        outputMeanAttr (om.MObject): Attribute for output mean value.
        outputStdAttr (om.MObject): Attribute for output standard deviation value.
    Methods:
        __init__(): Initializes the RegressionNode instance.
        compute(plug, dataBlock): Computes the output prediction based on input features, weights, and bias.
    """

    kNodeName = "RegressionNode"
    kNodeId = om.MTypeId(0x0007F7FB)  # Replace with a unique ID

    # Attributes
    featuresAttr = None
    weightsAttr = None
    biasAttr = None
    predictionAttr = None

    normalizeInputAttr = None
    inputMeanAttr = None
    inputStdAttr = None

    denormalizeOutputAttr = None
    outputMeanAttr = None
    outputStdAttr = None

    def __init__(self):
        super(RegressionNode, self).__init__()

    def compute(self, plug, dataBlock):
        """
        Compute the output prediction based on input features, weights, and bias.
        Args:
            plug (om.MPlug): The plug to compute.
            dataBlock (om.MDataBlock): The data block containing input and output attributes.
        Returns:
            om.kUnknownParameter if the plug is not the prediction attribute.
        """
        if plug != RegressionNode.predictionAttr:
            return om.kUnknownParameter

        # Retrieve input features
        featuresHandle = dataBlock.inputArrayValue(self.featuresAttr)
        features = []
        while not featuresHandle.isDone():
            features.append(featuresHandle.inputValue().asFloat())
            featuresHandle.next()
        features = np.array(features)

        # Retrieve weights and bias
        weights = np.array(dataBlock.inputValue(self.weightsAttr).asDoubleArray())
        bias = dataBlock.inputValue(self.biasAttr).asDouble()

        # Retrieve normalization settings
        normalizeInput = dataBlock.inputValue(self.normalizeInputAttr).asBool()
        inputMean = np.array(dataBlock.inputValue(self.inputMeanAttr).asDoubleArray())
        inputStd = np.array(dataBlock.inputValue(self.inputStdAttr).asDoubleArray())

        denormalizeOutput = dataBlock.inputValue(self.denormalizeOutputAttr).asBool()
        outputMean = np.array(dataBlock.inputValue(self.outputMeanAttr).asDoubleArray())
        outputStd = np.array(dataBlock.inputValue(self.outputStdAttr).asDoubleArray())

        # Validation
        # Ensure the number of features matches the number of weights
        if len(features) != len(weights):
            om.MGlobal.displayError("Mismatch: The number of features must match the number of weights.")
            return

        # Ensure input mean and std match the number of features if normalization is enabled
        if normalizeInput and (len(inputMean) != len(features) or len(inputStd) != len(features)):
            om.MGlobal.displayError("Mismatch: Input mean and std must match the number of features.")
            return

        # Ensure output mean and std have exactly one value if denormalization is enabled
        if denormalizeOutput and (len(outputMean) != 1 or len(outputStd) != 1):
            om.MGlobal.displayError("Mismatch: Output mean and std must have exactly one value.")
            return

        # Normalize inputs
        # If normalization is enabled, adjust the features using the mean and std
        if normalizeInput:
            features = (features - inputMean) / inputStd

        # Linear regression inference
        # Compute the prediction using the linear regression formula
        prediction = np.dot(features, weights) + bias

        # Denormalize output
        # If denormalization is enabled, adjust the prediction using the output mean and std
        if denormalizeOutput:
            prediction = prediction * outputStd[0] + outputMean[0]

        # Set output prediction
        # Set the computed prediction to the output attribute
        predictionHandle = dataBlock.outputArrayValue(self.predictionAttr)
        builder = predictionHandle.builder()
        outputHandle = builder.addElement(0)
        outputHandle.setFloat(prediction)
        predictionHandle.set(builder)
        dataBlock.setClean(plug)

# Plugin Initialization
def nodeCreator():
    return RegressionNode()

def nodeInitializer():
    # Create attribute functions
    numericAttr = om.MFnNumericAttribute()
    typedAttr = om.MFnTypedAttribute()

    # Features (array of floats)
    RegressionNode.featuresAttr = numericAttr.create("features", "feat", om.MFnNumericData.kFloat)
    numericAttr.array = True
    numericAttr.usesArrayDataBuilder = True
    numericAttr.readable = True
    numericAttr.writable = True
    numericAttr.storable = True
    numericAttr.keyable = True
    RegressionNode.addAttribute(RegressionNode.featuresAttr)

    # Weights (double array)
    RegressionNode.weightsAttr = typedAttr.create("weights", "wght", om.MFnData.kDoubleArray)
    typedAttr.readable = True
    typedAttr.writable = True
    typedAttr.storable = True
    typedAttr.keyable = True
    RegressionNode.addAttribute(RegressionNode.weightsAttr)


    # Bias (double)
    RegressionNode.biasAttr = numericAttr.create("bias", "bias", om.MFnNumericData.kDouble, 0.0)
    numericAttr.readable = True
    numericAttr.writable = True
    numericAttr.storable = True
    numericAttr.keyable = True
    RegressionNode.addAttribute(RegressionNode.biasAttr)

    # Normalize input attributes
    RegressionNode.normalizeInputAttr = numericAttr.create("normalizeInput", "normIn", om.MFnNumericData.kBoolean, False)
    numericAttr.readable = True
    numericAttr.writable = True
    numericAttr.storable = True
    numericAttr.keyable = True
    RegressionNode.addAttribute(RegressionNode.normalizeInputAttr)

    RegressionNode.inputMeanAttr = typedAttr.create("inputMean", "inMean", om.MFnData.kDoubleArray)
    numericAttr.readable = True
    numericAttr.writable = True
    numericAttr.storable = True
    numericAttr.keyable = False
    RegressionNode.addAttribute(RegressionNode.inputMeanAttr)

    RegressionNode.inputStdAttr = typedAttr.create("inputStd", "inStd", om.MFnData.kDoubleArray)
    numericAttr.readable = True
    numericAttr.writable = True
    numericAttr.storable = True
    numericAttr.keyable = False
    RegressionNode.addAttribute(RegressionNode.inputStdAttr)

    # Denormalize output attributes
    RegressionNode.denormalizeOutputAttr = numericAttr.create("denormalizeOutput", "denormOut", om.MFnNumericData.kBoolean, False)
    numericAttr.readable = True
    numericAttr.writable = True
    numericAttr.storable = True
    numericAttr.keyable = True
    RegressionNode.addAttribute(RegressionNode.denormalizeOutputAttr)

    RegressionNode.outputMeanAttr = typedAttr.create("outputMean", "outMean", om.MFnData.kDoubleArray)
    numericAttr.readable = True
    numericAttr.writable = True
    numericAttr.storable = True
    numericAttr.keyable = False
    RegressionNode.addAttribute(RegressionNode.outputMeanAttr)

    RegressionNode.outputStdAttr = typedAttr.create("outputStd", "outStd", om.MFnData.kDoubleArray)
    numericAttr.readable = True
    numericAttr.writable = True
    numericAttr.storable = True
    numericAttr.keyable = False
    RegressionNode.addAttribute(RegressionNode.outputStdAttr)

    # Prediction (array of floats, output)
    RegressionNode.predictionAttr = numericAttr.create("prediction", "pred", om.MFnNumericData.kFloat)
    numericAttr.array = True
    numericAttr.usesArrayDataBuilder = True
    numericAttr.readable = True
    numericAttr.writable = False
    numericAttr.storable = False
    RegressionNode.addAttribute(RegressionNode.predictionAttr)

    # Set attribute dependencies
    RegressionNode.attributeAffects(RegressionNode.featuresAttr, RegressionNode.predictionAttr)
    RegressionNode.attributeAffects(RegressionNode.weightsAttr, RegressionNode.predictionAttr)
    RegressionNode.attributeAffects(RegressionNode.biasAttr, RegressionNode.predictionAttr)
    RegressionNode.attributeAffects(RegressionNode.normalizeInputAttr, RegressionNode.predictionAttr)
    RegressionNode.attributeAffects(RegressionNode.inputMeanAttr, RegressionNode.predictionAttr)
    RegressionNode.attributeAffects(RegressionNode.inputStdAttr, RegressionNode.predictionAttr)
    RegressionNode.attributeAffects(RegressionNode.denormalizeOutputAttr, RegressionNode.predictionAttr)
    RegressionNode.attributeAffects(RegressionNode.outputMeanAttr, RegressionNode.predictionAttr)
    RegressionNode.attributeAffects(RegressionNode.outputStdAttr, RegressionNode.predictionAttr)

# Plugin Registration
def initializePlugin(mobject):
    plugin = om.MFnPlugin(mobject)
    try:
        plugin.registerNode(
            RegressionNode.kNodeName,
            RegressionNode.kNodeId,
            nodeCreator,
            nodeInitializer
        )
    except:
        om.MGlobal.displayError("Failed to register node: " + RegressionNode.kNodeName)

def uninitializePlugin(mobject):
    plugin = om.MFnPlugin(mobject)
    try:
        plugin.deregisterNode(RegressionNode.kNodeId)
    except:
        om.MGlobal.displayError("Failed to deregister node: " + RegressionNode.kNodeName)
