from maya import OpenMaya as om, OpenMayaMPx as ompx
from maya import cmds, mel

import numpy as np

class PCABlendshapeDeformer(ompx.MPxDeformerNode):
    kPluginNodeId = om.MTypeId(0x0007F7FC)  # Replace with a unique ID
    kPluginNodeName = "pcaBlendshape"

    # Attributes
    aShapesWeightsAttr = None
    aPCAComponents = None
    aPCAMean = None
    aPCAWeights = None
    aCumulativeVarianceRatio = None
    aCompresionRatio = None
    aCompresionResult = None

    @staticmethod
    def creator():
        return ompx.asMPxPtr(PCABlendshapeDeformer())


    @staticmethod
    def initialize():
        nAttr = om.MFnNumericAttribute()
        tAttr = om.MFnTypedAttribute()

        # shapes weights (array of floats)
        PCABlendshapeDeformer.aShapesWeightsAttr = nAttr.create("shapeWeights", "sw", om.MFnNumericData.kFloat)
        nAttr.setArray(True)
        nAttr.setReadable(True)
        nAttr.setWritable(True)
        nAttr.setStorable(True)
        nAttr.setKeyable (True)
        PCABlendshapeDeformer.addAttribute(PCABlendshapeDeformer.aShapesWeightsAttr)
        
        # PCA Components
        PCABlendshapeDeformer.aPCAComponents = tAttr.create("pcaComponents", "pcc", om.MFnData.kDoubleArray)
        tAttr.setArray(True)
        PCABlendshapeDeformer.addAttribute(PCABlendshapeDeformer.aPCAComponents)

        # PCA Weights
        PCABlendshapeDeformer.aPCAWeights = tAttr.create("pcaWeights", "pcw", om.MFnData.kDoubleArray)
        tAttr.setArray(True)
        PCABlendshapeDeformer.addAttribute(PCABlendshapeDeformer.aPCAWeights)

        # PCA Mean
        PCABlendshapeDeformer.aPCAMean = tAttr.create("pcaMean", "pcm", om.MFnData.kDoubleArray)
        PCABlendshapeDeformer.addAttribute(PCABlendshapeDeformer.aPCAMean)

        # Explained Variance
        PCABlendshapeDeformer.aCumulativeVarianceRatio = tAttr.create("cumulativeVarianceRatio", "pev", om.MFnData.kDoubleArray)
        PCABlendshapeDeformer.addAttribute(PCABlendshapeDeformer.aCumulativeVarianceRatio)

        # Desired Variance
        PCABlendshapeDeformer.aCompresionRatio = nAttr.create("pcaCompressionRatio", "pcr", om.MFnNumericData.kFloat, 0.05)
        nAttr.setKeyable(True)
        nAttr.setMax(1.0)
        nAttr.setMin(0.0)
        PCABlendshapeDeformer.addAttribute(PCABlendshapeDeformer.aCompresionRatio)

        # Jusnt an output attribute to show how much we compressed the data
        PCABlendshapeDeformer.aCompresionResult = nAttr.create("compresionPercentage", "cpr", om.MFnNumericData.kFloat)
        nAttr.setWritable(False)
        nAttr.setStorable(False)
        PCABlendshapeDeformer.addAttribute(PCABlendshapeDeformer.aCompresionResult)

        # Affects output
        kOutputGeom = ompx.cvar.MPxGeometryFilter_outputGeom
        PCABlendshapeDeformer.attributeAffects(PCABlendshapeDeformer.aShapesWeightsAttr, kOutputGeom)
        PCABlendshapeDeformer.attributeAffects(PCABlendshapeDeformer.aPCAComponents, kOutputGeom)
        PCABlendshapeDeformer.attributeAffects(PCABlendshapeDeformer.aPCAMean, kOutputGeom)
        PCABlendshapeDeformer.attributeAffects(PCABlendshapeDeformer.aCumulativeVarianceRatio, kOutputGeom)
        PCABlendshapeDeformer.attributeAffects(PCABlendshapeDeformer.aCompresionRatio, kOutputGeom)
        PCABlendshapeDeformer.attributeAffects(PCABlendshapeDeformer.aCompresionRatio, PCABlendshapeDeformer.aCompresionResult)

        

    def deform(self, dataBlock, geoIter, matrix, multiIndex):

        # get shapes weights
        shapesWeightsHandle = dataBlock.inputArrayValue(self.aShapesWeightsAttr)
        shapesWeights = []
        for i in range(shapesWeightsHandle.elementCount()):
            shapesWeightsHandle.jumpToElement(i)
            weight = shapesWeightsHandle.inputValue().asFloat()
            shapesWeights.append(weight)
        shapesWeights = np.array(shapesWeights)

        # Read PCA attributes
        pcaComponents_handle = dataBlock.inputArrayValue(self.aPCAComponents)
        pcaComponents = list()
        for i in range(pcaComponents_handle.elementCount()):
            pcaComponents_handle.jumpToElement(i)
            component_handle = pcaComponents_handle.inputValue()
            pcaComponents_data = component_handle.data()
            if pcaComponents_data.isNull():
                return 
            pcaComponents_fn = om.MFnDoubleArrayData(pcaComponents_data)
            pcaComponents.append(pcaComponents_fn.array())
        pcaComponents = np.array(pcaComponents)

        pcaMean_handle = dataBlock.inputValue(self.aPCAMean)
        pcaMean_data = pcaMean_handle.data()
        if pcaMean_data.isNull():
            return 
        pcaMean_fn = om.MFnDoubleArrayData(pcaMean_data)
        pcaMean = np.array(pcaMean_fn.array())


        cumulativeVarianceRatio_handle = dataBlock.inputValue(self.aCumulativeVarianceRatio)
        cumulativeVarianceRatio_data = cumulativeVarianceRatio_handle.data()
        if cumulativeVarianceRatio_data.isNull():
            return 
        cumulativeVarianceRatio_fn = om.MFnDoubleArrayData(cumulativeVarianceRatio_data)
        cumulativeVarianceRatio = np.array(cumulativeVarianceRatio_fn.array())

        compresionRatio = dataBlock.inputValue(self.aCompresionRatio).asFloat()
        variance_keept = 1 - compresionRatio

        # Retrieve weights and bias
        pcaWeights_handle = dataBlock.inputArrayValue(self.aPCAWeights)
        pcaWeights = []
        for i in range(pcaWeights_handle.elementCount()):
            pcaWeights_handle.jumpToElement(i)
            weight_handle = pcaWeights_handle.inputValue()
            weights_data = weight_handle.data()
            if weights_data.isNull():
                continue
            weights_fn = om.MFnDoubleArrayData(weights_data)
            pcaWeights.append(weights_fn.array())
        pcaWeights = np.array(pcaWeights)

        # Use cumulative variance and determine number of components
        nComponents = np.searchsorted(cumulativeVarianceRatio, variance_keept) + 1
        # Select relevant components
        selectedComponents = pcaComponents[:nComponents]
        selectedWeights = pcaWeights[:, :nComponents]

        # Get defomration prediction
        # (weightsArray @ selectedComponents) computes the batch matrix multiplication
        reconstructedData = selectedWeights @ selectedComponents + pcaMean
        recontructed_shapes = reconstructedData.reshape(reconstructedData.shape[0], -1, 3)
        recontructed_shapes *= shapesWeights[:, None, None]
        result_points = np.sum(recontructed_shapes, axis=0)
        # Apply deformed points back to the geometry
        geoIter.reset()
        while not geoIter.isDone():
            pt = geoIter.position();
            result_point = result_points[geoIter.index()] + np.array([pt.x, pt.y, pt.z])
            geoIter.setPosition(om.MPoint(*result_point))
            geoIter.next()

        # set how much we comrpessed the data just for debugging purposes
        compresionResult_handle = dataBlock.outputValue(self.aCompresionResult)
        compressed_size = (selectedComponents.nbytes + selectedWeights.nbytes + pcaMean.nbytes) / (1024 ** 2)
        original_size = reconstructedData.nbytes / (1024 ** 2)
        compresionResult_handle.setFloat((1 - compressed_size / original_size) * 100)
        compresionResult_handle.setClean()
# Plugin registration
def initializePlugin(mobject):
    vendor = "Mauro Lopez"
    version = "1.0"
    pluginFn = ompx.MFnPlugin(mobject, vendor, version)

    try:
        pluginFn.registerNode(
            PCABlendshapeDeformer.kPluginNodeName,
            PCABlendshapeDeformer.kPluginNodeId,
            PCABlendshapeDeformer.creator,
            PCABlendshapeDeformer.initialize,
            ompx.MPxNode.kDeformerNode,
        )
        mel.eval(aetemplate)
    except Exception as e:
        om.MGlobal.displayError(f"Failed to register node: {e}")

def uninitializePlugin(mobject):
    pluginFn = ompx.MFnPlugin(mobject)
    try:
        pluginFn.deregisterNode(PCABlendshapeDeformer.kPluginNodeId)
    except Exception as e:
        om.MGlobal.displayError(f"Failed to deregister node: {e}")

# Define the MEL AE template as a string
aetemplate = """

global proc AEpcaBlendshapeTemplate(string $nodeName)
{
    editorTemplate -beginScrollLayout;

    // Add a section for PCA-related attributes
    editorTemplate -beginLayout "PCA Attributes" -collapse false;
    editorTemplate -addControl "pcaCompressionRatio";
    editorTemplate -addControl "compresionPercentage";
    editorTemplate -addControl "shapeWeights";
    editorTemplate -endLayout;

    // Add other controls (extras)
    editorTemplate -addExtraControls;

    editorTemplate -endScrollLayout;
}
"""
