"""GLCM / GLSZM texture descriptors used as the *radiomics texture* target.

Features are computed separately inside the GM (1), WM (2) and CSF (3) masks
produced by :mod:`gmwmcsf`, giving 24 descriptors per tissue = 72 in total.

Called by :mod:`extract_all_features`; not meant to be run standalone.
"""
import logging

import radiomics

logging.getLogger("radiomics").setLevel(logging.ERROR)
logging.getLogger("radiomics.glcm").setLevel(logging.ERROR)


def textureFeaturesExtractor(img, roi, roiNum):
    roi_name = ["", "GM", "WM", "CSF"]
    textureFeaturesDict = {}

    glcmFeatures = radiomics.glcm.RadiomicsGLCM(img, roi, label=roiNum, binCount=128, verbose=True, interpolator=None, symmetricalGLCM=True)
    glcmFeatures._initCalculation()
    glcmFeatures._calculateMatrix()
    glcmFeatures._calculateCoefficients()

    textureFeaturesDict['GLCM_Autocorrelation_' + roi_name[roiNum]] = glcmFeatures.getAutocorrelationFeatureValue().item()
    textureFeaturesDict['GLCM_ClusterTendency_' + roi_name[roiNum]] = glcmFeatures.getClusterTendencyFeatureValue().item()
    textureFeaturesDict['GLCM_Contrast_' + roi_name[roiNum]] = glcmFeatures.getContrastFeatureValue().item()
    textureFeaturesDict['GLCM_Correlation_' + roi_name[roiNum]] = glcmFeatures.getCorrelationFeatureValue().item()
    textureFeaturesDict['GLCM_DifferenceAverage_' + roi_name[roiNum]] = glcmFeatures.getDifferenceAverageFeatureValue().item()
    textureFeaturesDict['GLCM_DifferenceEntropy_' + roi_name[roiNum]] = glcmFeatures.getDifferenceEntropyFeatureValue().item()
    textureFeaturesDict['GLCM_DifferenceVariance_' + roi_name[roiNum]] = glcmFeatures.getDifferenceVarianceFeatureValue().item()
    textureFeaturesDict['GLCM_JointEnergy_' + roi_name[roiNum]] = glcmFeatures.getJointEnergyFeatureValue().item()
    textureFeaturesDict['GLCM_JointEntropy_' + roi_name[roiNum]] = glcmFeatures.getJointEntropyFeatureValue().item()
    textureFeaturesDict['GLCM_IMC1_' + roi_name[roiNum]] = glcmFeatures.getImc1FeatureValue().item()
    textureFeaturesDict['GLCM_IMC2_' + roi_name[roiNum]] = glcmFeatures.getImc2FeatureValue().item()
    textureFeaturesDict['GLCM_IDM_' + roi_name[roiNum]] = glcmFeatures.getIdmFeatureValue().item()
    textureFeaturesDict['GLCM_MCC_' + roi_name[roiNum]] = glcmFeatures.getMCCFeatureValue().item()
    textureFeaturesDict['GLCM_IDMN_' + roi_name[roiNum]] = glcmFeatures.getIdmnFeatureValue().item()
    textureFeaturesDict['GLCM_InverseDifference_' + roi_name[roiNum]] = glcmFeatures.getIdFeatureValue().item()
    textureFeaturesDict['GLCM_InverseVariance_' + roi_name[roiNum]] = glcmFeatures.getInverseVarianceFeatureValue().item()
    textureFeaturesDict['GLCM_MaximumProbability_' + roi_name[roiNum]] = glcmFeatures.getMaximumProbabilityFeatureValue().item()
    textureFeaturesDict['GLCM_SumAverage_' + roi_name[roiNum]] = glcmFeatures.getSumAverageFeatureValue().item()
    textureFeaturesDict['GLCM_SumEntropy_' + roi_name[roiNum]] = glcmFeatures.getSumEntropyFeatureValue().item()
    textureFeaturesDict['GLCM_SumofSquares_' + roi_name[roiNum]] = glcmFeatures.getSumSquaresFeatureValue().item()

    glszmFeatures = radiomics.glszm.RadiomicsGLSZM(img, roi, label=roiNum, binCount=128, verbose=True, interpolator=None)
    glszmFeatures._initCalculation()
    glszmFeatures._calculateMatrix()
    glszmFeatures._calculateCoefficients()

    textureFeaturesDict['GLSZM_LargeAreaEmphasis_' + roi_name[roiNum]] = glszmFeatures.getLargeAreaEmphasisFeatureValue().item()
    textureFeaturesDict['GLSZM_GLNN_' + roi_name[roiNum]] = glszmFeatures.getGrayLevelNonUniformityNormalizedFeatureValue().item()
    textureFeaturesDict['GLSZM_SZNN_' + roi_name[roiNum]] = glszmFeatures.getSizeZoneNonUniformityNormalizedFeatureValue().item()
    textureFeaturesDict['GLSZM_ZoneEntropy_' + roi_name[roiNum]] = glszmFeatures.getZoneEntropyFeatureValue().item()

    return textureFeaturesDict
