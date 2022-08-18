# -*- coding: utf-8 -*-

import json
import os

#Load configuration

config_path = os.path.split(__file__)[0]
config_file_path = os.path.join(config_path, "config.json")
print("Config path: ", config_file_path)

with open(config_file_path) as config_file:
    configdata = json.load(config_file)

#CHANGE STARTS HERE
# If .env file is present, the values in .env file should override the config.json values.
# This must be done only for the overlapping keys
envdata = {}
if os.path.exists(".env"):
    with open(".env")  as env_file:
        envdata = json.load(env_file)

config_keys = configdata.keys()
for k,v in envdata.items():
    if k in config_keys:
        configdata[k] = v
# If .env file is present, the values in .env file should override the config.json values.
# This must be done only for the overlapping keys
#CHANGE ENDS HERE


# In[Get Root folder]

def getRootFolderPath():
    return configdata["ROOT_FOLDER"]

def getTempFolderPath():
    return configdata["TEMP_FOLDER"]

def getPathFromRoot(name):
    folder = configdata[name]
    root_path = getRootFolderPath()
    folder_path = os.path.join(root_path, folder)
    return folder_path
def getZipCodeFilePAth():
    return configdata["ZIP_CODE_RANGE"]
# In[Get Common configuration]

def getPredictionEndpoint():
    return configdata["MODEL_PREDICTION_API_ENDPOINT"]

def getExtractionApiPort():
    return configdata["EXTRACTION_API_PORT"]

def getExternalApiPort():
    return configdata["EXTERNAL_API_PORT"]

def getBlobConnectionString():
    return configdata["BLOB_CONNECTION_STRING"]

def getBlobAccountName():
    return configdata["BLOB_ACCOUNT_NAME"]

def getBlobStorePreproc():
    return configdata["BLOB_FOLDER_PREPROC"]

def getBlobStoreExport():
    return configdata["BLOB_FOLDER_EXPORT"]

def getBlobOPFolder():
    return configdata["BLOB_OUTPUT_FOLDER"]

def getMimeTiff():
    return configdata["MIMETYPE_TIFF"]

def getMimePng():
    return configdata["MIMETYPE_PNG"]

def getMimeJson():
    return configdata["MIMETYPE_JSON"]

def getExtnTiff():
    return configdata["EXTN_TIFF"]

def getExtnPng():
    return configdata["EXTN_PNG"]

def getPreprocessedFileSuffix():
    return configdata["PREPROCESSED_FIL_SFX"]

# In[Feature Engineering Configs]

def getStopWordFilePath():
    return getPathFromRoot('STOP_WORD_FILE')

def getLabelKeywords():
    return getPathFromRoot('HDR_KEYWORDS')

def getlabelKeywordsNonTokenized():
    return getPathFromRoot('HDR_KEYWORDS_NONTOKEN')

def getlabelKeywordsNonTokenized_new():
    return getPathFromRoot('HDR_KEYWORDS_NONTOKEN_NEW')

def getFeatureTypes():
    return getPathFromRoot("FEATURE_TYPES_CSV")

def getLIKeywordsPath():
    return getPathFromRoot('LI_KEYWORDS')

def getHDRKeywords():
    return getPathFromRoot('HDR_KEYWORDS')

def getHDRKeywordsNonTokenized():
    return getPathFromRoot('HDR_KEYWORDS_NONTOKEN')

def getHDRKeywordsNonTokenized_new():
    return getPathFromRoot('HDR_KEYWORDS_NONTOKEN_NEW')

def getLIFeaturesId():
    return configdata['LINE_ITEM_FTRS_ID']

def getLI1FeaturesId():
    return configdata['LINE_ITEM1_FTRS_ID']

def getLINgbrFeaturesId():
    return configdata['NGBR_LINE_ITEM_FTRS_ID']

def getSpatialFeaturesId():
    return configdata['SPATIAL_FTRS_ID']

def getNumericFeaturesId():
    return configdata['NUMERIC_FTRS_ID']

def getOCRFeaturesId():
    return configdata['OCR_FTRS_ID']

def getGSTRates():
    return configdata['GST_RATES']

def getSurroundLabelFeatures():
    return configdata['surround_label_feature']

# In[3]: - OCR specific configuration

def getAzFuncOcrURL():
    return configdata["OCR_FUNCTION_URL"]

# In[Get extraction API configuration]

def getAzFuncPreProc():
    return configdata["FUNC_IMAGEPREPROC_URL"]

def getAzFuncOcrDeskew():
    return configdata["FUNC_COMBINE_OCR_DESKEW_URL"]

def getAzFuncFtrEngg():
    return configdata["FUNC_FT_ENGG_URL"]

def getAzFuncPredDS():
    return configdata["FUNC_PREDICT_DATASET_URL"]

# In[4]: - Configuration functions specific to predict_dataset

def getModelJoblib():
    return getPathFromRoot("MODEL_JOBLIB")

def getLabelMapping():
    return getPathFromRoot("LABEL_MAPPING_CSV")

def getNoOfModelCols():
    return configdata["NO_MODEL_COLUMNS"]

# In[4]: - OCR Function Configurations

def getFormRecognizerEndpoint():
    return configdata["FORM_RECOGNIZER_ENDPOINT"]

def getFormRecognizerAPIKey():
    return configdata["FORM_RECOGNIZER_API_KEY"]

def getFormRecognizerPostURL():
    return configdata["FORM_RECOGNIZER_POST_URL"]

def getOcrTries():
    return configdata["NO_OF_OCR_TRIES"]

def getOcrWaitTime():
    return configdata["OCR_WAIT_TIME"]

# In[] - Image Pre-proc configurations

def getCropBorder():
    return configdata["CROP_BORDER"]

def getCropRequired():
    return configdata["CROP_REQUIRED"]

def getDeskewingRequired():
    return configdata["DESKEWING_REQUIRED"]

# In[] - Subscription related

def getSubDbConn():
    return configdata["SUB_DB_CONN_STRING"]

def getSubPvtEncryptKey():
    return configdata["SUB_PVT_ENCRYPT_KEY"]

def getAuthTknExp():
    return configdata["AUTH_TKN_EXP"]

def getAuthTknExpKey():
    return configdata["AUTH_TKN_EXP_KEY"]

def getAuthTknExpTopic():
    return configdata["AUTH_TKN_EXP_TOPIC"]

def getServicePrinciple():
    return configdata['SERVICE_PRINCIPAL']

def getSecretVault():
    return configdata['SECRET_VAULT']
