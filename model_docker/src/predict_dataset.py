#!/usr/bin/env python
# coding: utf-8

# In[1]: - Import packages
import traceback
import pandas as pd
import time
import os
from joblib import load
import numpy as np
import json
import util as util
import config as cfg
from datetime import datetime
import sys

# In[2]: - Util functions for predict_dataset

script_dir = os.path.dirname(__file__)
print("Script Directory:", script_dir)

@util.timing
def getModelJoblib():

    try:
        modelJoblibPath = cfg.getModelJoblib()
        fullPath = os.path.join(script_dir,modelJoblibPath)
        model = load(fullPath)
        return model
    except Exception as e:
        print(traceback.print_exc(),e)
        return None

# In[3]: - Variable Declaration

ROOT_FOLDER = cfg.getRootFolderPath()
outputFolder = cfg.getBlobOPFolder()
blobAccount = cfg.getBlobAccountName()

# In[3]: - # Loading Model file, feature files and labels
model = getModelJoblib()
print("Model file reading done!!!")
no_model_cols = cfg.getNoOfModelCols()


# In[]: - # Define return json structures

def uploadStatsToBlob(client_blob_folder,
                        auth_token,
                        documentId,
                        sub_id, 
                        body, 
                        stage,
                        success,
                        start, 
                        end):
    def updateStats():
        stats = {}
        stats['auth_token'] = auth_token
        stats['document_id'] = documentId
        stats['sub_id'] = sub_id
        stats['body'] = body
        stats['stage'] = stage
        stats['success'] = success
        stats['create_time'] = str(datetime.now())
        stats['is_start'] = start
        stats['is_end'] = end 
        response = json.dumps(stats,
                              indent = 4,
                              sort_keys = False,
                              default = str)
        return response
    
    file_contents = updateStats()
    file_name = str(auth_token)+"__" + str(stage) +'.json'
    file_path = os.path.join(client_blob_folder, file_name)
    with open(file_path,"w") as f:
            f.write(file_contents)
            
    uploaded, URI = util.uploadFilesToBlobStore(sub_id, file_path)
    
    if uploaded:
        resp_code = 200
    else:
        resp_code = 404
        
    docInfo = {}
    docInfo['status'] = "Success"
    docInfo['responseCode'] = resp_code
    
    response = json.dumps(docInfo,
                          indent = 4,
                          sort_keys = False,
                          default = str)
    return response
    

def returnFailure(client_blob_folder, 
                  auth_token,
                  sub_id,
                  documentId, 
                  stage):     
    """
    

    Parameters
    ----------
    accountName : TYPE
        DESCRIPTION.
    client_blob_folder : TYPE
        DESCRIPTION.
    accessKey : TYPE
        DESCRIPTION.
    auth_token : TYPE
        DESCRIPTION.
    sub_id : TYPE
        DESCRIPTION.
    documentId : TYPE
        DESCRIPTION.
    stage : TYPE
        DESCRIPTION.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    
    def getFailedStats():
        
        def failedStats(auth_token, sub_id):
            stats = {}
            stats['auth_token'] = auth_token
            stats['document_id'] = documentId
            stats['sub_id'] = sub_id
            stats['body'] = json.dumps({})
            stats['stage'] = "extraction"
            stats['success'] = 0
            stats['create_time'] = str(datetime.now())
            stats['is_start'] = 0
            stats['is_end'] = 1
            response = json.dumps(stats,
                                  indent = 4,
                                  sort_keys = False,
                                  default = str)
            return response
        print("Failed Auh token {} for Sub ID {}".format(auth_token, sub_id))
        file_contents = failedStats(auth_token, sub_id)
        print("Failed ", file_contents)
        file_name = str(auth_token)+"__failed.json"
        file_path = os.path.join(client_blob_folder, file_name)
        print("Failed File: ", file_path)
        with open(file_path,"w") as f:
            f.write(file_contents)
                
        uploaded, URI = util.uploadFilesToBlobStore(sub_id, file_path)
        print("Failed status updated: {}, at {}".format(uploaded, URI))
        if uploaded:
            try:
                os.remove(file_path)
            except Exception:
                print(traceback.print_exc())
        
        return uploaded
    failed = getFailedStats()
    if failed:
        resp_code = 404
    else:
        resp_code = 200
    docInfo = {}
    docInfo['status'] = "Failure"
    docInfo['responseCode'] = resp_code
    response = json.dumps(docInfo,
                          indent = 4,
                          sort_keys = False,
                          default = str)
    return response



# In[3]: Define functions
@util.timing
def get_prediction(docInput):

    @util.timing
    def prepare_dataset(df):
        def getFeatures():
            try:
                featureTypesPath = cfg.getFeatureTypes()
                fullPath = os.path.join(script_dir,featureTypesPath)
                all_features = pd.read_csv(fullPath)
                model_features = list(
                    all_features[all_features[
                        "model_input"] == 1]["Column_Names"].unique())
                return all_features,model_features
            except:
                print("get_prediction.prepare_dataset.getFeatures",
                      traceback.print_exc())
                return None

        #Get Feature definitions
        try:
            
            all_features,model_features = getFeatures()
        
            feature_type_dict = {}
            feature_fill_dict = {}
        
            # print(all_features['Column_Names'].to_list())
        
            for ind,row in all_features.iterrows():
                if row["Column_Names"] in model_features:
                    if row['Var_Type'] == 'Discrete':
                        feature_type_dict[row['Column_Names']] = int
                        feature_fill_dict[row['Column_Names']] = 0
                    elif row['Var_Type'] == 'Continous':
                        feature_type_dict[row['Column_Names']] = float
                        feature_fill_dict[row['Column_Names']] = 0.0
        
            df = df[model_features]
            df.fillna(feature_fill_dict, inplace=True)
            df = df.astype(feature_type_dict)
    
            return df
        except:
            print("get_prediction.prepare_dataset",
                  traceback.print_exc())
            return None

    @util.timing
    def make_prediction(df,df_prepared):

        try:
            print("DATASET SHAPE: ",
                  df.shape,
                  df_prepared.shape,
                  no_model_cols)
            #Check if no of features are same as what is required in model
            assert df_prepared.shape[1] == no_model_cols

            #Execute prediction of the model
            t = time.time()
            p = model.predict(df_prepared,
                              prediction_type='Probability')
            print("prediction Done:", time.time() - t)
            DF = df.copy()
            print("TESTE:",
                  len(p),
                  DF.shape,
                  p.shape)
            DF[probabilty] = pd.DataFrame(p,
                                          index = DF.index)

            DF[prediction] = pd.DataFrame(p.argsort(axis=-1),
                                          index=DF.index)

            return DF
        except:
            print("get_prediction.make_prediction",
                  traceback.print_exc())
            return None

    @util.timing
    def getLabelMapping():

        try:
            labelMappingPath = cfg.getLabelMapping()
            fullPath = os.path.join(script_dir,labelMappingPath)
            label_df = pd.read_csv(fullPath)
            label_df = label_df.loc[label_df['used'] == 1]
            label_df = label_df.loc[:, 'new_label':'new_label_cat']
            label_df.drop_duplicates(subset = "new_label",
                                     keep = 'last',
                                     inplace = True)
            
            # Prepare Label Catergory dict
            label_cat = label_df.set_index('new_label').T.to_dict('list')
            for i in label_cat:
                label_cat[i] = label_cat[i][0]
            
            no_of_labels = label_df.shape[0]
            
            label_cat_rev = label_df.set_index('new_label_cat').T.to_dict('list')
            for i in label_cat_rev:
                label_cat_rev[i] = label_cat_rev[i][0]

            return label_cat_rev,no_of_labels
        except:
            print("get_prediction.getLabelMapping",
                  traceback.print_exc())
            return None, None

    try:
        t = time.time()
        documentId = docInput["documentId"]
        input_blob_path = docInput["input_path"]
        client_folder = docInput["client_folder"]
        container = docInput["container"]

        os.makedirs(client_folder, exist_ok=True)

        #Download input path to local folder
        input_file_name = input_blob_path.split("/")[-1]
        input_path = os.path.join(ROOT_FOLDER,
                                  input_file_name)

        downloaded = util.downloadFilesFromBlobStore(container,
                                                     blobAccount + "/" + input_blob_path,
                                                     input_path)
        if not downloaded:
            return None

        # Read and Prepare Label Mapping Dataframe
        label_cat_rev,no_of_labels = getLabelMapping()

        # Creating Prob and Pred Column List
        probabilty = []
        prediction = []

        print("No of Label Class:", no_of_labels)

        for i in range(0,no_of_labels):
            probabilty.append("prob_" + label_cat_rev[i])

        for i in range(no_of_labels,0,-1):
            prediction.append("pred_" + str(i))


        print("Reading Feature Input Started", t)
        df_test = pd.read_csv(input_path, index_col=0)
        df_test.reset_index(drop=True, inplace=True)
        print("Reading Feature Input Done:", time.time() - t)

        t = time.time()
        print("Preparing Dataset Started", t)
        df_prepared = prepare_dataset(df_test)
        if df_prepared is None:
            raise ValueError("df_prepared Failed")
        print("Preparing Dataset Done:", time.time() - t)

        t = time.time()
        print("make_prediction Started", t)
        DF_PRED = make_prediction(df_test,df_prepared)
        if DF_PRED is None:
            raise ValueError("make_prediction Failed")
        print("make_prediction Done:", time.time() - t)

        df_test = None
        df_prepared = None

        DF_PRED['predict_label_cat'] = DF_PRED['pred_1']
        DF_PRED['predict_label'] = DF_PRED['predict_label_cat'].map(label_cat_rev)
        probcols = list(DF_PRED.columns.values)
        probcols = [col for col in probcols if "prob_" in col]
        DF_PRED["prediction_probability"] = np.nanmax(
            DF_PRED[probcols].values,
            axis = 1)
        print("RETURNING PREDICTION")
        DF_PRED["FileName"] = documentId
        pred_file_name = documentId + "_pred.csv"
        pred_file_path = os.path.join(ROOT_FOLDER,
                                      client_folder,
                                      pred_file_name)
        DF_PRED.to_csv(pred_file_path)

        #Upload the prediction dataset to blob storage
        # blob_destination = client_folder + "/" + pred_file_name
        uploaded, URI = util.uploadFilesToBlobStore(container,
                                                    pred_file_path)
        if not uploaded:
            print("Failed in uploading prediction file to blob")
            return None
        else:
            try:
                os.remove(pred_file_path)
                os.remove(input_path)
            except:
                pass

        return json.dumps({"status_code":200,"URI":URI})

    except ValueError as e:
        print("Prediction Function: Get Prediction: ",e,
              traceback.print_exc())
        return json.dumps({"status_code":500,"URI":None})


def predict(docInput):
    # Logging
    # print("Before parsing input json")
    # rawContent = docInput.content.read()
    # encodedContent = rawContent.decode("utf-8")
    # stats = json.loads(encodedContent)
    # print("Printing docInfo content: ", stats)
    topic = docInput['topic']
    subject = docInput['subject']
    print("predict started", topic, subject)

    # Get storage account details 
    storage_account, container_name, blob_path, file_name = util.parse_blob_triggers(topic,
                                                                                     subject)
    sub_id = container_name
    print("Parsed Values: ",
          storage_account,
          container_name,
          blob_path,
          file_name)

    fileURI = os.path.basename(subject)
    print("Testing input: ",
          sub_id,
          file_name,
          fileURI)
    # Download content of HTTP request 
    content, received = util.downloadStreamFromBlob(sub_id, fileURI)

    # Get auth_token, sub_id from content
    print("Printing input JSON")
    print(content)
    auth_token = content["auth_token"]
    sub_id = content["sub_id"]
    # stage = content["stage"]
    # is_start = content["is_start"]
    # is_end = content["is_end"]

    ftr_result = json.loads(content['body'])
    ftr_file_blob_path = ftr_result.get("URI")
    documentId = ftr_result["documentId"]
    
    print("Before logging")
    std_out = sys.stdout
    std_err = sys.stderr
    logfilepath = str(documentId) + "_model.log"
    logfile = open(logfilepath,"w")
    sys.stdout = logfile
    sys.stderr = logfile


    if "client_blob_folder" in docInput.keys():
        client_blob_folder = content["client_blob_folder"]
    else:
        client_blob_folder = "COMMON"

    os.makedirs(client_blob_folder,
                exist_ok=True)

    prediction_input = {"input_path":ftr_file_blob_path,
                        "documentId":documentId,
                        "client_folder":client_blob_folder,
                        "container" : sub_id}
    print("Input for predict dataset function ", prediction_input)

    try:
        t = time.time()
    
        # Get URI of the pred file in json
        extractionResult = get_prediction(prediction_input)
        print("TIME TAKEN FOR DATASET PREDICTION FUNCTION: ",
              time.time() - t)
        response = json.loads(extractionResult)
        print("Prediction done!")
        print(response)
        if response.get('status_code') != 200:
            return returnFailure(client_blob_folder, 
                                  auth_token, 
                                  sub_id, 
                                  documentId, 
                                  "extraction")
    
        else:
            return uploadStatsToBlob(client_blob_folder,
                               auth_token,
                              documentId,
                              sub_id,
                              json.dumps({"pages" : ftr_result['pages'], 
                                          "pred_file" : response.get("URI")}),
                                          "extraction",1,0,0)
    except:
        print(traceback.print_exc())
        return returnFailure(client_blob_folder, 
                                  auth_token, 
                                  sub_id, 
                                  documentId, 
                                  "extraction")
    finally:
        if not logfile.closed:
            logfile.close()
            sys.stdout = std_out
            sys.stderr = std_err
            # destination = client_blob_folder + "/" + os.path.basename(logfilepath)
            uploaded, URI = util.uploadFilesToBlobStore(sub_id,
                                                        logfilepath)
            try:
                os.remove(logfilepath)
            except:
                pass
            if uploaded:
                print("prediction log file uploaded to Blob")

def main():
    topic = "/subscriptions/3d34cc1f-baa0-4d2e-80b3-95a1834afe2f/resourceGroups/TAPP/providers/Microsoft.Storage/storageAccounts/submasterstorage"
    subject = "/blobServices/default/containers/f3cce063-3d5b-4cec-ab48-623a853f859c/blobs/2fd446ab-c246-4e05-a223-1e2b7c164179__feature_engg.json"
    docInput = {"topic":topic,
                "subject":subject}
    _ = predict(docInput)

main()



