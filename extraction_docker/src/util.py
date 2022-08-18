# -*- coding: utf-8 -*-
from curses.ascii import isdigit
import traceback
import os
import time
from PIL import Image
import cv2
import imutils
import pandas as pd
import re
import json
import psycopg2
import config as config
from datetime import datetime, timedelta
from cryptography.fernet import Fernet
from azure.keyvault.secrets import SecretClient
from azure.identity import ClientSecretCredential
import base64
import networkx 
from networkx.algorithms.components.connected import connected_components
import copy 


# from get_sas_token import gen_sas_token
#get script directory
script_dir = os.path.dirname(__file__)
print("Script Directory:", script_dir)

# In[Create a connection to Blob Storage]
from azure.storage.blob import BlobServiceClient
from azure.storage.blob import generate_account_sas, ResourceTypes, AccountSasPermissions

conn_string = config.getSubDbConn()
encryption_key = config.getSubPvtEncryptKey()

# In[amount patterns]
ptn1 = "[0-9]{1,3}[,]{1}[0-9]{3}[,]{1}[0-9]{3}[,]{1}[0-9]{3}[.]{1}[0-9]{1,4}"
ptn2 = "[0-9]{1,3}[,]{1}[0-9]{3}[,]{1}[0-9]{3}[.]{1}[0-9]{1,4}"
ptn3 = "[0-9]{1,3}[,]{1}[0-9]{3}[.]{1}[0-9]{1,4}"
ptn4 = "[0-9]{1,3}[.]{1}[0-9]{1,4}"

ptn5 = "[0-9]{1,3}[.]{1}[0-9]{3}[.]{1}[0-9]{3}[.]{1}[0-9]{3}[,]{1}[0-9]{1,4}"
ptn6 = "[0-9]{1,3}[.]{1}[0-9]{3}[.]{1}[0-9]{3}[,]{1}[0-9]{1,4}"
ptn7 = "[0-9]{1,3}[.]{1}[0-9]{3}[,]{1}[0-9]{1,4}"
ptn8 = "[0-9]{1,3}[,]{1}[0-9]{1,4}"

ptns = [ptn1,ptn2,ptn3,ptn4,ptn5,ptn6,ptn7,ptn8]

# In[1]: Timer function

def timing(f):
    """
    Function decorator to get execution time of a method
    :param f:
    :return:
    """

    def wrap(*args, **kwargs):
        time1 = time.time()
        ret = f(*args, **kwargs)
        time2 = time.time()
        print('{:s}: {:.3f} sec'.format(f.__name__, (time2 - time1)))
        return ret

    return wrap



# In[Machine ID]

def __getMacAddress__():
    """
    Returns
    -------
    get_mac : st
        MAC ID.
    """
    from uuid import getnode as get_mac
    return get_mac()

# In[Subscription credentials]

# Get storage account details from database.
# Storae account details presumed to be named as subscription ID of the client

def parse_blob_triggers(topic, subject):
    """
    Parameters: topic, subject 
    Returns: storage_account ,container_name ,blob_path ,file_name 
    
    """
    m = topic.split('/')
    storage_account = m[m.index('storageAccounts')+1]
    l = subject.split('/')
    container_name = l[l.index('containers')+1]
    blob_path = '/'.join(l[l.index('blobs')+1:])
    file_name = os.path.basename(blob_path)
    
    return storage_account, container_name, blob_path, file_name

def get_blob_account_details_old(sub_id):
        """
        Parameters: sub_id
        Returns: access_key, account_name, container
        """
        try:
            with psycopg2.connect(conn_string) as conn:
                with conn.cursor() as cursor:
                    print("select access_key,account_name,container_name from storage_account WHERE sub_id='"+sub_id+"'")
                    cursor.execute("select access_key,account_name,container_name from storage_account WHERE sub_id='"+sub_id+"'")
                    DF = pd.DataFrame(cursor.fetchall(),columns = ["access_key","account_name","container_name"])
                    if DF.shape[0]==1:
                        return DF.iloc[0]["access_key"], DF.iloc[0]["account_name"],DF.iloc[0]["container_name"]
                    else:
                        return None,None,None
        except Exception as e:
            print(e)
            return None,None,None
        
def get_service_principal():

    encodedString = config.getServicePrinciple()
    base64_string_byte = encodedString.encode("UTF-8")
    oringal_string_bytes = base64.b64decode(base64_string_byte)
    orignal_string = oringal_string_bytes.decode("UTF-8")
    sp_list = orignal_string.split(';')
    tenant_id, client_secret, client_id = sp_list[0],sp_list[1],sp_list[2]
    
    return tenant_id, client_secret, client_id

def get_blob_account_details(sub_id):
    '''
    

    Parameters
    ----------
    sub_id : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    '''
    try:
        keyVaultName = config.getSecretVault()
        KVUri = f"https://{keyVaultName}.vault.azure.net"
        tenant_id, client_secret, client_id = get_service_principal()
        credential = ClientSecretCredential(tenant_id = tenant_id,
                                    client_id = client_id,
                                    client_secret = client_secret)

        
        client = SecretClient(vault_url=KVUri, credential=credential)
        access_key = client.get_secret('access').value
        account_name = client.get_secret('account').value
        container_name = sub_id
        
        return access_key, account_name, container_name
    except Exception as e:
        print(e)
        return None,None,None
        
    # Decode the service principle
    
    

# In[Blob access related functions]

def check_subid_DB(sub_id,mac_id):
    try:
        with psycopg2.connect(conn_string) as conn:
            with conn.cursor() as cursor:
                select_sql = f"SELECT sub_id FROM allowed_machines where sub_id = '{sub_id}' and mac_id = '{mac_id}'"
                # logger.debug(select_sql)
                print("GET SAS Token",select_sql)
                cursor.execute(select_sql)
                # logger.debug("executed cursor")
                DF = pd.DataFrame(cursor.fetchall(),columns = ["sub_id"])
                # logger.debug("fetched data from cursor")
                # logger.debug("DataFrame shape is:" + str(DF.shape))
                if DF.shape[0] >= 1:
                    sub_id = DF['sub_id'].iloc[0]
                    # logger.debug("Subscriber id is:" +str(sub_id))
                    return str(sub_id)
                else:
                    return None
    except:
        print("subid not found in DB")
        return None

def generate_SAS_token(sub_id,
                       activity,
                       file_size = None,
                       IP = None):

    try:

        # logger.debug("generate sas token, sub id:" + sub_id)
        access_key,account_name,container_name = get_blob_account_details(sub_id)
        time_mins = 10
        if access_key is not None and account_name is not None:
            account_url="https://"+account_name+".blob.core.windows.net"
            if activity.lower() == "upload":
                sas_token = generate_account_sas(
                    account_name=account_name,
                    account_key=access_key,
                    resource_types=ResourceTypes(object=True),
                    ip=IP,
                    permission=AccountSasPermissions(write=True),
                    expiry=datetime.utcnow() + timedelta(minutes=time_mins))
                # logger.debug("successfully generated token: upload")
            elif activity.lower() == "download":
                sas_token = generate_account_sas(
                    account_name=account_name,
                    account_key=access_key,
                    resource_types=ResourceTypes(object=True),
                    ip=IP,
                    permission=AccountSasPermissions(read=True),
                    expiry=datetime.utcnow() + timedelta(minutes=time_mins))
                # logger.debug("successfully generated token: download")
            elif activity.lower() == "delete":
                sas_token = generate_account_sas(
                    account_name=account_name,
                    account_key=access_key,
                    resource_types=ResourceTypes(object=True),
                    ip=IP,
                    permission=AccountSasPermissions(delete=True),
                    expiry=datetime.utcnow() + timedelta(minutes=time_mins))
                # logger.debug("successfully generated token: delete")
            else:
                sas_token = None
                account_url = None
                # logger.debug("not generated token")
            # logger.debug("Account details: " + account_url + sas_token + container_name)
            return account_url,sas_token,container_name
    except:
        print("failure while sas token generation")
        return None,None,None

def gen_sas_token(sub_id,mac_id,IP,activity):
    try:
        # sub_id = check_subid_DB(sub_id, mac_id)
        print(sub_id, "subid")
        if sub_id is None:
            return None,None, None
        account_url,credential,container_name=generate_SAS_token(sub_id,
                                                                 activity,
                                                                 None,
                                                                 IP=IP)
        if account_url is None:
            return None,None,None
        return account_url,credential,container_name
    except:
        print(traceback.print_exc())
        return None, None, None


# In[cryptography]

def decryptMessage(encrpyted_json):
    try:
        fernet = Fernet(encryption_key)
        enc_message=bytes(encrpyted_json,'utf-8')
        message = fernet.decrypt(enc_message).decode()
        return message
    except:
        print("Decryption fails")
        return None

def encrypt_message(input_json):
    try:
        f = Fernet(encryption_key)
        encrypted_token = f.encrypt(bytes(input_json, 'utf-8'))
        encrypted_token=encrypted_token.decode("utf-8")
        enc_message=json.dumps({"message": encrypted_token,
                                "status":200,
                                "error_message":""})
        return enc_message
    except:
        enc_message={"message":"",
                     "status":404,
                     "error_message":"issue in encryption"}
        return json.dumps(enc_message)


# In[File access from/to blob]

def uploadFilesToBlobStore(sub_id,filePath):
    """

    Parameters
    ----------
    sub_id : TYPE
        DESCRIPTION.
    filePath : TYPE
        DESCRIPTION.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """    

    try:
        mac_id = __getMacAddress__()
        mac_id = '105852647403622'
        account_url,credential,container_name = gen_sas_token(sub_id,
                                                              mac_id,
                                                              None,
                                                              "upload")
        blob_service_client = BlobServiceClient(account_url=account_url,
                                                credential=credential)
        
        fileName = os.path.basename(filePath)
        blob_client = blob_service_client.get_blob_client(container = container_name,
                                                          blob = fileName)
        with open(filePath, "rb") as data:
            blob_client.upload_blob(data,
                                    overwrite=True)
        blobPath = container_name + "/" + fileName

        return True,blobPath
    except:
        print(traceback.print_exc())
        return False, None

def downloadFilesFromBlobStore(sub_id,
                               fileURI,
                               localPath):

    try:
        mac_id = __getMacAddress__()
        mac_id = '105852647403622'
        account_url,credential,container_name = gen_sas_token(sub_id,
                                                              mac_id,
                                                              None,
                                                              "download")
        blob_service_client = BlobServiceClient(account_url=account_url,
                                                credential=credential)
        print("Download From Blob Store", fileURI)

        splitURI = fileURI.split(container_name+"/")
        # print(splitURI)

        if len(splitURI) == 1:
            blobname = splitURI[0]
        elif len(splitURI) >1:
            blobname = splitURI[-1]
        else:
            return False
        print(blobname)

        blob_client = blob_service_client.get_blob_client(container=container_name,
                                                          blob=blobname)
        time.sleep(0.01)
        with open(localPath,"wb") as download_file:
            download_file.write(blob_client.download_blob().readall())


        return True
    except:
        print(traceback.print_exc())
        return False

def downloadStreamFromBlob(sub_id, fileURI):

    try:
        mac_id = __getMacAddress__()
        mac_id = '105852647403622'
        account_url,credential,container_name = gen_sas_token(sub_id,
                                                              mac_id,
                                                              None,
                                                              "download")
        blob_service_client = BlobServiceClient(account_url=account_url,
                                                credential=credential)
        print("Download From Blob Store", fileURI)
        blob_client = blob_service_client.get_blob_client(container=container_name,
                                                          blob=fileURI)
        streamdownloader = blob_client.download_blob()
        content = json.loads(streamdownloader.readall())
    
        
        return content, True
    except:
        print(traceback.print_exc())
        return {}, False


# In[Image related functions]

def rotateImageByAngle(imgpath, rotationAngle):
    #Takes image path, reads the image with opencv
    #Takes angle, rotates the image by angle
    #Saves image back in same path -> return type::boolean
    try:
        img1 = cv2.imread(imgpath)
        rotated_image = imutils.rotate_bound(img1,rotationAngle)
        return cv2.imwrite(imgpath, rotated_image)
    except:
        print("rotateImageByAngle",
              traceback.print_exc())
        return None

def converTiffToPng(image_path,png_path):
    '''
    '''
    try:
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        cpy = Image.fromarray(img)
        
        cpy.save(png_path)
        return png_path
    except:
        print(traceback.print_exc())
        return None

# In[Token related functions]
def remove_rupee_symbol(text):
    text_copy = copy.deepcopy(text)
    try:
        lb = len(text)
        text = str(text).replace(u'\u20B9','') # removing Rupee symbol
        la = len(text)
        if (la <lb):
            if text.isdigit():
                text = str(float(text))
                return text
            else:
                return text
        return text

    except:
        print("Removing rupee  sysmbol from text exception :",traceback.print_exc())
        return False, text_copy

def isAmount(s):
    try:
        # print("text befor rplaceing rupyee:",s)
        s = remove_rupee_symbol(s)
        # print("text after rplaceing rupyee:",s)
        for ptn in ptns:
            l = re.findall(ptn,s)
            l1 = [g for g in l if len(g) > 0]
            if len(l1) >= 1:
                # print("isAmount written true")
                return True
    except:
        print("isamount exception:",traceback.print_exc())
        return False
    return False

def isTokenAmount(s):

    try:
        t = s.replace(" ","")
        l = len(t)

        for ptn in ptns:
            search = re.search(ptn,t)
            if search is not None:
                if search.start() == 0 and search.end() == l:
                    return True
        return False
    except Exception as e:
        print(e)
        return False

def isNumber(s):

    try:
        cnt = 0
        tot = 0
        for itm in list(s):
            if str(itm).isnumeric():
                cnt += 1
            tot += 1
        if cnt/tot > .75:
            return True
    except:
        return False
    return False

def wordshape(text):
    if not(pd.isna(text)):
        t1 = re.sub('[A-Z]', 'X',text)
        t2 = re.sub('[a-z]', 'x', t1)
        return re.sub('[0-9]', 'd', t2)
    return text

def is_alpha_numeric(text):
    """
    Checks wheter a string a alphanumeric (Letters and Digit) or not
    Return: 1 when Alphanumeric, 0 otherwise
    """
    return 1 if not(pd.isna(text)) and text.isalnum() else 0

def is_alpha(text):
    """
    Checks wheter a string a alph (Letters) or not
    Return: 1 when Alpha, 0 otherwise
    """
    return 1 if not(pd.isna(text)) and str(text).isalpha() else 0

def onehot(df,col_name):

    # Get one-hot encoding for categorical features
    one_hot = pd.get_dummies(df[col_name])
    df = df.drop(col_name, axis=1)

    return df,one_hot

def getUniqueWords(words):
    words = list(set(words))
    if '' in words:
        words.remove('')
    return words

def gstin_pattern():
    format_ = r'\d{2}[A-Z]{5}\d{4}[A-Z]{1}[A-Z\d]{1}[Z|2]{1}[A-Z\d]{1}'
    compiled_ = re.compile(format_)
    return format_,compiled_

import traceback
def pan_pattern(string):
    try:
        string =str(string)
        GSTIN = re.compile("\d{2}[A-Z]{5}\d{4}[A-Z]{1}[A-Z\d]{1}[Z|2]{1}[A-Z\d]{1}")
        is_gstin = re.search(GSTIN,string)
        PAN = re.compile("[A-Z]{5}[0-9]{4}[A-Z]{1}")
        is_pan = re.search(PAN,string)
        #print("PAN :",is_pan,"\tGSTIN : ",is_gstin)
        if (is_pan is not None) and (is_gstin is None):
            return 1
        else:
            return 0
    except:
        print(traceback.print_exc())
        return 0


def email_pattern():
    format_ = r"[^@]+@[^@]+\.[^@]+"
    compiled_ = re.compile(format_)
    return format_,compiled_


# In[] - utility functions for feature engineering

def getLabelKeywords():

    try:
        labelKeywordPath = config.getLabelKeywords()
        fullPath = os.path.join(script_dir,
                                labelKeywordPath)
        print("Label Keywords Path: ", fullPath)
        with open(fullPath,"rb") as json_file:
            json_content = json_file.read()
            labelKeywords = json.loads(json_content)
            for key in labelKeywords.keys():
                labels = labelKeywords[key]
                labels = [l.lower().strip() for l in labels]
                labelKeywords[key] = labels
        return labelKeywords
    except:
        print(traceback.print_exc())
        return None

def getlabelKeywords_nonTokenized():  # to preProcUtilies files
    try:
        labelKeywordPath = config.getlabelKeywordsNonTokenized()
        labelKeywordPath_fullPath = os.path.join(script_dir,
                                                 labelKeywordPath)

        print("New Label Keywords Path: ",
              labelKeywordPath_fullPath)
        with open(labelKeywordPath_fullPath,
                  "rb") as json_file:
            json_content = json_file.read()
            json_obj = json.loads(json_content)
        return json_obj

    except:
        print(traceback.print_exc())
        return None

def getlabelKeywords_nonTokenized_new():  # to preProcUtilies files
    try:
        labelKeywordPath = config.getlabelKeywordsNonTokenized_new()
        labelKeywordPath_fullPath = os.path.join(script_dir,
                                                 labelKeywordPath)

        print("New Label Keywords Path: ",
              labelKeywordPath_fullPath)
        with open(labelKeywordPath_fullPath,
                  "rb") as json_file:
            json_content = json_file.read()
            json_obj = json.loads(json_content)
        return json_obj

    except:
        print(traceback.print_exc())
        return None

def getFeatureTypes():

    try:
        featureTypesPath = config.getFeatureTypes()
        fullPath = os.path.join(script_dir,featureTypesPath)
        dfFtrType = pd.read_csv(fullPath)
        return dfFtrType
    except:
        print("getFeatureTypes",
              traceback.print_exc())
        return None

# In[Network functions to find common nodes in a network of lists]

def to_edges(l):
    """ 
        treat `l` as a Graph and returns it's edges 
        to_edges(['a','b','c','d']) -> [(a,b), (b,c),(c,d)]
    """
    it = iter(l)
    last = next(it)

    for current in it:
        yield last, current
        last = current 


def to_graph(l):
    G = networkx.Graph()
    for part in l:
        # each sublist is a bunch of nodes
        G.add_nodes_from(part)
        # it also imlies a number of edges:
        G.add_edges_from(to_edges(part))
    return G

def connect_lines(lines):
    #convert lines to graph
    G = to_graph(lines)
    #connect graph into regions
    return list(connected_components(G))

# In[Dataframe related]
@timing
def reduce_mem_usage(df, int_cast=True, obj_to_category=False, subset=None):
    """
    Iterate through all the columns of a dataframe and modify the data type to reduce memory usage.
    :param df: dataframe to reduce (pd.DataFrame)
    :param int_cast: indicate if columns should be tried to be casted to int (bool)
    :param obj_to_category: convert non-datetime related objects to category dtype (bool)
    :param subset: subset of columns to analyse (list)
    :return: dataset with the column dtypes adjusted (pd.DataFrame)
    """
    import numpy as np
    start_mem = df.memory_usage().sum() / 1024 ** 2;
    # gc.collect()
    print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))

    cols = subset if subset is not None else df.columns.tolist()

    for col in cols:
        col_type = df[col].dtype

        if col_type != object and col_type.name != 'category' and 'datetime' not in col_type.name:
            # c_min = df[col].min()
            # c_max = df[col].max()

            # test if column can be converted to an integer
            if str(col_type)[:3] == 'int':
                df[col] = df[col].fillna(0)
                df[col] = df[col].astype(np.int16)
            elif str(col_type)[:5] == 'float':
                df[col] = df[col].fillna(0.0)
                df[col] = df[col].astype(np.float16)

    end_mem = df.memory_usage().sum() / 1024 ** 2
    print('Memory usage after optimization is: {:.3f} MB'.format(end_mem))
    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))

    return df
# In[assign Values to DF from dictionary based on token_id]

def assignVavluesToDf(col_name,col_vals,df,
                      base_col = "token_id"):

    df_copy = df.copy(deep = True)
    try:
        import numpy as np
        new_col = col_name + "_new"
        df[new_col] = df[base_col].map(col_vals)
        df[col_name] = np.where(df[new_col].isnull(),
                                df[col_name],
                                df[new_col])
        return df
    except:
        print("assignVavluesToDf",
              traceback.print_exc())
        return df_copy
