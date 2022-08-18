# -*- coding: utf-8 -*-
import traceback
import os
import time
import pandas as pd
import json
import psycopg2
import config as config
from datetime import datetime, timedelta
#get script directory
script_dir = os.path.dirname(__file__)
print("Script Directory:", script_dir)

# In[Create a connection to Blob Storage]
from azure.storage.blob import BlobServiceClient
from azure.storage.blob import generate_account_sas, ResourceTypes, AccountSasPermissions


from azure.keyvault.secrets import SecretClient
from azure.identity import ClientSecretCredential
import base64

conn_string = config.getSubDbConn()
encryption_key = config.getSubPvtEncryptKey()


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
        # access_key,account_name,container_name=get_blob_account_details(sub_id)
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
        print(splitURI)
        
        if len(splitURI) == 1:
            blobname = splitURI[0]
        elif len(splitURI) >1:
            blobname = splitURI[-1]
        else:
            return False
        print(blobname)
        
        blob_client = blob_service_client.get_blob_client(container=container_name,
                                                          blob=blobname)
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
