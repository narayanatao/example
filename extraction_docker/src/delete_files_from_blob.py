#!/usr/bin/env python
# coding: utf-8

# In[1]:

import psycopg2
import pandas as pd
import traceback
import json
from azure.storage.blob import BlobServiceClient
import config as cfg
import util as util

import logging 

logger = logging.getLogger('name')
logger.setLevel(logging.DEBUG)
sh = logging.StreamHandler()
sh.setLevel(logging.INFO)
logger.addHandler(sh)
logger.info('This will work.')


# In[2]:

# DB credentials
conn_string = cfg.getSubDbConn()

def get_call_subid(auth_token):
    try:
        with psycopg2.connect(conn_string) as conn:
            with conn.cursor() as cursor:
                select_sql = f"SELECT sub_id,call_id from subscription_auth_tokens WHERE auth_token = '{auth_token}'"
                print("sub_id, call_id sql",
                      select_sql)
                cursor.execute(select_sql)
                DF = pd.DataFrame(cursor.fetchall(),
                                  columns = ["sub_id","call_id"])
                if DF.shape[0]==1:
                        return DF.iloc[0]["sub_id"], DF.iloc[0]["call_id"]
                else:
                    return None,None
    except:
        print(traceback.print_exc())
        return None,None

def change_delete_flag(auth_token):
    try:
        with psycopg2.connect(conn_string) as conn:
            with conn.cursor() as cursor:
                upd_sql = "UPDATE subscription_auth_tokens SET all_files_deleted = 1 WHERE auth_token = '"+auth_token+"'"
                print("Update deleted flag",
                      upd_sql)
                cursor.execute(upd_sql)
                conn.commit()
        return True
        
    except:
        print(traceback.print_exc())
        return False

def delete_files_from_container(access_key,
                                account_name,
                                container_name,
                                call_id,
                                auth_token):
    try:
        conn_str="DefaultEndpointsProtocol=https;AccountName="+account_name+";AccountKey="+access_key+";EndpointSuffix=core.windows.net"
        source_blob_service_client = BlobServiceClient.from_connection_string(conn_str)
        source_container_client = source_blob_service_client.get_container_client(container_name)
        source_blobs_auth_token = source_container_client.list_blobs(name_starts_with = auth_token)
        source_blobs_auth_token = list(source_blobs_auth_token)
        source_blobs_call_id = source_container_client.list_blobs(name_starts_with = call_id)
        source_blobs_call_id = list(source_blobs_call_id)
        source_blobs = source_blobs_auth_token + source_blobs_call_id
        source_container_client.delete_blobs(*source_blobs)
        return True
    except:
        logger.debug(traceback.print_exc())
        return False 


def delete_records(input_json):
    try:
        logger.debug("%%%%%%%%%%%%%%%%%%")
        logger.debug(str(input_json))
        enc_message=json.loads(input_json)
        auth_token = enc_message["auth_token"]
        sub_id,call_id=get_call_subid(auth_token)
        logger.debug(str(sub_id))
        # access_key,account_name,container_name=get_blob_account_details(sub_id)
        access_key,account_name,container_name=util.get_blob_account_details(sub_id)
        deleted = delete_files_from_container(access_key,
                                              account_name,
                                              container_name,
                                              call_id,
                                              auth_token)
        logger.debug(str(deleted))
        if deleted:
            updated = change_delete_flag(auth_token)
            if updated:
                print("Successfully updated delete flag")
            return json.dumps({"response_code":200})
        else:
            return json.dumps({"response_code":500})
    finally:
        return json.dumps({"response_code":500})


def main():
	auth_token=json.dumps({"auth_token" : "6cd0d909-1ce6-49db-b21e-4f91bc5bb920"})

if __name__ == "__main__":
	auth_token=main()
	delete_records(auth_token)