#!/usr/bin/env python
# coding: utf-8

import os
import traceback
import json
from requests import post
import psycopg2
from datetime import datetime
import uuid
import delete_files_from_blob as delblob
import config as cfg
import util as util

# In[1]: - Get database credentials

import logging 

logger = logging.getLogger('name_1')
logger.setLevel(logging.DEBUG)
sh = logging.StreamHandler()
sh.setLevel(logging.INFO)
logger.addHandler(sh)
logger.info('This will work.')

# DB credentials
conn_string = cfg.getSubDbConn()

auth_tkn_exp_url = cfg.getAuthTknExp()
auth_tkn_exp_key = cfg.getAuthTknExpKey()
auth_tkn_exp_topic = cfg.getAuthTknExpTopic()

# In[2]: - Send call o delete files via event

headers = {"aeg-sas-key": auth_tkn_exp_key,
            "Content-Type":"application/json"}

post_url = auth_tkn_exp_url


# In[2]: - Input file endpoints to monitor
suffix_extensions = tuple(['__submit.json',
                           '__ocr.json',
                           '__extraction.json',
                           '__islaststage.json',
                           '__failed.json'])

# In[3]: - Clean Blob Files

def delete_request(delete_data):
    resp = post(url = post_url,
                    data = delete_data,
                    headers = headers)
    print("DELETE REQUEST SENT:")
    print(resp.status_code)
    print(resp.content)
    if resp.status_code != 202:
        print("Delete trigger failed:\n%s" % resp.text)
        return False
    else:
        return True

def generate_body(auth_token, document_id):
    try:
        body = [{"id": str(uuid.uuid1()).split('-')[0],
                "eventType": "recordInserted",
                "subject": "delete/by/authtoken/or/callid",
                "eventTime": str(datetime.now()),
                "data": {'auth_token': auth_token, 'document_id': document_id},
                "dataVersion": "1.0",
                "metadataVersion": "1",
                "topic": auth_tkn_exp_topic
            }]
        return json.dumps(body)
    except:
        return None

# In[3]: - Update database
def update_db(docInfo):
    """
    Parameters
    ----------
    docInfo : JSON
    Returns: RESPONSE.
    """
    print("docInfo: ",docInfo)
    #logger.debug("Entering the trigger")
    def get_storage_details(topic, subject):
        try:

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
        except:
            return None, None, None, None

    def update_db_status(stats):
        """
        Parameters: stats
        Returns: None
        """
        try:
            print("Input: ",stats)
            auth_token = stats['auth_token']
            status = stats["success"]
            stage = stats['stage']
            document_id = stats['document_id']
            body = stats["body"]
            if isinstance(body,dict):
                for key,val in body.items():
                    if isinstance(val,str):
                        body[key] = val.replace("'","__")
                body = json.dumps(body)
                print("Body:",body)
            elif isinstance(body,str):
                body = body.replace("'","__")

            create_time = stats["create_time"]
            is_start = stats['is_start']
            is_end = stats['is_end']
            with psycopg2.connect(conn_string) as conn:
                with conn.cursor() as cursor:
                    ins_sql_1 = "INSERT INTO call_log(auth_token,stage,status,time,is_start,is_end) VALUES ('"+auth_token+"','"+stage+"','"+str(status)+"',TIMESTAMP '"+str(create_time)+"','"+str(is_start)+"','"+str(is_end)+"')"
                    print("Insert SQL 1:",ins_sql_1)
                    cursor.execute(ins_sql_1)
                    ins_sql_2 = f"INSERT INTO stage_result(auth_token, body, stage) VALUES ('{auth_token}', '{body}', '{stage}')"
                    print("Insert SQL 2:",ins_sql_2)
                    cursor.execute(ins_sql_2)
                    # if (status == 0) or (status == 1 and is_end == 1):
                    #     upd_sql = "UPDATE subscription_call_records SET status = -1 WHERE auth_token = '"+auth_token+"'"
                    #     print("Update SQL:", upd_sql)
                    #     cursor.execute(upd_sql)
                    #     print("delete request")
                    #     deleted = delblob.delete_records(auth_token)
                    #     if deleted:
                    #         print("files deleted:",auth_token)
                    if (status == 0):
                        upd_sql = "UPDATE subscription_call_records SET status = -1 WHERE auth_token = '"+auth_token+"'"
                        print("Update SQL:", upd_sql)
                        cursor.execute(upd_sql)
                        print("delete request")
                        #publish event to topic: tokenExpiration
                        deleted = delblob.delete_records(auth_token)
                        if deleted:
                            print("files deleted:",auth_token)
                    elif (status == 1 and is_end == 1):
                        upd_sql = "UPDATE subscription_call_records SET status = 1 WHERE auth_token = '"+auth_token+"'"
                        print("Update SQL:", upd_sql)
                        cursor.execute(upd_sql)
                        print("delete request")
                        deleted = delblob.delete_records(auth_token)
                        if deleted:
                            print("files deleted:",auth_token)

                    conn.commit()
                    return True
        except:
            print("Database connection failed")
            print(traceback.print_exc())
            return False

    try:
        docInfo = json.loads(docInfo)
        topic = docInfo['topic']
        subject = docInfo['subject']
        storage_account, container_name, blob_path, file_name = get_storage_details(topic,
                                                                                    subject)
        sub_id = container_name
        if storage_account is None:
            return {"status": "Failed: No Storage account details"}
        if file_name.endswith(suffix_extensions):
            print("Processing: ", file_name)
            stats, downloaded = util.downloadStreamFromBlob(sub_id, blob_path)
            if not downloaded:
                return json.dumps({"status": "Failed: To download Json file"})
            updated = update_db_status(stats)
            if not updated:
                return json.dumps({"status": "Failed: To update DB"})
            return json.dumps({"status": "Stage updated"})
        else:
            print("File name received is: ",file_name)
            return json.dumps({"status": "Failed: Incorrect file name"})
    except:
        print(traceback.print_exc())
        return json.dumps({"status":"Error: Failed in update call log"})
