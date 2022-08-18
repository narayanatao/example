# -*- coding: utf-8 -*-

import psycopg2
import pandas as pd
import traceback
import uuid
from datetime import datetime,timedelta
import json
import os
import util as util
import config as cfg
import sys

files_dir = os.path.join(os.getcwd(),"files")

# import logging

# logger = logging.getLogger('name_1')
# logger.setLevel(logging.DEBUG)
# sh = logging.StreamHandler()
# sh.setLevel(logging.INFO)
# logger.addHandler(sh)
# logger.info('This will work.')


# Update connection string information
# DB credentials
conn_string = cfg.getSubDbConn()

#Sub pvt encryption key
pvt_key = cfg.getSubPvtEncryptKey()

# Daaframe should return only one record
# Date needs to be authenticated
# Pages against the limit need to be authenticated

def extraction(encryptedInput):

    print("Entered function")
    auth_info = {}
    auth_info['auth_token'] = None
    auth_info['success'] = 0
    auth_info["auth_message"] = "Invalid Subscription or invalid input"
    auth_info["exp_time"] = datetime.now()
    print("variables initialized")

    def triggerExtractionSubmit(auth_token,
                                sub_id,
                                document_id,
                                messageBody):
        try:
            subscription_info = {}
            subscription_info['auth_token'] = auth_token
            subscription_info['sub_id'] = sub_id
            subscription_info["document_id"] = document_id
            subscription_info["body"] = messageBody
            subscription_info['stage'] = "submit"
            subscription_info['create_time'] = str(datetime.now())
            subscription_info['success'] = 1
            subscription_info['is_start'] = 1
            subscription_info['is_end'] = 0
            os.makedirs(files_dir,exist_ok = True)
            json_path = os.path.join(files_dir,
                                     str(auth_token) + "__submit.json")
    
            sub_json = json.dumps(subscription_info)
            with open(json_path, "w") as f:
                f.write(sub_json)
                print("File successfully written, ", json_path)
    
            # Blob trigger will initiate the actual extraction
            uploaded = util.uploadFilesToBlobStore(sub_id, json_path)
            try:
                os.remove(json_path)
            except:
                pass
            return uploaded
        except:
            print(traceback.print_exc())
            return False


    def updateAuthInDb(auth_token,
                       sub_id,
                       document_id,
                       auth_time,
                       exp_time,
                       no_of_pages
                       ):
        try:
            with psycopg2.connect(conn_string) as conn:
                with conn.cursor() as cursor:
                    ins_string = "INSERT INTO subscription_auth_tokens "
                    ins_string += "(sub_id, call_id,auth_token,authenticated_time, expiration_time) "
                    ins_string += "VALUES ('"+sub_id + \
                        "','"+document_id+"','"+auth_token
                    ins_string += "',TIMESTAMP '" + \
                        str(auth_time)+"',TIMESTAMP '"+str(exp_time)+"')"
                    print(ins_string)
                    print("Insert to auth_tokens extraction submit", ins_string)
                    cursor.execute(ins_string)
    
                    ins_string = "INSERT INTO subscription_call_records"
                    ins_string += "(auth_token,status,modified_time,pages_requested)"
                    ins_string += "VALUES ('"+auth_token+"',-3,TIMESTAMP '"
                    ins_string += str(auth_time)+"',"+str(no_of_pages)+")"
                    print(ins_string)
                    print("Insert to auth_tokens extraction submit", ins_string)
                    cursor.execute(ins_string)
    
                    conn.commit()
                return True
        except:
            print(traceback.print_exc())
            return False

    def returnFailure():
        auth_info = {}
        auth_info['auth_token'] = None
        auth_info['status_code'] = 404
        auth_info["auth_message"] = "Invalid Subscription or invalid input"
        auth_info["exp_time"] = str(datetime.now())
        return json.dumps(auth_info)

    # Message will be encrypted by a secret key for a subscriber

    def auth_subscriber(sub_id, mac_id):
        try:
            select_sql = f"SELECT sub_id FROM allowed_machines where sub_id = '{sub_id}' and mac_id = '{mac_id}' "
            with psycopg2.connect(conn_string) as conn:
                with conn.cursor() as cursor:
                    cursor.execute(select_sql)
                    DF = pd.DataFrame(cursor.fetchall(), columns=["sub_id"])
                    if DF.shape[0] >= 1:
                        sub_id = DF['sub_id'].iloc[0]
                        return sub_id
                    else:
                        return None

        except:
            print(traceback.print_exc())
            return None

    def check_subscriber_limit(sub_id_, no_of_pages):
        def check_validity(df):
            if df.shape[0] != 1:
                print("Corrupted dataset")
                return False, 100

            # Authenticate date
            df['date'] = pd.to_datetime(df['end_date'],
                                        yearfirst=True)
            print(df['date'], "data")

            if df[df['date'] > pd.Timestamp.now()].shape[0] != 1:
                print("Outdated subscription")
                return False, 101
            df["input_pages"] = no_of_pages
            print(df["input_pages"], "pages")
            # Authenticate pages
            df['page_diff'] = df['allotted'].sub(df['pages_requested'], axis=0)

            df["page_diff"] = df["page_diff"].sub(df["input_pages"],
                                                  axis=0)
            print(df["page_diff"], "xxx")
            if df['page_diff'].iloc[0] <= 0:
                print("Request limit exhausted")
                return False, 102
            return True, 1

        try:
            select_sql = "SELECT * FROM auth_by_sub_id where sub_id = '{}'".format(
                sub_id_)
            with psycopg2.connect(conn_string) as conn:
                with conn.cursor() as cursor:
                    cursor.execute(select_sql)
                    DF = pd.DataFrame(cursor.fetchall(),
                                      columns=["sub_id",
                                               "pages_requested",
                                               "allotted",
                                               "end_date"])
                    print(DF, "DF")
                    valid, err = check_validity(DF)
    
                    return valid, err
        except:
            return False, -100  # Connection error

    def generate_auth_token(sub_id, no_of_pages):
        try:
            auth_token = str(uuid.uuid4())
            n = 2 * no_of_pages  # No of minutes to be taken from config file
            auth_time = datetime.now()
            exp_time = auth_time + timedelta(minutes=n)
            return auth_token, auth_time, exp_time, timedelta(minutes=n)
        except:
            print(traceback.print_exc())
            return None, None, None, None

    try:
        # Input is an encrypted message:
        print("entered try" + str(encryptedInput))

        enc_message = json.loads(encryptedInput)
        print("json loads done")
        enc_message = enc_message["message"]
        # Decrypt the message
        print("before decryption")
        message = util.decryptMessage(enc_message)
        print("after decryption")
        message = json.loads(message)
        print("after json loads")
        print(message, "msg")
        print("decryption done")

        # Subscriber ID comes as encrypted message along with mac address and datetime
        inputHash = message["input_hash"]
        # Decrypt Subscriber ID
        hash_msg = util.decryptMessage(inputHash)
        print("decryption of sub id done" + str(hash_msg))

        # sub-id__macaddess__datetime
        sub_id, mac_id, _ = [*hash_msg.split("__")]
        print("unpacked sub id" + str(sub_id))

        #pages = message["pages"]
        request = message.get("request")
        pages = request.get("pages")
        print("unpacked pages" + str(pages))
        document_id = message["documentId"]

        # Create logging based on document_id
        print("Before logging")
        std_out = sys.stdout
        std_err = sys.stderr
        logfilepath = str(document_id) + "_auth.log"
        logfile = open(logfilepath,"w")
        sys.stdout = logfile
        sys.stderr = logfile

        print("unpacked document id" + str(document_id))
        no_of_pages = len(pages)
        print("unpacked message")

        # Check if subscriber is valid by checking the db entry
        sub_id = auth_subscriber(sub_id, mac_id)
        print("sub auth done")

        print(sub_id, "sub_id")
        if sub_id is None:
            # return a json with auth-token:none,authenticated:false,err:"Invalid subscription id"
            print("Failed in auth_sub")
            return returnFailure()

        # Check the page limit available for the subscriber
        authenticated, err = check_subscriber_limit(sub_id,
                                                    no_of_pages)
        print("check_sub_limit success")
        #print("Checking subscriber limit: ", authenticated,err)

        if not authenticated:
            auth_info['auth_message'] = "Subscription expired"
            # return a json with auth-token:none,authenticated:false,err:<appropriate error message>
            print("auth failed")
            return returnFailure()

        # Authentication is successful, generate an auth token
        print("before generate auth token")
        auth_token, auth_time, exp_time, allotted_time = generate_auth_token(sub_id,
                                                                             no_of_pages)
        print("after generate auth token")

        if auth_token is None:
            print("generate auth token failed")
            return returnFailure()
        print("generate auth token success")
        messageBody = json.dumps({"request": request})
        print("generate message body for extraction submit")

        # Trigger Extraction submission by adding a blob trigger
        triggered = triggerExtractionSubmit(auth_token,
                                            sub_id,
                                            document_id,
                                            messageBody)
        if not triggered:
            print("trigger ext submit failed")
            return returnFailure()

        print("trigger ext submit success")

        # Update subscription db that extraction has been triggered
        updated = updateAuthInDb(auth_token,
                                 sub_id,
                                 document_id,
                                 auth_time,
                                 exp_time,
                                 no_of_pages)
        if not updated:
            print("Update Auth In Db failed")
            return returnFailure()

        print("Update Auth In Db success")

        return json.dumps({"auth_token": auth_token,
                           "status_code": 200,
                           "message": "Extraction submitted",
                           "allotted_time": str(allotted_time)})

    except:
        print(traceback.print_exc())
        return returnFailure()
    finally:
        if not logfile.closed:
            logfile.close()
            sys.stdout = std_out
            sys.stderr = std_err
            uploaded, URI = util.uploadFilesToBlobStore(sub_id,
                                                        logfilepath)
            # try:
            #     os.remove(logfilepath)
            # except:
            #     pass
            if uploaded:
                print("AUTH token log file uploaded to Blob")


def main():
    input = '{"message": "gAAAAABhcOz5_oFfk58rELwMIzA8nhGbBjON6lL0BMgQW8nCP9yoI5F6lvLbsDNQ3xrzyit4TNqxnVj5y9DJFCigTvqS8QT7DkFpaTT6EZj2ardOfWgpE4v082mbsAKdD8TYB9pX3xk4Da67JgY2QZkFUt7a4r2eOrQJTjKhQh2R2sBb75sUWx6TtEsAeAFuJmlwqwU7njhgryRhbz9rNdeTC3QBHI8830Gf-BkYY-M4kc3O8LJ3TAn-96eGJotScKOwnEP_mWXTpZcMrlgMrQb4dEY89goUjB9KfRh9UM9Pr5jYbd8iJjRXaO-5IA4TDMHA5-TITXwWuYm6pImR_UbLcv8lXYK5q_rTyD6gijQ4fFfkrm6MTvlTJE4G3j_EeBfWx7So6Uiq22a5EatguxbatimG4XqTi7oznIqxWNnO7VtDDXNe7Q__Bb7S54D3iYU4DineQDoh56-XMqt8og7OM3_tA_Q4vmcmILFwUeaWYjyVMRXjSMCN2olZOSom36AvcldDb6mF-ffvb32QmFE-fCZ6QWrLugi6rutIYLUC0E0YoctvTUWYsVdXhabFqml-z8jxW5YToZKg70jyR1h3gZ7X6tTdS-zPzvkQVU3ButtokSgF2SQZv_gWg8Dfpz-zYpQDs04ur3MqHTS9krL6XNUrlzZbWMMbTP-DWTuJ5UNoR8hLuhuMnzA374U7NoetlfhoU06ycJPxMwDVXqZIyoITGAiW6kBuO3yCo1dKliAslV9bsT3mbt8-9_EMX2BUQtkFBYr69gBUeNp-rQCqLx4t7iT64VBSlYve9qV0CTWJwkGN50yXyp3Ej10J6NlHSEIeVtAHHeAOXuLGnQyOoTgMZyJ9x0GeXJuLtSny42Iy31jOyWJUlTJvSPhurgVMojQhxvl4pIWWkiHng6SSKkuvsHU28tTXTBpMMcApcaCy6xIH4c-mprsDr_ZKI1FCyumQszbM31gn7Mf2pLSlbmbrAHwrWOj5Nb4hNuOtolWnZWoh7gu5tloRGR8icT5tb54SzjL0lTVpURxY8nX0yoxaTCkFCUX2NG-jnH_IEjwiC0rKctfab0Ea_fu_BGfwJ77O0ZDHCcWOCX0SB_aAXLGg_pzKLivBsUvAuKjyxAN7RtqTycYspTL8VlrPt4G0sBTq-HW07X6cXvyrpzpT9EMdP1NoEwslsZA2nJ9dEMxOvn2sKE7motm92ZkNVTrwpGUr79vXjS5xBNPYuLrQSERNXzxQ8rqCWfw5LZ8tJugDK9-Hmvop7iqScherq2Hrjwz1nQuBPJB01euvEllBgTIj9bhhzJ5pVXb5fAX6UDvPh_Q960uCQNL1otyKnP1ifxJLwC3e8TbD7Zo0wTyT1rO2fE1MmL-CmVx_ijg9m37KZiVJ5KNSCCp4Xzkc7dYs7p36b_UoNwcsMsCiUuZn99AbAXl2Crw4RNGLHGexA-pVL4dnbnZaL1zx6uUEBx5mTqUXTbSA16VtPPDQCTQ-He_zuNeg5lKWI8vigBcVplGiyvG3bMdEGU-vnvkOM3t7RtlRUrLFl3cffNQgA3wtbo-fiJSB7RDkXey0pOwZlV-EpGp9xhE0rj4xKW-DF4nBjHx23lNMA07YpmGISCoxJGgcKxdZE7ywORK89wLaapSm15gQGL1PBM4VcMGPnn8HuIt3bweVOm_0Ri8DK8vgz6PXgGi0Z_FlOqqP3X95Vcgj4v5YenAiiWMDxi31mgtUDxlWNTdpBqD9QTURzuqyE1OKP-EA9WOgG6EjwMaiZQ0zBzKTeO0FhSn5oFRyexQdsPX6W_oKn3HxF9jzHB-ernGdzXa_c_mJzj1utaf1HMsZe7lpdCyT0TO55wZ7bylZcTKgWCEmR5F1b2CQUFAp3GRGIzt6lg=="}'
    extract = extraction(input)
