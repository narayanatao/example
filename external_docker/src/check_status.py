#!/usr/bin/env python
# coding: utf-8

import psycopg2
import pandas as pd
import traceback
import json
import config as cfg
import util as util
import sys


#DB credentials
conn_string = cfg.getSubDbConn()

def check_status_DB(decrypted_auth_token):
    try:
        with psycopg2.connect(conn_string) as conn:
            with conn.cursor() as cursor:
                cursor.execute(f"SELECT status from subscription_call_records WHERE auth_token = '{decrypted_auth_token}'")
                DF = pd.DataFrame(cursor.fetchall(),columns = ["status"])
                if DF.shape[0] > 0:
                    status = DF['status'].iloc[0]
                    return status
                else:
                    return None
    except:
        print(traceback.print_exc())
        return None


def status_json(decrypted_auth_token,status):
    try:
        # result = None
        if status == 0:
            with psycopg2.connect(conn_string) as conn:
                with conn.cursor() as cursor:
                    cursor.execute(f"SELECT body from stage_result where stage = 'extraction' and auth_token = '{decrypted_auth_token}'")
                    DF = pd.DataFrame(cursor.fetchall(),columns = ["body"])
                    if DF.shape[0]>=1:
                        body = DF['body'].iloc[0]
                        if isinstance(body,dict):
                            body = json.dumps(body)
                        print("body:",body)
                        body = json.loads(body)
                        message = util.encrypt_message(
                            json.dumps({"status_code":200,
                                        "status":"Extracted",
                                        "auth_token": decrypted_auth_token,
                                        "result":body,
                                        "error_message":""}
                                       ))
                        print("Extraction done")
                        return message
                    else:
                        print("Still processing")
                        message = util.encrypt_message(
                            json.dumps(
                                {"status_code":200,
                                 "status":"Processing",
                                 "auth_token": decrypted_auth_token,
                                 "result":"","error_message":""
                                 }
                                ))
                        return message
        elif status == -1:
            print("Extraction Failed")
            message = util.encrypt_message(
                json.dumps(
                    {"status_code":200,
                     "status": "Failed",
                     "auth_token": decrypted_auth_token,
                     "result":"",
                     "error_message":"Failed in extraction"
                     }
                    ))
            print("Message", message)
            return message
        elif status == -2:
            print("Extraction Expired")
            message = util.encrypt_message(
                json.dumps(
                    {"status_code":200,
                     "status": "Expired",
                     "auth_token": decrypted_auth_token,
                     "result":"",
                     "error_message":"Authentication token expired"
                     }
                    ))
            # result = json.dumps({"message":message})
            return message
        elif status == -3:
            print("Extraction not started")
            message = util.encrypt_message(
                json.dumps(
                    {"status_code":200,
                     "status": "Submitted",
                     "auth_token": decrypted_auth_token,
                     "result":"",
                     "error_message":"Submitted, but extraction not started"
                     }
                    ))
            return message
        
        elif status == 1:
            print("Extraction Success")
            message = util.encrypt_message(
                json.dumps(
                    {"status_code":200,
                     "status": "Success",
                     "auth_token": decrypted_auth_token,
                     "result":"",
                     "error_message":""
                     }
                    ))
            return message
    except:
        print(traceback.print_exc())
        return None

def returnFailure(auth_token,errCode,errMsg):
    try:
        message = util.encrypt_message(json.dumps({"status_code":500,
                                                   "auth_token":auth_token,
                                                   "errCode":errCode,
                                                   "errMsg":errMsg,
                                                   "result":""}))
        return message
        # return {"message":message}
    except:
        print(traceback.print_exc())
        return None

def status_recovery(input_json):

    decrypted_auth_token = ""
    logfile = None

    try:
        
        message=json.loads(input_json)
        enc_message = message["message"]

        decrypted_json = util.decryptMessage(enc_message)
        if decrypted_json is None:
            return returnFailure("",
                                 "DECRYPT",
                                 "Error in decrypting the input message")
        print("Decryption done")

        decrypted_json=json.loads(decrypted_json)

        print(str(decrypted_json))

        decrypted_auth_token=decrypted_json["auth_token"]
        documentId = decrypted_json["documentId"]
        sub_id = decrypted_json["sub_id"]

        print(str(decrypted_auth_token))
        print("unpacked decrypted auth token")
        print("Before logging")
        std_out = sys.stdout
        std_err = sys.stderr
        logfilepath = str(decrypted_auth_token) + "_auth.log"
        logfile = open(logfilepath,"w")
        sys.stdout = logfile
        sys.stderr = logfile

        print(decrypted_auth_token,"decrypted")

        status=check_status_DB(decrypted_auth_token)
        if status is None:
            return returnFailure(decrypted_auth_token,
                                 "DBCHECK",
                                 "Error in retrieving status from DB")

        print("Check Status DB")

        status_returned = status_json(decrypted_auth_token,
                                      status)
        if status_returned is None:
            return returnFailure(decrypted_auth_token,
                                 "Failed to get results back")
            
        print("Status returned")
        print(str(status_returned))

        return status_returned
    except:
        print("status_recovery",
              traceback.print_exc())
        return returnFailure("",
                             "EXCEPTION",
                             "Failed during extracting results")
    finally:
        if logfile is not None:
            if not logfile.closed:
                logfile.close()
                sys.stdout = std_out
                sys.stderr = std_err
                uploaded, URI = util.uploadFilesToBlobStore(sub_id,
                                                            logfilepath)
                if uploaded:
                    print("Auth status log file uploaded to Blob")
        else:
            return json.dumps({"status_code":404,
                               "auth_token": "",
                               "error_message":"Failed during api call",
                               "result": ""})


if __name__ == "__main__":
    inp_json = {'message': 'gAAAAABiRflyISuD2Mc40pzYNUw4g3U0tNVHN2-tSk639dlxRv4Q4Ue1GHX5HVzNbmGJQn81kQLBcGYdvmWOEh3g0Tiw_zP5GoTThr_BrpwwQW1VPjmGw-Iy0UMt2RC1-3KQdxbqdUbpv_cHwxk4AWYVBAdmlZezxg=='}
    # input_json=json.dumps({"message": "gAAAAABhlkqcyiwygkr7dtCY_2bCz0f2syUk-Yc3LkXGFLQKE543gTRdEPrExplV0OIm8dA1YmCMVHR7Cp5uTMFyXZPvgLX5klswcCh9ucBdqeMIr1iio1f1XIu8aHlodOTqBtjfzSBbkHDrlCbdD7DhZ_ErlpzNyg=="})
    input_json=json.dumps(inp_json)
    status_json_return=status_recovery(input_json)
    print(status_json_return," Successfully Returned")
