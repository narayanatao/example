#!/usr/bin/env python
# coding: utf-8

import traceback
import json
import config as cfg
import util as util
import sys
from time import time, ctime
import os

# In[293]:
# DB credentials
conn_string = cfg.getSubDbConn()

# In[295]:
def returnFailure():
    return json.dumps({"status":404})

# In[296]:

#encrypt message to return json

def get_SAS_token(input_json):
    print("Before logging")
    std_out = sys.stdout
    std_err = sys.stderr
    logfilepath = str(ctime(time())) + "_sas_token.log"
    logfile = open(logfilepath,"w")
    sys.stdout = logfile
    sys.stderr = logfile

    try:
        print("Inside cli debugger")
        print(str(input_json))

        enc_message = json.loads(input_json)
        enc_message = enc_message["message"]

        print("Encrypted message")
        print(str(enc_message))

        #decrypts the encrypted message
        decrypted_input_json = util.decryptMessage(enc_message)

        print("Decrypted message")
        print(decrypted_input_json)

        print(decrypted_input_json,"decrypted message")

        decrypted_input_json=json.loads(decrypted_input_json)
        print("Decrypted json")
        print(str(decrypted_input_json))
        inputHash = decrypted_input_json["input_hash"]
        print("input hash")
        print(str(inputHash))

        hash_msg = util.decryptMessage(inputHash)
        print("input hash decrypt")
        print(str(hash_msg))
        print("Input hash", hash_msg)
        sub_id,mac_id,_=[*hash_msg.split("__")]
        print("input decrypt")
        print(sub_id)
        activity = decrypted_input_json["activity"]

        IP = decrypted_input_json.get("IP")
        file_size=decrypted_input_json.get("file_size")
        print("input activity, ip, file_size")
        print(str(activity))

        print(sub_id,
              mac_id,
              activity,
              IP,
              file_size,
              "Values of decrypted json")

        account_url,credential,container_name = util.gen_sas_token(sub_id,
                                                                   mac_id,
                                                                   IP,
                                                                   activity)
        print("after generate sas token")
        if account_url is None:
            return returnFailure()

        input_json=json.dumps({"sas_token":credential,
                               "account_url":account_url,
                               "container":container_name})

        returnMessage = util.encrypt_message(input_json)
        return returnMessage

    except:
        print("get_SAS_token",traceback.print_exc())
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
                print("SAS token log file uploaded to Blob")


if __name__ == "__main__":
    # encrypted_input_json=main()
    enc_message = {"message": "gAAAAABhX9Wdbh5J-ZL11FQREdTxUhmUUWVkp0jQEgK9PaRFKAVhf6d9ZsKqQY3b9xxdK3ioMeFiq5shklNxfp-J54k76McKSMInstLZ2qt_z_SNaf1YH90u988ZzKd1cS5eWjQBq2flyOX2KKyNmv_eIZUGn4rgi9T3s3_ZEU0GTZWeHWxN4AMYvygNXe-NuQioAfBW-nb5FYKSHTdISb065LtHn3A4LjoT4C__7FDlSc7W8a02gYVgakQvopMQ16Nhs3IJT3PJl2dkB9Fe4703sxlOEqEsl0iwiAONPDkp736pJMMKuri-aD5gsNuGaNnNJn0vUk5nd7Hm7xOlAZcfn5Nxyb0s-8aCfmE2MtsTcV9goXNDI1z8KA8PW4il-leSKlEZlWgt1M0mG_4fQEXOxwFMI0i-JezSLPOzDZTUvnR0s9Mwutw="}
    enc_message = json.dumps(enc_message)
    encrpted_msg_to_client=get_SAS_token(enc_message)
