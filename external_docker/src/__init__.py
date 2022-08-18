# -*- coding: utf-8 -*-
"""
Created on Fri Oct 22 22:22:51 2021

@author: Hari
"""

from klein import Klein
import config as cfg
import get_sas_token as sas
import auth_subscription as sub_auth
import check_status as status_auth
import json
import datetime

app = Klein()
appPort = cfg.getApiPort()

@app.route('/authtoken/status/get', methods=['POST'])
def auth_get_status(request):
    try:
        rawContent = request.content.read()
        encodedContent = rawContent.decode("utf-8")
        return status_auth.status_recovery(encodedContent)
    except:
        return json.dumps({"status_code":404,
                           "auth_token": "",
                           "error_message":"Failed during api call",
                           "result": ""})

@app.route('/authtoken/generate', methods=['POST'])
def auth_generate(request):
    try:
        rawContent = request.content.read()
        encodedContent = rawContent.decode("utf-8")
        return sub_auth.extraction(encodedContent)
    except:
        auth_info = {}
        auth_info['auth_token'] = None
        auth_info['status_code'] = 404
        auth_info["auth_message"] = "Invalid Subscription or invalid input"
        auth_info["exp_time"] = str(datetime.now())
        return json.dumps(auth_info)

@app.route('/blob/sastoken/generate', methods=['POST'])
def blob_sastoken_generate(request):
    try:
        rawContent = request.content.read()
        encodedContent = rawContent.decode("utf-8")
        return sas.get_SAS_token(encodedContent)
    except:
        return json.dumps({"status":404})


if __name__ == "__main__":
    app.run("0.0.0.0", appPort)

