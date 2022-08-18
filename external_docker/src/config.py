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

# In[] - IP Address and Port

def getApiPort():
    return configdata["EXTERNAL_API_PORT"]

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
