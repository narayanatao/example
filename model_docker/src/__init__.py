# -*- coding: utf-8 -*-
"""
Created on Fri Oct 22 22:22:51 2021

@author: Hari
"""

from klein import Klein
import config
import json

import predict_dataset as pred

app = Klein()
appPort = config.getModelApiPort()

@app.route('/paiges/model/predict', methods=['POST'])
def predict(request):
    try:
        rawContent = request.content.read()
        encodedContent = rawContent.decode("utf-8")
        print(".....")
        print(request)
        print()
        print(encodedContent)
        print(type(encodedContent))
        return pred.predict(json.loads(encodedContent))
    except:
        return json.dumps({"status":"Error in calling model predict"})

if __name__ == "__main__":
    app.run("0.0.0.0", appPort)

