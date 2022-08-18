# -*- coding: utf-8 -*-
# In[1]: - All Imports

import json
import time
from requests import get, post
import os

import config as config
import util as util
import traceback

# In[1]: - Variable declarations

rootPath = config.getRootFolderPath()
mime_tiff = config.getMimeTiff()
tempFolderPath = config.getTempFolderPath()

# OCR URL and access key
endpoint = config.getFormRecognizerEndpoint()
apim_key = config.getFormRecognizerAPIKey()
post_url = endpoint + config.getFormRecognizerPostURL()

# OCR wait time and number of tries
no_tries = config.getOcrTries()
ocr_wait_time = config.getOcrWaitTime()
blobAccount = config.getBlobAccountName()


# In[1]: - Define functions
@util.timing
def get_analyseLayout(docInput,
                      sub_id = None):

    headers = {
        # Request headers
        'Content-Type': mime_tiff,
        'Ocp-Apim-Subscription-Key': apim_key,
    }
    resp_code = 404
    resp_status = "Failed"
    resp_body = ""
    resp_final = {"response_code":resp_code,
                  "response_status":resp_status,
                  "response_body":resp_body}
    imgpath = docInput.get("file_path")
    container = docInput.get('container')
    img_az_func_path = os.path.join(tempFolderPath,
                                    os.path.basename(imgpath))
    print("Checking ocr input: ", imgpath, img_az_func_path)
    downloaded = util.downloadFilesFromBlobStore(container,
                                                 imgpath,
                                                 img_az_func_path)
    if not downloaded:
        return None

    #Convert the image to bytes to input to the OCR API
    with open(img_az_func_path, "rb") as f:
        data_bytes = f.read()

    #Call the POST method to Azure OCR API to register the file for OCR
    try:
        t = time.time()
        resp = post(url = post_url,
                    data = data_bytes,
                    headers = headers)
        td = time.time() - t
        if resp.status_code != 202:
            print("POST analyze failed:\n%s" % resp.text)
            quit()
        print("POST analyze succeeded:\n%s" % resp.headers)
        get_url = resp.headers["operation-location"]
        print("Time taken for submitting the request to OCR",td)
        print("OCR location",get_url)
    except Exception as e:
        print("POST analyze failed:\n%s" % str(e))
        return json.dumps(resp_final)

    n_try = 0
    t = time.time()
    while n_try < no_tries:
        try:
            resp = get(url = get_url,
                       headers = {"Ocp-Apim-Subscription-Key": apim_key})
            resp_json = json.loads(resp.text)
            status_code = resp.status_code
            status = resp_json["status"]
            resp_final["response_code"] = status_code
            resp_final["response_status"] = status
            if status_code != 200:
                print("GET Layout results failed:\n%s" % resp_json)

            if status == "succeeded":
                print("Layout Analysis succeeded:\n%s" % imgpath)
                resp_final["response_body"] = resp_json
                break
            if status == "failed":
                print("Layout Analysis failed:\n%s" % resp_json)
            # Analysis still running. Wait and retry.
            # time.sleep(ocr_wait_time)
            n_try += 1
        except Exception as e:
            msg = "GET analyze results failed:\n%s" % str(e)
            print("ERROR:",msg)
            break
    td = time.time() - t
    print("Time taken for OCR extraction to be completed",td)

    return json.dumps(resp_final)

# In[Fnction I/P and O/P related]

# def get_response_struct():
#     resp_code = 404
#     resp_status = "Failed"
#     resp_body = ""
#     resp_final = {"response_code":resp_code,
#                   "response_status":resp_status,
#                   "response_body":resp_body}
#     return resp_final

# def parse_doc_input(docInput):
#     blobPath = docInput.get("file_path")
#     container = docInput.get('container')
#     imgPath = os.path.join(tempFolderPath,
#                                     os.path.basename(blobPath))
#     return blobPath,container,imgPath

# def download_file(info):
#     imgPath = info.get("imgPath")
#     container = info.get("container")
#     blobPath = info.get("blobPath")
#     downloaded = util.downloadFilesFromBlobStore(container,
#                                                  imgPath,
#                                                  blobPath)
#     if not downloaded:
#         return False
#     else:
#         return True


# In[Deskew related]

@util.timing
def check_skewness(ocr_out):
    ocr_op = ocr_out['analyzeResult']['readResults'][0]
    page_angle = ocr_op['angle']
    return page_angle

@util.timing
def rotate_page(imgPath,page_angle):
    rotated = False
    if abs(page_angle) > 0.05:
        deskew = util.rotateImageByAngle(imgPath,
                                         (-1 * page_angle))
        rotated = deskew is not None
    return rotated

# In[Azure ocr related]

# OCR URL and access key
endpoint = config.getFormRecognizerEndpoint()
apim_key = config.getFormRecognizerAPIKey()
post_url = endpoint + config.getFormRecognizerPostURL()

# OCR wait time and number of tries
no_tries = config.getOcrTries()
ocr_wait_time = config.getOcrWaitTime()
blobAccount = config.getBlobAccountName()

# @util.timing
def submit_ocr(imgPath):

    headers = {
        # Request headers
        'Content-Type': mime_tiff,
        'Ocp-Apim-Subscription-Key': apim_key,
    }

    #Call the POST method to Azure OCR API to register the file for OCR
    try:
        #Convert the image to bytes to input to the OCR API
        with open(imgPath, "rb") as f:
            data_bytes = f.read()
        resp = post(url = post_url,
                    data = data_bytes,
                    headers = headers)
        if resp.status_code != 202:
            print("OCR failed:\n%s" % resp.text)
            return None
        print("POST analyze succeeded:\n%s" % resp.headers)
        get_url = resp.headers["operation-location"]
        return get_url
    except:
        print("POST analyze failed:\n%s",
              traceback.print_exc())
        return None

# @util.timing
def get_ocr(get_url):

    n_try = 0
    while n_try < 200:
        try:
            resp = get(url = get_url,
                       headers = {"Ocp-Apim-Subscription-Key": apim_key})
            status_code = resp.status_code
            #Jul 12 2022 - exit if ocr fails
            if status_code != 200:
                print("GET Layout results failed")
                return None
            #Jul 12 2022 - exit if ocr fails
            resp_json = json.loads(resp.text)
            status = resp_json["status"]
            if status_code != 200:
                print("GET Layout results failed:\n%s" % resp_json)
                return None

            if status == "succeeded":
                print("Layout Analysis succeeded:\n%s")
                return resp_json
            if status == "failed":
                print("Layout Analysis failed:\n%s" % resp_json)
                return None
            time.sleep(5)
            n_try += 1
        except:
            print("Azure OCR check_status:",
                  traceback.print_exc())
            return None
    return None

# @util.timing
def execute_ocr(imgPath):

    try:
        print("time before ocr request submitted",time.time())
        t = time.time()
        get_url = submit_ocr(imgPath)
        print("time after ocr request subitted successfully", time.time(),
              "time diff = ", time.time() - t)
        if get_url is None:
            return None

        t = time.time()
        ocr_out = get_ocr(get_url)
        #Jul 12 2022 if output is None, return immediately
        if ocr_out is None:
            print("time after ocr didn't extract any o/p", time.time(),
                  "time diff = ", time.time() - t)
            return None
        #Jul 12 2022 if output is None, return immediately

        print("time after ocr extracted successfully", time.time(),
              "time diff = ", time.time() - t)
        if ocr_out is None:
            return None

        page_angle = check_skewness(ocr_out)
        #Jul 12 2022 add print statements
        print("Check Skewness completed")
        #Jul 12 2022 add print statements

        if rotate_page(imgPath, page_angle):

            #Jul 12 2022 add print statements
            print("time before ocr request submitted",time.time())
            #Jul 12 2022 add print statements

            #Jul 12 2022 add print statements
            t = time.time()
            #Jul 12 2022 add print statements
            get_url = submit_ocr(imgPath)
            if get_url is None:
                #Jul 12 2022 add print statements
                print("time after ocr request submission failed", time.time(),
                      "time diff = ", time.time() - t)
                #Jul 12 2022 add print statements
                return None

            #Jul 12 2022 add print statements
            print("time after ocr request subitted successfully", time.time(),
                  "time diff = ", time.time() - t)
            #Jul 12 2022 add print statements

            #Jul 12 2022 add print statements
            t = time.time()
            #Jul 12 2022 add print statements
            ocr_out = get_ocr(get_url)
            if ocr_out is None:
                #Jul 12 2022 add print statements
                print("time after ocr didn't extract any o/p", time.time(),
                      "time diff = ", time.time() - t)
                #Jul 12 2022 add print statements
                return None
            #Jul 12 2022 add print statements
            print("time after ocr extracted successfully", time.time(),
                  "time diff = ", time.time() - t)
            #Jul 12 2022 add print statements

        return ocr_out

    except:
        print("execute_ocr",
              traceback.print_exc())
        return None

#@util.timing
def ocr_images(imgPaths):
    from multiprocessing import Pool
    # try:
    #     set_start_method("spawn")
    # except:
    #     pass
    pool_size = min(len(imgPaths),10)
    pool = Pool(pool_size)
    res = pool.map_async(execute_ocr,
                         (imgpath for imgpath in imgPaths))
    pool.close()
    pool.join()
    ocr_ops = res.get()
    pool.terminate()
    return ocr_ops
