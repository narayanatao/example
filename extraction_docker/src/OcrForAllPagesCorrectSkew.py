# -*- coding: utf-8 -*-
"""
Created on Tue Aug 24 18:12:17 2021

@author: Hari
"""
# In[1]: - Importing required packages
import traceback
import os
from PIL import Image
import json
import warnings
import sys
import azureOcr as ocr
import config as config
import util as util

warnings.filterwarnings("ignore")

# In[1]: - Util functions specific to OCR function

#get script directory
script_dir = os.path.dirname(__file__)
print("Script Directory:", script_dir)



# In[2]: - Defining the file paths for model, fb model, vectorDimensionalimit

root_folder = config.getRootFolderPath()
outputFolder = config.getBlobOPFolder()
tempFolderPath = config.getTempFolderPath()


#Get config for blob store or folder
blobAccount = config.getBlobAccountName()
blobOutputFolder = config.getBlobOPFolder()
blobPreproc = config.getBlobStorePreproc()

#Mime Types and Extensions
mimePng = config.getMimePng()
mimeJson = config.getMimeJson()
mimeTiff = config.getMimeTiff()
extnTiff = config.getExtnTiff()

#Service URL
# ocr_url = config.getAzFuncOcrURL()

# ocr_url = "https://azocrfunc.azurewebsites.net/api/AzureOCRFunc?"
# In[1]: - Define Logging Function

def logClose(logfile):
    try:
        if not logfile.closed:
            logfile.close()
    except Exception as e:
        print("Logging Issue: \n",traceback.print_exc(), e)
        pass
    
def logAppend(logfilepath):
    try:
        logfile = open(logfilepath,"a")
        sys.stdout = logfile
        sys.stderr = logfile
        return logfile
    except:
        return None

# In[4]: - Define functions

@util.timing
def correct_skewness(dict_file_pages, ocr_out, container, docInput):
    """
	Method added to identify skeweness and correct it 
    """
    print("Inside correct_skewness")
    skewness_corrected = False
    print("Doc Input:", docInput)
    extn = docInput.get("extn")
    orgFileLocation = docInput.get("orgFileLocation")
    # imgpath = docInput.get("file_path")
    # print(extn, orgFileLocation, imgpath, os.path.basename(imgpath))
    rotation = False

    for page_result in ocr_out['analyzeResult']['readResults']:
        page_num = page_result['page']
        page_angle = page_result['angle']
        abs_page_angle = abs(page_angle)
        try:
            if abs(abs_page_angle) > 0.0: # Check for the abs(angle) >= 0.5
                identified_page = [p for p in dict_file_pages 
                                   if dict_file_pages[p].get('document_page_number') == str(page_num-1)][0]
                img_file_path = dict_file_pages[identified_page]['file_path']
                png_file_path = dict_file_pages[identified_page]['png_path']
                png_local_path = os.path.join(os.path.split(img_file_path)[0],
                                              os.path.split(png_file_path)[1])
                print("Skewness Identified in page:", img_file_path, png_local_path)
                deskew_check = util.rotateImageByAngle(img_file_path,(-1 * page_angle))
                if deskew_check:
                    uploaded, URI = util.uploadFilesToBlobStore(container,
                                                                os.path.basename(img_file_path))
                    skewness_corrected = True
                    if uploaded:
                        rotation = True
                    png_local_path = util.converTiffToPng(img_file_path,
                                                           png_local_path)
                    if png_local_path is not None:
                        print("Deskewed Image updated to: ", png_local_path)
                        uploaded, URI = util.uploadFilesToBlobStore(container,
                                                                os.path.basename(png_local_path))
                        if uploaded:
                            os.remove(png_local_path)
                        else:
                            print("Failed to upload")
                else:
                    print("Error in rotation")
        except Exception as e:
            print("correct_skewness",
                  traceback.print_exc())
            pass
    # Check extn, if extn == pdf: 
    # Call a function for combining tiff to pdf and write it back to orgFileLocation
    if ".pdf" in extn.lower() and orgFileLocation and rotation:
        images = []
        for page_num in dict_file_pages.keys():
            images.append(Image.open(dict_file_pages[page_num]['file_path']))
        im1 = images[0]
        local_pdf_path = os.path.join(tempFolderPath,
                                      os.path.basename(orgFileLocation))
        im1.save(local_pdf_path,
                 "PDF",
                 resolution=100.0,
                 save_all=True,
                 append_images=images[1:])
        # print("Reformed PDF uploaded to: ", local_pdf_path)
        uploaded, URI = util.uploadFilesToBlobStore(container, local_pdf_path)
    return skewness_corrected

# In[5]: - Run OCR for all pages

@util.timing
def combine_document_run_ocr(docInput,
                             check_skewness = True):
    """
    """
    try:
        print("Inside combine_document_run_ocr")
        documentId = docInput.get("documentId")
        dict_file_pages = docInput.get("page_details")
        client_folder = docInput.get("client_folder")
        logfilepath = docInput.get("logfilepath")
        container = docInput.get("container")
        extn = docInput.get("extn")
        orgFileLocation = docInput.get("orgFileLocation")

        print("Dict file pages: ", dict_file_pages)
        # Combine the file and write it back and run the OCR
        first_page = None
        other_pages = {}
        combined_file_path = None
        dict_pages = {}
        for page_name, value in dict_file_pages.items():
            blobPath = value["image_path"]
            fileName = blobPath.split("/")[-1]
            local_file_path = os.path.join(root_folder,
                                           fileName)
            print("Downloaded input image",page_name)
            # local_file_paths.append(local_file_path)
            document_page_number = value['document_page_number']
            dict_pages[str(document_page_number)] = {}
            print(container, blobAccount + "/" + blobPath, local_file_path)
            downloaded = util.downloadFilesFromBlobStore(container,
                                                         blobAccount + "/" + blobPath,
                                                         local_file_path)
            if not downloaded:
                return None
            else:
                dict_pages[str(document_page_number)]["file_path"] = local_file_path
                dict_pages[str(document_page_number)]["png_path"] = value["png_path"]
                dict_pages[str(document_page_number)]["document_page_number"] = document_page_number

            # Make combined file path by adding _combined to the first file name
            print("Combined file path creation")
            if combined_file_path is None:
                p,ext = os.path.splitext(local_file_path)
                p = p + "_combined"
                combined_file_path = os.path.join(tempFolderPath,p + ext)

            print(combined_file_path)

            # Get individual OCR out paths for each page
            print("Page-wise ocr out path")
            o_out_path = os.path.join(client_folder,
                                      fileName + ".ocr.json")
            value['ocr_out_path'] = o_out_path

            # print(o_out_path)

            pageNo = int(document_page_number)
            if pageNo == 0:
                first_page = Image.open(local_file_path)
            else:
                other_pages[pageNo] = local_file_path

        print("Saving combined file")
        other_pages = dict(sorted(other_pages.items()))
        page_array = list(other_pages.values())
        page_array = [Image.open(p) for p in page_array]

        first_page.save(combined_file_path,
                        save_all = True,
                        append_images = page_array)
        print("Saving combined file - Success")

        # Call OCR service
        print("Calling OCR service",
              combined_file_path,
              extnTiff)

        uploaded, URI =  util.uploadFilesToBlobStore(container,
                                                     combined_file_path)
        if uploaded:
            data = {"file_path": container + "/" + URI,
                    "file_extension": extnTiff,
                    "logfilepath": logfilepath,
                    "container": container}
        else:
            print("Combined Image Path not uploaded to blob storage")
            return None

        # Check if extn is PDF; if extn == pdf combined_file_path
        print("Extention and file location ", extn, orgFileLocation)
        if ".pdf" in extn.lower() and orgFileLocation:
            print("PDF file found :",orgFileLocation)
            # data = {"file_path": orgFileLocation,
            #         "file_extension": ".pdf",
            #         "logfilepath": logfilepath,
            #         "container": container}
        print("Calling OCR for {}".format(data.get('file_path')))
        ocr_result = ocr.get_analyseLayout(data)
        ocr_result = json.loads(ocr_result)
        #Get OCR Status
        ocr_status = ocr_result.get("response_code") == 200 ##replace with the code that parses status
        ocr_out = None
        if ocr_status:
            ocr_out = ocr_result.get("response_body")
        else:
            return None

        # Check for Skewness and Correct it
        skewness_corrected = False
        print("Skewness Correction",ocr_status)
        print("Input for skew correction: {}".format(dict_pages))
        if ocr_status and check_skewness:
            skewness_corrected = correct_skewness(dict_pages,
                                                  ocr_out,
                                                  container,
                                                  docInput)
        print("Skewness Correction Completed",skewness_corrected)

        if skewness_corrected:
            # If skewness is corrected, call the method again with check_skewness flag as False
            print("Call OCR Again")
            return combine_document_run_ocr(docInput,
                                     check_skewness = False)
    
        # Take the OCR output, convert it into page wise output and write them back
        print("OCR and Deskew completed")

        print("Parsing OCR result and returning it to caller")
        doc_ocr_header = {}
        for k,v in ocr_out.items():
            if k!= "analyzeResult":
                doc_ocr_header[k] = v
        page_ocr_header = {}
        for k,v in ocr_out['analyzeResult'].items():
            if k!= "readResults":
                page_ocr_header[k] = v

        output = {}
        output_pages = []
        for page_result in ocr_out['analyzeResult']['readResults']:
            if len(page_result['lines'])>=1:
                output_page = {}
                page_num = page_result['page']
                # Form Page OCR Output
                ocr_page = {}
                ocr_page = {**doc_ocr_header, **ocr_page}
                ocr_page_analyzeResult = {}
                ocr_page_analyzeResult = {**ocr_page_analyzeResult,
                                        **page_ocr_header}
                ocr_page_analyzeResult["readResults"] = [page_result]
                ocr_page["analyzeResult"] = ocr_page_analyzeResult
                
        
                identified_page = [p for p in dict_file_pages 
                                if dict_file_pages[p].get('document_page_number') == str(page_num-1)][0]
                ocr_out_path = dict_file_pages[identified_page]['ocr_out_path']
                # Write the Page OCR back
                json_text = json.dumps(ocr_page)
                with open(ocr_out_path,"w") as f:
                    f.write(json_text)
                
                ocr_file_name = os.path.basename(ocr_out_path)
                blob_destination = client_folder + "/" + ocr_file_name
                uploaded, URI = util.uploadFilesToBlobStore(container,ocr_out_path)
                if not uploaded:
                    return None
                else:
                    try:
                        os.remove(ocr_out_path)
                    except:
                        pass

                output_page["ocr_path"] = URI
                output_page["document_page_number"] = dict_file_pages[identified_page]['document_page_number']
                output_page["image_path"] = dict_file_pages[identified_page]['image_path']
                output_pages.append(output_page)
        output["ocr_pages"] = output_pages
        output["documentId"] = documentId

        for page_name, value in dict_file_pages.items():
            value["ocrStatus"] = ocr_status

        for page_name, value in dict_pages.items():
            try:
                os.remove(value["file_path"])
            except:
                pass

        return output
    except:
        print("OCR and Deskew Pages",
              traceback.print_exc())
        return None

def write_ocr(container,
              path,
              content):

    try:
        json_text = json.dumps(content)
        with open(path,"w") as f:
            f.write(json_text)
        # ocr_file_name = os.path.basename(ocr_out_path)
        # blob_destination = client_folder + "/" + ocr_file_name
        uploaded, URI = util.uploadFilesToBlobStore(container,
                                                    path)
        if not uploaded:
            return None
        else:
            try:
                os.remove(path)
            except:
                pass
        return URI
    except:
        print("write_ocr",
              traceback.print_exc())
        return None

def crOcrOutput(ocr_results,
                dict_file_pages,
                container):
    try:
        output = {}
        output_pages = []
        for page_num,page_result in enumerate(ocr_results):
            doc_ocr_header = {}
            for k,v in page_result.items():
                if k!= "analyzeResult":
                    doc_ocr_header[k] = v
            page_ocr_header = {}
            for k,v in page_result['analyzeResult'].items():
                if k!= "readResults":
                    page_ocr_header[k] = v
            page_ocr_result = page_result['analyzeResult']['readResults']
            if len(page_ocr_result[0]['lines']) >= 1:
                output_page = {}
                # page_num = page_result['page']
                # Form Page OCR Output
                ocr_page = {}
                ocr_page = {**doc_ocr_header,
                            **ocr_page}
                ocr_page_analyzeResult = {}
                ocr_page_analyzeResult = {**ocr_page_analyzeResult,
                                        **page_ocr_header}
                ocr_page_analyzeResult["readResults"] = [page_ocr_result]
                ocr_page["analyzeResult"] = ocr_page_analyzeResult
    
                identified_page = [p for p in dict_file_pages 
                                if dict_file_pages[p].get('document_page_number') == str(page_num)][0]

                blobPath = dict_file_pages[identified_page]['image_path']
                fileName = blobPath.split("/")[-1]
                
                ocr_out_path = os.path.join(root_folder, fileName + ".ocr.json")
                # ocr_out_path = dict_file_pages[identified_page]['ocr_out_path']
                output_page["result"] = ocr_page
                output_page["ocr_out_path"] = ocr_out_path

                # Write the Page OCR back
                URI = write_ocr(container,
                                ocr_out_path,
                                ocr_page)
                if URI is None:
                    return None
    
                output_page["ocr_path"] = URI
                output_page["document_page_number"] = dict_file_pages[identified_page]['document_page_number']
                output_page["image_path"] = dict_file_pages[identified_page]['image_path']
                output_pages.append(output_page)
                print("OCR result successfully added",page_num)
        output["ocr_pages"] = output_pages
        return output
    except:
        print("crOcrOutput",
              traceback.print_exc())
        return None


def getFilesToBeOcred(docInput):

    try:
        print("Inside getFilesToBeOcred")
        dict_file_pages = docInput.get("page_details")
        container = docInput.get("container")

        print("Dict file pages: ", dict_file_pages)
        dict_pages = {}
        # local_file_paths = []

        for page_name, value in dict_file_pages.items():
            blobPath = value["image_path"]
            fileName = blobPath.split("/")[-1]
            local_file_path = os.path.join(root_folder,
                                           fileName)
            print("Downloaded input image",page_name)
            document_page_number = value['document_page_number']
            dict_pages[str(document_page_number)] = {}
            print("File Paths:",
                  container,
                  blobAccount + "/" + blobPath,
                  local_file_path)
            downloaded = util.downloadFilesFromBlobStore(container,
                                                         blobAccount + "/" + blobPath,
                                                         local_file_path)
            if not downloaded:
                return None
            else:
                dict_pages[str(document_page_number)]["file_path"] = local_file_path
                dict_pages[str(document_page_number)]["png_path"] = value["png_path"]
                dict_pages[str(document_page_number)]["document_page_number"] = document_page_number
        return dict_pages
    except:
        print("getFilesToBeOcred",
              traceback.print_exc())
        return None


@util.timing
def single_document_run_ocr(docInput):
    """
    """
    try:
        print("Inside single_document_run_ocr")
        documentId = docInput.get("documentId")
        dict_file_pages = docInput.get("page_details")
        container = docInput.get("container")

        dict_pages = getFilesToBeOcred(docInput)
        if dict_pages is None:
            return None

        #Get local path of the downloaded images
        local_file_paths = []
        local_file_paths = [dict_pages[k]["file_path"] for k in dict_pages]
        #OCR and Deskew happens here
        if len(local_file_paths) == len(dict_file_pages.items()):
            ocr_results = ocr.ocr_images(local_file_paths)
        else:
            return None
        print("OCR and Deskew completed")

        #Take the deskewed tiff files and save it as png and upload to png path
        png_paths = []
        png_paths = [dict_pages[k]["png_path"] for k in dict_pages]
        for tiff_path,png_path in zip(local_file_paths,png_paths):
            # print("Tiff and png images",tiff_path,png_path)
            try:
                local_png_path = png_path.split("/")[-1]
                img = Image.open(tiff_path)
                img.save(local_png_path)
                uploaded = util.uploadFilesToBlobStore(container, local_png_path)
                if uploaded:
                    print("Updated png file after deskew",png_path)
                    try:
                        os.remove(local_png_path)
                    except:
                        pass
                uploaded = util.uploadFilesToBlobStore(container, tiff_path)
                if uploaded:
                    print("Updated tiff file after deskew",tiff_path)
                    try:
                        os.remove(tiff_path)
                    except:
                        pass
                    

            except:
                print("updating png and tiff",traceback.print_exc())
                pass


        print("Delete the local images")
        for page_name, value in dict_pages.items():
            try:
                os.remove(value["file_path"])
            except:
                pass

        # # Take the OCR output, convert it into page wise output and write them back

        print("Parsing OCR result and returning it to caller")
        output = crOcrOutput(ocr_results,
                             dict_file_pages,
                             container)
        output["documentId"] = documentId

        for page_name, value in dict_file_pages.items():
            value["ocrStatus"] = True

        return output
    except:
        print("OCR and Deskew Pages",traceback.print_exc())
        return None
