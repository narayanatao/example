# -*- coding: utf-8 -*-
# In[1]: All Imports
import traceback
# import os
# import cv2
# import numpy as np
import imutils
from deskew import determine_skew
# from PIL import Image
import warnings

import util as util
import config as config

warnings.filterwarnings("ignore")

# In[]: Cropping a PDF

def pdfCrop(cropBorder,
            pdf_page,
            img_):

    try:
        #height,width of image
        img_ht = img_.shape[0]
        img_wd = img_.shape[1]
        #height, width of pdf
        pdf_ht = pdf_page.cropBox.upperRight[1]
        pdf_wd = pdf_page.cropBox.upperRight[0]
    
        v_ratio = pdf_ht / img_ht
        h_ratio = pdf_wd / img_wd
    
        # mod_left = left * h_ratio
        # mod_right = right * h_ratio
        # mod_top = top * v_ratio
        # mod_bottom = top * v_ratio
    
        mod_left = cropBorder[0] * h_ratio
        mod_right = cropBorder[2] * h_ratio
        mod_top = cropBorder[1] * v_ratio
        mod_bottom = cropBorder[3] * v_ratio
    
        pdf_page.cropBox.lowerLeft = (mod_left,mod_bottom)
        pdf_page.cropBox.upperRight = (mod_right,mod_top)
        return pdf_page
    except:
        print("crop pdf", traceback.print_exc())
        return None


# In[5]: Image Pre-processing - Cropping an Image

@util.timing
def cropImageNew(img, pdfPage = None):

    #Detect regions in an image
    import cv2
    import numpy as np
    cropBorder = config.getCropBorder()

    def detectRegions(img):
        try:
            mser = cv2.MSER_create()
            regions, bboxes = mser.detectRegions(img)
            return regions, bboxes
        except:
            print("detectRegions",traceback.print_exc())
            return None, None

    #This function crops an image to scale
    try:
        #crop White borders on all sides
        #Remove outlier boxes
        #For this, first detect regions
        #For each region find the aspect ratios i.e. height/width ratio
        #Find mean and SD of these ratios
        #Normalize the ratios to mean = 0 and SD = 1
        #Delete regions where the ratio is greater than 2 SD or less than -2 SD

        maxBorder = cropBorder
        regions, boxes = detectRegions(img)
        aspectRatios = [box[2]//box[3] for box in boxes]
        meanAspect = np.mean(aspectRatios)
        stdAspect = np.std(aspectRatios)
        normAspects = [((aspect - meanAspect)/stdAspect) for aspect in aspectRatios]
        outliers = [i for i, aspect in enumerate(normAspects) if (aspect < -2) or (aspect > 2)]
        boxCopy = boxes.copy()
        boxCopy = np.delete(boxes,outliers,axis = 0)

        if len(boxCopy) == 0:
            return None
        #Keep the maximum border as 50 pixels (maxBorder)
        minLeft = min(np.min([box[0] for box in boxCopy]),maxBorder)
        minTop = min(np.min([box[1] for box in boxCopy]),maxBorder)
        maxRight = max(np.max([box[0] + box[2] for box in boxCopy]),img.shape[1] - maxBorder)
        maxBottom = max(np.max([box[1] + box[3] for box in boxCopy]),img.shape[0] - maxBorder)
        print("Cropped dimensions are:", minLeft,minTop,maxRight,maxBottom)

        #Crop the image
        cpy = img.copy()
        cpy = cpy[minTop:maxBottom,minLeft:maxRight]
        img_border = cv2.copyMakeBorder(
                cpy,
                top=maxBorder,
                bottom=maxBorder,
                left=maxBorder,
                right=maxBorder,
                borderType=cv2.BORDER_CONSTANT,
                value=[255,255,255]
                )
        #Write the cropped image to a file
        # return img_border,pdfPage
        return img_border,(minLeft,minTop,maxRight,maxBottom)
    except:
        print("cropImageNew",
              traceback.print_exc())
        # return None,None
        return None, None

# In[6]: Image Pre-processing - Deskew an Image

@util.timing
def deskewImageNew(img):
    #Correct Skewness of image and return the image
    angle = 0
    try:
        gray = img.copy()
        skewangle = determine_skew(gray[gray.shape[0] // 2:,:])
        if skewangle is not None:
            if (abs(skewangle) <= 30) and (abs(skewangle) != 0):
                angle = skewangle
                print("Rotation angle is: ", -angle)
                img1 = imutils.rotate_bound(gray,-angle)
                return img1
            else:
                return gray
        else:
            return gray
    except:
        print("deskewImageNew",
              traceback.print_exc())
        return img

# In[7]: Image Pre-processing - Image Enhancement

@util.timing
def detectAndChangeBackColor(binarizedImg):
    #After image binarization an image can have white background with black fonts
    #OR black background with white fonts. The second case is a noise and we need to convert
    #to white background with black fonts. This function does that
    import numpy as np
    try:
        u, indices = np.unique(binarizedImg,
                               return_inverse = True)
        axis = 0
        a = u[np.argmax(np.apply_along_axis(np.bincount,
                                            axis,
                                            indices.reshape(binarizedImg.shape),
                                            None,
                                            np.max(indices) + 1),
                        axis = axis)]
        bckCol = np.argmax(np.bincount(a))
        if bckCol == 0:
            return np.bitwise_not(binarizedImg)
        else:
            return binarizedImg
    except:
        print("detectAndChangeBackColor",
              traceback.print_exc())
        return None

@util.timing
def imageEnhancementNew(pre):
    #Apply image enhancement at a page level and
    #return the image object to the calling function
    #If required, this image will be stored in blobstore and URI will be sent
    #Read the image as a binary file also
    try:
        #Apply MedianBlur to remove Watermarks or other noisy and data with light fonts
        #pre = cv2.medianBlur(pre,3)

        #Binarization of image - Make it strictly black or white 0 or 255
        # pre = cv2.threshold(pre, 210, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

        #Detect if background color is white or black. If black, change it to White
        pre = detectAndChangeBackColor(pre)

        return pre
    except:
        print(traceback.print_exc())
        return None

# In[8]: Image Pre-processing - Main/Caller Function

@util.timing
def pageProcessNew(page,
                   documentId,
                   client_folder,
                   container,
                   extn = ".tif",
                   pdfPage = None):

    # import cv2
    import os
    import shutil
    from PIL import Image
    # import util as util
    # import config as config


    #1. image enhancement (Sharpening, Filtering)
    #2. Image de-skewing
    #3. Image Cropping
    #4. Converts tiff image to .png file to be displayed in the website
    #5. Combines single page tiff files to a multipage tiff.
    #This needs to be used for ABBYY extraction to avoid scaling issues
    tempFolderPath = config.getTempFolderPath()

    #Get threshold values for image processing and document extraction
    # cropBorder = config.getCropBorder()
    # cropRequired = config.getCropRequired()
    # deskewingRequired = config.getDeskewingRequired()

    #Get config for blob store or folder
    blobAccount = config.getBlobAccountName()

    #Get suffixes, mimetype and extensions
    preprocFileSuffix = config.getPreprocessedFileSuffix()
    extnPng = config.getExtnPng()

    try:
        print("Extension at the start of pageProcessNew",extn)
        pageInfo = {}
        pageInfo["pageNo"] = ""
        pageInfo["pngURI"] = ""
        pageInfo["image"] = ""
        pageInfo["height"] = ""
        pageInfo["width"] = ""
        
        pageLocation = page.get("location")
        pageNo = page.get("pageNumber")
        pageFileName = pageLocation.split("/")[-1]
        pagePath = os.path.join(tempFolderPath,
                                pageFileName)
        pageBlobPath = blobAccount + "/" + pageLocation
        print("Preprocess input path",
              pageBlobPath)
        print("Checking download input before imgpreproc: ", container, pageBlobPath,
                                                pagePath)
        downloaded = util.downloadFilesFromBlobStore(container, pageBlobPath,
                                                pagePath)
        if not downloaded:
            return None

        print("TESTING: ", pagePath)
        # pageNo = os.path.splitext(os.path.split(pagePath)[1])[0].split("-")[-1]
        pageInfo["pageNo"] = pageNo

        #Read the Image of each page
        # img = cv2.imread(pagePath,
        #                  cv2.IMREAD_GRAYSCALE)
        img = Image.open(pagePath)
        wd,ht = img.size
        pageInfo["height"] = ht
        pageInfo["width"] = wd

        #Convert the pre-processed image to png to be displayed in UI
        if img is not None:
            #Save the pre-processed image to a blob storage
            localFileName = os.path.basename(pagePath)
            filename, file_extension = os.path.splitext(localFileName)
            pngLocalFileName = filename + preprocFileSuffix + extnPng
            localFileName = filename + preprocFileSuffix + file_extension
            localFilePath = os.path.join(tempFolderPath,
                                         localFileName)
            print("Printing path variables")
            print(pngLocalFileName)
            print(localFilePath)
            # cv2.imwrite(localFilePath,img)
            shutil.copy2(pagePath, localFilePath)
            localFileName = os.path.basename(localFilePath)
            uploaded, URI = util.uploadFilesToBlobStore(container,
                                                        localFilePath)
            if not uploaded:
                return None
            else:
                try:
                    os.remove(localFilePath)
                except:
                    pass

            pageInfo["image"] = URI
            pageInfo["height"] = ht
            pageInfo["width"] = wd

            #Convert image to png for displaying in UI
            pngPath = os.path.join(tempFolderPath,
                                   pngLocalFileName)
            # cpy = Image.fromarray(img)
            # cpy.save(pngPath)
            img.save(pngPath)
            if pngPath is not None:
                #Upload png files to blob store
                #for pdf file, get the uncropped png file
                print("Extension file is:",extn,
                      "orgPngPath present","orgPngPath" in pageInfo.keys())
                if ("pdf" in extn.lower()) and ("orgPngPath" in pageInfo.keys()):
                    pngPath = pageInfo.get("orgPngPath")
                    print("Uncropped png file",pngPath)
                uploaded, URI = util.uploadFilesToBlobStore(container, pngPath)
                if uploaded:
                    pageInfo["pngURI"] = URI
                try:
                    os.remove(pngPath)
                except:
                    pass

                print("Image converted to png ", str(pageNo))

            else:
                print("Error in png file conversion")

        # return pageInfo,cropPdf
        return pageInfo
    except:
        print("pageProcessNew",
              traceback.print_exc())
        # return None,None
        return None

@util.timing
def pageProcessNew_old(page,
                       documentId,
                       client_folder,
                       container,
                       extn = ".tif",
                       pdfPage = None):

    import cv2
    import os
    from PIL import Image
    # import util as util
    # import config as config


    #1. image enhancement (Sharpening, Filtering)
    #2. Image de-skewing
    #3. Image Cropping
    #4. Converts tiff image to .png file to be displayed in the website
    #5. Combines single page tiff files to a multipage tiff.
    #This needs to be used for ABBYY extraction to avoid scaling issues
    tempFolderPath = config.getTempFolderPath()

    #Get threshold values for image processing and document extraction
    cropBorder = config.getCropBorder()
    cropRequired = config.getCropRequired()
    deskewingRequired = config.getDeskewingRequired()

    #Get config for blob store or folder
    blobAccount = config.getBlobAccountName()

    #Get suffixes, mimetype and extensions
    preprocFileSuffix = config.getPreprocessedFileSuffix()
    extnPng = config.getExtnPng()

    try:
        print("Extension at the start of pageProcessNew",extn)
        pageInfo = {}
        pageInfo["pageNo"] = ""
        pageInfo["pngURI"] = ""
        pageInfo["image"] = ""
        pageInfo["height"] = ""
        pageInfo["width"] = ""
        
        pageLocation = page.get("location")
        pageNo = page.get("pageNumber")
        pageFileName = pageLocation.split("/")[-1]
        pagePath = os.path.join(tempFolderPath,
                                pageFileName)
        pageBlobPath = blobAccount + "/" + pageLocation
        print("Preprocess input path",
              pageBlobPath)
        print("Checking download input before imgpreproc: ", container, pageBlobPath,
                                                pagePath)
        downloaded = util.downloadFilesFromBlobStore(container, pageBlobPath,
                                                pagePath)
        if not downloaded:
            return None

        print("TESTING: ", pagePath)
        # pageNo = os.path.splitext(os.path.split(pagePath)[1])[0].split("-")[-1]
        pageInfo["pageNo"] = pageNo

        #Read the Image of each page
        img = cv2.imread(pagePath,
                         cv2.IMREAD_GRAYSCALE)

        pageInfo["height"] = img.shape[0]
        pageInfo["width"] = img.shape[1]

        #Enhance the quality of the image
        if img is not None:
            enhImg = imageEnhancementNew(img)
            if enhImg is not None:
                print("Image Enhancement done for ", str(pageNo))
                img = enhImg
                pageInfo["height"] = img.shape[0]
                pageInfo["width"] = img.shape[1]
            else:
                print("Error in Image Enhancement")
        
        #Deskew the image if manual scanning is not in upright position
        if img is not None:
            # deskewImg = deskewImageNew(img)
            deskewImg = None
            if deskewImg is not None:
                print("Deskew done for ", str(pageNo))
                if deskewingRequired == 1:
                    img = deskewImg
                pageInfo["height"] = img.shape[0]
                pageInfo["width"] = img.shape[1]

            else:
                print("Error in Deskew")

        #Create an uncropped png file in case of pdf
        if img is not None:
            orgPngPath = pageFileName + extnPng
            cpy = Image.fromarray(img)
            cpy.save(orgPngPath)
            pageInfo["orgPngPath"] = orgPngPath
        #Create an uncropped png file in case of pdf

        #Crop image to have a fixed border for all images to give a standard scale for all images
        #This helps ML algo to find a pattern
        if img is not None:
            if deskewImg is not None:
                cropImg, cropBorder = cropImageNew(deskewImg)
            else:
                cropImg, cropBorder = cropImageNew(img)
            if cropImg is not None:
                print("Cropping done for ", str(pageNo))
                if cropRequired == 1:
                    img = cropImg
                    pageInfo["height"] = img.shape[0]
                    pageInfo["width"] = img.shape[1]
                    pageInfo["cropped"] = 1
                    pageInfo["cropped_border"] = cropBorder

                pageInfo["cropHeight"] = cropImg.shape[0]
                pageInfo["cropWidth"] = cropImg.shape[1]
            else:
                print("Error in Cropping")

        print("After Cropping:", img is None)

        #Convert the pre-processed image to png to be displayed in UI
        if img is not None:
            #Save the pre-processed image to a blob storage
            localFileName = os.path.basename(pagePath)
            filename, file_extension = os.path.splitext(localFileName)
            pngLocalFileName = filename + preprocFileSuffix + extnPng
            localFileName = filename + preprocFileSuffix + file_extension
            localFilePath = os.path.join(tempFolderPath,
                                         localFileName)
            print("Printing path variables")
            print(pngLocalFileName)
            print(localFilePath)
            cv2.imwrite(localFilePath,img)
            localFileName = os.path.basename(localFilePath)
            uploaded, URI = util.uploadFilesToBlobStore(container,
                                                        localFilePath)
            if not uploaded:
                return None
            else:
                try:
                    os.remove(localFilePath)
                except:
                    pass

            pageInfo["image"] = URI
            pageInfo["height"] = img.shape[0]
            pageInfo["width"] = img.shape[1]

            #Convert image to png for displaying in UI
            pngPath = os.path.join(tempFolderPath,
                                   pngLocalFileName)
            cpy = Image.fromarray(img)
            cpy.save(pngPath)
            if pngPath is not None:
                #Upload png files to blob store
                #for pdf file, get the uncropped png file
                print("Extension file is:",extn,
                      "orgPngPath present","orgPngPath" in pageInfo.keys())
                if ("pdf" in extn.lower()) and ("orgPngPath" in pageInfo.keys()):
                    pngPath = pageInfo.get("orgPngPath")
                    print("Uncropped png file",pngPath)
                uploaded, URI = util.uploadFilesToBlobStore(container, pngPath)
                if uploaded:
                    pageInfo["pngURI"] = URI
                try:
                    os.remove(pngPath)
                except:
                    pass

                print("Image converted to png ", str(pageNo))

            else:
                print("Error in png file conversion")

        # return pageInfo,cropPdf
        return pageInfo
    except:
        print(traceback.print_exc())
        # return None,None
        return None


def process_single_page(docInfo):

    try:
        page = docInfo["page"]
        documentId = docInfo["documentId"]
        client_folder = docInfo["client_folder"]
        container = docInfo["container"]
        extn = docInfo["extn"]
        result = pageProcessNew(page,
                                documentId,
                                client_folder, 
                                container,
                                extn)
        return result
    except:
        print("process_single_page",
              traceback.print_exc())
        return None

@util.timing
def preprocessPages(docInput):

    from multiprocessing import Pool

    try:
        documentId = docInput["documentId"]
        client_folder = docInput["client_folder"]
        splitPageFiles = docInput['pages']
        container = docInput["container"]
        extn = docInput["extn"]
        results = []
        #Implement parallel processing of image pre-processing - 01 Apr 2022
        pages = []
        for page in splitPageFiles:
            split_page = {}
            print("Extension in processpages",extn)
            split_page["page"] = page
            split_page["documentId"] = documentId
            split_page["client_folder"] = client_folder
            split_page["container"] = container
            split_page["extn"] = extn
            pages.append(split_page)

        # if len(pages) > 0:
        #     # try:
        #     #     set_start_method("spawn")
        #     # except:
        #     #     pass
        #     pool_size = min(len(pages),10)
        #     pool = Pool(pool_size)
        #     res = pool.map_async(process_single_page,pages)
        #     pool.close()
        #     pool.join()
        #     results = res.get()
        # else:
        #     return None
        if len(pages) > 0:
            for page_ind,page in enumerate(pages):
                result = process_single_page(page)
                if result is None:
                    return None
                results.append(result)
        else:
            return None
        
        return results
    except:
        print("Process Pages",
              traceback.print_exc())
        return None
