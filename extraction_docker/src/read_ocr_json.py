# -*- coding: utf-8 -*-
"""
Created on Thu Oct 28 18:27:16 2021

@author: Hari
"""
import json
import pandas as pd
import numpy as np
import time
import cv2
import string
import traceback
import os

import warnings
warnings.filterwarnings("ignore")

import util as util
import config
# In[get script directory]

script_dir = os.path.dirname(__file__)
LIKeyWordsPath = config.getLIKeywordsPath()

with open(LIKeyWordsPath) as data:
    LIKeyWordData = json.load(data)

lineInfo = LIKeyWordData['lineInfo']


def splitLineTextVLines(df):

    df_copy = df.copy(deep = True)
    try:
        df = df.sort_values(by = ["page_num","line_num","word_num"],
                            ascending = [True,True,True])

        lines = list(df.groupby(by = ["page_num",
                                      "line_top",
                                      "line_left"]).groups)

        for line_ind,line in enumerate(lines):
            df_lines = df[(df["page_num"] == line[0]) &
                          (df["line_top"] == line[1]) &
                          (df["line_left"] == line[2])]

            right_vals = list(df_lines["line_right_y1"].unique())
            if len(right_vals) == 1:
                continue
            #Finding the text indices that are close to lines on the right
            sno = 0
            rightLineCoords = []
            line_breaks = []
            l = list(df_lines[["left","right"]].values.tolist())
            df_lines_iter = df_lines.iterrows()
            for line_ind, row in df_lines_iter:
                if row["lineRight"] == 1:
                    rightLineCoords = [row["line_right_y1"],
                                       row["line_right_y2"]]
                    sno = l.index(row[["left","right"]].tolist())
                    sub_rows = df_lines.iloc[sno+1:]
                    if sub_rows.shape[0] > 0:
                        for sub_ind,sub_row in sub_rows.iterrows():
                            sub_rightLineCoords = [sub_row["line_right_y1"],
                                                   sub_row["line_right_y2"]]
                            if (rightLineCoords == sub_rightLineCoords):
                                try:
                                    next(df_lines_iter)
                                except:
                                    pass
                            else:
                                sno = l.index(sub_row[["left",
                                                       "right"]].tolist()) - 1
                                line_breaks.append(sno)
                                break

            #Split the line text
            print("Line breaks:",line_breaks)
            if len(line_breaks) > 0:
                #Modify the indices to row ranges for easy selection
                j = []
                end = 0
                for i in range(len(line_breaks)+1):
                    if i==0:
                        j.append((0,line_breaks[i]+1))
                        end = line_breaks[i] + 1
                    elif i == len(line_breaks):
                        if end <= df_lines.shape[0]:
                            j.append((end,":"))
                    else:
                        j.append((end,line_breaks[i]+1))
                        end = line_breaks[i] + 1
                #Using ranges, create new line items with proper
                for i in j:
                    if i[1] != ":":
                        sub_lines = df_lines.iloc[i[0]:i[1]]
                    else:
                        sub_lines = df_lines.iloc[i[0]:]

                    line_texts = " ".join(list(sub_lines["text"].values))
                    line_right = max(list(sub_lines["right"].values))
                    line_left = min(list(sub_lines["left"].values))
                    line_width = line_right - line_left
                    cnt = 0
                    for sub_ind,sub_row in sub_lines.iterrows():
                        df.loc[df["token_id"] == sub_row["token_id"],
                               ["line_right",
                                "line_left",
                                "line_width",
                                "line_text",
                                "word_num"]
                               ] = [line_right,
                                    line_left,
                                    line_width,
                                    line_texts,
                                    cnt]

                        cnt += 1
            else:
                print("into other rows")
        df = df.sort_values(by = ["page_num",
                                  "line_num",
                                  "line_left",
                                  "word_num"],
                            ascending = [True,
                                         True,
                                         True,
                                         True])
        line_no = -1
        prev_ln_text = ""
        prev_ln_top = -1
        prev_ln_left = -1
        for ind, row in df.iterrows():
            ln_text = row["line_text"]
            ln_top = row["line_top"]
            ln_left = row["line_left"]
            token_id = row["token_id"]
            if not ((prev_ln_left == ln_left) and (prev_ln_top == ln_top) and (prev_ln_text == ln_text)):
                line_no += 1

            df.loc[df["token_id"] == token_id,
                   "line_num"] = line_no
            prev_ln_left = ln_left
            prev_ln_top = ln_top
            prev_ln_text = ln_text

        return df
    except:
        print("splitLineTextVLines",traceback.print_exc())
        return df_copy


# In[7]: Declare Image Feature extraction functions
@util.timing
def extract_image_features(df,imgFilePath):

    def getLineInfo(DF, imagepath, page_count):

        def findZeroPattern(vals):
            binaryVals = vals // 255
            arr1 = np.concatenate(([0],binaryVals))
            arr2 = np.concatenate((binaryVals,[0]))
            ptnVals = np.bitwise_and(arr1,arr2)[1:]

            iszero = np.concatenate(([0],
                                     np.equal(ptnVals, 0).view(np.int8),
                                     [0]))
            absdiff = np.abs(np.diff(iszero))
            ranges = np.where(absdiff == 1)[0].reshape(-1, 2)
            return ranges

        def findLines(arr):

            height = arr.shape[0]
            width = arr.shape[1]
            vlines = []
            hlines = []
            thresh = 50

            #Identify vertical lines
            for i in range(width):
                single = arr[:,i:i+1][:,0]

                if len(single) > 0:
                    vLines1 = findZeroPattern(single)
                    for line in vLines1:
                        # print("VLine Detected",
                        #       line[1] - line[0])
                        if line[1] - line[0] >= thresh:
                            coord = (i,line[0],i,line[1])
                            vlines.append(coord)

            #Identify horizontal lines
            for i in range(height):
                single = arr[i:i+1,:][0]
                if len(single) > 0:
                    hLines1 = findZeroPattern(single)
                    for line in hLines1:
                        if line[1] - line[0] >= thresh:
                            coord = (line[0],i,line[1],i)
                            hlines.append(coord)
            return vlines,hlines

        def findLinesClose(row,hlines,vlines):

            def sortHline(val):
                return val[1],val[3]

            def sortVline(val):
                return val[0],val[2]

            def vOverlapRight(line, word):
                if word[1] > line[3]:
                    return False
                elif word[3] < line[1]:
                    return False
                elif word[2] > line[0]:
                    return False
                return True

            def vOverlapLeft(line, word):
                if word[1] > line[3]:
                    return False
                elif word[3] < line[1]:
                    return False
                elif word[0] < line[0]:
                    return False
                return True

            def hOverlapTop(line, word):
                if word[0] > line[2]:
                    return False
                elif word[2] < line[0]:
                    return False
                elif word[1] < line[1]:
                    return False
                return True

            def hOverlapDown(line, word):
                if word[0] > line[2]:
                    return False
                elif word[2] < line[0]:
                    return False
                elif word[3] > line[1]:
                    return False
                return True


            isAbove = 0
            lenAbove = 0
            above_x1 = (-1,-1)
            above_x2 = (-1,-1)
            isBelow = 0
            lenBelow = 0
            below_x1 = (-1,-1)
            below_x2 = (-1,-1)
            isLeft = 0
            lenLeft = 0
            left_y1 = (-1,-1)
            left_y2 = (-1,-1)
            isRight = 0
            lenRight = 0
            right_y1 = (-1,-1)
            right_y2 = (-1,-1)

            wordBB = [row["left"],row["top"],
                      row["right"],row["bottom"]]

            #find line above
            topLines = [hline for hline in hlines
                          if hline[1] < wordBB[1]]
            topLines.sort(key = sortHline,
                        reverse = True)
            for hline in topLines:
                line_coord = (hline[0],
                              hline[1],
                              hline[2],
                              hline[3])
                overlap = hOverlapTop(line_coord,
                                      wordBB)
                if overlap:
                    isAbove = 1
                    lenAbove = hline[2] - hline[0]
                    above_x1 = (hline[0],hline[1])
                    above_x2 = (hline[2],hline[3])
                    break

            #find Line below
            belowLines = [hline for hline in hlines
                          if hline[1] > wordBB[3]]
            belowLines.sort(key = sortHline)
            for hline in belowLines:
                line_coord = (hline[0],
                              hline[1],
                              hline[2],
                              hline[3])
                overlap = hOverlapDown(line_coord,
                                       wordBB)
                if overlap:
                    isBelow = 1
                    lenBelow = hline[2] - hline[0]
                    below_x1 = (hline[0],
                                hline[1])
                    below_x2 = (hline[2],
                                hline[3])
                    break

            #Find line Left
            leftLines = [vline for vline in vlines
                          if vline[0] < wordBB[0]]
            leftLines.sort(key = sortVline,
                            reverse = True)
            for vline in leftLines:
                line_coord = (vline[0],
                              vline[1],
                              vline[2],
                              vline[3])
                overlap = vOverlapLeft(line_coord,
                                       wordBB)
                if overlap:
                    isLeft = 1
                    lenLeft = vline[3] - vline[1]
                    left_y1 = (vline[0],vline[1])
                    left_y2 = (vline[2],vline[3])
                    break

            #Find Line Right
            rightLines = [vline for vline in vlines
                          if vline[0] > wordBB[2]]
            rightLines.sort(key = sortVline)
            for vline in rightLines:
                line_coord = (vline[0],
                              vline[1],
                              vline[2],
                              vline[3])
                overlap = vOverlapRight(line_coord,
                                        wordBB)
                if overlap:
                    isRight = 1
                    lenRight = vline[3] - vline[1]
                    right_y1 = (vline[0],vline[1])
                    right_y2 = (vline[2],vline[3])
                    break

            return pd.Series([isAbove, lenAbove, above_x1, above_x2,
                              isBelow, lenBelow, below_x1, below_x2,
                              isLeft, lenLeft, left_y1, left_y2,
                              isRight, lenRight, right_y1, right_y2
                              ])

        DF_new = pd.DataFrame()

        ret, imgs = cv2.imreadmulti(imagepath)

        for i in range(page_count):
            im = imgs[i]

            if len(imgs[i].shape) == 3:
                imgs[i] = cv2.cvtColor(imgs[i],
                                       cv2.COLOR_BGR2GRAY)
            blur = imgs[i]
            pre = cv2.threshold(blur, 200, 255,
                                cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
            t = time.time()
            vlines, hlines = findLines(pre)
            for vline in vlines:
                cv2.line(im,(vline[0],vline[1]),
                         (vline[2],vline[3]),(0,255,255))
            for hline in hlines:
                cv2.line(im,(hline[0],hline[1]),
                         (hline[2],hline[3]),(255,0,255))
            # cv2.imwrite("out.jpg",im)

            height = pre.shape[0]
            width = pre.shape[1]

            hlines = [(hline[0] / width,hline[1] / height,hline[2] / width,
                       hline[3] / height) for hline in hlines]
            vlines = [(vline[0] / width,vline[1] / height,vline[2] / width,
                       vline[3] / height) for vline in vlines]

            tempar = DF.copy()

            if tempar.shape[0] > 0:
                t = time.time()
                tempar[lineInfo] = tempar.apply(findLinesClose,
                      args=(hlines,vlines),axis = 1)
                print("FindLinesClose: ", time.time() - t)
                if DF_new.shape[0] == 0:
                    DF_new = tempar
                else:
                    DF_new = DF_new.append(tempar)

        return DF_new

    if imgFilePath != '':
        temp_df = getLineInfo(df, imgFilePath, 1)
        #print("Each File",type(temp_df),temp_df.shape)
    else:
        print("\t\tNo Image File found for ",imgFilePath)

    return temp_df


# In[8]: Declare Functions that extract features from ocr files
@util.timing
def read_ocr_json_file(path,imgpath):

    @util.timing
    def read_result_tag(resultList):
        rows = []
        tokenid = 10000

        for resultobj in resultList:
            pageNo = resultobj["page"]
            lines = resultobj["lines"]

            unit = resultobj["unit"]
            unit_in_pixel = 96 if unit == "inch" else 1

            width = int(resultobj["width"] * unit_in_pixel)
            height = int(resultobj["height"] * unit_in_pixel)
            lineNo = 0
            for line in lines:
                value = line["text"]
                bb = line["boundingBox"]
                left = min(bb[0] * unit_in_pixel, bb[6] * unit_in_pixel) / width
                right = max(bb[2] * unit_in_pixel, bb[4] * unit_in_pixel) / width
                top = min(bb[1] * unit_in_pixel,bb[3] * unit_in_pixel) / height
                down = max(bb[5] * unit_in_pixel,bb[7] * unit_in_pixel) / height
                # print(pageNo,lineNo,value,left,right,top,down)

                words = line["words"]
                wordNo = 0
                for word in words:
                    row = {}
                    wordValue = word["text"]
                    wordConfidence = word["confidence"]
                    wordBB = word["boundingBox"]
                    wordLeft = min(wordBB[0] * unit_in_pixel, wordBB[6] * unit_in_pixel) / width
                    wordRight = max(wordBB[2] * unit_in_pixel, wordBB[4] * unit_in_pixel) / width
                    wordTop = min(wordBB[1] * unit_in_pixel,wordBB[3] * unit_in_pixel) / height
                    wordDown = max(wordBB[5] * unit_in_pixel,wordBB[7] * unit_in_pixel) / height

                    row["token_id"] = tokenid
                    row["page_num"] = pageNo
                    row["line_num"] = lineNo
                    row["line_text"] = value

                    row["line_left"] = left
                    row["line_top"] = top
                    row["line_height"] = down - top
                    row["line_width"] = right - left
                    row["line_right"] = right
                    row["line_down"] = down

                    row["word_num"] = wordNo
                    row["text"] = wordValue
                    row["conf"] = wordConfidence

                    row["left"] = wordLeft
                    row["top"] = wordTop
                    row["height"] = wordDown - wordTop
                    row["width"] = wordRight - wordLeft
                    row["right"] = wordRight
                    row["bottom"] = wordDown
                    row["image_height"] = height
                    row["image_widht"] = width

                    row["is_portrait_page"] = 1 if height > width else 0
                    row["page_ratio"] = round(height/width,3)
                    rows.append(row)
                    wordNo = wordNo + 1
                    tokenid = tokenid + 1
                lineNo = lineNo + 1

        return rows

    @util.timing
    def read_page_result_tag(pageList, df_ocr):
        df_ocr['rows_in_table'] = 0
        df_ocr['cols_in_table'] = 0
        df_ocr['table_num'] = 0
        df_ocr['row_num'] = 0
        df_ocr['col_num'] = 0
        df_ocr.astype({'line_num': 'int32', 'word_num': 'int32'}).dtypes

        rows = []
        for pages in pageList:

            tables = pages['tables']
            for no, table in enumerate(tables):
                table_num = no + 1
                rows_in_table = table['rows']
                cols_in_table = table['columns']
                table_contents = table['cells']

                for cells in table_contents:
                    row_num = cells['rowIndex'] + 1
                    col_num = cells['columnIndex'] + 1
                    cell_text = cells['text']
                    elements = cells['elements']

                    for i, element in enumerate(elements):
                        row = {}
                        element_list = element.split("/")
                        if len(element_list) < 6:
                            continue
                        page_num = int(element_list[2]) + 1
                        line_num = element_list[4]
                        word_num = element_list[6]
                        try:
                            text = cell_text.split(" ")[i]
                        except:
                            print("Read Page Result Tag:",
                                  traceback.print_exc())
                            text = " "
                            pass

                        row['page_num'] = page_num
                        row['line_num'] = line_num
                        row['word_num'] = word_num
                        row['table_num'] = table_num
                        row['row_num'] = row_num
                        row['col_num'] = col_num
                        row['rows_in_table'] = rows_in_table
                        row['cols_in_table'] = cols_in_table
                        row['text'] = text
                        rows.append(row)
                        df_ocr.loc[(df_ocr['page_num'].astype(int) == int(page_num)) &
                                   (df_ocr['line_num'].astype(int) == int(line_num)) &
                                   (df_ocr['word_num'].astype(int) == int(word_num)) &
                                   (df_ocr['text'] == text) ,
                                   ['table_num','row_num','col_num','rows_in_table','cols_in_table']
                                  ] = [table_num,row_num,col_num,rows_in_table,cols_in_table]

        return df_ocr

    def correctAmountTokens(df):
        df_copy = df.copy(deep = True)
        try:
            pun = list(string.punctuation)
            pun.remove(".")
            pun_str = "".join(pun)
            df["text_mod"] = df["text"]
            df["text_mod"].astype(str)
            df["text_mod"] = df["text_mod"].apply(lambda x:x.rstrip(pun_str))
            df["text_mod"] = df["text_mod"].apply(lambda x:x.lstrip(pun_str))
            #print("Strip completed")
            df["cond1"] = df["text_mod"].apply(lambda s:s[len(s) - s[::-1].find(".")-2::-1].replace(",","").replace(".","").isdigit())
            #print("condition 1")
            df["cond2"] = df["text_mod"].apply(lambda s:s[len(s):len(s) - s[::-1].find(".") - 1:-1].isdigit())
            #print("condition 2")
            df["cond"] = df["cond1"] & df["cond2"]
            #print(df)
            df.loc[df["cond"] == True,"text"] = df.loc[df["cond"] == True,
                                                       "text_mod"]
            df.drop(["cond","cond1","cond2","text_mod"],
                    inplace = True,
                    axis = 1)
            return df
        except:
            print("correctAmountTokens Failed:",
                  traceback.print_exc())
            return df_copy

    ###Function to correct OCR o/p when there is a single amount field is split into two tokens
    def correctAmountLineTokens(df):

        df_copy = df.copy(deep = True)

        try:
            #Group the ocr o/p by Azure Lines
            lines = df.groupby(by = ["line_text","line_top","line_left"]).groups
            #For each line check if the number of tokens is 2.
            # If 2, combine the tokens and check if it's in amount format
            #If yes, update the text same as the combined text and also update the line dimensions
            for line_text,line_top,line_left in lines:
                df_line = df[(df["line_text"] == line_text) &
                             (df["line_top"] == line_top) &
                             (df["line_left"] == line_left)]

                texts = df_line["text"].to_list()
                text = ""
                for i in range(df_line.shape[0]):
                    text += texts[i]
                    
                df_line.sort_values(["word_num"],
                                    ascending = [True],
                                    inplace = True)
                if util.isTokenAmount(text):
                    line_down = df_line["line_down"]
                    line_right = df_line["line_right"]
                    line_height = df_line["line_height"]
                    line_width = df_line["line_width"]
                    df.loc[(df["line_text"] == line_text) &
                           (df["line_top"] == line_top) &
                           (df["line_left"] == line_left),
                           ["top","bottom","left","right","height","width",
                            "text","line_text"]] = [line_top,
                                                    line_down,
                                                    line_left,
                                                    line_right,
                            line_height,line_width,text,text]
            df.drop_duplicates(["text","top","bottom","left","right"],
                               inplace = True)
            return df
        except:
            print("Correct Amount Line Tokens",
                  traceback.print_exc())
            return df_copy


    @util.timing
    def read_lines_from_table(df):

        df["isTableLine"] = 0
        df["noNeighbours"] = 0
        df["isLastAmt"] = 0
        df["tableLineNo"] = 0

        df = df.sort_values(["line_top","line_left"])

        df["temp"] = df["line_top"].astype(str)
        df["temp"] = df["temp"] + "---"
        df["temp"] = df["temp"] + df["line_down"].astype(str)
        df["temp"] = df["temp"] + "---"
        df["temp"] = df["temp"] + df["line_left"].astype(str)
        df["temp"] = df["temp"] + "---"
        df["temp"] = df["temp"] + df["line_right"].astype(str)

        unqs = list(df["temp"].unique())
        df = df.drop(["temp"], axis = 1)

        for line_no,unq in enumerate(unqs):
            dims = unq.split("---")

            top = float(dims[0])
            bottom = float(dims[1])
            left = float(dims[2])
            right = float(dims[3])

            dffilt1 = df[(df["line_top"] == top) &
                        (df["line_down"] == bottom) &
                        (df["line_left"] == left) &
                        (df["line_right"] == right) &
                        (df["isTableLine"] == 0)
                        ]

            if dffilt1.shape[0] == 0: # case where isTableLine is already one
                continue   # Still match with next lines

            dffilt2 = df[(((df["line_top"] < bottom) &
                     (df["line_down"] > top)) |
                    ((df["line_top"] > bottom) &
                     (df["line_down"] < top))) &
                     (df["line_left"] != left) &
                     (df["line_right"] != right) &
                     (df["isTableLine"] == 0)
                     ]

            dffilt = dffilt1.append(dffilt2)
            if dffilt.shape[0] > 0:
                dffilt = dffilt.sort_values(["line_left"],
                                            ascending = [False])
                first = dffilt.iloc[0]
                text = first["text"]
                is_amt = int(util.isAmount(text))
                for ind, row in dffilt.iterrows():
                    df.loc[(df["line_left"] == row["line_left"]) &
                           (df["line_right"] == row["line_right"]) &
                           (df["line_top"] == row["line_top"]) &
                           (df["line_down"] == row["line_down"]),
                           ["isTableLine","noNeighbours",
                           "isLastAmt","tableLineNo"]] = [1,
                            len(dffilt['line_text'].unique()),is_amt,
                            line_no + 1]

        return df

    def token_distances(DF):

        df = pd.DataFrame()
        for file in DF['page_num'].unique():
            temp_DF = DF[DF['page_num']==file]
            temp_DF.sort_values(['token_id'], inplace=True)

            # Calculate distances between tokens

            temp_DF['token_dist_next'] = temp_DF['bottom'] - temp_DF['top'].shift(-1)
            temp_DF['token_dist_prev'] = temp_DF['bottom'].shift(1) - temp_DF['top']

            temp_DF['token_dist_forward'] = temp_DF['right'].shift(1) - temp_DF['left']
            temp_DF['token_dist_backward'] = temp_DF['right'] - temp_DF['left'].shift(-1)

            df = df.append(temp_DF, ignore_index=True)

        df.fillna({'token_dist_prev':0,'token_dist_next':0,
                  'token_dist_forward':0, 'token_dist_backward':0}, inplace=True)

        return df

    def position_binning(DF):
        bins = [0,0.24,0.49,0.74,1]
        labels = [1,2,3,4]
        total_grids = len(labels) ** 2

        # Assign X and Y level bins
        DF['X_text_start'] = pd.cut(DF['left'],bins = bins, labels = labels,include_lowest=True)
        DF['y_text_start'] = pd.cut(DF['top'],bins = bins, labels = labels,include_lowest=True)
        DF['X_text_end'] = pd.cut(DF['right'],bins = bins, labels = labels,include_lowest=True)
        DF['y_text_end'] = pd.cut(DF['bottom'],bins = bins, labels = labels,include_lowest=True)

        DF['X_line_start'] = pd.cut(DF['line_left'],bins = bins, labels = labels,include_lowest=True)
        DF['y_line_start'] = pd.cut(DF['line_top'],bins = bins, labels = labels,include_lowest=True)
        DF['X_line_end'] = pd.cut(DF['line_right'],bins = bins, labels = labels,include_lowest=True)
        DF['y_line_end'] = pd.cut(DF['line_down'],bins = bins, labels = labels,include_lowest=True)

        # Calculate Grid value
        y_inc = 0
        for y_bin in range(1,len(labels)+1):
            for x_bin in range(1,len(labels)+1):
                grid_value = (int(x_bin)+y_inc)/total_grids

                DF.loc[((DF['y_text_start'] == y_bin) & (DF['X_text_start'] == x_bin)), 'text_start_grid'] = grid_value
                DF.loc[((DF['y_text_end'] == y_bin) & (DF['X_text_end'] == x_bin)), 'text_end_grid'] = grid_value
                DF.loc[((DF['y_line_start'] == y_bin) & (DF['X_line_start'] == x_bin)), 'line_start_grid'] = grid_value
                DF.loc[((DF['y_line_end'] == y_bin) & (DF['X_line_end'] == x_bin)), 'line_end_grid'] = grid_value

            y_inc += len(labels)

        return DF

    print(path)
    f = open(path, "r", encoding = "utf8")
    o = f.read()
    f.close()
    j = json.loads(o)
    resultList = j["analyzeResult"]["readResults"]
    # pageList = j["analyzeResult"]["pageResults"]
    pageList = []

    rows = read_result_tag(resultList)
    if len(rows) == 0 :
        return None

    df = pd.DataFrame(rows)
    df = read_page_result_tag(pageList, df)
    df = correctAmountLineTokens(df)
    df = correctAmountTokens(df)
    #Extract Image features and split Line Text VLines here
    df = extract_image_features(df,imgpath)
    df.to_csv("temp_df_0.csv",index = False)
    df = splitLineTextVLines(df)
    df.to_csv("temp_df_1.csv",index = False)

    df = read_lines_from_table(df)
    df = token_distances(df)
    df = position_binning(df)

    return df

def main():
    imgpath = r"C:\Users\Hari\Downloads\doc_1635425116289_a9a9baa89b9-6-pre.tiff"
    ocrpath = r"C:\Users\Hari\Downloads\doc_1635425116289_a9a9baa89b9-6-pre.tiff.ocr.json"
    df = read_ocr_json_file(ocrpath, imgpath)
    print("completed")
    
if __name__ == "__main__":
    main()
    

