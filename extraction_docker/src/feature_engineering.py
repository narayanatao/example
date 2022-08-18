#!/usr/bin/env python
# coding: utf-8
# In[1]: All Imports

import json
import pandas as pd
import numpy as np
import re
import time
from price_parser import parse_price
import random
import cv2
import string
import traceback
import decimal
import os
# from fuzzywuzzy import process as fz
# from fuzzywuzzy import fuzz
from functools import reduce
from rapidfuzz import process as rp_fz, utils as rp_utils, fuzz as rp_fuzz

import ast

import warnings
warnings.filterwarnings("ignore")

import config as config
import util as util
from util import pan_pattern

# import regex
from get_bill2ship2_details import get_bill_to_ship_to_details

# In[get script directory]

script_dir = os.path.dirname(__file__)

# In[Declare variables]

random.seed(0)

ROOT_FOLDER = config.getRootFolderPath()
#  Blob container
outputFolder = config.getBlobOPFolder()
blobAccount = config.getBlobAccountName()
INDIA_ZIP_CODE_RANGE = config.getZipCodeFilePAth()

labelKeywords = util.getLabelKeywords()
labelKeywords_nonToken = util.getlabelKeywords_nonTokenized()
#Jul 11 2022 - New label keywords for weighted fuzzy score for left, above keywords
labelKeywords_nonToken_new = util.getlabelKeywords_nonTokenized_new()
#Jul 11 2022 - New label keywords for weighted fuzzy score for left, above keywords
# companyNameList = hdr_tokens['companyNameList']
# addressNameList = hdr_tokens["addressNameList"]
billToNames_kws = labelKeywords_nonToken["lblBillingNames"]
shipToNames_kws = labelKeywords_nonToken["lblShippingNames"]

df_ftr = util.getFeatureTypes()

li_ftrs = df_ftr[df_ftr["class"] == config.getLIFeaturesId()]["Column_Names"].to_list()
li_ftrs1 = df_ftr[df_ftr["class"] == config.getLI1FeaturesId()]["Column_Names"].to_list()
li_ngbr_ftrs = df_ftr[df_ftr["class"] == config.getLINgbrFeaturesId()]["Column_Names"].to_list()
spatial_ftrs = df_ftr[df_ftr["class"] == config.getSpatialFeaturesId()]["Column_Names"].to_list()
numeric_ftrs = df_ftr[df_ftr["class"] == config.getNumericFeaturesId()]["Column_Names"].to_list()
ocr_ftrs = df_ftr[df_ftr["class"] == config.getOCRFeaturesId()]["Column_Names"].to_list()
non_mdl_ftrs = df_ftr[df_ftr["model_input"] == 0]["Column_Names"].to_list()

# gstin Number Format
gstin_pattern, _ = util.gstin_pattern()
_, EMAIL_REGEX = util.email_pattern()

foo = lambda x: pd.Series([i for i in x])

gst_rates = config.getGSTRates()


# In[spacy loading]

print("spacy loading")
#spacy commented
# nlp = en.load(disable=["parser"])
# nlp.add_pipe(nlp.create_pipe('sentencizer'))

print("spacy loaded")

# In[Load Keywords]

LIKeyWordsPath = config.getLIKeywordsPath()

with open(LIKeyWordsPath) as data:
    LIKeyWordData = json.load(data)

cat_encoding = LIKeyWordData['cat_encoding']
noOfPuncs_list = LIKeyWordData['noOfPuncs_list']

lineInfo = LIKeyWordData['lineInfo']

po_shapes = LIKeyWordData['po_shapes']

pixelThresh = LIKeyWordData['pixelThresh']
verticalThresh = LIKeyWordData['verticalThresh']
sideThresh = LIKeyWordData['sideThresh']

neighbWordsVec = LIKeyWordData['neighbWordsVec']
noNeighbors = LIKeyWordData['noNeighbors']
aggregated_neighbours = {"Above":["W" + str(i) + "Ab" for i in range(1, noNeighbors+1)],
                         "Left":["W" + str(i) + "Lf" for i in range(1, noNeighbors+1)]}

specific = LIKeyWordData['specific']
common = LIKeyWordData['common']
vendor_specific = LIKeyWordData['vendor_specific']
hdr_specific = specific + common + vendor_specific

hdr_values = LIKeyWordData['hdr_values']
hdr_amount = LIKeyWordData['hdr_amount']
footer_keywords = LIKeyWordData['footer_keywords']
desc_key = LIKeyWordData['desc_key']
desc_vendor = LIKeyWordData['desc_vendor']
code_key = LIKeyWordData['code_key']
code_vendor = LIKeyWordData['code_vendor']
qty_key = LIKeyWordData['qty_key']
qty_vendor = LIKeyWordData['qty_vendor']
price_key = LIKeyWordData['price_key']
price_vendor = LIKeyWordData['price_vendor']
val_key = LIKeyWordData['val_key']
val_vendor = LIKeyWordData['val_vendor']
tax_key = LIKeyWordData['tax_key']
uom_key = LIKeyWordData['uom_key']
uom_val = LIKeyWordData['uom_val']
hsn_key = LIKeyWordData['hsn_key']
tax_rate_key = LIKeyWordData['tax_rate_key']
cgst = LIKeyWordData['cgst']
sgst = LIKeyWordData['sgst']
igst = LIKeyWordData['igst']
disc_key = LIKeyWordData['discount_key']

code_key1 = LIKeyWordData['code_key1']
qty_key1 = LIKeyWordData['qty_key1']
price_key1 = LIKeyWordData['price_key1']
val_key1 = LIKeyWordData['val_key1']
uom_key1 = LIKeyWordData['uom_key1']
disc_key1 = LIKeyWordData['discount_key1']
hsn_key1 = LIKeyWordData['hsn_key1']
tax_rate_key1 = LIKeyWordData['tax_rate_key1']
cgst1 = LIKeyWordData['cgst1']
sgst1 = LIKeyWordData['sgst1']
igst1 = LIKeyWordData['igst1']
disc_key1 = LIKeyWordData['discount_key1']
desc_key.extend(desc_vendor)

LI_FIELDS = LIKeyWordData["LI_FIELDS"]

surrounding_label_feature = config.getSurroundLabelFeatures()


# In[4]: Declare Functions to extract feastures for line items
# @util.timing
def check_hdr(DF,hdr_line):
    try:
        texts = DF.loc[(DF["tableLineNo"] == hdr_line)]["text"].to_list()
        line_texts = DF.loc[(DF["tableLineNo"] == hdr_line)]["line_text"].to_list()
        # no_count = sum([1 if (util.isNumber(text) or util.isAmount(text))
        #                 else 0 for text,line_text in zip(texts,line_texts)])
        no_count = sum([1 if (util.isAmount(text))
                        else 0 for text,line_text in zip(texts,line_texts)])
        if no_count > 0:
            return False
        else:
            return True
    except:
        print("Failed in checking if it's header",
              traceback.print_exc())
        return True

# @util.timing
def getHdrScore_1(row):
    text_list = row['text']
    text_list = list(set([str(text).lower().strip() for text in text_list]))
    score = score = sum([1 if text in hdr_specific else 0 for text in text_list])
    return score

# @util.timing
def getHdrScore(row):
    punc = string.punctuation.replace('#', '').replace('%', '')
    punc = punc + '0123456789'
    text_list = row['line_text']
    text_list = [str(text).lower().strip().translate(
        str.maketrans('', '',punc)) for text in text_list]

    score = 0
    for text in text_list:
        words = text.split(" ")
        words = [text for text in words if text.strip() != '']
        words = [text for text in words if len(text.strip()) > 2]

        if len(words) <= 5:
            # print(text)
            result = rp_fz.extract(text,
                                   hdr_amount,
                                   scorer = rp_fuzz.token_set_ratio)
            if result is None:
                amt_max = 0
            else:
                amt_max = max([res[1] for res in result])

            # print(text)
            result = rp_fz.extract(text,
                                   hdr_values,
                                   scorer = rp_fuzz.token_set_ratio)
            if result is None:
                hdr_max = 0
            else:
                hdr_max = max([res[1] for res in result])

            # print(text)
            result = rp_fz.extract(text,
                                   vendor_specific,
                                   scorer = rp_fuzz.token_set_ratio)
            if result is None:
                vendor_max = 0
            else:
                vendor_max = max([res[1] for res in result])

            max_list = [amt_max,
                        hdr_max,
                        vendor_max]
            max_val = max(max_list)

            if max_val < 75:
                max_val = 0
            if max_val in max_list:
                max_ind = max_list.index(max_val)
            else:
                max_ind = 0
            score += max_val * (max_ind + 1)

    return score

@util.timing
def addLineItemFeatures(DF):

    def fz_match_hdrTxt(txt,keys):
        try:
            if isinstance(keys[0],dict):
                scores = []
                for word_weights in keys:
                    score = 0.0
                    for word_weight in word_weights.keys():
                        words = word_weights[word_weight]
                        #Change for using RapidFuzz instead of FuzzyWuzzy
                        # part_score = fz.extractBests(txt,
                        #                              words,
                        #                              scorer = fuzz.WRatio,
                        #                              limit = 1)[0][1] / 100
                        part = rp_fz.extractOne(txt,
                                                words,
                                                scorer = rp_fuzz.WRatio)
                        if part is None:
                            part_score = 0
                        else:
                            part_score = part[1] / 100
                        score += part_score * float(word_weight)
                    scores.append(score)
                score = max(scores)
                print("Fuzzy Score:",txt,keys,score)
                return score
            else:
                # score = fz.extractBests(txt,
                #                 keys,
                #                 scorer = fuzz.WRatio,
                #                 limit = 1)[0][1] / 100
                match = rp_fz.extractOne(txt,
                                         keys,
                                         scorer = rp_fuzz.WRatio)
                if match is None:
                    score = 0
                else:
                    score = match[1] / 100
                return score
        except:
            print("Failed in fz_match_hdrTxt",
                  traceback.print_exc())
            return 0.0

    #Initialize all the line item features
    DF["score"] = 0
    DF["score_1"] = 0
    DF["is_HDR"] = 0
    DF["line_noanchor"] = 0
    DF["line_ngbr"] = 0
    DF["line_dist"] = 0
    DF["line_valign"] = 0
    DF["line_amount"] = 0
    DF["line_item"] = 0
    DF["line_row"] = 0
    DF["amount_in_line"] = 0
    DF["is_item_desc"] = 0
    DF["is_item_code"] = 0
    DF["is_qty"] = 0
    DF["is_unit_price"] = 0
    DF["is_item_val"] = 0
    DF["is_uom"] = 0
    DF["is_hsn_key"] = 0
    DF["is_tax_rate_key"] = 0
    DF["is_cgst"] = 0
    DF["is_sgst"] = 0
    DF["is_igst"] = 0
    DF["is_disc"] = 0

    DF["is_qty1"] = 0
    DF["is_unit_price1"] = 0
    DF["is_item_val1"] = 0
    DF["is_uom1"] = 0
    DF["is_hsn_key1"] = 0
    DF["is_tax_rate_key1"] = 0
    DF["is_cgst1"] = 0
    DF["is_sgst1"] = 0
    DF["is_igst1"] = 0
    DF["is_disc1"] = 0

    #Header Line Identification
    #Score the header line using fuzzy logic
    print("Updated line item columns")
    t = time.time()
    DF.sort_values(["page_num","line_num","word_num"],
                   ascending=[True,True,True])
    DF_hdr = DF.groupby(['tableLineNo'])['line_text'].unique().apply(list).reset_index()
    DF_hdr['score'] = DF_hdr.apply(getHdrScore, axis=1)
    for ind, row in DF_hdr.iterrows():
        DF.loc[(DF["tableLineNo"] == row["tableLineNo"]),
                "score"] = row['score']

    print("UpdateScore: ", time.time() - t)

    t = time.time()
    DF_hdr_1 = DF.groupby(['tableLineNo'])['text'].apply(list).reset_index()
    DF_hdr_1['score'] = DF_hdr_1.apply(getHdrScore_1, axis=1)
    for ind, row in DF_hdr_1.iterrows():
        DF.loc[(DF["tableLineNo"] == row["tableLineNo"]),
                "score_1"] = row['score']

    print("UpdateScore 2: ", time.time() - t)


    #Update header line
    t = time.time()
    lines = list(DF_hdr["tableLineNo"])
    lines = sorted(lines)

    hdr_val = DF_hdr.nlargest(1,["score"])["score"].values[0]

    hdr_line = min(lines)
    hdr_found = False
    if hdr_val > 200:
        DF_hdr_filt = DF_hdr[DF_hdr["score"] == hdr_val]
        DF_hdr_filt_len = DF_hdr_filt.shape[0]

        if DF_hdr_filt_len == 1:

            hdr_line = DF_hdr_filt["tableLineNo"].values[0]
            print("Initial Header Line:", hdr_line, hdr_val)
            #Check if any amount is there on header line. If it is, then it's not a header
            if check_hdr(DF,hdr_line):
                DF.loc[(DF["tableLineNo"] == hdr_line),"is_HDR"] = 1
                hdr_found = True
                hdr_line_comp = hdr_line

            #Check if below wrapped lines are part of the header
            line_down = max(list(set(list(DF[DF["tableLineNo"] == hdr_line]["line_down"]))))
            scr = 1
            rowcount = 0
            while scr > 0:
                lines = sorted(lines)
                print("Lines:",lines)
                next_lines = lines[lines.index(hdr_line)+1:]
                print("Next Line Start:",hdr_line,
                      (len(next_lines) > 0) and (rowcount < len(next_lines)))
                if (len(next_lines) > 0) and (rowcount < len(next_lines)):
                    #Hdr line is updated with next line. Hence always taking from first index
                    #For prev Line, this is different
                    next_line = next_lines[0]
                    # print("Next Line is:", next_line)
                    line_top = min(list(set(list(DF[DF["tableLineNo"] == next_line]["line_top"]))))
                    # print("line_top",line_top,line_down)
                    if (line_top - line_down < .02) or True:
                        line_down = max(list(set(list(DF[DF["tableLineNo"] == next_line]["line_down"]))))
                        scr = DF_hdr[DF_hdr["tableLineNo"] == next_line]["score"].values[0]
                        print("Next Line is",next_line)
                        if scr >= 200:
                        # if scr >= 100:
                            # print("Score of next line",next_line," is higher")
                            if check_hdr(DF,next_line):
                                # print("Next Line", next_line, " is now hdr_line")
                                hdr_line = next_line
                                DF.loc[(DF["tableLineNo"] == hdr_line),
                                       "is_HDR"] = 1
                                if not hdr_found:
                                    hdr_found = True
                                    hdr_line_comp = hdr_line
                        else:
                            scr = 0
                    else:
                        scr = 0
                    rowcount += 1
                    if rowcount > 2 or rowcount >= len(next_lines):
                        scr = 0
                else:
                    scr = 0

            #Check if top wrapped lines are part of the header
            # print("check if lines top are part of the header")
            line_top = max(list(set(list(DF[DF["tableLineNo"] == hdr_line]["line_top"]))))
            scr = 1
            rowcount = 0
            while scr > 0:
                if lines.index(hdr_line) == 0:
                    break
                prev_lines = lines[:lines.index(hdr_line) - 1][::-1]
                if len(prev_lines) > 0 and (rowcount < len(prev_lines)):
                    # print("Previous lines",len(prev_lines),rowcount)
                    prev_line = prev_lines[rowcount]
                    #print("initial previous line",prev_line)
                    line_down = min(list(set(list(DF[DF["tableLineNo"] == prev_line]["line_down"]))))
                    if line_top - line_down < .02:
                        line_top = max(list(set(list(DF[DF["tableLineNo"] == prev_line]["line_top"]))))
                        scr = DF_hdr[DF_hdr["tableLineNo"] == prev_line]["score"].values[0]
                        if scr >= 200:
                            if check_hdr(DF,prev_line):
                                DF.loc[(DF["tableLineNo"] == prev_line),
                                       "is_HDR"] = 1
                                if not hdr_found:
                                    hdr_found = True
                                    hdr_line_comp = hdr_line
                        else:
                            scr = 0
                    else:
                        scr = 0
                    rowcount += 1
                    if rowcount > 2 or rowcount >= len(prev_lines):
                        scr = 0
                else:
                    scr = 0

    if not hdr_found:
        return DF

    line_noanchor = []
    line_valign = []
    line_ngbr = []
    line_dist = []

    pot_lines = lines[lines.index(hdr_line) + 1:]

    punc = string.punctuation.replace('#', '').replace('/', '')
    punc = punc + '0123456789'

    if len(pot_lines) > 0:

        #Check if lines have anchor words
        t = time.time()
        found = False
        for line_no in pot_lines:
            if found:
                break
            DF_line = DF[DF['tableLineNo'] == line_no]

            for i,row in DF_line.iterrows():
                line_text = str(row["line_text"]).lower().strip()
                line_text = line_text.translate(str.maketrans('', '', punc))

                # matches = fz.extractBests(line_text,
                #                           footer_keywords,
                #                           scorer = fuzz.ratio,
                #                           score_cutoff = 85,
                #                           limit = 1)
                # if len(matches) > 0:
                #     found = True
                #     if line_no in line_noanchor:
                #         line_noanchor.remove(line_no)
                #     break
                # else:
                #     if line_no not in line_noanchor:
                #         line_noanchor.append(line_no)

                matches = rp_fz.extractOne(line_text,
                                           footer_keywords,
                                           scorer = rp_fuzz.ratio,
                                           score_cutoff = 85)
                if matches:
                    found = True
                    if line_no in line_noanchor:
                        line_noanchor.remove(line_no)
                    break
                else:
                    if line_no not in line_noanchor:
                        line_noanchor.append(line_no)


        #Check if lines have same amount of neighbours -
        #THIS WILL WORK WELL FOR LINE TABLE WITH MORE LINE ITEMS
        t = time.time()
        if len(line_noanchor) >  0:
            noanchor = max(line_noanchor)
        else:
            noanchor = pot_lines[:-2:-1][0]

        pot_reduced_lines = pot_lines[:pot_lines.index(noanchor)+1]

        ngbrs = []
        for line_no in pot_reduced_lines:
            DF_line = DF[DF['tableLineNo'] == line_no]
            neighbour = max(list(DF_line["noNeighbours"]))
            ngbrs.append(neighbour)

        rowparts = list(range(min(len(pot_reduced_lines),5),0,-1))

        found = False
        rowpart = 0
        times = 0
        for m in rowparts:
            sum_pattern = [sum(ngbrs[i*m:i*m+m]) for i in range(len(ngbrs)//m)]
            diffs = [sum_pattern[j] - sum_pattern[j+1]
                     for j in range(len(sum_pattern)-1)]
            sum_diffs = 0
            for i in range(len(diffs)):
                if abs(diffs[i]) <= 1:
                    sum_diffs += diffs[i]
                    if sum_diffs > 2:
                        break
                    else:
                        found = True
                        rowpart = m
                        times = i + 1
                else:
                    break
            if found:
                break

        if rowpart == 0:
            line_ngbr.append(pot_lines[0])
        else:
            ntimes = rowpart * times
            if ntimes <= len(pot_reduced_lines):
                line_ngbr.extend(pot_reduced_lines[:ntimes])
            else:
                line_ngbr.extend(pot_reduced_lines[:])

        #Check if distance between the lines have a pattern -
        #THIS WILL WORK WELL FOR LINE TABLE WITH MORE COLUMNS
        t = time.time()
        pot_reduced_lines = pot_lines[:pot_lines.index(noanchor)+1]
        rowparts = list(range(min(len(pot_reduced_lines),10),0,-1))
        coords = []
        for line_ind, line_no in enumerate(pot_reduced_lines):
            DF_line = DF[DF['tableLineNo'] == line_no]
            line_top = max(list(DF_line["line_top"]))
            line_down = min(list(DF_line["line_down"]))
            coords.append((line_down,line_top))

        #print("Line Items before line distance",pot_reduced_lines)
        line_dist.append(pot_reduced_lines[0])
        for coord_ind in range(len(coords) - 1):
            dist = coords[coord_ind + 1][1] - coords[coord_ind][0]
            if dist <= .1:
                line_dist.append(pot_reduced_lines[coord_ind + 1])
            else:
                break

        #Check if most of the columns are aligned vertically
        t = time.time()
        pot_reduced_lines = pot_lines[:pot_lines.index(noanchor)+1]
        line_valign.append(pot_reduced_lines[0])
        hdrs = DF[DF["tableLineNo"] == pot_reduced_lines[0]][["line_down",
                                                              "line_left",
                                                              "line_right",
                                                              "line_text"]
                                                             ].drop_duplicates(
                                                                 keep = 'first'
                                                                 ).apply(dict)
        first_lefts = list(hdrs["line_left"].values())
        first_rights = list(hdrs["line_right"].values())
        line_down = max(list(hdrs["line_down"].values()))
        lines_notfound = 0

        for line_ind in range(1,len(pot_reduced_lines)):
            line = DF[DF["tableLineNo"] == pot_reduced_lines[line_ind]][
                ["line_top","line_left",
                 "line_right","line_text"]].drop_duplicates(
                     keep = 'first').apply(dict)
            lefts = list(line["line_left"].values())
            rights = list(line["line_right"].values())
            line_top = min(list(line["line_top"].values()))
            dist = line_top - line_down
            mtch = 0
            if dist < .1:
                for j in zip(first_lefts,first_rights):
                    for k in zip(lefts,rights):
                        if (k[0] > j[1]) or (k[1] < j[0]):
                            continue
                        else:
                            mtch += 1
            else:
                mtch = 0
                break
            if mtch == 0:
                lines_notfound += 1
            else:
                lines_notfound = 0
                line_valign.append(pot_reduced_lines[line_ind])
                hdrs = DF[DF["tableLineNo"] == pot_reduced_lines[line_ind]][["line_down","line_left",
                         "line_right","line_text"]].drop_duplicates(keep = 'first').apply(dict)
                first_lefts = list(hdrs["line_left"].values())
                first_rights = list(hdrs["line_right"].values())
                line_down = max(list(hdrs["line_down"].values()))
            if lines_notfound > 3:
                line_down = 0
                break

        #Update row breaks
        t = time.time()
        max_lefts = []
        max_alls = []
        line_dist_align = list(set(line_dist).union(set(line_valign)))
        line_dist_align = sorted(line_dist_align)
        for line_no in line_dist_align:
            lefts = list(DF[DF["tableLineNo"] == line_no]["line_left"])
            max_left = max(lefts)
            max_lefts.append(max_left)
            max_ind = lefts.index(max_left)

            rights = list(DF[DF["tableLineNo"] == line_no]["line_right"])
            max_right = rights[max_ind]

            max_alls.append((max_left,max_right,line_no))

        max_left = max(max_lefts)
        max_ind = max_lefts.index(max_left)
        max_all = max_alls[max_ind]
        max_right = max_all[1]
        row_number = 1
        row_first = False

        for all_ind,all_ in enumerate(max_alls):
            left = all_[0]
            right = all_[1]
            line_no = all_[2]

            DF.loc[(DF["tableLineNo"] == line_no),
                   "line_row"] = row_number
            if (right < max_left) or (left > max_right):
                continue
            else:
                if all_ind == 0:
                    row_number = 0
                    row_first = True
                row_number += 1
                if row_first:
                    DF.loc[(DF["tableLineNo"] == line_no),
                           "line_row"] = row_number

        #Check if lines have amount and how many amount fields are there
        t = time.time()
        pot_reduced_lines = pot_lines[:pot_lines.index(noanchor)+1]
        line_nums = []

        for line_no in pot_reduced_lines:
            texts = list(DF[DF["tableLineNo"] == line_no]["line_text"])
            count = sum([util.isAmount(text) for text in texts])
            if count == 0:
                count = sum([util.isNumber(text) for text in texts])

            DF.loc[(DF["tableLineNo"] == line_no)
                   ,["line_amount","amount_in_line"]] = [int(count > 0),count]
            if count > 0:
                line_nums.append(line_no)


        #Check the line item type
        t = time.time()
        DF = DF.sort_values(by = ["line_num"],ascending = [True])
        hdrs = DF[DF["tableLineNo"] == hdr_line_comp][["line_text",
                  "line_left","line_top","line_down",
                  "line_right"]].drop_duplicates(keep = 'first').apply(dict)

        print("Get Text and Dimensions of other lines")
        oth_hdrs = DF[DF["is_HDR"] == 1][
            ["line_text",
             "line_top",
             "line_down",
             "line_left",
             "line_right",
             "line_num"]].drop_duplicates(keep = 'first').apply(dict)
        print("Get Text and Dimensions of other lines","succes")

        print("List Text and Dimensions of other lines")
        oth_hdr_lefts = list(oth_hdrs["line_left"].values())
        oth_hdr_rights = list(oth_hdrs["line_right"].values())
        oth_hdr_tops = list(oth_hdrs["line_top"].values())
        oth_hdr_downs = list(oth_hdrs["line_down"].values())
        oth_hdr_texts = list(oth_hdrs["line_text"].values())
        oth_line_nums = list(oth_hdrs["line_num"].values())
        others = list(zip(oth_hdr_lefts,oth_hdr_tops,oth_hdr_rights,
                          oth_hdr_downs,oth_hdr_texts,oth_line_nums))
        print("List Text and Dimensions of other lines","success")

        #Append header text with prev and next lines if header is wrapped around
        print("Append Header Text")
        othrindx = []
        hdr_texts = []
        # print("Others:",others)
        for ind,obj in enumerate(others):
            hdr_texts.append("")
            for other_ind,other in enumerate(others):
                # print("Other vals:",
                #       obj[4],
                #       other[4],
                #       obj == other,
                #       (obj != other) and (obj[5] < other[5]))
                if (obj != other) and (obj[5] < other[5]) and (
                        other_ind not in othrindx) and (ind not in othrindx):
                    if (obj[1] > other[1]) and not ((obj[0] > other[2]) or (
                            obj[2] < other[0])):
                        hdr_texts[ind] = other[4] + " " + hdr_texts[ind]
                        oth_hdr_lefts[ind] = min(obj[0],other[0])
                        oth_hdr_rights[ind] = max(obj[2],other[2])
                        # print("Found match prev",hdr_texts[ind],other[4])
                        othrindx.append(other_ind)
                    elif (obj[1] < other[1]) and not ((obj[0] > other[2]) or (
                            obj[2] < other[0])):
                        hdr_texts[ind] = hdr_texts[ind] + " " + other[4]
                        oth_hdr_lefts[ind] = min(obj[0],other[0])
                        oth_hdr_rights[ind] = max(obj[2],other[2])
                        # print("Found match next",hdr_texts[ind],other[4])
                        othrindx.append(other_ind)
                elif obj == other:
                    hdr_texts[ind] = obj[4]
                    oth_hdr_lefts[ind] = obj[0]
                    oth_hdr_rights[ind] = obj[2]
                    if (ind in othrindx):
                        hdr_texts[ind] = " "
                # print("header text:",hdr_texts[ind])
        # print("Header Texts:",hdr_texts)


        line_dist_valign = list(set(line_dist).union(set(line_valign)))
        line_dist_valign = sorted(line_dist_valign)
        for line_no in line_dist_valign:
            #Working code before 15th June 2021
            line = DF[DF["tableLineNo"] == line_no][
                ["text","left","right","top"]].drop_duplicates(
                    keep = 'first').apply(dict)
            lefts = list(line["left"].values())
            rights = list(line["right"].values())
            line_top = min(list(line["top"].values()))
            texts = list(line["text"].values())
            for ind,coord in enumerate(zip(lefts,rights,texts)):
                is_item_desc_fz = 0
                is_item_code_fz = 0
                is_qty_fz = 0
                is_unit_price_fz = 0
                is_item_val_fz = 0
                is_uom_fz = 0
                is_hsn_key_fz = 0
                is_tax_rate_key_fz = 0
                is_cgst_fz = 0
                is_sgst_fz = 0
                is_igst_fz = 0
                is_disc_fz = 0

                is_qty1_fz = 0
                is_unit_price1_fz = 0
                is_item_val1_fz = 0
                is_uom1_fz = 0
                is_hsn_key1_fz = 0
                is_tax_rate_key1_fz = 0
                is_cgst1_fz = 0
                is_sgst1_fz = 0
                is_igst1_fz = 0
                is_disc1_fz = 0
                align_cnt = 0

                maxBoundingArea = 0.0

                for hdr_ind,hdr_coord in enumerate(zip(oth_hdr_lefts,
                                                       oth_hdr_rights)):
                    # print("Find Matching Header Column for:",coord[2])
                    # print("Header text",hdr_texts[hdr_ind])
                    # print("Match:",not((hdr_coord[0] > coord[1]) or
                    #                     (hdr_coord[1] < coord[0])))
                    # print("Coords:",hdr_coord[0],hdr_coord[1],
                    #       coord[0],coord[1])
                    if (hdr_coord[0] > coord[1]) or (hdr_coord[1] < coord[0]):
                        continue
                    else:
                        minLeft = min(hdr_coord[0],coord[0])
                        minRight = min(hdr_coord[1],coord[1])
                        maxLeft = max(hdr_coord[0],coord[0])
                        maxRight = max(hdr_coord[1],coord[1])
                        boundingArea = ((minRight - maxLeft)/(maxRight - minLeft))
                        if boundingArea < maxBoundingArea:
                            continue
                        maxBoundingArea = boundingArea
                        hdr_text = hdr_texts[hdr_ind]
                        print("Matching header text:", hdr_text)
                        if len(hdr_text.strip()) <= 2:
                            continue
                        hdr_text = str(hdr_text).lower().strip()
                        # fz_scores_exist = DF.loc[(DF["tableLineNo"] == line_no) &
                        #                          (DF["left"] == coord[0]) &
                        #                          (DF["right"] == coord[1])]
                        # ["is_item_desc","is_item_code",
                        #  "is_item_val","is_unit_price",
                        #  "is_uom","is_qty","is_hsn_key",
                        #  "is_tax_rate_key","is_cgst","is_sgst",
                        #  "is_igst","is_disc",
                        #  "is_qty1","is_unit_price1",
                        #  "is_item_val1","is_uom1",
                        #  "is_hsn_key1","is_tax_rate_key1",
                        #  "is_cgst1","is_sgst1","is_igst1",
                        #  "is_disc1"]
                        align_cnt += 1

                        is_item_desc_fz += fz_match_hdrTxt(hdr_text,
                                                           desc_key)
                        is_item_code_fz += fz_match_hdrTxt(hdr_text,
                                                           code_key)
                        is_qty_fz += fz_match_hdrTxt(hdr_text,
                                                     qty_key)
                        is_qty1_fz += fz_match_hdrTxt(hdr_text,
                                                     qty_key1)
                        is_unit_price_fz += fz_match_hdrTxt(hdr_text,
                                                            price_key)
                        is_unit_price1_fz += fz_match_hdrTxt(hdr_text,
                                                            price_key1)
                        is_item_val_fz += fz_match_hdrTxt(hdr_text,
                                                          val_key)
                        is_item_val1_fz += fz_match_hdrTxt(hdr_text,
                                                          val_key1)
                        is_uom_fz += fz_match_hdrTxt(hdr_text,
                                                     uom_key)
                        is_uom1_fz += fz_match_hdrTxt(hdr_text,
                                                     uom_key1)
                        is_hsn_key_fz += fz_match_hdrTxt(hdr_text,
                                                         hsn_key)
                        is_hsn_key1_fz += fz_match_hdrTxt(hdr_text,
                                                         hsn_key1)
                        is_tax_rate_key_fz += fz_match_hdrTxt(hdr_text,
                                                              tax_rate_key)
                        is_tax_rate_key1_fz += fz_match_hdrTxt(hdr_text,
                                                              tax_rate_key1)
                        is_cgst_fz += fz_match_hdrTxt(hdr_text,
                                                      cgst)
                        is_cgst1_fz += fz_match_hdrTxt(hdr_text,
                                                      cgst1)
                        is_sgst_fz += fz_match_hdrTxt(hdr_text,
                                                      sgst)
                        is_sgst1_fz += fz_match_hdrTxt(hdr_text,
                                                      sgst1)
                        is_igst_fz += fz_match_hdrTxt(hdr_text,
                                                      igst)
                        is_igst1_fz += fz_match_hdrTxt(hdr_text,
                                                      igst1)
                        is_disc_fz += fz_match_hdrTxt(hdr_text,
                                                      disc_key)
                        is_disc1_fz += fz_match_hdrTxt(hdr_text,
                                                      disc_key1)


                        # is_item_desc_fz += max(fz_match_hdrTxt(hdr_text,
                        #                                    desc_key),
                        #                        fz_scores_exist["is_item_desc"])
                        # is_item_code_fz += max(fz_match_hdrTxt(hdr_text,
                        #                                    code_key),
                        #                        fz_scores_exist["is_item_code"])
                        # is_qty_fz += max(fz_match_hdrTxt(hdr_text,
                        #                              qty_key),
                        #                  fz_scores_exist["is_qty"])
                        # is_qty1_fz += max(fz_match_hdrTxt(hdr_text,
                        #                              qty_key1),
                        #                   fz_scores_exist["is_qty1"])
                        # is_unit_price_fz += max(fz_match_hdrTxt(hdr_text,
                        #                                     price_key),
                        #                         fz_scores_exist["is_unit_price"])
                        # is_unit_price1_fz += max(fz_match_hdrTxt(hdr_text,
                        #                                     price_key1),
                        #                          fz_scores_exist["is_unit_price1"])
                        # is_item_val_fz += max(fz_match_hdrTxt(hdr_text,
                        #                                   val_key),
                        #                       fz_scores_exist["is_item_val"])
                        # is_item_val1_fz += max(fz_match_hdrTxt(hdr_text,
                        #                                   val_key1),
                        #                        fz_scores_exist["is_item_val1"])
                        # is_uom_fz += max(fz_match_hdrTxt(hdr_text,
                        #                              uom_key),
                        #                  fz_scores_exist["is_uom"])
                        # is_uom1_fz += max(fz_match_hdrTxt(hdr_text,
                        #                              uom_key1),
                        #                   fz_scores_exist["is_uom1"])
                        # is_hsn_key_fz += max(fz_match_hdrTxt(hdr_text,
                        #                                  hsn_key),
                        #                      fz_scores_exist["is_hsn_key"])
                        # is_hsn_key1_fz += max(fz_match_hdrTxt(hdr_text,
                        #                                  hsn_key1),
                        #                       fz_scores_exist["is_hsn_key1"])
                        # is_tax_rate_key_fz += max(fz_match_hdrTxt(hdr_text,
                        #                                       tax_rate_key),
                        #                           fz_scores_exist["is_tax_rate_key"])
                        # is_tax_rate_key1_fz += max(fz_match_hdrTxt(hdr_text,
                        #                                       tax_rate_key1),
                        #                            fz_scores_exist["is_tax_rate_key1"])
                        # is_cgst_fz += max(fz_match_hdrTxt(hdr_text,
                        #                               cgst),
                        #                   fz_scores_exist["is_cgst"])
                        # is_cgst1_fz += max(fz_match_hdrTxt(hdr_text,
                        #                               cgst1),
                        #                    fz_scores_exist["is_cgst1"])
                        # is_sgst_fz += max(fz_match_hdrTxt(hdr_text,
                        #                               sgst),
                        #                   fz_scores_exist["is_sgst"])
                        # is_sgst1_fz += max(fz_match_hdrTxt(hdr_text,
                        #                               sgst1),
                        #                    fz_scores_exist["is_sgst1"])
                        # is_igst_fz += max(fz_match_hdrTxt(hdr_text,
                        #                               igst),
                        #                   fz_scores_exist["is_igst"])
                        # is_igst1_fz += max(fz_match_hdrTxt(hdr_text,
                        #                               igst1),
                        #                    fz_scores_exist["is_igst1"])
                        # is_disc_fz += max(fz_match_hdrTxt(hdr_text,
                        #                               disc_key),
                        #                   fz_scores_exist["is_disc"])
                        # is_disc1_fz += max(fz_match_hdrTxt(hdr_text,
                        #                               disc_key1),
                        #                    fz_scores_exist["is_disc1"])

                        fz_scores = [is_item_desc_fz/align_cnt,
                                     is_item_code_fz/align_cnt,
                                     is_item_val_fz/align_cnt,
                                     is_unit_price_fz/align_cnt,
                                     is_uom_fz/align_cnt,
                                     is_qty_fz/align_cnt,
                                     is_hsn_key_fz/align_cnt,
                                     is_tax_rate_key_fz/align_cnt,
                                     is_cgst_fz/align_cnt,
                                     is_sgst_fz/align_cnt,
                                     is_igst_fz/align_cnt,
                                     is_disc_fz/align_cnt]
                        fz_scores1 = [is_qty1_fz/align_cnt,
                                      is_unit_price1_fz/align_cnt,
                                      is_item_val1_fz/align_cnt,
                                      is_uom1_fz/align_cnt,
                                      is_hsn_key1_fz/align_cnt,
                                      is_tax_rate_key1_fz/align_cnt,
                                      is_cgst1_fz/align_cnt,
                                      is_sgst1_fz/align_cnt,
                                      is_igst1_fz/align_cnt,
                                      is_disc1_fz/align_cnt]
                        # fz_scores = fz_scores + fz_scores1

                        print("Item Desc:",is_item_desc_fz/align_cnt)
                        print("Item Code:",is_item_code_fz/align_cnt)
                        print("Item Val:",is_item_val_fz/align_cnt)
                        print("Item Rate:",is_unit_price_fz/align_cnt)
                        print("Item UOM:",is_uom_fz/align_cnt)
                        print("Item Qty:",is_qty_fz/align_cnt)
                        print("Item HSN:",is_hsn_key_fz/align_cnt)
                        print("Tax Rate:",is_tax_rate_key_fz/align_cnt)
                        print("CGST:",is_cgst_fz/align_cnt)
                        print("SGST:",is_sgst_fz/align_cnt)
                        print("IGST:",is_igst_fz/align_cnt)

                        print("Fuzzy Score adjustment started",fz_scores)
                        gt95 = sum([int(scr > .95) for scr in fz_scores])
                        if gt95 == 1:
                            fz_scores = [min(i,.5) if i <=.95 else i for i in fz_scores]
                        print("Fuzzy Scores:", fz_scores)

                        DF.loc[(DF["tableLineNo"] == line_no) &
                               (DF["left"] == coord[0]) &
                               (DF["right"] == coord[1])
                               ,["is_item_desc",
                                 "is_item_code",
                                 "is_item_val",
                                 "is_unit_price",
                                 "is_uom",
                                 "is_qty",
                                 "is_hsn_key",
                                 "is_tax_rate_key",
                                 "is_cgst",
                                 "is_sgst",
                                 "is_igst",
                                 "is_disc",
                                 "is_qty1",
                                 "is_unit_price1",
                                 "is_item_val1",
                                 "is_uom1",
                                 "is_hsn_key1",
                                 "is_tax_rate_key1",
                                 "is_cgst1",
                                 "is_sgst1",
                                 "is_igst1",
                                 "is_disc1"]] = fz_scores + fz_scores1

        union = list(set(line_noanchor).union(set(line_ngbr)).union(
            set(line_dist)).union(set(line_valign)))

        t = time.time()
        if hdr_found:
            for line_no in union:
                score = 0
                cnt = 0
                anchor = 0
                ngbr = 0
                valign = 0
                dist = 0
                anchor = int(line_no in line_noanchor)
                score += anchor
                cnt += 1

                ngbr = int(line_no in line_ngbr)
                score += ngbr
                cnt += 1

                dist = int(line_no in line_dist)
                score += dist
                cnt += 1

                valign = int(line_no in line_valign)
                score += valign
                cnt += 1

                DF.loc[(DF["tableLineNo"] == line_no),
                       ["line_noanchor","line_ngbr",
                        "line_valign","line_dist",
                        "line_item"]] = [anchor, ngbr,
                                         valign, dist,
                                         score / cnt]


    DF['number_sum'] = DF['is_number'] + DF['is_amount']

    def updateScores(x):
        if x.line_item == 0:
            return pd.Series([0]*22)
        else:
            return pd.Series([x.is_item_desc,
                              x.is_item_code,
                              x.is_item_val,
                              x.is_unit_price,
                              x.is_uom, x.is_qty,
                              x.line_item,
                              x.is_hsn_key,
                              x.is_tax_rate_key,
                              x.is_cgst,
                              x.is_sgst,
                              x.is_igst,
                              x.is_qty1,
                              x.is_unit_price1,
                              x.is_item_val1,
                              x.is_uom1,
                              x.is_hsn_key1,
                              x.is_tax_rate_key1,
                              x.is_cgst1,
                              x.is_sgst1,
                              x.is_igst1,
                              x.is_disc1])

    DF[['is_item_desc',
        'is_item_code',
        'is_item_val',
        'is_unit_price',
        'is_uom',
        'is_qty',
        'line_item',
        'is_hsn_key',
        'is_tax_rate_key',
        'is_cgst',
        'is_sgst',
        'is_igst',
        'is_qty1',
        'is_unit_price1',
        'is_item_val1',
        'is_uom1',
        'is_hsn_key1',
        'is_tax_rate_key1',
        'is_cgst1',
        'is_sgst1',
        'is_igst1',
        'is_disc1']] = DF.apply(lambda x: updateScores(x), axis=1)

    DF.drop(['number_sum'],
            axis = 1,
            inplace=True)

    return DF

@util.timing
def correctLineRows(df):
    df_copy = df.copy(deep = True)
    try:
        df.loc[(df["is_HDR"] == 1),"line_row"] = 0
        pages = df.groupby(["page_num"])
        for page in pages.groups:
            headers = list(df[(df["page_num"] == page) &
                                (df["is_HDR"] == 1)]["token_id"])
            if len(headers) > 0:
                header_start = min(headers)
                df.loc[(df["page_num"] == page) & 
                       (df["token_id"] < header_start) &
                       (df["line_row"] > 0),"line_row"] = 0
        return df
    except:
        print("correctLineRows",traceback.print_exc())
        return df_copy

def trimListAtMatch(l,matchval):
    try:
        return l[(l.index(matchval) -1 if l.index(matchval) > 0 else 0):]
    except:
        return l

def getCorrectColHdrForNone(l,matchval):

    try:

        l = trimListAtMatch(l,matchval)
        l = trimListAtMatch(l[::-1],matchval)[::-1]

        non_nulls = [a for a in l if a != matchval]

        if len(non_nulls) == 3:
            return non_nulls[1]
        elif len(non_nulls) == 2:
            if l[-1] == 'none':
                return non_nulls[1]
            else:
                return non_nulls[0]
        elif len(non_nulls) == 1:
            return non_nulls[0]
        else:
            return None
    except:
        return None

@util.timing
def correctLineRowsFzScore(df):

    df_copy = df.copy(deep = True)

    print("Inside correctLineRowsFzScore")
    try:
        line_cols = ['score','score_1','is_HDR','line_noanchor',
                     'line_ngbr','line_dist','line_valign',
                     'line_amount','line_item','line_row',
                     'amount_in_line','is_item_desc','is_item_code',
                     'is_item_val','is_unit_price','is_uom','is_qty',
                     'is_hsn_key','is_tax_rate_key','is_cgst','is_sgst',
                     'is_igst','is_disc','is_item_code1','is_qty1',
                     'is_unit_price1','is_item_val1','is_uom1',
                     'is_hsn_key1','is_tax_rate_key1','is_cgst1',
                     'is_sgst1','is_igst1','is_disc1','tbl_col_hdr']
        df["tbl_col_hdr_null"] = df["tbl_col_hdr"].isnull() | df["tbl_col_hdr"].isna()
        df["tbl_col_hdr_null"] = df["tbl_col_hdr_null"] | (df["tbl_col_hdr"] == "")
        df.loc[(df["tbl_col_hdr_null"] == True),
                 "tbl_col_hdr"] = "none"
        filt = df[df["line_row"] > 0]
        pages = filt.groupby(["page_num",
                            "line_row",
                            "line_num"])
        for page,line_row,line_num in pages.groups:
            line = filt[(filt["page_num"] == page) &
                        (filt["line_row"] == line_row) &
                        (filt["line_num"] == line_num)]
            col_hdrs = list(line["tbl_col_hdr"])
            matchval = 'none'
            #Jun 16 2022 - change the logic to identify correct column header
            text = getCorrectColHdrForNone(col_hdrs,matchval)
            # print("colhdrs",0,col_hdrs)
            # unq_col_hdrs = list(set(col_hdrs))
            # if len(unq_col_hdrs) < 2:
            #     continue

            # while len(unq_col_hdrs) > 2:
            #     col_hdr = col_hdrs[0]
            #     if col_hdr != "none":
            #         col_hdrs = col_hdrs[1:]
            #     else:
            #         break
            #     #Jun 15 2022 - calculate unique headers without none
            #     unq_col_hdrs = list(set(col_hdrs))
            #     # unq_col_hdrs = list(set([col_hdr for col_hdr in col_hdrs if col_hdr != "none"]))
            #     #Jun 15 2022 - calculate unique headers without none

            # print("colhdrs",1,col_hdrs,unq_col_hdrs)

            # while len(unq_col_hdrs) > 2:
            #     col_hdr = col_hdrs[-1]
            #     if col_hdr != "none":
            #         col_hdrs.pop()
            #     else:
            #         break
            #     #Jun 15 2022 - calculate unique headers without none
            #     unq_col_hdrs = list(set(col_hdrs))
            #     # unq_col_hdrs = list(set([col_hdr for col_hdr in col_hdrs if col_hdr != "none"]))
            #     #Jun 15 2022 - calculate unique headers without none

            # print("colhdrs",2,col_hdrs,unq_col_hdrs)
            # #Jun 15 2022 - calculate unique headers without none
            # # col_hdrs = unq_col_hdrs
            # col_hdrs = list(set(col_hdrs))
            # print("colhdrs",3,col_hdrs,unq_col_hdrs)
            # #Jun 15 2022 - calculate unique headers without none

            # if len(col_hdrs) == 2:
            #     nulls = sum([1 if a == "none" else -1 for a in col_hdrs])
            #     print("colhdrs",3,nulls)
            #     if nulls != 0:
            #         continue
            #     text = ""
            #     text1 = col_hdrs[0]
            #     text2 = col_hdrs[1]
            #     if text1 == "none":
            #         text = text2
            #     elif text2 == "none":
            #         text = text1
                # print("colhdrs",4,text1,text2,text)
            
            #Jun 16 2022 - change the logic to identify correct column header
            if text is not None:
                vals = df[(df["page_num"] == page) &
                          (df["line_num"] == line_num) &
                          (df["tbl_col_hdr"] == text)][line_cols].values.tolist()[0]
                # print(vals,page,line_row,line_num)
                df.loc[(df["page_num"] == page) &
                       (df["line_num"] == line_num) &
                       (df["line_row"] == line_row) &
                       (df["tbl_col_hdr"] == "none")
                       ,line_cols] = vals
            else:
                continue

        return df
    except:
        print("correctLineRowsFzScore",
              traceback.print_exc())
        return df_copy

def fz_match_hdrTxt(txt,keys):
    try:
        if isinstance(keys[0],dict):
            scores = []
            for word_weights in keys:
                score = 0.0
                reduce = False
                for word_weight in word_weights.keys():
                    if word_weight.lower() != "exclude":
                        words = word_weights[word_weight]
                        #Change for using RapidFuzz instead of FuzzyWuzzy
                        part = rp_fz.extractOne(txt,
                                                words,
                                                scorer = rp_fuzz.WRatio)
                        # print("Fuzz",txt,words,part)
                        if part is None:
                            part_score = 0
                        else:
                            part_score = part[1] / 100
                        score += part_score * float(word_weight)
                    elif word_weight.lower() == "exclude":
                        words = word_weights[word_weight]
                        down = rp_fz.extractOne(txt,
                                                words,
                                                scorer = rp_fuzz.WRatio)
                        if down is not None:
                            # print("down",down)
                            if down[1] / 100 >= 0.9:
                                reduce = True
                    elif word_weight.lower() == "exact":
                        words = word_weights[word_weight]
                        
                if reduce:
                    score = score * 0.5
                scores.append(score)
            score = max(scores)
            # print("Fuzzy Score:",txt,keys,score)
            return score
        else:
            match = rp_fz.extractOne(txt,
                                     keys,
                                     scorer = rp_fuzz.WRatio)
            # print("Fuzz",txt,keys,match)
            if match is None:
                score = 0
            else:
                score = match[1] / 100
            return score
    except:
        print("Failed in fz_match_hdrTxt",
              traceback.print_exc())
        return 0.0

@util.timing
def addLineItemFeatures_New(DF):
    import logging
    logging.getLogger().setLevel(logging.ERROR)

    def updateScores(x):
        if x.line_item == 0:
            return pd.Series([0]*22)
        else:
            return pd.Series([x.is_item_desc,
                              x.is_item_code,
                              x.is_item_val,
                              x.is_unit_price,
                              x.is_uom, x.is_qty,
                              x.line_item,
                              x.is_hsn_key,
                              x.is_tax_rate_key,
                              x.is_cgst,
                              x.is_sgst,
                              x.is_igst,
                              x.is_item_code1,
                              x.is_qty1,
                              x.is_unit_price1,
                              x.is_item_val1,
                              x.is_uom1,
                              x.is_hsn_key1,
                              x.is_tax_rate_key1,
                              x.is_cgst1,
                              x.is_sgst1,
                              x.is_igst1,
                              x.is_disc1])

    try:
        
        #Initialize all the line item features
        DF["score"] = 0
        DF["score_1"] = 0
        DF["is_HDR"] = 0
        DF["line_noanchor"] = 0
        DF["line_ngbr"] = 0
        DF["line_dist"] = 0
        DF["line_valign"] = 0
        DF["line_amount"] = 0
        DF["line_item"] = 0
        DF["line_row"] = 0
        DF["amount_in_line"] = 0
        DF[LI_FIELDS] = 0
        DF["tbl_col_hdr"] = ""

        #Header Line Identification
        #Score the header line using fuzzy logic
        # print("Updated line item columns")

        DF.sort_values(["page_num",
                        "line_num",
                        "word_num"],
                       ascending=[True,
                                  True,
                                  True])
        DF_hdr = DF.groupby(['page_num',
                              'tableLineNo']
                            )['line_text'].unique().apply(list).reset_index()
        DF_hdr['score'] = DF_hdr.apply(getHdrScore, axis=1)

        for ind, row in DF_hdr.iterrows():
            DF.loc[(DF["page_num"] == row["page_num"]) &
                    (DF["tableLineNo"] == row["tableLineNo"]),
                    "score"] = row['score']

        DF_hdr_1 = DF.groupby(['page_num',
                                'tableLineNo']
                              )['text'].apply(list).reset_index()
        DF_hdr_1['score'] = DF_hdr_1.apply(getHdrScore_1, axis=1)
        for ind, row in DF_hdr_1.iterrows():
            DF.loc[(DF["page_num"] == row["page_num"]) &
                    (DF["tableLineNo"] == row["tableLineNo"]),
                    "score_1"] = row['score']

        #get max of score by page_num
        max_scores = DF_hdr.groupby(["page_num"]
                                    )["score"].agg("max").reset_index()
        # print(max_scores,max_scores.columns.values)
        all_pages = list(max_scores["page_num"].unique())
        max_scores = max_scores[max_scores["score"] > 200]
        pages = list(max_scores["page_num"].unique())
        # print("Pages:",pages,all_pages)
    
        page_cnt = 0
    
        for max_row_ind,max_row in max_scores.iterrows():

            #Update header line
            hdr_val = max_row["score"]
            min_page = pages[page_cnt]
            max_page = -1
            if len(pages) == 1:
                max_page = all_pages[-1]
            else:
                if page_cnt < len(pages) - 1:
                    max_page = pages[page_cnt + 1]
                else:
                    max_page = all_pages[-1]
            page_cnt += 1
            #Changes made on Apr 06, 2022
            # lines = list(DF_hdr[(DF_hdr["page_num"] >= min_page) &
            #                     (DF_hdr["page_num"] <= max_page)]["tableLineNo"])
            lines = list(DF_hdr[(DF_hdr["page_num"] >= min_page) &
                                (DF_hdr["page_num"] <= max_page)]["tableLineNo"])
            #Changes made on Apr 06, 2022
            lines = sorted(lines)

            hdr_line = min(lines)
            org_hdr_line = -1

            hdr_found = False
            # print("Header found",hdr_line)

            if hdr_val > 200:
                DF_hdr_filt = DF_hdr[(DF_hdr["page_num"] == min_page) & 
                                     (DF_hdr["score"] == hdr_val)]
                DF_hdr_filt_len = DF_hdr_filt.shape[0]
                # print("Header found",hdr_val,DF_hdr_filt_len)

                if DF_hdr_filt_len == 1:

                    hdr_line = DF_hdr_filt["tableLineNo"].values[0]
                    # print("Initial Header Line:", hdr_line, hdr_val)
                    #Check if any amount is there on header line. If it is, then it's not a header
                    # print("Header found",check_hdr(DF,hdr_line),hdr_line)
                    # print("Is header",check_hdr(DF,hdr_line),hdr_line)
                    if check_hdr(DF,hdr_line):
                        DF.loc[(DF["tableLineNo"] == hdr_line),
                               "is_HDR"] = 1
                        org_hdr_line = hdr_line
                        hdr_found = True
                        hdr_line_comp = hdr_line

                    #Check if below wrapped lines are part of the header
                    line_down = max(list(set(list(DF[DF["tableLineNo"] == hdr_line]["line_down"]))))
                    scr = 1
                    rowcount = 0
                    while scr > 0:
                        lines = sorted(lines)
                        next_lines = lines[lines.index(hdr_line)+1:]
                        # print("next Lines:",next_lines)
                        if (len(next_lines) > 0) and (rowcount < len(next_lines)):
                            #Hdr line is updated with next line. Hence always taking from first index
                            #For prev Line, this is different
                            next_line = next_lines[0]
                            # print("Next Line is:", next_line)
                            line_top = min(list(set(list(DF[DF["tableLineNo"] == next_line]["line_top"]))))
                            # print("line_top",line_top,line_down)
                            if (line_top - line_down < .02) or True:
                                line_down = max(list(set(list(DF[DF["tableLineNo"] == next_line]["line_down"]))))
                                scr = DF_hdr[DF_hdr["tableLineNo"] == next_line]["score"].values[0]
                                scr_1 = DF_hdr_1[DF_hdr_1["tableLineNo"] == next_line]["score"].values[0]
                                # print("Next Line is",next_line)
                                if scr >= 200 and scr_1 > 0:
                                    # print("Score of next line",next_line," is higher")
                                    # print("Is header next line",check_hdr(DF,next_line),next_line)
                                    if check_hdr(DF,next_line):
                                        # print("Next Line", next_line, " is now hdr_line")
                                        hdr_line = next_line
                                        DF.loc[(DF["tableLineNo"] == hdr_line),
                                               "is_HDR"] = 1
                                        if not hdr_found:
                                            hdr_found = True
                                            hdr_line_comp = hdr_line
                                else:
                                    scr = 0
                            else:
                                scr = 0
                            rowcount += 1
                            if rowcount > 2:
                                scr = 0
                        else:
                            scr = 0

                    #Check if top wrapped lines are part of the header
                    # print("check if lines top are part of the header")
                    line_top = max(list(set(list(DF[DF["tableLineNo"] == hdr_line]["line_top"]))))
                    scr = 1
                    rowcount = 0
                    while scr > 0:
                        # print("rowcount",rowcount)
                        if org_hdr_line > 0:
                            # print("lines index",lines,org_hdr_line)
                            if lines.index(org_hdr_line) == 0:
                                break
                        else:
                            break
                        lines = sorted(lines)
                        # print("rowcount",rowcount)
                        # print("header line",org_hdr_line)
                        prev_lines = lines[:lines.index(org_hdr_line)][::-1]
                        # print("Previous lines",prev_lines)
                        # print("rowcount",rowcount)
                        if len(prev_lines) > 0  and (rowcount < len(prev_lines)):
                            # print("Previous Lines",len(prev_lines),rowcount)
                            prev_line = prev_lines[rowcount]
                            line_down = min(list(set(list(DF[DF["tableLineNo"] == prev_line]["line_down"]))))
                            # print("rowcount",rowcount)
                            # print("initial previous line",rowcount,prev_line,
                            #       line_top,line_down)
                            if line_top - line_down < .05:
                                line_top = max(list(set(list(DF[DF["tableLineNo"] == prev_line]["line_top"]))))
                                scr = DF_hdr[DF_hdr["tableLineNo"] == prev_line]["score"].values[0]
                                scr_1 = DF_hdr_1[DF_hdr_1["tableLineNo"] == prev_line]["score"].values[0]
                                # print("Is header previous line",check_hdr(DF,prev_line),prev_line,scr)
                                if scr >= 200 and scr_1 > 0:
                                    if check_hdr(DF,prev_line):
                                        DF.loc[(DF["tableLineNo"] == prev_line),
                                               "is_HDR"] = 1
                                        if not hdr_found:
                                            hdr_found = True
                                            hdr_line_comp = hdr_line
                                else:
                                    scr = 0
                            else:
                                scr = 0
                            rowcount += 1
                            if rowcount > 2:
                                scr = 0
                        else:
                            scr = 0
        
            if not hdr_found:
                return DF
        
            line_noanchor = []
            line_valign = []
            line_ngbr = []
            line_dist = []
        
            pot_lines = lines[lines.index(hdr_line) + 1:]
    
            punc = string.punctuation.replace('#', '').replace('/', '')
            punc = punc + '0123456789'

            if len(pot_lines) > 0:

                #Check if lines have anchor words
                # t = time.time()
                found = False
                for line_no in pot_lines:
                    if found:
                        break
                    DF_line = DF[DF['tableLineNo'] == line_no]

                    for i,row in DF_line.iterrows():
                        line_text = str(row["line_text"]).lower().strip()
                        line_text = line_text.translate(str.maketrans('', '', punc))

                        matches = rp_fz.extractOne(line_text,
                                                   footer_keywords,
                                                   scorer = rp_fuzz.ratio,
                                                   score_cutoff = 90)
                        # matches = rp_fz.extractOne(line_text,
                        #                            footer_keywords,
                        #                            scorer = rp_fuzz.token_set_ratio,
                        #                            score_cutoff = 90)
                        # if line_text.lower() == "total amount inr".lower():
                        #     print("matches",
                        #           line_text,
                        #           matches,
                        #           footer_keywords)
                        # print("matches",
                        #       line_text,
                        #       matches,
                        #       footer_keywords)

                        if matches:
                            found = True
                            # print("matched Text",
                            #       line_text,
                            #       matches,
                            #       line_no,
                            #       line_noanchor,
                            #       found)
                            if line_no in line_noanchor:
                                line_noanchor.remove(line_no)
                            break
                        else:
                            if line_no not in line_noanchor:
                                line_noanchor.append(line_no)

                # print("Line Anchor",line_noanchor)
                #Check if lines have same amount of neighbours -
                #THIS WILL WORK WELL FOR LINE TABLE WITH MORE LINE ITEMS
                # t = time.time()
                if len(line_noanchor) >  0:
                    noanchor = max(line_noanchor)
                else:
                    noanchor = pot_lines[:-2:-1][0]

                pot_reduced_lines = pot_lines[:pot_lines.index(noanchor)+1]

                print("Potential reduced lines",
                      pot_reduced_lines)
                ngbrs = []
                for line_no in pot_reduced_lines:
                    DF_line = DF[DF['tableLineNo'] == line_no]
                    neighbour = max(list(DF_line["noNeighbours"]))
                    ngbrs.append(neighbour)

                rowparts = list(range(min(len(pot_reduced_lines),5),0,-1))

                found = False
                rowpart = 0
                times = 0
                for m in rowparts:
                    sum_pattern = [sum(ngbrs[i*m:i*m+m]) for i in range(len(ngbrs)//m)]
                    diffs = [sum_pattern[j] - sum_pattern[j+1]
                             for j in range(len(sum_pattern)-1)]
                    sum_diffs = 0
                    for i in range(len(diffs)):
                        if abs(diffs[i]) <= 1:
                            sum_diffs += diffs[i]
                            if sum_diffs > 2:
                                break
                            else:
                                found = True
                                rowpart = m
                                times = i + 1
                        else:
                            break
                    if found:
                        break

                if rowpart == 0:
                    line_ngbr.append(pot_lines[0])
                else:
                    ntimes = rowpart * times
                    if ntimes <= len(pot_reduced_lines):
                        line_ngbr.extend(pot_reduced_lines[:ntimes])
                    else:
                        line_ngbr.extend(pot_reduced_lines[:])

                #Check if distance between the lines have a pattern -
                #THIS WILL WORK WELL FOR LINE TABLE WITH MORE COLUMNS
                # t = time.time()
                pot_reduced_lines = pot_lines[:pot_lines.index(noanchor)+1]
                rowparts = list(range(min(len(pot_reduced_lines),10),0,-1))
                coords = []
                for line_ind, line_no in enumerate(pot_reduced_lines):
                    DF_line = DF[DF['tableLineNo'] == line_no]
                    line_top = max(list(DF_line["line_top"]))
                    line_down = min(list(DF_line["line_down"]))
                    coords.append((line_down,line_top))
        
                #print("Line Items before line distance",pot_reduced_lines)
                line_dist.append(pot_reduced_lines[0])
                for coord_ind in range(len(coords) - 1):
                    dist = coords[coord_ind + 1][1] - coords[coord_ind][0]
                    if dist <= .1:
                        line_dist.append(pot_reduced_lines[coord_ind + 1])
                    else:
                        break

                #Check if most of the columns are aligned vertically
                # t = time.time()
                pot_reduced_lines = pot_lines[:pot_lines.index(noanchor)+1]
                line_valign.append(pot_reduced_lines[0])
                hdrs = DF[DF["tableLineNo"] == pot_reduced_lines[0]][["line_down",
                                                                      "line_left",
                                                                      "line_right",
                                                                      "line_text"]
                                                                     ].drop_duplicates(
                                                                         keep = 'first'
                                                                         ).apply(dict)
                first_lefts = list(hdrs["line_left"].values())
                first_rights = list(hdrs["line_right"].values())
                line_down = max(list(hdrs["line_down"].values()))
                lines_notfound = 0

                for line_ind in range(1,len(pot_reduced_lines)):
                    line = DF[DF["tableLineNo"] == pot_reduced_lines[line_ind]][
                        ["line_top","line_left",
                         "line_right","line_text"]].drop_duplicates(
                             keep = 'first').apply(dict)
                    lefts = list(line["line_left"].values())
                    rights = list(line["line_right"].values())
                    line_top = min(list(line["line_top"].values()))
                    dist = line_top - line_down
                    mtch = 0
                    if dist < .1:
                        for j in zip(first_lefts,first_rights):
                            for k in zip(lefts,rights):
                                if (k[0] > j[1]) or (k[1] < j[0]):
                                    continue
                                else:
                                    mtch += 1
                    else:
                        mtch = 0
                        break
                    if mtch == 0:
                        lines_notfound += 1
                    else:
                        lines_notfound = 0
                        line_valign.append(pot_reduced_lines[line_ind])
                        hdrs = DF[DF["tableLineNo"] == pot_reduced_lines[line_ind]][["line_down","line_left",
                                 "line_right","line_text"]].drop_duplicates(keep = 'first').apply(dict)
                        first_lefts = list(hdrs["line_left"].values())
                        first_rights = list(hdrs["line_right"].values())
                        line_down = max(list(hdrs["line_down"].values()))
                    if lines_notfound > 3:
                        line_down = 0
                        break
    
                #Update row breaks
                # t = time.time()
                max_lefts = []
                max_rights = []
                max_alls = []
                line_dist_align = list(set(line_dist).union(set(line_valign)))
                line_dist_align = sorted(line_dist_align)
                for line_index,line_no in enumerate(line_dist_align):
                    if line_index == 0:
                        neighbours = list(DF[DF["tableLineNo"] == line_no])
                        if len(neighbours) == 1:
                            continue
                    lefts = list(DF[DF["tableLineNo"] == line_no]["line_left"])
                    max_left = max(lefts)
                    max_lefts.append(max_left)
                    max_ind = lefts.index(max_left)
    
                    rights = list(DF[DF["tableLineNo"] == line_no]["line_right"])
                    max_right = rights[max_ind]
                    max_rights.append(max_right)

                    max_alls.append((max_left,
                                     max_right,
                                     line_no))

                max_left = max(max_lefts)
                max_ind = max_lefts.index(max_left)
                max_all = max_alls[max_ind]
                max_right = max_all[1]
                row_number = 1
                row_first = False
                # print("Max",max_alls)

                for all_ind,all_ in enumerate(max_alls):
                    left = all_[0]
                    right = all_[1]
                    line_no = all_[2]
                    # print("Line row",row_number,line_no,right,
                    #       max_left,left,max_right,
                    #       "page_num",min_page)
                    DF.loc[(DF["tableLineNo"] == line_no),
                            "line_row"] = row_number
                    if (right < max_left) or (left > max_right):
                        continue
                    else:
                        if all_ind == 0:
                            row_number = 0
                            row_first = True
                        row_number += 1
                        if row_first:
                            DF.loc[(DF["tableLineNo"] == line_no),
                                   "line_row"] = row_number
                        # print("Line row",row_number,line_no,right,
                        #       max_left,left,max_right,
                        #       "page_num",min_page)
                # DF.to_csv(r"d:\ftrs_1.csv",index = True)
                #Check if lines have amount and how many amount fields are there
                # t = time.time()
                pot_reduced_lines = pot_lines[:pot_lines.index(noanchor)+1]
                line_nums = []
    
                for line_no in pot_reduced_lines:
                    texts = list(DF[DF["tableLineNo"] == line_no]["line_text"])
                    count = sum([util.isAmount(text) for text in texts])
                    if count == 0:
                        count = sum([util.isNumber(text) for text in texts])
    
                    DF.loc[(DF["tableLineNo"] == line_no)
                           ,["line_amount","amount_in_line"]] = [int(count > 0),count]
                    if count > 0:
                        line_nums.append(line_no)

                #Check the line item type
                # t = time.time()
                DF = DF.sort_values(by = ["page_num",
                                          "line_num"],
                                    ascending = [True,True])

                hdrs = DF[DF["tableLineNo"] == hdr_line_comp][["line_text",
                          "line_left","line_top","line_down",
                          "line_right"]].drop_duplicates(keep = 'first').apply(dict)

                # print("Get Text and Dimensions of other lines")
                pages_hdr = list(DF[DF["is_HDR"] == 1]["page_num"])
                page_num = max([int(page_hdr) for page_hdr in pages_hdr])
                # print("Header Page",page_num,pages)
                oth_hdrs = DF[(DF["is_HDR"] == 1) &
                              (DF["page_num"] == page_num)][
                    ["line_text",
                      "line_top",
                      "line_down",
                      "line_left",
                      "line_right",
                      "line_num",
                      "lineLeft",
                      "line_left_y1",
                      "lineRight",
                      "line_right_y1"]].drop_duplicates(keep = 'first').apply(dict)
                # print("Get Text and Dimensions of other lines",oth_hdrs)

                # print("List Text and Dimensions of other lines")
                oth_hdr_lefts = list(oth_hdrs["line_left"].values())
                oth_hdr_rights = list(oth_hdrs["line_right"].values())
                oth_hdr_tops = list(oth_hdrs["line_top"].values())
                oth_hdr_downs = list(oth_hdrs["line_down"].values())
                oth_hdr_texts = list(oth_hdrs["line_text"].values())
                oth_line_nums = list(oth_hdrs["line_num"].values())
                oth_lineLefts = list(oth_hdrs["lineLeft"].values())
                oth_line_left_y1s = list(oth_hdrs["line_left_y1"].values())
                oth_lineRights = list(oth_hdrs["lineRight"].values())
                oth_line_right_y1s = list(oth_hdrs["line_right_y1"].values())
                others = list(zip(oth_hdr_lefts,oth_hdr_tops,oth_hdr_rights,
                                  oth_hdr_downs,oth_hdr_texts,oth_line_nums,
                                  oth_lineLefts,oth_line_left_y1s,
                                  oth_lineRights,oth_line_right_y1s))

                #Append header text with prev and next lines if header is wrapped around
                othrindx = []
                hdr_texts = []
                # print("Others:",others)
                for ind,obj in enumerate(others):
                    hdr_texts.append("")
                    for other_ind,other in enumerate(others):
                        #Use column dividing line to calculate width of the text within - April 07, 2022
                        # print("Text:",obj[4],other[4])
                        left_obj = obj[0]
                        right_obj = obj[2]
                        #Jun 15, 2022 - keep lbound and ubound as 0 and 100
                        # left_x_solid_line = -1
                        # right_x_solid_line = -1
                        left_x_solid_line = 0
                        right_x_solid_line = 100
                        #Jun 15, 2022 - keep lbound and ubound as 0 and 100
                        if obj[6] == 1:
                            left_x_solid_line = obj[7]
                            if isinstance(left_x_solid_line,str):
                                left_x_solid_line = ast.literal_eval(left_x_solid_line)[0]
                            else:
                                left_x_solid_line = left_x_solid_line[0]
                        if obj[8] == 1:
                            right_x_solid_line = obj[9]
                            if isinstance(right_x_solid_line,str):
                                right_x_solid_line = ast.literal_eval(right_x_solid_line)[0]
                            else:
                                right_x_solid_line = right_x_solid_line[0]

                        # print("Assign header dimension",
                        #       obj[4],obj[6],obj[0],left_x_solid_line)
                        # print("Assign header dimension",
                        #       obj[4],obj[8],obj[2],right_x_solid_line)
                        if -0.005 < (obj[0] - left_x_solid_line):
                            #Jun 15, 2022 - keep lbound and ubound as 0 and 100
                            # if left_x_solid_line != -1:
                            if left_x_solid_line != 0:
                            #Jun 15, 2022 - keep lbound and ubound as 0 and 100
                                left_obj = left_x_solid_line + 0.00001
                        if -0.005 < (right_x_solid_line - obj[2]):
                            #Jun 15, 2022 - keep lbound and ubound as 0 and 100
                            # if right_x_solid_line != -1:
                            if right_x_solid_line != 100:
                            #Jun 15, 2022 - keep lbound and ubound as 0 and 100
                                right_obj = right_x_solid_line - 0.00001
                        
                        #Jun 01, 2022 - Check if the difference between
                        #left and right line <= 50% to assign extended line
                        # if abs(obj[0] - left_obj) >= 0.05:
                        #     left_obj = obj[0]
                        # if abs(obj[2] - right_obj) >= 0.05:
                        #     right_obj = obj[2]
                        # print("Dimension diff",
                        #       obj[0],left_obj,right_obj,obj[2])
                        #Jun 15, 2022 - Check if the diff between two lines are 50% or less
                        # if abs(obj[0] - left_obj) >= 0.5:
                        #     left_obj = obj[0]
                        # if abs(obj[2] - right_obj) >= 0.5:
                        #     right_obj = obj[2]

                        if right_x_solid_line - left_x_solid_line >= 0.5:
                            left_obj = obj[0]
                        # if abs(obj[2] - right_obj) >= 0.5:
                            right_obj = obj[2]
                        #Jun 15, 2022 - Check if the diff between two lines are 50% or less
                        #Jun 01, 2022 - Check if the difference is <= 50% to assign extended line

                        #May 02, 2022 - consider lines between column header to determin column width
                        #only if the lines are closer to the text
                        # left_obj = obj[0]
                        # right_obj = obj[2]
                        #May 02, 2022 - consider lines between column header to determin column width
                        #only if the lines are closer to the text

                        left_other = other[0]
                        right_other = other[2]
                        #Jun 15, 2022 - keep lbound and ubound as 0 and 100
                        # left_x_solid_line = -1
                        # right_x_solid_line = -1
                        left_x_solid_line = 0
                        right_x_solid_line = 100
                        #Jun 15, 2022 - keep lbound and ubound as 0 and 100
                        if other[6] == 1:
                            left_x_solid_line = other[7]
                            if isinstance(left_x_solid_line,str):
                                left_x_solid_line = ast.literal_eval(left_x_solid_line)[0]
                            else:
                                left_x_solid_line = left_x_solid_line[0]
                        if other[8] == 1:
                            right_x_solid_line = other[9]
                            if isinstance(right_x_solid_line,str):
                                right_x_solid_line = ast.literal_eval(right_x_solid_line)[0]
                            else:
                                right_x_solid_line = right_x_solid_line[0]

                        if 0.0 < (other[0] - left_x_solid_line):
                        #Jun 15, 2022 - keep lbound and ubound as 0 and 100
                            # if left_x_solid_line != -1:
                            if left_x_solid_line != 0:
                        #Jun 15, 2022 - keep lbound and ubound as 0 and 100
                                left_other = left_x_solid_line + 0.00001
                        if 0.0 < (right_x_solid_line - other[2]):
                        #Jun 15, 2022 - keep lbound and ubound as 0 and 100
                            # if right_x_solid_line != -1:
                            if right_x_solid_line != 100:
                        #Jun 15, 2022 - keep lbound and ubound as 0 and 100
                                right_other = right_x_solid_line - 0.00001

                        #Jun 01, 2022 - Check if the difference is <= 50% to assign extended line
                        # if abs(other[0] - left_other) >= 0.05:
                        #     left_other = other[0]
                        # if abs(other[2] - right_other) >= 0.05:
                        #     right_other = other[2]

                        # print("Dimension diff other",
                        #       other[0],left_other,right_other,other[2])
                        #Jun 15, 2022 - Check if the diff between two lines are 50% or less
                        # if abs(other[0] - left_other) >= 0.5:
                        #     left_other = other[0]
                        # if abs(other[2] - right_other) >= 0.5:
                        #     right_other = other[2]

                        if right_x_solid_line - left_x_solid_line >= 0.5:
                            left_other = other[0]
                            right_other = other[2]
                        #Jun 15, 2022 - Check if the diff between two lines are 50% or less
                        #Jun 01, 2022 - Check if the difference is <= 50% to assign extended line

                        #May 02, 2022 - consider lines between column header to determin column width
                        #only if the lines are closer to the text
                        # left_other = other[0]
                        # right_other = other[2]
                        #May 02, 2022 - consider lines between column header to determin column width
                        #only if the lines are closer to the text
                        #Use column dividing line to calculate width of the text within - April 07, 2022
                        if (obj != other) and (obj[5] < other[5]) and (
                                other_ind not in othrindx) and (ind not in othrindx):
                            #Use column dividing line to calculate width of the text within - April 07, 2022
                            if (obj[1] > other[1]) and not ((left_obj > right_other) or (
                                    right_obj < left_other)):
                                hdr_texts[ind] = other[4] + " " + hdr_texts[ind]
                                # print("header text 1",hdr_texts[ind],ind)
                        if (obj != other) and (obj[5] < other[5]) and (
                                other_ind not in othrindx) and (ind not in othrindx):
                            if (obj[1] > other[1]) and not ((left_obj > right_other) or (
                                    left_obj < left_other)):
                                hdr_texts[ind] = other[4] + " " + hdr_texts[ind]
                                # print("header text 2",hdr_texts[ind],ind)
                                #Use column dividing line to calculate width of the text within - April 07, 2022
                                oth_hdr_lefts[ind] = min(left_obj,left_other)
                                oth_hdr_rights[ind] = max(right_obj,right_other)
                                #Use column dividing line to calculate width of the text within - April 07, 2022
                                othrindx.append(other_ind)
                                #Use column dividing line to calculate width of the text within - April 07, 2022
                                othrindx.append(other_ind)
                                #Use column dividing line to calculate width of the text within - April 07, 2022
                            elif (obj[1] < other[1]) and not ((left_obj > right_other) or (
                                    right_obj < left_other)):
                                hdr_texts[ind] = hdr_texts[ind] + " " + other[4]
                                # print("header text 3",hdr_texts[ind],ind,
                                #       left_obj,right_obj,left_other,right_other)
                                #Use column dividing line to calculate width of the text within - April 07, 2022
                                oth_hdr_lefts[ind] = min(left_obj,left_other)
                                oth_hdr_rights[ind] = max(right_obj,right_other)
                                #Use column dividing line to calculate width of the text within - April 07, 2022
                                othrindx.append(other_ind)
                        elif obj == other:
                            hdr_texts[ind] = other[4]
                            # print("header text 4",hdr_texts[ind],ind)
                            #Use column dividing line to calculate width of the text within - April 07, 2022
                            oth_hdr_lefts[ind] = left_obj
                            oth_hdr_rights[ind] = right_obj
                            #Use column dividing line to calculate width of the text within - April 07, 2022
                            if (ind in othrindx):
                                hdr_texts[ind] = " "


                line_dist_valign = list(set(line_dist).union(set(line_valign)))
                line_dist_valign = sorted(line_dist_valign)
                for line_no in line_dist_valign:
                    #Working code before 15th June 2021
                    line = DF[DF["tableLineNo"] == line_no][
                        ["text","left","right","top"]].drop_duplicates(
                            keep = 'first').apply(dict)
                    lefts = list(line["left"].values())
                    rights = list(line["right"].values())
                    line_top = min(list(line["top"].values()))
                    texts = list(line["text"].values())
                    for ind,coord in enumerate(zip(lefts,rights,texts)):
                        is_item_desc_fz = 0
                        is_item_code_fz = 0
                        is_qty_fz = 0
                        is_unit_price_fz = 0
                        is_item_val_fz = 0
                        is_uom_fz = 0
                        is_hsn_key_fz = 0
                        is_tax_rate_key_fz = 0
                        is_cgst_fz = 0
                        is_sgst_fz = 0
                        is_igst_fz = 0
                        is_disc_fz = 0
    
                        is_item_code1_fz = 0
                        is_qty1_fz = 0
                        is_unit_price1_fz = 0
                        is_item_val1_fz = 0
                        is_uom1_fz = 0
                        is_hsn_key1_fz = 0
                        is_tax_rate_key1_fz = 0
                        is_cgst1_fz = 0
                        is_sgst1_fz = 0
                        is_igst1_fz = 0
                        is_disc1_fz = 0
                        align_cnt = 0

                        maxBoundingArea = 0.0
    
                        for hdr_ind,hdr_coord in enumerate(zip(oth_hdr_lefts,
                                                               oth_hdr_rights)):
                            # print("Find Matching Header Column for:",coord[2])
                            # print("Header text",hdr_texts[hdr_ind])
                            # print("Match:",not((hdr_coord[0] > coord[1]) or
                            #                     (hdr_coord[1] < coord[0])))
                            # print("Coords:",hdr_coord[0],hdr_coord[1],
                            #       coord[0],coord[1])
                            if (hdr_coord[0] > coord[1]) or (hdr_coord[1] < coord[0]):
                                continue
                            else:
                                minLeft = min(hdr_coord[0],coord[0])
                                minRight = min(hdr_coord[1],coord[1])
                                maxLeft = max(hdr_coord[0],coord[0])
                                maxRight = max(hdr_coord[1],coord[1])
                                boundingArea = ((minRight - maxLeft)/(maxRight - minLeft))
                                if boundingArea < maxBoundingArea:
                                    continue
                                maxBoundingArea = boundingArea
                                hdr_text = hdr_texts[hdr_ind]
                                # print("Matching header text:", hdr_text)
                                if len(hdr_text.strip()) <= 2:
                                    continue
                                hdr_text = str(hdr_text).lower().strip()

                                #hdr_text can be added as the column header for this field
                                #To be added on 11-Feb-2022

                                align_cnt += 1
    
                                is_item_desc_fz += fz_match_hdrTxt(hdr_text,
                                                                   desc_key)
                                is_item_code_fz += fz_match_hdrTxt(hdr_text,
                                                                   code_key)
                                is_item_code1_fz += fz_match_hdrTxt(hdr_text,
                                                                    code_key1)
                                is_qty_fz += fz_match_hdrTxt(hdr_text,
                                                             qty_key)
                                is_qty1_fz += fz_match_hdrTxt(hdr_text,
                                                             qty_key1)
                                is_unit_price_fz += fz_match_hdrTxt(hdr_text,
                                                                    price_key)
                                is_unit_price1_fz += fz_match_hdrTxt(hdr_text,
                                                                    price_key1)
                                is_item_val_fz += fz_match_hdrTxt(hdr_text,
                                                                  val_key)
                                is_item_val1_fz += fz_match_hdrTxt(hdr_text,
                                                                  val_key1)
                                is_uom_fz += fz_match_hdrTxt(hdr_text,
                                                             uom_key)
                                is_uom1_fz += fz_match_hdrTxt(hdr_text,
                                                             uom_key1)
                                is_hsn_key_fz += fz_match_hdrTxt(hdr_text,
                                                                 hsn_key)
                                is_hsn_key1_fz += fz_match_hdrTxt(hdr_text,
                                                                 hsn_key1)
                                is_tax_rate_key_fz += fz_match_hdrTxt(hdr_text,
                                                                      tax_rate_key)
                                is_tax_rate_key1_fz += fz_match_hdrTxt(hdr_text,
                                                                      tax_rate_key1)
                                is_cgst_fz += fz_match_hdrTxt(hdr_text,
                                                              cgst)
                                is_cgst1_fz += fz_match_hdrTxt(hdr_text,
                                                              cgst1)
                                is_sgst_fz += fz_match_hdrTxt(hdr_text,
                                                              sgst)
                                is_sgst1_fz += fz_match_hdrTxt(hdr_text,
                                                              sgst1)
                                is_igst_fz += fz_match_hdrTxt(hdr_text,
                                                              igst)
                                is_igst1_fz += fz_match_hdrTxt(hdr_text,
                                                              igst1)
                                is_disc_fz += fz_match_hdrTxt(hdr_text,
                                                              disc_key)
                                is_disc1_fz += fz_match_hdrTxt(hdr_text,
                                                              disc_key1)
    
                                fz_scores = [is_item_desc_fz/align_cnt,
                                             is_item_code_fz/align_cnt,
                                             is_item_val_fz/align_cnt,
                                             is_unit_price_fz/align_cnt,
                                             is_uom_fz/align_cnt,
                                             is_qty_fz/align_cnt,
                                             is_hsn_key_fz/align_cnt,
                                             is_tax_rate_key_fz/align_cnt,
                                             is_cgst_fz/align_cnt,
                                             is_sgst_fz/align_cnt,
                                             is_igst_fz/align_cnt,
                                             is_disc_fz/align_cnt]
                                fz_scores1 = [is_item_code1_fz/align_cnt,
                                              is_qty1_fz/align_cnt,
                                              is_unit_price1_fz/align_cnt,
                                              is_item_val1_fz/align_cnt,
                                              is_uom1_fz/align_cnt,
                                              is_hsn_key1_fz/align_cnt,
                                              is_tax_rate_key1_fz/align_cnt,
                                              is_cgst1_fz/align_cnt,
                                              is_sgst1_fz/align_cnt,
                                              is_igst1_fz/align_cnt,
                                              is_disc1_fz/align_cnt]
    
                                # print("Item Desc:",is_item_desc_fz/align_cnt)
                                # print("Item Code:",is_item_code_fz/align_cnt)
                                # print("Item Val:",is_item_val_fz/align_cnt)
                                # print("Item Rate:",is_unit_price_fz/align_cnt)
                                # print("Item UOM:",is_uom_fz/align_cnt)
                                # print("Item Qty:",is_qty_fz/align_cnt)
                                # print("Item HSN:",is_hsn_key_fz/align_cnt)
                                # print("Tax Rate:",is_tax_rate_key_fz/align_cnt)
                                # print("CGST:",is_cgst_fz/align_cnt)
                                # print("SGST:",is_sgst_fz/align_cnt)
                                # print("IGST:",is_igst_fz/align_cnt)
    
                                # print("Fuzzy Score adjustment started",fz_scores)
                                gt95 = sum([int(scr > .95) for scr in fz_scores])
                                if gt95 == 1:
                                    fz_scores = [min(i,.5) if i <=.95 else i for i in fz_scores]
                                # print("Fuzzy Scores:", fz_scores)
    
                                DF.loc[(DF["tableLineNo"] == line_no) &
                                       (DF["left"] == coord[0]) &
                                       (DF["right"] == coord[1])
                                       ,LI_FIELDS] = fz_scores + fz_scores1
    
                                #Update what is the column header for this token
                                DF.loc[(DF["tableLineNo"] == line_no) &
                                       (DF["left"] == coord[0]) &
                                       (DF["right"] == coord[1])
                                       ,["tbl_col_hdr"]] = hdr_text

                union = list(set(line_noanchor).union(set(line_ngbr)).union(
                    set(line_dist)).union(set(line_valign)))

                # t = time.time()
                if hdr_found:
                    for line_no in union:
                        score = 0
                        cnt = 0
                        anchor = 0
                        ngbr = 0
                        valign = 0
                        dist = 0
                        anchor = int(line_no in line_noanchor)
                        score += anchor
                        cnt += 1

                        ngbr = int(line_no in line_ngbr)
                        score += ngbr
                        cnt += 1

                        dist = int(line_no in line_dist)
                        score += dist
                        cnt += 1

                        valign = int(line_no in line_valign)
                        score += valign
                        cnt += 1

                        DF.loc[(DF["tableLineNo"] == line_no),
                               ["line_noanchor",
                                "line_ngbr",
                                "line_valign",
                                "line_dist",
                                "line_item"]] = [anchor, ngbr,
                                                 valign, dist,
                                                 score / cnt]


            DF['number_sum'] = DF['is_number'] + DF['is_amount']

            DF[LI_FIELDS] = DF.apply(lambda x: updateScores(x), axis=1)

            DF.drop(['number_sum'],
                    axis = 1,
                    inplace=True)

        return DF
    except:
        print("addLineItemFeatures_New",
              traceback.print_exc())
        return None

@util.timing
def addLineItemNeighbours_new(DF):

    DF["ngbr_item_desc"] = 0
    DF["ngbr_item_code"] = 0
    DF["ngbr_item_val"] = 0
    DF["ngbr_unit_price"] = 0
    DF["ngbr_uom"] = 0
    DF["ngbr_qty"] = 0
    DF["ngbr_hsn"] = 0
    DF["ngbr_tax_rate"] = 0
    DF["ngbr_cgst"] = 0
    DF["ngbr_sgst"] = 0
    DF["ngbr_igst"] = 0
    DF["ngbr_distance"] = 0
    return DF


@util.timing
def addLineItemNeighbours(DF):

    DF["ngbr_item_desc"] = 0
    DF["ngbr_item_code"] = 0
    DF["ngbr_item_val"] = 0
    DF["ngbr_unit_price"] = 0
    DF["ngbr_uom"] = 0
    DF["ngbr_qty"] = 0
    DF["ngbr_hsn"] = 0
    DF["ngbr_tax_rate"] = 0
    DF["ngbr_cgst"] = 0
    DF["ngbr_sgst"] = 0
    DF["ngbr_igst"] = 0
    DF["ngbr_distance"] = 0

    line_items = DF[DF["line_item"] > 0]

    #Score the header line using fuzzy logic
    # print("Updated line item columns")
    DF_hdr = DF.groupby(['tableLineNo'])['line_text'].unique().apply(list).reset_index()
    DF_hdr['score'] = DF_hdr.apply(getHdrScore, axis=1)

    #Update header line
    lines = list(DF_hdr["tableLineNo"])
    hdr_val = DF_hdr.nlargest(1,["score"])["score"].values[0]
    hdr_line = min(lines)
    hdr_found = False
    if hdr_val > 200:
        DF_hdr_filt = DF_hdr[DF_hdr["score"] == hdr_val]
        DF_hdr_filt_len = DF_hdr_filt.shape[0]
        if DF_hdr_filt_len == 1:
            hdr_line = DF_hdr_filt["tableLineNo"].values[0]
            if check_hdr(DF,hdr_line):
                hdr_found = True

    if not hdr_found:
        return DF

    if (line_items.shape[0] > 0) and (hdr_found):

        DF_hdr = DF[DF["tableLineNo"] == hdr_line]

        for line_ind, line_item in line_items.iterrows():
            left = line_item["left"]
            right = line_item["right"]
            DF_Not_overlap_token = []

            DF_hdr_lefts_token = list(DF_hdr[DF_hdr["right"] < left]['token_id'])
            DF_hdr_rights_token = list(DF_hdr[DF_hdr["left"] > right]['token_id'])

            DF_hdr_lefts_token.extend(DF_hdr_rights_token)
            DF_Not_overlap_token.extend(DF_hdr_lefts_token)

            hdr_collinear = DF_hdr.loc[~ (DF_hdr["token_id"].isin(DF_Not_overlap_token))]
            if hdr_collinear.shape[0] != 0:
                hdr_text = hdr_collinear['line_text'].values[0]
                hdr_distance = abs(DF_hdr['left'].values[0] - left)
            else:
                DF_hdr_lefts = DF_hdr[DF_hdr["right"] < left]
                if DF_hdr_lefts.shape[0] > 0:
                    lft_ngbr_pt = max(list(DF_hdr_lefts["left"]))
                    lft_ngr = DF_hdr_lefts[DF_hdr_lefts["left"] == lft_ngbr_pt].drop_duplicates(["left"],
                                          keep = "first")
                    hdr_text_left = lft_ngr["line_text"]
                    hdr_left = lft_ngr['line_left'].values[0]
                    hdr_right = lft_ngr['line_right'].values[0]
                    hdr_left_pt = hdr_right - left
                else:
                    hdr_left_pt = 100

                DF_hdr_rights = DF_hdr[DF_hdr["left"] > right]
                if DF_hdr_rights.shape[0] > 0:
                    rgt_ngbr_pt = min(list(DF_hdr_rights["left"]))
                    rgt_ngr = DF_hdr_rights[DF_hdr_rights["left"] == rgt_ngbr_pt].drop_duplicates(["left"],
                                          keep = "first")

                    hdr_text_right = rgt_ngr["line_text"]
                    hdr_left = rgt_ngr['line_left'].values[0]
                    hdr_right = rgt_ngr['line_right'].values[0]
                    hdr_right_pt = hdr_left - right
                else:
                  hdr_right_pt = 100

                # Find Close based on center
                diff_right = abs(hdr_right_pt)
                diff_left = abs(hdr_left_pt)
                #print("L:",diff_left,"R:",diff_right)
                close = "Right" if diff_right < diff_left else "Left"
                if close == 'Right':
                    hdr_distance = diff_right
                    hdr_text = hdr_text_right
                else:
                    hdr_distance = diff_left
                    hdr_text = hdr_text_left

            rgt_ngr_text = hdr_text


            is_item_desc_fz = 0
            is_item_code_fz = 0
            is_unit_price_fz = 0
            is_qty_fz = 0
            is_item_val_fz = 0
            is_uom_fz = 0

            is_hsn_key_fz = 0
            is_tax_rate_key_fz = 0
            is_cgst_fz = 0
            is_sgst_fz = 0
            is_igst_fz = 0

            rgt_ngr_text_proc = str(rgt_ngr_text).lower().strip()
            # is_item_desc_fz += fz.extractBests(str(rgt_ngr_text).lower().strip(),
            #                                    desc_key,scorer = fuzz.WRatio,
            #                                    limit = 1)[0][1] / 100

            is_item_desc_mtch = rp_fz.extractOne(rgt_ngr_text_proc,
                                                 desc_key,
                                                 scorer = rp_fuzz.WRatio)
            if is_item_desc_mtch is None:
                is_item_desc_fz = 0
            else:
                is_item_desc_fz = is_item_desc_mtch[1] / 100

            # is_item_code_fz += fz.extractBests(str(rgt_ngr_text).lower().strip(),
            #                                    code_key,scorer = fuzz.WRatio,
            #                                    limit = 1)[0][1] / 100

            is_item_code_mtch = rp_fz.extractOne(rgt_ngr_text_proc,
                                                 code_key,
                                                 scorer = rp_fuzz.WRatio)
            if is_item_code_mtch is None:
                is_item_code_fz = 0
            else:
                is_item_code_fz = is_item_code_mtch[1] / 100

            # is_item_val_fz += fz.extractBests(str(rgt_ngr_text).lower().strip(),
            #                                   val_key,scorer = fuzz.WRatio,
            #                                   limit = 1)[0][1] / 100
            is_item_val_mtch = rp_fz.extractOne(rgt_ngr_text_proc,
                                                val_key,
                                                scorer = rp_fuzz.WRatio)
            if is_item_val_mtch is None:
                is_item_val_fz = 0
            else:
                is_item_val_fz = is_item_val_mtch[1] / 100


            # is_unit_price_fz += fz.extractBests(str(rgt_ngr_text).lower().strip(),
            #                                     price_key,scorer = fuzz.WRatio,
            #                                     limit = 1)[0][1] / 100
            is_unit_price_mtch = rp_fz.extractOne(rgt_ngr_text_proc,
                                             price_key,
                                             scorer = rp_fuzz.WRatio)
            if is_unit_price_mtch is None:
                is_unit_price_fz = 0
            else:
                is_unit_price_fz = is_unit_price_mtch[1] / 100


            # is_uom_fz += fz.extractBests(str(rgt_ngr_text).lower().strip(),
            #                              uom_key,scorer = fuzz.WRatio,
            #                              limit = 1)[0][1] / 100
            is_uom_mtch = rp_fz.extractOne(rgt_ngr_text_proc,
                                           uom_key,
                                           scorer = rp_fuzz.WRatio)
            if is_uom_mtch is None:
                is_uom_fz = 0
            else:
                is_uom_fz = is_uom_mtch[1] / 100

            # is_qty_fz += fz.extractBests(str(rgt_ngr_text).lower().strip(),
            #                              qty_key,scorer = fuzz.WRatio,
            #                              limit = 1)[0][1] / 100
            is_qty_mtch = rp_fz.extractOne(rgt_ngr_text_proc,
                                           qty_key,
                                           scorer = rp_fuzz.WRatio)
            if is_qty_mtch is None:
                is_qty_fz = 0
            else:
                is_qty_fz = is_qty_mtch[1] / 100


            # is_hsn_key_fz += fz.extractBests(str(rgt_ngr_text).lower().strip(),
            #                             hsn_key,scorer = fuzz.WRatio,
            #                             limit = 1)[0][1] / 100
            is_hsn_key_mtch = rp_fz.extractOne(rgt_ngr_text_proc,
                                               hsn_key,
                                               scorer = rp_fuzz.WRatio)
            if is_hsn_key_mtch is None:
                is_hsn_key_fz = 0
            else:
                is_hsn_key_fz = is_hsn_key_mtch[1] / 100


            # is_tax_rate_key_fz += fz.extractBests(str(rgt_ngr_text).lower().strip(),
            #                             tax_rate_key,scorer = fuzz.WRatio,
            #                             limit = 1)[0][1] / 100
            is_tax_rate_key_mtch = rp_fz.extractOne(rgt_ngr_text_proc,
                                                    tax_rate_key,
                                                    scorer = rp_fuzz.WRatio)
            if is_tax_rate_key_mtch is None:
                is_tax_rate_key_fz = 0
            else:
                is_tax_rate_key_fz = is_tax_rate_key_mtch[1] / 100


            # is_cgst_fz += fz.extractBests(str(rgt_ngr_text).lower().strip(),
            #                             cgst,scorer = fuzz.WRatio,
            #                             limit = 1)[0][1] / 100
            is_cgst_mtch = rp_fz.extractOne(rgt_ngr_text_proc,
                                            cgst,
                                            scorer = rp_fuzz.WRatio)
            if is_cgst_mtch is None:
                is_cgst_fz = 0
            else:
                is_cgst_fz = is_cgst_mtch[1] / 100


            # is_sgst_fz += fz.extractBests(str(rgt_ngr_text).lower().strip(),
            #                             sgst,scorer = fuzz.WRatio,
            #                             limit = 1)[0][1] / 100
            is_sgst_mtch = rp_fz.extractOne(rgt_ngr_text_proc,
                                            sgst,
                                            scorer = rp_fuzz.WRatio)
            if is_sgst_mtch is None:
                is_sgst_fz = 0
            else:
                is_sgst_fz = is_sgst_mtch[1] / 100


            # is_igst_fz += fz.extractBests(str(rgt_ngr_text).lower().strip(),
            #                             igst,scorer = fuzz.WRatio,
            #                             limit = 1)[0][1] / 100
            is_igst_mtch = rp_fz.extractOne(rgt_ngr_text_proc,
                                            igst,
                                            scorer = rp_fuzz.WRatio)
            if is_igst_mtch is None:
                is_igst_fz = 0
            else:
                is_igst_fz = is_igst_mtch[1] / 100


            fz_scrs = [is_item_desc_fz,is_item_code_fz,is_item_val_fz,
                          is_unit_price_fz,is_uom_fz,is_qty_fz,
                          is_hsn_key_fz,is_tax_rate_key_fz,is_cgst_fz,
                          is_sgst_fz,is_igst_fz]
            max_scr = max(fz_scrs)
            bin_scrs = [int(max_scr == scr) for scr in fz_scrs]

            DF.loc[(DF["token_id"] == line_item["token_id"]),
                   ["ngbr_item_desc","ngbr_item_code","ngbr_item_val",
                    "ngbr_unit_price","ngbr_uom","ngbr_qty",
                    "ngbr_hsn","ngbr_tax_rate","ngbr_cgst",
                    "ngbr_sgst","ngbr_igst"]] = bin_scrs

            DF.loc[(DF["token_id"] == line_item["token_id"]),
                   "ngbr_distance"] = abs(hdr_distance)

    return DF



# In[5]: Declare Functions that extract features from each row
@util.timing
def findCloseWordsNew(df):

    df_copy = df.copy(deep = True)

    try:
        noNeighbours = 7
        cols = ["left","right",
                "top","bottom","text",
                "page_num","line_num"]
        xCols = [x + "_x" for x in cols]

        xCols_addl = xCols.copy()
        xCols_addl.extend(list(range(noNeighbours)))

        df1 = df[cols]
        df2 = df1.copy(deep = True)

        df_merge = df1.merge(df2, how = "cross")
        df_x = df_merge[xCols]
        df_x = df_x.drop_duplicates(keep = "first")

        #Find Above Words
        words_col_ab = ["W" + str(i) + "Ab" for i in range(1,
                                                           noNeighbours + 1)]
        dists_col_ab = ["d" + str(i) + "Ab" for i in range(1,
                                                           noNeighbours + 1)]
        df_above = df_merge[(df_merge["top_x"] > df_merge["bottom_y"]) &
                      (df_merge["top_x"] <= df_merge["bottom_y"] + verticalThresh) &
                      (df_merge["right_y"] >= df_merge["left_x"]) &
                      (df_merge["left_y"] <= df_merge["right_x"]) &
                      (df_merge["page_num_x"] == df_merge["page_num_y"])]
        df_above["dist"] = df_above["top_y"] - df_above["top_x"]
        df_above = df_above.sort_values(["top_x","left_x","bottom_y","left_y"],
                                        ascending = [True,True,False,True])
        df_above["grp1"] = df_above.groupby(by = xCols).cumcount()
        df_above["grp2"] = df_above["grp1"] + 100
        df_above = df_above[df_above["grp1"] < noNeighbours]

        df_ab = df_above.pivot(index = xCols,columns = ['grp1','grp2'],
                                values = ["text_y","dist"]).reset_index()

        col_count = len(df_ab.columns.values)
        if col_count == 7: # (7,8)
            df_ab.insert(7, "text_y_1", "")
            df_ab.insert(8, "dist_1", 0)
            col_count = 9

        if col_count == 9: # (8,10)
            df_ab.insert(8, "text_y_2", "")
            df_ab.insert(10, "dist_2", 0)
            col_count = 11

        if col_count == 11: # (9,12)
            df_ab.insert(9, "text_y_3", "")
            df_ab.insert(12, "dist_3", 0)
            col_count = 13

        if col_count == 13: # (10, 15)
            df_ab.insert(10, "text_y_4", "")
            df_ab.insert(14, "dist_4", 0)
            col_count = 15

        if col_count == 15: # (11,16)
            df_ab.insert(11, "text_y_5", "")
            df_ab.insert(16, "dist_5", 0)
            col_count = 17

        if col_count == 17: # (11,16)
            df_ab.insert(12, "text_y_6", "")
            df_ab.insert(18, "dist_6", 0)
            col_count = 19

        if col_count == 19: # (11,16)
            df_ab.insert(13, "text_y_7", "")
            df_ab.insert(20, "dist_7", 0)

        # print("Columns:", df_ab.columns.to_list())
        df_ab.columns = cols + words_col_ab + dists_col_ab
        df_ab[dists_col_ab] = df_ab[dists_col_ab].fillna(0)

        #Find Below Words
        words_col_be = ["W" + str(i) + "Be" for i in range(1,noNeighbours + 1)]
        dists_col_be = ["d" + str(i) + "Be" for i in range(1,noNeighbours + 1)]
        df_below = df_merge[(df_merge["top_y"] > df_merge["bottom_x"]) &
                      (df_merge["top_y"] - df_merge["bottom_x"] <= verticalThresh) &
                      (df_merge["right_y"] >= df_merge["left_x"]) &
                      (df_merge["left_y"] <= df_merge["right_x"]) &
                      (df_merge["page_num_x"] == df_merge["page_num_y"])]
        df_below["dist"] = df_below["top_y"] - df_below["top_x"]
        df_below = df_below.sort_values(["top_x","left_x","top_y","left_y"],
                                        ascending = [True,True,True,True])
        df_below["grp1"] = df_below.groupby(by = xCols).cumcount()
        df_below["grp2"] = df_below["grp1"] + 100
        df_below = df_below[df_below["grp1"] < noNeighbours]
        df_be = df_below.pivot(index = xCols,columns = ['grp1','grp2'],
                                values = ["text_y","dist"]).reset_index()

        col_count = len(df_be.columns.values)
        if col_count == 7: # (7,8)
            df_be.insert(7, "text_y_1", "")
            df_be.insert(8, "dist_1", 0)
            col_count = 9

        if col_count == 9: # (8,10)
            df_be.insert(8, "text_y_2", "")
            df_be.insert(10, "dist_2", 0)
            col_count = 11

        if col_count == 11: # (9,12)
            df_be.insert(9, "text_y_3", "")
            df_be.insert(12, "dist_3", 0)
            col_count = 13

        if col_count == 13: # (10, 15)
            df_be.insert(10, "text_y_4", "")
            df_be.insert(14, "dist_4", 0)
            col_count = 15

        if col_count == 15: # (11,16)
            df_be.insert(11, "text_y_5", "")
            df_be.insert(16, "dist_5", 0)
            col_count = 17

        if col_count == 17: # (11,16)
            df_be.insert(12, "text_y_6", "")
            df_be.insert(18, "dist_6", 0)
            col_count = 19

        if col_count == 19: # (11,16)
            df_be.insert(13, "text_y_7", "")
            df_be.insert(20, "dist_7", 0)

        df_be.columns = cols + words_col_be + dists_col_be
        df_be[dists_col_be] = df_be[dists_col_be].fillna(0)

        #Find Left Words
        words_col_lf = ["W" + str(i) + "Lf" for i in range(1,noNeighbours + 1)]
        dists_col_lf = ["d" + str(i) + "Lf" for i in range(1,noNeighbours + 1)]
        azlns_col_lf = ["is" + str(i) + "lfAzLn" for i in range(1,noNeighbours + 1)]
        df_left = df_merge[(df_merge["right_y"] < df_merge["left_x"]) &
                      (df_merge["bottom_y"] >= df_merge["top_x"]) &
                      (df_merge["top_y"] <= df_merge["bottom_x"]) &
                      (df_merge["page_num_x"] == df_merge["page_num_y"])]
        df_left["dist"] = df_left["left_y"] - df_left["left_x"]
        df_left["isAzLn"] = df_left["line_num_x"] == df_left["line_num_y"]
        df_left = df_left.sort_values(["top_x","left_x","right_y"],
                                        ascending = [True,True,False])
        df_left["grp1"] = df_left.groupby(by = xCols).cumcount()
        df_left["grp2"] = df_left["grp1"] + 100
        df_left["grp3"] = df_left["grp1"] + 200
        df_left = df_left[df_left["grp1"] < noNeighbours]
        df_lf = df_left.pivot(index = xCols,columns = ['grp1','grp2','grp3'],
                                values = ["text_y","dist","isAzLn"]).reset_index()

        col_count = len(df_lf.columns.values)
        if col_count == 7:
            df_lf.insert(7, "text_y_1", "")
            df_lf.insert(8, "dist_1", 0)
            df_lf.insert(9, "azln_1", 0)
            col_count = 10

        if col_count == 10:
            df_lf.insert(8, "text_y_2", "")
            df_lf.insert(10, "dist_2", 0)
            df_lf.insert(12, "azln_2", 0)
            col_count = 13

        if col_count == 13:
            df_lf.insert(9, "text_y_3", "")
            df_lf.insert(12, "dist_3", 0)
            df_lf.insert(15, "azln_3", 0)
            col_count = 16

        if col_count == 16: # (10, 15)
            df_lf.insert(10, "text_y_4", "")
            df_lf.insert(14, "dist_4", 0)
            df_lf.insert(18, "azln_4", 0)
            col_count = 19

        if col_count == 19: # (11,16)
            df_lf.insert(11, "text_y_5", "")
            df_lf.insert(16, "dist_5", 0)
            df_lf.insert(21, "azln_5", 0)
            col_count = 22

        if col_count == 22: # (11,16)
            df_lf.insert(12, "text_y_6", "")
            df_lf.insert(18, "dist_6", 0)
            df_lf.insert(24, "azln_6", 0)
            col_count = 25

        if col_count == 25: # (11,16)
            df_lf.insert(13, "text_y_7", "")
            df_lf.insert(20, "dist_7", 0)
            df_lf.insert(27, "azln_7", 0)


        df_lf.columns = cols + words_col_lf + dists_col_lf + azlns_col_lf
        df_lf[dists_col_lf] = df_lf[dists_col_lf].fillna(0)
        df_lf[azlns_col_lf] = df_lf[azlns_col_lf].fillna(0)
        df_lf[azlns_col_lf] = df_lf[azlns_col_lf].astype(int)

        #Find Right Words
        words_col_rg = ["W" + str(i) + "Rg" for i in range(1,noNeighbours + 1)]
        dists_col_rg = ["d" + str(i) + "Rg" for i in range(1,noNeighbours + 1)]
        azlns_col_rg = ["is" + str(i) + "rgAzLn" for i in range(1,noNeighbours + 1)]
        df_right = df_merge[(df_merge["left_y"] > df_merge["right_x"]) &
                      (df_merge["bottom_y"] >= df_merge["top_x"]) &
                      (df_merge["top_y"] <= df_merge["bottom_x"]) &
                      (df_merge["page_num_x"] == df_merge["page_num_y"])]
        df_right["dist"] = df_right["left_y"] - df_right["left_x"]
        df_right["isAzLn"] = df_right["line_num_x"] == df_right["line_num_y"]
        df_right = df_right.sort_values(["top_x","left_x","left_y"],
                                        ascending = [True,True,True])
        df_right["grp1"] = df_right.groupby(by = xCols).cumcount()
        df_right["grp2"] = df_right["grp1"] + 100
        df_right["grp3"] = df_right["grp1"] + 200
        df_right = df_right[df_right["grp1"] < noNeighbours]
        df_rg = df_right.pivot(index = xCols,columns = ['grp1','grp2','grp3'],
                                values = ["text_y","dist","isAzLn"]).reset_index()

        col_count = len(df_rg.columns.values)
        if col_count == 7:
            df_rg.insert(7, "text_y_1", "")
            df_rg.insert(8, "dist_1", 0)
            df_rg.insert(9, "azln_1", 0)
            col_count = 10

        if col_count == 10:
            df_rg.insert(8, "text_y_2", "")
            df_rg.insert(10, "dist_2", 0)
            df_rg.insert(12, "azln_2", 0)
            col_count = 13

        if col_count == 13:
            df_rg.insert(9, "text_y_3", "")
            df_rg.insert(12, "dist_3", 0)
            df_rg.insert(15, "azln_3", 0)
            col_count = 16

        if col_count == 16: # (10, 15)
            df_rg.insert(10, "text_y_4", "")
            df_rg.insert(14, "dist_4", 0)
            df_rg.insert(18, "azln_4", 0)
            col_count = 19

        if col_count == 19: # (11,16)
            df_rg.insert(11, "text_y_5", "")
            df_rg.insert(16, "dist_5", 0)
            df_rg.insert(21, "azln_5", 0)
            col_count = 22

        if col_count == 22: # (11,16)
            df_rg.insert(12, "text_y_6", "")
            df_rg.insert(18, "dist_6", 0)
            df_rg.insert(24, "azln_6", 0)
            col_count = 25

        if col_count == 25: # (11,16)
            df_rg.insert(13, "text_y_7", "")
            df_rg.insert(20, "dist_7", 0)
            df_rg.insert(27, "azln_7", 0)


        df_rg.columns = cols + words_col_rg + dists_col_rg + azlns_col_rg
        df_rg[dists_col_rg] = df_rg[dists_col_rg].fillna(0)
        df_rg[azlns_col_rg] = df_rg[azlns_col_rg].fillna(0)
        df_rg[azlns_col_rg] = df_rg[azlns_col_rg].astype(int)

        #Merge all independent dataframe
        df_ngbrs = [df_ab,df_be,df_lf,df_rg]
        df_ngbr = reduce(lambda left,right: pd.merge(left,
                                                      right,
                                                      on = cols,
                                                      how = 'outer'),df_ngbrs)

        df_all = pd.merge(df,
                          df_ngbr,
                          on = cols,
                          how = "outer")
        return df_all
    except:
        print("FindCloseWordsNew",
              traceback.print_exc())
        return df_copy


# In[6]: Declare Function that updates null values to a default value

@util.timing
def fillNullsWithDefaults(df):
    """
    Extract and build model features
    :param df:
    :return:
    """
    # print(df.columns.values)
    #Discrete Features
    ftr_discrete = list(df_ftr[(df_ftr["Used"] == 1) &
                                (df_ftr["model_input"] == 1) &
                               (df_ftr["Var_Type"] == "Discrete")]["Column_Names"])

    for b in ftr_discrete:
        try:
            df[b] = df[b].astype(np.int32)
            df[b] = df[b].fillna(0)
        except:
            print('Discrete Column:', b, df[b].dtypes,"->",
                  traceback.print_exc())
            pass

    #Continuous Features
    ftr_cont = list(df_ftr[(df_ftr["Used"] == 1) &
                            (df_ftr["model_input"] == 1) &
                               (df_ftr["Var_Type"] == "Continous")]["Column_Names"])

    for b in ftr_cont:
        try:
            df[b] = df[b].astype(np.float32)
            df[b] = df[b].fillna(0.0)
        except:
            print('Continous Column:', b, df[b].dtypes,"->",
                  traceback.print_exc())
            pass

    return df

# In[6b]: Declare Functions that extract text based features


@util.timing
def extract_text_features(df):

    @util.timing
    def extract_number_words(df):
        """
        Extracts features related to number of words in the line
        :param df:
        :return:
        """
        temp = df.groupby(["page_num",
                           "line_num"])[['text']].count().reset_index()
        temp = temp.rename(columns={'text': 'words_on_line'})

        df = pd.merge(df, temp, on=['page_num','line_num'],
                      how = 'left')
        return df

    @util.timing
    def is_label(DF):
        """
        Checks whether a text belongs to predefined label or not
        """
        # cols = ["token_id","page_num","line_num","word_num",
        #         "line_text"]
        # DF = df[cols]
        punc = string.punctuation.replace('#', '').replace('%', '')
        punc = punc+'0123456789'
        pat = re.compile(f'[{punc}]')
        # t = time.time()
        DF['line_text_processed'] = DF["line_text"].replace(pat,"").str.strip().str.lower()
        DF['line_text_processed'] = DF['line_text_processed'].fillna('')
        DF['line_text_processed'] = DF['line_text_processed'].str.split(" ")
        # print("isLabel_1",time.time() - 1)

        # Create labelKeyword columns with the List as values
        # t = time.time()
        for i in labelKeywords:
            DF[i] = DF.apply(lambda x: labelKeywords[i], axis=1)
        # print("isLabel_2",time.time() - 1)

        # Create Column Lists
        list1 = ['line_text_processed']
        labelKeywords_list = list(labelKeywords.keys())
        list1.extend(labelKeywords_list)
        # print("List of cols",labelKeywords_list)

        # t = time.time()
        DF[labelKeywords_list] = DF[list1].apply(lambda x:pd.Series(np.mean([1 if z in x[i] else 0 for z in x[0]])
            for i in range(1,len(labelKeywords_list)+1)), axis=1)
        # print("isLabel_3",time.time() - t)

        del DF['line_text_processed']
        # df[labelKeywords_list] = DF[labelKeywords_list]

        return DF

    @util.timing
    def is_label_zero(DF):
        """
        Checks whether a text belongs to predefined label or not
        """
        labelKeywords_list = list(labelKeywords.keys())
        DF[labelKeywords_list] = 0

        return DF

    # @util.timing
    # def extract_ner_spacy(df):
    #     """
    #     Extract Named Entity Recognition from Spacy
    #     :param df:
    #     :return:
    #     """
    #     df['NER_Spacy'] = "UKWN"

    #     set_labels = set(cat_encoding['NER_Spacy'])
    #     for index, row in df.iterrows():
    #         text = str(row['text']).lower()
    #         doc = nlp(text)
    #         try:
    #             for ent in doc.ents:
    #                 label = ent.label_
    #                 if label in set_labels:
    #                     df.at[index, 'NER_Spacy'] = label
    #                     break
    #         except:
    #             pass

    @util.timing
    def extract_is_email(df):
        """
        Extract whether text is in email format or not
        :param df:
        :return:
        """
        def check_email(text):
            """
            Check valid email
            :param s:
            :return:
            """
            if not EMAIL_REGEX.match(str(text)):
                return 0
            else:
                return 1

        df['is_email'] = df['text'].apply(check_email)

    @util.timing
    def extract_token_specific_features(df):
        """
        Extract specific token related features
        :param df:
        :return:
        """
        is_date = []
        is_number = []
        has_currency = []
        total_len_text = []
        len_digit = []
        len_alpha = []
        len_spaces = []

        for index, row in df.iterrows():
            text = row['text']
            try:
                pd.to_datetime(text)
                is_date.append(1)
            except:
                is_date.append(0)
                pass
            try:
                float(text.replace(',', '').replace(' ',''))
                is_number.append(1)
            except:
                is_number.append(0)
                pass
            try:
                price = parse_price(text)
                if (not price.amount is None) and (not price.currency is None):
                    has_currency.append(1)
                else:
                    has_currency.append(0)
            except:
                has_currency.append(0)
                pass
            try:
                total_len_text.append(len(text))
                len_digit.append(sum(c.isdigit() for c in text))
                len_alpha.append(sum(c.isalpha() for c in text))
                len_spaces.append(sum(c.isspace() for c in text))
            except:
                total_len_text.append(0)
                len_digit.append(0)
                len_alpha.append(0)
                len_spaces.append(0)
                pass

        df['is_date'] = is_date
        df['is_number'] = is_number
        df['has_currency'] = has_currency
        df['total_len_text'] = total_len_text
        df['len_digit'] = len_digit
        df['len_alpha'] = len_alpha
        df['len_spaces'] = len_spaces
        df['len_others'] = df['total_len_text'] - df['len_digit'] - df['len_alpha'] - df['len_spaces']

    def no_of_puncs(text):
        """
        Count Number of colons, semicolons, hypens, commas and periods in a text
        Return: Returns total and individual counts
        """
        count = 0
        colon = 0
        semicolon = 0
        hyphen = 0
        comma = 0
        period = 0

        if not type(text) == float:
            for i in range (0, len(str(text))):
                txt = text[i]
                if txt in ('!', "," ,"\'" ,";" ,"\"",":", ".", "-" ,"?"):
                    if txt == ':':
                        colon = colon + 1
                    elif txt == ';':
                        semicolon = semicolon + 1
                    elif txt == '-':
                        hyphen = hyphen + 1
                    elif txt == ',':
                        comma = comma + 1
                    elif txt == '.':
                        period = period + 1
                    count = count + 1

        return [count,colon,semicolon,hyphen,comma,period]

    def is_amount(text):
        """
        Checks whether passed string is valid amount or not
        Returns: 1 if amount, 0 otherwise
        """
        try:
            if util.isAmount(text):
                p = parse_price(text)
                if p.amount is not None:
                    return 1
        except:
            print("is_amount except:",traceback.print_exc())
            return 0
        return 0

    # @util.timing
    def is_ID(row):
        """
        Checks whether text is ID or not
        Conditions for being ID: Lenght of digit should be greater than length of letters
        Total length of text should be greater than 7
        Returns: 1 if ID, 0 otherwise
        """
        if (not pd.isna(row['text'])) and row["len_digit"] > 0 and row["len_digit"] > row["len_alpha"] and len(str(row['text'])) > 7 and row["is_date"] == 0 and row["DATE"] == 0 and row["is_email"] == 0 and row["is_amount"] == 0:
            list_id = 1
        else:
            list_id = 0

        return list_id

    # @util.timing
    def spacy_onehot(df):

        # Get one-hot encoding for categorical features
        # NER_Spacy
        col_name = 'NER_Spacy'
        df, one_hot = util.onehot(df,col_name)

        for c in cat_encoding[col_name]:
            df[c] = 0
        df[one_hot.columns] = one_hot
        return df

    #Fill isLabel Features
    # df = is_label(df)
    df = is_label_zero(df)

    # print(list(df.columns.values))

    #Fill Nearby Words
    df = findCloseWordsNew(df)

    # print(list(df.columns.values))

    #Extract number of words in a line
    df = extract_number_words(df)

    #Extract token specific features
    extract_token_specific_features(df)
    #Check if a token is an e-mail
    extract_is_email(df)
    #Extract token specific features entity types given by Spacy
    df["NER_Spacy"] = "UKWN"
    # extract_ner_spacy(df)
    #Convert spacy features to one-hor encodings
    df = spacy_onehot(df)


    #Extract other token specific features for the text and the nearby words
    #Extract other token specific features for the text
    time1 = time.time()
    df.loc[df['text'].str.contains(gstin_pattern,
                                   regex= True,
                                   na=False), 'is_gstin_format'] = 1
    df["is_pan_format"] = df["text"].apply(pan_pattern)    
    print("is_pan_format unique value counts :",df["is_pan_format"].value_counts())
    df["wordshape"] = df['text'].apply(util.wordshape)
    df["is_alpha_numeric"] = df['text'].apply(util.is_alpha_numeric)
    df["is_alpha"] = df['text'].apply(util.is_alpha)
    df["is_amount"] = df['text'].apply(is_amount)
    df["chars_wo_punct"] = df["text"].str.replace('[^\w\s]','')
    df["is_alpha_wo_punct"] = df["chars_wo_punct"].str.isalpha()
    df["is_alnum_wo_punct"] = df["chars_wo_punct"].str.isalnum()
    df["is_num_wo_punct"] = df["chars_wo_punct"].str.isnumeric()
    df["is_nothing"] = (df["is_alpha_wo_punct"] == False) & (df["is_alnum_wo_punct"] == False) & (df["is_num_wo_punct"] == False)
    df["noOfPuncs_list"] = df['text'].apply(no_of_puncs)
    df[noOfPuncs_list] = df["noOfPuncs_list"].apply(foo)
    df.loc[(df['is_alpha_numeric']==1) & (df['is_alpha']==0) &
            (df['is_amount']==0) & (df['is_date']==0) &
            (df['is_number']==0), 'is_only_alpha_numeric'] = 1

    df['is_only_alpha_numeric'].fillna(0, inplace=True)
    df["is_ID"] = df.apply(is_ID, axis=1)
    # print("token specific: {:.3f} sec".format(time.time() - time1))

    del df['noOfPuncs_list']

    #Extract other token specific features for the nearby words
    # print("Neighbours",neighbWordsVec)
    # print(list(df.columns.values))
    time1 = time.time()
    # for i in neighbWordsVec:
    #     print("neighborvector",i)
    #     temp_list = []
    #     df[i + "_" + "is_alpha_numeric"] = df[i].apply(util.is_alpha_numeric)
    #     print(i + "_" + "is_alpha_numeric")
    #     df[i + "_" + "is_alpha"] = df[i].apply(util.is_alpha)
    #     df[i + "_" + "is_amount"] = df[i].apply(is_amount)
    #     df[i + "_" + "noOfPuncs_list"] = df[i].apply(no_of_puncs)
    #     print(i + "_" + "noOfPuncs_list")
    #     for j in noOfPuncs_list:
    #         temp_list.append(str(i) + "_" + str(j))
    #     df[temp_list] = df[i + "_" + str("noOfPuncs_list")].apply(foo)
    #     print("temp",temp_list)
    #     del df[i + "_" + str("noOfPuncs_list")]
    for i in neighbWordsVec:
        # print("neighborvector",i)
        temp_list = []
        df[i + "_" + "is_alpha_numeric"] = 0
        df[i + "_" + "is_alpha"] = 0
        df[i + "_" + "is_amount"] = 0
        # df[i + "_" + "noOfPuncs_list"] = df[i].apply(no_of_puncs)
        # print(i + "_" + "noOfPuncs_list")
        for j in noOfPuncs_list:
            temp_list.append(str(i) + "_" + str(j))
        df[temp_list] = 0
        # print("temp",temp_list)
        # del df[i + "_" + str("noOfPuncs_list")]
    print("Neighbour Token specific: {:.3f} sec".format(time.time() - time1))

    return df

# In[6b]: Split OCR lines based on lines on the right
@util.timing
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
            # print("Line breaks:",line_breaks)
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

def splitLineTextVLines_old(df):

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
                                sno = l.index(sub_row[["left","right"]].tolist()) - 1
                                line_breaks.append(sno)
                                break

            #Split the line text
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
                #Using ranges, create new line items with proper line text, x coord,width
                for i in j:
                    if i[1] != ":":
                        sub_lines = df_lines.iloc[i[0]:i[1]]
                    else:
                        sub_lines = df_lines.iloc[i[0]:]

                    line_texts = " ".join(list(sub_lines["text"].values))
                    line_right = max(list(sub_lines["right"].values))
                    line_left = min(list(sub_lines["left"].values))
                    line_width = line_right - line_left
                    for sub_ind,sub_row in sub_lines.iterrows():
                        df.loc[df["token_id"] == sub_row["token_id"],
                               ["line_right",
                                "line_left",
                                "line_width",
                                "line_text"]
                               ] = [line_right,
                                    line_left,
                                    line_width,
                                    line_texts]

        return df
    except:
        print("splitLineTextVLines",traceback.print_exc())
        return df_copy


# In[7]: Declare Image Feature extraction functions
@util.timing
def extract_image_features(df,imgFilePath):

    def getLineInfo(DF, imagepath, page_count):

        def findZeroPattern(vals):
            #Apr-13,2022 changed - The binarization takes place here in the calling function based on grayscale values
            binaryVals = vals // 255
            # binaryVals = vals
            #Apr-13,2022 changed - The binarization takes place here in the calling function based on grayscale values
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
                    if (word[2] - line[0] > 0.001):
                        return False
                return True

            def vOverlapLeft(line, word):
                if word[1] > line[3]:
                    return False
                elif word[3] < line[1]:
                    return False
                elif word[0] < line[0]:
                    if (line[0] - word[0] > 0.001):
                        return False
                return True

            def hOverlapTop(line, word):
                if word[0] > line[2]:
                    return False
                elif word[2] < line[0]:
                    return False
                elif word[1] < line[1]:
                    if line[1] - word[1] > 0.001:
                        return False
                return True

            def hOverlapDown(line, word):
                if word[0] > line[2]:
                    return False
                elif word[2] < line[0]:
                    return False
                elif word[3] > line[1]:
                    if word[3] - line[1] > 0.001:
                        return False
                return True


            isAbove = 0
            lenAbove = 0
            above_x1 = (-1,-1)
            above_x2 = (-1,-1)
            above_x1_x = -1
            above_x1_y = -1
            above_x2_x = -1
            above_x2_y = -1
            isBelow = 0
            lenBelow = 0
            below_x1 = (-1,-1)
            below_x2 = (-1,-1)
            below_x1_x = -1
            below_x1_y = -1
            below_x2_x = -1
            below_x2_y = -1
            isLeft = 0
            lenLeft = 0
            left_y1 = (-1,-1)
            left_y2 = (-1,-1)
            left_y1_x = -1
            left_y1_y = -1
            left_y2_x = -1
            left_y2_y = -1
            isRight = 0
            lenRight = 0
            right_y1 = (-1,-1)
            right_y2 = (-1,-1)
            right_y1_x = -1
            right_y1_y = -1
            right_y2_x = -1
            right_y2_y = -1

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
                    above_x1_x = hline[0]
                    above_x1_y = hline[1]
                    above_x2 = (hline[2],hline[3])
                    above_x2_x = hline[2]
                    above_x2_y = hline[3]
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
                    below_x1_x = hline[0]
                    below_x1_y = hline[1]
                    below_x2 = (hline[2],
                                hline[3])
                    below_x2_x = hline[2]
                    below_x2_y = hline[3]
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
                    left_y1_x = vline[0]
                    left_y1_y = vline[1]
                    left_y2 = (vline[2],vline[3])
                    left_y2_x = vline[2]
                    left_y2_y = vline[3]
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
                # print(wordBB,line_coord)
                if overlap:
                    isRight = 1
                    lenRight = vline[3] - vline[1]
                    right_y1 = (vline[0],vline[1])
                    right_y1_x = vline[0]
                    right_y1_y = vline[1]
                    right_y2 = (vline[2],vline[3])
                    right_y2_x = vline[2]
                    right_y1_y = vline[3]
                    break

            return pd.Series([isAbove,lenAbove,
                              above_x1,above_x1_x,above_x1_y,
                              above_x2,above_x2_x,above_x2_y,
                              isBelow, lenBelow,
                              below_x1,below_x1_x,below_x1_y,
                              below_x2,below_x2_x,below_x2_y,
                              isLeft, lenLeft,
                              left_y1,left_y1_x,left_y1_y,
                              left_y2,left_y2_x,left_y2_y,
                              isRight, lenRight,
                              right_y1,right_y1_x,right_y1_y,
                              right_y2,right_y2_x,right_y2_y
                              ])

        DF_new = pd.DataFrame()

        ret, imgs = cv2.imreadmulti(imagepath)
        for i in range(page_count):
            # im = imgs[i]

            blur = imgs[i]

            if len(blur.shape) == 3:
                blur = cv2.cvtColor(blur,cv2.COLOR_BGR2GRAY)

            #Apr-13,2022 changed - The binarization is simply based on grayscale pixel values
            # pre = cv2.threshold(blur, 200, 255,
            #                     cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
            # pre = pre/255
            # cv2.imwrite(imgFilePath + "_out1.jpg",pre)

            pre = cv2.threshold(blur,200,255,cv2.THRESH_BINARY)[1]
            # pre = np.where(blur > 150,1,0)
            # cv2.imwrite(imgFilePath + "_out2.jpg",pre)
            #Apr-13,2022 changed - The binarization is simply based on grayscale pixel values

            vlines, hlines = findLines(pre)

            # for vline in vlines:
            #     cv2.line(pre,(vline[0],vline[1]),
            #               (vline[2],vline[3]),(0,255,255))
            # for hline in hlines:
            #     cv2.line(pre,(hline[0],hline[1]),
            #               (hline[2],hline[3]),(0,255,255))
            # cv2.imwrite(imgFilePath + "_out.jpg",pre)

            height = pre.shape[0]
            width = pre.shape[1]

            hlines = [(hline[0] / width,hline[1] / height,hline[2] / width,
                       hline[3] / height) for hline in hlines]
            vlines = [(vline[0] / width,vline[1] / height,vline[2] / width,
                       vline[3] / height) for vline in vlines]

            tempar = DF.copy()

            if tempar.shape[0] > 0:
                tempar[lineInfo] = tempar.apply(findLinesClose,
                      args=(hlines,vlines),axis = 1)
                # print("FindLinesClose: ", time.time() - t)
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

@util.timing
def correctImageFeatures(df):

    df_copy = df.copy(deep = True)
    try:
        df_lines = df.groupby(["page_num",
                               "line_num",
                               "line_right_y1_x",
                               "line_left_y1_x"])
        grps = list(df_lines.groups)
        grps_wo_left = [(grp[0],grp[1],grp[2]) for grp in grps]
        from collections import Counter as ct
        d = dict(ct(grps_wo_left))
        outliers = [m for m in d.keys() if d[m] > 1]
        line_right_x = {}
        line_right_y1 = {}
        line_right_y2 = {}
        leftLineLen = {}
        for outlier in outliers:
            page_num = outlier[0]
            line_num = outlier[1]
            filt = df[(df["page_num"] == page_num) &
                      (df["line_num"] == line_num)]
            filt = filt.sort_values(["word_num"],
                                    ascending = [True])
            right = -1
            left = -1
            tkn_id = -1
            for f_ind,f_row in enumerate(filt.itertuples()):
                if f_ind != 0:
                    if (right == f_row.line_right_y1_x) and (left != f_row.line_left_y1_x):
                        if f_row.line_left_y1_x < right:
                            line_right_x[tkn_id] = f_row.line_left_y1_x
                            line_right_y1[tkn_id] = f_row.line_left_y1
                            line_right_y2[tkn_id] = f_row.line_left_y2
                            y1 = f_row.line_left_y1_y
                            y2 = f_row.line_left_y2_y
                            dist = y2 - y1
                            leftLineLen[tkn_id] = dist
                right = f_row.line_right_y1_x
                left = f_row.line_left_y1_x
                tkn_id = f_row.token_id
        df = util.assignVavluesToDf("line_right_y1_x",line_right_x,df)
        df = util.assignVavluesToDf("line_right_y2_x",line_right_x,df)
        df = util.assignVavluesToDf("line_right_y1",line_right_y1,df)
        df = util.assignVavluesToDf("line_right_y2",line_right_y2,df)
        df = util.assignVavluesToDf("leftLineLen",leftLineLen,df)
        return df
    except:
        print("correctImageFeatures",
              traceback.print_exc())
        return df_copy


# In[8]: Declare Functions that extract features from ocr files
@util.timing
def correctAmountTokens(df):
    df_copy = df.copy(deep = True)
    try:
        pun = list(string.punctuation)
        pun.remove(".")
        pun_str = "".join(pun)
        df["text_mod"] = df["text"]
        df["text_mod"].astype(str)
        #Strip spaces on the left and right
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
@util.timing
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

    df = df.sort_values(["line_top",
                            "line_left"])

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

@util.timing
def read_lines_from_table_new(DF):

    DF["tableLineNo"] = 0
    DF["isTableLine"] = 1
    DF["noNeighbours"] = 0
    DF["isLastAmt"] = 0

    DF["tableLineNo"] = DF["tableLineNo"].astype(np.int16)
    DF["isTableLine"] = DF["isTableLine"].astype(np.int16)
    DF["noNeighbours"] = DF["noNeighbours"].astype(np.int16)
    DF["isLastAmt"] = DF["isLastAmt"].astype(np.int16)

    DF = DF.sort_values(["page_num",
                         "line_num",
                         "word_num"],
                        ascending=[True,True,True])
    cols = ["page_num","line_num","word_num",
            "line_left","line_right",
            "line_top","line_down",
            "text","token_id",
            "isTableLine","noNeighbours","isLastAmt"]
    df = DF[cols]
    tblLineNo = {}
    noNgbrs = {}
    isLastAmt = {}
    try:
        page_lines = []
        for row in df.itertuples():
            page_num = int(row.page_num)
            line_num = int(row.line_num)
            row_left = row.line_left
            row_down = row.line_down
            row_top = row.line_top
            line = []
            line.append(str(page_num) + "-" + str(line_num))
            # t = time.time()
            cond = (df["page_num"] == page_num)
            cond = cond & (df["line_right"] < row_left)
            cond = cond & (~(df["line_top"] > row_down))
            cond = cond & (~(df["line_down"] < row_top))

            candidates = df[cond]
            # print("time filt", time.time() - t)

            candidates["page_num"] = candidates["page_num"].astype(np.int16)
            candidates["line_num"] = candidates["line_num"].astype(np.int16)
            if candidates.shape[0] > 0:
                for cand_row in candidates.itertuples():
                    if (row.line_top >= cand_row.line_top) and (row.line_down >= cand_row.line_down):
                        #To check if they overlap
                        if (cand_row.line_down > row.line_top) and (row.line_down > cand_row.line_top):
                            if (row.line_top - cand_row.line_top <= 0.005) and (row.line_down - cand_row.line_down <= 0.005):
                                line.append(str(page_num) + "-" + str(cand_row.line_num))
                    elif (cand_row.line_top >= row.line_top) and (cand_row.line_down >= row.line_down):
                        #To check if they overlap
                        if (row.line_down > cand_row.line_top) and (cand_row.line_down > row.line_top):
                            if (cand_row.line_top - row.line_top <= 0.005) and (cand_row.line_down - row.line_down <= 0.005):
                                line.append(str(page_num) + "-" + str(cand_row.line_num))
                    elif (cand_row.line_top >= row.line_top) and (cand_row.line_down <= row.line_down):
                        #To check if they overlap
                        line.append(str(page_num) + "-" + str(cand_row.line_num))
                    elif (cand_row.line_top <= row.line_top) and (cand_row.line_down >= row.line_down):
                        #To check if they overlap
                        line.append(str(page_num) + "-" + str(cand_row.line_num))

            page_lines.append(line)

        # t = time.time()
        table_lines = util.connect_lines(page_lines)
        # print("network", time.time() - t)

        for table_line_ind,table_line in enumerate(table_lines):
            no_neighbours = 0
            line_nums = []
            for tbl_line in table_line:
                page_num = int(tbl_line.split("-")[0])
                line_num = int(tbl_line.split("-")[1])
                line_nums.append(line_num)
                filt = df[((df["page_num"] == page_num) &
                        (df["line_num"] == line_num))]
                no_neighbours += filt.shape[0]

            lineNo = (table_line_ind + 1) + (page_num * 1000)

            filt.sort_values(["word_num"],
                             ascending = [True],
                             inplace = True)
            last_row = filt.iloc[-1]
            is_amt = int(util.isAmount(last_row["text"]))
            for line_num in line_nums:
                tkns = list(df[(df["page_num"] == page_num) &
                               (df["line_num"] == line_num)]["token_id"])
                tblLineNo.update({tkn:lineNo for tkn in tkns})
                noNgbrs.update({tkn:no_neighbours for tkn in tkns})
                isLastAmt.update({tkn:is_amt for tkn in tkns})

        DF["tableLineNo_New"] = DF["token_id"].map(tblLineNo)
        DF["noNeighbours_New"] = DF["token_id"].map(noNgbrs)
        DF["isLastAmt_New"] = DF["token_id"].map(isLastAmt)
        DF["tableLineNo"] = np.where(DF["tableLineNo_New"].isnull(),
                                     DF["tableLineNo"],
                                     DF["tableLineNo_New"])
        DF["noNeighbours"] = np.where(DF["noNeighbours_New"].isnull(),
                                      DF["noNeighbours"],
                                      DF["noNeighbours_New"])
        DF["isLastAmt"] = np.where(DF["isLastAmt_New"].isnull(),
                                   DF["isLastAmt"],
                                   DF["isLastAmt_New"])

        return DF
    except:
        print("Read_lines_from_table_new",
                traceback.print_exc())
        return None

@util.timing
def read_lines_from_table_new_bef_Apr9(df):

    df["tableLineNo"] = 0
    df["isTableLine"] = 0
    df["noNeighbours"] = 0
    df["isLastAmt"] = 0
    try:
        page_lines = []
        for ind, row in df.iterrows():
            page_num = int(row["page_num"])
            line_num = int(row["line_num"])
            row_left = row["line_left"]
            row_down = row["line_down"]
            row_top = row["line_top"]
            line = []
            line.append(str(page_num) + "-" + str(line_num))
            cond = (df["page_num"] == page_num)
            cond = cond & (df["line_right"] < row_left)
            cond = cond & (~(df["line_top"] > row_down))
            cond = cond & (~(df["line_down"] < row_top))

            candidates = df[cond]
            # print(list(candidates.columns.values))
            candidates["page_num"] = candidates["page_num"].astype(int)
            candidates["line_num"] = candidates["line_num"].astype(int)
            if candidates.shape[0] > 0:
                for cand_row_ind, cand_row in candidates.iterrows():
                    if (row["line_top"] >= cand_row["line_top"]) and (row["line_down"] >= cand_row["line_down"]):
                        #To check if they overlap
                        if (cand_row["line_down"] > row["line_top"]) and (row["line_down"] > cand_row["line_top"]):
                            if (row["line_top"] - cand_row["line_top"] <= 0.005) and (row["line_down"] - cand_row["line_down"] <= 0.005):
                                line.append(str(page_num) + "-" + str(cand_row["line_num"]))
                    elif (cand_row["line_top"] >= row["line_top"]) and (cand_row["line_down"] >= row["line_down"]):
                        #To check if they overlap
                        if (row["line_down"] > cand_row["line_top"]) and (cand_row["line_down"] > row["line_top"]):
                            if (cand_row["line_top"] - row["line_top"] <= 0.005) and (cand_row["line_down"] - row["line_down"] <= 0.005):
                                line.append(str(page_num) + "-" + str(cand_row["line_num"]))
                    elif (cand_row["line_top"] >= row["line_top"]) and (cand_row["line_down"] <= row["line_down"]):
                        #To check if they overlap
                        line.append(str(page_num) + "-" + str(cand_row["line_num"]))
                    elif (cand_row["line_top"] <= row["line_top"]) and (cand_row["line_down"] >= row["line_down"]):
                        #To check if they overlap
                        line.append(str(page_num) + "-" + str(cand_row["line_num"]))

            page_lines.append(line)

        table_lines = util.connect_lines(page_lines)

        for table_line_ind,table_line in enumerate(table_lines):
            no_neighbours = 0
            line_nums = []
            for tbl_line in table_line:
                page_num = int(tbl_line.split("-")[0])
                line_num = int(tbl_line.split("-")[1])
                line_nums.append(line_num)
                filt = df[((df["page_num"] == page_num) &
                        (df["line_num"] == line_num))]
                no_neighbours += filt.shape[0]

            lineNo = (table_line_ind + 1) + (page_num * 1000)

            filt.sort_values(["word_num"],
                             ascending = [True],
                             inplace = True)
            last_row = filt.iloc[-1]
            is_amt = int(util.isAmount(last_row["text"]))
            for line_num in line_nums:
                df.loc[((df["page_num"] == page_num) &
                        (df["line_num"] == line_num)),
                        ["tableLineNo","isTableLine",
                        "noNeighbours","isLastAmt"]] = [lineNo,1,
                                                        no_neighbours,
                                                        is_amt]

        return df
    except:
        print("Read_lines_from_table_new",
                traceback.print_exc())
        return None


@util.timing
def read_lines_from_table_old(df):

    try:
        df["tableLineNo"] = 0
        df["isTableLine"] = 0
        df["noNeighbours"] = 0
        df["isLastAmt"] = 0
        page_lines = []
        for ind, row in df.iterrows():
            page_num = int(row["page_num"])
            line_num = int(row["line_num"])
            # row_top = row["line_top"]
            # row_down = row["line_down"]
            line = []
            line.append(str(page_num) + "-" + str(line_num))

            candidates = df[(df["page_num"] == page_num) &
                            (df["line_num"] == line_num + 1)
                            ]
            # print(list(candidates.columns.values))
            candidates["page_num"] = candidates["page_num"].astype(int)
            candidates["line_num"] = candidates["line_num"].astype(int)
            if candidates.shape[0] > 0:
                for cand_row_ind, cand_row in candidates.iterrows():
                    if (row["line_top"] >= cand_row["line_top"]) and (row["line_down"] >= cand_row["line_down"]):
                        #To check if they overlap
                        if (cand_row["line_down"] > row["line_top"]) and (row["line_down"] > cand_row["line_top"]):
                            if (row["line_top"] - cand_row["line_top"] <= 0.005) and (row["line_down"] - cand_row["line_down"] <= 0.005):
                                line.append(str(page_num) + "-" + str(cand_row["line_num"]))
                    elif (cand_row["line_top"] >= row["line_top"]) and (cand_row["line_down"] >= row["line_down"]):
                        #To check if they overlap
                        if (row["line_down"] > cand_row["line_top"]) and (cand_row["line_down"] > row["line_top"]):
                            if (cand_row["line_top"] - row["line_top"] <= 0.005) and (cand_row["line_down"] - row["line_down"] <= 0.005):
                                line.append(str(page_num) + "-" + str(cand_row["line_num"]))
                    elif (cand_row["line_top"] >= row["line_top"]) and (cand_row["line_down"] <= row["line_down"]):
                        #To check if they overlap
                        line.append(str(page_num) + "-" + str(cand_row["line_num"]))
                    elif (cand_row["line_top"] <= row["line_top"]) and (cand_row["line_down"] >= row["line_down"]):
                        #To check if they overlap
                        line.append(str(page_num) + "-" + str(cand_row["line_num"]))

            page_lines.append(line)

        table_lines = util.connect_lines(page_lines)

        for table_line_ind,table_line in enumerate(table_lines):
            no_neighbours = 0
            line_nums = []
            for tbl_line in table_line:
                page_num = int(tbl_line.split("-")[0])
                line_num = int(tbl_line.split("-")[1])
                line_nums.append(line_num)
                filt = df[((df["page_num"] == page_num) &
                        (df["line_num"] == line_num))]
                no_neighbours += filt.shape[0]

            lineNo = (table_line_ind + 1) + (page_num * 1000)

            filt.sort_values(["word_num"],
                                ascending = [True],
                                inplace = True)
            last_row = filt.iloc[-1]
            is_amt = int(util.isAmount(last_row["text"]))
            for line_num in line_nums:
                df.loc[((df["page_num"] == page_num) &
                        (df["line_num"] == line_num)),
                        ["tableLineNo","isTableLine",
                        "noNeighbours","isLastAmt"]] = [lineNo,1,
                                                        no_neighbours,
                                                        is_amt]

        return df
    except:
        print("Read_lines_from_table_new",
                traceback.print_exc())
        return None

@util.timing
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

@util.timing
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

@util.timing
def correctOcrLineForNonEngTkn(df):
    df_copy = df.copy(deep = True)

    def isHOverlap(line1,line2):
        val = (line1["line_top"] > line2["line_down"])
        val = val or (line1["line_down"] < line2["line_top"])
        val = val or (line2["line_top"] > line1["line_down"])
        val = val or (line2["line_down"] < line1["line_top"])
        return not(val)
    
    def distance(line1,line2):
        try:
            if line1["line_left"] > line2["line_left"]:
                return [line1["line_left"] - line2["line_right"],"L"]
            elif line1["line_left"] < line2["line_left"]:
                return [line2["line_left"] - line1["line_right"],"R"]
            else:
                return [-1,"O"]
        except:
            print(traceback.print_exc())
            return [-1,"O"]

    try:

        for ind, row in df.iterrows():
            text = row["line_text"]
            if not(any(str(s).isalnum() for s in list(text))):
                df_filt = df[(df["page_num"] == row["page_num"]) &
                                (df["line_num"] == row["line_num"]) &
                                (df["token_id"] != row["token_id"])]
                distances = []
                for ind_1, row_1 in df_filt.iterrows():
                    line1 = {"line_left":row["line_left"],
                             "line_right":row["line_right"],
                             "line_top":row["line_top"],
                             "line_down":row["line_down"],
                             "page_num":row["page_num"],
                             "line_num":row["line_num"],
                             "line_text":row["line_text"]
                             }
                    line2 = {"line_left":row_1["line_left"],
                             "line_right":row_1["line_right"],
                             "line_top":row_1["line_top"],
                             "line_down":row_1["line_down"],
                             "page_num":row_1["page_num"],
                             "line_num":row_1["line_num"],
                             "line_text":row_1["line_text"]
                             }
                    if isHOverlap(line1, line2):
                        dist = distance(line1,line2)
                        if (dist[0] <= 0.05) and ((dist[1] != "L") or (dist[1] != "R")):
                            distances.append((dist,line2))
                s_distances = sorted(distances)
                if len(s_distances) > 0:
                    selected = s_distances[0]
                    # print("distances",selected)
                    dist = selected[0]
                    line = selected[1]
                    if dist[1] == "L":
                        # print("text",
                        #         row["line_text"],
                        #         line["line_text"],
                        #         line["line_text"] + " " + row["line_text"])
                        df.loc[(df["page_num"] == line["page_num"]) &
                                (df["line_num"] == line["line_num"]),
                                ["line_text",
                                 "line_right"]] = [line["line_text"] + " " + row["line_text"],
                                                   row["line_right"]]
                        #Modify the line that has the non-english characters
                        #Also modify the word_num for the merged line
                        max_wrd_num = max(df[(df["page_num"] == line["page_num"])
                                                & (df["line_num"] == line["line_num"])]["word_num"])
                        df.loc[(df["page_num"] == row["page_num"]) &
                                (df["line_num"] == row["line_num"]),
                                "word_num"] += max_wrd_num + 1
                        df.loc[(df["page_num"] == row["page_num"]) &
                                (df["line_num"] == row["line_num"]),
                                ["line_num",
                                 "line_text"]] = [line["line_num"],
                                                line["line_text"] + " " + row["line_text"]]

                    elif dist[1] == "R":
                        # print("text",
                        #         row["line_text"],
                        #         line["line_text"],
                        #         row["line_text"] + " " + line["line_text"])
                        df.loc[(df["page_num"] == line["page_num"]) &
                                (df["line_num"] == line["line_num"]),
                                ["line_text",
                                 "line_left"]] = [row["line_text"] + " " + line["line_text"],
                                                  row["line_left"]]
                        #Modify the line that has the non-english characters
                        #Also modify the word_num for the merged line
                        max_wrd_num = max(df[(df["page_num"] == row["page_num"])
                                                & (df["line_num"] == row["line_num"])]["word_num"])
                        df.loc[(df["page_num"] == line["page_num"]) &
                                (df["line_num"] == line["line_num"]),
                                "word_num"] += max_wrd_num + 1

                        df.loc[(df["page_num"] == row["page_num"]) &
                                (df["line_num"] == row["line_num"]),
                                ["line_num","line_text"]] = [line["line_num"],
                                                            row["line_text"] + " " + line["line_text"]]
        return df
    except:
        print("correctOcrLineForNonEngTkn",
              traceback.print_exc())
        return df_copy

def read_ocr_json_file(path,imgpath):

    @util.timing
    def get_image_dimensions(imgpath):
        """
        Return PNG/TIFF image dimensions

        """
        try:
            img = cv2.imread(imgpath, cv2.IMREAD_GRAYSCALE)
            img_h,img_w = img.shape[:2]
            return img_h, img_w
        except:
            print("get_image_dimensions",
                  traceback.print_exc())
            return None, None

    @util.timing
    def read_result_tag(resultList, img_h = None, img_w= None):
        rows = []
        tokenid = 10000

        for resultobj in resultList:
            pageNo = resultobj[0]["page"]
            lines = resultobj[0]["lines"]

            unit = resultobj[0]["unit"]
            unit_in_pixel = 96 if unit == "inch" else 1

            width = int(resultobj[0]["width"] * unit_in_pixel)
            height = int(resultobj[0]["height"] * unit_in_pixel)

            if img_h and img_w:
                width_scaler,height_scaler = float(img_w/width),float(img_h/height)
            else:
                width_scaler,height_scaler = 1.0,1.0

            lineNo = 0
            for line in lines:
                value = line["text"]
                bb = line["boundingBox"]
                left = min(bb[0] * unit_in_pixel, bb[6] * unit_in_pixel) / width
                right = max(bb[2] * unit_in_pixel, bb[4] * unit_in_pixel) / width
                top = min(bb[1] * unit_in_pixel,bb[3] * unit_in_pixel) / height
                down = max(bb[5] * unit_in_pixel,bb[7] * unit_in_pixel) / height

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
                    row["height_scaler"] = height_scaler
                    row["width_scaler"] = width_scaler

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

    #Apr 08, 2022 code to make token IDs unique
    @util.timing
    def makeTokenIdsUnq(df):
        base_tokenid = 10000
        df["token_id"] = df["token_id"] - base_tokenid
        df["token_id"] = (df["page_num"] * base_tokenid) + df["token_id"]
        return df
    #Apr 08, 2022 code to make token IDs unique


    # print(path)
    f = open(path, "r", encoding = "utf8")
    o = f.read()
    f.close()
    j = json.loads(o)
    resultList = j["analyzeResult"]["readResults"]
    # pageList = j["analyzeResult"]["pageResults"]
    pageList = []

    img_h, img_w = get_image_dimensions(imgpath)
    rows = read_result_tag(resultList,img_h, img_w)
    if len(rows) == 0 :
        return None

    df = pd.DataFrame(rows)
    df = read_page_result_tag(pageList, df)
    #Apr 08, 2022 code to make token IDs unique
    # df = makeTokenIdsUnq(df)
    #Apr 08, 2022 code to make token IDs unique

    return df

@util.timing
def neighborLabels(df):

    punc = string.punctuation.replace('#', '').replace('%', '')
    punc = punc + '0123456789'
    pat = re.compile(f'[{punc}]')

    @util.timing
    def neighborLabelsNew(df):

        list_dict = {}
        for key in aggregated_neighbours:
            list_dict[key] = []
            for i in labelKeywords:
                df[i+"_"+key] = df.apply(lambda x: labelKeywords[i], axis=1)
                list_dict[key].append(i+"_"+key)

        for key in aggregated_neighbours:
            neighbors = aggregated_neighbours[key]
            for n in neighbors:
                df['temp'] = df[n].replace(pat,"")
                df['temp'] = df['temp'].astype("string")
                df[key+"_"+n+"_processed"] = df['temp'].str.strip().str.lower()
                df[key+"_"+n+'_processed'] = df[key+"_"+n+"_processed"].fillna('')

            filter_col = [col for col in df if col.startswith(key+"_")]
            df[key+"_processed"] = df[filter_col].values.tolist()
            df[key+"_processed"] = df[key+"_processed"].apply(util.getUniqueWords)

            list1 = [key+"_processed"]
            labelKeywords_list = list_dict[key]
            list1.extend(labelKeywords_list)
            # print("LabelKeywords",labelKeywords_list)

            df[labelKeywords_list] = df[list1].apply(lambda x:pd.Series(np.mean([1 if z in x[i] else 0 for z in x[0]])
                              for i in range(1,len(labelKeywords_list)+1)),
                                                     axis=1)

        return df

    @util.timing
    def neighborLabelsZero(df):

        list_dict = {}
        for key in aggregated_neighbours:
            list_dict[key] = []
            for i in labelKeywords:
                list_dict[key].append(i+"_"+key)

        for key in aggregated_neighbours:
            labelKeywords_list = list_dict[key]
            df[labelKeywords_list] = [0] * len(labelKeywords_list)

        return df

    @util.timing
    def fuzzyNeighborLabels(df):

        def getUniqueSurroundingWords(sequence):
            seen = set()
            sourrounding = [x for x in sequence 
                            if not (x in seen or seen.add(x))]

            if '' in sourrounding:
                sourrounding.remove('')
            if np.NaN in sourrounding:
                sourrounding.remove(np.NaN)

            if len(sourrounding) > 3:
                sourrounding = sourrounding[:3]

            line = ' '.join(map(str,sourrounding))
            return line.strip()

        def getFuzzyScore(row, keywords):
            if row != '' and row != '%':
                match = rp_fz.extractOne(row,
                                         keywords,
                                         scorer = rp_fuzz.token_sort_ratio)
                if match:
                    score = match[1] / 100
                else:
                    score = 0
                return score
            return 0

        for label in labelKeywords_nonToken:
            keywords = labelKeywords_nonToken[label]
            df['temp'] = df['line_text'].replace(pat,"").str.strip().str.lower().replace('#'," number")
            df['temp'] = df['temp'].fillna('')
            df['fz_'+label] = df['temp'].apply(getFuzzyScore,
                                               args=(keywords,))
            df['fz_'+label] = df['fz_'+label].fillna(0)

        for key in aggregated_neighbours:
            neighbors = aggregated_neighbours[key]
            dfcol = df[neighbors]
            for n in neighbors:
                dfcol[n] = dfcol[n].str.lower().str.strip()
                dfcol[n] = dfcol[n].replace(pat,"").replace('#',"number")
                dfcol[n] = dfcol[n].fillna('')
            dfcol['combine_'+key] = dfcol[neighbors].values.tolist()
            dfcol['combine_'+key] = dfcol['combine_'+key].apply(getUniqueSurroundingWords)
            # print("Key for fuzzy neighbours:",key)

            for i in {k: v for k, v in labelKeywords_nonToken.items() 
                      if k in surrounding_label_feature}:
                keywords = labelKeywords_nonToken[i]
                # print("Keywords:",keywords)
                df['fz_'+i + "_" + key] = dfcol['combine_'+key].apply(getFuzzyScore,
                                                                      args=(keywords,))
                df['fz_'+i + "_" + key] = df['fz_'+i + "_" + key].fillna(0)
                # print("fz_"+i+"_"+key,df['fz_'+i + "_" + key])

        df.drop(['temp'], axis = 1, inplace=True)
        return df


    # df = neighborLabelsNew(df)
    df = neighborLabelsZero(df)
    #Commented on Feb-07.
    #New functions findLfNgbrs, findAbNgbrs, fuzzyMatchNgrs is better
    df = fuzzyNeighborLabels(df)

    return df

@util.timing
def extract_amount_features(DF):
    """
    Needs entire Document output as the input
    Returns back updated DataFrame with features
    """

    def extract_amount(text):
        """
        Checks whether passed string is valid amount or not
        Returns: 1 if amount, 0 otherwise
        """

        try:

            # preproc the amounts
            text = str(text)
            index_last_dot = text.rfind(".")
            if index_last_dot != -1:
                text = text.replace(".", ",")
                text = list(text)
                text[index_last_dot] = "."
                text = "".join(text)
                if (len(text) - index_last_dot) == 4:
                    text = list(text)
                    text.append("0")
                    text = "".join(text)

            if util.isAmount(text):
                p = parse_price(text)
                if p.amount is not None:
                    return (p.amount,1)
        except:
            return (0.0, 0)
        return (0.0, 0)

    DF['extracted_amount'], DF['is_amount'] = zip(*DF['text'].apply(extract_amount))
    temp_df = DF.loc[DF['extracted_amount'] > 0.0]
    temp_df.sort_values(['extracted_amount'], ascending=False, inplace=True)

    while temp_df.shape[0] > 1:
        # Remove max value if it is out of range
        max_val = temp_df.iloc[0]['extracted_amount']
        second_max_val = temp_df.iloc[1]['extracted_amount']
        if max_val > 2*second_max_val:
            temp_df = temp_df.iloc[1:]
        else:
            break

    extracted_amounts = set(temp_df['extracted_amount'])
    #print(extracted_amounts)
    if len(extracted_amounts) == 1:
        # Single amount found in the document and hence the Assumption: Single Page Invoice
        # Mark the bottom most as probable totalAmount and second bottom most as probable subTotal
        # if length is more than two, If only two amounts- mark the second bottom most as probable LI_itemvalue
        a = temp_df.sort_values(['top'], ascending=[False])

        probable_TA = (a.iloc[0][['token_id', 'page_num']])
        probable_TA['probable_totalAmount'] = 1
        probable_TA = (pd.DataFrame(probable_TA).T)
        DF = DF.merge(probable_TA, on=['token_id', 'page_num'], how="left")
        if temp_df.shape[0] > 2:
            probable_ST = (a.iloc[1][['token_id', 'page_num']])
            probable_ST['probable_subTotal'] = 1
            probable_ST = (pd.DataFrame(probable_ST).T)
            DF = DF.merge(probable_ST, on=['token_id', 'page_num'], how="left")
        elif temp_df.shape[0] == 2:
            probable_IV = (a.iloc[1][['token_id', 'page_num']])
            probable_IV['probable_LI_itemValue'] = 1
            probable_IV = (pd.DataFrame(probable_IV).T)
            DF = DF.merge(probable_IV, on=['token_id', 'page_num'], how="left")
    elif len(extracted_amounts) > 1:
        # Multiple amount found
        # Find the max amount (totalAmount) and the sum of two amounts (subTotal+taxAmount)
        total_amount = max(extracted_amounts)
        extracted_amounts.remove(total_amount)
        sub_total = 0
        tax_amount = 0
        for a in extracted_amounts:
            if total_amount - a in extracted_amounts:
                sub_total = max(a, total_amount-a)
                tax_amount = min(a, total_amount-a)
                break
        # Mark total amount
        if sub_total > 0:
            a = temp_df.loc[temp_df['extracted_amount'] == total_amount]
            a.sort_values(['top'], ascending=[False], inplace=True)
            probable_TA = (a.iloc[0][['token_id', 'page_num']])
            probable_TA['probable_totalAmount'] = 1
            probable_TA = (pd.DataFrame(probable_TA).T)
            DF = DF.merge(probable_TA, on=['token_id', 'page_num'], how="left")

            a = temp_df.loc[temp_df['extracted_amount'] == sub_total]
            a.sort_values(['top'], ascending=[False], inplace=True)
            probable_ST = (a.iloc[0][['token_id', 'page_num']])
            probable_ST['probable_subTotal'] = 1
            probable_ST = (pd.DataFrame(probable_ST).T)
            DF = DF.merge(probable_ST, on=['token_id', 'page_num'], how="left")

            a = temp_df.loc[temp_df['extracted_amount'] == tax_amount]
            a.sort_values(['top'], ascending=[False], inplace=True)
            probable_TAX = (a.iloc[0][['token_id', 'page_num']])
            probable_TAX['probable_taxAmount'] = 1
            probable_TAX = (pd.DataFrame(probable_TAX).T)
            DF = DF.merge(probable_TAX, on=['token_id', 'page_num'], how="left")


    # Add code for GST
    DF["probable_gstAmount_range"] = 0
    DF["probable_subtotal_range"] = 0
    DF["probable_tcsAmount_range"] = 0

    DF['PROBABLE_GST_AMOUNT_SLAB'] = 0
    DF['PROBABLE_SUBTOTAL_SLAB'] = 0

    if len(extracted_amounts) > 1:
        dict_gst_amounts_range = {key:0 for key in extracted_amounts}
        dict_subtotal_range = {key:0 for key in extracted_amounts}
        dict_tcsAmount_range = {key:0 for key in extracted_amounts}

        dict_gst_amounts_slab = {key:0 for key in extracted_amounts}
        dict_subtotal_slab = {key:0 for key in extracted_amounts}

        for key, val in dict_gst_amounts_range.items():
            for a in extracted_amounts:
                low_range = a*decimal.Decimal(0.044)
                high_range = a*decimal.Decimal(0.2)
                if (key >= low_range) & (key <= high_range):
                    dict_gst_amounts_range[key] = 1
                    dict_subtotal_range[a] += 1
                # TCS Range
                tax_low_range = a*decimal.Decimal(0.0075)
                tax_high_range = a*decimal.Decimal(0.075)
                if (key >= tax_low_range) & (key <= tax_high_range):
                    dict_tcsAmount_range[key] = 1
                for rate in gst_rates:
                    low_range = float(a)*rate*0.98
                    high_range = float(a)*rate*1.02
                    if (key >= low_range) & (key <= high_range):
                        dict_gst_amounts_slab[key] = 1
                        dict_subtotal_slab[a] += 1

        dict_gst_amounts_range[0.0] = 0
        dict_subtotal_range[0.0] = 0
        dict_tcsAmount_range[0.0] = 0

        dict_gst_amounts_slab[0.0] = 0
        dict_subtotal_slab[0.0] = 0

        DF["probable_gstAmount_range"] = DF["extracted_amount"].map(
            dict_gst_amounts_range)
        DF["probable_subtotal_range"] = DF["extracted_amount"].map(
            dict_subtotal_range)
        DF["probable_tcsAmount_range"] = DF["extracted_amount"].map(
            dict_tcsAmount_range)
        DF["PROBABLE_GST_AMOUNT_SLAB"] = DF["extracted_amount"].map(
            dict_gst_amounts_slab)
        DF["PROBABLE_SUBTOTAL_SLAB"] = DF["extracted_amount"].map(
            dict_subtotal_slab)


    DF.loc[DF['probable_subtotal_range'] > 0, 'probable_subtotal_range'] = 1
    DF.loc[DF['PROBABLE_SUBTOTAL_SLAB'] > 0, 'PROBABLE_SUBTOTAL_SLAB'] = 1

    if not 'probable_totalAmount' in DF.columns:
        DF['probable_totalAmount'] = 0
    if not 'probable_subTotal' in DF.columns:
        DF['probable_subTotal'] = 0
    if not 'probable_taxAmount' in DF.columns:
        DF['probable_taxAmount'] = 0
    if not 'probable_LI_itemValue' in DF.columns:
        DF['probable_LI_itemValue'] = 0
    if not 'probable_tcsAmount_range' in DF.columns:
        DF['probable_tcsAmount_range'] = 0
    if not 'PROBABLE_GST_AMOUNT_SLAB' in DF.columns:
        DF['PROBABLE_GST_AMOUNT_SLAB'] = 0
    if not 'PROBABLE_SUBTOTAL_SLAB' in DF.columns:
        DF['PROBABLE_SUBTOTAL_SLAB'] = 0

    DF[
        ['probable_totalAmount', 'probable_subTotal',
        'probable_taxAmount', 'probable_LI_itemValue',
        'probable_gstAmount_range', 'probable_subtotal_range',
        'probable_tcsAmount_range', 'PROBABLE_GST_AMOUNT_SLAB',
        'PROBABLE_SUBTOTAL_SLAB']
    ] = DF[['probable_totalAmount', 'probable_subTotal',
    'probable_taxAmount', 'probable_LI_itemValue',
    'probable_gstAmount_range','probable_subtotal_range',
    'probable_tcsAmount_range', 'PROBABLE_GST_AMOUNT_SLAB',
    'PROBABLE_SUBTOTAL_SLAB']
    ].fillna(value=0)

    return DF

@util.timing
def makeNonLIftrsZero_new(df):

    df_copy = df.copy(deep = True)

    try:
        # Code added to populate GST features for non-lineitem tokens

        df["probable_totalAmount"] = np.where(df["line_row"] > 0,
                                              0,
                                              df["probable_totalAmount"])
        df["probable_subTotal"] = np.where(df["line_row"] > 0,
                                           0,
                                           df["probable_subTotal"])
        df["probable_gstAmount_range"] = np.where(df["line_row"] > 0,
                                                  0,
                                                  df["probable_gstAmount_range"])
        df["probable_subtotal_range"] = np.where(df["line_row"] > 0,
                                                 0,
                                                 df["probable_subtotal_range"])
        df["probable_tcsAmount_range"] = np.where(df["line_row"] > 0,
                                                  0,
                                                  df["probable_tcsAmount_range"])
        df["PROBABLE_GST_AMOUNT_SLAB"] = np.where(df["line_row"] > 0,
                                                  0,
                                                  df["PROBABLE_GST_AMOUNT_SLAB"])
        df["PROBABLE_SUBTOTAL_SLAB"] = np.where(df["line_row"] > 0,
                                                0,
                                                df["PROBABLE_SUBTOTAL_SLAB"])


        #Default all other features to 0
        po_fields = ["fz_lblPoNumber_Above","fz_lblPoNumber_Left",
                     "fz_lblPoNumber_Above_rank","fz_lblPoNumber_Left_rank",
                     "is_ponumber_max"]

        cols = list(df.columns.values)
        cols_noli = [col for col in cols
                      if col not in li_ftrs + li_ftrs1 +
                      li_ngbr_ftrs + spatial_ftrs + 
                      numeric_ftrs + ocr_ftrs +
                      non_mdl_ftrs]
        
        for col in cols_noli:
            if col not in po_fields:
                if df[col].dtype == float:
                    df[col] = np.where(df["line_row"] > 0,0.0,
                                       df[col])
                elif df[col].dtype == int:
                    df[col] = np.where(df["line_row"] > 0,0,
                                       df[col])

        return df
    except:
        print("makeNonLIftrs",
              traceback.print_exc())
        return df_copy


@util.timing
def makeNonLIftrsZero(df):

    df_copy = df.copy(deep = True)

    try:
        # Code added to populate GST features for non-lineitem tokens
        # df.loc[df['line_row'] > 0, 'probable_totalAmount'] = 0
        # df.loc[df['line_row'] > 0, 'probable_subTotal'] = 0
        # df.loc[df['line_row'] > 0, 'probable_gstAmount_range'] = 0
        # df.loc[df['line_row'] > 0, 'probable_subtotal_range'] = 0
        # df.loc[df['line_row'] > 0, 'probable_tcsAmount_range'] = 0
        # df.loc[df['line_row'] > 0, 'PROBABLE_GST_AMOUNT_SLAB'] = 0
        # df.loc[df['line_row'] > 0, 'PROBABLE_SUBTOTAL_SLAB'] = 0

        df["probable_totalAmount"] = np.where(df["line_row"] > 0,
                                              0,
                                              df["probable_totalAmount"])
        df["probable_subTotal"] = np.where(df["line_row"] > 0,
                                           0,
                                           df["probable_subTotal"])
        df["probable_gstAmount_range"] = np.where(df["line_row"] > 0,
                                                  0,
                                                  df["probable_gstAmount_range"])
        df["probable_subtotal_range"] = np.where(df["line_row"] > 0,
                                                 0,
                                                 df["probable_subtotal_range"])
        df["probable_tcsAmount_range"] = np.where(df["line_row"] > 0,
                                                  0,
                                                  df["probable_tcsAmount_range"])
        df["PROBABLE_GST_AMOUNT_SLAB"] = np.where(df["line_row"] > 0,
                                                  0,
                                                  df["PROBABLE_GST_AMOUNT_SLAB"])
        df["PROBABLE_SUBTOTAL_SLAB"] = np.where(df["line_row"] > 0,
                                                0,
                                                df["PROBABLE_SUBTOTAL_SLAB"])


        #Default all other features to 0
        po_ab = list(df["fz_lblPoNumber_Above"])
        po_lf = list(df["fz_lblPoNumber_Left"])
        po_ab_rk = list(df["fz_lblPoNumber_Above_rank"])
        po_lf_rk = list(df["fz_lblPoNumber_Left_rank"])
        is_po_max = list(df["is_ponumber_max"])

        cols = list(df.columns.values)
        cols_noli = [col for col in cols
                      if col not in li_ftrs + li_ftrs1 +
                      li_ngbr_ftrs + spatial_ftrs + 
                      numeric_ftrs + ocr_ftrs +
                      non_mdl_ftrs]
        
        for col in cols_noli:
            if df[col].dtype == float:
                df.loc[(df["line_row"] != 0)
                        ,col] = 0.0
            elif df[col].dtype == int:
                df.loc[(df["line_row"] != 0)
                        ,col] = 0
        df["fz_lblPoNumber_Above"] = po_ab
        df["fz_lblPoNumber_Left"] = po_lf
        df["fz_lblPoNumber_Above_rank"] = po_ab_rk
        df["fz_lblPoNumber_Left_rank"] = po_lf_rk
        df["is_ponumber_max"] = is_po_max

        return df
    except:
        print("makeNonLIftrs",
              traceback.print_exc())
        return df_copy

@util.timing
def extract_max_amount_features_new(DF):
    """
    Needs entire Document output as the input
    Returns back updated DataFrame with features
    """
    DF[['first_max_amount',
        'second_max_amount',
        'extract_amount_rank']] = [0,0,0]
    df_copy = DF.copy(deep = True)
    try:
        max_upd = {}
        sec_max_upd = {}
        DF["extract_amount_rank"] = DF["extracted_amount"].rank(method = "min",
                                                                ascending = False)
        max_amount = DF["extracted_amount"].max()
        if max_amount > 0:
            max_amts = list(DF[(DF["extracted_amount"] == max_amount)]["token_id"])
            for amt_id in max_amts:
                max_upd[amt_id] = 1
            sec_max_amount = DF[DF["extracted_amount"] < max_amount]["extracted_amount"].max()
            if sec_max_amount > 0:
                max_amts = list(DF[(DF["extracted_amount"] == sec_max_amount)]["token_id"])
                print("Max Amounts",max_amts,max_amount,sec_max_amount,
                      max_amount - sec_max_amount <= 2.0)
                for amt_id in max_amts:
                    if max_amount - sec_max_amount <= 2.0:
                        max_upd[amt_id] = 1
                    else:
                        sec_max_upd[amt_id] = 1
                print("Second Max is empty",
                      sec_max_upd == {},sec_max_upd)
                if sec_max_upd == {}:
                    print("Second Max",sec_max_amount)
                    sec_max_amount = DF[DF["extracted_amount"] < sec_max_amount]["extracted_amount"].max()
                    print("Second Max",sec_max_amount)
                    if sec_max_amount > 0:
                        max_amts = list(DF[(DF["extracted_amount"] == sec_max_amount)]["token_id"])
                        for amt_id in max_amts:
                            sec_max_upd[amt_id] = 1
        if max_upd != {}:
            DF = util.assignVavluesToDf("first_max_amount",max_upd,DF)
        if sec_max_upd != {}:
            DF = util.assignVavluesToDf("second_max_amount",sec_max_upd,DF)
        print("extract max amount features done")
        return DF
    except:
        print("extract_max_amount_features_new exception",
              traceback.print_exc())
        return df_copy


@util.timing
def extract_max_amount_features(DF):
    """
    Needs entire Document output as the input
    Returns back updated DataFrame with features
    """
    DF[['first_max_amount',
        'second_max_amount']] = [0,0]
    df_copy = DF.copy(deep = True)
    try:

        temp_df = DF.loc[DF['extracted_amount'] > 0.0]
        temp_df.sort_values(['extracted_amount'],
                            ascending=False,
                            inplace=True)
        temp_df.drop_duplicates(['extracted_amount'],
                                keep='last',
                                inplace = True)

        while temp_df.shape[0] > 1:
            # Remove max value if it is out of range
            max_val = temp_df.iloc[0]['extracted_amount']
            second_max_val = temp_df.iloc[1]['extracted_amount']
            if max_val > 2*second_max_val:
                temp_df = temp_df.iloc[1:]
            else:
                break

        extracted_amounts = set(temp_df['extracted_amount'])
        print("Extracted Amounts",extracted_amounts)

        total_amount = 0
        sub_total = 0

        if len(extracted_amounts) == 1:
            total_amount = max(extracted_amounts)
        elif len(extracted_amounts) >= 2:
            total_amount = max(extracted_amounts)
            extracted_amounts.remove(total_amount)
            sub_total = max(extracted_amounts)
        print("Total",total_amount)
        if total_amount != 0:
            a = temp_df.loc[temp_df['extracted_amount'] == total_amount]
            a.sort_values(['top'],
                          ascending=[False],
                          inplace=True)
            probable_TA = (a.iloc[0][['token_id', 'page_num']])
            probable_TA['first_max_amount'] = 1
            probable_TA = (pd.DataFrame(probable_TA).T)
            DF = DF.merge(probable_TA,
                          on=['token_id',
                              'page_num'],
                          how="left")

        if sub_total != 0:
            a = temp_df.loc[temp_df['extracted_amount'].astype(int) == int(sub_total)]
            a.sort_values(['top'],
                          ascending=[False],
                          inplace=True)
            probable_ST = (a.iloc[0][['token_id',
                                      'page_num']])
            probable_ST['second_max_amount'] = 1
            probable_ST = (pd.DataFrame(probable_ST).T)
            DF = DF.merge(probable_ST,
                          on=['token_id',
                              'page_num'],
                          how = "left")

        if not 'first_max_amount' in DF.columns:
            DF['first_max_amount'] = 0
        if not 'second_max_amount' in DF.columns:
            DF['second_max_amount'] = 0

        DF[['first_max_amount',
            'second_max_amount']] = DF[['first_max_amount',
                                        'second_max_amount']].fillna(value=0)
        return DF
    except:
        print("extract_max_amount_features",
              traceback.print_exc())
        return df_copy



# In[8b]: Is_Prob_Item_Value

def potential_itemValue_zero(df):
    df['is_prob_itemValue'] = 0
    df['is_prob_itemQty'] = 0
    df['is_prob_unitPrice'] = 0
    return df


@util.timing
def potential_itemValue(df):

    import numpy as np
    from itertools import product
    import string

    def convert_to_float(row):
        try:
            value = 0.0
            punc = string.punctuation
            punc = punc.replace(".","")
            alpha = string.ascii_lowercase
            all_ = punc + alpha
            text = str(row["text"])
            #Remove alphabets and punctuation except decimal point
            mod_text = "".join([g for g in list(text.lower()) if g not in all_])
            #Replace "." other than actual decimals
            dec_count = mod_text.count(".")
            mod_text = mod_text.replace(".", "",dec_count - 1)
            mod_text = mod_text.strip()

            #Return the value in float
            if mod_text.replace(".","").isdigit():
                value = float(mod_text)
            return value
        except:
            print("Convert to Decimal - Potential Item Value:",
                  traceback.print_exc(),
                  row["text"])
            return 0.0

    def find_lineAmounts(qty,rate,vals):
        results = []
        try:
            combs = product(qty,rate)
            itm_vals = [val[-1] for val in vals]
            for comb in combs:
                if (comb[0][-1] > 0.0) and (comb[1][-1] > 0.0):
                    prod = comb[0][-1] * comb[1][-1]
                    mod1 = prod * 0.99
                    mod2 = prod * 1.01
                    for itm_indx,itm_val in enumerate(itm_vals):
                        if (itm_val >= mod1) and (itm_val <= mod2):
                            results.append((comb[0],
                                            comb[1],
                                            vals[itm_indx]))
                            # print("Qty, Rate, ItemValue",
                            #       comb[0],comb[1],vals[itm_indx])
            return results
        except:
            print("find_lineAmounts - potential item value:",
                  traceback.print_exc())
            return []

    df['is_prob_itemValue'] = 0
    df['is_prob_itemQty'] = 0
    df['is_prob_unitPrice'] = 0
    df_copy = df.copy(deep = True)
    try:
        print("Probable Item Value started")
        for page in df['page_num'].unique():
            # print("Loop page_num:", page)
            temp_df = df[(df['page_num'] == page) &
                         (df['line_row'] > 0)]

            if temp_df.shape[0] > 0:
                for line in temp_df['line_row'].unique():
                    qty_vals = []
                    rate_vals = []
                    val_vals = []
                    line_df = temp_df[(temp_df['line_row'] == line)]
                    qtys = line_df[(line_df["is_qty"] >= 0.85)]
                    # print("Actual Qty",qtys.shape)
                    if qtys.shape[0] == 0:
                        qtys = line_df[(line_df["is_qty"] == 0.0) &
                                       (line_df["ngbr_qty"] == 1)]
                    else:
                        max_qty = max(list(qtys["is_qty"]))
                        qtys = line_df[(line_df["is_qty"] == max_qty)]
                    # print("Actual Qty",qtys.shape)
                    if qtys.shape[0] > 0:
                        qtys["qty_val"] = qtys.apply(
                            lambda row:convert_to_float(row),
                            axis = 1)
                        qty_vals = qtys[["left",
                                         "top",
                                         "text",
                                         "qty_val"]].values.tolist()

                    # print("Quantities:",qty_vals)
                    rates = line_df[(line_df["is_unit_price"] >= 0.85)]
                    if rates.shape[0] == 0:
                        rates = line_df[(line_df["is_unit_price"] == 0.0) &
                                        (line_df["ngbr_unit_price"] == 1)]
                    else:
                        max_rate = max(list(rates["is_unit_price"]))
                        rates = line_df[(line_df["is_unit_price"] == max_rate)]
                    if rates.shape[0] > 0:
                        rates["rate_val"] = rates.apply(
                            lambda row:convert_to_float(row),
                            axis = 1)
                        rate_vals = rates[["left",
                                         "top",
                                         "text",
                                         "rate_val"]].values.tolist()
                    # print("Rates:",rate_vals)
                    vals = line_df[(line_df["is_item_val"] >= 0.85)]
                    if vals.shape[0] == 0:
                        vals = line_df[(line_df["is_item_val"] == 0.0) &
                                       (line_df["ngbr_item_val"] == 1)]
                    else:
                        max_val = max(list(vals["is_item_val"]))
                        vals = line_df[(line_df["is_item_val"] == max_val)]
                    if vals.shape[0] > 0:
                        vals["item_val_val"] = vals.apply(
                            lambda row:convert_to_float(row),
                            axis = 1)
                        val_vals = vals[["left",
                                         "top",
                                         "text",
                                         "item_val_val"]].values.tolist()
                    # print("Item Value:",val_vals)
                    results = find_lineAmounts(qty_vals,
                                               rate_vals,
                                               val_vals)
                    # print("Results:",results)
                    if len(results) > 0:
                        res_index = np.argmax([res[2][-1] for res in results])
                        result = results[res_index]
                        qty_row = result[0]
                        rate_row = result[1]
                        val_row = result[2]
                        # print("Result Qty, Rate, value:",
                        #       qty_row,rate_row,val_row)
                        df.loc[
                            (df["page_num"] == page) &
                            (df["line_row"] == line) &
                            (df["top"] == qty_row[1]) &
                            (df["left"] == qty_row[0]) &
                            (df["text"] == qty_row[2]),
                            ["is_qty",
                             "ngbr_qty",
                             "is_prob_itemQty"]] = [1.0,1,1]
                        df.loc[
                            (df["page_num"] == page) &
                            (df["line_row"] == line) &
                            (df["top"] != qty_row[1]) &
                            (df["left"] != qty_row[0]),
                            ["is_qty",
                             "ngbr_qty"]] = [0.5,0]
                        df.loc[
                            (df["page_num"] == page) &
                            (df["line_row"] == line) &
                            (df["top"] == rate_row[1]) &
                            (df["left"] == rate_row[0]) &
                            (df["text"] == rate_row[2]),
                            ["is_unit_price",
                             "ngbr_unit_price",
                             "is_prob_unitPrice"]] = [1.0,1,1]
                        df.loc[
                            (df["page_num"] == page) &
                            (df["line_row"] == line) &
                            (df["top"] != rate_row[1]) &
                            (df["left"] != rate_row[0]),
                            ["is_unit_price",
                             "ngbr_unit_price"]] = [0.5,0]
                        df.loc[
                            (df["page_num"] == page) &
                            (df["line_row"] == line) &
                            (df["top"] == val_row[1]) &
                            (df["left"] == val_row[0]) &
                            (df["text"] == val_row[2]),
                            ["is_item_val",
                              "ngbr_item_val",
                              "is_prob_itemValue"]] = [1.0,1,1]
                        df.loc[
                            (df["page_num"] == page) &
                            (df["line_row"] == line) &
                            (df["top"] != val_row[1]) &
                            (df["left"] != val_row[0]),
                            ["is_item_val",
                              "ngbr_item_val",
                              "is_prob_itemValue"]] = [0.5,0,0]

        return df
    except:
        print("Potential Item Value:",
              traceback.print_exc())
        return df_copy

# In[8b]: New GST features
@util.timing
def extract_GST_features_label_overlap(DF):

    df_copy = DF.copy(deep = True)

    try:
        CGST_LABELS = ['CGST', 'CENTRAL']
        SGST_LABELS = ['SGST', 'STATE']
        IGST_LABELS = ['IGST', 'INTEGRATED']
        GST_LEFT_ANCHOR_LABELS = ['TOTAL']
        TEMP = DF.loc[DF['is_amount'] == 1]
        DF["IGST_ABOVE"] = 0
        DF["IGST_ABOVE_DISTANCE"] = 1

        DF["CGST_ABOVE"] = 0
        DF["CGST_ABOVE_DISTANCE"] = 1

        DF["SGST_ABOVE"] = 0
        DF["SGST_ABOVE_DISTANCE"] = 1

        DF["IGST_LEFT"] = 0
        DF["IGST_LEFT_DISTANCE"] = 1

        DF["CGST_LEFT"] = 0
        DF["CGST_LEFT_DISTANCE"] = 1

        DF["SGST_LEFT"] = 0
        DF["SGST_LEFT_DISTANCE"] = 1

        DF["TOTAL_LEFT"] = 0

        for idx, row in TEMP.iterrows():
            left = row['left']
            right = row['right']
            top = row['top']
            bottom = row['bottom']
            token_id = row['token_id']
            page_num = row['page_num']

            LINE = DF.loc[(DF['line_left'] <= right) & (DF['line_right'] >=  left)
                          & (DF['line_down'] <= top)][['line_text',
                                                       'line_left',
                                                       'line_right',
                                                       'line_down']].drop_duplicates()
            LINE = LINE.loc[~LINE['line_text'].isna()]
            LINE['line_text'] = LINE['line_text'].str.upper()
            LINE['line_text'] = LINE['line_text'].str.replace(' ', '')

            L_IGST = LINE.loc[LINE['line_text'].str.contains('|'.join(IGST_LABELS))]
            L_CGST = LINE.loc[LINE['line_text'].str.contains('|'.join(CGST_LABELS))]
            L_SGST = LINE.loc[LINE['line_text'].str.contains('|'.join(SGST_LABELS))]

            LINE_LEFT = DF.loc[(DF['line_top'] <= bottom) & (DF['line_down'] >=  top)
                          & (DF['line_right'] <= left)][['line_text',
                                                         'line_left',
                                                         'line_right',
                                                         'line_down']].drop_duplicates()
            LINE_LEFT = LINE_LEFT.loc[~LINE_LEFT['line_text'].isna()]
            LINE_LEFT['line_text'] = LINE_LEFT['line_text'].str.upper()
            LINE_LEFT['line_text'] = LINE_LEFT['line_text'].str.replace(' ', '')

            L_LEFT_TOTAL = LINE_LEFT.loc[LINE_LEFT['line_text'].str.contains('|'.join(GST_LEFT_ANCHOR_LABELS))]

            if L_LEFT_TOTAL.shape[0] > 0:
                DF.loc[((DF['token_id'] == token_id) & (DF['page_num'] == page_num)), 'TOTAL_LEFT'] = 1

            L_LEFT_IGST = LINE_LEFT.loc[LINE_LEFT['line_text'].str.contains('|'.join(IGST_LABELS))]
            L_LEFT_CGST = LINE_LEFT.loc[LINE_LEFT['line_text'].str.contains('|'.join(CGST_LABELS))]
            L_LEFT_SGST = LINE_LEFT.loc[LINE_LEFT['line_text'].str.contains('|'.join(SGST_LABELS))]

            if L_LEFT_IGST.shape[0] > 0:
                L_LEFT_IGST.sort_values(['line_right'], ascending=[False], inplace=True)
                l = L_LEFT_IGST.iloc[0]['line_left']
                r = L_LEFT_IGST.iloc[0]['line_right']
                d = L_LEFT_IGST.iloc[0]['line_down']
                distance = left - r
                DF.loc[((DF['token_id'] == token_id) & (DF['page_num'] == page_num)), 'IGST_LEFT'] = 1
                DF.loc[((DF['token_id'] == token_id) & (DF['page_num'] == page_num)),
                        'IGST_LEFT_DISTANCE'] = distance

            if L_LEFT_CGST.shape[0] > 0:
                L_LEFT_CGST.sort_values(['line_right'], ascending=[False], inplace=True)
                l = L_LEFT_CGST.iloc[0]['line_left']
                r = L_LEFT_CGST.iloc[0]['line_right']
                d = L_LEFT_CGST.iloc[0]['line_down']
                distance = left - r
                DF.loc[((DF['token_id'] == token_id) & (DF['page_num'] == page_num)), 'CGST_LEFT'] = 1
                DF.loc[((DF['token_id'] == token_id) & (DF['page_num'] == page_num)),
                        'CGST_LEFT_DISTANCE'] = distance

            if L_LEFT_SGST.shape[0] > 0:
                L_LEFT_SGST.sort_values(['line_right'], ascending=[False], inplace=True)
                l = L_LEFT_SGST.iloc[0]['line_left']
                r = L_LEFT_SGST.iloc[0]['line_right']
                d = L_LEFT_SGST.iloc[0]['line_down']
                distance = left - r
                DF.loc[((DF['token_id'] == token_id) & (DF['page_num'] == page_num)), 'SGST_LEFT'] = 1
                DF.loc[((DF['token_id'] == token_id) & (DF['page_num'] == page_num)),
                        'SGST_LEFT_DISTANCE'] = distance

            if L_IGST.shape[0] > 0:
                L_IGST.sort_values(['line_down'], ascending=[False], inplace=True)
                l = L_IGST.iloc[0]['line_left']
                r = L_IGST.iloc[0]['line_right']
                d = L_IGST.iloc[0]['line_down']
                distance = top - d
                DF.loc[((DF['token_id'] == token_id) & (DF['page_num'] == page_num)), 'IGST_ABOVE'] = 1
                DF.loc[((DF['token_id'] == token_id) & (DF['page_num'] == page_num)),
                        'IGST_ABOVE_DISTANCE'] = distance

            if L_CGST.shape[0] > 0:
                L_CGST.sort_values(['line_down'], ascending=[False], inplace=True)
                l = L_CGST.iloc[0]['line_left']
                r = L_CGST.iloc[0]['line_right']
                d = L_CGST.iloc[0]['line_down']
                distance = top - d
                DF.loc[((DF['token_id'] == token_id) & (DF['page_num'] == page_num)), 'CGST_ABOVE'] = 1
                DF.loc[((DF['token_id'] == token_id) & (DF['page_num'] == page_num)),
                        'CGST_ABOVE_DISTANCE'] = distance

            if L_SGST.shape[0] > 0:
                L_SGST.sort_values(['line_down'], ascending=[False], inplace=True)
                # l = L_SGST.iloc[0]['line_left']
                r = L_SGST.iloc[0]['line_right']
                d = L_SGST.iloc[0]['line_down']
                distance = top - d
                DF.loc[((DF['token_id'] == token_id) & (DF['page_num'] == page_num)), 'SGST_ABOVE'] = 1
                DF.loc[((DF['token_id'] == token_id) & (DF['page_num'] == page_num)),
                        'SGST_ABOVE_DISTANCE'] = distance

        return DF
    except:
        print("extract GST Features",
              traceback.print_exc())
        return df_copy

# In[8c]: New GST features

def populate_max_probableGST_amount(DF):

    df_copy = DF.copy(deep = True)
    try:
        """
        Depends on PROBABLE_GST_AMOUNT_SLAB feature
        """
        TEMP = DF.loc[DF['PROBABLE_GST_AMOUNT_SLAB'] == 1]
        TEMP = TEMP[['token_id',
                     'page_num',
                     'extracted_amount']]

        l = list(set(TEMP['extracted_amount']))
        l.sort(reverse=True)
        # print(l)
        DF['PROBABLE_GST_MAX'] = 0
        DF['PROBABLE_GST_SECOND_MAX'] = 0

        if len(l) > 0:
            # Mark probable GST Max Amount
            max_amount = int(l[0])
            DF.loc[((DF['PROBABLE_GST_AMOUNT_SLAB'] == 1) & 
                    (DF['extracted_amount'] >= max_amount)),
                   'PROBABLE_GST_MAX'] = 1

        if len(l) > 1:
            # Mark probable GST Max Amount
            max_amount = int(l[0])
            second_max_amount = int(l[1])
            DF.loc[((DF['PROBABLE_GST_AMOUNT_SLAB'] == 1) &
                    (DF['extracted_amount'] < max_amount) &
                    (DF['extracted_amount'] >= second_max_amount)),
                   'PROBABLE_GST_SECOND_MAX'] = 1

        return DF
    except:
        print("populate max probableGST_amount",
              traceback.print_exc())
        return df_copy

# In[]: Declare Functions that finds neighbours and does fuzzy match with keywords
@util.timing
def findAbNgbrs(DF):

    DF["above_processed_ngbr"] = ""
    df_copy = DF.copy(deep = True)

    try:

        cols = ["page_num","line_num","word_num",
                "right","left","line_down","line_top",
                "line_left","line_right","line_text",
                "tableLineNo","lineLeft","line_left_y1",
                "above_processed_ngbr","line_row","is_HDR",
                "lineRight","line_right_y1_x"]

        DF.sort_values(["page_num",
                        "line_num",
                        "word_num"],
                       ascending = [True,True,True],
                       inplace = True)
        df = DF[cols]
        results = []
        results_bb = []
        for row in df.itertuples():
            page_num = row.page_num
            # line_num = row.line_num
            # line_no = row.tableLineNo
            lft_ln = row.lineLeft
            # print("Type of left_y1: ",type(row["line_left_y1"]))
            if isinstance(row.line_left_y1, str):
                lft_ln_x = (ast.literal_eval(row.line_left_y1)[0]
                            if lft_ln == 1 else 10)
            elif isinstance(row.line_left_y1, tuple):
                lft_ln_x = (row.line_left_y1[0]
                            if lft_ln == 1 else 10)

            #Jul 01 2022 - Take above neighbour as line header instead of line items
            # candidates = df[(df.page_num == page_num) &
            #                 (
            #                     (df.tableLineNo <= line_no) &
            #                     # (df["line_num"] < line_num) &
            #                     (df.line_top < row.line_top) &
            #                     (row.line_left - df.line_left <= 0.05) &
            #                     (df.line_top - row.line_top >= -0.1) #&
            #                 )
            #                 ]
            # candidates = df[(df.page_num == page_num) &
            #                 (
            #                     (df.tableLineNo <= line_no) &
            #                     # (df["line_num"] < line_num) &
            #                     (df.line_top < row.line_top) &
            #                     (row.line_left - df.line_left <= 0.05) &
            #                     (df.line_top - row.line_top >= -0.1) &
            #                     (df.line_row == 0)
            #                 )
            #                 ]
            candidates = df[(df.page_num == page_num) &
                            (
                                # (df.tableLineNo <= line_no) &
                                # (df["line_num"] < line_num) &
                                (df.line_top < row.line_top) &
                                # (row.line_left - df.line_left <= 0.05) &
                                # (df.line_top - row.line_top >= -0.1) &
                                # (df.line_down < row.line_top) &
                                (df.line_row == 0)
                            )
                            ]
            #Jul 01 2022 - Take above neighbour as line header instead of line items
            # print("Ab Ngbrs filter", time.time() - t)
            result = []
            result_bb = []
            ind = 0
            if candidates.shape[0] > 0:
                candidates.sort_values(["line_top"],
                                       ascending = [False],
                                       inplace = True)
                cur_line = 0
                prev_top = 0
                prev_down = 0
                for cand_row in candidates.itertuples():
                    if cand_row.line_num == cur_line:
                        continue
                    #Jul 11 2022 increase the distance threshold so that more neighbours are found
                    # if (row.line_top - cand_row.line_down > 0.02) & (ind == 0):
                    if (row.line_top - cand_row.line_down > 0.15) & (ind == 0):
                    #Jul 11 2022 increase the distance threshold so that more neighbours are found
                        if cand_row.is_HDR != 1:
                            break
                    cur_line = cand_row.line_num
                    cand_lft_ln = cand_row.lineLeft
                    if isinstance(cand_row.line_left_y1,str):
                        cand_lft_ln_x = (ast.literal_eval(cand_row.line_left_y1)[0]
                                          if cand_lft_ln == 1 else 10)
                    elif isinstance(cand_row.line_left_y1,tuple):
                        cand_lft_ln_x = (cand_row.line_left_y1[0]
                                          if cand_lft_ln == 1 else 10)

                    lft_ln_diff = abs(lft_ln_x - cand_lft_ln_x)
                    lft_ln_same = [1 if lft_ln_diff <= 0.001 else 0]

                    #check if lines overlap. If overlaps and left is closer and closest from top, then it must be the only ngbr
                    #Check if the line coordinates above to the current token and candidates are same
                    #Check if line coordinates left to the current token and candidates are same
                    #Check if line coordinates right to the current token and candidates are same
                    #check if candidate's line_left is far away. If yes, don't consider
                    if cand_row.line_left >= row.line_left:
                        #Jul 11 2022 increase the distance threshold so that more neighbours are found
                        # if cand_row.line_left <= row.line_right:
                        #Jul 13 2022 - If overlap is there, simply add the ngbr
                        if row.line_right - cand_row.line_left >= 0.015:
                            #Jul 19 2022 - capture line_top and line_down and compare it with previous neighbour
                            if prev_down == 0 or prev_top == 0:
                                if row.line_top - cand_row.line_down <= 0.03:
                                    result.append(cand_row.line_text)
                                    result_bb.append((cand_row.line_text,
                                                      cand_row.line_top,
                                                      cand_row.line_down,
                                                      cand_row.line_left,
                                                      cand_row.line_right))
                                elif cand_row.is_HDR == 1:
                                    result.append(cand_row.line_text)
                                    result_bb.append((cand_row.line_text,
                                                      cand_row.line_top,
                                                      cand_row.line_down,
                                                      cand_row.line_left,
                                                      cand_row.line_right))
                            else:
                                if prev_top - cand_row.line_down <= 0.03:
                                    result.append(cand_row.line_text)
                                    result_bb.append((cand_row.line_text,
                                                      cand_row.line_top,
                                                      cand_row.line_down,
                                                      cand_row.line_left,
                                                      cand_row.line_right))
                            prev_down = cand_row.line_down
                            prev_top = cand_row.line_top
                            # result.append(cand_row.line_text)
                            #Jul 19 2022 - capture line_top and line_down and compare it with previous neighbour
                        # else:

                        # if row.line_right - cand_row.line_left >= 0.01:
                            #Jul 11 2022 increase the distance threshold so that more neighbours are found
                            # result.append(cand_row.line_text)
                            #Jul 11 2022 increase the distance threshold so that more neighbours are found
                        # elif (cand_row.line_left - row.line_right <= 0.005) and lft_ln_same:
                            # if cand_row.line_left - row.line_left <= 0.02:
                                # result.append(cand_row.line_text)
                            #Jul 11 2022 increase the distance threshold so that more neighbours are found
                    else:
                        #Jul 12 2022 - if there is overlap, then don't check for any other condition
                        # if (cand_row.line_right > row.line_left) and (cand_row.line_right <= row.line_right):
                        if (cand_row.line_right > row.line_left):
                        #Jul 12 2022 - if there is overlap, then don't check for any other condition
                            #Jul 19 2022 - capture line_top and line_down and compare it with previous neighbour
                            if prev_down == 0 or prev_top == 0:
                                if row.line_top - cand_row.line_down <= 0.03:
                                    result.append(cand_row.line_text)
                                    result_bb.append((cand_row.line_text,
                                                      cand_row.line_top,
                                                      cand_row.line_down,
                                                      cand_row.line_left,
                                                      cand_row.line_right))
                                elif cand_row.is_HDR == 1:
                                    result.append(cand_row.line_text)
                                    result_bb.append((cand_row.line_text,
                                                      cand_row.line_top,
                                                      cand_row.line_down,
                                                      cand_row.line_left,
                                                      cand_row.line_right))
                            else:
                                if prev_top - cand_row.line_down <= 0.03:
                                    result.append(cand_row.line_text)
                                    result_bb.append((cand_row.line_text,
                                                      cand_row.line_top,
                                                      cand_row.line_down,
                                                      cand_row.line_left,
                                                      cand_row.line_right))
                            prev_down = cand_row.line_down
                            prev_top = cand_row.line_top
                            # result.append(cand_row.line_text)
                            #Jul 19 2022 - capture line_top and line_down and compare it with previous neighbour
                        else:
                            #Jul 11 2022 increase the distance threshold so that more neighbours are found
                            # if row.line_left - cand_row.line_left <= 0.06:
                            #Jul 12 2022 - if no overlap, check the distance between extremities are lt thresh
                            # if row.line_left - cand_row.line_left <= 0.06:
                            if (row.line_left - cand_row.line_right <= 0.01):
                            #Jul 12 2022 - if no overlap, check the distance between extremities are lt thresh
                            #Jul 11 2022 increase the distance threshold so that more neighbours are found
                            #Jul 19 2022 - capture line_top and line_down and compare it with previous neighbour
                                if prev_down == 0 or prev_top == 0:
                                    if row.line_top - cand_row.line_down <= 0.03:
                                        result.append(cand_row.line_text)
                                        result_bb.append((cand_row.line_text,
                                                          cand_row.line_top,
                                                          cand_row.line_down,
                                                          cand_row.line_left,
                                                          cand_row.line_right))
                                else:
                                    if prev_top - cand_row.line_down <= 0.03:
                                        result.append(cand_row.line_text)
                                        result_bb.append((cand_row.line_text,
                                                          cand_row.line_top,
                                                          cand_row.line_down,
                                                          cand_row.line_left,
                                                          cand_row.line_right))
                                prev_down = cand_row.line_down
                                prev_top = cand_row.line_top
                                # result.append(cand_row.line_text)
                            #Jul 19 2022 - capture line_top and line_down and compare it with previous neighbour
            results.append(result)
            results_bb.append(result_bb)
        # df["above_processed_ngbr"] = results
        DF["above_processed_ngbr"] = results
        DF["above_processed_ngbr_bb"] = results_bb
        return DF
    except:
        print("findAbNgbrs",traceback.print_exc())
        return df_copy

@util.timing
def findAbNgbrs_old(df):

    df["above_processed_ngbr"] = ""
    df_copy = df.copy(deep = True)

    try:

        cols = ["page_num","line_num","word_num",
                "right","left","line_down","line_top",
                "line_left","line_right","line_text",
                "tableLineNo","lineLeft","line_left_y1",
                "above_processed_ngbr"]

        df.sort_values(["page_num",
                        "line_num",
                        "word_num"],
                       ascending = [True,True,True],
                       inplace = True)
        results = []
        for ind,row in df.iterrows():
            page_num = row["page_num"]
            line_num = row["line_num"]
            line_no = row["tableLineNo"]
            lft_ln = row["lineLeft"]
            # print("Type of left_y1: ",type(row["line_left_y1"]))
            if isinstance(row["line_left_y1"], str):
                lft_ln_x = (ast.literal_eval(row["line_left_y1"])[0]
                            if lft_ln == 1 else 10)
            elif isinstance(row["line_left_y1"], tuple):
                lft_ln_x = (row["line_left_y1"][0]
                            if lft_ln == 1 else 10)

            candidates = df[(df["page_num"] == page_num) &
                            (
                                (df["tableLineNo"] <= line_no) &
                                # (df["line_num"] < line_num) &
                                (df["line_top"] < row["line_top"]) &
                                (row["line_left"] - df["line_left"] <= 0.05) &
                                (df["line_top"] - row["line_top"] >= -0.1) #&
                            )
                            ]
            # print("Ab Ngbrs filter", time.time() - t)
            result = []
            candidates.sort_values(["line_top"],
                                   ascending = [False])
            ind = 0                                    
            if candidates.shape[0] > 0:
                candidates.sort_values(["line_top"],
                                       ascending = [False],inplace=True)
                cur_line = 0
                for cand_ind,cand_row in candidates.iterrows():
                    if cand_row["line_num"] == cur_line:
                        continue
                    if (row["line_top"] - cand_row["line_down"] > 0.02) & (ind == 0):
                        break
                    cur_line = cand_row["line_num"]
                    cand_lft_ln = cand_row["lineLeft"]
                    if isinstance(cand_row["line_left_y1"],str):
                        cand_lft_ln_x = (ast.literal_eval(cand_row["line_left_y1"])[0]
                                         if cand_lft_ln == 1 else 10)
                    elif isinstance(cand_row["line_left_y1"],tuple):
                        cand_lft_ln_x = (cand_row["line_left_y1"][0]
                                         if cand_lft_ln == 1 else 10)


                    lft_ln_diff = abs(lft_ln_x - cand_lft_ln_x)
                    lft_ln_same = [1 if lft_ln_diff <= 0.001 else 0]
    
                    #check if lines overlap. If overlaps and left is closer and closest from top, then it must be the only ngbr
                    #Check if the line coordinates above to the current token and candidates are same
                    #Check if line coordinates left to the current token and candidates are same
                    #Check if line coordinates right to the current token and candidates are same
                    #check if candidate's line_left is far away. If yes, don't consider
                    if cand_row["line_left"] > row["line_left"]:
                        if cand_row["line_left"] <= row["line_right"]:
                            # if cand_row["line_left"] - row["line_left"] <= 0.02:
                            result.append(cand_row["line_text"])
                        elif (cand_row["line_left"] - row["line_right"] <= 0.005) and lft_ln_same:
                            if cand_row["line_left"] - row["line_left"] <= 0.02:
                                result.append(cand_row["line_text"])
                    else:
                        if (cand_row["line_right"] > row["line_left"]) and (cand_row["line_right"] <= row["line_right"]):
                            result.append(cand_row["line_text"])
                        else:
                            if row["line_left"] - cand_row["line_left"] <= 0.06:
                                result.append(cand_row["line_text"])
            results.append(result)
        df["above_processed_ngbr"] = results
        # DF["above_processed_ngbr"] = results
        return df
    except:
        print("findAbNgbrs",traceback.print_exc())
        return df_copy

@util.timing
def findLfNgbrs(DF):

    DF["left_processed_ngbr"] = ""
    df_copy = DF.copy(deep = True)
    try:
        cols = ["page_num","line_num","word_num",
                "right","left","line_down","line_top",
                "line_left","line_right","line_text",
                "left_processed_ngbr","line_row",
                "text"]
        DF.sort_values(["page_num",
                        "line_num",
                        "word_num"],
                       ascending = [True,True,True],
                       inplace = True)
        df = DF[cols]
        results = []
        for row in df.itertuples():
            page_num = row.page_num
            cond1 = (df["right"] < row.left)
            cond1 = cond1 & (~(df["line_down"] < row.line_top))
            cond1 = cond1 & (~(df["line_top"] > row.line_down))
            cond1 = cond1 & (df["page_num"] == page_num)
            #Jul 04, 2022 - Neighbours should not be from line items
            cond1 = cond1 & (df["line_row"] == 0)
            #Jul 04, 2022 - Neighbours should not be from line items
            candidates = df[cond1]
            candidates.sort_values(["line_num",
                                    "word_num"],
                                   ascending = [False,False],
                                   inplace = True)
            if row.text == "4636.26":
                print("Candidates",candidates)
            # print("Left Ngbrs filter time",time.time() - t)
            result = []
            # print("text", row["text"])

            # t = time.time()
            if candidates.shape[0] > 0:
                candidates.sort_values(["left"],
                                       ascending = [False],
                                       inplace = True)
                for cand_row in candidates.itertuples():
                    ln_text = ""
                    overlap_area = 0.0
                    lft = 0.0
                    rgt = 0.0
                    if (row.line_num == cand_row.line_num) and (row.word_num > cand_row.word_num):
                        ln_texts = cand_row.line_text.split(" ")
                        ln_text = " ".join([x for i,x in enumerate(ln_texts) 
                                            if i < row.word_num])
                        overlap_area = 2.0
                        lft = cand_row.line_left
                        rgt = cand_row.line_right
                    elif (row.line_num != cand_row.line_num):
                        if cand_row.line_right > row.line_left:
                            continue
                        if (row.line_top >= cand_row.line_top) and (row.line_down <= cand_row.line_down):
                            ln_text = cand_row.line_text
                            overlap_area = 1.0
                            lft = cand_row.line_left
                            rgt = cand_row.line_right
                        elif (row.line_top <= cand_row.line_top) and (row.line_down >= cand_row.line_down):
                            ln_text = cand_row.line_text
                            overlap_area = 1.0
                            lft = cand_row.line_left
                            rgt = cand_row.line_right
                        elif (row.line_top > cand_row.line_top) and (row.line_down > cand_row.line_down):
                            range_ht = row.line_down - cand_row.line_top
                            overlap_ht = cand_row.line_down - row.line_top
                            overlap_area = overlap_ht / range_ht
                            ln_text = cand_row.line_text
                            lft = cand_row.line_left
                            rgt = cand_row.line_right

                        elif (row.line_top < cand_row.line_top) and (row.line_down < cand_row.line_down):
                            range_ht = cand_row.line_down - row.line_top
                            overlap_ht = row.line_down - cand_row.line_top
                            overlap_area = overlap_ht / range_ht
                            ln_text = cand_row.line_text
                            lft = cand_row.line_left
                            rgt = cand_row.line_right

                    if (ln_text,overlap_area,lft,rgt) not in result:
                        result.append((ln_text,
                                       overlap_area,
                                       lft,rgt))
                    # print("intermediate result",result)

                #Take only one left ngbr if there are two overlapping neighbors.
                #This can happen especially when there are some left regions
                #that actually belongs to the top or below line but because
                #document is folded, the alignment won't be proper
                filtered = [r for r in result if r[1] < 1.0]
                # print("filtered result", filtered)
                del_rows = []
                for r in filtered:
                    for g in filtered:
                        if g != r:
                            g_lft = g[2]
                            g_rgt = g[3]
                            r_lft = r[2]
                            r_rgt = r[3]
                            g_area = g[1]
                            r_area = r[1]
                            if not((g_rgt < r_lft) or (g_lft > r_rgt)):
                                if g_area > r_area:
                                    del_rows.append(r)
                                else:
                                    del_rows.append(g)
                # print("del_rows", del_rows)
                # del_rows = []
                result = [r[0] for r in result if r not in del_rows]
                # print("final", result)
            # print("Left Ngbrs for loop",time.time() - t)

            results.append(result)
        # df["left_processed_ngbr"] = results
        DF["left_processed_ngbr"] = results
        return DF
    except:
        print("findLfNgbrs",
              traceback.print_exc())
        return df_copy

@util.timing
def findLfNgbrs_old(df):

    df["left_processed_ngbr"] = ""
    df_copy = df.copy(deep = True)
    try:
        cols = ["page_num","line_num","word_num",
                "right","left","line_down","line_top",
                "line_left","line_right","line_text",
                "left_processed_ngbr"]
        df.sort_values(["page_num",
                        "line_num",
                        "word_num"],
                       ascending = [True,True,True],
                       inplace = True)
        results = []
        for ind,row in df.iterrows():
            page_num = row["page_num"]
            cond1 = (df["right"] < row["left"])
            cond1 = cond1 & (~(df["line_down"] < row["line_top"]))
            cond1 = cond1 & (~(df["line_top"] > row["line_down"]))
            cond1 = cond1 & (df["page_num"] == page_num)
            candidates = df[cond1]
            candidates.sort_values(["line_num",
                                    "word_num"],
                                   ascending = [False,False],
                                   inplace = True)
            # print("Left Ngbrs filter time",time.time() - t)
            result = []
            # print("text", row["text"])

            # t = time.time()
            if candidates.shape[0] > 0:
                candidates.sort_values(["left"],
                                       ascending = [False],
                                       inplace=True)
                for cand_ind,cand_row in candidates.iterrows():
                    ln_text = ""
                    overlap_area = 0.0
                    lft = 0.0
                    rgt = 0.0
                    if (row["line_num"] == cand_row["line_num"]) and (row["word_num"] > cand_row["word_num"]):
                        ln_texts = cand_row["line_text"].split(" ")
                        ln_text = " ".join([x for i,x in enumerate(ln_texts) 
                                            if i < row["word_num"]])
                        overlap_area = 2.0
                        lft = cand_row["line_left"]
                        rgt = cand_row["line_right"]
                    elif (row["line_num"] != cand_row["line_num"]):
                        if cand_row["line_right"] > row["line_left"]:
                            continue
                        if (row["line_top"] >= cand_row["line_top"]) and (row["line_down"] <= cand_row["line_down"]):
                            ln_text = cand_row["line_text"]
                            overlap_area = 1.0
                            lft = cand_row["line_left"]
                            rgt = cand_row["line_right"]
                        elif (row["line_top"] <= cand_row["line_top"]) and (row["line_down"] >= cand_row["line_down"]):
                            ln_text = cand_row["line_text"]
                            overlap_area = 1.0
                            lft = cand_row["line_left"]
                            rgt = cand_row["line_right"]
                        elif (row["line_top"] > cand_row["line_top"]) and (row["line_down"] > cand_row["line_down"]):
                            range_ht = row["line_down"] - cand_row["line_top"]
                            overlap_ht = cand_row["line_down"] - row["line_top"]
                            overlap_area = overlap_ht / range_ht
                            ln_text = cand_row["line_text"]
                            lft = cand_row["line_left"]
                            rgt = cand_row["line_right"]
                            # if overlap_area >= 0.3:
                            #     ln_text = cand_row["line_text"]
                        elif (row["line_top"] < cand_row["line_top"]) and (row["line_down"] < cand_row["line_down"]):
                            range_ht = cand_row["line_down"] - row["line_top"]
                            overlap_ht = row["line_down"] - cand_row["line_top"]
                            overlap_area = overlap_ht / range_ht
                            ln_text = cand_row["line_text"]
                            lft = cand_row["line_left"]
                            rgt = cand_row["line_right"]
                            # if overlap_area >= 0.3:
                            #     ln_text = cand_row["line_text"]
                            #     if row["text"] == "PCSPL/2021/0008":
                            #         print(ln_text)
                    # if overlap_area > 0.0:
                    if (ln_text,overlap_area,lft,rgt) not in result:
                        result.append((ln_text,
                                       overlap_area,
                                       lft,rgt))
                    # print("intermediate result",result)

                #Take only one left ngbr if there are two overlapping neighbors.
                #This can happen especially when there are some left regions
                #that are actually belongs to the top or below line but because
                #document is folded, the alignment won't be proper
                filtered = [r for r in result if r[1] < 1.0]
                # print("filtered result", filtered)
                del_rows = []
                for r in filtered:
                    for g in filtered:
                        if g != r:
                            g_lft = g[2]
                            g_rgt = g[3]
                            r_lft = r[2]
                            r_rgt = r[3]
                            g_area = g[1]
                            r_area = r[1]
                            if not((g_rgt < r_lft) or (g_lft > r_rgt)):
                                if g_area > r_area:
                                    del_rows.append(r)
                                else:
                                    del_rows.append(g)
                # print("del_rows", del_rows)
                result = [r[0] for r in result if r not in del_rows]
                # print("final", result)
            # print("Left Ngbrs for loop",time.time() - t)

            results.append(result)
        df["left_processed_ngbr"] = results
        # DF["left_processed_ngbr"] = results
        return df
    except:
        print("findLfNgbrs",traceback.print_exc())
        return df_copy

def rankList(l):
    try:
        d = {j:[] for i,j in enumerate(l)}
        for i,j in enumerate(l):
            d[j].append(i)
        s = list(sorted(set(l),
                        reverse = True))
        c = len(s)
        ranks = list(range(1,c+1))
        r = l.copy()
    
        for i,j in enumerate(s):
            rank = ranks[i]
            indices = d[j]
            for ind in indices:
                r[ind] = rank
        return r
    except:
        print("rankList", traceback.print_exc())
        return None

@util.timing
def fuzzyMatchNgbrs(df,orientation,subject):

    #Jul 11 2022 - get the new label keywords for finding weighted average of fuzzy scores
    # hdr_keys_vals = labelKeywords_nonToken
    hdr_keys_vals = labelKeywords_nonToken_new
    #Jul 11 2022 - get the new label keywords for finding weighted average of fuzzy scores

    df["processed_text_" + orientation] = ""
    for hdr_key in hdr_keys_vals:
        df["fz_" + hdr_key + "_" + orientation] = 0.0
        df["fz_" + hdr_key + "_" + orientation + "_rank"] = 0

    df_copy = df.copy(deep = True)

    try:

        df.sort_values(["page_num",
                        "line_num",
                        "word_num"],
                       ascending=[True,True,True],
                       inplace = True)

        results = {}
        for hdr_key in hdr_keys_vals:
            results["fz_" + hdr_key + "_" + orientation] = []
            results["fz_" + hdr_key + "_" + orientation + "_rank"] = []

        results["processed_text_" + orientation] = []
        overallCount = 0
        #May 13 2022 - using itertuples instead of iterrows
        for ind_row, row in df.iterrows():
        # for row in df.itertuples():
        #May 13 2022 - using itertuples instead of iterrows
            resultCount = 0
            #May 13 2022 - slicing done using . operator
            texts = row[subject]
            # texts = row.subject

            #May 13 2022 - slicing done using . operator
            # print_true = False
            if isinstance(texts,str):
                texts = ast.literal_eval(texts)
            text_to_compare = ""
            word_seq_len = 0
            max_score = 0.0
            prev_tmp_result = {}
            tmp_result = {}
            for hdr_key in hdr_keys_vals:
                tmp_result["fz_" + hdr_key + "_" + orientation] = 0.0
                tmp_result["fz_" + hdr_key + "_" + orientation + "_rank"] = 0

            results["processed_text_" + orientation].append(texts)

            if len(texts) <= 0:
                for hdr_key in hdr_keys_vals:
                    results["fz_" + hdr_key + "_" + orientation].append(0.0)
                    results["fz_" + hdr_key + "_" + orientation + "_rank"].append(0)

                resultCount += 1
                overallCount += 1
                continue
            for text_ind,text in enumerate(texts):
                prev_tmp_result = tmp_result.copy()
                #Remove punctuations from string to compare
                mod_text = "".join([txt if (txt.isalpha() or txt == "#") else " "
                                    for txt in text
                                    if txt.isalpha()
                                    or txt in string.punctuation
                                    or txt == " "])
                #Standardize the string to lowercase for comparison. Remove space in the end
                mod_text = mod_text.lower().strip()
                #Replace # with the word number
                mod_text = mod_text.replace("#","number")
                #Remove extra spaces in the middle to 1 string
                mod_text = re.sub(' +',' ',mod_text)

                text_to_compare = mod_text + " " + text_to_compare
                text_to_compare = text_to_compare.strip()
                text_split = text_to_compare.split(" ")
                #Remove extra whitespaces
                text_to_compare = " ".join(text for text in text_split if text != "")
                len_words = len(text_to_compare.split(" "))
                if text_to_compare == "":
                    if text_ind < len(texts) - 1:
                        continue
                    else:
                        for hdr_key_ind,hdr_key in enumerate(hdr_keys_vals):
                            # if "igst" in text_to_compare:
                            #     print(prev_tmp_result["fz_" + hdr_key + "_" + orientation],
                            #           tmp_result["fz_" + hdr_key + "_" + orientation])
                            results["fz_" + hdr_key + "_" + orientation].append(
                                prev_tmp_result["fz_" + hdr_key + "_" + orientation])
                            results["fz_" + hdr_key + "_" + orientation + "_rank"].append(-1)
                        l = []
                        for hdr_key_ind,hdr_key in enumerate(hdr_keys_vals):
                            l.append(results["fz_" + hdr_key + "_" + orientation][-1])
                        ranked = rankList(l)
                        # if "igst" in text_to_compare:
                        #     print("text_ind < len(texts)",l,ranked)
                        for hdr_key_ind,hdr_key in enumerate(hdr_keys_vals):
                            results["fz_" + hdr_key + "_" + orientation + "_rank"][-1] = ranked[hdr_key_ind]
                        break
                if len_words > 0:
                    word_seq_len += len_words
                elif word_seq_len > 0:
                    for hdr_key_ind,hdr_key in enumerate(hdr_keys_vals):
                        # if "igst" in text_to_compare and hdr_key == "lblIGSTAmount":
                        #     print("word seq > 0",
                        #           prev_tmp_result["fz_" + hdr_key + "_" + orientation],
                        #           tmp_result["fz_" + hdr_key + "_" + orientation])
                        results["fz_" + hdr_key + "_" + orientation].append(
                            prev_tmp_result["fz_" + hdr_key + "_" + orientation])
                        results["fz_" + hdr_key + "_" + orientation + "_rank"].append(0)
                    l = []
                    for hdr_key_ind,hdr_key in enumerate(hdr_keys_vals):
                        l.append(results["fz_" + hdr_key + "_" + orientation][-1])
                    ranked = rankList(l)
                    # if "igst" in text_to_compare:
                    #     print("last word seq len > 0",l,ranked)
                    for hdr_key_ind,hdr_key in enumerate(hdr_keys_vals):
                        results["fz_" + hdr_key + "_" + orientation + "_rank"][-1] = ranked[hdr_key_ind]

                    resultCount += 1
                    overallCount += 1
                    break
                else:
                    if text_ind < len(texts) - 1:
                        continue
                    else:
                        for hdr_key in hdr_keys_vals:
                            results["fz_" + hdr_key + "_" + orientation].append(0.0)
                            results["fz_" + hdr_key + "_" + orientation+"_rank"].append(0)
                        resultCount += 1
                        overallCount += 1
                        break

                if word_seq_len > 4:
                    for hdr_key_ind,hdr_key in enumerate(hdr_keys_vals):
                        # if "igst" in text_to_compare:
                        #     print(prev_tmp_result["fz_" + hdr_key + "_" + orientation],
                        #           tmp_result["fz_" + hdr_key + "_" + orientation])
                        results["fz_" + hdr_key + "_" + orientation].append(
                            tmp_result["fz_" + hdr_key + "_" + orientation])
                        results["fz_" + hdr_key + "_" + orientation + "_rank"].append(0)
                    l = []
                    for hdr_key_ind,hdr_key in enumerate(hdr_keys_vals):
                        l.append(results["fz_" + hdr_key + "_" + orientation][-1])
                    ranked = rankList(l)
                    # if "igst" in text_to_compare:
                    #     print("last - word seq len > 4",l,ranked)
                    for hdr_key_ind,hdr_key in enumerate(hdr_keys_vals):
                        results["fz_" + hdr_key + "_" + orientation + "_rank"][-1] = ranked[hdr_key_ind]

                    resultCount += 1
                    overallCount += 1
                    break

                #Calculate score for each header keywords
                scores = []
                tmp_max_score = 0.0
                for key in hdr_keys_vals:
                    vals = hdr_keys_vals[key]
                    #Jul 11 2022 - Use fuzzyHdrText function to find weighted fuzzy score
                    # processed_text = [rp_utils.default_process(val) for val in vals]
                    processed_text = [rp_utils.default_process(str(val)) for val in vals]
                    #Jul 11 2022 - Use fuzzyHdrText function to find weighted fuzzy score
                    #Jun 16 2022 - changed ratio to WRatio
                    # match = rp_fz.extractOne(text_to_compare,
                    #                          processed_text,
                    #                          processor=None,
                    #                          scorer = rp_fuzz.ratio,
                    #                          score_cutoff = 0.01)
                    #Jul 11 2022 - Use fuzzyHdrText function to find weighted fuzzy score
                    # match = rp_fz.extractOne(text_to_compare,
                    #                          processed_text,
                    #                          processor = None,
                    #                          scorer = rp_fuzz.WRatio,
                    #                          score_cutoff = 0.01)
                    if isinstance(vals[0],str):
                        match = fz_match_hdrTxt(text_to_compare, processed_text)
                    else:
                        match = fz_match_hdrTxt(text_to_compare, vals)
                    if row.text == "248.37" and not isinstance(vals[0],str) and key in ['lblCGSTAmount1']:
                        print("matches for tax",
                              match,
                              text_to_compare,
                              vals)
                    #Jul 11 2022 - Use fuzzyHdrText function to find weighted fuzzy score
                    #Jun 16 2022 - changed ratio to WRatio
                    # if print_true:
                    #     print("Match",
                    #           match is None,
                    #           match,
                    #           text_to_compare,
                    #           processed_text)
                    # print("Match of neighbour fuzzy",match)
                    tmp_score = 0
                    if match:
                        #Jul 11 2022 - Use fuzzyHdrText function to find weighted fuzzy score
                        # tmp_score = match[1] / 100
                        tmp_score = match
                        #Jul 11 2022 - Use fuzzyHdrText function to find weighted fuzzy score
                        scores.append((key,tmp_score))
                        if tmp_score > tmp_max_score:
                            tmp_max_score = tmp_score
                    if "fz_" + key + "_" + orientation in tmp_result.keys():
                        prev_result = tmp_result["fz_" + key + "_" + orientation]
                        # if "igst" in text_to_compare and key == "lblIGSTAmount":
                        #     print("Prev and now", prev_result, tmp_score)

                        if prev_result < tmp_score:
                            tmp_result["fz_" + key + "_" + orientation] = tmp_score
                    else:
                        tmp_result["fz_" + key + "_" + orientation] = tmp_score

                    # if "cgst" in text_to_compare and key == "lblCGSTAmount1" and orientation.lower() == "above":
                    #     if "fz_" + key + "_" + orientation in tmp_result.keys():
                    #         print("Temp result before assigning",
                    #               tmp_result["fz_" + key + "_" + orientation],
                    #               tmp_score,
                    #               text,
                    #               text_to_compare,
                    #               key)

                    # if "igst" in text_to_compare:
                    #     print(prev_tmp_result["fz_" + hdr_key + "_" + orientation],
                    #           tmp_result["fz_" + hdr_key + "_" + orientation])
                    # if "igst" in mod_text:
                    #     print(processed_text,text_to_compare,
                    #           text,mod_text,tmp_score,max_score,
                    #           tmp_max_score)
                # if row.text == "248.37" and row.line_num == 127:
                #     print("scores tmp_result",
                #           tmp_max_score,
                #           max_score,
                #           tmp_result["fz_lblCGSTAmount1_Above"])
                if tmp_max_score > max_score:
                    max_score = tmp_max_score
                    if text_ind < len(texts) - 1:
                        continue
                for hdr_key_ind, hdr_key in enumerate(hdr_keys_vals):
                    results["fz_" + hdr_key + "_" + orientation].append(
                        tmp_result["fz_" + hdr_key + "_" + orientation])
                    results["fz_" + hdr_key + "_" + orientation + "_rank"].append(0)
                    # if "igst" in text_to_compare and hdr_key == "lblIGSTAmount":
                    #     print("normal",
                    #           results["fz_" + hdr_key + "_" + orientation])
                # if row.text == "248.37" and row.line_num == 127 and orientation.lower() == "above":
                #     print("scores results",
                #           tmp_max_score,
                #           max_score,
                #           results["fz_lblCGSTAmount1_Above"][-1])
                l = []
                for hdr_key_ind,hdr_key in enumerate(hdr_keys_vals):
                    l.append(results["fz_" + hdr_key + "_" + orientation][-1])
                # if row.text == "248.37" and row.line_num == 127 and orientation.lower() == "above":
                #     print("scores l",
                #           tmp_max_score,
                #           max_score,
                #           results["fz_lblCGSTAmount1_Above"][-1])
                ranked = rankList(l)
                # if "igst" in text_to_compare:
                #     print("last",l,ranked)
                for hdr_key_ind,hdr_key in enumerate(hdr_keys_vals):
                    results["fz_" + hdr_key + "_" + orientation + "_rank"][-1] = ranked[hdr_key_ind]

                resultCount += 1
                # if row.text == "248.37" and row.line_num == 127:
                #     print("scores",
                #           tmp_max_score,
                #           max_score,
                #           tmp_result["fz_lblCGSTAmount1_Above"])
                break
        # df_texts = list(df["text"])
        # IGSTs = results["fz_lblIGSTAmount_" + orientation]
        # d = pd.DataFrame({"text":df_texts,
        #      "IGST_fz_left":IGSTs})
        # d.to_csv(r"d:/sdsd.csv",index = False)
        # print("Dimension:",df.shape,len(results["fz_lblCGSTAmount1_Above"]))

        for hdr_key in hdr_keys_vals:
            #May 13 2022 - slice dataframe using . operator and not square bracket
            df["fz_" + hdr_key + "_" + orientation] = results["fz_" + hdr_key + "_" + orientation]
            df["fz_" + hdr_key + "_" + orientation + "_rank"] = results["fz_" + hdr_key + "_" + orientation + "_rank"]
            # fld = "fz_" + hdr_key + "_" + orientation
            # rank_fld = "fz_" + hdr_key + "_" + orientation + "rank"
            # df.fld = results["fz_" + hdr_key + "_" + orientation]
            # df.rank_fld = results["fz_" + hdr_key + "_" + orientation + "_rank"]
            #May 13 2022 - slice dataframe using . operator and not square bracket
        #May 13 2022 - slice dataframe using . operator and not square bracket
        df["processed_text_" + orientation] = results["processed_text_" + orientation]
        # df_ = df[["text","fz_lblIGSTAmount_" + orientation]]
        # df_.to_csv(r"d:/cddd.csv",index = False)
        # fld = "processed_text_" + orientation
        # df.fld = results["processed_text_" + orientation]
        #May 13 2022 - slice dataframe using . operator and not square bracket
        print("FuzzyMatchNgbrs done !")
        # df.to_csv(r"d:\dd.csv",index = False)
        return df
    except:
        print("FuzzyMatchNgbrs",
              traceback.print_exc())
        return df_copy

# In["Create a feature to check if the fuzzy score of the critical hdr fields is max"]:

@util.timing
def isHeaderFzMax(df):

    hdr_fields = {"is_ponumber_max":
                   [("fz_lblPoNumber_Above_rank",
                     "fz_lblPoNumber_Above"),
                    ("fz_lblPoNumber_Left_rank",
                     "fz_lblPoNumber_Left")],
                   "is_invoicenumber_max":
                       [("fz_lblInvoiceNumber_Above_rank",
                         "fz_lblInvoiceNumber_Above"),
                        ("fz_lblInvoiceNumber_Left_rank",
                         "fz_lblInvoiceNumber_Left")],
                       "is_invoicedate_max":
                           [("fz_lblInvoicedate_Above_rank",
                             "fz_lblInvoicedate_Above"),
                            ("fz_lblInvoicedate_Left_rank",
                             "fz_lblInvoicedate_Left")],
                           "is_cgstamount_max":
                               [("fz_lblCGSTAmount_Above_rank",
                                 "fz_lblCGSTAmount_Above"),
                                ("fz_lblCGSTAmount_Left_rank",
                                 "fz_lblCGSTAmount_Left")],
                              "is_sgstamount_max":
                                  [("fz_lblSGSTAmount_Above_rank",
                                    "fz_lblSGSTAmount_Above"),
                                   ("fz_lblSGSTAmount_Left_rank",
                                    "fz_lblSGSTAmount_Left")],
                                  "is_igstamount_max":
                                      [("fz_lblIGSTAmount_Above_rank",
                                        "fz_lblIGSTAmount_Above"),
                                       ("fz_lblIGSTAmount_Left_rank",
                                        "fz_lblIGSTAmount_Left")],
                                      "is_totalamount_max":
                                          [("fz_lblTotalAmount_Above_rank",
                                            "fz_lblTotalAmount_Above"),
                                           ("fz_lblTotalAmount_Left_rank",
                                            "fz_lblTotalAmount_Left")]}

    for hdr_field in hdr_fields:
        df[hdr_field] = 0
    df_copy = df.copy(deep = True)

    try:
        for hdr_field in hdr_fields:
            ftr1_1 = hdr_fields[hdr_field][0][0]
            ftr1_2 = hdr_fields[hdr_field][0][1]
            ftr2_1 = hdr_fields[hdr_field][1][0]
            ftr2_2 = hdr_fields[hdr_field][1][0]
            df[hdr_field] = ((df[ftr1_1] == 1) & 
                             (df[ftr1_2] > 0.0)) | ((df[ftr2_1] == 1) & 
                                                    (df[ftr2_2] > 0.0))
            df[hdr_field] = df[hdr_field].astype(int)

        return df
    except:
        print("isHeaderMax",
              traceback.print_exc())
        return df_copy

@util.timing
def isLImax(df):

    # max_pairs = {"item_code_max":('is_item_code','is_item_code1'),
    #              "item_desc_max":('is_item_desc','is_item_desc'),
    #              "item_val_max":('is_item_val','is_item_val1'),
    #              "item_unit_price_max":('is_unit_price','is_unit_price1'),
    #              "item_uom_max":('is_uom','is_uom1'),
    #              "item_qty_max":('is_qty','is_qty1'),
    #              "item_hsn_max":('is_hsn_key','is_hsn_key1'),
    #              "item_cgst_max":('is_cgst','is_cgst1'),
    #              "item_sgst_max":('is_sgst','is_sgst1'),
    #              "item_igst_max":('is_igst','is_igst1'),
    #              "item_disc_max":('is_disc','is_disc1')
    #              }

    # li_fields = {
    #         "is_item_code_max":["item_code_max",
    #                             "item_code_rank"],
    #         "is_item_desc_max":["item_desc_max",
    #                             "item_desc_rank"],
    #         "is_item_val_max":["item_val_max",
    #                            "item_val_rank"],
    #         "is_item_unit_price_max":["item_unit_price_max",
    #                                   "item_unit_price_rank"],
    #         "is_item_uom_max":["item_uom_max",
    #                            "item_uom_rank"],
    #         "is_item_qty_max":["item_qty_max",
    #                            "item_qty_rank"],
    #         "is_item_hsn_max":["item_hsn_max",
    #                            "item_hsn_rank"],
    #         "is_item_cgst_max":["item_cgst_max",
    #                             "item_cgst_rank"],
    #         "is_item_sgst_max":["item_sgst_max",
    #                             "item_sgst_rank"],
    #         "is_item_igst_max":["item_igst_max",
    #                             "item_igst_rank"],
    #         "is_item_disc_max":["item_disc_max",
    #                             "item_disc_rank"]
    #         }
    li_fields = {
            "is_item_code_max":["is_item_code1",
                                "item_code_rank"],
            "is_item_desc_max":["is_item_desc",
                                "item_desc_rank"],
            "is_item_val_max":["is_item_val1",
                               "item_val_rank"],
            "is_item_unit_price_max":["is_unit_price1",
                                      "item_unit_price_rank"],
            "is_item_uom_max":["is_uom1",
                               "item_uom_rank"],
            "is_item_qty_max":["is_qty1",
                               "item_qty_rank"],
            "is_item_hsn_max":["is_hsn_key1",
                               "item_hsn_rank"],
            "is_item_cgst_max":["is_cgst1",
                                "item_cgst_rank"],
            "is_item_sgst_max":["is_sgst1",
                                "item_sgst_rank"],
            "is_item_igst_max":["is_igst1",
                                "item_igst_rank"],
            "is_item_disc_max":["is_disc1",
                                "item_disc_rank"]
            }
    for li_field in li_fields:
        df[li_field] = 0
    df_copy = df.copy(deep = True)

    try:
        for li_field in li_fields:
            ftr1 = li_fields[li_field][0]
            ftr2 = li_fields[li_field][1]
            df[li_field] = (df[ftr1] > 0.0) & (df[ftr2] == 1)
            df[li_field] = df[li_field].astype(int)
        return df
    except:
        print("isLImax",
              traceback.print_exc())
        return df_copy

    
# In["Rank Line Item Features"]: Find rank of LI features
@util.timing
def upd_item_rank_fields(df):

    LI_RANK_FIELDS = ["item_code_rank",
                      "item_desc_rank",
                      "item_val_rank",
                      "item_unit_price_rank",
                      "item_uom_rank",
                      "item_qty_rank",
                      "item_hsn_rank",
                      "item_cgst_rank",
                      "item_sgst_rank",
                      "item_igst_rank",
                      "item_disc_rank"]

    df[LI_RANK_FIELDS] = 0
    df_copy = df.copy(deep = True)
    keys = ["is_item_code1","is_item_desc",
            "is_item_val1","is_unit_price1",
            "is_uom1","is_qty1","is_hsn_key1",
            "is_cgst1","is_sgst1","is_igst1",
            "is_disc1"]

    try:
        df[LI_RANK_FIELDS] = df[keys].rank(axis = 1,
                                           method = "dense",
                                           ascending = False).astype(int)
        return df
    except:
        print("upd_item_rank_fields",
              traceback.print_exc())
        return df_copy

# In[Find Regions]: Find regions

def findRegions(df):

    def findLineNums(df):

        df["region"] = 0
        # regionNo = 0

        df.sort_values(["page_num",
                        "line_num",
                        "word_num"],
                        ascending = [True,True,True],
                        inplace = True)

        lines = []
        page_line_nums = []

        for ind,row in df.iterrows():
            page_num = row["page_num"]
            line_no = row["line_num"]
            line_down = row["line_down"]
            if (page_num,line_no) in page_line_nums:
                continue
            else:
                page_line_nums.append((page_num,line_no))

            # rgnNo = row["region"]

            down_line_len = row["downLineLen"] if row["downLineLen"] > 0.1 else 0

            if down_line_len > 0:
                line_down_x1 = None
                if isinstance(row["line_down_x1"],str):
                    line_down_x1 = ast.literal_eval(row["line_down_x1"])
                else:
                    line_down_x1 = row["line_down_x1"]
                down_line_y = round(line_down_x1[1],2)
            else:
                down_line_y = -1
            top_line_len = row["topLineLen"] if row["topLineLen"] > 0.1 else 0

            if top_line_len > 0:
                line_top_x1 = None
                if isinstance(row["line_top_x1"],str):
                    line_top_x1 = ast.literal_eval(row["line_top_x1"])
                else:
                    line_top_x1 = row["line_top_x1"]
                top_line_y = round(line_top_x1[1],2)
            else:
                top_line_y = -1

            line = []
            line.append(str(page_num) + "," + str(line_no))

            candidates = df[(df["page_num"] == page_num) &
                            (df["line_top"] - line_down <= 0.005) &
                            (df["line_top"] > row["line_top"]) &
                            (df["line_left"] < row["line_right"]) &
                            (df["line_right"] > row["line_left"])
                            ]
            s = set()
            if candidates.shape[0] > 0:
                for cand_row_ind, cand_row in candidates.iterrows():
                    down_line_len_cand = cand_row["downLineLen"] if cand_row["downLineLen"] > 0.1 else 0
                    if down_line_len_cand > 0:
                        cand_line_down_x1 = None
                        if isinstance(cand_row["line_down_x1"],str):
                            cand_line_down_x1 = ast.literal_eval(cand_row["line_down_x1"])
                        else:
                            cand_line_down_x1 = cand_row["line_down_x1"]
                        down_line_y_cand = round(cand_line_down_x1[1],2)
                    else:
                        down_line_y_cand = -1
                    top_line_len_cand = cand_row["topLineLen"] if cand_row["topLineLen"] > 0.1 else 0
                    if top_line_len_cand > 0:
                        cand_line_top_x1 = None
                        if isinstance(cand_row["line_down_x1"],str):
                            cand_line_top_x1 = ast.literal_eval(cand_row["line_top_x1"])
                        else:
                            cand_line_top_x1 = cand_row["line_top_x1"]
                        top_line_y_cand = round(cand_line_top_x1[1],2)
                    else:
                        top_line_y_cand = -1
                if (down_line_y == down_line_y_cand):
                    s.add(cand_row["line_num"])

                l = list(s)
                for f in l:
                    line.append(str(page_num) + "," + str(f))

            lines.append(line)

        return lines

    df["region"] = 0
    # regionNo = 0
    df_copy = df.copy(deep = True)

    try:

        #Find neighbour lines that are potential to become regions
        line_nums = findLineNums(df)
        
        #Create a graph of lines
        region_lines = util.connect_lines(line_nums)
    
        #update df with region numbers on the appropriate lines
        for region_ind,region in enumerate(region_lines):
            region_num = region_ind + 1
            for line in region:
                page_num_str = line.split(",")[0]
                line_num_str = line.split(",")[1]

                page_num = int(ast.literal_eval(page_num_str))
                line_num = int(ast.literal_eval(line_num_str))
                df.loc[(df["page_num"] == int(page_num)) & 
                       (df["line_num"] == int(line_num))
                       ,["region"]] = [region_num]
    
        return df
    except:
        print("Find Regions", traceback.print_exc())
        return df_copy

# In[]: Find Address, Company Name and identify address regions
# Word level matching
def checkMatch(s, l):
    #print("str : ", s)
    #print("l : ", l)
    for words in l:
        #print(" words", words.lower(), "text :", text.lower())
        if words.lower() in s.lower():
            print("prasent :", words)
            return 1
        else:
            #print(" not match ", words)
            continue 
    return 0

# #Charactor level matchigg
# def regex_match(s,ptn):
#     try:
#         m = regex.match('.*(' + ptn.lower() + '){e<=1}',
#                         s.lower())
#         return m is not None
#     except:
#         print(s)
#         return False

# def findMatches(s, l):
#     for ptn in l:
#         is_match = regex_match(s, ptn)
#         # print(ptn,s,is_match)
#         if is_match:
#             return is_match
#         else:
#             continue
#     return False

def noOfWordsLinesInRegions(df):
    df_groups = df.groupby(["page_num",
                            "region"])[
                                "token_id"].agg("count").reset_index()
    df = df.merge(df_groups,
                  on = ["page_num","region"],
                  how = "inner")
    df.rename(columns={"token_id_y":"words_on_region",
                       "token_id_x":"token_id"},
              inplace=True)

    df_groups = df.groupby(["page_num",
                            "line_num",
                            "region"])[
                                "token_id"].agg("count").reset_index()
    df = df.merge(df_groups,
                  on = ["page_num",
                        "line_num",
                        "region"],
                  how = "inner")
    df.rename(columns={"token_id_y":"lines_on_region",
                       "token_id_x":"token_id"},
              inplace=True)

    return df

def is_address_region(df):
    df["company_or_address"] = df["is_company_name"] + df["is_address"]
    df_groups = df.groupby(["page_num",
                            "region"])[
                                "company_or_address"].agg(
                                    "sum").reset_index()
    df_groups["company_or_address"] = df_groups["company_or_address"] > 0
    df = df.merge(df_groups,
                  on = ["page_num","region"],
                  how = "inner")
    df.rename(columns={"company_or_address_y":"is_address_region"},
              inplace=True)
    df["is_address_region"] = df["is_address_region"].fillna(0).astype(int)
    return df



# In[9]: Declare Functions that calls all other functions

@util.timing
def get_ocr_df(ocr_path,imgpath):
    """
    Function needs OCR and Label blob paths
    """
    df_ocr = read_ocr_json_file(ocr_path,imgpath)

    return df_ocr

@util.timing
def process_df_for_feature_extraction(df):

    df = df.loc[df['conf'] != '-1']
    df = df.reset_index(drop=True)
    # df['conf'] = df['conf'].astype(float)

    # Clean Amount tokens in OCR DF
    #Jul 13, 2022 - add unmodified original text in a separate column to add ay validation
    df["original_text"] = df["text"]
    #Jul 13, 2022 - add unmodified original text in a separate column to add ay validation
    df = correctAmountLineTokens(df)
    df = correctAmountTokens(df)

    df = correctOcrLineForNonEngTkn(df) # OCR line correction
    df = token_distances(df)
    df = position_binning(df)

    df = read_lines_from_table_new(df) # Updated new lines of table

    #Extract all token specific or text features
    #It also extracts nearby words and their token specific features
    df = extract_text_features(df)

    #Add isLabel features for the neighboring words
    #Added on Jul 15, 2020
    df = neighborLabels(df)

    #Add isProbable fields for amounts
    df = extract_amount_features(df)

    #Create a new feature isPONumber - Aug 13
    df['is_PO_number'] = 0
    df.loc[df['wordshape'].isin(po_shapes), 'is_PO_number'] = 1

    #Populate GST features
    df = extract_GST_features_label_overlap(df)
    df = populate_max_probableGST_amount(df)

    #June 15, 2021 code
    df = extract_max_amount_features(df)

    #Call new code for finding neighbours
    df = findLfNgbrs(df)
    df = findAbNgbrs(df)

    #Do a fuzzy match on the new neighbours and rank the left & above fz scores
    print("Calling left proceessed ngbr")
    df = fuzzyMatchNgbrs(df,
                         orientation = "Left",
                         subject = "left_processed_ngbr")
    print("Calling right proceessed ngbr")
    df = fuzzyMatchNgbrs(df,
                         orientation = "Above",
                         subject = "above_processed_ngbr")

    #Also find the max fuzzy score
    df = isHeaderFzMax(df)

    #findRegions of text
    df = findRegions(df)

    # df = addLineItemNeighbours(df)
    df = addLineItemNeighbours_new(df)

    #Add Line Item Features and neighbors
    # df = addLineItemFeatures(df)
    df = addLineItemFeatures_New(df)
    
    #Added code to correct line_row assignent if present before header row
    df = correctLineRows(df)
    #May 28, 2022 - Added code to adjust description if their column header is not overlapping
    df = correctLineRowsFzScore(df)

    #Added on Feb-13-2022 - Find Rank of LI Item Fields and find max of fz
    # df = util.reduce_mem_usage(df)
    df = upd_item_rank_fields(df)
    df = isLImax(df)
    #isProbable_itemvalue feature is populated here
    # df = potential_itemValue(df)
    #isProbable_itemvalue feature -make it zero - Apr 10, 2022
    df = potential_itemValue_zero(df)

    # Code added to populate GST features for non-lineitem tokens
    df.loc[df['line_row'] > 0, 'probable_totalAmount'] = 0
    df.loc[df['line_row'] > 0, 'probable_subTotal'] = 0
    df.loc[df['line_row'] > 0, 'probable_gstAmount_range'] = 0
    df.loc[df['line_row'] > 0, 'probable_subtotal_range'] = 0
    df.loc[df['line_row'] > 0, 'probable_tcsAmount_range'] = 0
    df.loc[df['line_row'] > 0, 'PROBABLE_GST_AMOUNT_SLAB'] = 0
    df.loc[df['line_row'] > 0, 'PROBABLE_SUBTOTAL_SLAB'] = 0

    # df = util.reduce_mem_usage(df)
    # df = makeNonLIftrsZero(df)
    #Apr 11, 2022 - better way to update dataframe
    df = makeNonLIftrsZero_new(df)

    #One hot encoding of entities, apply NaNs with 0 for all integer & float
    df = fillNullsWithDefaults(df)

    return df

@util.timing
def fuzzyRelatedProcess(df):

    df_copy = df.copy(deep = True)
    try:
        #Jul 04 2022 - process for Fuzzy here
        #Call new code for finding neighbours
        
        df = findLfNgbrs(df)
        df = findAbNgbrs(df)
    
        #Do a fuzzy match on the new neighbours and rank the left & above fz scores
        print("Calling left proceessed ngbr")
        df = fuzzyMatchNgbrs(df,
                             orientation = "Left",
                             subject = "left_processed_ngbr")
        print("Calling Above proceessed ngbr inside process df ")
        df = fuzzyMatchNgbrs(df,
                             orientation = "Above",
                             subject = "above_processed_ngbr")
    
        #Also find the max fuzzy score
        df = isHeaderFzMax(df)
    
    
        # getting bill to ship to feature
        df  = get_bill_to_ship_to_details(df,
                                          INDIA_ZIP_CODE_RANGE,
                                          billToNames_kws,
                                          shipToNames_kws) 
        print("process df features done")
        return df
        #Jul 04 2022 - process for Fuzzy here
    except:
        print("fuzzyRelatedProcess",
              traceback.print_exc())
        return df_copy


def process_df_features(df):

    try:
        df = df.loc[df['conf'] != '-1']
        df = df.reset_index(drop=True)

        # Clean Amount tokens in OCR DF
        #Jul 13, 2022 - add unmodified original text in a separate column to add ay validation
        df["original_text"] = df["text"]
        #Jul 13, 2022 - add unmodified original text in a separate column to add ay validation
        df = correctAmountLineTokens(df)
        df = correctAmountTokens(df)

        df = correctOcrLineForNonEngTkn(df) # OCR line correction
        df = token_distances(df)
        df = position_binning(df)

        df = read_lines_from_table_new(df) # Updated new lines of table

        #Extract all token specific or text features
        #It also extracts nearby words and their token specific features
        df = extract_text_features(df)

        #Add isLabel features for the neighboring words
        #Added on Jul 15, 2020
        df = neighborLabels(df)

        #Add isProbable fields for amounts
        df = extract_amount_features(df)

        #Create a new feature isPONumber - Aug 13
        df['is_PO_number'] = 0
        df.loc[df['wordshape'].isin(po_shapes), 'is_PO_number'] = 1

        #Populate GST features
        df = extract_GST_features_label_overlap(df)
        #df = populate_max_probableGST_amount(df)

        #June 15, 2021 code
        #df = extract_max_amount_features(df)

        #findRegions of text
        df = findRegions(df)

        #Call new code for finding neighbours
        # df = findLfNgbrs(df)
        # df = findAbNgbrs(df)

        # #Do a fuzzy match on the new neighbours and rank the left & above fz scores
        # print("Calling left proceessed ngbr")
        # df = fuzzyMatchNgbrs(df,
        #                      orientation = "Left",
        #                      subject = "left_processed_ngbr")
        # print("Calling Above proceessed ngbr inside process df ")
        # df = fuzzyMatchNgbrs(df,
        #                      orientation = "Above",
        #                      subject = "above_processed_ngbr")

        # #Also find the max fuzzy score
        # df = isHeaderFzMax(df)


        # # getting bill to ship to feature
        # df  = get_bill_to_ship_to_details(df,
        #                                   INDIA_ZIP_CODE_RANGE,
        #                                   billToNames_kws,
        #                                   shipToNames_kws) 
        # print("process df features done")

        return df
    except:
        print("process_df_features exception",
              traceback.print_exc())
        return None

def process_df_lines(df):

    try:
        df = addLineItemNeighbours_new(df)
    
        #Add Line Item Features and neighbors
        # df = addLineItemFeatures(df)
        df = addLineItemFeatures_New(df)
        #Added code to correct line_row assignent if present before header row
        df = correctLineRows(df)
        #May 28, 2022 - Added code to adjust description if their column header is not overlapping
        df = correctLineRowsFzScore(df)

        #Added on Feb-13-2022 - Find Rank of LI Item Fields and find max of fz
        df = util.reduce_mem_usage(df)
        df = upd_item_rank_fields(df)
        df = isLImax(df)
        #isProbable_itemvalue feature is populated here
        df = potential_itemValue(df)

        # Code added to populate GST features for non-lineitem tokens
        df.loc[df['line_row'] > 0, 'probable_totalAmount'] = 0
        df.loc[df['line_row'] > 0, 'probable_subTotal'] = 0
        df.loc[df['line_row'] > 0, 'probable_gstAmount_range'] = 0
        df.loc[df['line_row'] > 0, 'probable_subtotal_range'] = 0
        df.loc[df['line_row'] > 0, 'probable_tcsAmount_range'] = 0
        df.loc[df['line_row'] > 0, 'PROBABLE_GST_AMOUNT_SLAB'] = 0
        df.loc[df['line_row'] > 0, 'PROBABLE_SUBTOTAL_SLAB'] = 0

        # df = util.reduce_mem_usage(df)
        #Jul 04 2022 - Don' call it here. Call it after fuzzy score generated
        # df = makeNonLIftrsZero(df)
        #Jul 04 2022 - Don' call it here. Call it after fuzzy score generated

        #One hot encoding of entities, apply NaNs with 0 for all integer & float
        # df = fillNullsWithDefaults(df)
        df = util.reduce_mem_usage(df)

        return df
    except:
        print("process_df_lines",
              traceback.print_exc())
        return None


@util.timing
def process_single_file(image_path,
                        ocr_path,
                        page_num,
                        container):
    """
    Function to read ocr.json files and extarct features
    """
    try:

        #Convert the OCR into a dataframe and forms the basic features
        image_filename = image_path.split("/")[-1]
        ocr_filename = ocr_path.split("/")[-1]
        ocrLocalPath = os.path.join(ROOT_FOLDER,
                                    ocr_filename)
        imgLocalPath = os.path.join(ROOT_FOLDER,
                                    image_filename)
        downloaded = [util.downloadFilesFromBlobStore(container,
                                                      blobAccount + "/" + image_path,
                                                      imgLocalPath),
                      util.downloadFilesFromBlobStore(container,
                                                      blobAccount + "/" + ocr_path,
                                                      ocrLocalPath)]
        if not all(downloaded):
            return None
        DF = get_ocr_df(ocrLocalPath,
                        imgLocalPath)

        # Clean Amount tokens in OCR DF
        # DF = correctAmountLineTokens(DF)
        # DF = correctAmountTokens(DF)

        # Call Image features by passing above DF
        DF = extract_image_features(DF,imgLocalPath) # Extract image features
        DF = correctImageFeatures(DF)
        DF = splitLineTextVLines(DF) # splt Line Text VLines

        # DF = correctOcrLineForNonEngTkn(DF) # OCR line correction
        # DF = token_distances(DF) 
        # DF = position_binning(DF)

        # DF = read_lines_from_table_new(DF) # Updated new lines of table

        #Extract all other features using the token text, image, etc.,
        DF["page_num"] = int(page_num)

        try:
            os.remove(ocrLocalPath)
            os.remove(imgLocalPath)
        except:
            pass

        return(DF)

    except:
        print("process single file:",
              traceback.print_exc())
        return None

def process_image_ocr(page):

    try:
        DF = process_single_file(page.get("image_path"),
                                 page.get("ocr_path"),
                                 page.get("document_page_number"),
                                 page.get("container"))
        return DF
    except:
        print("process_image_ocr",
              traceback.print_exc())
        return None

@util.timing
def makeTokenIdsUnq(df):
    base_tokenid = 10000
    df["token_id"] = df["token_id"] - base_tokenid
    df["token_id"] = (df["page_num"] * base_tokenid) + df["token_id"]
    return df
#Apr 08, 2022 code to make token IDs unique

    

@util.timing
def extract_features(docInput):

    from multiprocessing import Pool

    documentId = ""
    try:
        documentId = docInput.get("documentId")
        pages = docInput.get("page_details")
        client_folder = docInput["client_folder"]
        container = docInput["container"]
        # print("Pages...: ", pages)
        for page_no,page in enumerate(pages):
            page["container"] = container

        # t = time.time()
        # for page_no,page in enumerate(pages):
        #     DF = process_single_file(page.get("image_path"),
        #                               page.get("ocr_path"),
        #                               page.get("document_page_number"),
        #                               container)
        #     if DF is None:
        #         return None
        #     DF_All = DF_All.append(DF,
        #                             ignore_index = True)
        # # print(type(pages),pages)
        # print("process_image_ocr",time.time() - t)

        t = time.time()
        pool_size = min(len(pages),10)
        # try:
        #     set_start_method("spawn")
        # except:
        #     print("extract_features failed in set_start_method spawn")
        #     pass
        pool = Pool(pool_size)
        res = pool.map_async(process_image_ocr,
                              pages)
        pool.close()
        pool.join()
        DFs = res.get()
        pool.terminate()
        print("process_image_ocr",
              time.time() - t)
        # print("Dataframe type",type(DFs),DFs)
        # df = pd.DataFrame()
        # for DF in DFs:
        #     if DF is None:
        #         return None
        #     df = df.append(DF,ignore_index = True)
        # df.to_csv("d://ftrs_image.csv",index = False)
        t = time.time()
        # try:
        #     set_start_method("fork")
        # except:
        #     print("set start method back to fork")
        #     pass
        pool_size = min(len(DFs),10)
        pool = Pool(pool_size)
        res = pool.map_async(process_df_features,DFs)
        pool.close()
        pool.join()
        DFs = res.get()
        pool.terminate()
        print("process_df_features",
              time.time() - t)

        DF_All = pd.DataFrame()
        for DF in DFs:
            if DF is None:
                return None
            DF_All = DF_All.append(DF,
                                    ignore_index = True)

        DF_All = DF_All.sort_values(["page_num",
                                     "line_num",
                                     "word_num"])
        DF_All.reset_index(inplace = True)

        DF_All = makeTokenIdsUnq(DF_All)
        #May 06, 2022 - These functions should be acted on the entire dataframe and not page wise
        # DF_All = extract_max_amount_features(DF_All)
        DF_All = extract_max_amount_features_new(DF_All)

        DF_All = populate_max_probableGST_amount(DF_All)
        #May 06, 2022 - These functions should be acted on the entire dataframe and not page wise

        #Call feature engineering for all pages at once
        t = time.time()
        DF_All = process_df_lines(DF_All)
        DF_All = DF_All.sort_values(["page_num",
                                     "line_num",
                                     "word_num"])
        print("process_df_lines",time.time() - t)

        #Jul 04 2022 - process fuzzy related features here
        print("Is line_row present", "line_row" in DF_All.columns.values)
        DF_All = fuzzyRelatedProcess(DF_All)
        DF_All = makeNonLIftrsZero(DF_All)
        #Jul 04 2022 - process fuzzy related features here

        blob_ftr_file_name = documentId + "_ftrs.csv"
        blob_ftr_file_path = os.path.join(ROOT_FOLDER,
                                          client_folder,
                                          blob_ftr_file_name)
        if os.path.exists("/Users/Parmesh/Downloads"):
            DF_All.to_csv("/Users/Parmesh/Downloads/ftrs.csv",index = False)
        DF_All.to_csv(blob_ftr_file_path)

        uploaded, URI = util.uploadFilesToBlobStore(container,
                                                    blob_ftr_file_path)
        if not uploaded:
            return json.dumps({"status_code":500,
                               "error":"Error in uploading",
                               "documentId":documentId,
                               "URI":""})
        else:
            try:
                os.remove(blob_ftr_file_path)
            except:
                pass
            print("feature file uploaded to Blob")

        return json.dumps({"status_code":200,
                           "error":"",
                           "documentId":documentId,
                           "URI":URI})
    except:
        print("Feature For All Pages",traceback.print_exc())
        return json.dumps({"status_code":500,
                           "error":"Exception",
                           "documentId":documentId,
                           "URI":""})

if __name__ == "__main__":
    inp = {
           "documentId":"doc_1658728784118_baa99abb8b9",
           "client_folder":"",
           "container":"6bf8ef2f-4b83-4ef1-ac1f-af94d4d58df9",
           "page_details":
               [
                   {"image_path":"6bf8ef2f-4b83-4ef1-ac1f-af94d4d58df9/doc_1658728784118_baa99abb8b9-0-pre.tiff",
                    "ocr_path":"6bf8ef2f-4b83-4ef1-ac1f-af94d4d58df9/doc_1658728784118_baa99abb8b9-0-pre.tiff.ocr.json",
                     "document_page_number":"0"}#,
                     #{"image_path":"79c5de9a-2d6b-4cc6-b05f-3974217ea41b/08585452121727097137684559406CU27--1-1-pre.tiff",
                    #"ocr_path":"79c5de9a-2d6b-4cc6-b05f-3974217ea41b/08585452121727097137684559406CU27--1-1-pre.tiff.ocr.json",
                     #"document_page_number":"1"}
                ]
               }

    result = extract_features(inp)



