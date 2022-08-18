#!/usr/bin/env python
# coding: utf-8
from ast import literal_eval
from copy import deepcopy
from lib2to3.pgen2 import token

from pyparsing import WordStart
import config as config
import util as util
import json
import regex
from functools import reduce
import pandas as pd 
import traceback
import re
from rapidfuzz.process import extractOne
from rapidfuzz.fuzz import ratio


LIKeyWordsPath = config.getLIKeywordsPath()
INDIA_ZIP_CODE_RANGE = config.getZipCodeFilePAth()

with open(LIKeyWordsPath) as data:
    LIKeyWordData = json.load(data)
verticalThresh = LIKeyWordData['verticalThresh']
hdr_TokenJson = config.getlabelKeywordsNonTokenized()
with open(hdr_TokenJson) as f:
    hdr_tokens =  json.load(f)

companyNameList = hdr_tokens['companyNameList']
addressNameList = hdr_tokens["addressNameList"]
billToNames_kws = hdr_tokens["lblBillingNames"]
shipToNames_kws = hdr_tokens["lblShippingNames"]

############## Identtify bill to ship to name ##############
def checkMatch(s, l):
    #print("str : ", s)
    #print("l : ", l)
    try:
        for words in l:
            #print(" words", words.lower(), "text :", text.lower())
            if str(words).lower() in str(s).lower():
                #print("prasent :", words)
                return 1
            else:
                #print(" not match ", words)
                continue 
        return 0
    except: 
        print(traceback.print_exc())
        return 0
    

def address_match(text,words):
    try:
        text = remove_chars(str(text),['/',':','(',')','.',"'",","])
        #print("clean text ",text)
        text = text.split(' ')
        text = [x.lower() for x in text]
        words = [x.lower() for x in words]
        #print("line text words :",text)
        #print("address word :",words)
        for tkn in text:
            for wds in words:
                if tkn == wds:
                    return 1
        return 0
    except:
        print("Address match exception",traceback.print_exc())
        return 0


def containsCompanyName(df):
    df["is_company_name"] = 0
    df_copy = df.copy(deep = True)
    try:
        df["is_company_name"] = df["line_text"].apply(checkMatch,args = (companyNameList,))
        return df
    except:
        print(traceback.print_exc())
        return df_copy

def containsAddress(df):
    df["is_address"] = 0
    df_copy = df.copy(deep =True)
    try:
        df["is_address"] = df["line_text"].apply(address_match, args = (addressNameList,))
        return df
    except: 
        print(traceback.print_exc())
        return df_copy

# fuzzy score for for bill to label & ship to 
def get_fz_lbl_score(string, keyWords_list):
    try:
        string = remove_chars(str(string),['/',':','(',')','.',"'",","])
        # print("line text :",string)
        match_score = extractOne(string, keyWords_list, scorer=ratio)
        return match_score[1]
    except:
        print(traceback.print_exc())
        return 0

def fz_lbl_bill_to(df):
    df["fz_lbl_bill_to"] = 0
    df_copy = df.copy(deep =True)
    try:
        df["fz_lbl_bill_to"] = df["line_text"].apply(get_fz_lbl_score,args = (billToNames_kws,))
        return df
    except: 
        print(traceback.print_exc())
        return df_copy

def fz_lbl_ship_to(df):
    df["fz_lbl_ship_to"] = 0
    df_copy = df.copy(deep =True)
    try:
        df["fz_lbl_ship_to"] = df["line_text"].apply(get_fz_lbl_score,args = (shipToNames_kws,))
        return df
    except: 
        print(traceback.print_exc())
        return df_copy



def contains_bill_to_name(df):
    df["contains_bill_to_name"] = 0
    df_copy = df.copy(deep =True)
    try:
        df["contains_bill_to_name"] = df["line_text"].apply(checkMatch,args = (billToNames_kws,))
        return df
    except: 
        print(traceback.print_exc())
        return df_copy

def contains_ship_to_name(df):
    df["contains_ship_to_name"] = 0
    df_copy = df.copy(deep =True)
    try:
        df["contains_ship_to_name"] = df["line_text"].apply(checkMatch,args = (shipToNames_kws,))
        return df
    except: 
        print("contains ship to name exception :",traceback.print_exc())
        return df_copy


def noOfWordsLinesInRegions(df):
    df_copy = df.copy(deep = True)
    try:
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
    except: 
        print("No of words line in ragion exception :",traceback.print_exc())
        return df_copy

def is_address_region(df):
    df["company_or_address"] = 0
    df_copy = df.copy(deep = True)
    try:
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
    except : 
        print("company or address exception :",traceback.print_exc())
        return df_copy

# finding above / left processed lines#
# def find_left_above_lines(df):
#     noNeighbours = 1
#     cols = ["left","right", "top","bottom","line_text", "page_num","line_num"]
#     xCols = [x + "_x" for x in cols]
#     #print("x cols Name",xCols)
#     xCols_addl = xCols.copy()
#     xCols_addl.extend(list(range(noNeighbours)))

#     df1 = df[cols]
#     df2 = df1.copy(deep = True)

#     df_merge = df1.merge(df2, how = "cross")
#     df_x = df_merge[xCols]
#     df_x = df_x.drop_duplicates(keep = "first")
#     above_processed_line = ["above_processed_line"]
#     #above_processed_line = ["line_" + str(i) + "_above" for i in range(1,noNeighbours + 1)] #['above_processed_line']
#     print("above_processed_line :",above_processed_line)
#     df_above = df_merge[(df_merge["top_x"] > df_merge["bottom_y"]) &
#                   (df_merge["top_x"] <= df_merge["bottom_y"] + verticalThresh) &
#                   (df_merge["right_y"] >= df_merge["left_x"]) &
#                   (df_merge["left_y"] <= df_merge["right_x"]) &
#                   (df_merge["page_num_x"] == df_merge["page_num_y"])]

#     df_above = df_above.sort_values(["top_x","left_x","bottom_y","left_y"], ascending = [True,True,False,True])
#     df_above["grp1"] = df_above.groupby(by = xCols).cumcount()
#     df_above["grp2"] = df_above["grp1"] + 100
#     df_above = df_above[df_above["grp1"] < noNeighbours]
#     df_ab = df_above.pivot(index = xCols,columns = ['grp1','grp2'], values = ["line_text_y"]).reset_index()
#     #print(df_ab.columns.values)
#     df_ab.columns = cols + above_processed_line

#     #words_col_lf = ["line_" + str(i) + "_left" for i in range(1,noNeighbours + 1)]#["left_processed_line"]
#     words_col_lf = ["left_processed_line"]
#     print("left_processed_line :",words_col_lf)
#     df_left = df_merge[(df_merge["right_y"] < df_merge["left_x"]) &
#                   (df_merge["bottom_y"] >= df_merge["top_x"]) &
#                   (df_merge["top_y"] <= df_merge["bottom_x"]) &
#                   (df_merge["page_num_x"] == df_merge["page_num_y"])]
#     df_left = df_left.sort_values(["top_x","left_x","right_y"], ascending = [True,True,False])
#     df_left["grp1"] = df_left.groupby(by = xCols).cumcount()
#     df_left["grp2"] = df_left["grp1"] + 100
#     df_left["grp3"] = df_left["grp1"] + 200
#     df_left = df_left[df_left["grp1"] < noNeighbours]
#     df_lf = df_left.pivot(index = xCols,columns = ['grp1','grp2','grp3'], values = ["line_text_y"]).reset_index()
#     print("dfleft cols :",df_lf.columns)
#     df_lf.columns = cols + words_col_lf 
#     df_ngbrs = [df_ab,df_lf]
#     df_ngbr = reduce(lambda left,right: pd.merge(left, right, on = cols,how = 'outer'),df_ngbrs)

#     df = pd.merge(df, df_ngbr, on = cols, how = "outer")
#     return df


def is_bill_to_names(df,bill_to_Names_kws):
    df["is_bill_to_name"] =0
    df_copy = df.copy(deep =True)
    try:
        # print("no_items :",no_items)
        for idx, row in df.iterrows():
            ab_ngbr_lst = str(row["above_processed_ngbr"]) # literal_eval(row["above_processed_ngbr"])
            if row["is_company_name"]==1:
                # print("ab_ngbr_lst :",ab_ngbr_lst)
                if ab_ngbr_lst:
                    for item  in bill_to_Names_kws:                    
                        if item.lower() in ab_ngbr_lst.lower():
                            # print("line match : ",line,"at index :",idx )
                            # print("b4r upd",df["is_bill_to_name"][idx])
                            df.loc[[idx],["is_bill_to_name"]] = 1
                            # print("updted indx :",df["is_bill_to_name"][idx], idx)
                            #print("Breaked at inner loop")
                            break
        if df["is_bill_to_name"].sum()<=0:
            print(" Matching in left_processed_ngbr ")
            for idx, row in df.iterrows():
                lf_ngbr_lst = str(row["left_processed_ngbr"]) #literal_eval(row["left_processed_ngbr"])
                if row["is_company_name"]==1:
                    if lf_ngbr_lst:
                        for item  in bill_to_Names_kws:                    
                            if item.lower() in lf_ngbr_lst.lower():
                                # print("line match : ",line,"at index :",idx )
                                df.loc[[idx],["is_bill_to_name"]] = 1
                                break

        print("items frequency:",df["is_bill_to_name"].value_counts())
        return df
    except: 
        print("Is bill to name exception :",traceback.print_exc())
        return df_copy

#### extract zip code digit #####
def Get_ZipCode_Digit(df):
    df["zipCode_digits"] = 0
    df_copy = df.copy(deep = True)
    try:

        for idx, row in df.iterrows():
            #print(idx, row["line_num"], row["line_text"],row["word_num"],row["text"])
            text = str(row["text"])
            text = list(text[::-1])
            text_ = text.copy()

            for i, s in enumerate(text_):
                #print(i , s)
                if s == ' ':
                    text.pop(i)

            #print("after deleting whitespace",text)
            for i,s in enumerate(text_):
                if s.isnumeric():
                    break
                else:
                    del text[0]
            #print("after deleting ", text)

            text = list(text[::-1])
            #print("after back 2 reverse deleting ", text)
            text_ = text.copy()

            for i,s in enumerate(text_):
                if s.isnumeric():
                    break
                else:
                    del text[0]
            for i,s in enumerate(text_):
                if s.isnumeric():
                    break

            updated_text= ("".join(text))
            
            #print("Extracted text string :",updated_text)
            if updated_text.isdigit():
                if(len(updated_text)==6):
                    df.loc[[idx],["zipCode_digits"]] = updated_text
                    #print("6 digit zipCode_digits ")
                    #continue 
                else:
                    if(len(updated_text)==3):
                        previous_lineNo = df["line_num"][idx-1]
                        #print("previous_lineNo :", previous_lineNo,"Present Line no ", row["line_num"])
                        if (row["line_num"] == previous_lineNo):
                                previous_word_num = df["word_num"][idx-1]
                                previous_word_num = previous_word_num + 1
                                if previous_word_num == row["word_num"]:
                                    last_item = df['zipCode_digits'][idx - 1] #lst[-1]
                                    if len(last_item) == len(updated_text):
                                        updated_text = last_item + updated_text
                                        print("combine zipCode_digits",updated_text)
                                        df.loc[[idx],["zipCode_digits"]] = updated_text
                                        df.loc[[idx-1],["zipCode_digits"]] = updated_text
                                    else:
                                        df.loc[[idx],["zipCode_digits"]] = updated_text
                                        last_item = df['zipCode_digits'][idx - 1]
                                        if len(last_item)==3:
                                            print("b4 replace block 1 :",last_item)
                                            if idx > 0:
                                                df.loc[[idx - 1],["zipCode_digits"]] = "0"
                                                print("after replace block 2:",last_item)
                                            else:
                                                df.loc[[idx],["zipCode_digits"]] = "0"
                                            print("after replace block 1 :",last_item)
                                            
                                        
                                else:
                                    print(" Not Matched")
                                    df.loc[[idx],["zipCode_digits"]]= updated_text
                                    last_item = df['zipCode_digits'][idx - 1]
                                    if len(last_item)==3:
                                        print("b4 replace block 1 :",last_item)
                                        if idx > 0:
                                            df.loc[[idx - 1],["zipCode_digits"]] = "0"
                                            print("after replace block 2:",last_item)
                                        else:
                                            df.loc[[idx],["zipCode_digits"]] = "0"
                                        print("after replace block 1 :",last_item)
                                    
                        else:
                            df.loc[[idx],["zipCode_digits"]]= updated_text
                            last_item = df["zipCode_digits"][idx-1]
                            if len(last_item)==3:
                                print("b4 replace block 2 :",last_item)
                                if idx > 0:
                                    df.loc[[idx - 1],["zipCode_digits"]] = "0"
                                    print("after replace block 2:",last_item)
                                else:
                                    df.loc[[idx],["zipCode_digits"]] = "0"
                    else:
                        df.loc[[idx],["zipCode_digits"]]= "0"
            else:
                df.loc[[idx],["zipCode_digits"]]= "0"
        
        return df
    except:
        print("Extract zipcode digit exception :",traceback.print_exc())
        return df_copy

## zip code validation
def is_zipCode_India(df, INDIA_ZIP_CODE_RANGE):
    df["is_zipCode_India"] = 0
    df_copy = df.copy(deep = True)
    try:
        ZIP_CODE_RANGE = pd.read_csv(INDIA_ZIP_CODE_RANGE,encoding ="unicode_escape")    
        for idx, row in df.iterrows():
            for ind, val in ZIP_CODE_RANGE.iterrows():
                if (int(row["zipCode_digits"]) >=int(val["Minimum_Pincode"]) and int(row["zipCode_digits"]) <= int(val["Maximum_Pincode"])):
                    df.loc[[idx],["is_zipCode_India"]] = 1
                    # print(" PIN Code :",row)
                    break 
        INDIA_ZIP_CODE_RANGE = None               
        return df
    except:
        print("is_zipCode_India exception", traceback.print_exc())
        return df_copy

# finding ship to Name
# def is_ship_to_name(df,ship_to_Names_kws):
#     '''
#     Retutns df
#     ''' 
#     df["is_ship_to_name"] =""
#     l = []
#     no_items = len(ship_to_Names_kws)
#     #print("no_items :",no_items)
#     for idx, row in df.iterrows():
#         txt = str(row["above_processed_line"])
#         #print("line text :",txt)
#         #print("above line",row["above_processed_line"])
#         if row["is_company_name"]==1:
#             #print("Txt lenth :",len(txt))
#             if len(txt)>0:
#                 i = 0
#                 for item in ship_to_Names_kws:
                    
#                     if item.lower() in txt.lower():
#                         l.append(1)
#                         #print("items matched : ",item , txt )
#                         break 
#                     else:
#                         i = i+1
#                         if i == no_items:
#                             l.append(0)
#                             #print("i :",i)
#                             #print("items not matched: ",item , txt )
                
#             else:
#                # print("test 0 ",txt)
#                 l.append(0)
#         else:
#             #print(" text null")
#             l.append(0)
#     print("df shape :",df.shape)
#     print("list lenth",len(l))
#     df["is_ship_to_name"] = l
#     print("items frequency:",df["is_ship_to_name"].value_counts())
#     return df
def is_ship_to_names(df,ship_to_Names_kws):
    df["is_ship_to_name"] =0
    df_copy = df.copy(deep =True)
    try:
        print("Matching in above_processed_ngbr ")
        for idx, row in df.iterrows():
            ab_ngbr_lst = str(row["above_processed_ngbr"])
            # print("ab_ngbr_lst of type ",type(ab_ngbr_lst))
            if row["is_company_name"]==1:
                # print(" ab_ngbr_lst :",ab_ngbr_lst)
                if ab_ngbr_lst:
                    for item  in ship_to_Names_kws:                    
                        if item.lower() in ab_ngbr_lst.lower():
                            # print("line match : ",line,"at index :",idx )
                            # print("b4r upd",df["is_bill_to_name"][idx])
                            df.loc[[idx],["is_ship_to_name"]] = 1
                            # print("updted indx :",df["is_bill_to_name"][idx], idx)
                            break 
            
        if df["is_ship_to_name"].sum()<=0:
            print(" Matching in left_processed_ngbr ")
            for idx, row in df.iterrows():
                lf_ngbr_lst = str(row["left_processed_ngbr"])
                if row["is_company_name"]==1:
                    if lf_ngbr_lst:
                        for item  in ship_to_Names_kws:                    
                            if item.lower() in lf_ngbr_lst.lower():
                                # print("line match : ",line,"at index :",idx )
                                # print("b4r upd",df["is_bill_to_name"][idx])
                                df.loc[[idx],["is_ship_to_name"]] = 1
                                break
        print("df shape :",df.shape)
        #print("list lenth",len(is_bill_list))
        print("items frequency:",df["is_ship_to_name"].value_counts())
        return df
    except: 
        print(" ship to name exception :",traceback.print_exc())
        return df_copy

def contains_bill2ship2_feature(df):
    df["contains_bill2ship2_feature"] = 0
    df_copy = df.copy(deep =True)
    try:
        for idx , row in df.iterrows():
            if row["is_gstin_format"] ==1:
                df.loc[[idx],["contains_bill2ship2_feature"]] = 1
                continue 
            else:
                if row["is_company_name"] == 1:
                    df.loc[[idx],["contains_bill2ship2_feature"]] = 1
                    continue 
                else: 
                    if row["is_address"] ==1:
                        df.loc[[idx],["contains_bill2ship2_feature"]] = 1
                        continue 
                    else:
                        if row["is_bill_to_name"] == 1:
                            df.loc[[idx],["contains_bill2ship2_feature"]] = 1
                            continue 
                        else :
                            if row["is_ship_to_name"] == 1:
                                df.loc[[idx],["contains_bill2ship2_feature"]] = 1
                                continue 
                            else:
                                if row["is_zipCode_India"] == 1:
                                    df.loc[[idx],["contains_bill2ship2_feature"]] = 1
                                    continue
                                else:
                                    if row["contains_bill_to_name"] == 1:
                                        df.loc[[idx],["contains_bill2ship2_feature"]] = 1
                                    else:
                                        if row["contains_ship_to_name"] == 1:
                                            df.loc[[idx],["contains_bill2ship2_feature"]] = 1

        return df
    except: 
        print("Contains bill2 ship2 feature exception :",traceback.print_exc())
        return df_copy

## geting vendor and buyers details
def remove_chars(s, chars):
    """
    """
    return re.sub('[' + re.escape(''.join(chars)) + ']', '', s)
chars = ['/',':']

def get_vendor_buyers_details_old(DF):
    """
    It tooks DF as input and populates vendor/billing/shipping Name, GSTIN and add address. return DF
    """

    DF["is_vendorGSTIN"] = 0
    DF["is_bilingGSTIN"] = 0
    DF["is_shippingGSTIN"] = 0
    DF["above_gstin_line"] = 0
    DF["vendorName"] = 0
    DF["vendorAddress"] = 0
    DF["billingName"] = 0
    DF["billingAddress"]= 0
    DF["shippingName"] = 0
    DF["shippingAddress"] = 0
    df_copy = DF.copy(deep = True)
    try:
        DF.sort_values(["page_num", "line_num", "word_num"], ascending = [True,True,True],inplace = True)
        if(DF["is_gstin_format"].sum()==0):
            print("GSTIN Not Present")
            df = DF[DF["contains_bill2ship2_feature"]==1]
            print("df shape",df.shape)
            results = []
            for row in df.itertuples():
                bill2address_tokens = []
                ship2address_tokens = []
                is_billing = None
                is_shipping = None
                page_num = row.page_num
                line_num = row.line_num
                line_no = row.tableLineNo
                #clean_line_text = remove_chars(str(row.line_text), chars)
                candidates = df[(df.page_num == page_num) & 
                                ((df.tableLineNo <= line_no) & 
                                 (df.line_top < row.line_top) &
                                 #(df.line_left <= row.line_right) &
                                 ((row.line_right - df.line_left) >= -0.05)&
                                 ((row.line_left - df.line_left)<=0.2)&
                                 (df.line_top - row.line_top >= -0.25)
                                )]
                if candidates.shape[0] > 0:
                    candidates.sort_values(["line_top"], ascending = [False],inplace=True)
                    above_lines = candidates["line_text"].to_list()
                    result = []
                    [result.append(x) for x in above_lines if x not in result] 

                    #DF.loc[[idx],["above_gstin_line"]] = str(result)
                    print("\nPage No :",page_num)
                    print("row.line_text :",row.line_text, " row.is_company_name :",row.is_company_name,"row.is_address",row.is_address)
                    print("Above lines of line_text: ",result)
                    breaker = None
                    for cand_row in candidates.itertuples():
                        if cand_row.is_ship_to_name == 1:
                            #ship2address_tokens.append(row.token_id)

                            if cand_row.line_left < row.line_right:
                                print("less than text line right")
                                is_shipping = True
                                breaker = True
                                break
                            else:
                                print(" Greater than line_text right")
                        if cand_row.is_bill_to_name == 1:
                            if cand_row.line_left < row.line_right:
                                print("less than text line right")
                                is_billing = True
                                breaker = True
                                break
                            else:
                                print(" Greater than line_text right")
                    if is_billing == True and is_shipping == None:
                        if row.is_address == 1:
                            DF.loc[(DF["token_id"]==row.token_id),"billingAddress"] = 1
                            print("Updated as Billing address : ",row.line_text)
                        if row.is_company_name == 1:
                            DF.loc[(DF["token_id"]==row.token_id),"billingName"] = 1
                            print("Updated as Billing name : ",row.line_text)

                    if is_shipping == True and is_billing == None:
                        if row.is_address == 1:
                            DF.loc[(DF["token_id"]==row.token_id),"shippigAddress"] = 1
                            print("Updated as shipping address: ",row.line_text)
                        if row.is_company_name == 1:
                            DF.loc[(DF["token_id"]==row.token_id),"shippingName"] = 1
                            print("Updated as shipping Name : ",row.line_text)
                    if is_shipping == True and is_billing == True:
                        print("Bill to ship to both match found : ",row.line_text)
                if is_billing == None and is_shipping == None:
                    print("is_shipping :",is_shipping, "is_billing",is_billing)
                    if row.is_address == 1:
                        DF.loc[(DF["token_id"]==row.token_id),"vendorAddress"] = 1
                        print("Updated as vendor address ",row.line_text)
#                     if row.is_company_name == 1 and row.is_address == 0:
#                         DF.loc[(DF["token_id"]==row.token_id),"vendorName"] = 1
#                         print("Updated as vendor Name :",row.line_text)

        else:
            print("GSTIN present there")
            cols = ["token_id","page_num","line_num","word_num",
                "right","left","line_down","line_top",
                "line_left","line_right","line_text",
                "tableLineNo","lineLeft","line_left_y1",
                "above_processed_ngbr","is_gstin_format","text","is_address","is_company_name"]

            DF.sort_values(["page_num",
                            "line_num",
                            "word_num"],
                        ascending = [True,True,True],
                        inplace = True)
            df = DF[cols]        
            for idx, row in df.iterrows():
                b_gstin = None
                s_gstin = None
                v_gstin = None
                bill2address_tokens = []
                ship2address_tokens = []
                page_num = row["page_num"]
                line_num = row["line_num"]
                line_no = row["tableLineNo"]
                lft_ln = row["lineLeft"]
                is_gst = row["is_gstin_format"]
                cleanGSTIN = remove_chars(str(row["text"]), chars)
                add_tokens = []
                comp_tokens = []
                if is_gst == 1:
                    candidates = df[(df.page_num == page_num) &
                                    (
                                        (df.tableLineNo <= line_no) &
                                        # (df["line_num"] < line_num) &
                                        (df.line_top < row.line_top) &

                                        #(df.line_left <= row.line_right) &
                                        ((row.line_right - df.line_left) >= -0.05)&
                                        ((row.line_left - df.line_left)<=0.2)&
                                        (df.line_top - row.line_top >= -0.25)
                                    )
                                    ]
                    # print("Ab Ngbrs filter", time.time() - t)
                    if candidates.shape[0] > 0:
                        candidates.sort_values(["line_top"],
                                               ascending = [False],inplace=True)
                        # candidates.to_csv("candidates08.csv")

                        ab_gstin_lines = candidates["line_text"].to_list()
                        result = []
                        [result.append(x) for x in ab_gstin_lines if x not in result] 

                        DF.loc[[idx],["above_gstin_line"]] = str(result)
                        # print("\nPage No :",page_num)
                        # print("lines above GSTIN : ",result)
                        breaker = None
                        for i, cand_row in candidates.iterrows():
                            cand_row_text = remove_chars(str(cand_row["line_text"]),['/',':','(',')','.',"'",","])
                            # print("cand_row_text :",cand_row_text.lower())
                            for b in billToNames_kws:
                                if b.lower() in str(cand_row_text).lower():
                                    match_billing_word = b
                                    b_gstin = cleanGSTIN
                                    breaker = True 
                                    break
                            for s in shipToNames_kws:
                                if s in str(cand_row_text).lower():
                                    match_shipping_word = s
                                    s_gstin = cleanGSTIN
                                    breaker = True
                                    break
                            for name in companyNameList:
                                if name in str(cand_row_text).lower():
                                    # print("line_text",cand_row_text.lower())
                                    if cand_row["is_address"] == 0:
                                        comp_tokens.append(cand_row["token_id"])
                                        break

                            if (b_gstin is not None) & (s_gstin is None):
                                print("BillingGSTIN ",b_gstin)
                                print("match_billing_keys :",match_billing_word)
                                DF.loc[[idx],["is_bilingGSTIN"]] = 1
                                for item in comp_tokens:
                                    #print("Billing token_id:",item)
                                    DF.loc[(DF["token_id"]==item),"billingName"] = 1
                                for items in add_tokens:
                                    DF.loc[(DF["token_id"]==items),"billingAddress"] = 1
                            if (s_gstin is not None) & (b_gstin is None):
                                print("shippingGSTIN ",s_gstin)
                                print("match_shipping_word :",match_shipping_word)
                                DF.loc[[idx],["is_shippingGSTIN"]] = 1
                                for item in comp_tokens:
                                    DF.loc[(DF["token_id"]==item),"shippingName"] = 1
                                for items in add_tokens:
                                    DF.loc[(DF["token_id"]==items),"shippingAddress"] = 1
                            if (b_gstin is not None) & (s_gstin is not None):
                                print("Page No :",page_num)
                                print("Both Ship to and bill to keyword matches found ")
                            if breaker:
                                break 
                            if cand_row["is_company_name"] == 1 & cand_row["is_address"] == 1:
                                add_tokens.append(cand_row["token_id"])
                            if cand_row["is_company_name"] != 1:
                                add_tokens.append(cand_row["token_id"])
                        if (b_gstin is None) & (s_gstin is None):
                            print("Default set")
                            print("\nPage No :",page_num)
                            print("VendorGSTIN ",cleanGSTIN)
                            # print("comp_tokens:",comp_tokens)
                            DF.loc[[idx],["is_vendorGSTIN"]] = 1
                            for item in comp_tokens:
                                DF.loc[(DF["token_id"]==item),"vendorName"] = 1
                            for items in add_tokens:
                                DF.loc[(DF["token_id"]==items),"vendorAddress"] = 1
                    else:
                        DF.loc[[idx],["is_vendorGSTIN"]] = 1
                        print("\nPage No :",page_num )
                        print("vendorGSTIN ",cleanGSTIN)
                        print("Zero Candidates to match")

        return DF
    except:
        print(traceback.print_exc())
        return df_copy

def get_vendor_buyers_details(DF):
    """
    It tooks DF as input and populates vendor/billing/shipping Name, GSTIN and add address. return DF
    """

    DF["is_vendorGSTIN"] = 0
    DF["is_bilingGSTIN"] = 0
    DF["is_shippingGSTIN"] = 0
    DF["above_lines_of_line"] = 0
    DF["vendorName"] = 0
    DF["vendorAddress"] = 0
    DF["billingName"] = 0
    DF["billingAddress"]= 0
    DF["shippingName"] = 0
    DF["shippingAddress"] = 0
    df_copy = DF.copy(deep = True)
    try:
        DF.sort_values(["page_num", "line_num", "word_num"], ascending = [True,True,True],inplace = True)
        if(DF["is_gstin_format"].sum()==0):
            print("GSTIN Not Present")
            df = DF[DF["contains_bill2ship2_feature"]==1]
            print("df shape",df.shape)
            results = []
            for row in df.itertuples():
                bill2address_tokens = []
                ship2address_tokens = []
                is_billing = None
                is_shipping = None
                page_num = row.page_num
                line_num = row.line_num
                line_no = row.tableLineNo
                #clean_line_text = remove_chars(str(row.line_text), chars)
                candidates = df[(df.page_num == page_num) & 
                                ((df.tableLineNo <= line_no) & 
                                 (df.line_top < row.line_top) &
                                 #(df.line_left <= row.line_right) &
                                 ((row.line_right - df.line_left) >= -0.05)&
                                 ((row.line_left - df.line_left)<=0.2)&
                                 (df.line_top - row.line_top >= -0.25)
                                )]
                if candidates.shape[0] > 0:
                    candidates.sort_values(["line_top"], ascending = [False],inplace=True)
                    above_lines = candidates["line_text"].to_list()
                    result = []
                    [result.append(x) for x in above_lines if x not in result] 

                    DF.loc[(DF["token_id"]==row.token_id),"above_lines_of_line"] = str(result)
                    # print("\nPage No :",page_num)
                    # print("row.line_text :",row.line_text, " row.is_company_name :",row.is_company_name,"row.is_address",row.is_address)
                    # print("Above lines of line_text: ",result)
                    breaker = None
                    for cand_row in candidates.itertuples():
                        cand_row_text = remove_chars(str(cand_row.line_text),['/',':','(',')','.',"'",","])
                        # print("cand_row_text :",cand_row_text.lower())
                        for b in billToNames_kws:
                            if b.lower() in str(cand_row_text).lower():
                                match_billing_word = b
                                is_billing = True
                                breaker = True 
                                break
                        for s in shipToNames_kws:
                            if s in str(cand_row_text).lower():
                                match_shipping_word = s
                                is_shipping = True
                                breaker = True
                                break
                        if (is_billing == True) & (is_shipping is None):
                            if (row.is_address ==1):
                                print("line text of Address ",row.line_text)
                                print("match_billing_keys :",match_billing_word)
                                DF.loc[(DF["token_id"]==row.token_id),"billingAddress"] = 1
                            if (row.is_company_name == 1) & (row.is_address==0):
                                print("line text of Company ",row.line_text)
                                print("match_billing_keys :",match_billing_word)
                                DF.loc[(DF["token_id"]==row.token_id),"billingName"] = 1
                                
                        if (is_shipping is not None) & (is_billing is None):
                            if (row.is_address ==1):
                                print("line text of Address ",row.line_text)
                                print("match_billing_keys :",match_shipping_word)
                                DF.loc[(DF["token_id"]==row.token_id),"shippingAddress"] = 1
                            if (row.is_company_name == 1) & (row.is_address==0):
                                print("line text of Company ",row.line_text)
                                print("match_billing_keys :",match_shipping_word)
                                DF.loc[(DF["token_id"]==row.token_id),"shippingName"] = 1
                        if (is_shipping is not None) & (is_billing is not None):
                            print("Page No :",page_num)
                            print("Both Ship to and bill to keyword matches found ")
                        if breaker:
                            break 
                    if (is_shipping is None) & (is_billing is None):
                        if (row.is_address ==1):
                            print("line text of Address ",row.line_text)
                            DF.loc[(DF["token_id"]==row.token_id),"vendorAddress"] = 1
                        if (row.is_company_name == 1) & (row.is_address==0):
                            print("line text of Company ",row.line_text)
                            DF.loc[(DF["token_id"]==row.token_id),"VendorName"] = 1
                    
        else:
            print("GSTIN present there")
            df = DF
            for idx, row in df.iterrows():
                b_gstin = None
                s_gstin = None
                v_gstin = None
                bill2address_tokens = []
                ship2address_tokens = []
                page_num = row["page_num"]
                line_num = row["line_num"]
                line_no = row["tableLineNo"]
                lft_ln = row["lineLeft"]
                is_gst = row["is_gstin_format"]
                cleanGSTIN = remove_chars(str(row["text"]), chars)
                add_tokens = []
                comp_tokens = []
                if is_gst == 1:
                    candidates = df[(df.page_num == page_num) &
                                    (
                                        (df.tableLineNo <= line_no) &
                                        # (df["line_num"] < line_num) &
                                        (df.line_top < row.line_top) &

                                        #(df.line_left <= row.line_right) &
                                        ((row.line_right - df.line_left) >= -0.05)&
                                        ((row.line_left - df.line_left)<=0.2)&
                                        (df.line_top - row.line_top >= -0.25)
                                    )
                                    ]
                    # print("Ab Ngbrs filter", time.time() - t)
                    if candidates.shape[0] > 0:
                        candidates.sort_values(["line_top"], ascending = [False],inplace=True)
                        # candidates.to_csv("candidates08.csv")

                        ab_gstin_lines = candidates["line_text"].to_list()
                        result = []
                        [result.append(x) for x in ab_gstin_lines if x not in result] 

                        DF.loc[[idx],["above_lines_of_line"]] = str(result)
                        # print("\nPage No :",page_num)
                        print("lines above GSTIN : ",result)
                        breaker = None
                        for i, cand_row in candidates.iterrows():
                            cand_row_text = remove_chars(str(cand_row["line_text"]),['/',':','(',')','.',"'",","])
                            # print("cand_row_text :",cand_row_text.lower())
                            for b in billToNames_kws:
                                if b.lower() in str(cand_row_text).lower():
                                    match_billing_word = b
                                    b_gstin = cleanGSTIN
                                    breaker = True 
                                    break
                            for s in shipToNames_kws:
                                if s in str(cand_row_text).lower():
                                    match_shipping_word = s
                                    s_gstin = cleanGSTIN
                                    breaker = True
                                    break
                            for name in companyNameList:
                                if name in str(cand_row_text).lower():
                                    # print("line_text",cand_row_text.lower())
                                    if cand_row["is_address"] == 0:
                                        comp_tokens.append(cand_row["token_id"])
                                        break
                            if cand_row["is_address"] == 1:
                                add_tokens.append(cand_row["token_id"])
                                
                            if (b_gstin is not None) & (s_gstin is None):
                                print("BillingGSTIN ",b_gstin)
                                print("match_billing_keys :",match_billing_word)
                                DF.loc[[idx],["is_bilingGSTIN"]] = 1
                                for item in comp_tokens:
                                    #print("Billing token_id:",item)
                                    DF.loc[(DF["token_id"]==item),"billingName"] = 1
                                for items in add_tokens:
                                    DF.loc[(DF["token_id"]==items),"billingAddress"] = 1
                            if (s_gstin is not None) & (b_gstin is None):
                                print("shippingGSTIN ",s_gstin)
                                print("match_shipping_word :",match_shipping_word)
                                DF.loc[[idx],["is_shippingGSTIN"]] = 1
                                for item in comp_tokens:
                                    DF.loc[(DF["token_id"]==item),"shippingName"] = 1
                                for items in add_tokens:
                                    DF.loc[(DF["token_id"]==items),"shippingAddress"] = 1
                            if (b_gstin is not None) & (s_gstin is not None):
                                print("Page No :",page_num)
                                print("Both Ship to and bill to keyword matches found ")
                            if breaker:
                                break 
                            if ((cand_row["is_company_name"] == 1) & (cand_row["is_address"] == 1)):
                                add_tokens.append(cand_row["token_id"])
                            if cand_row["is_company_name"] != 1:
                                add_tokens.append(cand_row["token_id"])
                        if (b_gstin is None) & (s_gstin is None):
                            print("Default set")
                            print("\nPage No :",page_num)
                            print("VendorGSTIN ",cleanGSTIN)
                            # print("comp_tokens:",comp_tokens)
                            TEMP_DF = DF[DF["page_num"]== page_num]
                            if (TEMP_DF["is_vendorGSTIN"].sum() < 1):
                                DF.loc[[idx],["is_vendorGSTIN"]] = 1
                                for item in comp_tokens:
                                    DF.loc[(DF["token_id"]==item),"vendorName"] = 1
                                for items in add_tokens:
                                    DF.loc[(DF["token_id"]==items),"vendorAddress"] = 1
                    else:
                        DF.loc[[idx],["is_vendorGSTIN"]] = 1
                        print("\nPage No :",page_num )
                        print("vendorGSTIN ",cleanGSTIN)
                        print("Zero Candidates to match")
         
        return DF
    except:
        print(traceback.print_exc())
        return df_copy


def get_bill_to_ship_to_details(df, zipcodes, bill_to_Names_kws, ship_to_Names_kws):
    
    # extract zip code digit from line text
    df = Get_ZipCode_Digit(df)
    print(df.shape)
    # Validate zipcode
    df = is_zipCode_India(df,zipcodes)
    #print(" df shape after zip valid",df.shape)
    # find is the company name     
    df = containsCompanyName(df)

    # checki if line of text contains address
    df = containsAddress(df)

    # check is adress region
    df = is_address_region(df)
    #df = find_left_above_lines(df)
    df = fz_lbl_bill_to(df)
    df = fz_lbl_ship_to(df)
    # is bill to name
    df = contains_bill_to_name(df)
    # check if contains ship to Name
    df = contains_ship_to_name(df)
    # check if Bill To Company Name
    df = is_bill_to_names(df,bill_to_Names_kws)
    # check if ship to company Name
    df = is_ship_to_names(df,ship_to_Names_kws)
    # Check if billTOShip to feature
    df = contains_bill2ship2_feature(df)
    df = get_vendor_buyers_details(df)
    return df

if __name__ == "__main__":
    df = pd.read_csv(r"C:\Users\OVE\Downloads\GSTIN_extraction_pred\2022-06-16T11-00-23-00-00--2_pred.csv",index_col = False)
    df = get_bill_to_ship_to_details(df,INDIA_ZIP_CODE_RANGE, billToNames_kws, shipToNames_kws)
    df.to_csv(r"C:\Users\OVE\Downloads\GSTIN_extraction_pred\2022-06-16T11-00-23-00-00--2_pred1.csv.csv")


