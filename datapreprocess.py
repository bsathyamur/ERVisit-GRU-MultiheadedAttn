"""
- Script takes year as a input and uses the claims and enroll_synth 
- data files as input. data preprocessing is performed and 
- It then sorts the claims based on claim from dates
- converts the diagnosis codes to CCS codes based on the data in the
- ccs_diag_xwalk.xls file and  
- sets the target 
"""
import os
import sys
import time
import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.simplefilter("ignore")
import logging
import yaml
import pickle

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from multiprocessing import cpu_count
n_cores = cpu_count()

RANDOM_STATE = 2

# retrieve the configuration values
with open('config/default.yml') as f:
    preprocess_config = yaml.load(f, Loader=yaml.FullLoader)

input_years = preprocess_config['input_years']

starttime = time.time()

input_years.sort()

logging.info(f"Data prepocessing for the years {input_years}")

assert len(input_years) > 0, "at least 1 input year is needed"

# Loading the claims data files

CLAIM_PATH = Path("inputFiles/claims/")
df_claims = pd.concat(
    [pd.read_csv(CLAIM_PATH / Path(f"claims_{year}.dat"),dtype='unicode',sep = "|") for year in input_years]
)

logging.info(f"claims total rows for processing is {df_claims.shape[0]}")

# Loading the enrollment data file

ENROLL_PATH = Path("inputFiles/enroll/")

df_enroll = pd.read_csv(ENROLL_PATH / Path("enroll_synth.dat"),dtype='unicode',sep = "|")

logging.info(f"total enroll records is {df_enroll.shape[0]}")

# converting the claim from and to date to datetime
df_claims["from_dt"] = pd.to_datetime(df_claims["from_dt"])
df_claims["to_dt"] = pd.to_datetime(df_claims["to_dt"])

# fill Nan with space
df_claims = df_claims.fillna('')

#remove pharmacy claims
df_claims = df_claims[df_claims.rectype != "P"]

logging.info(f"claims total rows after removing pharmacy records is {df_claims.shape[0]}")

# filter for patients with more than one claims
df_claims = df_claims.groupby('pat_id').filter(lambda x : len(x) > 1)

logging.info(f"claims total rows after patients with just 1 claim is {df_claims.shape[0]}")

# remove null rectype records
df_claims = df_claims[df_claims.rectype != '']

logging.info(f"claims total rows after null rec type records is {df_claims.shape[0]}")

# sort claims by patient id and claim from date

df_claims = df_claims.sort_values(by=['pat_id','from_dt'])

# Identity emergency claims based on revenue code
emergency_rev_cd = preprocess_config['emergency_rev_cd']
df_claims["ER_Ind"] = np.where(df_claims.rev_code.isin(emergency_rev_cd),"Y","N")

logging.info(f"List of emergency revenue codes is {emergency_rev_cd}")

total_emergency_claims =df_claims[df_claims.ER_Ind == "Y"]["pat_id"].count()

logging.info(f"Total emergency claims identified is {total_emergency_claims}")

# reset the claims index due to sorting and deletion
df_claims = df_claims.reset_index()

# Load the CCS conversion codes to a dictionary
CCS_PATH = Path("inputFiles/CCS/")

df_ccs_xwlk = pd.read_csv (CCS_PATH / Path("ccs_xwlk.csv"),dtype='unicode') 

ccs_dict = {row.proc_cd:row.CCSR_label for index,row in df_ccs_xwlk.iterrows()}

logging.info(f"Total CCS codes rows loaded is {len(ccs_dict)}")

def generateDiagSequence(df_claims,ccs_dict):
    
    """
    Function to generate the Diag code sequences for each claims in the
    dataframe using the ccs_dictionary
    """

    cnt = 0

    for index,row in df_claims.iterrows():
        
        seq = []
    
        # For all 12 diagnosis codes in the claims file

        if len(row["diag1"].strip()) > 0 and row["diag1"] in ccs_dict:
            seq.append(int(ccs_dict[row["diag1"]]))
                
        if len(row["diag2"].strip()) > 0 and row["diag2"] in ccs_dict:
            seq.append(int(ccs_dict[row["diag2"]]))           
                
        if len(row["diag3"].strip()) > 0 and row["diag3"] in ccs_dict:
            seq.append(int(ccs_dict[row["diag3"]]))

        if len(row["diag4"].strip()) > 0 and row["diag4"] in ccs_dict:
            seq.append(int(ccs_dict[row["diag4"]]))

        if len(row["diag5"].strip()) > 0 and row["diag5"] in ccs_dict:
            seq.append(int(ccs_dict[row["diag5"]]))        

        if len(row["diag6"].strip()) > 0 and row["diag6"] in ccs_dict:
            seq.append(int(ccs_dict[row["diag6"]]))
        
        if len(row["diag7"].strip()) > 0 and row["diag7"] in ccs_dict:
            seq.append(int(ccs_dict[row["diag7"]]))
                
        if len(row["diag8"].strip()) > 0 and row["diag8"] in ccs_dict:
            seq.append(int(ccs_dict[row["diag8"]]))
                
        if len(row["diag9"].strip()) > 0 and row["diag9"] in ccs_dict:
            seq.append(int(ccs_dict[row["diag9"]]))
                
        if len(row["diag10"].strip()) > 0 and row["diag10"] in ccs_dict:
            seq.append(int(ccs_dict[row["diag10"]]))
                        
        if len(row["diag11"].strip()) > 0 and row["diag11"] in ccs_dict:
            seq.append(int(ccs_dict[row["diag11"]]))
                
        if len(row["diag12"].strip()) > 0 and row["diag12"] in ccs_dict:
            seq.append(int(ccs_dict[row["diag12"]]))

        if len(seq) > 0:
            df_claims.loc[index, 'diagSeq'] = sorted(seq)
            cnt += 1

    return df_claims,cnt

# Create a empty column for diagnosis sequence
df_claims = df_claims.astype('object')
df_claims["diagSeq"] = ""

df_claims,row_cnt = generateDiagSequence(df_claims,ccs_dict)

logging.info(f"Total claims with diag sequence generated is {row_cnt}")

# remove null diag Sequence
df_claims = df_claims[df_claims.diagSeq != '']

logging.info(f"claims total rows after removing empty diagnosis sequence is {df_claims.shape[0]}")

# Store the diag sequences, ER_Indicator and Claims Days duration is a dictionary
diagSeqs = {}
ER_Ind = {}
maxClms = preprocess_config['max_claims']

logging.info(f"Generating diagnosis sequence dictionary for maximum no claims threshold = {maxClms}")

for index,row in df_claims.iterrows():
    
    pat_id = row["pat_id"]
    
    if pat_id not in ER_Ind:        
        if row["ER_Ind"] == "N":
            ER_Ind[pat_id] = 0
        else:
            ER_Ind[pat_id] = 1      
    else:
        if row["ER_Ind"] == "Y" and ER_Ind[pat_id] == 0:
            ER_Ind[pat_id] = 1
        
    if ER_Ind[pat_id] == 0:
        
        if pat_id in diagSeqs:
            
            if len(diagSeqs[pat_id]) < maxClms:            
                diagSeqs[pat_id].append(row["diagSeq"]) 
                
        else:
            diagSeqs[pat_id] = [row["diagSeq"]]

logging.info(f"Total diagnosis sequence is {len(diagSeqs)}")
logging.info(f"Total target label(ER Indicator) is {len(ER_Ind)}")

# Process member demographics data

df_enroll = df_enroll[["pat_id","pat_region","mh_cd","der_yob","der_sex"]]

df_enroll = df_enroll[~df_enroll.der_yob.isin(["0",np.nan])]
df_enroll["der_yob"] = df_enroll["der_yob"].astype(int)

df_enroll = df_enroll.fillna('')

member_demo = {}

calc_age = max(input_years)

logging.info(f"Calculating member age in demographics data for {calc_age}")

for _,row in df_enroll.iterrows():
    
    pat_id = row.pat_id
    
    if pat_id is not None:

        age_diff = calc_age - row.der_yob
        age = 0

        if age_diff > 0 and age_diff <= 120:
            age = age_diff
        elif age_diff > 120:
            age = 120

        # label for gender
        if row.der_sex == "M":
            gender = 0
        elif row.der_sex == "F":
            gender = 1
        else:
            gender = 2
            
        member_demo[pat_id] = {"age": age,"gender": gender}

logging.info(f"Total member demo loaded is {len(member_demo)}")

# Generating the final diagnosis sequence, claims days, member demographics and target (ER Indicator)

diagSeqs_final = {}
ER_Ind_final = {}
member_demo_final = {}

patid = 0

for key in diagSeqs.keys():
    
    # Check if the key is present in all the 4 dicts
    if key in diagSeqs and key in ER_Ind and key in member_demo:
        diagSeqs_final[patid] = diagSeqs[key]
        ER_Ind_final[patid] = ER_Ind[key]
        member_demo_final[patid] = member_demo[key]
        
        patid += 1

logging.info(f"total number of diagnosis records is {len(diagSeqs_final)}")
logging.info(f"total number of member demographics records is {len(member_demo_final)}")
logging.info(f"total number of target labels is {len(ER_Ind_final)}")

# Writing the data to a object

file_to_write = open("output/diagSeqs.pickle", "wb")
pickle.dump(diagSeqs_final, file_to_write)

file_to_write = open("output/ER_Ind.pickle", "wb")
pickle.dump(ER_Ind_final, file_to_write)

file_to_write = open("output/member_demo.pickle", "wb")
pickle.dump(member_demo_final, file_to_write)

logging.info(f"data preprocessing completed in {(time.time() - starttime)/60:.4f} minutes.")