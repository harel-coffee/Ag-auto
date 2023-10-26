# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.5
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%

# %%
import pandas as pd
import numpy as np

# %%
merged_pred_dir = "/Users/hn/Documents/01_research_data/NASA/ML_data_Oct17/DeskRejectResults/01_merged_preds/"

# %%
DL_preds = pd.read_csv(merged_pred_dir + "all_DL_preds.csv")
DL_preds.head(2)

# %%
DL_preds_train = DL_preds[DL_preds.train_test == "train"].copy()
DL_preds_train.reset_index(drop=True, inplace=True)
DL_preds_train.head(2)

# %%
print (sorted(DL_preds_train.SR.unique()))
print (sorted(DL_preds_train.train_ID.unique()))

# %%
del(DL_preds_train)
DL_preds_test = DL_preds[DL_preds.train_test == "test"].copy()
DL_preds_test.reset_index(drop=True, inplace=True)
DL_preds_test.head(2)

# %%
test_results = {}
best_params = {}

for a_train_ID in sorted(DL_preds_test.train_ID.unique()):
    a_test_set = DL_preds_test[DL_preds_test.train_ID == a_train_ID].copy()
    
    curr_best_params = best_params["train_ID" + str(a_train_ID)]
    
    best_cut = curr_best_params["cut"]
    best_ratio = curr_best_params["ratio"]
    
    a_test_set_SR = a_test_set[a_test_set.SR == best_ratio].copy()
    
    colName = "NDVI_SG_DL_p" + str(int(best_cut*10))
    a_test_set_SR[colName] = -1
    a_test_set_SR.loc[a_test_set_SR.prob_single < a_cut, colName] = 2
    a_test_set_SR.loc[a_test_set_SR.prob_single >= a_cut, colName] = 1
    
    
    error_count = (a_test_set_SR.Vote != a_test_set_SR[colName]).sum()
    accuracy = (len(a_test_set_SR) - error_count) / len(a_test_set_SR)

    double_agreement_df = a_test_set_SR[a_test_set_SR.Vote==2]
    double_agreement = (double_agreement_df.Vote == double_agreement_df[colName]).sum()
    double_disagreement = len(double_agreement_df) - double_agreement
    
    curr_test_result = {"acc" : accuracy,
                        "err_count" : error_count,
                        "double_agreement": double_agreement,
                        "double_disagreement":double_disagreement
                       }
    
    
    test_results["train_ID" + str(a_train_ID)] = curr_test_result

# %%
for a_key in test_results.keys():
    print (a_key)
    print (test_results[a_key])
    print ("==========================================================================")


# %%
# Hyper by test set:

# %%
cuts_ = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
best_params = {}

for a_train_ID in sorted(DL_preds_test.train_ID.unique()):
    print (f"{a_train_ID = }")
    best_in_set = {# "train_ID": a_train_ID,
                   "cut":0, 
                   "ratio":0, 
                   "acc.":-1, 
                   "double_agreement":-10, 
                   "double_disagreement":-10, 
                   "err_count":100000}
    a_test_set = DL_preds_test[DL_preds_test.train_ID == a_train_ID].copy()
    
    for a_SR in sorted(a_test_set.SR.unique()):
        a_test_set_SR = a_test_set[a_test_set.SR == a_SR].copy()
        
        for a_cut in cuts_:
            colName = "NDVI_SG_DL_p" + str(int(a_cut*10))
            a_test_set_SR[colName] = -1
            a_test_set_SR.loc[a_test_set_SR.prob_single < a_cut, colName] = 2
            a_test_set_SR.loc[a_test_set_SR.prob_single >= a_cut, colName] = 1
            
            error_count = (a_test_set_SR.Vote != a_test_set_SR[colName]).sum()
            accuracy = (len(a_test_set_SR) - error_count) / len(a_test_set_SR)
            
            double_agreement_df = a_test_set_SR[a_test_set_SR.Vote==2]
            double_agreement = (double_agreement_df.Vote == double_agreement_df[colName]).sum()
            double_disagreement = len(double_agreement_df) - double_agreement
            
            if error_count < best_in_set["err_count"]:
                best_in_set["cut"] = a_cut
                best_in_set["ratio"] = a_SR
                best_in_set["acc."] = accuracy
                best_in_set["double_agreement"] = double_agreement
                best_in_set["double_disagreement"] = double_disagreement
                best_in_set["err_count"] = error_count
                
            elif error_count == best_in_set["err_count"]:
                if double_agreement > best_in_set["double_agreement"]:
                    best_in_set["cut"] = a_cut
                    best_in_set["ratio"] = a_SR
                    best_in_set["acc."] = accuracy
                    best_in_set["double_agreement"] = double_agreement
                    best_in_set["double_disagreement"] = double_disagreement
                    best_in_set["err_count"] = error_count
                    
    print (best_in_set)
    best_params["train_ID" + str(a_train_ID)] = best_in_set
    print ("=============================================================================")

# %%

# %%
