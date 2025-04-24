"""
    Last Modified: 2025/04/04
    Author: Veerupaksh (Veeru) Singla (singla2@purdue.edu)
    Description: make csv for each model each split accuracy.
"""


import os


this_script_dir = os.path.dirname(os.path.realpath(__file__))
os.chdir(this_script_dir)


import sys
sys.path.append(os.path.join(this_script_dir, ".."))


from utils import *


k_fold_accuracy_analysis_dir = os.path.join(analyses_dir, "k_fold_accuracy_analysis")

xgb_direct_preds_dir = os.path.join(preds_dir, "xgb_direct")
xgb_delta_preds_dir = os.path.join(preds_dir, "xgb_delta")

chemprop_direct_preds_dir = os.path.join(preds_dir, "chemprop_direct")
chemprop_delta_preds_dir = os.path.join(preds_dir, "chemprop_delta")

refine_fn_1 = "wb97xd"
refine_fn_2 = "b2plypd3"
chemprop_direct_transfer_preds_dir_1 = os.path.join(preds_dir, f"chemprop_direct_transfer_{refine_fn_1}")
chemprop_direct_transfer_preds_dir_2 = os.path.join(preds_dir, f"chemprop_direct_transfer_{refine_fn_2}")
chemprop_delta_transfer_preds_dir_1 = os.path.join(preds_dir, f"chemprop_delta_transfer_{refine_fn_1}")
chemprop_delta_transfer_preds_dir_2 = os.path.join(preds_dir, f"chemprop_delta_transfer_{refine_fn_2}")


def detect_ground_truth_ea_dict(preds_json_name):
    if "combined" in preds_json_name:
        return "combined"
    elif "preds_fwd" not in preds_json_name and "preds_rev" not in preds_json_name:
        if "fwd" in preds_json_name:
            return "fwd"
        elif "rev" in preds_json_name:
            return "rev"
        else:
            raise ValueError(f"Ground truth EA not found in {preds_json_name}")
    elif "preds_fwd" in preds_json_name:
        return "fwd"
    elif "preds_rev" in preds_json_name:
        return "rev"
    else:
        raise ValueError(f"Ground truth EA not found in {preds_json_name}")


def get_train_test_mr_ar_ea_ea_preds_lists(k_fold_preds_mr_ar_ea_dict, mr_ar_ea_dict, k_fold_splits_dict):
    train_mr_ar_list = []
    train_ea_mr_list = []
    train_ea_list = []
    train_ea_pred_list = []
    
    test_mr_ar_list = []
    test_ea_mr_list = []
    test_ea_list = []
    test_ea_pred_list = []
    
    for fold in k_fold_splits_dict:
        fold_all_mr_ar_set = set(k_fold_preds_mr_ar_ea_dict[fold].keys())
        fold_test_mr_ar_set = fold_all_mr_ar_set & set(k_fold_splits_dict[fold][-1])
        fold_test_mr_ar_list = list(fold_test_mr_ar_set)
        fold_train_mr_ar_list = list(fold_all_mr_ar_set - fold_test_mr_ar_set)  ## includes validation set as well
        
        fold_test_ea_list = [mr_ar_ea_dict[mr_ar] for mr_ar in fold_test_mr_ar_list]
        fold_train_ea_list = [mr_ar_ea_dict[mr_ar] for mr_ar in fold_train_mr_ar_list]
        
        fold_test_ea_mr_list = [mr_ar_ea_dict[ar_name_to_mr_name(mr_ar)] for mr_ar in fold_test_mr_ar_list]
        fold_train_ea_mr_list = [mr_ar_ea_dict[ar_name_to_mr_name(mr_ar)] for mr_ar in fold_train_mr_ar_list]
        
        fold_test_ea_pred_list = [k_fold_preds_mr_ar_ea_dict[fold][mr_ar] for mr_ar in fold_test_mr_ar_list]
        fold_train_ea_pred_list = [k_fold_preds_mr_ar_ea_dict[fold][mr_ar] for mr_ar in fold_train_mr_ar_list]
        
        train_mr_ar_list.extend(fold_train_mr_ar_list)
        train_ea_mr_list.extend(fold_train_ea_mr_list)
        train_ea_list.extend(fold_train_ea_list)
        train_ea_pred_list.extend(fold_train_ea_pred_list)
        
        test_mr_ar_list.extend(fold_test_mr_ar_list)
        test_ea_mr_list.extend(fold_test_ea_mr_list)
        test_ea_list.extend(fold_test_ea_list)
        test_ea_pred_list.extend(fold_test_ea_pred_list)
        
    return train_mr_ar_list, train_ea_mr_list, train_ea_list, train_ea_pred_list, test_mr_ar_list, test_ea_mr_list, test_ea_list, test_ea_pred_list


def preds_dir_to_k_fold_test_accuracy_csv_list(preds_dir, mr_ar_ea_name_dict, k_fold_splits_name_dict):
    csv_list = ["preds_json_name,r2_train,median_ae_train,mae_train,rmse_train,r2_test,median_ae_test,mae_test,rmse_test"]
    
    preds_json_list = os.listdir(preds_dir)
    preds_json_list.sort()
    for preds_json_name in tqdm(preds_json_list):
        ea_dict_name = detect_ground_truth_ea_dict(preds_json_name)
        mr_ar_ea_dict_to_use = mr_ar_ea_name_dict[ea_dict_name]
        if ea_dict_name == "combined":
            if "stratified" in preds_json_name:
                k_fold_splits_dict_to_use = k_fold_splits_name_dict["stratified_combined"]
            elif "random" in preds_json_name:
                k_fold_splits_dict_to_use = k_fold_splits_name_dict["random_combined"]
            elif "mr_scaffold" in preds_json_name:
                k_fold_splits_dict_to_use = k_fold_splits_name_dict["mr_scaffold_combined"]
        else:
            if "stratified" in preds_json_name:
                k_fold_splits_dict_to_use = k_fold_splits_name_dict["stratified"]
            elif "random" in preds_json_name:
                k_fold_splits_dict_to_use = k_fold_splits_name_dict["random"]
            elif "mr_scaffold" in preds_json_name:
                k_fold_splits_dict_to_use = k_fold_splits_name_dict["mr_scaffold"]
        
        preds_dict = json.load(open(os.path.join(preds_dir, preds_json_name), "r"))
        if len(preds_dict["0"]) == 5:
            ## inc train percent preds. [20%, 40%, 60%, 80%, 100%]
            
            ## for combined trained models, we are only interested in fwd preds to check the effect of data augmentation with reverse rxn data
            
            if ea_dict_name == "combined":
                train_20_k_fold_preds_mr_ar_ea_dict = {fold: {k: v for k, v in zip(preds_dict[fold][0][0], preds_dict[fold][0][1]) if "_rev" not in k} for fold in k_fold_splits_dict_to_use.keys()}
                train_40_k_fold_preds_mr_ar_ea_dict = {fold: {k: v for k, v in zip(preds_dict[fold][1][0], preds_dict[fold][1][1]) if "_rev" not in k} for fold in k_fold_splits_dict_to_use.keys()}
                train_60_k_fold_preds_mr_ar_ea_dict = {fold: {k: v for k, v in zip(preds_dict[fold][2][0], preds_dict[fold][2][1]) if "_rev" not in k} for fold in k_fold_splits_dict_to_use.keys()}
                train_80_k_fold_preds_mr_ar_ea_dict = {fold: {k: v for k, v in zip(preds_dict[fold][3][0], preds_dict[fold][3][1]) if "_rev" not in k} for fold in k_fold_splits_dict_to_use.keys()}
                train_100_k_fold_preds_mr_ar_ea_dict = {fold: {k: v for k, v in zip(preds_dict[fold][4][0], preds_dict[fold][4][1]) if "_rev" not in k} for fold in k_fold_splits_dict_to_use.keys()}
            else:
                train_20_k_fold_preds_mr_ar_ea_dict = {fold: {k: v for k, v in zip(preds_dict[fold][0][0], preds_dict[fold][0][1])} for fold in k_fold_splits_dict_to_use.keys()}
                train_40_k_fold_preds_mr_ar_ea_dict = {fold: {k: v for k, v in zip(preds_dict[fold][1][0], preds_dict[fold][1][1])} for fold in k_fold_splits_dict_to_use.keys()}
                train_60_k_fold_preds_mr_ar_ea_dict = {fold: {k: v for k, v in zip(preds_dict[fold][2][0], preds_dict[fold][2][1])} for fold in k_fold_splits_dict_to_use.keys()}
                train_80_k_fold_preds_mr_ar_ea_dict = {fold: {k: v for k, v in zip(preds_dict[fold][3][0], preds_dict[fold][3][1])} for fold in k_fold_splits_dict_to_use.keys()}
                train_100_k_fold_preds_mr_ar_ea_dict = {fold: {k: v for k, v in zip(preds_dict[fold][4][0], preds_dict[fold][4][1])} for fold in k_fold_splits_dict_to_use.keys()}
        
            train_20_train_mr_ar_list, train_20_train_ea_mr_list, train_20_train_ea_list, train_20_train_ea_pred_list, train_20_test_mr_ar_list, train_20_test_ea_mr_list, train_20_test_ea_list, train_20_test_ea_pred_list = get_train_test_mr_ar_ea_ea_preds_lists(train_20_k_fold_preds_mr_ar_ea_dict, mr_ar_ea_dict_to_use, k_fold_splits_dict_to_use)
            train_40_train_mr_ar_list, train_40_train_ea_mr_list, train_40_train_ea_list, train_40_train_ea_pred_list, train_40_test_mr_ar_list, train_40_test_ea_mr_list, train_40_test_ea_list, train_40_test_ea_pred_list = get_train_test_mr_ar_ea_ea_preds_lists(train_40_k_fold_preds_mr_ar_ea_dict, mr_ar_ea_dict_to_use, k_fold_splits_dict_to_use)
            train_60_train_mr_ar_list, train_60_train_ea_mr_list, train_60_train_ea_list, train_60_train_ea_pred_list, train_60_test_mr_ar_list, train_60_test_ea_mr_list, train_60_test_ea_list, train_60_test_ea_pred_list = get_train_test_mr_ar_ea_ea_preds_lists(train_60_k_fold_preds_mr_ar_ea_dict, mr_ar_ea_dict_to_use, k_fold_splits_dict_to_use)
            train_80_train_mr_ar_list, train_80_train_ea_mr_list, train_80_train_ea_list, train_80_train_ea_pred_list, train_80_test_mr_ar_list, train_80_test_ea_mr_list, train_80_test_ea_list, train_80_test_ea_pred_list = get_train_test_mr_ar_ea_ea_preds_lists(train_80_k_fold_preds_mr_ar_ea_dict, mr_ar_ea_dict_to_use, k_fold_splits_dict_to_use)
            train_100_train_mr_ar_list, train_100_train_ea_mr_list, train_100_train_ea_list, train_100_train_ea_pred_list, train_100_test_mr_ar_list, train_100_test_ea_mr_list, train_100_test_ea_list, train_100_test_ea_pred_list = get_train_test_mr_ar_ea_ea_preds_lists(train_100_k_fold_preds_mr_ar_ea_dict, mr_ar_ea_dict_to_use, k_fold_splits_dict_to_use)
            
            train_20_train_stats_dict = return_stats(train_20_train_ea_list, train_20_train_ea_pred_list)
            train_20_test_stats_dict = return_stats(train_20_test_ea_list, train_20_test_ea_pred_list)
            train_40_train_stats_dict = return_stats(train_40_train_ea_list, train_40_train_ea_pred_list)
            train_40_test_stats_dict = return_stats(train_40_test_ea_list, train_40_test_ea_pred_list)
            train_60_train_stats_dict = return_stats(train_60_train_ea_list, train_60_train_ea_pred_list)
            train_60_test_stats_dict = return_stats(train_60_test_ea_list, train_60_test_ea_pred_list)
            train_80_train_stats_dict = return_stats(train_80_train_ea_list, train_80_train_ea_pred_list)
            train_80_test_stats_dict = return_stats(train_80_test_ea_list, train_80_test_ea_pred_list)
            train_100_train_stats_dict = return_stats(train_100_train_ea_list, train_100_train_ea_pred_list)
            train_100_test_stats_dict = return_stats(train_100_test_ea_list, train_100_test_ea_pred_list)
            
            ## using mr ea as direct prediction for ar ea
            train_20_train_stats_mr_surrogate_dict = return_stats(train_20_train_ea_list, train_20_train_ea_mr_list)
            train_20_test_stats_mr_surrogate_dict = return_stats(train_20_test_ea_list, train_20_test_ea_mr_list)
            train_40_train_stats_mr_surrogate_dict = return_stats(train_40_train_ea_list, train_40_train_ea_mr_list)
            train_40_test_stats_mr_surrogate_dict = return_stats(train_40_test_ea_list, train_40_test_ea_mr_list)
            train_60_train_stats_mr_surrogate_dict = return_stats(train_60_train_ea_list, train_60_train_ea_mr_list)
            train_60_test_stats_mr_surrogate_dict = return_stats(train_60_test_ea_list, train_60_test_ea_mr_list)
            train_80_train_stats_mr_surrogate_dict = return_stats(train_80_train_ea_list, train_80_train_ea_mr_list)
            train_80_test_stats_mr_surrogate_dict = return_stats(train_80_test_ea_list, train_80_test_ea_mr_list)
            train_100_train_stats_mr_surrogate_dict = return_stats(train_100_train_ea_list, train_100_train_ea_mr_list)
            train_100_test_stats_mr_surrogate_dict = return_stats(train_100_test_ea_list, train_100_test_ea_mr_list)
            
            
            if ea_dict_name == "combined":
                csv_list.append(f"{preds_json_name.split('.json')[0]+'_fwd_20'},{train_20_train_stats_dict['r2']},{train_20_train_stats_dict['median_ae']},{train_20_train_stats_dict['mae']},{train_20_train_stats_dict['rmse']},{train_20_test_stats_dict['r2']},{train_20_test_stats_dict['median_ae']},{train_20_test_stats_dict['mae']},{train_20_test_stats_dict['rmse']}")
                csv_list.append(f"{preds_json_name.split('.json')[0]+'_fwd_40'},{train_40_train_stats_dict['r2']},{train_40_train_stats_dict['median_ae']},{train_40_train_stats_dict['mae']},{train_40_train_stats_dict['rmse']},{train_40_test_stats_dict['r2']},{train_40_test_stats_dict['median_ae']},{train_40_test_stats_dict['mae']},{train_40_test_stats_dict['rmse']}")
                csv_list.append(f"{preds_json_name.split('.json')[0]+'_fwd_60'},{train_60_train_stats_dict['r2']},{train_60_train_stats_dict['median_ae']},{train_60_train_stats_dict['mae']},{train_60_train_stats_dict['rmse']},{train_60_test_stats_dict['r2']},{train_60_test_stats_dict['median_ae']},{train_60_test_stats_dict['mae']},{train_60_test_stats_dict['rmse']}")
                csv_list.append(f"{preds_json_name.split('.json')[0]+'_fwd_80'},{train_80_train_stats_dict['r2']},{train_80_train_stats_dict['median_ae']},{train_80_train_stats_dict['mae']},{train_80_train_stats_dict['rmse']},{train_80_test_stats_dict['r2']},{train_80_test_stats_dict['median_ae']},{train_80_test_stats_dict['mae']},{train_80_test_stats_dict['rmse']}")
                csv_list.append(f"{preds_json_name.split('.json')[0]+'_fwd_100'},{train_100_train_stats_dict['r2']},{train_100_train_stats_dict['median_ae']},{train_100_train_stats_dict['mae']},{train_100_train_stats_dict['rmse']},{train_100_test_stats_dict['r2']},{train_100_test_stats_dict['median_ae']},{train_100_test_stats_dict['mae']},{train_100_test_stats_dict['rmse']}")
                
                ## using mr ea as direct prediction for ar ea
                csv_list.append(f"MR_surrogate_{preds_json_name.split('.json')[0]+'_fwd_20'},{train_20_train_stats_mr_surrogate_dict['r2']},{train_20_train_stats_mr_surrogate_dict['median_ae']},{train_20_train_stats_mr_surrogate_dict['mae']},{train_20_train_stats_mr_surrogate_dict['rmse']},{train_20_test_stats_mr_surrogate_dict['r2']},{train_20_test_stats_mr_surrogate_dict['median_ae']},{train_20_test_stats_mr_surrogate_dict['mae']},{train_20_test_stats_mr_surrogate_dict['rmse']}")
                csv_list.append(f"MR_surrogate_{preds_json_name.split('.json')[0]+'_fwd_40'},{train_40_train_stats_mr_surrogate_dict['r2']},{train_40_train_stats_mr_surrogate_dict['median_ae']},{train_40_train_stats_mr_surrogate_dict['mae']},{train_40_train_stats_mr_surrogate_dict['rmse']},{train_40_test_stats_mr_surrogate_dict['r2']},{train_40_test_stats_mr_surrogate_dict['median_ae']},{train_40_test_stats_mr_surrogate_dict['mae']},{train_40_test_stats_mr_surrogate_dict['rmse']}")
                csv_list.append(f"MR_surrogate_{preds_json_name.split('.json')[0]+'_fwd_60'},{train_60_train_stats_mr_surrogate_dict['r2']},{train_60_train_stats_mr_surrogate_dict['median_ae']},{train_60_train_stats_mr_surrogate_dict['mae']},{train_60_train_stats_mr_surrogate_dict['rmse']},{train_60_test_stats_mr_surrogate_dict['r2']},{train_60_test_stats_mr_surrogate_dict['median_ae']},{train_60_test_stats_mr_surrogate_dict['mae']},{train_60_test_stats_mr_surrogate_dict['rmse']}")
                csv_list.append(f"MR_surrogate_{preds_json_name.split('.json')[0]+'_fwd_80'},{train_80_train_stats_mr_surrogate_dict['r2']},{train_80_train_stats_mr_surrogate_dict['median_ae']},{train_80_train_stats_mr_surrogate_dict['mae']},{train_80_train_stats_mr_surrogate_dict['rmse']},{train_80_test_stats_mr_surrogate_dict['r2']},{train_80_test_stats_mr_surrogate_dict['median_ae']},{train_80_test_stats_mr_surrogate_dict['mae']},{train_80_test_stats_mr_surrogate_dict['rmse']}")
                csv_list.append(f"MR_surrogate_{preds_json_name.split('.json')[0]+'_fwd_100'},{train_100_train_stats_mr_surrogate_dict['r2']},{train_100_train_stats_mr_surrogate_dict['median_ae']},{train_100_train_stats_mr_surrogate_dict['mae']},{train_100_train_stats_mr_surrogate_dict['rmse']},{train_100_test_stats_mr_surrogate_dict['r2']},{train_100_test_stats_mr_surrogate_dict['median_ae']},{train_100_test_stats_mr_surrogate_dict['mae']},{train_100_test_stats_mr_surrogate_dict['rmse']}")
            else:
                csv_list.append(f"{preds_json_name.split('.json')[0]+'_20'},{train_20_train_stats_dict['r2']},{train_20_train_stats_dict['median_ae']},{train_20_train_stats_dict['mae']},{train_20_train_stats_dict['rmse']},{train_20_test_stats_dict['r2']},{train_20_test_stats_dict['median_ae']},{train_20_test_stats_dict['mae']},{train_20_test_stats_dict['rmse']}")
                csv_list.append(f"{preds_json_name.split('.json')[0]+'_40'},{train_40_train_stats_dict['r2']},{train_40_train_stats_dict['median_ae']},{train_40_train_stats_dict['mae']},{train_40_train_stats_dict['rmse']},{train_40_test_stats_dict['r2']},{train_40_test_stats_dict['median_ae']},{train_40_test_stats_dict['mae']},{train_40_test_stats_dict['rmse']}")
                csv_list.append(f"{preds_json_name.split('.json')[0]+'_60'},{train_60_train_stats_dict['r2']},{train_60_train_stats_dict['median_ae']},{train_60_train_stats_dict['mae']},{train_60_train_stats_dict['rmse']},{train_60_test_stats_dict['r2']},{train_60_test_stats_dict['median_ae']},{train_60_test_stats_dict['mae']},{train_60_test_stats_dict['rmse']}")
                csv_list.append(f"{preds_json_name.split('.json')[0]+'_80'},{train_80_train_stats_dict['r2']},{train_80_train_stats_dict['median_ae']},{train_80_train_stats_dict['mae']},{train_80_train_stats_dict['rmse']},{train_80_test_stats_dict['r2']},{train_80_test_stats_dict['median_ae']},{train_80_test_stats_dict['mae']},{train_80_test_stats_dict['rmse']}")
                csv_list.append(f"{preds_json_name.split('.json')[0]+'_100'},{train_100_train_stats_dict['r2']},{train_100_train_stats_dict['median_ae']},{train_100_train_stats_dict['mae']},{train_100_train_stats_dict['rmse']},{train_100_test_stats_dict['r2']},{train_100_test_stats_dict['median_ae']},{train_100_test_stats_dict['mae']},{train_100_test_stats_dict['rmse']}")
                
                ## using mr ea as direct prediction for ar ea
                csv_list.append(f"MR_surrogate_{preds_json_name.split('.json')[0]+'_20'},{train_20_train_stats_mr_surrogate_dict['r2']},{train_20_train_stats_mr_surrogate_dict['median_ae']},{train_20_train_stats_mr_surrogate_dict['mae']},{train_20_train_stats_mr_surrogate_dict['rmse']},{train_20_test_stats_mr_surrogate_dict['r2']},{train_20_test_stats_mr_surrogate_dict['median_ae']},{train_20_test_stats_mr_surrogate_dict['mae']},{train_20_test_stats_mr_surrogate_dict['rmse']}")
                csv_list.append(f"MR_surrogate_{preds_json_name.split('.json')[0]+'_40'},{train_40_train_stats_mr_surrogate_dict['r2']},{train_40_train_stats_mr_surrogate_dict['median_ae']},{train_40_train_stats_mr_surrogate_dict['mae']},{train_40_train_stats_mr_surrogate_dict['rmse']},{train_40_test_stats_mr_surrogate_dict['r2']},{train_40_test_stats_mr_surrogate_dict['median_ae']},{train_40_test_stats_mr_surrogate_dict['mae']},{train_40_test_stats_mr_surrogate_dict['rmse']}")
                csv_list.append(f"MR_surrogate_{preds_json_name.split('.json')[0]+'_60'},{train_60_train_stats_mr_surrogate_dict['r2']},{train_60_train_stats_mr_surrogate_dict['median_ae']},{train_60_train_stats_mr_surrogate_dict['mae']},{train_60_train_stats_mr_surrogate_dict['rmse']},{train_60_test_stats_mr_surrogate_dict['r2']},{train_60_test_stats_mr_surrogate_dict['median_ae']},{train_60_test_stats_mr_surrogate_dict['mae']},{train_60_test_stats_mr_surrogate_dict['rmse']}")
                csv_list.append(f"MR_surrogate_{preds_json_name.split('.json')[0]+'_80'},{train_80_train_stats_mr_surrogate_dict['r2']},{train_80_train_stats_mr_surrogate_dict['median_ae']},{train_80_train_stats_mr_surrogate_dict['mae']},{train_80_train_stats_mr_surrogate_dict['rmse']},{train_80_test_stats_mr_surrogate_dict['r2']},{train_80_test_stats_mr_surrogate_dict['median_ae']},{train_80_test_stats_mr_surrogate_dict['mae']},{train_80_test_stats_mr_surrogate_dict['rmse']}")
                csv_list.append(f"MR_surrogate_{preds_json_name.split('.json')[0]+'_100'},{train_100_train_stats_mr_surrogate_dict['r2']},{train_100_train_stats_mr_surrogate_dict['median_ae']},{train_100_train_stats_mr_surrogate_dict['mae']},{train_100_train_stats_mr_surrogate_dict['rmse']},{train_100_test_stats_mr_surrogate_dict['r2']},{train_100_test_stats_mr_surrogate_dict['median_ae']},{train_100_test_stats_mr_surrogate_dict['mae']},{train_100_test_stats_mr_surrogate_dict['rmse']}")
        
        elif len(preds_dict["0"]) == 1:
            ## full train preds. [100%]
            if ea_dict_name == "combined":
                train_100_k_fold_preds_mr_ar_ea_dict = {fold: {k: v for k, v in zip(preds_dict[fold][0][0], preds_dict[fold][0][1]) if "_rev" not in k} for fold in k_fold_splits_dict_to_use.keys()}
            else:
                train_100_k_fold_preds_mr_ar_ea_dict = {fold: {k: v for k, v in zip(preds_dict[fold][0][0], preds_dict[fold][0][1])} for fold in k_fold_splits_dict_to_use.keys()}
            
            train_100_train_mr_ar_list, train_100_train_ea_mr_list, train_100_train_ea_list, train_100_train_ea_pred_list, train_100_test_mr_ar_list, train_100_test_ea_mr_list, train_100_test_ea_list, train_100_test_ea_pred_list = get_train_test_mr_ar_ea_ea_preds_lists(train_100_k_fold_preds_mr_ar_ea_dict, mr_ar_ea_dict_to_use, k_fold_splits_dict_to_use)
            
            train_100_train_stats_dict = return_stats(train_100_train_ea_list, train_100_train_ea_pred_list)
            train_100_test_stats_dict = return_stats(train_100_test_ea_list, train_100_test_ea_pred_list)
            
            ## using mr ea as direct prediction for ar ea
            train_100_train_stats_mr_surrogate_dict = return_stats(train_100_train_ea_list, train_100_train_ea_mr_list)
            train_100_test_stats_mr_surrogate_dict = return_stats(train_100_test_ea_list, train_100_test_ea_mr_list)
            
            if ea_dict_name == "combined":
                csv_list.append(f"{preds_json_name.split('.json')[0]+'_fwd_100'},{train_100_train_stats_dict['r2']},{train_100_train_stats_dict['median_ae']},{train_100_train_stats_dict['mae']},{train_100_train_stats_dict['rmse']},{train_100_test_stats_dict['r2']},{train_100_test_stats_dict['median_ae']},{train_100_test_stats_dict['mae']},{train_100_test_stats_dict['rmse']}")
                
                ## using mr ea as direct prediction for ar ea
                csv_list.append(f"MR_surrogate_{preds_json_name.split('.json')[0]+'_fwd_100'},{train_100_train_stats_mr_surrogate_dict['r2']},{train_100_train_stats_mr_surrogate_dict['median_ae']},{train_100_train_stats_mr_surrogate_dict['mae']},{train_100_train_stats_mr_surrogate_dict['rmse']},{train_100_test_stats_mr_surrogate_dict['r2']},{train_100_test_stats_mr_surrogate_dict['median_ae']},{train_100_test_stats_mr_surrogate_dict['mae']},{train_100_test_stats_mr_surrogate_dict['rmse']}")
            else:
                csv_list.append(f"{preds_json_name.split('.json')[0]+'_100'},{train_100_train_stats_dict['r2']},{train_100_train_stats_dict['median_ae']},{train_100_train_stats_dict['mae']},{train_100_train_stats_dict['rmse']},{train_100_test_stats_dict['r2']},{train_100_test_stats_dict['median_ae']},{train_100_test_stats_dict['mae']},{train_100_test_stats_dict['rmse']}")
                
                ## using mr ea as direct prediction for ar ea
                csv_list.append(f"MR_surrogate_{preds_json_name.split('.json')[0]+'_100'},{train_100_train_stats_mr_surrogate_dict['r2']},{train_100_train_stats_mr_surrogate_dict['median_ae']},{train_100_train_stats_mr_surrogate_dict['mae']},{train_100_train_stats_mr_surrogate_dict['rmse']},{train_100_test_stats_mr_surrogate_dict['r2']},{train_100_test_stats_mr_surrogate_dict['median_ae']},{train_100_test_stats_mr_surrogate_dict['mae']},{train_100_test_stats_mr_surrogate_dict['rmse']}")
        
        else:
            raise ValueError(f"Invalid number of preds in {preds_json_name}")
        
    
    return csv_list


def make_train_percent_test_accuracy_plot(median_ae_direct_list, mae_direct_list, median_ae_delta_list, mae_delta_list, median_ae_mr_surrogate_list, mae_mr_surrogate_list, save_name):
    
    train_perc_x_list = [1, 2, 3, 4, 5]
    train_perc_x_list_mean = [i-0.15 for i in train_perc_x_list]
    train_perc_x_list_median = [i+0.15 for i in train_perc_x_list]
    train_perc_label_list = ["20", "40", "60", "80", "100"]
    
    plt.clf()
    fig, ax = plt.subplots(figsize=(1.10, 2.25))
    # ax.set_aspect(1)
    
    median_linestyle = "--"
    mean_linestyle = "-"
    
    median_marker = "*"
    mean_marker = "."
    markersize = 6
    linewidth = 1
    
    alpha = 0.8
    
    ax.plot(train_perc_x_list_median, median_ae_mr_surrogate_list, linestyle=median_linestyle, linewidth=linewidth, marker=median_marker, markersize=markersize, markeredgewidth=0, color=vs_colors["orange"][0], label="Median MR Surrogate", alpha=alpha)
    ax.plot(train_perc_x_list_mean, mae_mr_surrogate_list, linestyle=mean_linestyle, linewidth=linewidth, marker=mean_marker, markersize=markersize, markeredgewidth=0, color=vs_colors["orange"][0], label="MAE MR Surrogate", alpha=alpha)
    
    ax.plot(train_perc_x_list_median, median_ae_direct_list, linestyle=median_linestyle, linewidth=linewidth, marker=median_marker, markersize=markersize, markeredgewidth=0, color=vs_colors["red"][0], label="Median Direct", alpha=alpha)
    ax.plot(train_perc_x_list_mean, mae_direct_list, linestyle=mean_linestyle, linewidth=linewidth, marker=mean_marker, markersize=markersize, markeredgewidth=0, color=vs_colors["red"][0], label="MAE Direct", alpha=alpha)
    
    ax.plot(train_perc_x_list_median, median_ae_delta_list, linestyle=median_linestyle, linewidth=linewidth, marker=median_marker, markersize=markersize, markeredgewidth=0, color=vs_colors["purple"][0], label="Median Delta", alpha=alpha)
    ax.plot(train_perc_x_list_mean, mae_delta_list, linestyle=mean_linestyle, linewidth=linewidth, marker=mean_marker, markersize=markersize, markeredgewidth=0, color=vs_colors["purple"][0], label="MAE Delta", alpha=alpha)
    
    ax.grid(which='major', axis='y', color='gray', linestyle='-', linewidth=0.5, alpha=0.5, zorder=0)
    ax.grid(which='minor', axis='y', color='gray', linestyle=':', linewidth=0.5, alpha=0.3, zorder=0)
    ax.set_axisbelow(True)
    ax.minorticks_on()
    ax.tick_params(axis='both', which='both', labelsize=14, direction='in')
    ax.tick_params(axis='x', which='minor', bottom=False)  # Disable minor ticks on x-axis
    ax.tick_params(axis='y', which='minor', left=True)
    # ax.set_title(params["save_name"], fontsize=8)
    
    ax.set_xlim(0.4, 5.5)
    # ax.set_xlabel("Train %", fontsize=14)
    ax.set_xlabel("", fontsize=0)
    ax.set_xticks(train_perc_x_list)
    # ax.set_xticklabels(train_perc_label_list, fontsize=14)
    ax.set_xticklabels(["" for i in train_perc_label_list], fontsize=0)
    
    y_max = 17
    ax.set_ylim(0, y_max)
    # ax.set_ylabel("Absolute Error", fontsize=14)
    ax.set_ylabel("", fontsize=0)
    ax.set_yticks(np.arange(0, y_max + 0.1, 1))
    # ax.set_yticklabels(np.arange(0, y_max + 0.1, 1), fontsize=14)
    ax.set_yticklabels(["" for i in np.arange(0, y_max + 0.1, 1)], fontsize=0)
    
    # legend_handles = [
    #     Line2D([0], [0], linestyle=mean_linestyle, marker=mean_marker, color='none', markeredgewidth=0, markerfacecolor='black', markersize=14, label="MAE", alpha=alpha),
    #     Line2D([0], [0], linestyle=median_linestyle, marker=median_marker, color='none', markeredgewidth=0, markerfacecolor='black', markersize=14, label="MedAE", alpha=alpha),
    #     Line2D([0], [0], linestyle='-', color=vs_colors["orange"][0], linewidth=3, label="MR Surr.", alpha=alpha),
    #     Line2D([0], [0], linestyle='-', color=vs_colors["red"][0], linewidth=3, label="Direct Preds.", alpha=alpha),
    #     Line2D([0], [0], linestyle='-', color=vs_colors["purple"][0], linewidth=3, label="\u0394 Preds.", alpha=alpha)
    # ]
    # ax.legend(handles=legend_handles, loc='upper center', fontsize=14, handlelength=1, labelspacing=0.05, handletextpad=0.15, borderpad=0.15, borderaxespad=0.15, columnspacing=0.5, ncols=5) #, frameon=False
    
    plt.tight_layout()
    plt.savefig(transparent=True, fname=os.path.join(k_fold_accuracy_analysis_dir, f"{save_name}.pdf"), dpi=600, bbox_inches='tight', pad_inches=0.005)
    
    return


def read_csv(csv_name, base_preds_name):
    with open(os.path.join(k_fold_accuracy_analysis_dir, f"{csv_name}.csv"), "r") as f:
        for line in f:
            if f"MR_surrogate_{base_preds_name}" in line:
                ssl = line.strip().split(",")
                if f"MR_surrogate_{base_preds_name}_100" in line:
                    mr_surrogate_median_ae_100 = float(ssl[6])  # median absolute error for test
                    mr_surrogate_mae_100 = float(ssl[7])  # mean absolute error for test
                elif f"MR_surrogate_{base_preds_name}_80" in line:
                    mr_surrogate_median_ae_80 = float(ssl[6])
                    mr_surrogate_mae_80 = float(ssl[7])
                elif f"MR_surrogate_{base_preds_name}_60" in line:
                    mr_surrogate_median_ae_60 = float(ssl[6])
                    mr_surrogate_mae_60 = float(ssl[7])
                elif f"MR_surrogate_{base_preds_name}_40" in line:
                    mr_surrogate_median_ae_40 = float(ssl[6])
                    mr_surrogate_mae_40 = float(ssl[7])
                elif f"MR_surrogate_{base_preds_name}_20" in line:
                    mr_surrogate_median_ae_20 = float(ssl[6])
                    mr_surrogate_mae_20 = float(ssl[7])
                else:
                    # raise ValueError(f"Invalid line {line}")
                    print(f"Invalid line {line}")
            elif f"{base_preds_name}" in line:
                ssl = line.strip().split(",")
                if f"{base_preds_name}_100" in line:
                    median_ae_100 = float(ssl[6])
                    mae_100 = float(ssl[7])
                elif f"{base_preds_name}_80" in line:
                    median_ae_80 = float(ssl[6])
                    mae_80 = float(ssl[7])
                elif f"{base_preds_name}_60" in line:
                    median_ae_60 = float(ssl[6])
                    mae_60 = float(ssl[7])
                elif f"{base_preds_name}_40" in line:
                    median_ae_40 = float(ssl[6])
                    mae_40 = float(ssl[7])
                elif f"{base_preds_name}_20" in line:
                    median_ae_20 = float(ssl[6])
                    mae_20 = float(ssl[7])
                else:
                    # raise ValueError(f"Invalid line {line}")
                    print(f"Invalid line {line}")
            else:
                continue
    median_ae_list = [median_ae_20, median_ae_40, median_ae_60, median_ae_80, median_ae_100]
    mae_list = [mae_20, mae_40, mae_60, mae_80, mae_100]
    median_ae_mr_surrogate_list = [mr_surrogate_median_ae_20, mr_surrogate_median_ae_40, mr_surrogate_median_ae_60, mr_surrogate_median_ae_80, mr_surrogate_median_ae_100]
    mae_mr_surrogate_list = [mr_surrogate_mae_20, mr_surrogate_mae_40, mr_surrogate_mae_60, mr_surrogate_mae_80, mr_surrogate_mae_100]
    
    return median_ae_list, mae_list, median_ae_mr_surrogate_list, mae_mr_surrogate_list


def check_create_dirs():
    if not os.path.exists(k_fold_accuracy_analysis_dir):
        os.makedirs(k_fold_accuracy_analysis_dir)
    return


def main():
    check_create_dirs()
    
    ## load ea dicts
    ar_ea_fwd_dict_path = os.path.join(data_paper_data_dir, "ar_ea_fwd_dict.json")
    ar_ea_fwd_dict = json.load(open(ar_ea_fwd_dict_path, "r"))
    
    ar_ea_rev_dict_path = os.path.join(data_paper_data_dir, "ar_ea_rev_dict.json")
    ar_ea_rev_dict = json.load(open(ar_ea_rev_dict_path, "r"))
    
    mr_ea_fwd_dict_path = os.path.join(data_paper_data_dir, "mr_ea_fwd_dict.json")
    mr_ea_fwd_dict = json.load(open(mr_ea_fwd_dict_path, "r"))
    
    mr_ea_rev_dict_path = os.path.join(data_paper_data_dir, "mr_ea_rev_dict.json")
    mr_ea_rev_dict = json.load(open(mr_ea_rev_dict_path, "r"))
    
    mr_ar_ea_fwd_dict = deepcopy(mr_ea_fwd_dict)
    mr_ar_ea_fwd_dict.update(ar_ea_fwd_dict)
    
    mr_ar_ea_rev_dict = deepcopy(mr_ea_rev_dict)
    mr_ar_ea_rev_dict.update(ar_ea_rev_dict)
    
    mr_ar_ea_combined_dict = deepcopy(mr_ar_ea_fwd_dict)
    mr_ar_ea_combined_dict.update({f"{k}_rev": v for k, v in mr_ar_ea_rev_dict.items()})
    
    mr_ar_ea_name_dict = {
        "fwd": mr_ar_ea_fwd_dict,
        "rev": mr_ar_ea_rev_dict,
        "combined": mr_ar_ea_combined_dict
    }
    
    
    ## load splits
    splits_dir = os.path.join(data_dir, "splits")
    stratified_k_fold_splits_dict = json.load(open(os.path.join(splits_dir, "stratified_k_fold_splits_dict.json"), "r"))
    random_k_fold_splits_dict = json.load(open(os.path.join(splits_dir, "random_k_fold_splits_dict.json"), "r"))
    mr_scaffold_k_fold_splits_dict = json.load(open(os.path.join(splits_dir, "mr_scaffold_k_fold_splits_dict.json"), "r"))
    
    stratified_k_fold_splits_combined_dict = splits_dict_to_combined_splits_dict(stratified_k_fold_splits_dict)
    random_k_fold_splits_combined_dict = splits_dict_to_combined_splits_dict(random_k_fold_splits_dict)
    mr_scaffold_k_fold_splits_combined_dict = splits_dict_to_combined_splits_dict(mr_scaffold_k_fold_splits_dict)
    
    # k_fold_splits_name_dict = {
    #     "stratified": stratified_k_fold_splits_dict,
    #     "random": random_k_fold_splits_dict,
    #     "mr_scaffold": mr_scaffold_k_fold_splits_dict,
    #     "stratified_combined": stratified_k_fold_splits_combined_dict,
    #     "random_combined": random_k_fold_splits_combined_dict,
    #     "mr_scaffold_combined": mr_scaffold_k_fold_splits_combined_dict,
    # }
    
    # preds_dirs_list = [
    #     xgb_direct_preds_dir,
    #     xgb_delta_preds_dir,
    #     chemprop_direct_preds_dir,
    #     chemprop_delta_preds_dir,
    #     chemprop_direct_transfer_preds_dir_1,
    #     chemprop_direct_transfer_preds_dir_2,
    #     chemprop_delta_transfer_preds_dir_1,
    #     chemprop_delta_transfer_preds_dir_2
    # ]
    
    # out_csv_path_list = [os.path.join(k_fold_accuracy_analysis_dir, os.path.basename(preds_dir) + ".csv") for preds_dir in preds_dirs_list]
    
    # for preds_dir, out_csv_path in zip(preds_dirs_list, out_csv_path_list):
    #     csv_list = preds_dir_to_k_fold_test_accuracy_csv_list(preds_dir, mr_ar_ea_name_dict, k_fold_splits_name_dict)
    #     with open(out_csv_path, "w") as f:
    #         f.write("\n".join(csv_list))
    #         f.close()
    
    ## chemprop random split plot
    # train fwd, pred fwd
    median_ae_delta_list, mae_delta_list, median_ae_mr_surrogate_list, mae_mr_surrogate_list = read_csv("chemprop_delta", "random_k_fold_mr_mr_ea_concat_fwd_chemprop_delta_preds")
    median_ae_direct_list, mae_direct_list, _, _ = read_csv("chemprop_direct", "random_k_fold_fwd_chemprop_direct_preds")
    make_train_percent_test_accuracy_plot(median_ae_direct_list, mae_direct_list, median_ae_delta_list, mae_delta_list, median_ae_mr_surrogate_list, mae_mr_surrogate_list, "chemprop_random_split_train_fwd_pred_fwd")
    
    # train fwd, pred rev
    median_ae_delta_list, mae_delta_list, median_ae_mr_surrogate_list, mae_mr_surrogate_list = read_csv("chemprop_delta", "random_k_fold_mr_mr_ea_concat_fwd_chemprop_delta_preds_rev")
    median_ae_direct_list, mae_direct_list, _, _ = read_csv("chemprop_direct", "random_k_fold_fwd_chemprop_direct_preds_rev")
    make_train_percent_test_accuracy_plot(median_ae_direct_list, mae_direct_list, median_ae_delta_list, mae_delta_list, median_ae_mr_surrogate_list, mae_mr_surrogate_list, "chemprop_random_split_train_fwd_pred_rev")
    
    # train combined, pred fwd
    median_ae_delta_list, mae_delta_list, median_ae_mr_surrogate_list, mae_mr_surrogate_list = read_csv("chemprop_delta", "random_k_fold_mr_mr_ea_concat_combined_chemprop_delta_preds_fwd")
    median_ae_direct_list, mae_direct_list, _, _ = read_csv("chemprop_direct", "random_k_fold_combined_chemprop_direct_preds_fwd")
    make_train_percent_test_accuracy_plot(median_ae_direct_list, mae_direct_list, median_ae_delta_list, mae_delta_list, median_ae_mr_surrogate_list, mae_mr_surrogate_list, "chemprop_random_split_train_combined_pred_fwd")
    
    
    ## chemprop stratified split plot
    # train fwd, pred fwd
    median_ae_delta_list, mae_delta_list, median_ae_mr_surrogate_list, mae_mr_surrogate_list = read_csv("chemprop_delta", "stratified_k_fold_mr_mr_ea_concat_fwd_chemprop_delta_preds")
    median_ae_direct_list, mae_direct_list, _, _ = read_csv("chemprop_direct", "stratified_k_fold_fwd_chemprop_direct_preds")
    make_train_percent_test_accuracy_plot(median_ae_direct_list, mae_direct_list, median_ae_delta_list, mae_delta_list, median_ae_mr_surrogate_list, mae_mr_surrogate_list, "chemprop_stratified_split_train_fwd_pred_fwd")
    
    # train fwd, pred rev
    median_ae_delta_list, mae_delta_list, median_ae_mr_surrogate_list, mae_mr_surrogate_list = read_csv("chemprop_delta", "stratified_k_fold_mr_mr_ea_concat_fwd_chemprop_delta_preds_rev")
    median_ae_direct_list, mae_direct_list, _, _ = read_csv("chemprop_direct", "stratified_k_fold_fwd_chemprop_direct_preds_rev")
    make_train_percent_test_accuracy_plot(median_ae_direct_list, mae_direct_list, median_ae_delta_list, mae_delta_list, median_ae_mr_surrogate_list, mae_mr_surrogate_list, "chemprop_stratified_split_train_fwd_pred_rev")
    
    # train combined, pred fwd
    median_ae_delta_list, mae_delta_list, median_ae_mr_surrogate_list, mae_mr_surrogate_list = read_csv("chemprop_delta", "stratified_k_fold_mr_mr_ea_concat_combined_chemprop_delta_preds_fwd")
    median_ae_direct_list, mae_direct_list, _, _ = read_csv("chemprop_direct", "stratified_k_fold_combined_chemprop_direct_preds_fwd")
    make_train_percent_test_accuracy_plot(median_ae_direct_list, mae_direct_list, median_ae_delta_list, mae_delta_list, median_ae_mr_surrogate_list, mae_mr_surrogate_list, "chemprop_stratified_split_train_combined_pred_fwd")
    
    
    ## chemprop mr scaffold split plot
    # train fwd, pred fwd
    median_ae_delta_list, mae_delta_list, median_ae_mr_surrogate_list, mae_mr_surrogate_list = read_csv("chemprop_delta", "mr_scaffold_k_fold_mr_mr_ea_concat_fwd_chemprop_delta_preds")
    median_ae_direct_list, mae_direct_list, _, _ = read_csv("chemprop_direct", "mr_scaffold_k_fold_fwd_chemprop_direct_preds")
    make_train_percent_test_accuracy_plot(median_ae_direct_list, mae_direct_list, median_ae_delta_list, mae_delta_list, median_ae_mr_surrogate_list, mae_mr_surrogate_list, "chemprop_mr_scaffold_split_train_fwd_pred_fwd")
    
    # train fwd, pred rev
    median_ae_delta_list, mae_delta_list, median_ae_mr_surrogate_list, mae_mr_surrogate_list = read_csv("chemprop_delta", "mr_scaffold_k_fold_mr_mr_ea_concat_fwd_chemprop_delta_preds_rev")
    median_ae_direct_list, mae_direct_list, _, _ = read_csv("chemprop_direct", "mr_scaffold_k_fold_fwd_chemprop_direct_preds_rev")
    make_train_percent_test_accuracy_plot(median_ae_direct_list, mae_direct_list, median_ae_delta_list, mae_delta_list, median_ae_mr_surrogate_list, mae_mr_surrogate_list, "chemprop_mr_scaffold_split_train_fwd_pred_rev")
    
    # train combined, pred fwd
    median_ae_delta_list, mae_delta_list, median_ae_mr_surrogate_list, mae_mr_surrogate_list = read_csv("chemprop_delta", "mr_scaffold_k_fold_mr_mr_ea_concat_combined_chemprop_delta_preds_fwd")
    median_ae_direct_list, mae_direct_list, _, _ = read_csv("chemprop_direct", "mr_scaffold_k_fold_combined_chemprop_direct_preds_fwd")
    make_train_percent_test_accuracy_plot(median_ae_direct_list, mae_direct_list, median_ae_delta_list, mae_delta_list, median_ae_mr_surrogate_list, mae_mr_surrogate_list, "chemprop_mr_scaffold_split_train_combined_pred_fwd")
    
    
    ## xgb random split plot
    # train fwd, pred fwd
    median_ae_delta_list, mae_delta_list, median_ae_mr_surrogate_list, mae_mr_surrogate_list = read_csv("xgb_delta", "random_k_fold_mr_mr_ea_concat_fwd_xgb_delta_preds")
    median_ae_direct_list, mae_direct_list, _, _ = read_csv("xgb_direct", "random_k_fold_fwd_xgb_direct_preds")
    make_train_percent_test_accuracy_plot(median_ae_direct_list, mae_direct_list, median_ae_delta_list, mae_delta_list, median_ae_mr_surrogate_list, mae_mr_surrogate_list, "xgb_random_split_train_fwd_pred_fwd")
    
    # train fwd, pred rev
    median_ae_delta_list, mae_delta_list, median_ae_mr_surrogate_list, mae_mr_surrogate_list = read_csv("xgb_delta", "random_k_fold_mr_mr_ea_concat_fwd_xgb_delta_preds_rev")
    median_ae_direct_list, mae_direct_list, _, _ = read_csv("xgb_direct", "random_k_fold_fwd_xgb_direct_preds_rev")
    make_train_percent_test_accuracy_plot(median_ae_direct_list, mae_direct_list, median_ae_delta_list, mae_delta_list, median_ae_mr_surrogate_list, mae_mr_surrogate_list, "xgb_random_split_train_fwd_pred_rev")
    
    # train combined, pred fwd
    median_ae_delta_list, mae_delta_list, median_ae_mr_surrogate_list, mae_mr_surrogate_list = read_csv("xgb_delta", "random_k_fold_mr_mr_ea_concat_combined_xgb_delta_preds_fwd")
    median_ae_direct_list, mae_direct_list, _, _ = read_csv("xgb_direct", "random_k_fold_combined_xgb_direct_preds_fwd")
    make_train_percent_test_accuracy_plot(median_ae_direct_list, mae_direct_list, median_ae_delta_list, mae_delta_list, median_ae_mr_surrogate_list, mae_mr_surrogate_list, "xgb_random_split_train_combined_pred_fwd")
    
    
    ## xgb stratified split plot
    # train fwd, pred fwd
    median_ae_delta_list, mae_delta_list, median_ae_mr_surrogate_list, mae_mr_surrogate_list = read_csv("xgb_delta", "stratified_k_fold_mr_mr_ea_concat_fwd_xgb_delta_preds")
    median_ae_direct_list, mae_direct_list, _, _ = read_csv("xgb_direct", "stratified_k_fold_fwd_xgb_direct_preds")
    make_train_percent_test_accuracy_plot(median_ae_direct_list, mae_direct_list, median_ae_delta_list, mae_delta_list, median_ae_mr_surrogate_list, mae_mr_surrogate_list, "xgb_stratified_split_train_fwd_pred_fwd")
    
    # train fwd, pred rev
    median_ae_delta_list, mae_delta_list, median_ae_mr_surrogate_list, mae_mr_surrogate_list = read_csv("xgb_delta", "stratified_k_fold_mr_mr_ea_concat_fwd_xgb_delta_preds_rev")
    median_ae_direct_list, mae_direct_list, _, _ = read_csv("xgb_direct", "stratified_k_fold_fwd_xgb_direct_preds_rev")
    make_train_percent_test_accuracy_plot(median_ae_direct_list, mae_direct_list, median_ae_delta_list, mae_delta_list, median_ae_mr_surrogate_list, mae_mr_surrogate_list, "xgb_stratified_split_train_fwd_pred_rev")
    
    # train combined, pred fwd
    median_ae_delta_list, mae_delta_list, median_ae_mr_surrogate_list, mae_mr_surrogate_list = read_csv("xgb_delta", "stratified_k_fold_mr_mr_ea_concat_combined_xgb_delta_preds_fwd")
    median_ae_direct_list, mae_direct_list, _, _ = read_csv("xgb_direct", "stratified_k_fold_combined_xgb_direct_preds_fwd")
    make_train_percent_test_accuracy_plot(median_ae_direct_list, mae_direct_list, median_ae_delta_list, mae_delta_list, median_ae_mr_surrogate_list, mae_mr_surrogate_list, "xgb_stratified_split_train_combined_pred_fwd")
    
    
    ## xgb mr scaffold split plot
    # train fwd, pred fwd
    median_ae_delta_list, mae_delta_list, median_ae_mr_surrogate_list, mae_mr_surrogate_list = read_csv("xgb_delta", "mr_scaffold_k_fold_mr_mr_ea_concat_fwd_xgb_delta_preds")
    median_ae_direct_list, mae_direct_list, _, _ = read_csv("xgb_direct", "mr_scaffold_k_fold_fwd_xgb_direct_preds")
    make_train_percent_test_accuracy_plot(median_ae_direct_list, mae_direct_list, median_ae_delta_list, mae_delta_list, median_ae_mr_surrogate_list, mae_mr_surrogate_list, "xgb_mr_scaffold_split_train_fwd_pred_fwd")
    
    # train fwd, pred rev
    median_ae_delta_list, mae_delta_list, median_ae_mr_surrogate_list, mae_mr_surrogate_list = read_csv("xgb_delta", "mr_scaffold_k_fold_mr_mr_ea_concat_fwd_xgb_delta_preds_rev")
    median_ae_direct_list, mae_direct_list, _, _ = read_csv("xgb_direct", "mr_scaffold_k_fold_fwd_xgb_direct_preds_rev")
    make_train_percent_test_accuracy_plot(median_ae_direct_list, mae_direct_list, median_ae_delta_list, mae_delta_list, median_ae_mr_surrogate_list, mae_mr_surrogate_list, "xgb_mr_scaffold_split_train_fwd_pred_rev")
    
    # train combined, pred fwd
    median_ae_delta_list, mae_delta_list, median_ae_mr_surrogate_list, mae_mr_surrogate_list = read_csv("xgb_delta", "mr_scaffold_k_fold_mr_mr_ea_concat_combined_xgb_delta_preds_fwd")
    median_ae_direct_list, mae_direct_list, _, _ = read_csv("xgb_direct", "mr_scaffold_k_fold_combined_xgb_direct_preds_fwd")
    make_train_percent_test_accuracy_plot(median_ae_direct_list, mae_direct_list, median_ae_delta_list, mae_delta_list, median_ae_mr_surrogate_list, mae_mr_surrogate_list, "xgb_mr_scaffold_split_train_combined_pred_fwd")
    
if __name__ == "__main__":
    main()
