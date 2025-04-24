"""
    Last Modified: 2025/04/04
    Author: Veerupaksh (Veeru) Singla (singla2@purdue.edu)
    Description: train xgb regressors to delta predict Ea. Baseline to compare with delta models.
"""

import os


this_script_dir = os.path.dirname(os.path.realpath(__file__))
os.chdir(this_script_dir)


import sys
sys.path.append(os.path.join(this_script_dir, ".."))


from utils import *


xgb_delta_models_dir = os.path.join(models_dir, "xgb_delta")
xgb_delta_preds_dir = os.path.join(preds_dir, "xgb_delta")


def train_pred_save_delta_xgb(k_fold_splits_dict, mr_ar_fp_dict, mr_ar_ea_dict, kw, mr_ar_fp_dict_extra=None,
                                mr_ar_ea_dict_extra=None, kw_extra="rev", random_state=random_seed, params=None):
    if mr_ar_fp_dict_extra is not None:
        assert mr_ar_ea_dict_extra is not None, "mr_ar_ea_dict_extra should be provided if mr_ar_fp_dict_extra is not None."
    
    ## Create a dir to save the xgb models.
    xgb_delta_models_dir_kw = os.path.join(xgb_delta_models_dir, f"{kw}_xgb_delta_models")
    if not os.path.exists(xgb_delta_models_dir_kw):
        os.makedirs(xgb_delta_models_dir_kw)
    else:
        print("xgb_delta_models_dir_kw dir already exists.")
    
    mr_ar_delta_ea_dict = create_delta_ea_dict(mr_ar_ea_dict)  # delta_ea = ar_ea - mr_ea. ar_ea = mr_ea + delta_ea
    
    ## Initialize a dict to store the k-fold predictions.
    k_fold_preds_dict = {}  ## {fold: [[[mr_ar_list_train_perc_0], [preds_train_perc_0]], [[mr_ar_list_train_perc_1], [preds_train_perc_1]], ...]}, mr_ar_list is for all sets: train, val, test. preds is for all sets: train, val, test in the same order as mr_ar_list.
    
    if mr_ar_fp_dict_extra is not None:
        k_fold_preds_dict_extra = {}
    
    train_perc_list = ["20", "40", "60", "80", "100"]
    ## Loop over each fold in the k-fold splits.
    for fold, split in tqdm(k_fold_splits_dict.items(), total=len(k_fold_splits_dict), desc=f"Training k-fold delta xgb {kw}"):
        ## Get the train, val, and test sets for the current fold.
        
        val_mr_ar_list = split[1]
        test_mr_ar_list = split[2]
        for i_train_perc, train_mr_ar_list in enumerate(split[0]):
            ## Add the val set to the train set for xgb.
            train_mr_ar_list.extend(val_mr_ar_list)
        
            ## Convert the train and test sets to numpy arrays.
            X_train = np.array([mr_ar_fp_dict[mr_ar] for mr_ar in train_mr_ar_list])
            y_train = np.array([mr_ar_delta_ea_dict[mr_ar] for mr_ar in train_mr_ar_list])  ## delta to train
            y_train_ea = np.array([mr_ar_ea_dict[mr_ar] for mr_ar in train_mr_ar_list])  ## ea to test
            
            X_test = np.array([mr_ar_fp_dict[mr_ar] for mr_ar in test_mr_ar_list])
            y_test = np.array([mr_ar_delta_ea_dict[mr_ar] for mr_ar in test_mr_ar_list])  ## not used
            y_test_ea = np.array([mr_ar_ea_dict[mr_ar] for mr_ar in test_mr_ar_list])  ## ea to test
        
            if mr_ar_fp_dict_extra is not None:
                mr_ar_list_extra = list(mr_ar_fp_dict_extra.keys())
                X_extra = np.array([mr_ar_fp_dict_extra[mr_ar] for mr_ar in mr_ar_list_extra])
                y_ea_extra = np.array([mr_ar_ea_dict_extra[mr_ar] for mr_ar in mr_ar_list_extra])
        
            ## Train an xgb regressor.
            xgb_reg_delta, y_train_pred, y_test_pred = train_pred_xgb_regressor(X_train, y_train, X_test, y_test, random_state=random_state, params=params)
        
            y_train_pred = y_train_pred.tolist()
            y_test_pred = y_test_pred.tolist()
            y_train_pred_dict = {mr_ar: y_train_pred[i] for i, mr_ar in enumerate(train_mr_ar_list)}  ## delta prediction
            y_test_pred_dict = {mr_ar: y_test_pred[i] for i, mr_ar in enumerate(test_mr_ar_list)}
            
            y_train_ea_pred_dict = delta_pred_to_ea_dict(y_train_pred_dict, mr_ar_ea_dict)  ## ea prediction
            y_test_ea_pred_dict = delta_pred_to_ea_dict(y_test_pred_dict, mr_ar_ea_dict)
            y_train_ea_pred = np.array([y_train_ea_pred_dict[mr_ar] for mr_ar in train_mr_ar_list])
            y_test_ea_pred = np.array([y_test_ea_pred_dict[mr_ar] for mr_ar in test_mr_ar_list])
        
            ## Save the xgb model.
            xgb_reg_delta.save_model(os.path.join(xgb_delta_models_dir_kw, f"{fold}_{train_perc_list[i_train_perc]}.json"))
        
            ## Print train and test stats.
            print_train_test_stats(y_train_ea, y_train_ea_pred, y_test_ea, y_test_ea_pred)
        
            ## Store the predictions in the k_fold_preds_dict.
            if str(fold) not in k_fold_preds_dict:
                k_fold_preds_dict[str(fold)] = []
            k_fold_preds_dict[str(fold)].append([train_mr_ar_list + test_mr_ar_list, y_train_ea_pred.tolist() + y_test_ea_pred.tolist()])
        
            if mr_ar_fp_dict_extra is not None:
                y_extra_pred = pred_xgb_regressor(X_extra, xgb_reg_delta)
                y_extra_pred_dict = {mr_ar: y_extra_pred[i] for i, mr_ar in enumerate(mr_ar_list_extra)}
                y_ea_extra_pred_dict = delta_pred_to_ea_dict(y_extra_pred_dict, mr_ar_ea_dict_extra)
                y_ea_extra_pred = np.array([y_ea_extra_pred_dict[mr_ar] for mr_ar in mr_ar_list_extra])
                
                ## in this case, the whole actual data is train technically. and this extra is the test.
                print("Stats for extra: ")
                print_train_test_stats(np.concatenate((y_train_ea, y_test_ea)), np.concatenate((y_train_ea_pred, y_test_ea_pred)), y_ea_extra, y_ea_extra_pred)
                
                if str(fold) not in k_fold_preds_dict_extra:
                    k_fold_preds_dict_extra[str(fold)] = []
                k_fold_preds_dict_extra[str(fold)].append([mr_ar_list_extra, y_ea_extra_pred.tolist()])
            
            ## free memory since xgb doesnt do it automatically when training on gpu
            del xgb_reg_delta
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    ## Save the k_fold_preds_dict.
    json.dump(k_fold_preds_dict, open(os.path.join(xgb_delta_preds_dir, f"{kw}_xgb_delta_preds.json"), "w"), indent=4)
    
    if mr_ar_fp_dict_extra is not None:
        json.dump(k_fold_preds_dict_extra, open(os.path.join(xgb_delta_preds_dir, f"{kw}_xgb_delta_preds_{kw_extra}.json"), "w"), indent=4)
        return k_fold_preds_dict, k_fold_preds_dict_extra, xgb_delta_models_dir_kw
    
    return k_fold_preds_dict, xgb_delta_models_dir_kw


def check_create_dirs():
    # Check if xgb_delta_models_dir dir exists. Else create it.
    if not os.path.exists(xgb_delta_models_dir):
        os.makedirs(xgb_delta_models_dir)
    else:
        print("xgb_delta_models_dir dir already exists.")
        
    # Check if xgb_delta_preds_dir exists. else create it.
    if not os.path.exists(xgb_delta_preds_dir):
        os.makedirs(xgb_delta_preds_dir)
    else:
        print("xgb_delta_preds_dir dir already exists.")
        
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
    
    
    ## load rxn energy dicts (h, g).
    _, _, _, mr_ar_grxn_fwd_dict, mr_ar_grxn_rev_dict, mr_ar_grxn_combined_dict = get_default_g_mr_ar_dicts()
    
    
    ## load fps
    fps_dir = os.path.join(data_dir, "fps")
    
    mr_ar_drfp_dict_fwd_path = os.path.join(fps_dir, "mr_ar_fwd_drfp_dict.pkl")
    mr_ar_drfp_dict_fwd = pickle.load(open(mr_ar_drfp_dict_fwd_path, "rb"))
    
    mr_ar_drfp_dict_rev_path = os.path.join(fps_dir, "mr_ar_rev_drfp_dict.pkl")
    mr_ar_drfp_dict_rev = pickle.load(open(mr_ar_drfp_dict_rev_path, "rb"))
    
    mr_ar_drfp_dict_combined = deepcopy(mr_ar_drfp_dict_fwd)
    mr_ar_drfp_dict_combined.update({f"{k}_rev": v for k, v in mr_ar_drfp_dict_rev.items()})
    
    mr_ts_e3fp_dict_path = os.path.join(fps_dir, "mr_ts_e3fp_dict.pkl")
    mr_ts_e3fp_dict = pickle.load(open(mr_ts_e3fp_dict_path, "rb"))
    
    mr_ts_geofp_dict_path = os.path.join(fps_dir, "mr_ts_geofp_dict.pkl")  # custom mr ts geo fp
    mr_ts_geofp_dict = pickle.load(open(mr_ts_geofp_dict_path, "rb"))
    
    mr_ar_mr_concat_drfp_dict_fwd = create_mr_concat_fp_dict(mr_ar_drfp_dict_fwd)  ## mr fp concatenated to ar fp
    mr_ar_mr_concat_drfp_dict_rev = create_mr_concat_fp_dict(mr_ar_drfp_dict_rev)
    mr_ar_mr_concat_drfp_dict_combined = deepcopy(mr_ar_mr_concat_drfp_dict_fwd)
    mr_ar_mr_concat_drfp_dict_combined.update({f"{k}_rev": v for k, v in mr_ar_mr_concat_drfp_dict_rev.items()})
    
    mr_ar_mr_mr_ea_concat_drfp_dict_fwd = create_mr_ea_concat_fp_dict(mr_ar_mr_concat_drfp_dict_fwd, mr_ar_ea_fwd_dict)  ## mr fp and mr ea concatenated to ar fp
    mr_ar_mr_mr_ea_concat_drfp_dict_rev = create_mr_ea_concat_fp_dict(mr_ar_mr_concat_drfp_dict_rev, mr_ar_ea_rev_dict)
    mr_ar_mr_mr_ea_concat_drfp_dict_combined = deepcopy(mr_ar_mr_mr_ea_concat_drfp_dict_fwd)
    mr_ar_mr_mr_ea_concat_drfp_dict_combined.update({f"{k}_rev": v for k, v in mr_ar_mr_mr_ea_concat_drfp_dict_rev.items()})
    
    mr_ar_mr_mr_ea_mr_e_rxn_concat_drfp_dict_fwd = create_mr_ea_concat_fp_dict(mr_ar_mr_mr_ea_concat_drfp_dict_fwd, mr_ar_grxn_fwd_dict)  ## mr fp, mr ea, and mr e_rxn concatenated to ar fp
    mr_ar_mr_mr_ea_mr_e_rxn_concat_drfp_dict_rev = create_mr_ea_concat_fp_dict(mr_ar_mr_mr_ea_concat_drfp_dict_rev, mr_ar_grxn_rev_dict)
    mr_ar_mr_mr_ea_mr_e_rxn_concat_drfp_dict_combined = deepcopy(mr_ar_mr_mr_ea_mr_e_rxn_concat_drfp_dict_fwd)
    mr_ar_mr_mr_ea_mr_e_rxn_concat_drfp_dict_combined.update({f"{k}_rev": v for k, v in mr_ar_mr_mr_ea_mr_e_rxn_concat_drfp_dict_rev.items()})
    
    
    ## load splits
    splits_dir = os.path.join(data_dir, "splits")
    stratified_k_fold_splits_dict = json.load(open(os.path.join(splits_dir, "stratified_k_fold_splits_dict.json"), "r"))
    random_k_fold_splits_dict = json.load(open(os.path.join(splits_dir, "random_k_fold_splits_dict.json"), "r"))
    mr_scaffold_k_fold_splits_dict = json.load(open(os.path.join(splits_dir, "mr_scaffold_k_fold_splits_dict.json"), "r"))
    
    stratified_k_fold_splits_combined_dict = splits_dict_to_combined_splits_dict(stratified_k_fold_splits_dict)
    random_k_fold_splits_combined_dict = splits_dict_to_combined_splits_dict(random_k_fold_splits_dict)
    mr_scaffold_k_fold_splits_combined_dict = splits_dict_to_combined_splits_dict(mr_scaffold_k_fold_splits_dict)
    
    ###############################################################
    ## Single direction models training: forward reactions only. ##
    ###############################################################
    ## train with drfp and mr drfp and mr ea concatenated
    _, _, _ = train_pred_save_delta_xgb(mr_scaffold_k_fold_splits_dict, mr_ar_mr_mr_ea_concat_drfp_dict_fwd, mr_ar_ea_fwd_dict, "mr_scaffold_k_fold_mr_mr_ea_concat_fwd", mr_ar_fp_dict_extra=mr_ar_mr_mr_ea_concat_drfp_dict_rev, mr_ar_ea_dict_extra=mr_ar_ea_rev_dict, kw_extra="rev")
    _, _, _ = train_pred_save_delta_xgb(stratified_k_fold_splits_dict, mr_ar_mr_mr_ea_concat_drfp_dict_fwd, mr_ar_ea_fwd_dict, "stratified_k_fold_mr_mr_ea_concat_fwd", mr_ar_fp_dict_extra=mr_ar_mr_mr_ea_concat_drfp_dict_rev, mr_ar_ea_dict_extra=mr_ar_ea_rev_dict, kw_extra="rev")
    _, _, _ = train_pred_save_delta_xgb(random_k_fold_splits_dict, mr_ar_mr_mr_ea_concat_drfp_dict_fwd, mr_ar_ea_fwd_dict, "random_k_fold_mr_mr_ea_concat_fwd", mr_ar_fp_dict_extra=mr_ar_mr_mr_ea_concat_drfp_dict_rev, mr_ar_ea_dict_extra=mr_ar_ea_rev_dict, kw_extra="rev")
    
    ####################################################################################
    ## Combined direction models training: both forward and reverse trained together. ##
    ####################################################################################
    ## train with drfp and mr drfp and mr ea concatenated
    _, _ = train_pred_save_delta_xgb(mr_scaffold_k_fold_splits_combined_dict, mr_ar_mr_mr_ea_concat_drfp_dict_combined, mr_ar_ea_combined_dict, "mr_scaffold_k_fold_mr_mr_ea_concat_combined")
    _, _ = train_pred_save_delta_xgb(stratified_k_fold_splits_combined_dict, mr_ar_mr_mr_ea_concat_drfp_dict_combined, mr_ar_ea_combined_dict, "stratified_k_fold_mr_mr_ea_concat_combined")
    _, _ = train_pred_save_delta_xgb(random_k_fold_splits_combined_dict, mr_ar_mr_mr_ea_concat_drfp_dict_combined, mr_ar_ea_combined_dict, "random_k_fold_mr_mr_ea_concat_combined")
    
    return


if __name__ == "__main__":
    main()
