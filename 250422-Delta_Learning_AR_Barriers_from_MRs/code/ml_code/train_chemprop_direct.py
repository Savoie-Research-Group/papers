"""
    Last Modified: 2025/04/04
    Author: Veerupaksh (Veeru) Singla (singla2@purdue.edu)
    Description: train chemprop regressors to directly predict Ea. Baseline to compare with delta models.
"""

import os


this_script_dir = os.path.dirname(os.path.realpath(__file__))
os.chdir(this_script_dir)


import sys
sys.path.append(os.path.join(this_script_dir, ".."))


from utils import *


chemprop_direct_models_dir = os.path.join(models_dir, "chemprop_direct")
chemprop_direct_preds_dir = os.path.join(preds_dir, "chemprop_direct")


def train_pred_save_direct_chemprop(k_fold_splits_dict, mr_ar_atom_mapped_smi_dict, mr_ar_ea_dict, kw, mr_ar_atom_mapped_smi_dict_extra=None,
                                    mr_ar_ea_dict_extra=None, kw_extra="rev", random_state=random_seed, mr_e_rxn_dict=None, mr_e_rxn_dict_extra=None, concat_mr_feature=True,
                                    append_mr_e_rxn_feature=False, append_mr_ea_feature=False, delta_ea=False):
    
    if append_mr_e_rxn_feature:  # (heat or free anergy of reaction)
        assert mr_e_rxn_dict is not None, "mr_e_rxn_dict should be provided if append_mr_e_rxn_feature is True."
    
    if mr_ar_atom_mapped_smi_dict_extra is not None:
        assert mr_ar_ea_dict_extra is not None, "mr_ar_ea_dict_extra should be provided if mr_ar_atom_mapped_smi_dict_extra is not None."
        if append_mr_e_rxn_feature:
            assert mr_e_rxn_dict_extra is not None, "mr_e_rxn_dict_extra should be provided if append_mr_e_rxn_feature is True."
    
    extra_feature_len = 0
    if append_mr_e_rxn_feature:
        extra_feature_len += 1
    if append_mr_ea_feature:
        extra_feature_len += 1
    
    ## Create a dir to save the chemprop models.
    chemprop_direct_models_dir_kw = os.path.join(chemprop_direct_models_dir, f"{kw}_chemprop_direct_models")
    if not os.path.exists(chemprop_direct_models_dir_kw):
        os.makedirs(chemprop_direct_models_dir_kw)
    else:
        shutil.rmtree(chemprop_direct_models_dir_kw)
        os.makedirs(chemprop_direct_models_dir_kw)
        print("chemprop_direct_models_dir_kw dir already exists.")
        
    ## Initialize a dict to store the k-fold predictions.
    k_fold_preds_dict = {}  ## {fold: [[mr_ar_list], [preds]]}, mr_ar_list is for all sets: train, val, test. preds is for all sets: train, val, test in the same order as mr_ar_list.
    
    if mr_ar_atom_mapped_smi_dict_extra is not None:
        k_fold_preds_dict_extra = {}
    
    train_perc_list = ["20", "40", "60", "80", "100"]
    
    ## Loop over each fold in the k-fold splits.
    for fold, split in tqdm(k_fold_splits_dict.items(), total=len(k_fold_splits_dict), desc=f"Training k-fold direct chmemprop {kw}"):
        ## Get the train, val, and test sets for the current fold.
        
        val_mr_ar_list = split[1]
        test_mr_ar_list = split[2]
        
        for i_train_perc, train_mr_ar_list in enumerate(split[0]):
            ## For direct models, no MR info. Only use ar info.
            ## So, filter out the MR info from the train, val, and test sets.
            _, train_ar_list = filter_mr_ar_list(train_mr_ar_list)
            _, val_ar_list = filter_mr_ar_list(val_mr_ar_list)
            _, test_ar_list = filter_mr_ar_list(test_mr_ar_list)
            
            train_datapoint_list = create_chemprop_reaction_datapoint_list(train_ar_list, mr_ar_atom_mapped_smi_dict, mr_ar_ea_dict, mr_e_rxn_dict=mr_e_rxn_dict,
                                                                        concat_mr_feature=concat_mr_feature, append_mr_e_rxn_feature=append_mr_e_rxn_feature, append_mr_ea_feature=append_mr_ea_feature, delta_ea=delta_ea)
            val_datapoint_list = create_chemprop_reaction_datapoint_list(val_ar_list, mr_ar_atom_mapped_smi_dict, mr_ar_ea_dict, mr_e_rxn_dict=mr_e_rxn_dict,
                                                                        concat_mr_feature=concat_mr_feature, append_mr_e_rxn_feature=append_mr_e_rxn_feature, append_mr_ea_feature=append_mr_ea_feature, delta_ea=delta_ea)
            test_datapoint_list = create_chemprop_reaction_datapoint_list(test_ar_list, mr_ar_atom_mapped_smi_dict, mr_ar_ea_dict, mr_e_rxn_dict=mr_e_rxn_dict,
                                                                        concat_mr_feature=concat_mr_feature, append_mr_e_rxn_feature=append_mr_e_rxn_feature, append_mr_ea_feature=append_mr_ea_feature, delta_ea=delta_ea)
            
            if len(train_datapoint_list) > 2:
                train_datapoint_list = [deepcopy(train_datapoint_list)]
                val_datapoint_list = [deepcopy(val_datapoint_list)]
                test_datapoint_list = [deepcopy(test_datapoint_list)]
            
            if mr_ar_atom_mapped_smi_dict_extra is not None:
                mr_ar_list_extra = list(mr_ar_atom_mapped_smi_dict_extra.keys())
                datapoint_list_extra = create_chemprop_reaction_datapoint_list(mr_ar_list_extra, mr_ar_atom_mapped_smi_dict_extra, mr_ar_ea_dict_extra, mr_e_rxn_dict=mr_e_rxn_dict_extra,
                                                                            concat_mr_feature=concat_mr_feature, append_mr_e_rxn_feature=append_mr_e_rxn_feature, append_mr_ea_feature=append_mr_ea_feature, delta_ea=delta_ea)
                if len(datapoint_list_extra) > 2:
                    datapoint_list_extra = [deepcopy(datapoint_list_extra)]
            
            multicomponent=True
            ## preds are lists not np.ndarrays
            if train_perc_list[i_train_perc] == "100":
                patience = 30
                max_epochs = 60
            else:
                patience = 10
                max_epochs = 30
            mpnn_best_checkpoint, train_preds, val_preds, test_preds = train_chemprop(train_datapoint_list, val_datapoint_list, test_datapoint_list,
                                                                checkpoint_dir=chemprop_direct_models_dir_kw, kw=f"{fold}_{train_perc_list[i_train_perc]}",
                                                                multicomponent=multicomponent, random_state=random_state,
                                                                # warmup_epochs=2, patience=30, max_epochs=75,
                                                                warmup_epochs=2, patience=patience, max_epochs=max_epochs,
                                                                extra_feature_len=extra_feature_len
                                                                #    warmup_epochs=0, patience=1, max_epochs=1  ## testing single shot.
                                                                )
            
            if str(fold) not in k_fold_preds_dict:
                k_fold_preds_dict[str(fold)] = []
            k_fold_preds_dict[str(fold)].append([train_ar_list + val_ar_list + test_ar_list, train_preds + val_preds + test_preds])
            
            y_train = [mr_ar_ea_dict[ar] for ar in train_ar_list]
            y_val = [mr_ar_ea_dict[ar] for ar in val_ar_list]
            y_test = [mr_ar_ea_dict[ar] for ar in test_ar_list]
            
            y_train_pred = train_preds
            y_val_pred = val_preds
            y_test_pred = test_preds
            
            print_train_test_stats(np.array(y_train + y_val), np.array(y_train_pred + y_val_pred), np.array(y_test), np.array(y_test_pred))
        
            if mr_ar_atom_mapped_smi_dict_extra is not None:
                y_pred_extra = pred_chemprop(datapoint_list_extra, mpnn_best_checkpoint, multicomponent=multicomponent)
                
                if str(fold) not in k_fold_preds_dict_extra:
                    k_fold_preds_dict_extra[str(fold)] = []
                k_fold_preds_dict_extra[str(fold)].append([mr_ar_list_extra, y_pred_extra])
                
                y_extra = [mr_ar_ea_dict_extra[ar] for ar in mr_ar_list_extra]
                
                ## in this case, the whole actual data is train technically. and this extra is th test.
                print("Stats for extra: ")
                print_train_test_stats(np.array(y_train + y_val + y_test), np.array(y_train_pred + y_val_pred + y_test_pred), np.array(y_extra), np.array(y_pred_extra))
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    ## Save the k_fold_preds_dict.
    json.dump(k_fold_preds_dict, open(os.path.join(chemprop_direct_preds_dir, f"{kw}_chemprop_direct_preds.json"), "w"), indent=4)
    
    if mr_ar_atom_mapped_smi_dict_extra is not None:
        json.dump(k_fold_preds_dict_extra, open(os.path.join(chemprop_direct_preds_dir, f"{kw}_chemprop_direct_preds_{kw_extra}.json"), "w"), indent=4)
        return k_fold_preds_dict, k_fold_preds_dict_extra, chemprop_direct_models_dir_kw
    
    return k_fold_preds_dict, chemprop_direct_models_dir_kw


def check_create_dirs():
    # Check if chemprop_direct_models_dir dir exists. Else create it.
    if not os.path.exists(chemprop_direct_models_dir):
        os.makedirs(chemprop_direct_models_dir)
    else:
        print("chemprop_direct_models_dir dir already exists.")
    
    # Check if chemprop_direct_preds_dir exists. else create it.
    if not os.path.exists(chemprop_direct_preds_dir):
        os.makedirs(chemprop_direct_preds_dir)
    else:
        print("chemprop_direct_preds_dir dir already exists.")
    
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
    
    ## load smiles dicts. use atom mapped for chemprop.
    ar_atom_mapped_smi_dict_path = os.path.join(data_paper_data_dir, "ar_atom_mapped_smi_dict.json")
    ar_atom_mapped_smi_dict = json.load(open(ar_atom_mapped_smi_dict_path, "r"))
    
    mr_atom_mapped_smi_dict_path = os.path.join(data_paper_data_dir, "mr_atom_mapped_smi_dict.json")
    mr_atom_mapped_smi_dict = json.load(open(mr_atom_mapped_smi_dict_path, "r"))
    
    mr_ar_atom_mapped_smi_dict = deepcopy(mr_atom_mapped_smi_dict)
    mr_ar_atom_mapped_smi_dict.update(ar_atom_mapped_smi_dict)
    
    mr_ar_atom_mapped_smi_dict_rev = {}
    for k, v in mr_ar_atom_mapped_smi_dict.items():
        p, r = v.split(">>")
        mr_ar_atom_mapped_smi_dict_rev[k] = f"{r}>>{p}"
        
    mr_ar_atom_mapped_smi_dict_combined = deepcopy(mr_ar_atom_mapped_smi_dict)
    mr_ar_atom_mapped_smi_dict_combined.update({f"{k}_rev": v for k, v in mr_ar_atom_mapped_smi_dict_rev.items()})
    
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
    # train with no additional features.
    _, _, _ = train_pred_save_direct_chemprop(stratified_k_fold_splits_dict, mr_ar_atom_mapped_smi_dict, mr_ar_ea_fwd_dict, "stratified_k_fold_fwd",
                                           mr_ar_atom_mapped_smi_dict_extra=mr_ar_atom_mapped_smi_dict_rev, mr_ar_ea_dict_extra=mr_ar_ea_rev_dict, kw_extra="rev",
                                           concat_mr_feature=False, append_mr_ea_feature=False, delta_ea=False)
    _, _, _ = train_pred_save_direct_chemprop(mr_scaffold_k_fold_splits_dict, mr_ar_atom_mapped_smi_dict, mr_ar_ea_fwd_dict, "mr_scaffold_k_fold_fwd",
                                           mr_ar_atom_mapped_smi_dict_extra=mr_ar_atom_mapped_smi_dict_rev, mr_ar_ea_dict_extra=mr_ar_ea_rev_dict, kw_extra="rev",
                                           concat_mr_feature=False, append_mr_ea_feature=False, delta_ea=False)
    _, _, _ = train_pred_save_direct_chemprop(random_k_fold_splits_dict, mr_ar_atom_mapped_smi_dict, mr_ar_ea_fwd_dict, "random_k_fold_fwd",
                                           mr_ar_atom_mapped_smi_dict_extra=mr_ar_atom_mapped_smi_dict_rev, mr_ar_ea_dict_extra=mr_ar_ea_rev_dict, kw_extra="rev",
                                           concat_mr_feature=False, append_mr_ea_feature=False, delta_ea=False)
    
    ####################################################################################
    ## Combined direction models training: both forward and reverse trained together. ##
    ####################################################################################
    ## train with no additional features.
    _, _ = train_pred_save_direct_chemprop(stratified_k_fold_splits_combined_dict, mr_ar_atom_mapped_smi_dict_combined, mr_ar_ea_combined_dict, "stratified_k_fold_combined",
                                           concat_mr_feature=False, append_mr_ea_feature=False, delta_ea=False)
    _, _ = train_pred_save_direct_chemprop(mr_scaffold_k_fold_splits_combined_dict, mr_ar_atom_mapped_smi_dict_combined, mr_ar_ea_combined_dict, "mr_scaffold_k_fold_combined",
                                           concat_mr_feature=False, append_mr_ea_feature=False, delta_ea=False)
    _, _ = train_pred_save_direct_chemprop(random_k_fold_splits_combined_dict, mr_ar_atom_mapped_smi_dict_combined, mr_ar_ea_combined_dict, "random_k_fold_combined",
                                           concat_mr_feature=False, append_mr_ea_feature=False, delta_ea=False)
    
    return


if __name__ == "__main__":
    main()
