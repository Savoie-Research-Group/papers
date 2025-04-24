"""
    Last Modified: 2025/04/04
    Author: Veerupaksh (Veeru) Singla (singla2@purdue.edu)
    Description: train chemprop regressors to delta predict Ea.
"""

import os


this_script_dir = os.path.dirname(os.path.realpath(__file__))
os.chdir(this_script_dir)


import sys
sys.path.append(os.path.join(this_script_dir, ".."))


from utils import *


chemprop_delta_pretrained_models_dir = os.path.join(models_dir, "chemprop_delta")

refine_fn_1 = "wb97xd"
refine_fn_2 = "b2plypd3"

chemprop_delta_transfer_models_dir_1 = os.path.join(models_dir, f"chemprop_delta_transfer_{refine_fn_1}")
chemprop_delta_transfer_models_dir_2 = os.path.join(models_dir, f"chemprop_delta_transfer_{refine_fn_2}")

chemprop_delta_transfer_preds_dir_1 = os.path.join(preds_dir, f"chemprop_delta_transfer_{refine_fn_1}")
chemprop_delta_transfer_preds_dir_2 = os.path.join(preds_dir, f"chemprop_delta_transfer_{refine_fn_2}")


def train_pred_save_delta_transfer_chemprop(chemprop_delta_transfer_models_dir, chemprop_delta_transfer_preds_dir, mr_ar_ea_dict_refined_in, refine_fn,
                                            k_fold_splits_dict, mr_ar_atom_mapped_smi_dict, mr_ar_ea_dict, kw, mr_ar_atom_mapped_smi_dict_extra=None,
                                            mr_ar_ea_dict_extra=None, kw_extra="rev", random_state=random_seed, concat_mr_feature=True, append_mr_ea_feature=False, delta_ea=True):
    if mr_ar_atom_mapped_smi_dict_extra is not None:
        assert mr_ar_ea_dict_extra is not None, "mr_ar_ea_dict_extra should be provided if mr_ar_atom_mapped_smi_dict_extra is not None."
    
    extra_feature_len = 0
    if append_mr_ea_feature:
        extra_feature_len += 1
    
    chemprop_delta_transfer_models_dir_kw = os.path.join(chemprop_delta_transfer_models_dir, f"{kw}_chemprop_delta_transfer_models")
    if not os.path.exists(chemprop_delta_transfer_models_dir_kw):
        os.makedirs(chemprop_delta_transfer_models_dir_kw)
    else:
        shutil.rmtree(chemprop_delta_transfer_models_dir_kw)
        os.makedirs(chemprop_delta_transfer_models_dir_kw)
        
    ## Initialize a dict to store the k-fold predictions.
    k_fold_preds_dict = {}  ## {fold: [[mr_ar_list], [preds]]}, mr_ar_list is for all sets: train, val, test. preds is for all sets: train, val, test in the same order as mr_ar_list.
    k_fold_preds_dict_transfer = {}
    
    if mr_ar_atom_mapped_smi_dict_extra is not None:
        k_fold_preds_dict_extra = {}
    
    mr_ar_ea_dict_refined_delta = create_delta_ea_dict(mr_ar_ea_dict_refined_in)
    mr_ar_ea_dict_refined = {k:v for k, v in mr_ar_ea_dict_refined_in.items() if k in mr_ar_ea_dict_refined_delta}
    
    # train_perc_list = ["20", "40", "60", "80", "100"]
    train_perc_list = ["100"]  ## only refine models trained with 100% train data with transfer learning on the refined data.
    for fold, split in tqdm(k_fold_splits_dict.items(), total=len(k_fold_splits_dict), desc=f"Training k-fold delta transfer chemprop {kw}"):
        # these two only for preds
        val_mr_ar_list = split[1]
        test_mr_ar_list = split[2]
        
        # thse two for training transfer models
        val_mr_ar_list_transfer = [val_mr_ar for val_mr_ar in val_mr_ar_list if val_mr_ar in mr_ar_ea_dict_refined]
        test_mr_ar_list_transfer = [test_mr_ar for test_mr_ar in test_mr_ar_list if test_mr_ar in mr_ar_ea_dict_refined]
        
        for i_train_perc, train_mr_ar_list in enumerate(split[0][-1:]):  ## only train transfer models with 100% train data.
            # train_mr_ar_list only for preds. train_mr_ar_list_transfer only for training transfer models.
            train_mr_ar_list_transfer = [train_mr_ar for train_mr_ar in train_mr_ar_list if train_mr_ar in mr_ar_ea_dict_refined]
            
            # create datapoint lists only for preds
            train_datapoint_list = create_chemprop_reaction_datapoint_list(train_mr_ar_list, mr_ar_atom_mapped_smi_dict, mr_ar_ea_dict,
                                                                        concat_mr_feature=concat_mr_feature, append_mr_ea_feature=append_mr_ea_feature, delta_ea=delta_ea)
            val_datapoint_list = create_chemprop_reaction_datapoint_list(val_mr_ar_list, mr_ar_atom_mapped_smi_dict, mr_ar_ea_dict,
                                                                        concat_mr_feature=concat_mr_feature, append_mr_ea_feature=append_mr_ea_feature, delta_ea=delta_ea)
            test_datapoint_list = create_chemprop_reaction_datapoint_list(test_mr_ar_list, mr_ar_atom_mapped_smi_dict, mr_ar_ea_dict,
                                                                            concat_mr_feature=concat_mr_feature, append_mr_ea_feature=append_mr_ea_feature, delta_ea=delta_ea)
            
            # create datapoint lists for training transfer models
            train_datapoint_list_transfer = create_chemprop_reaction_datapoint_list(train_mr_ar_list_transfer, mr_ar_atom_mapped_smi_dict, mr_ar_ea_dict_refined,
                                                                                    concat_mr_feature=concat_mr_feature, append_mr_ea_feature=append_mr_ea_feature, delta_ea=delta_ea,
                                                                                    mr_ar_ea_dict_extra_for_append_mr_ea_feature=mr_ar_ea_dict
                                                                                    )
            val_datapoint_list_transfer = create_chemprop_reaction_datapoint_list(val_mr_ar_list_transfer, mr_ar_atom_mapped_smi_dict, mr_ar_ea_dict_refined,
                                                                                    concat_mr_feature=concat_mr_feature, append_mr_ea_feature=append_mr_ea_feature, delta_ea=delta_ea,
                                                                                    mr_ar_ea_dict_extra_for_append_mr_ea_feature=mr_ar_ea_dict
                                                                                    )
            test_datapoint_list_transfer = create_chemprop_reaction_datapoint_list(test_mr_ar_list_transfer, mr_ar_atom_mapped_smi_dict, mr_ar_ea_dict_refined,
                                                                                    concat_mr_feature=concat_mr_feature, append_mr_ea_feature=append_mr_ea_feature, delta_ea=delta_ea,
                                                                                    mr_ar_ea_dict_extra_for_append_mr_ea_feature=mr_ar_ea_dict
                                                                                    )
            
            if len(train_datapoint_list) > 2:
                train_datapoint_list = [deepcopy(train_datapoint_list)]
                val_datapoint_list = [deepcopy(val_datapoint_list)]
                test_datapoint_list = [deepcopy(test_datapoint_list)]
                
                train_datapoint_list_transfer = [deepcopy(train_datapoint_list_transfer)]
                val_datapoint_list_transfer = [deepcopy(val_datapoint_list_transfer)]
                test_datapoint_list_transfer = [deepcopy(test_datapoint_list_transfer)]
                
            if mr_ar_atom_mapped_smi_dict_extra is not None:
                mr_ar_list_extra = list(mr_ar_atom_mapped_smi_dict_extra.keys())
                datapoint_list_extra = create_chemprop_reaction_datapoint_list(mr_ar_list_extra, mr_ar_atom_mapped_smi_dict_extra, mr_ar_ea_dict_extra,
                                                                            concat_mr_feature=concat_mr_feature, append_mr_ea_feature=append_mr_ea_feature, delta_ea=delta_ea)
                if len(datapoint_list_extra) > 2:
                    datapoint_list_extra = [deepcopy(datapoint_list_extra)]
                    
            multicomponent=True
            ## preds are lists not np.ndarrays. preds for transfer data only directly obtained here.
            pretrained_checkpoint = os.path.join(chemprop_delta_pretrained_models_dir, f"{kw}_chemprop_delta_models", f"best_{fold}_{train_perc_list[i_train_perc]}.ckpt")
            mpnn_best_checkpoint_transfer, train_preds_transfer, val_preds_transfer, test_preds_transfer = train_chemprop_transfer(pretrained_checkpoint, train_datapoint_list_transfer, val_datapoint_list_transfer, test_datapoint_list_transfer,
                                                                                                                                checkpoint_dir=chemprop_delta_transfer_models_dir_kw, kw=f"{fold}_{train_perc_list[i_train_perc]}",
                                                                                                                                multicomponent=multicomponent, random_state=random_state,
                                                                                                                                patience=20, max_epochs=50,
                                                                                                                                extra_feature_len=extra_feature_len)
            
            # pred original data on transfer model
            train_preds = pred_chemprop(train_datapoint_list, mpnn_best_checkpoint_transfer, multicomponent=multicomponent)
            val_preds = pred_chemprop(val_datapoint_list, mpnn_best_checkpoint_transfer, multicomponent=multicomponent)
            test_preds = pred_chemprop(test_datapoint_list, mpnn_best_checkpoint_transfer, multicomponent=multicomponent)
            
            train_pred_dict = {mr_ar: train_preds[i] for i, mr_ar in enumerate(train_mr_ar_list)}
            val_pred_dict = {mr_ar: val_preds[i] for i, mr_ar in enumerate(val_mr_ar_list)}
            test_pred_dict = {mr_ar: test_preds[i] for i, mr_ar in enumerate(test_mr_ar_list)}
            
            train_pred_dict_transfer = {mr_ar: train_preds_transfer[i] for i, mr_ar in enumerate(train_mr_ar_list_transfer)}
            val_pred_dict_transfer = {mr_ar: val_preds_transfer[i] for i, mr_ar in enumerate(val_mr_ar_list_transfer)}
            test_pred_dict_transfer = {mr_ar: test_preds_transfer[i] for i, mr_ar in enumerate(test_mr_ar_list_transfer)}
            
            if delta_ea:
                train_ea_pred_dict = delta_pred_to_ea_dict(train_pred_dict, mr_ar_ea_dict)  ## ea prediction
                val_ea_pred_dict = delta_pred_to_ea_dict(val_pred_dict, mr_ar_ea_dict)
                test_ea_pred_dict = delta_pred_to_ea_dict(test_pred_dict, mr_ar_ea_dict)
                
                train_ea_pred_dict_transfer = delta_pred_to_ea_dict(train_pred_dict_transfer, mr_ar_ea_dict_refined)  ## ea prediction
                val_ea_pred_dict_transfer = delta_pred_to_ea_dict(val_pred_dict_transfer, mr_ar_ea_dict_refined)
                test_ea_pred_dict_transfer = delta_pred_to_ea_dict(test_pred_dict_transfer, mr_ar_ea_dict_refined)
            else:
                train_ea_pred_dict = train_pred_dict
                val_ea_pred_dict = val_pred_dict
                test_ea_pred_dict = test_pred_dict
                
                train_ea_pred_dict_transfer = train_pred_dict_transfer
                val_ea_pred_dict_transfer = val_pred_dict_transfer
                test_ea_pred_dict_transfer = test_pred_dict_transfer
            
            train_ea_pred = [train_ea_pred_dict[mr_ar] for mr_ar in train_mr_ar_list]
            val_ea_pred = [val_ea_pred_dict[mr_ar] for mr_ar in val_mr_ar_list]
            test_ea_pred = [test_ea_pred_dict[mr_ar] for mr_ar in test_mr_ar_list]
            
            train_ea_pred_transfer = [train_ea_pred_dict_transfer[mr_ar] for mr_ar in train_mr_ar_list_transfer]
            val_ea_pred_transfer = [val_ea_pred_dict_transfer[mr_ar] for mr_ar in val_mr_ar_list_transfer]
            test_ea_pred_transfer = [test_ea_pred_dict_transfer[mr_ar] for mr_ar in test_mr_ar_list_transfer]
            
            if str(fold) not in k_fold_preds_dict:
                k_fold_preds_dict[str(fold)] = []
            # while this will be the zeroth element since we are only running for 100% train data, important to remember. while generally if we run all, this will be the last or the 5th element.
            k_fold_preds_dict[str(fold)].append([train_mr_ar_list + val_mr_ar_list + test_mr_ar_list, train_ea_pred + val_ea_pred + test_ea_pred])
            
            if str(fold) not in k_fold_preds_dict_transfer:
                k_fold_preds_dict_transfer[str(fold)] = []
            # same as before, remember this will be the zeroth element since we are only running for 100% train data. while generally if we run all, this will be the last or the 5th element.
            k_fold_preds_dict_transfer[str(fold)].append([train_mr_ar_list_transfer + val_mr_ar_list_transfer + test_mr_ar_list_transfer, train_ea_pred_transfer + val_ea_pred_transfer + test_ea_pred_transfer])
            
            train_ea = [mr_ar_ea_dict[ar] for ar in train_mr_ar_list]
            val_ea = [mr_ar_ea_dict[ar] for ar in val_mr_ar_list]
            test_ea = [mr_ar_ea_dict[ar] for ar in test_mr_ar_list]
            
            train_ea_transfer = [mr_ar_ea_dict_refined[ar] for ar in train_mr_ar_list_transfer]
            val_ea_transfer = [mr_ar_ea_dict_refined[ar] for ar in val_mr_ar_list_transfer]
            test_ea_transfer = [mr_ar_ea_dict_refined[ar] for ar in test_mr_ar_list_transfer]
            
            print("Train-Test stats for original data:")
            print_train_test_stats(np.array(train_ea + val_ea), np.array(train_ea_pred + val_ea_pred), np.array(test_ea), np.array(test_ea_pred))
            print("Train-Test stats for transfer data:")
            print_train_test_stats(np.array(train_ea_transfer + val_ea_transfer), np.array(train_ea_pred_transfer + val_ea_pred_transfer), np.array(test_ea_transfer), np.array(test_ea_pred_transfer))
            
            if mr_ar_atom_mapped_smi_dict_extra is not None:
                preds_extra = pred_chemprop(datapoint_list_extra, mpnn_best_checkpoint_transfer, multicomponent=multicomponent)
                pred_dict_extra = {mr_ar: preds_extra[i] for i, mr_ar in enumerate(mr_ar_list_extra)}
                
                if delta_ea:
                    ea_pred_dict_extra = delta_pred_to_ea_dict(pred_dict_extra, mr_ar_ea_dict_extra)
                else:
                    ea_pred_dict_extra = pred_dict_extra
                    
                ea_pred_extra = [ea_pred_dict_extra[mr_ar] for mr_ar in mr_ar_list_extra]
                
                if str(fold) not in k_fold_preds_dict_extra:
                    k_fold_preds_dict_extra[str(fold)] = []
                ## again remember that this will be the zeroth element since we are only running for 100% train data. while generally if we run all, this will be the last or the 5th element.
                k_fold_preds_dict_extra[str(fold)].append([mr_ar_list_extra, ea_pred_extra])
                
                ea_extra = [mr_ar_ea_dict_extra[ar] for ar in mr_ar_list_extra]
                
                ## in this case, the whole actual data is train technically. and this extra is the test.
                print("Stats for extra: ")
                print_train_test_stats(np.array(train_ea + val_ea + test_ea), np.array(train_ea_pred + val_ea_pred + test_ea_pred), np.array(ea_extra), np.array(ea_pred_extra))
                
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
    ## Save the k_fold_preds_dict.
    json.dump(k_fold_preds_dict, open(os.path.join(chemprop_delta_transfer_preds_dir, f"{kw}_chemprop_delta_transfer_preds.json"), "w"), indent=4)
    json.dump(k_fold_preds_dict_transfer, open(os.path.join(chemprop_delta_transfer_preds_dir, f"{kw}_chemprop_delta_transfer_preds_{refine_fn}.json"), "w"), indent=4)
    
    if mr_ar_atom_mapped_smi_dict_extra is not None:
        json.dump(k_fold_preds_dict_extra, open(os.path.join(chemprop_delta_transfer_preds_dir, f"{kw}_chemprop_delta_transfer_preds_{kw_extra}.json"), "w"), indent=4)
        return k_fold_preds_dict, k_fold_preds_dict_extra, chemprop_delta_transfer_models_dir_kw
    
    return k_fold_preds_dict, chemprop_delta_transfer_models_dir_kw


def check_create_dirs():
    if not os.path.exists(chemprop_delta_transfer_models_dir_1):
        os.makedirs(chemprop_delta_transfer_models_dir_1)
    if not os.path.exists(chemprop_delta_transfer_models_dir_2):
        os.makedirs(chemprop_delta_transfer_models_dir_2)
    if not os.path.exists(chemprop_delta_transfer_preds_dir_1):
        os.makedirs(chemprop_delta_transfer_preds_dir_1)
    if not os.path.exists(chemprop_delta_transfer_preds_dir_2):
        os.makedirs(chemprop_delta_transfer_preds_dir_2)


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
    
    mr_ar_ea_combined_dict_refined_1 = json.load(open(os.path.join(data_dir, "high_level_single_pt_refinement", "mr_ar_ea_combined_dict_wb97xd.json"), "r"))
    mr_ar_ea_fwd_dict_refined_1 = {k: v for k, v in mr_ar_ea_combined_dict_refined_1.items() if "_rev" not in k}
    mr_ar_ea_rev_dict_refined_1 = {k.split("_rev")[0]: v for k, v in mr_ar_ea_combined_dict_refined_1.items() if "_rev" in k}
    
    mr_ar_ea_combined_dict_refined_2 = json.load(open(os.path.join(data_dir, "high_level_single_pt_refinement", "mr_ar_ea_combined_dict_b2plypd3.json"), "r"))
    mr_ar_ea_fwd_dict_refined_2 = {k: v for k, v in mr_ar_ea_combined_dict_refined_2.items() if "_rev" not in k}
    mr_ar_ea_rev_dict_refined_2 = {k.split("_rev")[0]: v for k, v in mr_ar_ea_combined_dict_refined_2.items() if "_rev" in k}
    
    
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
    ## train with mr graph and mr ea added as features.
    # wb97xd mr_scaffold_k_fold_splits
    _, _, _ = train_pred_save_delta_transfer_chemprop(chemprop_delta_transfer_models_dir_1, chemprop_delta_transfer_preds_dir_1, mr_ar_ea_fwd_dict_refined_1, refine_fn_1,
                                                    mr_scaffold_k_fold_splits_dict, mr_ar_atom_mapped_smi_dict, mr_ar_ea_fwd_dict, "mr_scaffold_k_fold_mr_mr_ea_concat_fwd",
                                                    mr_ar_atom_mapped_smi_dict_extra=mr_ar_atom_mapped_smi_dict_rev, mr_ar_ea_dict_extra=mr_ar_ea_rev_dict, kw_extra="rev",
                                                    concat_mr_feature=True, append_mr_ea_feature=True, delta_ea=True)
    
    # b2plypd3 mr_scaffold_k_fold_splits
    _, _, _ = train_pred_save_delta_transfer_chemprop(chemprop_delta_transfer_models_dir_2, chemprop_delta_transfer_preds_dir_2, mr_ar_ea_fwd_dict_refined_2, refine_fn_2,
                                                    mr_scaffold_k_fold_splits_dict, mr_ar_atom_mapped_smi_dict, mr_ar_ea_fwd_dict, "mr_scaffold_k_fold_mr_mr_ea_concat_fwd",
                                                    mr_ar_atom_mapped_smi_dict_extra=mr_ar_atom_mapped_smi_dict_rev, mr_ar_ea_dict_extra=mr_ar_ea_rev_dict, kw_extra="rev",
                                                    concat_mr_feature=True, append_mr_ea_feature=True, delta_ea=True)
    
    # wb97xd stratified_k_fold_splits
    _, _, _ = train_pred_save_delta_transfer_chemprop(chemprop_delta_transfer_models_dir_1, chemprop_delta_transfer_preds_dir_1, mr_ar_ea_fwd_dict_refined_1, refine_fn_1,
                                                    stratified_k_fold_splits_dict, mr_ar_atom_mapped_smi_dict, mr_ar_ea_fwd_dict, "stratified_k_fold_mr_mr_ea_concat_fwd",
                                                    mr_ar_atom_mapped_smi_dict_extra=mr_ar_atom_mapped_smi_dict_rev, mr_ar_ea_dict_extra=mr_ar_ea_rev_dict, kw_extra="rev",
                                                    concat_mr_feature=True, append_mr_ea_feature=True, delta_ea=True)
    
    # b2plypd3 stratified_k_fold_splits
    _, _, _ = train_pred_save_delta_transfer_chemprop(chemprop_delta_transfer_models_dir_2, chemprop_delta_transfer_preds_dir_2, mr_ar_ea_fwd_dict_refined_2, refine_fn_2,
                                                    stratified_k_fold_splits_dict, mr_ar_atom_mapped_smi_dict, mr_ar_ea_fwd_dict, "stratified_k_fold_mr_mr_ea_concat_fwd",
                                                    mr_ar_atom_mapped_smi_dict_extra=mr_ar_atom_mapped_smi_dict_rev, mr_ar_ea_dict_extra=mr_ar_ea_rev_dict, kw_extra="rev",
                                                    concat_mr_feature=True, append_mr_ea_feature=True, delta_ea=True)
    
    # wb97xd random_k_fold_splits
    _, _, _ = train_pred_save_delta_transfer_chemprop(chemprop_delta_transfer_models_dir_1, chemprop_delta_transfer_preds_dir_1, mr_ar_ea_fwd_dict_refined_1, refine_fn_1,
                                                    random_k_fold_splits_dict, mr_ar_atom_mapped_smi_dict, mr_ar_ea_fwd_dict, "random_k_fold_mr_mr_ea_concat_fwd",
                                                    mr_ar_atom_mapped_smi_dict_extra=mr_ar_atom_mapped_smi_dict_rev, mr_ar_ea_dict_extra=mr_ar_ea_rev_dict, kw_extra="rev",
                                                    concat_mr_feature=True, append_mr_ea_feature=True, delta_ea=True)
    
    # b2plypd3 random_k_fold_splits
    _, _, _ = train_pred_save_delta_transfer_chemprop(chemprop_delta_transfer_models_dir_2, chemprop_delta_transfer_preds_dir_2, mr_ar_ea_fwd_dict_refined_2, refine_fn_2,
                                                    random_k_fold_splits_dict, mr_ar_atom_mapped_smi_dict, mr_ar_ea_fwd_dict, "random_k_fold_mr_mr_ea_concat_fwd",
                                                    mr_ar_atom_mapped_smi_dict_extra=mr_ar_atom_mapped_smi_dict_rev, mr_ar_ea_dict_extra=mr_ar_ea_rev_dict, kw_extra="rev",
                                                    concat_mr_feature=True, append_mr_ea_feature=True, delta_ea=True)
    
    ####################################################################################
    ## Combined direction models training: both forward and reverse trained together. ##
    ####################################################################################
    ## train with mr graph and mr ea added as features.
    # wb97xd mr_scaffold_k_fold_splits
    _, _ = train_pred_save_delta_transfer_chemprop(chemprop_delta_transfer_models_dir_1, chemprop_delta_transfer_preds_dir_1, mr_ar_ea_combined_dict_refined_1, refine_fn_1,
                                                   mr_scaffold_k_fold_splits_combined_dict, mr_ar_atom_mapped_smi_dict_combined, mr_ar_ea_combined_dict, "mr_scaffold_k_fold_mr_mr_ea_concat_combined",
                                                   concat_mr_feature=True, append_mr_ea_feature=True, delta_ea=True)
    # b2plypd3 mr_scaffold_k_fold_splits
    _, _ = train_pred_save_delta_transfer_chemprop(chemprop_delta_transfer_models_dir_2, chemprop_delta_transfer_preds_dir_2, mr_ar_ea_combined_dict_refined_2, refine_fn_2,
                                                   mr_scaffold_k_fold_splits_combined_dict, mr_ar_atom_mapped_smi_dict_combined, mr_ar_ea_combined_dict, "mr_scaffold_k_fold_mr_mr_ea_concat_combined",
                                                   concat_mr_feature=True, append_mr_ea_feature=True, delta_ea=True)
    
    # wb97xd stratified_k_fold_splits
    _, _ = train_pred_save_delta_transfer_chemprop(chemprop_delta_transfer_models_dir_1, chemprop_delta_transfer_preds_dir_1, mr_ar_ea_combined_dict_refined_1, refine_fn_1,
                                                   stratified_k_fold_splits_combined_dict, mr_ar_atom_mapped_smi_dict_combined, mr_ar_ea_combined_dict, "stratified_k_fold_mr_mr_ea_concat_combined",
                                                   concat_mr_feature=True, append_mr_ea_feature=True, delta_ea=True)
    # b2plypd3 stratified_k_fold_splits
    _, _ = train_pred_save_delta_transfer_chemprop(chemprop_delta_transfer_models_dir_2, chemprop_delta_transfer_preds_dir_2, mr_ar_ea_combined_dict_refined_2, refine_fn_2,
                                                   stratified_k_fold_splits_combined_dict, mr_ar_atom_mapped_smi_dict_combined, mr_ar_ea_combined_dict, "stratified_k_fold_mr_mr_ea_concat_combined",
                                                   concat_mr_feature=True, append_mr_ea_feature=True, delta_ea=True)
    
    # wb97xd random_k_fold_splits
    _, _ = train_pred_save_delta_transfer_chemprop(chemprop_delta_transfer_models_dir_1, chemprop_delta_transfer_preds_dir_1, mr_ar_ea_combined_dict_refined_1, refine_fn_1,
                                                   random_k_fold_splits_combined_dict, mr_ar_atom_mapped_smi_dict_combined, mr_ar_ea_combined_dict, "random_k_fold_mr_mr_ea_concat_combined",
                                                   concat_mr_feature=True, append_mr_ea_feature=True, delta_ea=True)
    # b2plypd3 random_k_fold_splits
    _, _ = train_pred_save_delta_transfer_chemprop(chemprop_delta_transfer_models_dir_2, chemprop_delta_transfer_preds_dir_2, mr_ar_ea_combined_dict_refined_2, refine_fn_2,
                                                   random_k_fold_splits_combined_dict, mr_ar_atom_mapped_smi_dict_combined, mr_ar_ea_combined_dict, "random_k_fold_mr_mr_ea_concat_combined",
                                                   concat_mr_feature=True, append_mr_ea_feature=True, delta_ea=True)
    
    return


if __name__ == "__main__":
    main()
