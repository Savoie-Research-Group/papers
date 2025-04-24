"""
    Last Modified: 2025/04/04
    Author: Veerupaksh (Veeru) Singla (singla2@purdue.edu)
    Description: test improvement on an external set from alkane stability score paper. consists only of acyclic hydricarbons but has radicals.
"""


import os


this_script_dir = os.path.dirname(os.path.realpath(__file__))
os.chdir(this_script_dir)


import sys
sys.path.append(os.path.join(this_script_dir, ".."))


from utils import *


alk_pyrolysis_benchmark_dir = os.path.join(data_dir, "case_study_alkane_pyrolysis_mr_benchmark")

chemprop_delta_models_dir = os.path.join(models_dir, "chemprop_delta")
alk_pyrolysis_chemprop_delta_preds_dir = os.path.join(alk_pyrolysis_benchmark_dir, "chemprop_delta_preds")

refine_fn_1 = "wb97xd"
refine_fn_2 = "b2plypd3"
chemprop_delta_transfer_models_dir_1 = os.path.join(models_dir, f"chemprop_delta_transfer_{refine_fn_1}")
chemprop_delta_transfer_models_dir_2 = os.path.join(models_dir, f"chemprop_delta_transfer_{refine_fn_2}")
alk_pyrolysis_chemprop_delta_transfer_preds_dir_1 = os.path.join(alk_pyrolysis_benchmark_dir, f"chemprop_delta_transfer_preds_{refine_fn_1}")
alk_pyrolysis_chemprop_delta_transfer_preds_dir_2 = os.path.join(alk_pyrolysis_benchmark_dir, f"chemprop_delta_transfer_preds_{refine_fn_2}")


def check_create_dirs():
    if not os.path.exists(alk_pyrolysis_chemprop_delta_preds_dir):
        os.makedirs(alk_pyrolysis_chemprop_delta_preds_dir)
    if not os.path.exists(alk_pyrolysis_chemprop_delta_transfer_preds_dir_1):
        os.makedirs(alk_pyrolysis_chemprop_delta_transfer_preds_dir_1)
    if not os.path.exists(alk_pyrolysis_chemprop_delta_transfer_preds_dir_2):
        os.makedirs(alk_pyrolysis_chemprop_delta_transfer_preds_dir_2)
    return


def load_alk_pyrolysis_data(alk_pyrolysis_csv_path):
    ## update mr and ar names to match with the rest of the code. mr name to have 2 underscore. ar name to have 5 underscore.
    # csv header: MR_num,AR_am_smi,MR_am_smi,AR_Ea,MR_Ea
    
    mr_ar_am_smi_dict = {}
    mr_ar_ea_dict = {}
    with open(alk_pyrolysis_csv_path, "r") as f:
        for line in f:
            if "MR_num" in line:
                continue
            ssl = line.strip().split(",")
            mr_name = ssl[0] + "_0"
            ar_num = len({k: v for k, v in mr_ar_am_smi_dict.items() if mr_name in k and k != mr_name})
            ar_name = mr_name + "_" + str(ar_num) + "_0_0"
            
            print(mr_name, ar_name)

            ar_am_smi = ssl[1]
            mr_am_smi = ssl[2]
            ar_ea = float(ssl[3])
            mr_ea = float(ssl[4])
            mr_ar_am_smi_dict[mr_name] = mr_am_smi
            mr_ar_ea_dict[mr_name] = mr_ea
            mr_ar_am_smi_dict[ar_name] = ar_am_smi
            mr_ar_ea_dict[ar_name] = ar_ea
        f.close()
    return mr_ar_am_smi_dict, mr_ar_ea_dict


def pred_chemprop_delta(models_dir, preds_dir, mr_ar_am_smi_dict, mr_ar_ea_dict):
    mr_ar_list = list(mr_ar_am_smi_dict.keys())
    mr_list, ar_list = filter_mr_ar_list(mr_ar_list)
    mr_ea_list = [mr_ar_ea_dict[mr] for mr in mr_list]
    ar_ea_list = [mr_ar_ea_dict[ar] for ar in ar_list]
    
    ## test to see how well direct ea pred using mr ea works.
    ar_mr_ea_list = []
    for ar in ar_list:
        ar_mr = ar_name_to_mr_name(ar)
        ar_mr_ea_list.append(mr_ar_ea_dict[ar_mr])
    
    # print_train_test_stats(np.array(ar_ea_list), np.array(ar_mr_ea_list), np.array(mr_ea_list), np.array(mr_ea_list))
    # R^2 | train: 0.9587457523164064 | test: 1.0
    # Median AE | train: 0.7561375000170756 | test: 0.0
    # MAE | train: 1.013049185740301 | test: 0.0
    # RMSE | train: 1.3816162929669995 | test: 0.0
    
    for models_dir_kw in tqdm(os.listdir(models_dir)):
        print(models_dir_kw)
        if "mr_mr_ea_mr_e_rxn_concat" in models_dir_kw:
            continue
        elif "mr_mr_ea_concat" in models_dir_kw:
            ar_datapoint_list = create_chemprop_reaction_datapoint_list(ar_list,
                                                                        mr_ar_am_smi_dict, mr_ar_ea_dict,
                                                                        concat_mr_feature=True, append_mr_ea_feature=True, delta_ea=True)
            mr_datapoint_list = create_chemprop_reaction_datapoint_list(mr_list,
                                                                        mr_ar_am_smi_dict, mr_ar_ea_dict,
                                                                        concat_mr_feature=True, append_mr_ea_feature=True, delta_ea=True)
        elif "mr_concat" in models_dir_kw:
            ar_datapoint_list = create_chemprop_reaction_datapoint_list(ar_list,
                                                                        mr_ar_am_smi_dict, mr_ar_ea_dict,
                                                                        concat_mr_feature=True, append_mr_ea_feature=False, delta_ea=True)
            mr_datapoint_list = create_chemprop_reaction_datapoint_list(mr_list,
                                                                        mr_ar_am_smi_dict, mr_ar_ea_dict,
                                                                        concat_mr_feature=True, append_mr_ea_feature=False, delta_ea=True)
        else:
            ar_datapoint_list = create_chemprop_reaction_datapoint_list(ar_list,
                                                                        mr_ar_am_smi_dict, mr_ar_ea_dict,
                                                                        concat_mr_feature=False, append_mr_ea_feature=False, delta_ea=True)
            mr_datapoint_list = create_chemprop_reaction_datapoint_list(mr_list,
                                                                        mr_ar_am_smi_dict, mr_ar_ea_dict,
                                                                        concat_mr_feature=False, append_mr_ea_feature=False, delta_ea=True)
        
        if len(ar_datapoint_list) > 2:
            ar_datapoint_list = [deepcopy(ar_datapoint_list)]
        if len(mr_datapoint_list) > 2:
            mr_datapoint_list = [deepcopy(mr_datapoint_list)]
        
        ar_preds_list_list = []  ## since we have 10-fold validation models. will use average prediction later on.
        mr_preds_list_list = []  ## since we have 10-fold validation models. will use average prediction later on.
        for model_ckpt in os.listdir(os.path.join(models_dir, models_dir_kw)):
            if "_100.ckpt" not in model_ckpt:  ## only using models trained on all train data
                continue
            model_ckpt_path = os.path.join(models_dir, models_dir_kw, model_ckpt)
            
            ar_preds = pred_chemprop(ar_datapoint_list, model_ckpt_path, multicomponent=True)  # delta preds. to convert to ea preds.
            mr_preds = pred_chemprop(mr_datapoint_list, model_ckpt_path, multicomponent=True)  # delta preds. to convert to ea preds.
            
            ar_preds_dict = {ar_list[i]: ar_preds[i] for i in range(len(ar_list))}
            mr_preds_dict = {mr_list[i]: mr_preds[i] for i in range(len(mr_list))}
            
            ar_ea_preds_dict = delta_pred_to_ea_dict(ar_preds_dict, mr_ar_ea_dict)
            mr_ea_preds_dict = delta_pred_to_ea_dict(mr_preds_dict, mr_ar_ea_dict)
            
            ar_ea_preds = [ar_ea_preds_dict[ar] for ar in ar_list]
            mr_ea_preds = [mr_ea_preds_dict[mr] for mr in mr_list]
            
            ar_preds_list_list.append(ar_ea_preds)
            mr_preds_list_list.append(mr_ea_preds)
            
        # save all preds for later use so error bars can be calculated.
        print(len(ar_preds_list_list), len(mr_preds_list_list))
        preds_save_path = os.path.join(preds_dir, f"{models_dir_kw.replace('_models', '_preds')}.json")
        json.dump([ar_list, ar_preds_list_list, mr_list, mr_preds_list_list], open(preds_save_path, 'w'), indent=4)
        
        # for the purpose of printing accuracy here, take average across axis 0.
        ar_preds_avg = np.mean(np.array(ar_preds_list_list), axis=0)
        mr_preds_avg = np.mean(np.array(mr_preds_list_list), axis=0)
        
        # repurpose the train_test accuracy stats function to predict ar and mr prediction accuracy on the external set. here both ar and mr are external.
        print_train_test_stats(np.array(ar_ea_list), ar_preds_avg, np.array(mr_ea_list), mr_preds_avg)
        
    return


def main():
    check_create_dirs()
    
    # alk_pyrolysis_csv_path = os.path.join(alk_pyrolysis_benchmark_dir, "alkane_pyrolysis_mr_benchmark.csv")
    # mr_ar_am_smi_dict, mr_ar_ea_dict = load_alk_pyrolysis_data(alk_pyrolysis_csv_path)
    # json.dump(mr_ar_am_smi_dict, open(os.path.join(alk_pyrolysis_benchmark_dir, "mr_ar_am_smi_dict.json"), 'w'), indent=4)
    # json.dump(mr_ar_ea_dict, open(os.path.join(alk_pyrolysis_benchmark_dir, "mr_ar_ea_dict.json"), 'w'), indent=4)
    
    mr_ar_am_smi_dict = json.load(open(os.path.join(alk_pyrolysis_benchmark_dir, "mr_ar_am_smi_dict.json"), "r"))
    mr_ar_ea_dict = json.load(open(os.path.join(alk_pyrolysis_benchmark_dir, "mr_ar_ea_dict.json"), "r"))
    
    pred_chemprop_delta(chemprop_delta_models_dir, alk_pyrolysis_chemprop_delta_preds_dir, mr_ar_am_smi_dict, mr_ar_ea_dict)
    
    pred_chemprop_delta(chemprop_delta_transfer_models_dir_1, alk_pyrolysis_chemprop_delta_transfer_preds_dir_1, mr_ar_am_smi_dict, mr_ar_ea_dict)
    
    pred_chemprop_delta(chemprop_delta_transfer_models_dir_2, alk_pyrolysis_chemprop_delta_transfer_preds_dir_2, mr_ar_am_smi_dict, mr_ar_ea_dict)
    
    return


if __name__ == "__main__":
    main()
