"""
    Date Modified: 2024/01/23
    Author: Veerupaksh (Veeru) Singla (singla2@purdue.edu)
    Description: creating multiple train:test splits for extrapolation testing.
"""


import os
import json
import numpy as np
import pickle
import random
from rdkit import Chem
import random
import multiprocessing


def find_core_alk(alk_smi):
    return "C" * (int(np.max(Chem.GetDistanceMatrix(Chem.MolFromSmiles(alk_smi)))) + 1)


def find_core_branches(alk_smi):
    if "(" not in alk_smi:
        return 0
    
    core_alk = find_core_alk(alk_smi)
    alk_frags = Chem.MolToSmiles(Chem.ReplaceCore(Chem.MolFromSmiles(alk_smi), Chem.MolFromSmiles(core_alk)))
    if alk_frags == "":
        return 0
    
    return alk_frags.count(".") + 1


def find_all_branches(alk_smi):
    if "(" not in alk_smi:
        return 0
    
    core_alk = find_core_alk(alk_smi)
    frags_all = ["C" + smi_frag[4:] for smi_frag in Chem.MolToSmiles(Chem.ReplaceCore(Chem.MolFromSmiles(alk_smi), Chem.MolFromSmiles(core_alk))).split(".")]
    smi_frags_dict = {alk_smi: frags_all}
    for frag in frags_all:
        if "(" in frag:
            if frag in smi_frags_dict:
                frags_all.extend(smi_frags_dict[frag])
            else:
                core_alk = find_core_alk(frag)
                frag_frags = ["C" + smi_frag[4:] for smi_frag in Chem.MolToSmiles(Chem.ReplaceCore(Chem.MolFromSmiles(frag), Chem.MolFromSmiles(core_alk))).split(".")]
                smi_frags_dict[frag] = frag_frags
                frags_all.extend(frag_frags)
    
    return len(frags_all)


def lists_to_data(train_smi_list_in, test_smi_list_in, train_hl_list_in, test_hl_list_in):
    train_len = len(train_smi_list_in)
    test_len = len(test_smi_list_in)

    train_fp_list = np.array([smi_fp_dict[smi] for smi in train_smi_list_in])
    test_fp_list = np.array([smi_fp_dict[smi] for smi in test_smi_list_in])
    
    ## training on log(half-lives) to contain range of data
    train_hl_list = np.log(np.array(train_hl_list_in))
    test_hl_list = np.log(np.array(test_hl_list_in))
    
    ## training log(half-lives) mean and std to z-normalize resulting in mean 0 and std 1
    mean_ = train_hl_list.mean(axis=0)
    std_ = train_hl_list.std(axis=0)

    print(mean_, std_)

    train_hl_list = (train_hl_list - mean_) / std_
    test_hl_list = (test_hl_list - mean_) / std_

    train_hl_list = list(train_hl_list)
    test_hl_list = list(test_hl_list)

    csv_train_val_list = ["smiles,hl"]
    csv_test_val_list = ["smiles,hl"]

    for i in range(len(train_smi_list_in)):
        csv_train_val_list.append(f"{train_smi_list_in[i]},{train_hl_list[i]}")
        
    for i in range(len(test_smi_list_in)):
        csv_test_val_list.append(f"{test_smi_list_in[i]},{test_hl_list[i]}")

    train_hl_list = np.array(train_hl_list)
    test_hl_list = np.array(test_hl_list)

    np.save(os.path.join(save_path, "x_fp_train.npy"), train_fp_list)
    np.save(os.path.join(save_path, "x_fp_test.npy"), test_fp_list)
    np.save(os.path.join(save_path, "y_hl_train.npy"), train_hl_list)
    np.save(os.path.join(save_path, "y_hl_test.npy"), test_hl_list)

    with open(os.path.join(save_path, "train_val_smi_hl.csv"), "w") as f:
        f.write("\n".join(csv_train_val_list))
        f.close()
    
    with open(os.path.join(save_path, "test_smi_hl.csv"), "w") as f:
        f.write("\n".join(csv_test_val_list))
        f.close()

    with open(os.path.join(save_path, "data_info.txt"), "w") as f:
        f.write(f"train_len: {train_len}\ntest_len: {test_len}\nmean: {mean_}\nstd: {std_}")


def main():
    this_script_directory = os.path.dirname(os.path.realpath(__file__))
    os.chdir(this_script_directory)
    
    hl_data_dict_path = "../data/alk_smi_hl_dict_secs_hl_prune_till_c17_32421_vals.json"
    hl_data_dict = json.load(open(hl_data_dict_path, "r"))

    smi_fp_dict_path = "../data/alkanes_till_c17_canon_fp_2048_int8.p"
    smi_fp_dict = pickle.load(open(smi_fp_dict_path, "rb"))


    ### Create splits

    # 0. Random 90:10 Train:Test split
    save_path = "../data/splits/random_90-10_train-test_split"
    
    train_smi_list = []
    test_smi_list = []
    train_hl_list = []
    test_hl_list = []
    
    x_smi_tmp = list(hl_data_dict.keys())
    y_hl_tmp = list(hl_data_dict.values())

    xy_zip_temp = list(zip(x_smi_tmp, y_hl_tmp))
    random.Random(34).shuffle(xy_zip_temp)
    x_smi_tmp, y_hl_tmp = zip(*xy_zip_temp)

    x_smi_tmp = list(x_smi_tmp)
    y_hl_tmp = list(y_hl_tmp)

    len_90pc = int(0.9 * len(x_smi_tmp))

    train_smi_list = x_smi_tmp[:len_90pc]
    test_smi_list = x_smi_tmp[len_90pc:]
    train_hl_list = y_hl_tmp[:len_90pc]
    test_hl_list = y_hl_tmp[len_90pc:]
    
    lists_to_data(train_smi_list, test_smi_list, train_hl_list, test_hl_list)


    # 1. Train till c16, test c17
    save_path = "../data/splits/till_c16_train_c17_test"
    
    train_smi_list = []
    test_smi_list = []
    train_hl_list = []
    test_hl_list = []
    
    for smi, hl in hl_data_dict.items():
        if smi.count("C") < 17:
            train_smi_list.append(smi)
            train_hl_list.append(hl)
        else:
            test_smi_list.append(smi)
            test_hl_list.append(hl)
            
    lists_to_data(train_smi_list, test_smi_list, train_hl_list, test_hl_list)


    # 2. Train 4 or less core branch, test rest
    save_path = "../data/splits/4_or_less_core_branches_train_rest_test"
    
    train_smi_list = []
    test_smi_list = []
    train_hl_list = []
    test_hl_list = []
    
    for smi, hl in hl_data_dict.items():
        if find_core_branches(smi) <= 4:
            train_smi_list.append(smi)
            train_hl_list.append(hl)
        else:
            test_smi_list.append(smi)
            test_hl_list.append(hl)
            
    lists_to_data(train_smi_list, test_smi_list, train_hl_list, test_hl_list)


    # 3. Train 6 or less all branches, test rest
    save_path = "6_or_less_total_branches_train_rest_test"
    
    train_smi_list = []
    test_smi_list = []
    train_hl_list = []
    test_hl_list = []
    
    for smi, hl in hl_data_dict.items():
        if find_all_branches(smi) <= 6:
            train_smi_list.append(smi)
            train_hl_list.append(hl)
        else:
            test_smi_list.append(smi)
            test_hl_list.append(hl)
            
    lists_to_data(train_smi_list, test_smi_list, train_hl_list, test_hl_list)


    # 4. Train backbone <=10, test rest
    save_path = "backbone_smaller_than_equal_10_train_rest_test"
    
    train_smi_list = []
    test_smi_list = []
    train_hl_list = []
    test_hl_list = []
    
    for smi, hl in hl_data_dict.items():
        if len(find_core_alk(smi)) <= 10:
            train_smi_list.append(smi)
            train_hl_list.append(hl)
        else:
            test_smi_list.append(smi)
            test_hl_list.append(hl)
            
    lists_to_data(train_smi_list, test_smi_list, train_hl_list, test_hl_list)


if __name__ == "__main__":
    main()
