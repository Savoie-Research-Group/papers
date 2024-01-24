"""
    Date Modified: 2024/01/23
    Author: Veerupaksh (Veeru) Singla (singla2@purdue.edu)
    Description: calculate pairwise accuracy for all splits for both mlp and chemprop
"""


import os
import multiprocessing as mp
import numpy as np
import json
from tqdm import tqdm


this_script_directory = os.path.dirname(os.path.realpath(__file__))
os.chdir(this_script_directory)


def calc_accuracy(y_hl_train, y_hl_test, y_pred_train, y_pred_test):
    train_total_pairs = 0
    train_correct_pairs = 0
    test_total_pairs = 0
    test_correct_pairs = 0
    
    for i in range(len(y_hl_train) - 1):
        for j in range(i + 1, len(y_hl_train)):
            train_total_pairs += 1
            if (y_hl_train[i] >= y_hl_train[j]) == (y_pred_train[i] >= y_pred_train[j]):
                train_correct_pairs += 1
                
    for i in range(len(y_hl_test) - 1):
        for j in range(i + 1, len(y_hl_test)):
            test_total_pairs += 1
            if (y_hl_test[i] >= y_hl_test[j]) == (y_pred_test[i] >= y_pred_test[j]):
                test_correct_pairs += 1
                
    train_accuracy = train_correct_pairs / train_total_pairs
    test_accuracy = test_correct_pairs / test_total_pairs
    
    return train_accuracy, test_accuracy


def nn_accuracy(nn_kw, nn_data_path, nn_train_path):
    print(f"nn {nn_kw}")
    y_hl_train = [float(i) for i in np.load(f"{nn_data_path}/y_hl_train.npy")]
    y_hl_test = [float(i) for i in np.load(f"{nn_data_path}/y_hl_test.npy")]
    
    y_pred_train = [float(i[0]) for i in np.load(f"{nn_train_path}/y_pred_train.npy")]
    y_pred_test = [float(i[0]) for i in np.load(f"{nn_train_path}/y_pred_test.npy")]
    
    return calc_accuracy(y_hl_train, y_hl_test, y_pred_train, y_pred_test)
                

def cp_accuracy(cp_kw, cp_data_path, cp_train_path):
    print(f"cp {cp_kw}")
    y_hl_train = []
    y_hl_test = []
    
    y_pred_train = []
    y_pred_test = []
    
    with open(f"{cp_data_path}/train_val_smi_hl.csv") as f:
        f.__next__()
        for line in f:
            line = line.split(",")
            y_hl_train.append(float(line[1]))
    
    with open(f"{cp_data_path}/test_smi_hl.csv") as f:
        f.__next__()
        for line in f:
            line = line.split(",")
            y_hl_test.append(float(line[1]))
            
    with open(f"{cp_train_path}/train_val_smi_pred.csv") as f:
        f.__next__()
        for line in f:
            line = line.split(",")
            y_pred_train.append(float(line[1]))
    
    with open(f"{cp_train_path}/test_smi_pred.csv") as f:
        f.__next__()
        for line in f:
            line = line.split(",")
            y_pred_test.append(float(line[1]))
            
    return calc_accuracy(y_hl_train, y_hl_test, y_pred_train, y_pred_test)


def main():
    data_paths = [
        "../data/splits/random_90-10_train-test_split",
        "../data/splits/4_or_less_core_branches_train_rest_test",
        "../data/splits/6_or_less_total_branches_train_rest_test",
        "../data/splits/backbone_smaller_than_equal_10_train_rest_test",
        "../data/splits/till_c16_train_c17_test"
    ]
    
    keywords_list = [
        "random_split",
        "4_or_less_core_branches_train_rest_test",
        "6_or_less_total_branches_train_rest_test",
        "backbone_smaller_than_equal_10_train_rest_test",
        "till_c16_train_c17_test"
    ]
    
    cp_training_paths = [
        "../pretrained_models/chemprop_models/random_90-10_train-test_split",
        "../pretrained_models/chemprop_models/4_or_less_core_branches_train_rest_test",
        "../pretrained_models/chemprop_models/6_or_less_total_branches_train_rest_test",
        "../pretrained_models/chemprop_models/backbone_smaller_than_equal_10_train_rest_test",
        "../pretrained_models/chemprop_models/till_c16_train_c17_test"
    ]
    
    nn_training_paths = [
        "../pretrained_models/mlp_models/random_90-10_train-test_split",
        "../pretrained_models/mlp_models/4_or_less_core_branches_train_rest_test",
        "../pretrained_models/mlp_models/6_or_less_total_branches_train_rest_test",
        "../pretrained_models/mlp_models/backbone_smaller_than_equal_10_train_rest_test",
        "../pretrained_models/mlp_models/till_c16_train_c17_test"
    ]
    
    mp_cores = len(os.sched_getaffinity(0))
    print(f"Using {mp_cores} cores")
    pool = mp.Pool(mp_cores)
    
    cp_accuracy_run = pool.starmap_async(cp_accuracy, zip(keywords_list, data_paths, cp_training_paths))
    nn_accuracy_run = pool.starmap_async(nn_accuracy, zip(keywords_list, data_paths, nn_training_paths))
    
    nn_accuracy_run = nn_accuracy_run.get()
    cp_accuracy_run = cp_accuracy_run.get()
    
    pool.close()
    pool.join()
    
    nn_accuracy_dict = {kw: acc for kw, acc in zip(keywords_list, nn_accuracy_run)}
    cp_accuracy_dict = {kw: acc for kw, acc in zip(keywords_list, cp_accuracy_run)}
    
    with open("../data/mlp_accuracy.json", "w") as f:
        json.dump(nn_accuracy_dict, f)
        
    with open("../data/chemprop_accuracy.json", "w") as f:
        json.dump(cp_accuracy_dict, f)

if __name__ == "__main__":
    main()
