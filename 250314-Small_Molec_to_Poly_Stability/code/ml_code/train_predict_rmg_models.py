"""
    Date Modified: 2024/10/24
    Author: Veerupaksh (Veeru) Singla (singla2@purdue.edu)
    Corresponding Author: Brett M Savoie (bsavoie2@nd.edu)
    Description: Training stability models on active sampled data using rmg.
"""


import os
import numpy as np
import sys
import json
import pickle
from tqdm import tqdm


this_script_dir = os.path.dirname(os.path.realpath(__file__))
os.chdir(this_script_dir)


sys.path.append(this_script_dir)
from ml_utils import *


fp_radius = 2
fp_size = 2048

data_path_main = os.path.join(this_script_dir, "../../data")
rmg_data_path = os.path.join(data_path_main, "rmg_active_learning_data")
active_iteration_models_path = os.path.join(rmg_data_path, "trained_models_active_iterations_cummulative")
expt_small_molec_data_dir = os.path.join(data_path_main, "expt_small_molecule_decomp_temp_data")
expt_polymer_data_dir = os.path.join(data_path_main, "expt_polymer_decomp_temp_data")


def main():
    train()
    
    predict_expt_small_molec_data()
    get_accuracy_expt_small_molec_data()
    
    predict_expt_polymer_data()
    get_accuracy_expt_polymer_data()
    return


def train():
    sampled_smi_hl_dict_path = os.path.join(rmg_data_path, "sampled_smi_hl_dict.json")
    smi_hl_dict = json.load(open(sampled_smi_hl_dict_path, "r"))
    active_iterations_smi_lists_dir = os.path.join(rmg_data_path, "active_iterations_smi_lists")
    smi_fp_dict = gen_uint8_fp_from_smiles_list_parallel(list(smi_hl_dict.keys()), fp_radius=fp_radius, fp_size=fp_size)
    iteration_list = list(range(0, 11))
    
    iteration_smi_list_dict = {}
    for iteration in iteration_list:
        iter_smi_list_path = os.path.join(active_iterations_smi_lists_dir, f"iteration_{iteration}_smi_list.txt")
        iter_smi_list = []
        with open(iter_smi_list_path, "r") as f:
            for line in f:
                smi = line.strip()
                if smi in smi_hl_dict:
                    iter_smi_list.append(smi)
        iteration_smi_list_dict[iteration] = iter_smi_list
        
    iteration_fp_list_dict_cumm = {}
    for iteration in iteration_list:
        iter_fp_list = [smi_fp_dict[smi] for smi in iteration_smi_list_dict[iteration]]
        if iteration == 0:
            iter_fp_list_cumm = iter_fp_list
        else:
            iter_fp_list_cumm = iteration_fp_list_dict_cumm[iteration-1] + iter_fp_list
        iteration_fp_list_dict_cumm[iteration] = iter_fp_list_cumm

    iteration_hl_list_dict_cumm = {}
    for iteration in iteration_list:
        iter_hl_list = [smi_hl_dict[smi] for smi in iteration_smi_list_dict[iteration]]
        if iteration == 0:
            iter_hl_list_cumm = iter_hl_list
        else:
            iter_hl_list_cumm = iteration_hl_list_dict_cumm[iteration-1] + iter_hl_list
        iteration_hl_list_dict_cumm[iteration] = iter_hl_list_cumm
    
    for iteration in iteration_list:
        iter_model_dir = os.path.join(active_iteration_models_path, f"iteration_{iteration}")
        os.makedirs(iter_model_dir, exist_ok=True)
        
        print(f"Training model for iteration {iteration}")
        model = FFNN1(input_size=2048, hidden_sizes=[300, 300, 300, 300, 300], output_size=1)
        X = np.array(iteration_fp_list_dict_cumm[iteration])
        y = np.array(iteration_hl_list_dict_cumm[iteration])
        print(X.shape, y.shape)
        train_save_k_fold_model(
            model=model,
            save_dir=iter_model_dir,
            X=X,
            y=y,
            k=10,
            batch_size=128,
            learning_rate=0.001,
            max_epochs=600,
            patience=300,
            warmup_epochs=0
        )
    return


def predict_expt_small_molec_data():
    preds_rmg_active_learning_iterations_models_dir = os.path.join(expt_small_molec_data_dir, "preds_rmg_active_learning_iterations_models")
    smi_decomp_temp_dict_path = os.path.join(expt_small_molec_data_dir, "smi_expt_decomp_temp_dict_chon_f_cl.json")
    smi_decomp_temp_dict = json.load(open(smi_decomp_temp_dict_path, "r"))
    smi_list = list(smi_decomp_temp_dict.keys())
    smi_fp_dict = gen_uint8_fp_from_smiles_list_parallel(smi_list, fp_radius=fp_radius, fp_size=fp_size)
    fp_list = [smi_fp_dict[smi] for smi in smi_list]
    
    iteration_list = list(range(0, 11))
    for iteration in tqdm(iteration_list):
        model = FFNN1(input_size=2048, hidden_sizes=[300, 300, 300, 300, 300], output_size=1)
        iter_model_dir = os.path.join(active_iteration_models_path, f"iteration_{iteration}")
        k_fold_y_preds_dict = load_eval_k_fold_model(model, iter_model_dir, fp_list)
        k_fold_y_preds_dict = {fold: preds.flatten().tolist() for fold, preds in k_fold_y_preds_dict.items()}
        k_fold_smi_preds_dict = {k: dict(zip(smi_list, k_fold_y_preds_dict[k])) for k in k_fold_y_preds_dict}
        smi_preds_dict_path = os.path.join(preds_rmg_active_learning_iterations_models_dir, f"k_fold_smi_preds_dict_iteration_{iteration}.pkl")
        pickle.dump(k_fold_smi_preds_dict, open(smi_preds_dict_path, "wb"))
    return


def get_accuracy_expt_small_molec_data():
    preds_rmg_active_learning_iterations_models_dir = os.path.join(expt_small_molec_data_dir, "preds_rmg_active_learning_iterations_models")
    smi_decomp_temp_dict_path = os.path.join(expt_small_molec_data_dir, "smi_expt_decomp_temp_dict_chon_f_cl.json")
    smi_decomp_temp_dict = json.load(open(smi_decomp_temp_dict_path, "r"))
    smi_list = list(smi_decomp_temp_dict.keys())
    y_true_all = [smi_decomp_temp_dict[smi] for smi in smi_list]
    thresh_small_molec = 70.0  ## Threshold for pairwise accuracy calculation. 2*expt_error (35.0 from data source)
    iteration_list = list(range(0, 11))
    for iteration in iteration_list:
        smi_preds_dict_path = os.path.join(preds_rmg_active_learning_iterations_models_dir, f"k_fold_smi_preds_dict_iteration_{iteration}.pkl")
        smi_preds_dict = pickle.load(open(smi_preds_dict_path, "rb"))
        k_fold_pairwise_accuracy_dict = {}
        for k in smi_preds_dict:
            y_preds = [smi_preds_dict[k][smi] for smi in smi_list]
            acc_pairs, all_pairs = get_pairwise_accuracy(y_true_all, y_preds, y_true_thresh=thresh_small_molec)
            acc_pairs, all_pairs = float(acc_pairs), float(all_pairs)
            k_fold_pairwise_accuracy_dict[k] = [acc_pairs, all_pairs, acc_pairs/all_pairs]
        k_fold_pairwise_accuracy_dict_path = os.path.join(preds_rmg_active_learning_iterations_models_dir, f"k_fold_pairwise_accuracy_dict_iteration_{iteration}.json")
        json.dump(k_fold_pairwise_accuracy_dict, open(k_fold_pairwise_accuracy_dict_path, "w"), indent=4)
    
    return


def predict_expt_polymer_data():
    oligomer_preds_rmg_active_learning_iterations_models_dir_dict = {
        "dimer": os.path.join(expt_polymer_data_dir, "dimer_preds_rmg_active_learning_iterations_models"),
        "trimer": os.path.join(expt_polymer_data_dir, "trimer_preds_rmg_active_learning_iterations_models"),
        "tetramer": os.path.join(expt_polymer_data_dir, "tetramer_preds_rmg_active_learning_iterations_models")
    }
    
    for oligomer, preds_rmg_active_learning_iterations_models_dir in oligomer_preds_rmg_active_learning_iterations_models_dir_dict.items():
        abbr_oligomer_smi_dict_path = os.path.join(expt_polymer_data_dir, f"polymer_abbr_linear_{oligomer}_smi_dict.json")
        abbr_oligomer_smi_dict = json.load(open(abbr_oligomer_smi_dict_path, "r"))
        
        smi_list = list(abbr_oligomer_smi_dict.values())
        smi_fp_dict = gen_uint8_fp_from_smiles_list_parallel(smi_list, fp_radius=fp_radius, fp_size=fp_size)
        fp_list = [smi_fp_dict[smi] for smi in smi_list]
        
        iteration_list = list(range(0, 11))
        for iteration in tqdm(iteration_list):
            model = FFNN1(input_size=2048, hidden_sizes=[300, 300, 300, 300, 300], output_size=1)
            iter_model_dir = os.path.join(active_iteration_models_path, f"iteration_{iteration}")
            
            k_fold_y_preds_dict = load_eval_k_fold_model(model, iter_model_dir, fp_list)
            k_fold_y_preds_dict = {fold: preds.flatten().tolist() for fold, preds in k_fold_y_preds_dict.items()}
            k_fold_smi_preds_dict = {k: dict(zip(smi_list, k_fold_y_preds_dict[k])) for k in k_fold_y_preds_dict}
            smi_preds_dict_path = os.path.join(preds_rmg_active_learning_iterations_models_dir, f"k_fold_smi_preds_dict_iteration_{iteration}.pkl")
            pickle.dump(k_fold_smi_preds_dict, open(smi_preds_dict_path, "wb"))
    
    return


def get_accuracy_expt_polymer_data():
    expt_temp_type_list = ["tp", "td", "tp_td_mean"]
    
    oligomer_preds_rmg_active_learning_iterations_models_dir_dict = {
        "dimer": os.path.join(expt_polymer_data_dir, "dimer_preds_rmg_active_learning_iterations_models"),
        "trimer": os.path.join(expt_polymer_data_dir, "trimer_preds_rmg_active_learning_iterations_models"),
        "tetramer": os.path.join(expt_polymer_data_dir, "tetramer_preds_rmg_active_learning_iterations_models")
    }
    thresh_poly = 20.0  ## Threshold for pairwise accuracy calculation. 2*expt_error (10.0 from data source)
    for oligomer, preds_rmg_active_learning_iterations_models_dir in oligomer_preds_rmg_active_learning_iterations_models_dir_dict.items():
        abbr_oligomer_smi_dict_path = os.path.join(expt_polymer_data_dir, f"polymer_abbr_linear_{oligomer}_smi_dict.json")
        abbr_oligomer_smi_dict = json.load(open(abbr_oligomer_smi_dict_path, "r"))
        
        for expt_temp_type in expt_temp_type_list:
            abbr_temp_type_temp_dict_path = os.path.join(expt_polymer_data_dir, f"polymer_abbr_expt_{expt_temp_type}_dict.json")
            abbr_temp_type_temp_dict = json.load(open(abbr_temp_type_temp_dict_path, "r"))
            
            smi_decomp_temp_dict = {abbr_oligomer_smi_dict[abbr]: abbr_temp_type_temp_dict[abbr] for abbr in abbr_oligomer_smi_dict}
            smi_list = list(smi_decomp_temp_dict.keys())
            
            y_true_all = [smi_decomp_temp_dict[smi] for smi in smi_list]
            
            iteration_list = list(range(0, 11))
            for iteration in iteration_list:
                smi_preds_dict_path = os.path.join(preds_rmg_active_learning_iterations_models_dir, f"k_fold_smi_preds_dict_iteration_{iteration}.pkl")
                smi_preds_dict = pickle.load(open(smi_preds_dict_path, "rb"))
                
                k_fold_pairwise_accuracy_dict = {}
                for k in smi_preds_dict:
                    y_preds = [smi_preds_dict[k][smi] for smi in smi_list]
                    acc_pairs, all_pairs = get_pairwise_accuracy(y_true_all, y_preds, y_true_thresh=thresh_poly)
                    acc_pairs, all_pairs = float(acc_pairs), float(all_pairs)
                    k_fold_pairwise_accuracy_dict[k] = [acc_pairs, all_pairs, acc_pairs/all_pairs]
                
                k_fold_pairwise_accuracy_dict_path = os.path.join(preds_rmg_active_learning_iterations_models_dir, f"{expt_temp_type}_k_fold_pairwise_accuracy_dict_iteration_{iteration}.json")
                json.dump(k_fold_pairwise_accuracy_dict, open(k_fold_pairwise_accuracy_dict_path, "w"), indent=4)
    return


if __name__ == "__main__":
    main()
