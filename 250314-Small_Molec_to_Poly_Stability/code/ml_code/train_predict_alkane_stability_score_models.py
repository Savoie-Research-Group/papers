"""
    Date Modified: 2024/10/24
    Author: Veerupaksh (Veeru) Singla (singla2@purdue.edu)
    Corresponding Author: Brett M Savoie (bsavoie2@nd.edu)
    Description: Training models from alkane stability score paper: Machine learning of stability scores from kinetic data (https://doi.org/10.1039/D4DD00036F)
"""


import os
import numpy as np
import sys
import json
import pickle


this_script_dir = os.path.dirname(os.path.realpath(__file__))
os.chdir(this_script_dir)


sys.path.append(this_script_dir)
from ml_utils import *


fp_radius = 2
fp_size = 2048

data_path_main = os.path.join(this_script_dir, "../../data")
stab_score_paper_data_path = os.path.join(data_path_main, "alkane_stab_score_paper_data")
model_till_c15_path = os.path.join(stab_score_paper_data_path, "trained_model_till_c15")
model_till_c17_path = os.path.join(stab_score_paper_data_path, "trained_model_till_c17")
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
    alk_smi_hl_dict_path = os.path.join(stab_score_paper_data_path, "alk_smi_hl_dict_secs_hl_prune_till_c17_32421_vals.json")
    alk_smi_hl_dict = json.load(open(alk_smi_hl_dict_path, "r"))
    
    smi_fp_dict = gen_uint8_fp_from_smiles_list_parallel(list(alk_smi_hl_dict.keys()), fp_radius=fp_radius, fp_size=fp_size)
    
    X_till_c15, y_till_c15 = [], []
    X_till_c17, y_till_c17 = [], []
    for smi, hl in alk_smi_hl_dict.items():
        c_count = smi.count("C") + smi.count("c")
        if c_count <= 15:
            X_till_c15.append(smi_fp_dict[smi])
            y_till_c15.append(hl)
        if c_count <= 17:
            X_till_c17.append(smi_fp_dict[smi])
            y_till_c17.append(hl)
    
    print(len(X_till_c15), len(y_till_c15), len(X_till_c17), len(y_till_c17))
    
    print("Training model till C15")
    model_c15 = FFNN1(input_size=2048, hidden_sizes=[300, 300, 300, 300, 300], output_size=1)
    train_save_k_fold_model(
        model=model_c15,
        save_dir=model_till_c15_path,
        X=np.array(X_till_c15),
        y=np.log(np.array(y_till_c15)),
        k=10,
        batch_size=128,
        learning_rate=0.001,
        max_epochs=600,
        patience=300,
        warmup_epochs=0
    )
    
    print("Training model till C17")
    model_c17 = FFNN1(input_size=2048, hidden_sizes=[300, 300, 300, 300, 300], output_size=1)
    train_save_k_fold_model(
        model=model_c17,
        save_dir=model_till_c17_path,
        X=np.array(X_till_c17),
        y=np.log(np.array(y_till_c17)),
        k=10,
        batch_size=128,
        learning_rate=0.001,
        max_epochs=600,
        patience=300,
        warmup_epochs=0
    )
    return


def predict_expt_small_molec_data():
    preds_alkane_stab_score_models_dir = os.path.join(expt_small_molec_data_dir, "preds_alkane_stab_score_models")
    
    preds_till_c15_dict_path = os.path.join(preds_alkane_stab_score_models_dir, "k_fold_smi_preds_dict_alkane_stab_score_model_till_c15.pkl")
    preds_till_c17_dict_path = os.path.join(preds_alkane_stab_score_models_dir, "k_fold_smi_preds_dict_alkane_stab_score_model_till_c17.pkl")
    
    smi_decomp_temp_dict_path = os.path.join(expt_small_molec_data_dir, "smi_expt_decomp_temp_dict_chon_f_cl.json")
    smi_decomp_temp_dict = json.load(open(smi_decomp_temp_dict_path, "r"))
    
    smi_list = list(smi_decomp_temp_dict.keys())
    smi_fp_dict = gen_uint8_fp_from_smiles_list_parallel(smi_list, fp_radius=fp_radius, fp_size=fp_size)
    fp_list = [smi_fp_dict[smi] for smi in smi_list]
    
    model_c15 = FFNN1(input_size=2048, hidden_sizes=[300, 300, 300, 300, 300], output_size=1)
    model_c17 = FFNN1(input_size=2048, hidden_sizes=[300, 300, 300, 300, 300], output_size=1)
    
    k_fold_y_preds_dict_c15 = load_eval_k_fold_model(model_c15, model_till_c15_path, fp_list)
    # k_fold_y_preds_dict_c15 = {fold: np.exp(preds).flatten().tolist() for fold, preds in k_fold_y_preds_dict_c15.items()}
    k_fold_y_preds_dict_c15 = {fold: preds.flatten().tolist() for fold, preds in k_fold_y_preds_dict_c15.items()}
    
    k_fold_y_preds_dict_c17 = load_eval_k_fold_model(model_c17, model_till_c17_path, fp_list)
    # k_fold_y_preds_dict_c17 = {fold: np.exp(preds).flatten().tolist() for fold, preds in k_fold_y_preds_dict_c17.items()}
    k_fold_y_preds_dict_c17 = {fold: preds.flatten().tolist() for fold, preds in k_fold_y_preds_dict_c17.items()}
    
    k_fold_smi_preds_dict_c15 = {k: dict(zip(smi_list, k_fold_y_preds_dict_c15[k])) for k in k_fold_y_preds_dict_c15}
    k_fold_smi_preds_dict_c17 = {k: dict(zip(smi_list, k_fold_y_preds_dict_c17[k])) for k in k_fold_y_preds_dict_c17}
    
    pickle.dump(k_fold_smi_preds_dict_c15, open(preds_till_c15_dict_path, "wb"))
    pickle.dump(k_fold_smi_preds_dict_c17, open(preds_till_c17_dict_path, "wb"))
    
    return
    
    
def get_accuracy_expt_small_molec_data():
    preds_alkane_stab_score_models_dir = os.path.join(expt_small_molec_data_dir, "preds_alkane_stab_score_models")
    
    k_fold_pairwise_accuracy_till_c15_dict_path = os.path.join(preds_alkane_stab_score_models_dir, "k_fold_pairwise_accuracy_dict_alkane_stab_score_model_till_c15.json")
    k_fold_pairwise_accuracy_till_c17_dict_path = os.path.join(preds_alkane_stab_score_models_dir, "k_fold_pairwise_accuracy_dict_alkane_stab_score_model_till_c17.json")
    
    preds_till_c15_dict_path = os.path.join(preds_alkane_stab_score_models_dir, "k_fold_smi_preds_dict_alkane_stab_score_model_till_c15.pkl")
    preds_till_c15_dict = pickle.load(open(preds_till_c15_dict_path, "rb"))
    
    preds_till_c17_dict_path = os.path.join(preds_alkane_stab_score_models_dir, "k_fold_smi_preds_dict_alkane_stab_score_model_till_c17.pkl")
    preds_till_c17_dict = pickle.load(open(preds_till_c17_dict_path, "rb"))
    
    smi_decomp_temp_dict_path = os.path.join(expt_small_molec_data_dir, "smi_expt_decomp_temp_dict_chon_f_cl.json")
    smi_decomp_temp_dict = json.load(open(smi_decomp_temp_dict_path, "r"))
    
    smi_list = list(smi_decomp_temp_dict.keys())
        
    y_true_all = [smi_decomp_temp_dict[smi] for smi in smi_list]
    
    k_fold_pairwise_accuracy_till_c15_dict = {}
    k_fold_pairwise_accuracy_till_c17_dict = {}
    thresh_small_molec = 70.0  ## Threshold for pairwise accuracy calculation. 2*expt_error (35.0 from data source)
    for k in preds_till_c15_dict:
        y_pred_all_c15 = [preds_till_c15_dict[k][smi] for smi in smi_list]
        acc_pairs_all_c15, all_pairs_all_c15 = get_pairwise_accuracy(y_true_all, y_pred_all_c15, y_true_thresh=thresh_small_molec)
        acc_pairs_all_c15, all_pairs_all_c15 = float(acc_pairs_all_c15), float(all_pairs_all_c15)
        k_fold_pairwise_accuracy_till_c15_dict[k] = [acc_pairs_all_c15, all_pairs_all_c15, acc_pairs_all_c15/all_pairs_all_c15]
    
    for k in preds_till_c17_dict:
        y_pred_all_c17 = [preds_till_c17_dict[k][smi] for smi in smi_list]
        acc_pairs_all_c17, all_pairs_all_c17 = get_pairwise_accuracy(y_true_all, y_pred_all_c17, y_true_thresh=thresh_small_molec)
        acc_pairs_all_c17, all_pairs_all_c17 = float(acc_pairs_all_c17), float(all_pairs_all_c17)
        k_fold_pairwise_accuracy_till_c17_dict[k] = [acc_pairs_all_c17, all_pairs_all_c17, acc_pairs_all_c17/all_pairs_all_c17]
        
    json.dump(k_fold_pairwise_accuracy_till_c15_dict, open(k_fold_pairwise_accuracy_till_c15_dict_path, "w"), indent=4)
    json.dump(k_fold_pairwise_accuracy_till_c17_dict, open(k_fold_pairwise_accuracy_till_c17_dict_path, "w"), indent=4)
    
    return


def predict_expt_polymer_data():
    oligomer_preds_alkane_stab_score_models_dir_dict = {
        "dimer": os.path.join(expt_polymer_data_dir, "dimer_preds_alkane_stab_score_models"),
        "trimer": os.path.join(expt_polymer_data_dir, "trimer_preds_alkane_stab_score_models"),
        "tetramer": os.path.join(expt_polymer_data_dir, "tetramer_preds_alkane_stab_score_models")
    }
    
    for oligomer, preds_alkane_stab_score_models_dir in oligomer_preds_alkane_stab_score_models_dir_dict.items():
        preds_till_c15_dict_path = os.path.join(preds_alkane_stab_score_models_dir, "k_fold_smi_preds_dict_alkane_stab_score_model_till_c15.pkl")
        preds_till_c17_dict_path = os.path.join(preds_alkane_stab_score_models_dir, "k_fold_smi_preds_dict_alkane_stab_score_model_till_c17.pkl")
        
        abbr_oligomer_smi_dict_path = os.path.join(expt_polymer_data_dir, f"polymer_abbr_linear_{oligomer}_smi_dict.json")
        abbr_oligomer_smi_dict = json.load(open(abbr_oligomer_smi_dict_path, "r"))
        
        smi_list = list(abbr_oligomer_smi_dict.values())
        smi_fp_dict = gen_uint8_fp_from_smiles_list_parallel(smi_list, fp_radius=fp_radius, fp_size=fp_size)
        fp_list = [smi_fp_dict[smi] for smi in smi_list]
        
        model_c15 = FFNN1(input_size=2048, hidden_sizes=[300, 300, 300, 300, 300], output_size=1)
        model_c17 = FFNN1(input_size=2048, hidden_sizes=[300, 300, 300, 300, 300], output_size=1)
        
        k_fold_y_preds_dict_c15 = load_eval_k_fold_model(model_c15, model_till_c15_path, fp_list)
        # k_fold_y_preds_dict_c15 = {fold: np.exp(preds).flatten().tolist() for fold, preds in k_fold_y_preds_dict_c15.items()}
        k_fold_y_preds_dict_c15 = {fold: preds.flatten().tolist() for fold, preds in k_fold_y_preds_dict_c15.items()}
        
        k_fold_y_preds_dict_c17 = load_eval_k_fold_model(model_c17, model_till_c17_path, fp_list)
        # k_fold_y_preds_dict_c17 = {fold: np.exp(preds).flatten().tolist() for fold, preds in k_fold_y_preds_dict_c17.items()}
        k_fold_y_preds_dict_c17 = {fold: preds.flatten().tolist() for fold, preds in k_fold_y_preds_dict_c17.items()}
        
        k_fold_smi_preds_dict_c15 = {k: dict(zip(smi_list, k_fold_y_preds_dict_c15[k])) for k in k_fold_y_preds_dict_c15}
        k_fold_smi_preds_dict_c17 = {k: dict(zip(smi_list, k_fold_y_preds_dict_c17[k])) for k in k_fold_y_preds_dict_c17}
        
        pickle.dump(k_fold_smi_preds_dict_c15, open(preds_till_c15_dict_path, "wb"))
        pickle.dump(k_fold_smi_preds_dict_c17, open(preds_till_c17_dict_path, "wb"))
        
    return


def get_accuracy_expt_polymer_data():
    expt_temp_type_list = ["tp", "td", "tp_td_mean"]
    
    oligomer_preds_alkane_stab_score_models_dir_dict = {
        "dimer": os.path.join(expt_polymer_data_dir, "dimer_preds_alkane_stab_score_models"),
        "trimer": os.path.join(expt_polymer_data_dir, "trimer_preds_alkane_stab_score_models"),
        "tetramer": os.path.join(expt_polymer_data_dir, "tetramer_preds_alkane_stab_score_models")
    }
    thresh_poly = 20.0  ## Threshold for pairwise accuracy calculation. 2*expt_error (10.0 from data source)
    for oligomer, preds_alkane_stab_score_models_dir in oligomer_preds_alkane_stab_score_models_dir_dict.items():
        preds_till_c15_dict_path = os.path.join(preds_alkane_stab_score_models_dir, "k_fold_smi_preds_dict_alkane_stab_score_model_till_c15.pkl")
        preds_till_c15_dict = pickle.load(open(preds_till_c15_dict_path, "rb"))
        
        preds_till_c17_dict_path = os.path.join(preds_alkane_stab_score_models_dir, "k_fold_smi_preds_dict_alkane_stab_score_model_till_c17.pkl")
        preds_till_c17_dict = pickle.load(open(preds_till_c17_dict_path, "rb"))
        
        abbr_oligomer_smi_dict_path = os.path.join(expt_polymer_data_dir, f"polymer_abbr_linear_{oligomer}_smi_dict.json")
        abbr_oligomer_smi_dict = json.load(open(abbr_oligomer_smi_dict_path, "r"))
        
        for expt_temp_type in expt_temp_type_list:
            k_fold_pairwise_accuracy_till_c15_dict_path = os.path.join(preds_alkane_stab_score_models_dir, f"{expt_temp_type}_k_fold_pairwise_accuracy_dict_alkane_stab_score_model_till_c15.json")
            k_fold_pairwise_accuracy_till_c17_dict_path = os.path.join(preds_alkane_stab_score_models_dir, f"{expt_temp_type}_k_fold_pairwise_accuracy_dict_alkane_stab_score_model_till_c17.json")
            
            abbr_temp_type_temp_dict_path = os.path.join(expt_polymer_data_dir, f"polymer_abbr_expt_{expt_temp_type}_dict.json")
            abbr_temp_type_temp_dict = json.load(open(abbr_temp_type_temp_dict_path, "r"))
            
            smi_decomp_temp_dict = {abbr_oligomer_smi_dict[abbr]: abbr_temp_type_temp_dict[abbr] for abbr in abbr_oligomer_smi_dict}
            
            smi_list = list(smi_decomp_temp_dict.keys())
            
            y_true_all = [smi_decomp_temp_dict[smi] for smi in smi_list]
            
            k_fold_pairwise_accuracy_till_c15_dict = {}
            k_fold_pairwise_accuracy_till_c17_dict = {}
            
            for k in preds_till_c15_dict:
                y_pred_all_c15 = [preds_till_c15_dict[k][smi] for smi in smi_list]
                acc_pairs_all_c15, all_pairs_all_c15 = get_pairwise_accuracy(y_true_all, y_pred_all_c15, y_true_thresh=thresh_poly)
                acc_pairs_all_c15, all_pairs_all_c15 = float(acc_pairs_all_c15), float(all_pairs_all_c15)
                k_fold_pairwise_accuracy_till_c15_dict[k] = [acc_pairs_all_c15, all_pairs_all_c15, acc_pairs_all_c15/all_pairs_all_c15]
                
            for k in preds_till_c17_dict:
                y_pred_all_c17 = [preds_till_c17_dict[k][smi] for smi in smi_list]
                acc_pairs_all_c17, all_pairs_all_c17 = get_pairwise_accuracy(y_true_all, y_pred_all_c17, y_true_thresh=thresh_poly)
                acc_pairs_all_c17, all_pairs_all_c17 = float(acc_pairs_all_c17), float(all_pairs_all_c17)
                k_fold_pairwise_accuracy_till_c17_dict[k] = [acc_pairs_all_c17, all_pairs_all_c17, acc_pairs_all_c17/all_pairs_all_c17]
            
            json.dump(k_fold_pairwise_accuracy_till_c15_dict, open(k_fold_pairwise_accuracy_till_c15_dict_path, "w"), indent=4)
            json.dump(k_fold_pairwise_accuracy_till_c17_dict, open(k_fold_pairwise_accuracy_till_c17_dict_path, "w"), indent=4)
            
    return


if __name__ == "__main__":
    main()
