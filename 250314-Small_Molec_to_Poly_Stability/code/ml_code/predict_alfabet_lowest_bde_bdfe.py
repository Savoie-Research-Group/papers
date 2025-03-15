"""
    Date Modified: 2024/11/10
    Author: Veerupaksh (Veeru) Singla (singla2@purdue.edu)
    Corresponding Author: Brett M Savoie (bsavoie2@nd.edu)
"""


import os
from alfabet import model
import json
import numpy as np


this_script_dir = os.path.dirname(os.path.realpath(__file__))
os.chdir(this_script_dir)

from ml_utils import get_pairwise_accuracy

expt_small_molec_data_path = os.path.join(this_script_dir, "../../data/expt_small_molecule_decomp_temp_data")
expt_polymer_data_path = os.path.join(this_script_dir, "../../data/expt_polymer_decomp_temp_data")


def get_smi_min_bde_bdfe(smi):
    preds_df = model.predict([smi])
    print(preds_df)
    bde_min = float(np.min(preds_df['bde_pred']))
    bdfe_min = float(np.min(preds_df['bdfe_pred']))
    return bde_min, bdfe_min


def get_smi_list_min_bde_bdfe_dicts(smi_list):
    preds_df = model.predict(smi_list, batch_size=1, verbose=True)
    
    smi_bde_list_dict = preds_df.groupby("molecule").apply(lambda x: x["bde_pred"].tolist()).to_dict()
    smi_bdfe_list_dict = preds_df.groupby("molecule").apply(lambda x: x["bdfe_pred"].tolist()).to_dict()
    
    smi_min_bde_dict = {smi: float(np.min(bde_list)) for smi, bde_list in smi_bde_list_dict.items()}
    smi_min_bdfe_dict = {smi: float(np.min(bdfe_list)) for smi, bdfe_list in smi_bdfe_list_dict.items()}
    
    return smi_min_bde_dict, smi_min_bdfe_dict


def pred_polymer_data():
    alfabet_preds_dir = os.path.join(expt_polymer_data_path, "alfabet_min_bde_bdfe_preds")
    polymer_abbr_tp_td_mean_dict_path = os.path.join(expt_polymer_data_path, "polymer_abbr_expt_tp_td_mean_dict.json")
    polymer_abbr_tp_td_mean_dict = json.load(open(polymer_abbr_tp_td_mean_dict_path, "r"))
    poly_abbr_list = list(polymer_abbr_tp_td_mean_dict.keys())
    poly_tp_td_mean_list = [polymer_abbr_tp_td_mean_dict[poly_abbr] for poly_abbr in poly_abbr_list]
    poly_abbr_dimer_smi_dict_path = os.path.join(expt_polymer_data_path, "polymer_abbr_linear_dimer_smi_dict.json")
    poly_abbr_trimer_smi_dict_path = os.path.join(expt_polymer_data_path, "polymer_abbr_linear_trimer_smi_dict.json")
    poly_abbr_tetramer_smi_dict_path = os.path.join(expt_polymer_data_path, "polymer_abbr_linear_tetramer_smi_dict.json")
    poly_abbr_dimer_smi_dict = json.load(open(poly_abbr_dimer_smi_dict_path, "r"))
    poly_abbr_trimer_smi_dict = json.load(open(poly_abbr_trimer_smi_dict_path, "r"))
    poly_abbr_tetramer_smi_dict = json.load(open(poly_abbr_tetramer_smi_dict_path, "r"))
    
    dimer_smi_list = [poly_abbr_dimer_smi_dict[abbr] for abbr in poly_abbr_list]
    trimer_smi_list = [poly_abbr_trimer_smi_dict[abbr] for abbr in poly_abbr_list]
    tetramer_smi_list = [poly_abbr_tetramer_smi_dict[abbr] for abbr in poly_abbr_list]
    dimer_smi_min_bde_dict, dimer_smi_min_bdfe_dict = get_smi_list_min_bde_bdfe_dicts(dimer_smi_list)
    trimer_smi_min_bde_dict, trimer_smi_min_bdfe_dict = get_smi_list_min_bde_bdfe_dicts(trimer_smi_list)
    tetramer_smi_min_bde_dict, tetramer_smi_min_bdfe_dict = get_smi_list_min_bde_bdfe_dicts(tetramer_smi_list)
    
    json.dump(dimer_smi_min_bde_dict, open(alfabet_preds_dir + "/dimer_smi_min_bde_dict.json", "w"))
    json.dump(dimer_smi_min_bdfe_dict, open(alfabet_preds_dir + "/dimer_smi_min_bdfe_dict.json", "w"))
    json.dump(trimer_smi_min_bde_dict, open(alfabet_preds_dir + "/trimer_smi_min_bde_dict.json", "w"))
    json.dump(trimer_smi_min_bdfe_dict, open(alfabet_preds_dir + "/trimer_smi_min_bdfe_dict.json", "w"))
    json.dump(tetramer_smi_min_bde_dict, open(alfabet_preds_dir + "/tetramer_smi_min_bde_dict.json", "w"))
    json.dump(tetramer_smi_min_bdfe_dict, open(alfabet_preds_dir + "/tetramer_smi_min_bdfe_dict.json", "w"))
    
    thresh_poly = 20.0  ## Threshold for pairwise accuracy calculation. 2*expt_error (10.0 from data source)
    ## prelim test for accuracy using only tp_td_mean values
    dimer_acc_pairs_bde, dimer_total_pairs_bde = get_pairwise_accuracy(poly_tp_td_mean_list, [dimer_smi_min_bde_dict[smi] for smi in dimer_smi_list], thresh_poly)
    dimer_acc_pairs_bdfe, dimer_total_pairs_bdfe = get_pairwise_accuracy(poly_tp_td_mean_list, [dimer_smi_min_bdfe_dict[smi] for smi in dimer_smi_list], thresh_poly)
    trimer_acc_pairs_bde, trimer_total_pairs_bde = get_pairwise_accuracy(poly_tp_td_mean_list, [trimer_smi_min_bde_dict[smi] for smi in trimer_smi_list], thresh_poly)
    trimer_acc_pairs_bdfe, trimer_total_pairs_bdfe = get_pairwise_accuracy(poly_tp_td_mean_list, [trimer_smi_min_bdfe_dict[smi] for smi in trimer_smi_list], thresh_poly)
    tetramer_acc_pairs_bde, tetramer_total_pairs_bde = get_pairwise_accuracy(poly_tp_td_mean_list, [tetramer_smi_min_bde_dict[smi] for smi in tetramer_smi_list], thresh_poly)
    tetramer_acc_pairs_bdfe, tetramer_total_pairs_bdfe = get_pairwise_accuracy(poly_tp_td_mean_list, [tetramer_smi_min_bdfe_dict[smi] for smi in tetramer_smi_list], thresh_poly)
    
    print(dimer_acc_pairs_bde/dimer_total_pairs_bde, dimer_acc_pairs_bdfe/dimer_total_pairs_bdfe)  # 0.7241014799154334 0.7473572938689218
    print(trimer_acc_pairs_bde/trimer_total_pairs_bde, trimer_acc_pairs_bdfe/trimer_total_pairs_bdfe)  # 0.7389006342494715 0.7536997885835095
    print(tetramer_acc_pairs_bde/tetramer_total_pairs_bde, tetramer_acc_pairs_bdfe/tetramer_total_pairs_bdfe)  # 0.7452431289640592 0.7558139534883721
    
    return


def pred_small_molecule_data():
    smi_decomp_temp_dict_path = os.path.join(expt_small_molec_data_path, "smi_expt_decomp_temp_dict_chon_f_cl.json")
    smi_decomp_temp_dict = json.load(open(smi_decomp_temp_dict_path, "r"))
    smi_list = list(smi_decomp_temp_dict.keys())
    decomp_temp_list = [smi_decomp_temp_dict[smi] for smi in smi_list]
    
    smi_min_bde_dict, smi_min_bdfe_dict = get_smi_list_min_bde_bdfe_dicts(smi_list)
    
    json.dump(smi_min_bde_dict, open(expt_small_molec_data_path + "/smi_alfabet_min_bde_dict.json", "w"))
    json.dump(smi_min_bdfe_dict, open(expt_small_molec_data_path + "/smi_alfabet_min_bdfe_dict.json", "w"))
    
    smi_min_bde_dict = json.load(open(expt_small_molec_data_path + "/smi_alfabet_min_bde_dict.json", "r"))
    smi_min_bdfe_dict = json.load(open(expt_small_molec_data_path + "/smi_alfabet_min_bdfe_dict.json", "r"))
    ## for problem smiles (where the python alfabet package is not working), using alfabet website (https://bde.ml.nrel.gov/) to get data and manually add to dicts
    problem_smi_list = []
    for smi in smi_list:
        if smi not in smi_min_bde_dict or smi not in smi_min_bdfe_dict:
            problem_smi_list.append(smi)
    
    #  these smiles were somehow not working with the model. bde/bdfe values manually pulled from ALFABET frontend at: https://bde.ml.nrel.gov/
    problem_smi_list = ['COc1ccc(N2CC(=CN(C)C)C(=C(C#N)C#N)C2)cc1', 'CN1C(=Cc2cc(C(C)(C)C)c(O)c(C(C)(C)C)c2)C(=O)N=C1NC#N', 'CN1C(=Cc2cc(C(C)(C)C)c(O)c(C(C)(C)C)c2)C(=O)N=C1NC(=N)N',
                        'C(=Cc1ccccc1)C1=CCN(Cc2coc3ncccc23)CC1', 'C(=Cc1ccccc1)C1=CCN(Cc2ccc3[nH]ccc3c2)CC1', 'CC(C=CC1=C(C)CC(O)CC1(C)C)=CC=CC(C)=CC(=O)O', 'CC(C=CC1=C(C)C=CC(=O)C1(C)C)=CC=CC(C)=CC(=O)O']
    problem_smi_min_bde_dict = {
        'COc1ccc(N2CC(=CN(C)C)C(=C(C#N)C#N)C2)cc1': 61.0,
        'CN1C(=Cc2cc(C(C)(C)C)c(O)c(C(C)(C)C)c2)C(=O)N=C1NC#N': 76.2,
        'CN1C(=Cc2cc(C(C)(C)C)c(O)c(C(C)(C)C)c2)C(=O)N=C1NC(=N)N': 73.6,
        'C(=Cc1ccccc1)C1=CCN(Cc2coc3ncccc23)CC1': 70.3,
        'C(=Cc1ccccc1)C1=CCN(Cc2ccc3[nH]ccc3c2)CC1': 70.6,
        'CC(C=CC1=C(C)CC(O)CC1(C)C)=CC=CC(C)=CC(=O)O': 72.5,
        'CC(C=CC1=C(C)C=CC(=O)C1(C)C)=CC=CC(C)=CC(=O)O': 61.5
    }
    problem_smi_min_bdfe_dict = {
        'COc1ccc(N2CC(=CN(C)C)C(=C(C#N)C#N)C2)cc1': 47.8,
        'CN1C(=Cc2cc(C(C)(C)C)c(O)c(C(C)(C)C)c2)C(=O)N=C1NC#N': 62.7,
        'CN1C(=Cc2cc(C(C)(C)C)c(O)c(C(C)(C)C)c2)C(=O)N=C1NC(=N)N': 60.2,
        'C(=Cc1ccccc1)C1=CCN(Cc2coc3ncccc23)CC1': 56.9,
        'C(=Cc1ccccc1)C1=CCN(Cc2ccc3[nH]ccc3c2)CC1': 57.2,
        'CC(C=CC1=C(C)CC(O)CC1(C)C)=CC=CC(C)=CC(=O)O': 57.9,
        'CC(C=CC1=C(C)C=CC(=O)C1(C)C)=CC=CC(C)=CC(=O)O': 47.1
    }
    
    smi_min_bde_dict.update(problem_smi_min_bde_dict)
    smi_min_bdfe_dict.update(problem_smi_min_bdfe_dict)
    
    json.dump(smi_min_bde_dict, open(expt_small_molec_data_path + "/smi_alfabet_min_bde_dict.json", "w"))
    json.dump(smi_min_bdfe_dict, open(expt_small_molec_data_path + "/smi_alfabet_min_bdfe_dict.json", "w"))
    
    thresh_small_molec = 70.0  ## Threshold for pairwise accuracy calculation. 2*expt_error (35.0 from data source)
    ## prelim test for accuracy
    acc_pairs_bde, total_pairs_bde = get_pairwise_accuracy(decomp_temp_list, [smi_min_bde_dict[smi] for smi in smi_list], thresh_small_molec)
    acc_pairs_bdfe, total_pairs_bdfe = get_pairwise_accuracy(decomp_temp_list, [smi_min_bdfe_dict[smi] for smi in smi_list], thresh_small_molec)
    print(acc_pairs_bde/total_pairs_bde, acc_pairs_bdfe/total_pairs_bdfe)  ## 0.8615197110524837 0.8644894817617026
    
    return


def main():
    pred_polymer_data()
    pred_small_molecule_data()
    
    return


if __name__ == "__main__":
    main()
