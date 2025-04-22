"""
    Date Modified: 2025/22/04
    Author: Veerupaksh (Veeru) Singla (singla2@purdue.edu)
    Description: Create dictionaries of free energies of activation of forward and reverse model reactions and actual reactions in kcal/mol.
"""

import os
import json


this_script_dir = os.path.dirname(os.path.realpath(__file__))
os.chdir(this_script_dir)


from utils import data_path_main, analyses_path_main, plots_path_main, e_csv_to_dict, create_mr_ar_ea_dicts


def main():
    mr_smi_dict = json.load(open(os.path.join(data_path_main, "mr_smi_dict.json")))
    mr_ts_e_dict = e_csv_to_dict(os.path.join(data_path_main, "mr_transition_state_energy_list.csv"))
    
    ar_smi_dict = json.load(open(os.path.join(data_path_main, "ar_smi_dict.json")))
    ar_ts_e_dict = e_csv_to_dict(os.path.join(data_path_main, "ar_transition_state_energy_list.csv"))
    
    react_prod_smi_e_dict = e_csv_to_dict(os.path.join(data_path_main, "react_prod_smi_energy_list.csv"))
    
    ## all ea in kcal/mol
    mr_ea_fwd_dict, mr_ea_rev_dict, ar_ea_fwd_dict, ar_ea_rev_dict = create_mr_ar_ea_dicts(mr_smi_dict,
                                                                                              mr_ts_e_dict,
                                                                                              ar_smi_dict,
                                                                                              ar_ts_e_dict,
                                                                                              react_prod_smi_e_dict,
                                                                                              save_dicts=True)
    return


if __name__ == "__main__":
    main()
