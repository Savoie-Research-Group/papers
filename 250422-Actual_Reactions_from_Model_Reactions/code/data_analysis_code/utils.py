import os
import json
import numpy as np

from drfp import DrfpEncoder

import matplotlib as mpl
from matplotlib import pyplot as plt
from matplotlib.legend_handler import HandlerPathCollection
import matplotlib.lines as mlines
import matplotlib.patches as mpatches


this_script_dir = os.path.dirname(os.path.realpath(__file__))
os.chdir(this_script_dir)


random_seed = 42

data_path_main = "../../data"
analyses_path_main = "../../analyses_and_plots"
plots_path_main = "../../analyses_and_plots"

vs_colors = {"blue": ["#002E6B", [0, 46, 107]], "red": ["#A11D4B", [161, 29, 75]],
             "green": ["#155914", [21, 89, 20]], "orange": ["#D15019", [209, 80, 25]],
             "purple": ["#3C063D", [60, 6, 61]]}


class AlphaHandler(HandlerPathCollection):
    def __init__(self, alpha=1.0, **kw):
        HandlerPathCollection.__init__(self, **kw)
        self.alpha = alpha
    def create_collection(self, orig_handle, sizes, offsets, transOffset):
        collection = HandlerPathCollection.create_collection(self, orig_handle, sizes, offsets, transOffset)
        collection.set_alpha(self.alpha)
        return collection
    

def round_up_to_base(x, base=10):
    # https://stackoverflow.com/a/65725123
    return x + (-x % base)


def round_down_to_base(x, base=10):
    # https://stackoverflow.com/a/65725123
    return x - (x % base)


def rxn_smi_to_ha_count(rxn_smi):
    ## only for chon and same number of atoms on both sides
    react_smi = rxn_smi.split(">>")[0]
    ha_count = react_smi.count("c") + react_smi.count("C") + react_smi.count("n") + react_smi.count("N") + react_smi.count("o") + react_smi.count("O")
    return ha_count

    
def e_csv_to_dict(e_csv):
    e_dict = {}
    with open(e_csv, "r") as f:
        for line in f:
            ssl = line.strip().split(",")
            try:
                e_dict[ssl[0]] = [float(x) for x in ssl[1:]]
            except:
                e_dict[ssl[0]] = ssl[1:]
    return e_dict


def get_rxn_drfp(rxn_smi_list):
    ## drfp from article: https://doi.org/10.1039/D1DD00006C
    drfp_list = DrfpEncoder.encode(rxn_smi_list, show_progress_bar=True, include_hydrogens=True)
    return drfp_list


def create_mr_ar_ea_dicts(mr_smi_dict, mr_ts_e_dict, ar_smi_dict, ar_ts_e_dict, react_prod_smi_e_dict, save_dicts=True):
    hartree2kcalpermol = 627.5
    
    mr_ea_fwd_dict = {}
    mr_ea_rev_dict = {}
    
    ar_ea_fwd_dict = {}
    ar_ea_rev_dict = {}
    
    for mr, mr_smi in mr_smi_dict.items():
        ts_f = mr_ts_e_dict[mr][1]  ## free energy of transition state, hartree
        
        r_f = 0  ## free energy of reactants, hartree
        p_f = 0  ## free energy of products, hartree
        r_smi, p_smi = mr_smi.split(">>")
        for smi in r_smi.split("."):
            r_f += react_prod_smi_e_dict[smi][1]
        for smi in p_smi.split("."):
            p_f += react_prod_smi_e_dict[smi][1]
        
        ea_fwd = (ts_f - r_f) * hartree2kcalpermol  ## forward free energy of activation, kcal/mol
        ea_rev = (ts_f - p_f) * hartree2kcalpermol  ## reverse free energy of activation, kcal/mol
        mr_ea_fwd_dict[mr] = ea_fwd
        mr_ea_rev_dict[mr] = ea_rev
    
    for ar, ar_smi in ar_smi_dict.items():
        ts_f = ar_ts_e_dict[ar][1]
        
        r_f = 0
        p_f = 0
        r_smi, p_smi = ar_smi.split(">>")
        for smi in r_smi.split("."):
            r_f += react_prod_smi_e_dict[smi][1]
        for smi in p_smi.split("."):
            p_f += react_prod_smi_e_dict[smi][1]
        
        ea_fwd = (ts_f - r_f) * hartree2kcalpermol
        ea_rev = (ts_f - p_f) * hartree2kcalpermol
        
        ar_ea_fwd_dict[ar] = ea_fwd
        ar_ea_rev_dict[ar] = ea_rev
    
    print(ar_ea_fwd_dict)
    
    if save_dicts:
        json.dump(mr_ea_fwd_dict, open(os.path.join(analyses_path_main, "mr_ea_fwd_dict.json"), "w"))
        json.dump(mr_ea_rev_dict, open(os.path.join(analyses_path_main, "mr_ea_rev_dict.json"), "w"))
        
        json.dump(mr_ea_fwd_dict, open(os.path.join(data_path_main, "mr_ea_fwd_dict.json"), "w"))
        json.dump(mr_ea_rev_dict, open(os.path.join(data_path_main, "mr_ea_rev_dict.json"), "w"))
        
        json.dump(ar_ea_fwd_dict, open(os.path.join(analyses_path_main, "ar_ea_fwd_dict.json"), "w"))
        json.dump(ar_ea_rev_dict, open(os.path.join(analyses_path_main, "ar_ea_rev_dict.json"), "w"))
        
        json.dump(ar_ea_fwd_dict, open(os.path.join(data_path_main, "ar_ea_fwd_dict.json"), "w"))
        json.dump(ar_ea_rev_dict, open(os.path.join(data_path_main, "ar_ea_rev_dict.json"), "w"))
    
    return mr_ea_fwd_dict, mr_ea_rev_dict, ar_ea_fwd_dict, ar_ea_rev_dict

