"""
    Date Modified: 2024/11/04
    Author: Veerupaksh (Veeru) Singla (singla2@purdue.edu)
    Corresponding Author: Brett M Savoie (bsavoie2@nd.edu)
"""


import os

from matplotlib import pyplot as plt
from matplotlib.legend_handler import HandlerPathCollection
import matplotlib.lines as mlines
import matplotlib.patches as mpatches

import json
import pickle
import sys
import numpy as np


this_script_dir = os.path.dirname(os.path.realpath(__file__))
os.chdir(this_script_dir)


sys.path.append(this_script_dir + "/../ml_code")
from ml_utils import get_pairwise_accuracy


vs_colors = {"blue": ["#002E6B", [0, 46, 107]], "red": ["#A11D4B", [161, 29, 75]],
            "green": ["#155914", [21, 89, 20]], "orange": ["#D15019", [209, 80, 25]],
            "purple": ["#3C063D", [60, 6, 61]]}

expt_small_molec_data_path = os.path.join(this_script_dir, "../../data/expt_small_molecule_decomp_temp_data")
expt_polymer_data_path = os.path.join(this_script_dir, "../../data/expt_polymer_decomp_temp_data")
plots_path = os.path.join(this_script_dir, "../../data/plots")


class AlphaHandler(HandlerPathCollection):
    def __init__(self, alpha=1.0, **kw):
        HandlerPathCollection.__init__(self, **kw)
        self.alpha = alpha
    def create_collection(self, orig_handle, sizes, offsets, transOffset):
        collection = HandlerPathCollection.create_collection(self, orig_handle, sizes, offsets, transOffset)
        collection.set_alpha(self.alpha)
        return collection


def round_up_to_base(x, base=10):
    return x + (-x % base)


def round_down_to_base(x, base=10):
    return x - (x % base)


def expt_small_molec_data_analysis(alk_model=15):
    smi_decomp_temp_dict_path = os.path.join(expt_small_molec_data_path, "smi_expt_decomp_temp_dict_chon_f_cl.json")
    smi_decomp_temp_dict = json.load(open(smi_decomp_temp_dict_path, "r"))
    
    ## alkane stab score model trained till c15 is default in main text for consistency.
    ## other model included in si.
    if alk_model == 15:
        small_molecs_preds_dict_path = os.path.join(expt_small_molec_data_path, 
                                                    "preds_alkane_stab_score_models",
                                                    "k_fold_smi_preds_dict_alkane_stab_score_model_till_c15.pkl")
    elif alk_model == 17:
        small_molecs_preds_dict_path = os.path.join(expt_small_molec_data_path,
                                                    "preds_alkane_stab_score_models",
                                                    "k_fold_smi_preds_dict_alkane_stab_score_model_till_c17.pkl")
    else:
        print("Invalid alk_model. Choose 15 or 17")
        return {}
    
    small_molecs_preds_dict = pickle.load(open(small_molecs_preds_dict_path, "rb"))
    
    category_smi_list_dict = {
        "All": [],  # all
        "Acylic": [],  # acyclic
        "HC": [],  # only hc
        "HCX": [],  # hc + x (f, cl)
        "HCON": [],  # hc + o + n
        "<=15HA": [],  # <= 15 HA
        ">15HA": []  # > 15 HA
    }
    
    category_list = list(category_smi_list_dict.keys())
    for smi in smi_decomp_temp_dict:
        smi_l = smi.lower()
        smi_ha_dict = {
            "Cl": smi_l.count("cl"),
            "F": smi_l.count("f"),
            "C": smi_l.count("c") - smi_l.count("cl"),
            "O": smi_l.count("o"),
            "N": smi_l.count("n")
        }
        
        if smi_ha_dict["C"] == 0:
            continue
        
        category_smi_list_dict["All"].append(smi)
        
        if "1" not in smi:
            category_smi_list_dict["Acylic"].append(smi)
        
        if smi_ha_dict["Cl"] == 0 and smi_ha_dict["F"] == 0 and smi_ha_dict["O"] == 0 and smi_ha_dict["N"] == 0:
            category_smi_list_dict["HC"].append(smi)
        
        if (smi_ha_dict["Cl"] > 0 or smi_ha_dict["F"] > 0) and smi_ha_dict["O"] == 0 and smi_ha_dict["N"] == 0:
            category_smi_list_dict["HCX"].append(smi)
        
        if (smi_ha_dict["O"] > 0 or smi_ha_dict["N"] > 0) and smi_ha_dict["Cl"] == 0 and smi_ha_dict["F"] == 0:
            category_smi_list_dict["HCON"].append(smi)
        
        if sum(smi_ha_dict.values()) <= 15:
            category_smi_list_dict["<=15HA"].append(smi)
        else:
            category_smi_list_dict[">15HA"].append(smi)
    
    category_k_fold_accuracy_dict = {}
    thresh_small_molec = 70.0  ## Threshold for pairwise accuracy calculation. 2*expt_error (35.0 from data source)
    for category in category_list:
        category_smi_list = category_smi_list_dict[category]
        
        category_decomp_temp_list = [smi_decomp_temp_dict[smi] for smi in category_smi_list]
        category_preds_list_list = [[k_smi_preds_dict[smi] for smi in category_smi_list] for k, k_smi_preds_dict in small_molecs_preds_dict.items()]
        category_preds_accuracy_list = [get_pairwise_accuracy(category_decomp_temp_list, category_preds_list, thresh_small_molec) for category_preds_list in category_preds_list_list]
        category_preds_accuracy_list = [acc[0]/acc[1] for acc in category_preds_accuracy_list]
        
        category_k_fold_accuracy_dict[category] = category_preds_accuracy_list
    
    return category_k_fold_accuracy_dict


def plot_small_molec_category_k_fold_accuracy(small_molec_category_k_fold_accuracy_dict):
    ## bar plot (medians) with error bars (25th percentile, 75th percentile)
    categories_to_plot = ["All", "Acylic", "HCON", "<=15HA", ">15HA"]
    categories_alias = ["All", "Acylic", "CHON", "\u226415 HA", ">15 HA"]
    label_list = ["i", "ii", "iii", "iv", "v"]
    
    plt.clf()
    fig, ax = plt.subplots(figsize=(2, 4.6))
    
    for i, category in enumerate(categories_to_plot):
        category_accuracy_list = small_molec_category_k_fold_accuracy_dict[category]
        category_accuracy_list = np.array(category_accuracy_list)
        
        median = np.median(category_accuracy_list)
        q1 = np.percentile(category_accuracy_list, 25)
        q3 = np.percentile(category_accuracy_list, 75)
        
        ax.bar(i, median, width=0.75, color=vs_colors["orange"][0], alpha=0.5, label=f"{label_list[i]}. {categories_alias[i]}")
        ax.bar(i, median, width=0.75, yerr=[[median-q1], [q3-median]], capsize=4,  edgecolor='black', linewidth=1, facecolor='none')
        
    ax.grid(which='major', axis='y', color='gray', linestyle='-', linewidth=0.5, alpha=0.5, zorder=0)
    ax.grid(which='minor', axis='y', color='gray', linestyle=':', linewidth=0.5, alpha=0.3, zorder=0)
    ax.set_axisbelow(True)
    
    ax.minorticks_on()
    ax.xaxis.set_minor_locator(plt.NullLocator())
    
    ax.set_xlabel("Molecule\nCategory", fontsize=14)
    ax.set_ylabel("Pairwise Accuracy", fontsize=14)
    # ax.set_title("Reaction Feature Distribution", fontsize=14)
    
    ax.set_yticks(np.arange(0.0, 1.1, 0.1))
    ax.set_ylim(0.0, 1.0)
    
    ax.tick_params(axis='both', which='both', labelsize=14, direction='in')
    
    ax.set_xticks(range(len(categories_to_plot)))
    ax.set_xticklabels(categories_alias, fontsize=14, rotation=90)
    
    # ## remove color patches from legend
    # legend = ax.legend(loc='upper center', fontsize=10, ncol=2, frameon=False, handlelength=0.0, columnspacing=0.75, labelspacing=0.5, handletextpad=0.0) #, borderpad=0)

    plt.tight_layout()
    # plt.savefig(os.path.join(plots_path, "small_molec_category_k_fold_accuracy_alkane_stab_score_model.svg"), dpi=600, bbox_inches='tight', transparent=True, pad_inches=0.005)
    plt.savefig(os.path.join(plots_path, "small_molec_category_k_fold_accuracy_alkane_stab_score_model.pdf"), dpi=600, bbox_inches='tight', transparent=True, pad_inches=0.005)
    # plt.savefig(os.path.join(plots_path, "small_molec_category_k_fold_accuracy_alkane_stab_score_model.png"), dpi=600, bbox_inches='tight', transparent=True, pad_inches=0.005)
    
    return


def SI_plot_small_molec_category_k_fold_accuracy(small_molec_category_k_fold_accuracy_dict, alk_model=15):
    ## bar plot (medians) with error bars (25th percentile, 75th percentile)
    categories_to_plot = ["All", "Acylic", "HCON", "<=15HA", ">15HA"]
    categories_alias = ["All", "Acylic", "CHON", "\u226415 HA", ">15 HA"]
    label_list = ["i", "ii", "iii", "iv", "v"]
    
    plt.clf()
    fig, ax = plt.subplots(figsize=(2, 4.6))
    
    for i, category in enumerate(categories_to_plot):
        category_accuracy_list = small_molec_category_k_fold_accuracy_dict[category]
        category_accuracy_list = np.array(category_accuracy_list)
        
        median = np.median(category_accuracy_list)
        q1 = np.percentile(category_accuracy_list, 25)
        q3 = np.percentile(category_accuracy_list, 75)
        
        ax.bar(i, median, width=0.75, color=vs_colors["orange"][0], alpha=0.5, label=f"{label_list[i]}. {categories_alias[i]}")
        ax.bar(i, median, width=0.75, yerr=[[median-q1], [q3-median]], capsize=4,  edgecolor='black', linewidth=1, facecolor='none')
        
    ax.grid(which='major', axis='y', color='gray', linestyle='-', linewidth=0.5, alpha=0.5, zorder=0)
    ax.grid(which='minor', axis='y', color='gray', linestyle=':', linewidth=0.5, alpha=0.3, zorder=0)
    ax.set_axisbelow(True)
    
    ax.minorticks_on()
    ax.xaxis.set_minor_locator(plt.NullLocator())
    
    ax.set_xlabel("Molecule\nCategory", fontsize=14)
    ax.set_ylabel("Pairwise Accuracy", fontsize=14)
    # ax.set_title("Reaction Feature Distribution", fontsize=14)
    
    ax.set_yticks(np.arange(0.0, 1.1, 0.1))
    ax.set_ylim(0.0, 1.0)
    
    ax.tick_params(axis='both', which='both', labelsize=14, direction='in')
    
    ax.set_xticks(range(len(categories_to_plot)))
    ax.set_xticklabels(categories_alias, fontsize=14, rotation=90)
    
    # ## remove color patches from legend
    # legend = ax.legend(loc='upper center', fontsize=10, ncol=2, frameon=False, handlelength=0.0, columnspacing=0.75, labelspacing=0.5, handletextpad=0.0) #, borderpad=0)
    
    if alk_model == 15:
        save_prefix = "till_c15"
    elif alk_model == 17:
        save_prefix = "till_c17"
    else:
        print("Invalid alk_model. Choose 15 or 17")
        return
    
    plt.tight_layout()
    # plt.savefig(os.path.join(plots_path, f"SI_small_molec_category_k_fold_accuracy_alkane_stab_score_model_{save_prefix}.svg"), dpi=600, bbox_inches='tight', transparent=True, pad_inches=0.005)
    plt.savefig(os.path.join(plots_path, f"SI_small_molec_category_k_fold_accuracy_alkane_stab_score_model_{save_prefix}.pdf"), dpi=600, bbox_inches='tight', transparent=True, pad_inches=0.005)
    # plt.savefig(os.path.join(plots_path, f"SI_small_molec_category_k_fold_accuracy_alkane_stab_score_model_{save_prefix}.png"), dpi=600, bbox_inches='tight', transparent=True, pad_inches=0.005)
    plt.close()
    
    return


def expt_polymer_data_analysis(oligomer_len=3, alk_model=15):
    ## trimer in main text. dimer and tetramer in SI
    if oligomer_len == 3:
        poly_abbr_oligomer_dict_path = os.path.join(expt_polymer_data_path, "polymer_abbr_linear_trimer_smi_dict.json")
    elif oligomer_len == 2:
        poly_abbr_oligomer_dict_path = os.path.join(expt_polymer_data_path, "polymer_abbr_linear_dimer_smi_dict.json")
    elif oligomer_len == 4:
        poly_abbr_oligomer_dict_path = os.path.join(expt_polymer_data_path, "polymer_abbr_linear_tetramer_smi_dict.json")
    else:
        print("Invalid oligomer_len. Choose 2, 3 or 4")
        return {}, {}, {}
    poly_abbr_oligomer_dict = json.load(open(poly_abbr_oligomer_dict_path, "r"))
    
    chain_growth = ['PBDR', 'PBR', 'PCR', 'PCTFE', 'PECTFE', 'PETFE', 'PNR',
                    'P3FE', 'PA6', 'PAM', 'PAN', 'PBMA', 'PCL', 'PE-HD', 'PEA',
                    'PEMA', 'PEO', 'PMMA', 'PMS', 'POM', 'PP', 'PPOX', 'PS',
                    'PTFE', 'PVAC', 'PVC', 'PVDC', 'PVDF', 'PVF', 'PVK']
    
    step_growth = ['PA12', 'PA610', 'PA612', 'PA66', 'PEN', 'PVOH',
                'PPO', 'PX', 'PBT', 'PET', 'PVB']
    
    if alk_model not in [15, 17]:
        print("Invalid alk_model. Choose 15 or 17")
        return {}, {}, {}
    
    if oligomer_len == 3:
        oligomer_preds_dict_path = os.path.join(expt_polymer_data_path, 
                                        "trimer_preds_alkane_stab_score_models",
                                        f"k_fold_smi_preds_dict_alkane_stab_score_model_till_c{alk_model}.pkl")
    elif oligomer_len == 2:
        oligomer_preds_dict_path = os.path.join(expt_polymer_data_path,
                                        "dimer_preds_alkane_stab_score_models",
                                        f"k_fold_smi_preds_dict_alkane_stab_score_model_till_c{alk_model}.pkl")
    elif oligomer_len == 4:
        oligomer_preds_dict_path = os.path.join(expt_polymer_data_path,
                                        "tetramer_preds_alkane_stab_score_models",
                                        f"k_fold_smi_preds_dict_alkane_stab_score_model_till_c{alk_model}.pkl")
    else:
        print("Invalid oligomer_len. Choose 2, 3 or 4")
        return {}, {}, {}
    
    smi_preds_dict = pickle.load(open(oligomer_preds_dict_path, "rb"))
    
    poly_abbr_tp_dict_path = os.path.join(expt_polymer_data_path, "polymer_abbr_expt_tp_dict.json")
    poly_abbr_td_dict_path = os.path.join(expt_polymer_data_path, "polymer_abbr_expt_td_dict.json")
    poly_abbr_tp_td_mean_dict_path = os.path.join(expt_polymer_data_path, "polymer_abbr_expt_tp_td_mean_dict.json")
    poly_abbr_tp_dict = json.load(open(poly_abbr_tp_dict_path, "r"))
    poly_abbr_td_dict = json.load(open(poly_abbr_td_dict_path, "r"))
    poly_abbr_tp_td_mean_dict = json.load(open(poly_abbr_tp_td_mean_dict_path, "r"))
    
    smi_tp_dict = {poly_abbr_oligomer_dict[abbr]: tp for abbr, tp in poly_abbr_tp_dict.items()}
    smi_td_dict = {poly_abbr_oligomer_dict[abbr]: td for abbr, td in poly_abbr_td_dict.items()}
    smi_tp_td_mean_dict = {poly_abbr_oligomer_dict[abbr]: tp_td_mean for abbr, tp_td_mean in poly_abbr_tp_td_mean_dict.items()}
    
    category_smi_list_dict = {
        "All": list(poly_abbr_oligomer_dict.values()),  # all
        "Chain Growth": [smi for abbr, smi in poly_abbr_oligomer_dict.items() if abbr in chain_growth],  # chain growth
        "Step Growth": [smi for abbr, smi in poly_abbr_oligomer_dict.items() if abbr in step_growth],  # step growth
        "CH": [],  # only c, h
        "CHX Only": [],  # only c, h, x (f, cl)
        "CHN Only": [],  # only c, h, n
        "CHON Only": [],  # only c, h, o, n
        "CHO Only": [],  # only c, h, o
        "CHON All": []  # c, h, o, n
    }
    
    category_list = list(category_smi_list_dict.keys())
    for smi in poly_abbr_oligomer_dict.values():
        smi_l = smi.lower()
        smi_ha_dict = {
            "Cl": smi_l.count("cl"),
            "F": smi_l.count("f"),
            "C": smi_l.count("c") - smi_l.count("cl"),
            "O": smi_l.count("o"),
            "N": smi_l.count("n")
        }
        
        if smi_ha_dict["C"] == 0:
            continue
        
        if smi_ha_dict["Cl"] == 0 and smi_ha_dict["F"] == 0 and smi_ha_dict["O"] == 0 and smi_ha_dict["N"] == 0:
            category_smi_list_dict["CH"].append(smi)
        
        if (smi_ha_dict["Cl"] > 0 or smi_ha_dict["F"] > 0) and smi_ha_dict["O"] == 0 and smi_ha_dict["N"] == 0:
            category_smi_list_dict["CHX Only"].append(smi)
        
        if smi_ha_dict["O"] > 0 and smi_ha_dict["N"] == 0 and smi_ha_dict["Cl"] == 0 and smi_ha_dict["F"] == 0:
            category_smi_list_dict["CHO Only"].append(smi)
            category_smi_list_dict["CHON All"].append(smi)
        
        if smi_ha_dict["N"] > 0 and smi_ha_dict["O"] == 0 and smi_ha_dict["Cl"] == 0 and smi_ha_dict["F"] == 0:
            category_smi_list_dict["CHN Only"].append(smi)
            category_smi_list_dict["CHON All"].append(smi)
            
        if smi_ha_dict["O"] > 0 and smi_ha_dict["N"] > 0 and smi_ha_dict["Cl"] == 0 and smi_ha_dict["F"] == 0:
            category_smi_list_dict["CHON Only"].append(smi)
            category_smi_list_dict["CHON All"].append(smi)
        
    category_k_fold_accuracy_dict_tp = {}
    category_k_fold_accuracy_dict_td = {}
    category_k_fold_accuracy_dict_tp_td_mean = {}
    thresh_poly = 20.0  ## Threshold for pairwise accuracy calculation. 2*expt_error (10.0 from data source)
    for category in category_list:
        category_smi_list = category_smi_list_dict[category]
        
        category_tp_list = [smi_tp_dict[smi] for smi in category_smi_list]
        category_td_list = [smi_td_dict[smi] for smi in category_smi_list]
        category_tp_td_mean_list = [smi_tp_td_mean_dict[smi] for smi in category_smi_list]
        
        category_preds_list_list = [[k_smi_preds_dict[smi] for smi in category_smi_list] for k, k_smi_preds_dict in smi_preds_dict.items()]
        
        category_preds_accuracy_list_tp = [get_pairwise_accuracy(category_tp_list, category_preds_list, thresh_poly) for category_preds_list in category_preds_list_list]
        category_preds_accuracy_list_tp = [acc[0]/acc[1] for acc in category_preds_accuracy_list_tp]
        category_k_fold_accuracy_dict_tp[category] = category_preds_accuracy_list_tp
        
        category_preds_accuracy_list_td = [get_pairwise_accuracy(category_td_list, category_preds_list, thresh_poly) for category_preds_list in category_preds_list_list]
        category_preds_accuracy_list_td = [acc[0]/acc[1] for acc in category_preds_accuracy_list_td]
        category_k_fold_accuracy_dict_td[category] = category_preds_accuracy_list_td
        
        category_preds_accuracy_list_tp_td_mean = [get_pairwise_accuracy(category_tp_td_mean_list, category_preds_list, thresh_poly) for category_preds_list in category_preds_list_list]
        category_preds_accuracy_list_tp_td_mean = [acc[0]/acc[1] for acc in category_preds_accuracy_list_tp_td_mean]
        category_k_fold_accuracy_dict_tp_td_mean[category] = category_preds_accuracy_list_tp_td_mean
        
    ## print cho only accuracies
    print("CHO Only Accuracies")
    print(f"Tp: {np.quantile(category_k_fold_accuracy_dict_tp['CHO Only'], [0.25, 0.5, 0.75])}")
    print(f"Td: {np.quantile(category_k_fold_accuracy_dict_td['CHO Only'], [0.25, 0.5, 0.75])}")
    print(f"Tp+Td Mean: {np.quantile(category_k_fold_accuracy_dict_tp_td_mean['CHO Only'], [0.25, 0.5, 0.75])}")
        

    return category_k_fold_accuracy_dict_tp, category_k_fold_accuracy_dict_td, category_k_fold_accuracy_dict_tp_td_mean


def plot_polymer_category_k_fold_accuracy(category_k_fold_accuracy_dict_tp, category_k_fold_accuracy_dict_td, category_k_fold_accuracy_dict_tp_td_mean):
    categories_to_plot = ["All", "Chain Growth", "Step Growth", "CH", "CHX Only", "CHON All"]
    categories_alias = ["All", "Chain\nGrowth", "Step\nGrowth", "CH", "CHX", "CHON"]
    label_list = ["i", "ii", "iii", "iv", "v", "vi"]
    
    ## plot tp, td, tp_td_mean for each category under same tick. 3 bars per category.
    plt.clf()
    fig, ax = plt.subplots(figsize=(3.6, 4.5))
    
    bwidth = 0.8
    for i, category in enumerate(categories_to_plot):
        category_accuracy_list_tp = category_k_fold_accuracy_dict_tp[category]
        category_accuracy_list_td = category_k_fold_accuracy_dict_td[category]
        category_accuracy_list_tp_td_mean = category_k_fold_accuracy_dict_tp_td_mean[category]
        
        category_accuracy_list_tp = np.array(category_accuracy_list_tp)
        category_accuracy_list_td = np.array(category_accuracy_list_td)
        category_accuracy_list_tp_td_mean = np.array(category_accuracy_list_tp_td_mean)
        
        median_tp = np.median(category_accuracy_list_tp)
        q1_tp = np.percentile(category_accuracy_list_tp, 25)
        q3_tp = np.percentile(category_accuracy_list_tp, 75)
        
        median_td = np.median(category_accuracy_list_td)
        q1_td = np.percentile(category_accuracy_list_td, 25)
        q3_td = np.percentile(category_accuracy_list_td, 75)
        
        median_tp_td_mean = np.median(category_accuracy_list_tp_td_mean)
        q1_tp_td_mean = np.percentile(category_accuracy_list_tp_td_mean, 25)
        q3_tp_td_mean = np.percentile(category_accuracy_list_tp_td_mean, 75)
        
        ax.bar(3*i-bwidth-0.05, median_tp, width=bwidth, yerr=[[median_tp-q1_tp], [q3_tp-median_tp]], capsize=3,  edgecolor=vs_colors["blue"][0], linewidth=1, facecolor='none')
        ax.bar(3*i-bwidth-0.05, median_tp, width=bwidth, color=vs_colors["blue"][0], alpha=0.5)  #, label=f"{label_list[i]}. {categories_alias[i]} TP")
        
        ax.bar(3*i, median_td, width=bwidth, yerr=[[median_td-q1_td], [q3_td-median_td]], capsize=3,  edgecolor=vs_colors["red"][0], linewidth=1, facecolor='none')
        ax.bar(3*i, median_td, width=bwidth, color=vs_colors["red"][0], alpha=0.5)  #, label=f"{label_list[i]}. {categories_alias[i]} TD")
        
        ax.bar(3*i+bwidth+0.05, median_tp_td_mean, width=bwidth, yerr=[[median_tp_td_mean-q1_tp_td_mean], [q3_tp_td_mean-median_tp_td_mean]], capsize=3,  edgecolor=vs_colors["green"][0], linewidth=1, facecolor='none')
        ax.bar(3*i+bwidth+0.05, median_tp_td_mean, width=bwidth, color=vs_colors["green"][0], alpha=0.5)  #, label=f"{label_list[i]}. {categories_alias[i]} TP-TD Mean")
    
    ax.grid(which='major', axis='y', color='gray', linestyle='-', linewidth=0.5, alpha=0.5, zorder=0)
    ax.grid(which='minor', axis='y', color='gray', linestyle=':', linewidth=0.5, alpha=0.3, zorder=0)
    ax.set_axisbelow(True)
    
    ax.minorticks_on()
    ax.xaxis.set_minor_locator(plt.NullLocator())
    
    ax.set_xlabel("Polymer Category", fontsize=14)
    ax.set_ylabel("Pairwise Accuracy", fontsize=14)
    # ax.set_title("Reaction Feature Distribution", fontsize=14)
    
    ax.set_yticks(np.arange(0.0, 1.1, 0.1))
    ax.set_ylim(0.0, 1.0)
    
    ax.tick_params(axis='both', which='both', labelsize=14, direction='in')
    
    ax.set_xticks(3*np.arange(len(categories_to_plot)))
    ax.set_xticklabels(categories_alias, fontsize=14, rotation=90)
    
    ## custom legend patch. 3 patches for 3 bars.
    tp_patch = mpatches.Patch(color=vs_colors["blue"][0], alpha=0.5, label="Tp")
    td_patch = mpatches.Patch(color=vs_colors["red"][0], alpha=0.5, label="Td")
    tp_td_mean_patch = mpatches.Patch(color=vs_colors["green"][0], alpha=0.5, label="(Tp+Td)/2")
    
    legend = ax.legend(handles=[tp_patch, td_patch, tp_td_mean_patch], loc='upper left', fontsize=10, ncol=1, bbox_to_anchor=(0.02, 1.01)) #, frameon=False, handlelength=0.0, columnspacing=0.75, labelspacing=0.5, handletextpad=0.0) #, borderpad=0)
    
    plt.tight_layout()
    # plt.savefig(os.path.join(plots_path, "polymer_category_k_fold_accuracy_alkane_stab_score_model.svg"), dpi=600, bbox_inches='tight', transparent=True, pad_inches=0.005)
    plt.savefig(os.path.join(plots_path, "polymer_category_k_fold_accuracy_alkane_stab_score_model.pdf"), dpi=600, bbox_inches='tight', transparent=True, pad_inches=0.005)
    # plt.savefig(os.path.join(plots_path, "polymer_category_k_fold_accuracy_alkane_stab_score_model.png"), dpi=600, bbox_inches='tight', transparent=True, pad_inches=0.005)
    
    return


def SI_plot_polymer_category_k_fold_accuracy(category_k_fold_accuracy_dict_tp, category_k_fold_accuracy_dict_td, category_k_fold_accuracy_dict_tp_td_mean, oligomer_len=3, alk_model=15):
    categories_to_plot = ["All", "Chain Growth", "Step Growth", "CH", "CHX Only", "CHON All"]
    categories_alias = ["All", "Chain\nGrowth", "Step\nGrowth", "CH", "CHX", "CHON"]
    label_list = ["i", "ii", "iii", "iv", "v", "vi"]
    
    ## plot tp, td, tp_td_mean for each category under same tick. 3 bars per category.
    plt.clf()
    fig, ax = plt.subplots(figsize=(3.6, 4.5))
    
    bwidth = 0.8
    for i, category in enumerate(categories_to_plot):
        category_accuracy_list_tp = category_k_fold_accuracy_dict_tp[category]
        category_accuracy_list_td = category_k_fold_accuracy_dict_td[category]
        category_accuracy_list_tp_td_mean = category_k_fold_accuracy_dict_tp_td_mean[category]
        
        category_accuracy_list_tp = np.array(category_accuracy_list_tp)
        category_accuracy_list_td = np.array(category_accuracy_list_td)
        category_accuracy_list_tp_td_mean = np.array(category_accuracy_list_tp_td_mean)
        
        median_tp = np.median(category_accuracy_list_tp)
        q1_tp = np.percentile(category_accuracy_list_tp, 25)
        q3_tp = np.percentile(category_accuracy_list_tp, 75)
        
        median_td = np.median(category_accuracy_list_td)
        q1_td = np.percentile(category_accuracy_list_td, 25)
        q3_td = np.percentile(category_accuracy_list_td, 75)
        
        median_tp_td_mean = np.median(category_accuracy_list_tp_td_mean)
        q1_tp_td_mean = np.percentile(category_accuracy_list_tp_td_mean, 25)
        q3_tp_td_mean = np.percentile(category_accuracy_list_tp_td_mean, 75)
        
        ax.bar(3*i-bwidth-0.05, median_tp, width=bwidth, yerr=[[median_tp-q1_tp], [q3_tp-median_tp]], capsize=3,  edgecolor=vs_colors["blue"][0], linewidth=1, facecolor='none')
        ax.bar(3*i-bwidth-0.05, median_tp, width=bwidth, color=vs_colors["blue"][0], alpha=0.5)  #, label=f"{label_list[i]}. {categories_alias[i]} TP")
        
        ax.bar(3*i, median_td, width=bwidth, yerr=[[median_td-q1_td], [q3_td-median_td]], capsize=3,  edgecolor=vs_colors["red"][0], linewidth=1, facecolor='none')
        ax.bar(3*i, median_td, width=bwidth, color=vs_colors["red"][0], alpha=0.5)  #, label=f"{label_list[i]}. {categories_alias[i]} TD")
        
        ax.bar(3*i+bwidth+0.05, median_tp_td_mean, width=bwidth, yerr=[[median_tp_td_mean-q1_tp_td_mean], [q3_tp_td_mean-median_tp_td_mean]], capsize=3,  edgecolor=vs_colors["green"][0], linewidth=1, facecolor='none')
        ax.bar(3*i+bwidth+0.05, median_tp_td_mean, width=bwidth, color=vs_colors["green"][0], alpha=0.5)  #, label=f"{label_list[i]}. {categories_alias[i]} TP-TD Mean")
    
    ax.grid(which='major', axis='y', color='gray', linestyle='-', linewidth=0.5, alpha=0.5, zorder=0)
    ax.grid(which='minor', axis='y', color='gray', linestyle=':', linewidth=0.5, alpha=0.3, zorder=0)
    ax.set_axisbelow(True)
    
    ax.minorticks_on()
    ax.xaxis.set_minor_locator(plt.NullLocator())
    
    ax.set_xlabel("Polymer Category", fontsize=14)
    ax.set_ylabel("Pairwise Accuracy", fontsize=14)
    # ax.set_title("Reaction Feature Distribution", fontsize=14)
    
    ax.set_yticks(np.arange(0.0, 1.1, 0.1))
    ax.set_ylim(0.0, 1.0)
    
    ax.tick_params(axis='both', which='both', labelsize=14, direction='in')
    
    ax.set_xticks(3*np.arange(len(categories_to_plot)))
    ax.set_xticklabels(categories_alias, fontsize=14, rotation=90)
    
    ## custom legend patch. 3 patches for 3 bars.
    tp_patch = mpatches.Patch(color=vs_colors["blue"][0], alpha=0.5, label="Tp")
    td_patch = mpatches.Patch(color=vs_colors["red"][0], alpha=0.5, label="Td")
    tp_td_mean_patch = mpatches.Patch(color=vs_colors["green"][0], alpha=0.5, label="(Tp+Td)/2")
    
    legend = ax.legend(handles=[tp_patch, td_patch, tp_td_mean_patch], loc='upper left', fontsize=10, ncol=1, bbox_to_anchor=(0.02, 1.01)) #, frameon=False, handlelength=0.0, columnspacing=0.75, labelspacing=0.5, handletextpad=0.0) #, borderpad=0)
    
    oligomer_prefix_dict = {2: "dimer", 3: "trimer", 4: "tetramer"}
    alk_model_prefix_dict = {15: "till_c15", 17: "till_c17"}
    
    plt.tight_layout()
    # plt.savefig(os.path.join(plots_path, f"SI_{oligomer_prefix_dict[oligomer_len]}_category_k_fold_accuracy_alkane_stab_score_model_{alk_model_prefix_dict[alk_model]}.svg"), dpi=600, bbox_inches='tight', transparent=True, pad_inches=0.005)
    plt.savefig(os.path.join(plots_path, f"SI_{oligomer_prefix_dict[oligomer_len]}_category_k_fold_accuracy_alkane_stab_score_model_{alk_model_prefix_dict[alk_model]}.pdf"), dpi=600, bbox_inches='tight', transparent=True, pad_inches=0.005)
    # plt.savefig(os.path.join(plots_path, f"SI_{oligomer_prefix_dict[oligomer_len]}_category_k_fold_accuracy_alkane_stab_score_model_{alk_model_prefix_dict[alk_model]}.png"), dpi=600, bbox_inches='tight', transparent=True, pad_inches=0.005)
    plt.close()
    
    return


def main():
    small_molec_category_k_fold_accuracy_dict = expt_small_molec_data_analysis()
    plot_small_molec_category_k_fold_accuracy(small_molec_category_k_fold_accuracy_dict)
    
    category_k_fold_accuracy_dict_tp, category_k_fold_accuracy_dict_td, category_k_fold_accuracy_dict_tp_td_mean = expt_polymer_data_analysis()
    plot_polymer_category_k_fold_accuracy(category_k_fold_accuracy_dict_tp, category_k_fold_accuracy_dict_td, category_k_fold_accuracy_dict_tp_td_mean)
    
    small_molec_category_k_fold_accuracy_dict_c15 = expt_small_molec_data_analysis(alk_model=15)
    SI_plot_small_molec_category_k_fold_accuracy(small_molec_category_k_fold_accuracy_dict_c15, alk_model=15)
    
    small_molec_category_k_fold_accuracy_dict_c17 = expt_small_molec_data_analysis(alk_model=17)
    SI_plot_small_molec_category_k_fold_accuracy(small_molec_category_k_fold_accuracy_dict_c17, alk_model=17)
    
    for oligomer_len in [2, 3, 4]:
        for alk_model in [15, 17]:
            category_k_fold_accuracy_dict_tp, category_k_fold_accuracy_dict_td, category_k_fold_accuracy_dict_tp_td_mean = expt_polymer_data_analysis(oligomer_len=oligomer_len, alk_model=alk_model)
            SI_plot_polymer_category_k_fold_accuracy(category_k_fold_accuracy_dict_tp, category_k_fold_accuracy_dict_td, category_k_fold_accuracy_dict_tp_td_mean, oligomer_len=oligomer_len, alk_model=alk_model)
    
    return


if __name__ == "__main__":
    main()
