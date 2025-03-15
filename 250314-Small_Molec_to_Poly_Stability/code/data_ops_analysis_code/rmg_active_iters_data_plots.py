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

import dill
from copy import deepcopy
from tqdm import tqdm
import multiprocessing as mp


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


def return_iter_category_k_fold_accuracy_dict_dict_iter_small_molec(iter, category_list=None, category_smi_list_dict=None, smi_decomp_temp_dict=None):
    iter_category_k_fold_accuracy_dict_dict_iter = {}
    
    small_molecs_preds_dict_path = os.path.join(expt_small_molec_data_path,
                                                "preds_rmg_active_learning_iterations_models",
                                                f"k_fold_smi_preds_dict_iteration_{iter}.pkl")
    small_molecs_preds_dict = pickle.load(open(small_molecs_preds_dict_path, "rb"))
    
    thresh_small_molec = 70.0  # (2 * experimental error)
    for category in tqdm(category_list):
        category_smi_list = category_smi_list_dict[category]
        
        category_decomp_temp_list = [smi_decomp_temp_dict[smi] for smi in category_smi_list]
        category_preds_list_list = [[k_smi_preds_dict[smi] for smi in category_smi_list] for k, k_smi_preds_dict in small_molecs_preds_dict.items()]
        category_preds_accuracy_list = [get_pairwise_accuracy(category_decomp_temp_list, category_preds_list, thresh_small_molec) for category_preds_list in category_preds_list_list]
        category_preds_accuracy_list = [acc[0]/acc[1] for acc in category_preds_accuracy_list]
        
        iter_category_k_fold_accuracy_dict_dict_iter[category] = category_preds_accuracy_list
    
    return iter_category_k_fold_accuracy_dict_dict_iter


def expt_small_molec_data_analysis():
    smi_decomp_temp_dict_path = os.path.join(expt_small_molec_data_path, "smi_expt_decomp_temp_dict_chon_f_cl.json")
    smi_decomp_temp_dict = json.load(open(smi_decomp_temp_dict_path, "r"))
    
    category_smi_list_dict = {
        "All": [],  # all
        "Acylic": [],  # acyclic
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
        
        if (smi_ha_dict["O"] > 0 or smi_ha_dict["N"] > 0) and smi_ha_dict["Cl"] == 0 and smi_ha_dict["F"] == 0:
            category_smi_list_dict["HCON"].append(smi)
        
        if sum(smi_ha_dict.values()) <= 15:
            category_smi_list_dict["<=15HA"].append(smi)
        else:
            category_smi_list_dict[">15HA"].append(smi)
    
    # for iter in tqdm(iter_list):
    #     iter_category_k_fold_accuracy_dict_dict[iter] = return_iter_category_k_fold_accuracy_dict_dict_iter(iter)
    
    iter_list = list(range(11))
    mp_cores = len(os.sched_getaffinity(0))
    with mp.Pool(mp_cores) as pool:
        iter_category_k_fold_accuracy_dict_list = pool.starmap(return_iter_category_k_fold_accuracy_dict_dict_iter_small_molec, [(iter, category_list, category_smi_list_dict, smi_decomp_temp_dict) for iter in iter_list])
        
    iter_category_k_fold_accuracy_dict_dict = dict(zip(iter_list, iter_category_k_fold_accuracy_dict_list))
    
    return iter_category_k_fold_accuracy_dict_dict


def plot_small_molec_category_k_fold_accuracy(iter_category_k_fold_accuracy_dict_dict):
    categories_to_plot = ["All", "Acylic", "HCON", "<=15HA", ">15HA"]
    categories_alias = ["All", "Acylic", "CHON", "\u226415 HA", ">15 HA"]
    label_list = ["i", "ii", "iii", "iv", "v"]
    
    plt.clf()
    # fig, axs = plt.subplots(2, 6, figsize=(12, 9.2), sharex=True, sharey=True, constrained_layout=True)
    # fig.subplots_adjust(hspace=0, wspace=0)
    
    fig = plt.figure(figsize=(3.5, 9))
    # gs = fig.add_gridspec(2, 6, hspace=0, wspace=0)
    gs = fig.add_gridspec(3, 2, hspace=0, wspace=0)
    axs = gs.subplots(sharex=True, sharey=True)
    
    for iter in tqdm(range(0,11,2)):
        # ax = axs[iter//6, iter%6]
        in_ = iter//2
        ax = axs[in_//2, in_%2]
        
        for i, category in enumerate(categories_to_plot):
            iter_category_accuracy_list = iter_category_k_fold_accuracy_dict_dict[iter][category]
            iter_category_accuracy_list = np.array(iter_category_accuracy_list)
            
            median = np.median(iter_category_accuracy_list)
            q1 = np.percentile(iter_category_accuracy_list, 25)
            q3 = np.percentile(iter_category_accuracy_list, 75)
            
            ax.bar(i, median, width=0.75, color=vs_colors["orange"][0], alpha=0.5, label=f"{label_list[i]}. {categories_alias[i]}")
            ax.bar(i, median, width=0.75, yerr=[[median-q1], [q3-median]], capsize=4,  edgecolor='black', linewidth=1, facecolor='none')
            
        # ax.set_title(f"Iter {iter}", fontsize=12)
        ax.text(0.03, 0.94, f"Iter {iter}", horizontalalignment='left', verticalalignment='center', transform=ax.transAxes, fontsize=14)
        
        ax.grid(which='major', axis='y', color='gray', linestyle='-', linewidth=0.5, alpha=0.5, zorder=0)
        ax.grid(which='minor', axis='y', color='gray', linestyle=':', linewidth=0.5, alpha=0.3, zorder=0)
        ax.set_axisbelow(True)
        
        ax.minorticks_on()
        ax.xaxis.set_minor_locator(plt.NullLocator())
        
        ## set y label in iter 4 plot
        if iter == 4:
            ax.set_ylabel("Pairwise Accuracy", fontsize=14)
            
        ## set x label between iter 8 and 10 plots
        if iter == 8:
            ax.set_xlabel("Molecule Category", fontsize=14)
            ax.xaxis.set_label_coords(1.0, -0.4)
        
        ax.set_yticks(np.arange(0.1, 1.1, 0.1))
        ax.set_ylim(0.0, 1.0)
        
        ax.tick_params(axis='both', which='both', labelsize=14, direction='in')
        
        ax.set_xticks(range(len(categories_to_plot)))
        ax.set_xticklabels(categories_alias, fontsize=14, rotation=90)
    
    # plt.savefig(os.path.join(plots_path, "small_molec_category_k_fold_accuracy_rmg_active_learning_iterations_models.svg"), dpi=600, bbox_inches='tight', transparent=True, pad_inches=0.005)
    # plt.savefig(os.path.join(plots_path, "small_molec_category_k_fold_accuracy_rmg_active_learning_iterations_models.png"), dpi=600, bbox_inches='tight', transparent=True, pad_inches=0.005)
    plt.savefig(os.path.join(plots_path, "small_molec_category_k_fold_accuracy_rmg_active_learning_iterations_models.pdf"), dpi=600, bbox_inches='tight', transparent=True, pad_inches=0.005)
    plt.close()
    
    return


def SI_plot_small_molec_category_k_fold_accuracy(iter_category_k_fold_accuracy_dict_dict):
    categories_to_plot = ["All", "Acylic", "HCON", "<=15HA", ">15HA"]
    categories_alias = ["All", "Acylic", "CHON", "\u226415 HA", ">15 HA"]
    label_list = ["i", "ii", "iii", "iv", "v"]
    
    plt.clf()
    
    fig = plt.figure(figsize=(10.5, 6))
    gs = fig.add_gridspec(2, 6, hspace=0, wspace=0)
    axs = gs.subplots(sharex=True, sharey=True)
    
    for iter in tqdm(range(0,11)):
        if iter <= 5:
            ax = axs[iter//5, iter%5]
        else:
            ax = axs[iter//6, iter%6 + 1]
        
        for i, category in enumerate(categories_to_plot):
            iter_category_accuracy_list = iter_category_k_fold_accuracy_dict_dict[iter][category]
            iter_category_accuracy_list = np.array(iter_category_accuracy_list)
            
            median = np.median(iter_category_accuracy_list)
            q1 = np.percentile(iter_category_accuracy_list, 25)
            q3 = np.percentile(iter_category_accuracy_list, 75)
            
            ax.bar(i, median, width=0.75, color=vs_colors["orange"][0], alpha=0.5, label=f"{label_list[i]}. {categories_alias[i]}")
            ax.bar(i, median, width=0.75, yerr=[[median-q1], [q3-median]], capsize=4,  edgecolor='black', linewidth=1, facecolor='none')
            
        # ax.set_title(f"Iter {iter}", fontsize=12)
        ax.text(0.03, 0.94, f"Iter {iter}", horizontalalignment='left', verticalalignment='center', transform=ax.transAxes, fontsize=14)
        
        ax.grid(which='major', axis='y', color='gray', linestyle='-', linewidth=0.5, alpha=0.5, zorder=0)
        ax.grid(which='minor', axis='y', color='gray', linestyle=':', linewidth=0.5, alpha=0.3, zorder=0)
        ax.set_axisbelow(True)
        
        ax.minorticks_on()
        ax.xaxis.set_minor_locator(plt.NullLocator())
        
        ## set y label in iter 4 plot
        if iter == 0:
            ax.set_ylabel("Pairwise Accuracy", fontsize=14)
            ax.yaxis.set_label_coords(-0.3, 0.0)
            
        ## set x label between iter 8 and 10 plots
        if iter == 8:
            ax.set_xlabel("Molecule Category", fontsize=14)
            ax.xaxis.set_label_coords(0.0, -0.35)
        
        ax.set_yticks(np.arange(0.1, 1.1, 0.1))
        ax.set_ylim(0.0, 1.0)
        
        ax.tick_params(axis='both', which='both', labelsize=14, direction='in')
        
        ax.set_xticks(range(len(categories_to_plot)))
        ax.set_xticklabels(categories_alias, fontsize=14, rotation=90)
    
    fig.delaxes(axs[0, 5])
    
    # plt.savefig(os.path.join(plots_path, "SI_small_molec_category_k_fold_accuracy_rmg_active_learning_iterations_models.svg"), dpi=600, bbox_inches='tight', transparent=True, pad_inches=0.005)
    # plt.savefig(os.path.join(plots_path, "SI_small_molec_category_k_fold_accuracy_rmg_active_learning_iterations_models.png"), dpi=600, bbox_inches='tight', transparent=True, pad_inches=0.005)
    plt.savefig(os.path.join(plots_path, "SI_small_molec_category_k_fold_accuracy_rmg_active_learning_iterations_models.pdf"), dpi=600, bbox_inches='tight', transparent=True, pad_inches=0.005)
    plt.close()
    
    return


def return_iter_category_k_fold_accuracy_dict_dict_iter_polymer(iter, category_list=None, category_smi_list_dict=None, smi_tp_dict=None, smi_td_dict=None, smi_tp_td_mean_dict=None, oligomer_len=3):
    
    iter_category_k_fold_accuracy_dict_dict_iter_tp = {}
    iter_category_k_fold_accuracy_dict_dict_iter_td = {}
    iter_category_k_fold_accuracy_dict_dict_iter_tp_td_mean = {}
    
    if oligomer_len == 3:
        polymer_preds_dict_path = os.path.join(expt_polymer_data_path,
                                                "trimer_preds_rmg_active_learning_iterations_models",
                                                f"k_fold_smi_preds_dict_iteration_{iter}.pkl")
    elif oligomer_len == 2:
        polymer_preds_dict_path = os.path.join(expt_polymer_data_path,
                                                "dimer_preds_rmg_active_learning_iterations_models",
                                                f"k_fold_smi_preds_dict_iteration_{iter}.pkl")
    elif oligomer_len == 4:
        polymer_preds_dict_path = os.path.join(expt_polymer_data_path,
                                                "tetramer_preds_rmg_active_learning_iterations_models",
                                                f"k_fold_smi_preds_dict_iteration_{iter}.pkl")
    else:
        print("Invalid oligomer length. Choose 2, 3, or 4. Exiting.")
        return {}, {}, {}
    
    polymer_preds_dict = pickle.load(open(polymer_preds_dict_path, "rb"))
    thresh_poly = 20.0  ## Threshold for pairwise accuracy calculation. 2*expt_error (10.0 from data source)
    for category in tqdm(category_list):
        category_smi_list = category_smi_list_dict[category]
        
        category_tp_list = [smi_tp_dict[smi] for smi in category_smi_list]
        category_td_list = [smi_td_dict[smi] for smi in category_smi_list]
        category_tp_td_mean_list = [smi_tp_td_mean_dict[smi] for smi in category_smi_list]
        
        category_tp_preds_list_list = [[k_smi_preds_dict[smi] for smi in category_smi_list] for k, k_smi_preds_dict in polymer_preds_dict.items()]
        category_tp_preds_accuracy_list = [get_pairwise_accuracy(category_tp_list, category_tp_preds_list, thresh_poly) for category_tp_preds_list in category_tp_preds_list_list]
        category_tp_preds_accuracy_list = [acc[0]/acc[1] for acc in category_tp_preds_accuracy_list]
        
        category_td_preds_list_list = [[k_smi_preds_dict[smi] for smi in category_smi_list] for k, k_smi_preds_dict in polymer_preds_dict.items()]
        category_td_preds_accuracy_list = [get_pairwise_accuracy(category_td_list, category_td_preds_list, thresh_poly) for category_td_preds_list in category_td_preds_list_list]
        category_td_preds_accuracy_list = [acc[0]/acc[1] for acc in category_td_preds_accuracy_list]
        
        category_tp_td_mean_preds_list_list = [[k_smi_preds_dict[smi] for smi in category_smi_list] for k, k_smi_preds_dict in polymer_preds_dict.items()]
        category_tp_td_mean_preds_accuracy_list = [get_pairwise_accuracy(category_tp_td_mean_list, category_tp_td_mean_preds_list, thresh_poly) for category_tp_td_mean_preds_list in category_tp_td_mean_preds_list_list]
        category_tp_td_mean_preds_accuracy_list = [acc[0]/acc[1] for acc in category_tp_td_mean_preds_accuracy_list]
        
        iter_category_k_fold_accuracy_dict_dict_iter_tp[category] = category_tp_preds_accuracy_list
        iter_category_k_fold_accuracy_dict_dict_iter_td[category] = category_td_preds_accuracy_list
        iter_category_k_fold_accuracy_dict_dict_iter_tp_td_mean[category] = category_tp_td_mean_preds_accuracy_list
        
    return iter_category_k_fold_accuracy_dict_dict_iter_tp, iter_category_k_fold_accuracy_dict_dict_iter_td, iter_category_k_fold_accuracy_dict_dict_iter_tp_td_mean
    

def expt_polymer_data_analysis(oligomer_len=3):
    ## oligomer_len default is 3. trimer in main text. dimer and tetramer in SI
    if oligomer_len == 3:
        poly_abbr_oligomer_dict_path = os.path.join(expt_polymer_data_path, "polymer_abbr_linear_trimer_smi_dict.json")
    elif oligomer_len == 2:
        poly_abbr_oligomer_dict_path = os.path.join(expt_polymer_data_path, "polymer_abbr_linear_dimer_smi_dict.json")
    elif oligomer_len == 4:
        poly_abbr_oligomer_dict_path = os.path.join(expt_polymer_data_path, "polymer_abbr_linear_tetramer_smi_dict.json")
    else:
        print("Invalid oligomer length. Choose 2, 3, or 4. Exiting.")
        return {}, {}, {}
        
    poly_abbr_oligomer_dict = json.load(open(poly_abbr_oligomer_dict_path, "r"))
    
    chain_growth = ['PBDR', 'PBR', 'PCR', 'PCTFE', 'PECTFE', 'PETFE', 'PNR',
                    'P3FE', 'PA6', 'PAM', 'PAN', 'PBMA', 'PCL', 'PE-HD', 'PEA',
                    'PEMA', 'PEO', 'PMMA', 'PMS', 'POM', 'PP', 'PPOX', 'PS',
                    'PTFE', 'PVAC', 'PVC', 'PVDC', 'PVDF', 'PVF', 'PVK']
    
    step_growth = ['PA12', 'PA610', 'PA612', 'PA66', 'PEN', 'PVOH',
                'PPO', 'PX', 'PBT', 'PET', 'PVB']
    
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
        
        if (smi_ha_dict["O"] > 0 or smi_ha_dict["N"] > 0) and smi_ha_dict["Cl"] == 0 and smi_ha_dict["F"] == 0:
            category_smi_list_dict["CHON All"].append(smi)
    
    iter_list = list(range(11))
    mp_cores = len(os.sched_getaffinity(0))
    with mp.Pool(mp_cores) as pool:
        iter_category_k_fold_accuracy_dict_list = pool.starmap(return_iter_category_k_fold_accuracy_dict_dict_iter_polymer, [(iter, category_list, category_smi_list_dict, smi_tp_dict, smi_td_dict, smi_tp_td_mean_dict, oligomer_len) for iter in iter_list])
        
    iter_category_k_fold_accuracy_dict_dict_tp = dict(zip(iter_list, [iter_category_k_fold_accuracy_dict[0] for iter_category_k_fold_accuracy_dict in iter_category_k_fold_accuracy_dict_list]))
    iter_category_k_fold_accuracy_dict_dict_td = dict(zip(iter_list, [iter_category_k_fold_accuracy_dict[1] for iter_category_k_fold_accuracy_dict in iter_category_k_fold_accuracy_dict_list]))
    iter_category_k_fold_accuracy_dict_dict_tp_td_mean = dict(zip(iter_list, [iter_category_k_fold_accuracy_dict[2] for iter_category_k_fold_accuracy_dict in iter_category_k_fold_accuracy_dict_list]))
    
    return iter_category_k_fold_accuracy_dict_dict_tp, iter_category_k_fold_accuracy_dict_dict_td, iter_category_k_fold_accuracy_dict_dict_tp_td_mean


def plot_polymer_category_k_fold_accuracy(iter_category_k_fold_accuracy_dict_dict_tp, iter_category_k_fold_accuracy_dict_dict_td, iter_category_k_fold_accuracy_dict_dict_tp_td_mean):
    categories_to_plot = ["All", "Chain Growth", "Step Growth", "CH", "CHX Only", "CHON All"]
    categories_alias = ["All", "Chain\nGrowth", "Step\nGrowth", "CH", "CHX", "CHON"]
    label_list = ["i", "ii", "iii", "iv", "v", "vi"]
    
    plt.clf()
    
    fig = plt.figure(figsize=(6.0, 8.5))
    # gs = fig.add_gridspec(2, 6, hspace=0, wspace=0)
    gs = fig.add_gridspec(3, 2, hspace=0, wspace=0)
    axs = gs.subplots(sharex=True, sharey=True)
    
    bwidth = 0.8
    
    for iter in tqdm(range(0,11,2)):
        in_ = iter//2
        ax = axs[in_//2, in_%2]
        
        for i, category in enumerate(categories_to_plot):
            iter_category_accuracy_list_tp = iter_category_k_fold_accuracy_dict_dict_tp[iter][category]
            iter_category_accuracy_list_td = iter_category_k_fold_accuracy_dict_dict_td[iter][category]
            iter_category_accuracy_list_tp_td_mean = iter_category_k_fold_accuracy_dict_dict_tp_td_mean[iter][category]
            
            iter_category_accuracy_list_tp = np.array(iter_category_accuracy_list_tp)
            iter_category_accuracy_list_td = np.array(iter_category_accuracy_list_td)
            iter_category_accuracy_list_tp_td_mean = np.array(iter_category_accuracy_list_tp_td_mean)
            
            median_tp = np.median(iter_category_accuracy_list_tp)
            q1_tp = np.percentile(iter_category_accuracy_list_tp, 25)
            q3_tp = np.percentile(iter_category_accuracy_list_tp, 75)
            
            median_td = np.median(iter_category_accuracy_list_td)
            q1_td = np.percentile(iter_category_accuracy_list_td, 25)
            q3_td = np.percentile(iter_category_accuracy_list_td, 75)
            
            median_tp_td_mean = np.median(iter_category_accuracy_list_tp_td_mean)
            q1_tp_td_mean = np.percentile(iter_category_accuracy_list_tp_td_mean, 25)
            q3_tp_td_mean = np.percentile(iter_category_accuracy_list_tp_td_mean, 75)
            
            ax.bar(3*i-bwidth-0.05, median_tp, width=bwidth, color=vs_colors["blue"][0], alpha=0.5)
            ax.bar(3*i-bwidth-0.05, median_tp, width=bwidth, yerr=[[median_tp-q1_tp], [q3_tp-median_tp]], capsize=3,  edgecolor=vs_colors["blue"][0], linewidth=1, facecolor='none')
            
            ax.bar(3*i, median_td, width=bwidth, color=vs_colors["red"][0], alpha=0.5)
            ax.bar(3*i, median_td, width=bwidth, yerr=[[median_td-q1_td], [q3_td-median_td]], capsize=3,  edgecolor=vs_colors["red"][0], linewidth=1, facecolor='none')
            
            ax.bar(3*i+bwidth+0.05, median_tp_td_mean, width=bwidth, color=vs_colors["green"][0], alpha=0.5)
            ax.bar(3*i+bwidth+0.05, median_tp_td_mean, width=bwidth, yerr=[[median_tp_td_mean-q1_tp_td_mean], [q3_tp_td_mean-median_tp_td_mean]], capsize=3,  edgecolor=vs_colors["green"][0], linewidth=1, facecolor='none')
            
        ax.text(0.03, 0.94, f"Iter {iter}", horizontalalignment='left', verticalalignment='center', transform=ax.transAxes, fontsize=14)
        
        ax.grid(which='major', axis='y', color='gray', linestyle='-', linewidth=0.5, alpha=0.5, zorder=0)
        ax.grid(which='minor', axis='y', color='gray', linestyle=':', linewidth=0.5, alpha=0.3, zorder=0)
        ax.set_axisbelow(True)
        
        ax.minorticks_on()
        ax.xaxis.set_minor_locator(plt.NullLocator())
        
        if iter == 4:
            ax.set_ylabel("Pairwise Accuracy", fontsize=14)
            
        if iter == 8:
            ax.set_xlabel("Polymer Category", fontsize=14)
            ax.xaxis.set_label_coords(1.0, -0.4)
            
        ax.set_yticks(np.arange(0.1, 1.1, 0.1))
        ax.set_ylim(0.0, 1.0)
        
        ax.tick_params(axis='both', which='both', labelsize=14, direction='in')
        
        ax.set_xticks(3*np.arange(len(categories_to_plot)))
        ax.set_xticklabels(categories_alias, fontsize=14, rotation=90)
        
    tp_patch = mpatches.Patch(color=vs_colors["blue"][0], alpha=0.5, label="Tp")
    td_patch = mpatches.Patch(color=vs_colors["red"][0], alpha=0.5, label="Td")
    tp_td_mean_patch = mpatches.Patch(color=vs_colors["green"][0], alpha=0.5, label="(Tp+Td)/2")
    
    legend = fig.legend(handles=[tp_patch, td_patch, tp_td_mean_patch], loc='upper center', bbox_to_anchor=(0.5, 1.02), ncol=3, fontsize=12)
    
    plt.tight_layout()
    # plt.subplots_adjust(top=0.85)
    
    # plt.savefig(os.path.join(plots_path, "polymer_category_k_fold_accuracy_rmg_active_learning_iterations_models.svg"), dpi=600, bbox_inches='tight', transparent=True, pad_inches=0.005)
    # plt.savefig(os.path.join(plots_path, "polymer_category_k_fold_accuracy_rmg_active_learning_iterations_models.png"), dpi=600, bbox_inches='tight', transparent=True, pad_inches=0.005)
    plt.savefig(os.path.join(plots_path, "polymer_category_k_fold_accuracy_rmg_active_learning_iterations_models.pdf"), dpi=600, bbox_inches='tight', transparent=True, pad_inches=0.005)
    plt.close()
    
    return


def SI_plot_polymer_category_k_fold_accuracy(iter_category_k_fold_accuracy_dict_dict_tp, iter_category_k_fold_accuracy_dict_dict_td, iter_category_k_fold_accuracy_dict_dict_tp_td_mean, oligomer_len=3):
    categories_to_plot = ["All", "Chain Growth", "Step Growth", "CH", "CHX Only", "CHON All"]
    categories_alias = ["All", "Chain\nGrowth", "Step\nGrowth", "CH", "CHX", "CHON"]
    label_list = ["i", "ii", "iii", "iv", "v", "vi"]
    
    plt.clf()
    
    fig = plt.figure(figsize=(12.0, 8.5))
    # gs = fig.add_gridspec(3, 2, hspace=0, wspace=0)
    gs = fig.add_gridspec(3, 4, hspace=0, wspace=0)
    axs = gs.subplots(sharex=True, sharey=True)
    
    bwidth = 0.8
    
    for iter in tqdm(range(0,11)):
        if iter < 3:
            ax = axs[iter//3, iter%3]
        else:
            it1 = iter+1
            ax = axs[it1//4, it1%4]
        
        for i, category in enumerate(categories_to_plot):
            iter_category_accuracy_list_tp = iter_category_k_fold_accuracy_dict_dict_tp[iter][category]
            iter_category_accuracy_list_td = iter_category_k_fold_accuracy_dict_dict_td[iter][category]
            iter_category_accuracy_list_tp_td_mean = iter_category_k_fold_accuracy_dict_dict_tp_td_mean[iter][category]
            
            iter_category_accuracy_list_tp = np.array(iter_category_accuracy_list_tp)
            iter_category_accuracy_list_td = np.array(iter_category_accuracy_list_td)
            iter_category_accuracy_list_tp_td_mean = np.array(iter_category_accuracy_list_tp_td_mean)
            
            median_tp = np.median(iter_category_accuracy_list_tp)
            q1_tp = np.percentile(iter_category_accuracy_list_tp, 25)
            q3_tp = np.percentile(iter_category_accuracy_list_tp, 75)
            
            median_td = np.median(iter_category_accuracy_list_td)
            q1_td = np.percentile(iter_category_accuracy_list_td, 25)
            q3_td = np.percentile(iter_category_accuracy_list_td, 75)
            
            median_tp_td_mean = np.median(iter_category_accuracy_list_tp_td_mean)
            q1_tp_td_mean = np.percentile(iter_category_accuracy_list_tp_td_mean, 25)
            q3_tp_td_mean = np.percentile(iter_category_accuracy_list_tp_td_mean, 75)
            
            ax.bar(3*i-bwidth-0.05, median_tp, width=bwidth, color=vs_colors["blue"][0], alpha=0.5)
            ax.bar(3*i-bwidth-0.05, median_tp, width=bwidth, yerr=[[median_tp-q1_tp], [q3_tp-median_tp]], capsize=3,  edgecolor=vs_colors["blue"][0], linewidth=1, facecolor='none')
            
            ax.bar(3*i, median_td, width=bwidth, color=vs_colors["red"][0], alpha=0.5)
            ax.bar(3*i, median_td, width=bwidth, yerr=[[median_td-q1_td], [q3_td-median_td]], capsize=3,  edgecolor=vs_colors["red"][0], linewidth=1, facecolor='none')
            
            ax.bar(3*i+bwidth+0.05, median_tp_td_mean, width=bwidth, color=vs_colors["green"][0], alpha=0.5)
            ax.bar(3*i+bwidth+0.05, median_tp_td_mean, width=bwidth, yerr=[[median_tp_td_mean-q1_tp_td_mean], [q3_tp_td_mean-median_tp_td_mean]], capsize=3,  edgecolor=vs_colors["green"][0], linewidth=1, facecolor='none')
            
        ax.text(0.03, 0.94, f"Iter {iter}", horizontalalignment='left', verticalalignment='center', transform=ax.transAxes, fontsize=14)
        
        ax.grid(which='major', axis='y', color='gray', linestyle='-', linewidth=0.5, alpha=0.5, zorder=0)
        ax.grid(which='minor', axis='y', color='gray', linestyle=':', linewidth=0.5, alpha=0.3, zorder=0)
        ax.set_axisbelow(True)
        
        ax.minorticks_on()
        ax.xaxis.set_minor_locator(plt.NullLocator())
        
        if iter == 3:
            ax.set_ylabel("Pairwise Accuracy", fontsize=14)
            
        if iter == 8:
            ax.set_xlabel("Polymer Category", fontsize=14)
            ax.xaxis.set_label_coords(1.0, -0.35)
            
        ax.set_yticks(np.arange(0.1, 1.1, 0.1))
        ax.set_ylim(0.0, 1.0)
        
        ax.tick_params(axis='both', which='both', labelsize=14, direction='in')
        
        ax.set_xticks(3*np.arange(len(categories_to_plot)))
        ax.set_xticklabels(categories_alias, fontsize=14, rotation=90)
    
    fig.delaxes(axs[0, 3])
    
    tp_patch = mpatches.Patch(color=vs_colors["blue"][0], alpha=0.5, label="Tp")
    td_patch = mpatches.Patch(color=vs_colors["red"][0], alpha=0.5, label="Td")
    tp_td_mean_patch = mpatches.Patch(color=vs_colors["green"][0], alpha=0.5, label="(Tp+Td)/2")
    
    legend = fig.legend(handles=[tp_patch, td_patch, tp_td_mean_patch], loc='upper right', bbox_to_anchor=(0.94, 0.89), ncol=1, fontsize=12)
    
    plt.tight_layout()
    # plt.subplots_adjust(top=0.85)
    
    if oligomer_len == 3:
        plot_prefix = "trimer"
    elif oligomer_len == 2:
        plot_prefix = "dimer"
    elif oligomer_len == 4:
        plot_prefix = "tetramer"
    else:
        print("Invalid oligomer length. Choose 2, 3, or 4. Exiting.")
        return
    
    # plt.savefig(os.path.join(plots_path, f"SI_{plot_prefix}_category_k_fold_accuracy_rmg_active_learning_iterations_models.svg"), dpi=600, bbox_inches='tight', transparent=True, pad_inches=0.005)
    # plt.savefig(os.path.join(plots_path, f"SI_{plot_prefix}_category_k_fold_accuracy_rmg_active_learning_iterations_models.png"), dpi=600, bbox_inches='tight', transparent=True, pad_inches=0.005)
    plt.savefig(os.path.join(plots_path, f"SI_{plot_prefix}_category_k_fold_accuracy_rmg_active_learning_iterations_models.pdf"), dpi=600, bbox_inches='tight', transparent=True, pad_inches=0.005)
    plt.close()
    
    return


def main():
    iter_category_k_fold_accuracy_dict_dict = expt_small_molec_data_analysis()
    plot_small_molec_category_k_fold_accuracy(iter_category_k_fold_accuracy_dict_dict)
    SI_plot_small_molec_category_k_fold_accuracy(iter_category_k_fold_accuracy_dict_dict)
    
    iter_category_k_fold_accuracy_dict_dict_tp, iter_category_k_fold_accuracy_dict_dict_td, iter_category_k_fold_accuracy_dict_dict_tp_td_mean = expt_polymer_data_analysis()
    plot_polymer_category_k_fold_accuracy(iter_category_k_fold_accuracy_dict_dict_tp, iter_category_k_fold_accuracy_dict_dict_td, iter_category_k_fold_accuracy_dict_dict_tp_td_mean)
    
    iter_category_k_fold_accuracy_dict_dict_tp_dimer, iter_category_k_fold_accuracy_dict_dict_td_dimer, iter_category_k_fold_accuracy_dict_dict_tp_td_mean_dimer = expt_polymer_data_analysis(oligomer_len=2)
    SI_plot_polymer_category_k_fold_accuracy(iter_category_k_fold_accuracy_dict_dict_tp_dimer, iter_category_k_fold_accuracy_dict_dict_td_dimer, iter_category_k_fold_accuracy_dict_dict_tp_td_mean_dimer, oligomer_len=2)
    
    iter_category_k_fold_accuracy_dict_dict_tp_trimer, iter_category_k_fold_accuracy_dict_dict_td_trimer, iter_category_k_fold_accuracy_dict_dict_tp_td_mean_trimer = expt_polymer_data_analysis(oligomer_len=3)
    SI_plot_polymer_category_k_fold_accuracy(iter_category_k_fold_accuracy_dict_dict_tp_trimer, iter_category_k_fold_accuracy_dict_dict_td_trimer, iter_category_k_fold_accuracy_dict_dict_tp_td_mean_trimer, oligomer_len=3)
    
    iter_category_k_fold_accuracy_dict_dict_tp_tetramer, iter_category_k_fold_accuracy_dict_dict_td_tetramer, iter_category_k_fold_accuracy_dict_dict_tp_td_mean_tetramer = expt_polymer_data_analysis(oligomer_len=4)
    SI_plot_polymer_category_k_fold_accuracy(iter_category_k_fold_accuracy_dict_dict_tp_tetramer, iter_category_k_fold_accuracy_dict_dict_td_tetramer, iter_category_k_fold_accuracy_dict_dict_tp_td_mean_tetramer, oligomer_len=4)
    
    return


if __name__ == "__main__":
    main()
