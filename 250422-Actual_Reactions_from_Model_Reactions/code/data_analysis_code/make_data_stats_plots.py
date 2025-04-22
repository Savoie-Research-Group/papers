## rxn feature distribution bar plot. ha count histogram, mr ar count histogram. distribution of ea values (both mr and reactions on same plot)
"""
    Date Modified: 2025/22/04
    Author: Veerupaksh (Veeru) Singla (singla2@purdue.edu)
    Description: Create MR/AR stats related to feature distribution, HA count, Ea distribution.
"""

import os
import json
import numpy as np
from scipy import stats


this_script_dir = os.path.dirname(os.path.realpath(__file__))
os.chdir(this_script_dir)


from utils import *


def ea_dist_hist(ar_ea_fwd_dict, ar_ea_rev_dict, mr_ea_fwd_dict, mr_ea_rev_dict, save_name):
    bw = 25
    
    mr_ea_list_fwd = np.array(list(mr_ea_fwd_dict.values()))
    mr_ea_list_rev = np.array(list(mr_ea_rev_dict.values()))
    ar_ea_list_fwd = np.array(list(ar_ea_fwd_dict.values()))
    ar_ea_list_rev = np.array(list(ar_ea_rev_dict.values()))
    
    min_ea = 0
    max_ea = 175
    
    min_count = 0
    max_mr_count = 165
    max_ar_count = 5200
    
    bins = np.arange(min_ea, max_ea + 1, bw)
    
    plt.clf()
    fig = plt.figure(figsize=(3.25, 4.5))
    gs = fig.add_gridspec(2, 2, hspace=0.0, wspace=0)
    
    ax1 = fig.add_subplot(gs[0, 0])  ## mr fwd
    ax2 = fig.add_subplot(gs[0, 1])  #, sharey=ax1)  ## mr rev
    ax3 = fig.add_subplot(gs[1, 0])  #, sharex=ax1)  ## ar fwd
    ax4 = fig.add_subplot(gs[1, 1])  #, sharex=ax2, sharey=ax3)  ## ar rev
    
    mr_fwd_kde = stats.gaussian_kde(mr_ea_list_fwd)
    mr_rev_kde = stats.gaussian_kde(mr_ea_list_rev)
    ar_fwd_kde = stats.gaussian_kde(ar_ea_list_fwd)
    ar_rev_kde = stats.gaussian_kde(ar_ea_list_rev)
    
    x_ea = np.linspace(min_ea, max_ea, 2000)
    
    kde_mr_fwd = mr_fwd_kde(x_ea)
    kde_mr_rev = mr_rev_kde(x_ea)
    kde_ar_fwd = ar_fwd_kde(x_ea)
    kde_ar_rev = ar_rev_kde(x_ea)
    
    n1, bins1, patches1 = ax1.hist(mr_ea_list_fwd, bins=bins,
                                edgecolor="none",
                                linewidth=0.0,
                                facecolor="none")  # Custom hex color and alpha
    kde_mr_fwd = kde_mr_fwd / kde_mr_fwd.max() * max(n1)
    ax1.fill_between(x_ea, 0.0, 0.0 + kde_mr_fwd,
                    alpha=0.5, color=vs_colors["blue"][0])
    ax1.vlines(mr_ea_list_fwd, -kde_mr_fwd.max()/10, -kde_mr_fwd.max()/8 + kde_mr_fwd.max()/10,
            color=vs_colors["blue"][0], alpha=0.15, linewidth=0.5)
    ax1.hist(mr_ea_list_fwd, bins=bins1,
            edgecolor="black", linewidth=1.0, facecolor="none")
    for i in range(len(n1)):
        ax1.text(bins1[i] + bw/2, n1[i] + 1, str(int(n1[i])), ha='center', va='bottom', fontsize=8,
                 rotation=90, bbox=dict(facecolor='white', edgecolor='none', pad=0, alpha=0.8))
    ax1.set_xlim(min_ea, max_ea)
    ax1.set_xticks(range(0, max_ea+1, 25))
    ax1.set_xticklabels(["" for i in range(0, max_ea+1, 25)])
    ax1.set_ylabel("MR Count", fontsize=14)
    ax1.set_yticks(range(0, max_mr_count+1, 25))
    ax1.set_yticklabels([str(i) for i in range(0, max_mr_count+1, 25)])
    
    n2, bins2, patches2 = ax2.hist(mr_ea_list_rev, bins=bins,
                                edgecolor="none",
                                linewidth=0.0,
                                facecolor="none")  # Custom hex color and alpha
    kde_mr_rev = kde_mr_rev / kde_mr_rev.max() * max(n2)
    ax2.fill_between(x_ea, 0.0, 0.0 + kde_mr_rev,
                    alpha=0.5, color=vs_colors["blue"][0])
    ax2.vlines(mr_ea_list_rev, -kde_mr_rev.max()/10, -kde_mr_rev.max()/8 + kde_mr_rev.max()/10,
            color=vs_colors["blue"][0], alpha=0.15, linewidth=0.5)
    ax2.hist(mr_ea_list_rev, bins=bins2,
            edgecolor="black", linewidth=1.0, facecolor="none")
    for i in range(len(n2)):
        ax2.text(bins2[i] + bw/2, n2[i] + 1, str(int(n2[i])), ha='center', va='bottom', fontsize=8,
                 rotation=90, bbox=dict(facecolor='white', edgecolor='none', pad=0, alpha=0.8))
    ax2.set_xlim(min_ea, max_ea)
    ax2.set_xticks(range(0, max_ea+1, 25))
    ax2.set_xticklabels(["" for i in range(0, max_ea+1, 25)])
    ax1.set_ylim(min(-kde_mr_fwd.max()/7, -kde_mr_rev.max()/7), max_mr_count) ## put here since kde_mr_rev is calculated after ax1
    ax2.set_ylim(min(-kde_mr_fwd.max()/7, -kde_mr_rev.max()/7), max_mr_count)
    ax2.set_yticks(range(0, max_mr_count+1, 25))
    ax2.set_yticklabels(["" for i in range(0, max_mr_count+1, 25)])
    
    n3, bins3, patches3 = ax3.hist(ar_ea_list_fwd, bins=bins,
                                edgecolor="none",
                                linewidth=0.0,
                                facecolor="none")  # Custom hex color and alpha
    kde_ar_fwd = kde_ar_fwd / kde_ar_fwd.max() * max(n3)
    ax3.fill_between(x_ea, 0.0, 0.0 + kde_ar_fwd,
                    alpha=0.5, color=vs_colors["red"][0])
    ax3.vlines(ar_ea_list_fwd, -kde_ar_fwd.max()/10, -kde_ar_fwd.max()/8 + kde_ar_fwd.max()/10,
            color=vs_colors["red"][0], alpha=0.025, linewidth=0.5)
    ax3.hist(ar_ea_list_fwd, bins=bins3,
            edgecolor="black", linewidth=1.0, facecolor="none")
    for i in range(len(n3)):
        ax3.text(bins3[i] + bw/2, n3[i] + 50, str(int(n3[i])), ha='center', va='bottom', fontsize=8,
                 rotation=90, bbox=dict(facecolor='white', edgecolor='none', pad=0, alpha=0.8))
    ax3.set_xlabel("Ea Fwd (kcal/mol)", fontsize=14)
    ax3.set_xlim(min_ea, max_ea)
    ax3.set_xticks(range(0, max_ea+1, 25))
    ax3.set_xticklabels([str(i) if i%50==0 else "" for i in range(0, max_ea+1, 25)], rotation=90)
    ax3.set_ylabel("AR Count", fontsize=14)
    ax3.set_yticks(range(0, max_ar_count+1, 1000))
    ax3.set_yticklabels([str(i) for i in range(0, max_ar_count+1, 1000)])
    
    n4, bins4, patches4 = ax4.hist(ar_ea_list_rev, bins=bins,
                                edgecolor="none",
                                linewidth=0.0,
                                facecolor="none")  # Custom hex color and alpha
    kde_ar_rev = kde_ar_rev / kde_ar_rev.max() * max(n4)
    ax4.fill_between(x_ea, 0.0, 0.0 + kde_ar_rev,
                    alpha=0.5, color=vs_colors["red"][0])
    ax4.vlines(ar_ea_list_rev, -kde_ar_rev.max()/10, -kde_ar_rev.max()/8 + kde_ar_rev.max()/10,
            color=vs_colors["red"][0], alpha=0.025, linewidth=0.5)
    ax4.hist(ar_ea_list_rev, bins=bins4,
            edgecolor="black", linewidth=1.0, facecolor="none")
    for i in range(len(n4)):
        ax4.text(bins4[i] + bw/2, n4[i] + 50, str(int(n4[i])), ha='center', va='bottom', fontsize=8,
                 rotation=90, bbox=dict(facecolor='white', edgecolor='none', pad=0, alpha=0.8))
    ax4.set_xlabel("Ea Rev (kcal/mol)", fontsize=14)
    ax4.set_xlim(min_ea, max_ea)
    ax4.set_xticks(range(0, max_ea+1, 25))
    ax4.set_xticklabels([str(i) if i%50==0 else "" for i in range(0, max_ea+1, 25)], rotation=90)
    ax3.set_ylim(min(-kde_ar_fwd.max()/7, -kde_ar_rev.max()/7), max_ar_count) ## put here since kde_ar_rev is calculated after ax3
    ax4.set_ylim(min(-kde_ar_fwd.max()/7, -kde_ar_rev.max()/7), max_ar_count)
    ax4.set_yticks(range(0, max_ar_count+1, 1000))
    ax4.set_yticklabels(["" for i in range(0, max_ar_count+1, 1000)])
    
    for ax in (ax1, ax2, ax3, ax4):
        ax.grid(which='major', axis='y', color='gray', linestyle='-', linewidth=0.5, alpha=0.5, zorder=0)
        ax.grid(which='minor', axis='y', color='gray', linestyle=':', linewidth=0.5, alpha=0.3, zorder=0)
        ax.set_axisbelow(True)
        ax.minorticks_on()
        ax.xaxis.set_minor_locator(plt.NullLocator())
        ax.tick_params(axis='both', which='both', labelsize=14, direction='in')
        
    plt.tight_layout()
    plt.savefig(transparent=True, fname=save_name + ".pdf", dpi=300, bbox_inches='tight', pad_inches=0.005)
    plt.savefig(transparent=True, fname=save_name + ".svg", dpi=300, bbox_inches='tight', pad_inches=0.005)
    
    return
    

def mr_ar_dist_hist(mr_ar_count_dict, save_name):
    ## distribution showing number of ars corresponding to each mr
    bw = 5
    
    mr_ar_count_list = np.array(list(mr_ar_count_dict.values()))
    
    ## ar on x-axis, mr on y-axis
    min_ar_count = 0
    max_ar_count = 70
    min_mr_count = 0
    max_mr_count = 64
    
    bins = np.arange(min_ar_count, max_ar_count + 1, bw)
    
    plt.clf()
    fig = plt.figure(figsize=(3.25, 2.5))
    gs = fig.add_gridspec(1, 1, hspace=0.0, wspace=0)
    
    ax1 = fig.add_subplot(gs[0, 0])
    
    kde_mr_ar_count = stats.gaussian_kde(mr_ar_count_list)
    x_ar_count = np.linspace(min_ar_count, max_ar_count, 2000)
    kde_mr_ar_count = kde_mr_ar_count(x_ar_count)
    
    n1, bins1, patches1 = ax1.hist(mr_ar_count_list, bins=bins,
                                edgecolor="none",
                                linewidth=0.0,
                                facecolor="none")  # Custom hex color and alpha
    kde_mr_ar_count = kde_mr_ar_count / kde_mr_ar_count.max() * max(n1)
    ax1.fill_between(x_ar_count, 0.0, 0.0 + kde_mr_ar_count,
                    alpha=0.5, color=vs_colors["orange"][0])
    ax1.vlines(mr_ar_count_list, -kde_mr_ar_count.max()/10, -kde_mr_ar_count.max()/8 + kde_mr_ar_count.max()/10,
            color=vs_colors["orange"][0], alpha=0.1, linewidth=0.5)
    ax1.hist(mr_ar_count_list, bins=bins1,
            edgecolor="black", linewidth=1.0, facecolor="none")
    for i in range(len(n1)):
        ax1.text(bins1[i] + bw/2, n1[i] + 0.5, str(int(n1[i])), ha='center', va='bottom', fontsize=8,
                 rotation=0, bbox=dict(facecolor='white', edgecolor='none', pad=0, alpha=0.5))
    ax1.set_xlabel("AR Count", fontsize=14)
    ax1.set_xlim(min_ar_count, max_ar_count)
    ax1.set_xticks(range(0, max_ar_count+1, 5))
    ax1.set_xticklabels([str(i) if i%10==0 else "" for i in range(0, max_ar_count+1, 5)])
    ax1.set_ylabel("MR Count", fontsize=14)
    ax1.set_ylim(-kde_mr_ar_count.max()/7, max_mr_count)
    ax1.set_yticks(range(0, max_mr_count+1, 10))
    ax1.set_yticklabels([str(i) for i in range(0, max_mr_count+1, 10)])
    
    for ax in (ax1,):
        ax.grid(which='major', axis='y', color='gray', linestyle='-', linewidth=0.5, alpha=0.5, zorder=0)
        ax.grid(which='minor', axis='y', color='gray', linestyle=':', linewidth=0.5, alpha=0.3, zorder=0)
        ax.set_axisbelow(True)
        ax.minorticks_on()
        ax.xaxis.set_minor_locator(plt.NullLocator())
        ax.tick_params(axis='both', which='both', labelsize=14, direction='in')
        
    plt.tight_layout()
    plt.savefig(transparent=True, fname=save_name + ".pdf", dpi=300, bbox_inches='tight', pad_inches=0.005)
    plt.savefig(transparent=True, fname=save_name + ".svg", dpi=300, bbox_inches='tight', pad_inches=0.005)
    
    return


def mr_ar_ha_count_dist_hist(mr_ha_count_dict, ar_ha_count_dict, save_name):
    ## mr and ar heavy atom count distribution on same plot
    bw = 1
    
    mr_ha_count_list = np.array(list(mr_ha_count_dict.values()))
    ar_ha_count_list = np.array(list(ar_ha_count_dict.values()))
    
    min_ha_count_mr = 4
    max_ha_count_mr = 11
    min_ha_count_ar = 4
    max_ha_count_ar = 23
    
    min_rxn_count = 0
    max_rxn_count = 2700
    # max_rxn_count = 2100
    
    bins_mr  = np.arange(min_ha_count_mr, max_ha_count_mr + 1, bw)
    bins_ar = np.arange(min_ha_count_ar, max_ha_count_ar + 1, bw)
    
    plt.clf()
    fig = plt.figure(figsize=(4.5, 2))
    # fig = plt.figure(figsize=(7, 2))
    gs = fig.add_gridspec(1, 1, hspace=0.0, wspace=0)
    
    ax1 = fig.add_subplot(gs[0, 0])
    
    n2, bins2, patches2 = ax1.hist(ar_ha_count_list, bins=bins_ar,
                                edgecolor="none",
                                linewidth=0.0,
                                facecolor=vs_colors["red"][0], alpha=0.5, label="AR")  # Custom hex color and alpha
                                # facecolor=vs_colors["purple"][0], alpha=0.5, label="AR")  # Custom hex color and alpha
    ax1.hist(ar_ha_count_list, bins=bins2,
            edgecolor="black", linewidth=1.0, facecolor="none")
    
    n1, bins1, patches1 = ax1.hist(mr_ha_count_list, bins=bins_mr,
                                edgecolor="none",
                                linewidth=0.0,
                                facecolor=vs_colors["blue"][0], alpha=0.5, label="MR")  # Custom hex color and alpha
                                # facecolor=vs_colors["orange"][0], alpha=0.5, label="MR")  # Custom hex color and alpha
    ax1.hist(mr_ha_count_list, bins=bins1,
            edgecolor="black", linewidth=1.0, facecolor="none")
    
    for i in range(len(n1)):
        ## text for mr height based on rxn text height
        try:
            ax1.text(bins1[i] + bw/2, max(n1[i], n2[i]) + 700, str(int(n1[i])), ha='center', va='bottom', fontsize=8,
                    color=vs_colors["blue"][0], rotation=90, bbox=dict(facecolor='white', edgecolor='none', pad=0, alpha=0.8))
                    # color=vs_colors["orange"][0], rotation=0, bbox=dict(facecolor='white', edgecolor='none', pad=0, alpha=0.8))
        except:
            ax1.text(bins1[i] + bw/2, n2[i] + 700, str(int(n1[i])), ha='center', va='bottom', fontsize=8,
                    color=vs_colors["blue"][0], rotation=90, bbox=dict(facecolor='white', edgecolor='none', pad=0, alpha=0.8))
                    # color=vs_colors["orange"][0], rotation=0, bbox=dict(facecolor='white', edgecolor='none', pad=0, alpha=0.8))
    
    for i in range(len(n2)):
        try:
            ax1.text(bins2[i] + bw/2, max(n1[i], n2[i]) + 50, str(int(n2[i])), ha='center', va='bottom', fontsize=8,
                    color=vs_colors["red"][0], rotation=90, bbox=dict(facecolor='white', edgecolor='none', pad=0, alpha=0.8))
                    # color=vs_colors["purple"][0], rotation=0, bbox=dict(facecolor='white', edgecolor='none', pad=0, alpha=0.8))
        except:
            ax1.text(bins2[i] + bw/2, n2[i] + 50, str(int(n2[i])), ha='center', va='bottom', fontsize=8,
                    color=vs_colors["red"][0], rotation=90, bbox=dict(facecolor='white', edgecolor='none', pad=0, alpha=0.8))
                    # color=vs_colors["purple"][0], rotation=0, bbox=dict(facecolor='white', edgecolor='none', pad=0, alpha=0.8))
        
    ax1.set_xlabel("Heavy Atom Count", fontsize=14)
    ax1.set_xlim(min_ha_count_ar, max_ha_count_ar)
    ax1.set_xticks(range(min_ha_count_ar, max_ha_count_ar+1, 1))
    ax1.set_xticklabels([str(i) if i%2==0 else "" for i in range(min_ha_count_ar, max_ha_count_ar+1, 1)])
    ax1.set_ylabel("MR/AR Count", fontsize=14)
    ax1.set_ylim(0, max_rxn_count)
    ax1.set_yticks(range(0, max_rxn_count+1, 500))
    ax1.set_yticklabels([str(i) for i in range(0, max_rxn_count+1, 500)])
    
    for ax in (ax1,):
        ax.grid(which='major', axis='y', color='gray', linestyle='-', linewidth=0.5, alpha=0.5, zorder=0)
        ax.grid(which='minor', axis='y', color='gray', linestyle=':', linewidth=0.5, alpha=0.3, zorder=0)
        ax.set_axisbelow(True)
        ax.minorticks_on()
        ax.xaxis.set_minor_locator(plt.NullLocator())
        ax.tick_params(axis='both', which='both', labelsize=14, direction='in')
    
    # ax1.legend(loc="upper left", fontsize=12)
    ax1.legend([patches1[0], patches2[0]], ["MR", "AR"], loc="upper left", fontsize=12, borderaxespad=0.25)
    
    plt.tight_layout()
    plt.savefig(transparent=True, fname=save_name + ".pdf", dpi=600, bbox_inches='tight', pad_inches=0.005)
    plt.savefig(transparent=True, fname=save_name + ".svg", dpi=300, bbox_inches='tight', pad_inches=0.005)
    
    return
        

def rxn_features_fn_grps_dist_bar_plot(ar_smi_dict, ar_fn_groups_added_dict, save_name):
    fn_grp_roman_to_name = {"i": "    Anhydrides & Imides",
                            "ii": "   Cyclic Aromatics & Derivatives",
                            "iii": "  Amidines & Azos",
                            "iv": "  Nitrosos & Oximes",
                            "v": "   Amines, Imines, & Nitriles",
                            "vi": "  Acids, Alcohols, Aldehydes, Esters, Ethers, Ketones, & Peroxides",
                            "vii": " Amides & Isocyanates",
                            "viii": "Cycloalkanes & Derivatives"}
    
    fn_grp_roman_to_name["vi"] = "  Acids, Alcohols, Aldehydes, Esters,\n       Ethers, Ketones, & Peroxides"
    
    roman_fn_grp_class_dict = {
        "i": ["anhydrides_imides", 1],
        "ii": ["cyclic_aromatics_derivatives", 7],
        "iii": ["amidines_azos", 2],
        "iv": ["nitrosos_oximes", 3],
        "v": ["amines_imines_nitriles", 4],
        "vi": ["acids_alcohols_aldehydes_esters_ethers_ketones_peroxides", 5],
        "vii": ["amides_isocyanates", 6],
        "viii": ["cycloalkanes_derivatives", 8]
    }

    fn_grp_class_roman_dict = {v[1]: k for k, v in roman_fn_grp_class_dict.items()}
    
    features_count_dict = {
        "Fwd Unimolec": 0,
        "Fwd Bimolec": 0,
        "Fwd Trimolec": 0,
        "Rev Unimolec": 0,
        "Rev Bimolec": 0,
        "Rev Trimolec": 0,
        "i": 0,
        "ii": 0,
        "iii": 0,
        "iv": 0,
        "v": 0,
        "vi": 0,
        "vii": 0,
        "viii": 0
    }
    
    features_order = ["Fwd Unimolec", "Fwd Bimolec", "Fwd Trimolec", "Rev Unimolec", "Rev Bimolec", "Rev Trimolec",
                      "i", "ii", "iii", "iv", "v", "vi", "vii", "viii"]
        
    for ar, ar_smi in ar_smi_dict.items():
        react_smi, prod_smi = ar_smi.split(">>")
        react_molec = react_smi.count(".") + 1
        prod_molec = prod_smi.count(".") + 1
        
        if react_molec == 1:
            features_count_dict["Fwd Unimolec"] += 1
        elif react_molec == 2:
            features_count_dict["Fwd Bimolec"] += 1
        elif react_molec == 3:
            features_count_dict["Fwd Trimolec"] += 1
            
        if prod_molec == 1:
            features_count_dict["Rev Unimolec"] += 1
        elif prod_molec == 2:
            features_count_dict["Rev Bimolec"] += 1
        elif prod_molec == 3:
            features_count_dict["Rev Trimolec"] += 1
            
        for fn_added in ar_fn_groups_added_dict[ar]:
            fn_class_int = int(fn_added.split("_")[0])
            fn_class_roman = fn_grp_class_roman_dict[fn_class_int]
            features_count_dict[fn_class_roman] += 1
    
    print(features_count_dict)
    
    plt.clf()
    fig = plt.figure(figsize=(4.5, 4.25))
    gs = fig.add_gridspec(1, 1, hspace=0.0, wspace=0)
    
    ax1 = fig.add_subplot(gs[0, 0])
    
    ax1.bar(features_order, [features_count_dict[k] for k in features_order], edgecolor='none', linewidth=0.0, facecolor=vs_colors["green"][0], alpha=0.5)
    ax1.bar(features_order, [features_count_dict[k] for k in features_order], edgecolor='black', linewidth=1.0, facecolor='none')
    
    for i, feature_key in enumerate(features_order):
        ax1.text(i, features_count_dict[feature_key] + 100, str(features_count_dict[feature_key]), ha='center', va='bottom', fontsize=8,
                 rotation=90, bbox=dict(facecolor='white', edgecolor='none', pad=0, alpha=0.8))  # boxstyle='round,pad=0.1'
    
    ax1.set_xlabel("Features/Fn Grp Class", fontsize=14)
    ax1.set_xticks(range(0, len(features_order)))
    ax1.set_xticklabels(features_order, rotation=90) #, ha="right")
    ax1.set_ylabel("AR Count", fontsize=14)
    ax1.set_ylim(0, 11000)
    ax1.set_yticks(range(0, 11000, 2000))
    ax1.set_yticklabels([str(i) for i in range(0, 11000, 2000)])
    
    for ax in (ax1,):
        ax.grid(which='major', axis='y', color='gray', linestyle='-', linewidth=0.5, alpha=0.5, zorder=0)
        ax.grid(which='minor', axis='y', color='gray', linestyle=':', linewidth=0.5, alpha=0.3, zorder=0)
        ax.set_axisbelow(True)
        ax.minorticks_on()
        ax.xaxis.set_minor_locator(plt.NullLocator())
        ax.tick_params(axis='both', which='both', labelsize=14, direction='in')
        
    table_text = "\n".join([f"{k}. {v}" for k, v in fn_grp_roman_to_name.items()])
    ax1.text(0.33, 0.98, table_text, # ha='right', va='right',
             fontsize=8, transform=ax1.transAxes,
             verticalalignment='top', horizontalalignment='left', 
             bbox=dict(facecolor='white', edgecolor='none', boxstyle='round,pad=0.1', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(transparent=True, fname=save_name + ".pdf", dpi=300, bbox_inches='tight', pad_inches=0.005)
    plt.savefig(transparent=True, fname=save_name + ".svg", dpi=300, bbox_inches='tight', pad_inches=0.005)
    
    return


def main():
    ar_ea_fwd_dict = json.load(open(os.path.join(analyses_path_main, "ar_ea_fwd_dict.json"), "r"))
    ar_ea_rev_dict = json.load(open(os.path.join(analyses_path_main, "ar_ea_rev_dict.json"), "r"))
    mr_ea_fwd_dict = json.load(open(os.path.join(analyses_path_main, "mr_ea_fwd_dict.json"), "r"))
    mr_ea_rev_dict = json.load(open(os.path.join(analyses_path_main, "mr_ea_rev_dict.json"), "r"))
    
    mr_smi_dict = json.load(open(os.path.join(data_path_main, "mr_smi_dict.json"), "r"))
    ar_smi_dict = json.load(open(os.path.join(data_path_main, "ar_smi_dict.json"), "r"))
    
    mr_ar_list_dict = json.load(open(os.path.join(data_path_main, "mr_ar_list_dict.json"), "r"))
    
    ar_fn_groups_added_dict = json.load(open(os.path.join(data_path_main, "ar_fn_groups_added_dict.json"), "r"))
    
    mr_ar_count_dict = {k: len(v) for k, v in mr_ar_list_dict.items()}
    
    mr_ha_count_dict = {k: rxn_smi_to_ha_count(v) for k, v in mr_smi_dict.items()}
    ar_ha_count_dict = {k: rxn_smi_to_ha_count(v) for k, v in ar_smi_dict.items()}
    
    # ea_dist_save_name = os.path.join(plots_path_main, "ea_dist_hist")
    # ea_dist_hist(ar_ea_fwd_dict, ar_ea_rev_dict, mr_ea_fwd_dict, mr_ea_rev_dict, ea_dist_save_name)
    
    # mr_ar_dist_save_name = os.path.join(plots_path_main, "mr_ar_dist_hist")
    # mr_ar_dist_hist(mr_ar_count_dict, mr_ar_dist_save_name)
    
    # rxn_features_fn_grps_dist_save_name = os.path.join(plots_path_main, "rxn_features_fn_grps_dist_bar_plot")
    # rxn_features_fn_grps_dist_bar_plot(ar_smi_dict, ar_fn_groups_added_dict, rxn_features_fn_grps_dist_save_name)
    
    return


if __name__ == "__main__":
    main()
