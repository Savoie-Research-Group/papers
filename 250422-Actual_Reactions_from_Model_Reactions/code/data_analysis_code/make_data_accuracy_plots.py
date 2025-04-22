"""
    Date Modified: 2025/22/04
    Author: Veerupaksh (Veeru) Singla (singla2@purdue.edu)
    Description: Create accuracy plots of using MR eas as surrogate for corresponding AR eas for both fwd and rev MRs/ARs.
"""


import os
import json


this_script_dir = os.path.dirname(os.path.realpath(__file__))
os.chdir(this_script_dir)


from utils import *


def ar_mr_parity_plot(mr_ea_dict, ar_ea_dict, save_name, key="fwd"):
    ar_mr_ea_list = []  ## x-axis, mr ea value for each ar
    ar_ea_list = []  ## y-axis, ar ea values
    
    for ar, ar_ea in ar_ea_dict.items():
        ar_mr = "_".join(ar.split("_")[:3])  ## mr for this ar
        ar_mr_ea_list.append(mr_ea_dict[ar_mr])
        ar_ea_list.append(ar_ea)
    
    # max_ea = max(max(ar_mr_ea_list), max(ar_ea_list))
    max_ea = 170  ## kcal/mol
    
    # print(f"Max Ea: {max_ea}")
    
    r2 = np.corrcoef(ar_ea_list, ar_mr_ea_list)[0, 1]**2
    mae = np.mean([abs(x-y) for x, y in zip(ar_ea_list, ar_mr_ea_list)])  ## kcal/mol
    rmse = np.sqrt(np.mean([(x-y)**2 for x, y in zip(ar_ea_list, ar_mr_ea_list)]))  ## kcal/mol
    median_ae = np.median([abs(x-y) for x, y in zip(ar_ea_list, ar_mr_ea_list)])  ## kcal/mol
    
    plt.clf()
    fig, ax = plt.subplots(figsize=(2.75, 2.75))
    ax.set_aspect(1)
    
    if key == "fwd":
        ax.scatter(ar_ea_list, ar_mr_ea_list, s=2, color=vs_colors["orange"][0], alpha=0.10, label="AR")
    elif key == "rev":
        ax.scatter(ar_ea_list, ar_mr_ea_list, s=2, color=vs_colors["purple"][0], alpha=0.10, label="AR")
    else:
        raise ValueError("Invalid key")
    
    
    ax.plot([0, max_ea], [0, max_ea], color='black', linestyle='--', linewidth=1, label="Parity")

    ax.grid(which='major', color='gray', linestyle='-', linewidth=0.5, alpha=0.5, zorder=0)
    ax.grid(which='minor', axis='both', color='gray', linestyle=':', linewidth=0.5, alpha=0.3, zorder=0)
    ax.set_axisbelow(True)
    ax.minorticks_on()
    # ax.set_title("AR Ea vs MR Ea", fontsize=14)
    
    ax.tick_params(axis='both', which='both', labelsize=14, direction='in')
    
    ax.set_xlim(0, max_ea)
    ax.set_xlabel("AR Ea (kcal/mol)", fontsize=14)
    ax.set_xticks(np.arange(0, max_ea+1, 50))
    ax.set_ylim(0, max_ea)
    ax.set_ylabel("MR Ea (kcal/mol)", fontsize=14)
    ax.set_yticks(np.arange(0, max_ea+1, 50))

    ax.legend(handles=[ax.get_lines()[0]], fontsize=10, loc='upper left')
    
    ## plot r2 and mae values
    ax.text(0.96, 0.07,
            f"R\u00B2: {r2:.2f}\nMedAE: {median_ae:.2f}\nMAE: {mae:.2f}\nRMSE: {rmse:.2f}",
            transform=ax.transAxes, 
            verticalalignment='bottom', 
            horizontalalignment='right',
            fontsize=8,
            bbox=dict(facecolor='white', edgecolor='none', alpha=0.8))
    
    plt.tight_layout()
    
    plt.savefig(f"{save_name}.png", transparent=False, dpi=600, bbox_inches='tight')
    plt.savefig(f"{save_name}.pdf", transparent=False, dpi=600, bbox_inches='tight')
    
    return


def mr_med_mean_abs_error_plot(mr_ea_dict, ar_ea_dict, save_name):
    mr_abs_error_list_dict = {}
    
    for ar, ar_ea in ar_ea_dict.items():
        ar_mr = "_".join(ar.split("_")[:3])  ## mr for this ar
        ar_mr_ea = mr_ea_dict[ar_mr]
        if ar_mr not in mr_abs_error_list_dict:
            mr_abs_error_list_dict[ar_mr] = []
        mr_abs_error_list_dict[ar_mr].append(abs(ar_ea - ar_mr_ea))
        
    mr_list = sorted(list(mr_abs_error_list_dict.keys()))
    mean_abs_error_list = [np.mean(mr_abs_error_list_dict[mr]) for mr in mr_list]
    median_abs_error_list = [np.median(mr_abs_error_list_dict[mr]) for mr in mr_list]
    mr_ar_count_list = [mr_ar_count_dict[mr] for mr in mr_list]
    
    # csv_list = ["MR,Mean AE,Median AE,AR Count"]
    # for mr, mean_abs_error, median_abs_error, mr_ar_count in zip(mr_list, mean_abs_error_list, median_abs_error_list, mr_ar_count_list):
    #     csv_list.append(f"{mr},{mean_abs_error},{median_abs_error},{mr_ar_count}")
    # with open(f"{save_name}.csv", "w") as f:
    #     f.write("\n".join(csv_list))
        
    plt.clf()
    fig, ax = plt.subplots(figsize=(6, 3))
    
    ax.plot(mr_list, mean_abs_error_list, color=vs_colors["blue"][0], linestyle='-', marker='o', markersize=3, label="Mean AE")
    ax.plot(mr_list, median_abs_error_list, color=vs_colors["red"][0], linestyle='-', marker='o', markersize=3, label="Median AE")
    
    ax.grid(which='major', color='gray', linestyle='-', linewidth=0.5, alpha=0.5, zorder=0)
    ax.grid(which='minor', axis='both', color='gray', linestyle=':', linewidth=0.5, alpha=0.3, zorder=0)
    ax.set_axisbelow(True)
    
    ax.tick_params(axis='both', which='both', labelsize=14, direction='in')
    
    ax.set_xlabel("MR Num", fontsize=14)
    ax.set_xticks(np.arange(0, 175+1, 25))
    ax.set_xlim(0, 175)
    ax.set_ylabel("Mean/Median AE (kcal/mol)", fontsize=14)
    # ax.set_yticks(np.arange(0, 51, 5))
    
    ax.legend(fontsize=10)
    
    plt.tight_layout()
    
    # plt.savefig(f"{save_name}.png", transparent=False, dpi=600, bbox_inches='tight')
    plt.savefig(f"{save_name}.pdf", transparent=False, dpi=600, bbox_inches='tight')
    
    return
    

def rxn_features_fn_grps_accuracy_box(mr_ea_fwd_dict, mr_ea_rev_dict, ar_ea_fwd_dict, ar_ea_rev_dict, ar_smi_dict, ar_fn_groups_added_dict, save_name):
    fn_grp_roman_to_name = {"i": "     Anhydrides & Imides",
                            "ii": "    Cyclic Aromatics & Derivatives",
                            "iii": "   Amidines & Azos",
                            "iv": "   Nitrosos & Oximes",
                            "v": "    Amines, Imines, & Nitriles",
                            "vi": "   Acids, Alcohols, Aldehydes, Esters, Ethers, Ketones, & Peroxides",
                            "vii": "  Amides & Isocyanates",
                            "viii": " Cycloalkanes & Derivatives"}
    
    fn_grp_roman_to_name["vi"] = "  Acids, Alcohols, Aldehydes,\n       Esters, Ethers, Ketones,\n       & Peroxides"
    
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
    
    features_ar_list_dict = {
        "Fwd Unimolec": [],
        "Fwd Bimolec": [],
        "Fwd Trimolec": [],
        "Rev Unimolec": [],
        "Rev Bimolec": [],
        "Rev Trimolec": [],
        "i": [],
        "ii": [],
        "iii": [],
        "iv": [],
        "v": [],
        "vi": [],
        "vii": [],
        "viii": []
    }
    
    
    for ar, ar_smi in ar_smi_dict.items():
        react_smi, prod_smi = ar_smi.split(">>")
        react_molec = react_smi.count(".") + 1
        prod_molec = prod_smi.count(".") + 1
        
        if react_molec == 1:
            features_ar_list_dict["Fwd Unimolec"].append(ar)
        elif react_molec == 2:
            features_ar_list_dict["Fwd Bimolec"].append(ar)
        elif react_molec == 3:
            features_ar_list_dict["Fwd Trimolec"].append(ar)
            
        if prod_molec == 1:
            features_ar_list_dict["Rev Unimolec"].append(ar)
        elif prod_molec == 2:
            features_ar_list_dict["Rev Bimolec"].append(ar)
        elif prod_molec == 3:
            features_ar_list_dict["Rev Trimolec"].append(ar)
            
        for fn_added in ar_fn_groups_added_dict[ar]:
            fn_class_int = int(fn_added.split("_")[0])
            fn_class_roman = fn_grp_class_roman_dict[fn_class_int]
            features_ar_list_dict[fn_class_roman].append(ar)
            
    features_accuracy_list_dict_fwd = {
        "Unimolec": [abs(ar_ea_fwd_dict[ar] - mr_ea_fwd_dict["_".join(ar.split("_")[:3])]) for ar in features_ar_list_dict["Fwd Unimolec"]],
        "Bimolec": [abs(ar_ea_fwd_dict[ar] - mr_ea_fwd_dict["_".join(ar.split("_")[:3])]) for ar in features_ar_list_dict["Fwd Bimolec"]],
        "Trimolec": [abs(ar_ea_fwd_dict[ar] - mr_ea_fwd_dict["_".join(ar.split("_")[:3])]) for ar in features_ar_list_dict["Fwd Trimolec"]],
        "i": [abs(ar_ea_fwd_dict[ar] - mr_ea_fwd_dict["_".join(ar.split("_")[:3])]) for ar in features_ar_list_dict["i"]],
        "ii": [abs(ar_ea_fwd_dict[ar] - mr_ea_fwd_dict["_".join(ar.split("_")[:3])]) for ar in features_ar_list_dict["ii"]],
        "iii": [abs(ar_ea_fwd_dict[ar] - mr_ea_fwd_dict["_".join(ar.split("_")[:3])]) for ar in features_ar_list_dict["iii"]],
        "iv": [abs(ar_ea_fwd_dict[ar] - mr_ea_fwd_dict["_".join(ar.split("_")[:3])]) for ar in features_ar_list_dict["iv"]],
        "v": [abs(ar_ea_fwd_dict[ar] - mr_ea_fwd_dict["_".join(ar.split("_")[:3])]) for ar in features_ar_list_dict["v"]],
        "vi": [abs(ar_ea_fwd_dict[ar] - mr_ea_fwd_dict["_".join(ar.split("_")[:3])]) for ar in features_ar_list_dict["vi"]],
        "vii": [abs(ar_ea_fwd_dict[ar] - mr_ea_fwd_dict["_".join(ar.split("_")[:3])]) for ar in features_ar_list_dict["vii"]],
        "viii": [abs(ar_ea_fwd_dict[ar] - mr_ea_fwd_dict["_".join(ar.split("_")[:3])]) for ar in features_ar_list_dict["viii"]]
    }
    
    features_accuracy_list_dict_rev = {
        "Unimolec": [abs(ar_ea_rev_dict[ar] - mr_ea_rev_dict["_".join(ar.split("_")[:3])]) for ar in features_ar_list_dict["Rev Unimolec"]],
        "Bimolec": [abs(ar_ea_rev_dict[ar] - mr_ea_rev_dict["_".join(ar.split("_")[:3])]) for ar in features_ar_list_dict["Rev Bimolec"]],
        "Trimolec": [abs(ar_ea_rev_dict[ar] - mr_ea_rev_dict["_".join(ar.split("_")[:3])]) for ar in features_ar_list_dict["Rev Trimolec"]],
        "i": [abs(ar_ea_rev_dict[ar] - mr_ea_rev_dict["_".join(ar.split("_")[:3])]) for ar in features_ar_list_dict["i"]],
        "ii": [abs(ar_ea_rev_dict[ar] - mr_ea_rev_dict["_".join(ar.split("_")[:3])]) for ar in features_ar_list_dict["ii"]],
        "iii": [abs(ar_ea_rev_dict[ar] - mr_ea_rev_dict["_".join(ar.split("_")[:3])]) for ar in features_ar_list_dict["iii"]],
        "iv": [abs(ar_ea_rev_dict[ar] - mr_ea_rev_dict["_".join(ar.split("_")[:3])]) for ar in features_ar_list_dict["iv"]],
        "v": [abs(ar_ea_rev_dict[ar] - mr_ea_rev_dict["_".join(ar.split("_")[:3])]) for ar in features_ar_list_dict["v"]],
        "vi": [abs(ar_ea_rev_dict[ar] - mr_ea_rev_dict["_".join(ar.split("_")[:3])]) for ar in features_ar_list_dict["vi"]],
        "vii": [abs(ar_ea_rev_dict[ar] - mr_ea_rev_dict["_".join(ar.split("_")[:3])]) for ar in features_ar_list_dict["vii"]],
        "viii": [abs(ar_ea_rev_dict[ar] - mr_ea_rev_dict["_".join(ar.split("_")[:3])]) for ar in features_ar_list_dict["viii"]]
    }
    
    features_order = ["Unimolec", "Bimolec", "Trimolec", "i", "ii", "iii", "iv", "v", "vi", "vii", "viii"]
    
    plt.clf()
    fig = plt.figure(figsize=(5, 3))
    gs = fig.add_gridspec(1, 1, hspace=0.0, wspace=0)
    
    ax1 = fig.add_subplot(gs[0, 0])
    
    n_boxes = len(features_accuracy_list_dict_fwd)
    max_error = 25
    min_error = 0
    
    box_positions = np.arange(1, n_boxes+1)
    box_widths = 0.3
    
    ax1.boxplot([features_accuracy_list_dict_fwd[k] for k in features_order], positions=box_positions-box_widths/1.5, widths=box_widths, patch_artist=True, showfliers=False, boxprops=dict(facecolor=vs_colors["orange"][0], edgecolor='none', alpha=0.5, zorder=1), medianprops=dict(color='none'), whiskerprops=dict(color='none'), capprops=dict(color='none'), showmeans=True, meanline=True, meanprops=dict(color='green', linewidth=1, linestyle='-', zorder=2))
    ax1.boxplot([features_accuracy_list_dict_rev[k] for k in features_order], positions=box_positions+box_widths/1.5, widths=box_widths, patch_artist=True, showfliers=False, boxprops=dict(facecolor=vs_colors["purple"][0], edgecolor='none', alpha=0.45, zorder=1), medianprops=dict(color='none'), whiskerprops=dict(color='none'), capprops=dict(color='none'), showmeans=True, meanline=True, meanprops=dict(color='green', linewidth=1, linestyle='-', zorder=2))
    
    ax1.boxplot([features_accuracy_list_dict_fwd[k] for k in features_order], positions=box_positions-box_widths/1.5, widths=box_widths, patch_artist=False, showfliers=False, boxprops=dict(color='black', linewidth=1, zorder=3), whiskerprops=dict(color='black', linewidth=1), capprops=dict(color='black', linewidth=1), medianprops=dict(color='purple', linewidth=1, linestyle='-', zorder=2))
    ax1.boxplot([features_accuracy_list_dict_rev[k] for k in features_order], positions=box_positions+box_widths/1.5, widths=box_widths, patch_artist=False, showfliers=False, boxprops=dict(color='black', linewidth=1, zorder=3), whiskerprops=dict(color='black', linewidth=1), capprops=dict(color='black', linewidth=1), medianprops=dict(color='purple', linewidth=1, linestyle='-', zorder=2))
    
    ax1.set_xlabel("Features / Fn Group Class", fontsize=14)
    ax1.set_xticks(range(1, len(features_order)+1))
    ax1.set_xticklabels(features_order, rotation=90) #, ha="right")
    ax1.set_ylabel("Absolute error", fontsize=14)
    ax1.set_ylim(0, 40)
    ax1.set_yticks(range(0, max_error, 5))
    ax1.set_yticklabels([str(i) for i in range(0, max_error, 5)])
    
    for ax in (ax1,):
        ax.grid(which='major', axis='y', color='gray', linestyle='-', linewidth=0.5, alpha=0.5, zorder=0)
        ax.grid(which='minor', axis='y', color='gray', linestyle=':', linewidth=0.5, alpha=0.3, zorder=0)
        ax.set_axisbelow(True)
        ax.minorticks_on()
        ax.xaxis.set_minor_locator(plt.NullLocator())
        ax.tick_params(axis='both', which='both', labelsize=14, direction='in')
    
    # fwd_patch = mpatches.Patch(color=vs_colors["orange"][0], alpha=0.5, label='Fwd')
    # rev_patch = mpatches.Patch(color=vs_colors["purple"][0], alpha=0.5, label='Rev')
    # medae_patch = mlines.Line2D([0], [0], color='purple', linewidth=1, linestyle='-', label='Median')
    # mae_patch = mlines.Line2D([0], [0], color='green', linewidth=1, linestyle='-', label='Mean')
    # ax1.legend(handles=[fwd_patch, rev_patch, medae_patch, mae_patch], fontsize=11, loc='upper right',
    #           borderaxespad=0.25, ncol=2, columnspacing=0.75, handletextpad=0.25, handlelength=1.5) #, borderpad=0.25, labelspacing=0.25)
    
    
    table_text_col1 = "\n".join([f" {k}.{fn_grp_roman_to_name[k]}   " for k in ["i", "ii", "iii", "iv", "v"]])
    table_text_col2 = "\n".join([f" {k}.{fn_grp_roman_to_name[k]} " for k in ["vi", "vii", "viii"]])
    
    ax1.text(0.0148, 0.95, table_text_col1, # ha='right', va='right',
                fontsize=8, transform=ax1.transAxes,
                verticalalignment='top', horizontalalignment='left',
                bbox=dict(facecolor='white', edgecolor='none', boxstyle='square,pad=0.5', alpha=1))
    
    ax1.text(0.5415, 0.95, table_text_col2, # ha='right', va='right',
                fontsize=8, transform=ax1.transAxes,
                verticalalignment='top', horizontalalignment='left',
                bbox=dict(facecolor='white', edgecolor='none', boxstyle='square,pad=0.5', alpha=1))
    
    # table_text = "\n".join([f" {k}.{v}" for k, v in fn_grp_roman_to_name.items()])
    # ax1.text(0.0015, 0.9945, table_text, # ha='right', va='right',
    #          fontsize=7.5, transform=ax1.transAxes,
    #          verticalalignment='top', horizontalalignment='left', 
    #          bbox=dict(facecolor='white', edgecolor='none', boxstyle='round,pad=0.0', alpha=1))
    
    plt.tight_layout()
    plt.savefig(transparent=True, fname=save_name + ".pdf", dpi=300, bbox_inches='tight', pad_inches=0.005)
    plt.savefig(transparent=True, fname=save_name + ".svg", dpi=300, bbox_inches='tight', pad_inches=0.005)
    
    return


def ar_ha_count_accuracy_box(mr_ea_fwd_dict, mr_ea_rev_dict, ar_ea_fwd_dict, ar_ea_rev_dict, ar_smi_dict, save_name):
    ha_ar_list_dict = {i:[] for i in range(5, 23)}
    
    for ar, ar_smi in ar_smi_dict.items():
        ha_count = rxn_smi_to_ha_count(ar_smi)
        if ha_count in ha_ar_list_dict:
            ha_ar_list_dict[ha_count].append(ar)
        elif ha_count == 23:
            ha_ar_list_dict[22].append(ar)
        else:
            raise ValueError(f"Invalid ha count: {ha_count}")
    
    ha_accuracy_list_dict_fwd = {i: [abs(ar_ea_fwd_dict[ar] - mr_ea_fwd_dict["_".join(ar.split("_")[:3])]) for ar in ha_ar_list_dict[i]] for i in ha_ar_list_dict}
    ha_accuracy_list_dict_rev = {i: [abs(ar_ea_rev_dict[ar] - mr_ea_rev_dict["_".join(ar.split("_")[:3])]) for ar in ha_ar_list_dict[i]] for i in ha_ar_list_dict}
    
    ha_order = list(range(5, 23))
    
    plt.clf()
    fig = plt.figure(figsize=(5, 1.75))
    gs = fig.add_gridspec(1, 1, hspace=0.0, wspace=0)
    
    ax1 = fig.add_subplot(gs[0, 0])
    
    n_boxes = len(ha_accuracy_list_dict_fwd)
    max_error = 25
    min_error = 0
    
    box_positions = np.arange(1, n_boxes+1)
    box_widths = 0.3
    
    ax1.boxplot([ha_accuracy_list_dict_fwd[i] for i in ha_order], positions=box_positions-box_widths/1.5, widths=box_widths, patch_artist=True, showfliers=False, boxprops=dict(facecolor=vs_colors["orange"][0], edgecolor='none', alpha=0.5, zorder=1), medianprops=dict(color='none'), whiskerprops=dict(color='none'), capprops=dict(color='none'), showmeans=True, meanline=True, meanprops=dict(color='green', linewidth=1, linestyle='-', zorder=2))
    ax1.boxplot([ha_accuracy_list_dict_rev[i] for i in ha_order], positions=box_positions+box_widths/1.5, widths=box_widths, patch_artist=True, showfliers=False, boxprops=dict(facecolor=vs_colors["purple"][0], edgecolor='none', alpha=0.45, zorder=1), medianprops=dict(color='none'), whiskerprops=dict(color='none'), capprops=dict(color='none'), showmeans=True, meanline=True, meanprops=dict(color='green', linewidth=1, linestyle='-', zorder=2))
    
    ax1.boxplot([ha_accuracy_list_dict_fwd[i] for i in ha_order], positions=box_positions-box_widths/1.5, widths=box_widths, patch_artist=False, showfliers=False, boxprops=dict(color='black', linewidth=1, zorder=3), whiskerprops=dict(color='black', linewidth=1), capprops=dict(color='black', linewidth=1), medianprops=dict(color='purple', linewidth=1, linestyle='-', zorder=2))
    ax1.boxplot([ha_accuracy_list_dict_rev[i] for i in ha_order], positions=box_positions+box_widths/1.5, widths=box_widths, patch_artist=False, showfliers=False, boxprops=dict(color='black', linewidth=1, zorder=3), whiskerprops=dict(color='black', linewidth=1), capprops=dict(color='black', linewidth=1), medianprops=dict(color='purple', linewidth=1, linestyle='-', zorder=2))
    
    ax1.set_xlabel("AR Heavy Atom Count", fontsize=14)
    ax1.set_xticks(range(1, len(ha_order)+1))
    # ax1.set_xticklabels(ha_order) # , rotation=90) #, ha="right")
    ax1.set_xticklabels([str(i) if (i-1)%2==0 else "" for i in ha_order], rotation=0) #, ha="right")
    ax1.set_ylabel("Absolute error", fontsize=14)
    ax1.set_ylim(0, max_error)
    ax1.set_yticks(range(0, max_error, 5))
    ax1.set_yticklabels([str(i) for i in range(0, max_error, 5)])
    
    for ax in (ax1,):
        ax.grid(which='major', axis='y', color='gray', linestyle='-', linewidth=0.5, alpha=0.5, zorder=0)
        ax.grid(which='minor', axis='y', color='gray', linestyle=':', linewidth=0.5, alpha=0.3, zorder=0)
        ax.set_axisbelow(True)
        ax.minorticks_on()
        ax.xaxis.set_minor_locator(plt.NullLocator())
        ax.tick_params(axis='both', which='both', labelsize=14, direction='in')
        
    fwd_patch = mpatches.Patch(color=vs_colors["orange"][0], alpha=0.5, label='Fwd')
    rev_patch = mpatches.Patch(color=vs_colors["purple"][0], alpha=0.5, label='Rev')
    medae_patch = mlines.Line2D([0], [0], color='purple', linewidth=1, linestyle='-', label='Median')
    mae_patch = mlines.Line2D([0], [0], color='green', linewidth=1, linestyle='-', label='Mean')
    ax1.legend(handles=[fwd_patch, rev_patch, medae_patch, mae_patch], fontsize=12, loc='upper right', ncol=4,
                borderaxespad=0.2, columnspacing=0.75, handletextpad=0.25) #, handlelength=1.5) #, borderpad=0.25, labelspacing=0.25)
    
    plt.tight_layout()
    plt.savefig(transparent=True, fname=save_name + ".pdf", dpi=300, bbox_inches='tight', pad_inches=0.005)
    plt.savefig(transparent=True, fname=save_name + ".svg", dpi=300, bbox_inches='tight', pad_inches=0.005)
    
    return


def ar_count_per_mr_accuracy_box(mr_ea_fwd_dict, mr_ea_rev_dict, ar_ea_fwd_dict, ar_ea_rev_dict, mr_ar_list_dict, save_name):
    ar_count_order_list = ["[0,5)", "[5,10)", "[10,15)", "[15,20)", "[20,25)", "[25,30)", "[30,35)",
                            "[35,40)", "[40,45)", "[45,50)", "[50,55)", "[55,60)", "[60,65)", "[65,70]"]
    
    ar_count_ar_list_dict = {i:[] for i in ar_count_order_list}
    
    for mr, ar_list in mr_ar_list_dict.items():
        ar_count = len(ar_list)
        if ar_count < 5:
            ar_count_ar_list_dict["[0,5)"].extend(ar_list)
        elif ar_count < 10:
            ar_count_ar_list_dict["[5,10)"].extend(ar_list)
        elif ar_count < 15:
            ar_count_ar_list_dict["[10,15)"].extend(ar_list)
        elif ar_count < 20:
            ar_count_ar_list_dict["[15,20)"].extend(ar_list)
        elif ar_count < 25:
            ar_count_ar_list_dict["[20,25)"].extend(ar_list)
        elif ar_count < 30:
            ar_count_ar_list_dict["[25,30)"].extend(ar_list)
        elif ar_count < 35:
            ar_count_ar_list_dict["[30,35)"].extend(ar_list)
        elif ar_count < 40:
            ar_count_ar_list_dict["[35,40)"].extend(ar_list)
        elif ar_count < 45:
            ar_count_ar_list_dict["[40,45)"].extend(ar_list)
        elif ar_count < 50:
            ar_count_ar_list_dict["[45,50)"].extend(ar_list)
        elif ar_count < 55:
            ar_count_ar_list_dict["[50,55)"].extend(ar_list)
        elif ar_count < 60:
            ar_count_ar_list_dict["[55,60)"].extend(ar_list)
        elif ar_count < 65:
            ar_count_ar_list_dict["[60,65)"].extend(ar_list)
        elif ar_count <= 70:
            ar_count_ar_list_dict["[65,70]"].extend(ar_list)
        else:
            raise ValueError(f"Invalid ar count: {ar_count}")
        
    ar_count_accuracy_list_dict_fwd = {i: [abs(ar_ea_fwd_dict[ar] - mr_ea_fwd_dict["_".join(ar.split("_")[:3])]) for ar in ar_count_ar_list_dict[i]] for i in ar_count_ar_list_dict}
    ar_count_accuracy_list_dict_rev = {i: [abs(ar_ea_rev_dict[ar] - mr_ea_rev_dict["_".join(ar.split("_")[:3])]) for ar in ar_count_ar_list_dict[i]] for i in ar_count_ar_list_dict}
    
    plt.clf()
    fig = plt.figure(figsize=(5, 2.75))
    gs = fig.add_gridspec(1, 1, hspace=0.0, wspace=0)
    
    ax1 = fig.add_subplot(gs[0, 0])
    
    n_boxes = len(ar_count_accuracy_list_dict_fwd)
    max_error = 40
    min_error = 0
    
    box_positions = np.arange(1, n_boxes+1)
    box_widths = 0.3
    
    ax1.boxplot([ar_count_accuracy_list_dict_fwd[i] for i in ar_count_order_list], positions=box_positions-box_widths/1.5, widths=box_widths, patch_artist=True, showfliers=False, boxprops=dict(facecolor=vs_colors["orange"][0], edgecolor='none', alpha=0.5, zorder=1), medianprops=dict(color='none'), whiskerprops=dict(color='none'), capprops=dict(color='none'), showmeans=True, meanline=True, meanprops=dict(color='green', linewidth=1, linestyle='-', zorder=2))
    ax1.boxplot([ar_count_accuracy_list_dict_rev[i] for i in ar_count_order_list], positions=box_positions+box_widths/1.5, widths=box_widths, patch_artist=True, showfliers=False, boxprops=dict(facecolor=vs_colors["purple"][0], edgecolor='none', alpha=0.45, zorder=1), medianprops=dict(color='none'), whiskerprops=dict(color='none'), capprops=dict(color='none'), showmeans=True, meanline=True, meanprops=dict(color='green', linewidth=1, linestyle='-', zorder=2))
    
    ax1.boxplot([ar_count_accuracy_list_dict_fwd[i] for i in ar_count_order_list], positions=box_positions-box_widths/1.5, widths=box_widths, patch_artist=False, showfliers=False, boxprops=dict(color='black', linewidth=1, zorder=3), whiskerprops=dict(color='black', linewidth=1), capprops=dict(color='black', linewidth=1), medianprops=dict(color='purple', linewidth=1, linestyle='-', zorder=2))
    ax1.boxplot([ar_count_accuracy_list_dict_rev[i] for i in ar_count_order_list], positions=box_positions+box_widths/1.5, widths=box_widths, patch_artist=False, showfliers=False, boxprops=dict(color='black', linewidth=1, zorder=3), whiskerprops=dict(color='black', linewidth=1), capprops=dict(color='black', linewidth=1), medianprops=dict(color='purple', linewidth=1, linestyle='-', zorder=2))
    
    ax1.set_xlabel("AR Count per MR", fontsize=14)
    ax1.set_xticks(range(1, len(ar_count_order_list)+1))
    ax1.set_xticklabels(ar_count_order_list, rotation=90) #, ha="right")
    ax1.set_ylabel("Absolute error", fontsize=14)
    ax1.set_ylim(0, max_error)
    ax1.set_yticks(range(0, max_error, 5))
    ax1.set_yticklabels([str(i) for i in range(0, max_error, 5)])
    
    for ax in (ax1,):
        ax.grid(which='major', axis='y', color='gray', linestyle='-', linewidth=0.5, alpha=0.5, zorder=0)
        ax.grid(which='minor', axis='y', color='gray', linestyle=':', linewidth=0.5, alpha=0.3, zorder=0)
        ax.set_axisbelow(True)
        ax.minorticks_on()
        ax.xaxis.set_minor_locator(plt.NullLocator())
        ax.tick_params(axis='both', which='both', labelsize=14, direction='in')
        
    # fwd_patch = mpatches.Patch(color=vs_colors["orange"][0], alpha=0.5, label='Fwd')
    # rev_patch = mpatches.Patch(color=vs_colors["purple"][0], alpha=0.5, label='Rev')
    # medae_patch = mlines.Line2D([0], [0], color='purple', linewidth=1, linestyle='-', label='Median')
    # mae_patch = mlines.Line2D([0], [0], color='green', linewidth=1, linestyle='-', label='Mean')
    # ax1.legend(handles=[fwd_patch, rev_patch, medae_patch, mae_patch], fontsize=11, loc='upper right', ncol=2,
    #             borderaxespad=0.2, columnspacing=0.75, handletextpad=0.25) #, handlelength=1.5) #, borderpad=0.25, labelspacing=0.25)
    
    plt.tight_layout()
    plt.savefig(transparent=True, fname=save_name + ".pdf", dpi=300, bbox_inches='tight', pad_inches=0.005)
    plt.savefig(transparent=True, fname=save_name + ".svg", dpi=300, bbox_inches='tight', pad_inches=0.005)
    
    return


def main():
    mr_smi_dict = json.load(open(os.path.join(data_path_main, "mr_smi_dict.json")))
    mr_ts_e_dict = e_csv_to_dict(os.path.join(data_path_main, "mr_transition_state_energy_list.csv"))
    
    ar_smi_dict = json.load(open(os.path.join(data_path_main, "ar_smi_dict.json")))
    ar_ts_e_dict = e_csv_to_dict(os.path.join(data_path_main, "ar_transition_state_energy_list.csv"))
    
    react_prod_smi_e_dict = e_csv_to_dict(os.path.join(data_path_main, "react_prod_smi_energy_list.csv"))
    
    ar_fn_groups_added_dict = json.load(open(os.path.join(data_path_main, "ar_fn_groups_added_dict.json"), "r"))
    
    mr_ar_list_dict = json.load(open(os.path.join(data_path_main, "mr_ar_list_dict.json"), "r"))
    
    ## all ea in kcal/mol
    mr_ea_fwd_dict, mr_ea_rev_dict, ar_ea_fwd_dict, ar_ea_rev_dict = create_mr_ar_ea_dicts(mr_smi_dict,
                                                                                            mr_ts_e_dict,
                                                                                            ar_smi_dict,
                                                                                            ar_ts_e_dict,
                                                                                            react_prod_smi_e_dict,
                                                                                            save_dicts=False)
    
    ## fwd parity
    ar_mr_parity_plot_fwd_save_name = os.path.join(plots_path_main, "parity_plot_fwd")
    ar_mr_parity_plot(mr_ea_fwd_dict, ar_ea_fwd_dict, ar_mr_parity_plot_fwd_save_name, key="fwd")
    
    ## rev parity
    ar_mr_parity_plot_rev_save_name = os.path.join(plots_path_main, "parity_plot_rev")
    ar_mr_parity_plot(mr_ea_rev_dict, ar_ea_rev_dict, ar_mr_parity_plot_rev_save_name, key="rev")
    
    # ## fwd mean/median abs error
    # mr_med_mean_abs_error_plot_fwd_save_name = os.path.join(plots_path_main, "med_mean_abs_error_plot_fwd")
    # mr_med_mean_abs_error_plot(mr_ea_fwd_dict, ar_ea_fwd_dict, mr_med_mean_abs_error_plot_fwd_save_name)
    
    # ## rev mean/median abs error
    # mr_med_mean_abs_error_plot_rev_save_name = os.path.join(plots_path_main, "med_mean_abs_error_plot_rev")
    # mr_med_mean_abs_error_plot(mr_ea_rev_dict, ar_ea_rev_dict, mr_med_mean_abs_error_plot_rev_save_name)
    
    ## rxn features fn grps accuracy box
    rxn_features_fn_grps_accuracy_box_save_name = os.path.join(plots_path_main, "rxn_features_fn_grps_accuracy_box")
    rxn_features_fn_grps_accuracy_box(mr_ea_fwd_dict, mr_ea_rev_dict, ar_ea_fwd_dict, ar_ea_rev_dict, ar_smi_dict, ar_fn_groups_added_dict, rxn_features_fn_grps_accuracy_box_save_name)
    
    ## ar ha count accuracy box
    ar_ha_count_accuracy_box_save_name = os.path.join(plots_path_main, "ar_ha_count_accuracy_box")
    ar_ha_count_accuracy_box(mr_ea_fwd_dict, mr_ea_rev_dict, ar_ea_fwd_dict, ar_ea_rev_dict, ar_smi_dict, ar_ha_count_accuracy_box_save_name)
    
    ## ar count per mr accuracy box
    ar_count_per_mr_accuracy_box_save_name = os.path.join(plots_path_main, "ar_count_per_mr_accuracy_box")
    ar_count_per_mr_accuracy_box(mr_ea_fwd_dict, mr_ea_rev_dict, ar_ea_fwd_dict, ar_ea_rev_dict, mr_ar_list_dict, ar_count_per_mr_accuracy_box_save_name)
    
    return


if __name__ == "__main__":
    main()
