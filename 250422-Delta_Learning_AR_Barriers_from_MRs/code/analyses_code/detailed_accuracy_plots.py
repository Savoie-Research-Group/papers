"""
    Last Modified: 2025/04/04
    Author: Veerupaksh (Veeru) Singla (singla2@purdue.edu)
    Description: make csv for each model each split accuracy.
"""


import os


this_script_dir = os.path.dirname(os.path.realpath(__file__))
os.chdir(this_script_dir)


import sys
sys.path.append(os.path.join(this_script_dir, ".."))


from utils import *


detailed_accuracy_plots_dir = os.path.join(analyses_dir, "detailed_accuracy_plots")

xgb_direct_preds_dir = os.path.join(preds_dir, "xgb_direct")
xgb_delta_preds_dir = os.path.join(preds_dir, "xgb_delta")

chemprop_direct_preds_dir = os.path.join(preds_dir, "chemprop_direct")
chemprop_delta_preds_dir = os.path.join(preds_dir, "chemprop_delta")

refine_fn_1 = "wb97xd"
refine_fn_2 = "b2plypd3"
chemprop_direct_transfer_preds_dir_1 = os.path.join(preds_dir, f"chemprop_direct_transfer_{refine_fn_1}")
chemprop_direct_transfer_preds_dir_2 = os.path.join(preds_dir, f"chemprop_direct_transfer_{refine_fn_2}")
chemprop_delta_transfer_preds_dir_1 = os.path.join(preds_dir, f"chemprop_delta_transfer_{refine_fn_1}")
chemprop_delta_transfer_preds_dir_2 = os.path.join(preds_dir, f"chemprop_delta_transfer_{refine_fn_2}")


def detect_ground_truth_ea_dict(preds_json_name):
    if "combined" in preds_json_name:
        return "combined"
    elif "preds_fwd" not in preds_json_name and "preds_rev" not in preds_json_name:
        if "fwd" in preds_json_name:
            return "fwd"
        elif "rev" in preds_json_name:
            return "rev"
        else:
            raise ValueError(f"Ground truth EA not found in {preds_json_name}")
    elif "preds_fwd" in preds_json_name:
        return "fwd"
    elif "preds_rev" in preds_json_name:
        return "rev"
    else:
        raise ValueError(f"Ground truth EA not found in {preds_json_name}")


def return_ea_dict_splits_dict_to_use(preds_json_name, mr_ar_ea_name_dict, k_fold_splits_name_dict):
    ea_dict_name = detect_ground_truth_ea_dict(preds_json_name)
    mr_ar_ea_dict_to_use = mr_ar_ea_name_dict[ea_dict_name]
    if ea_dict_name == "combined":
        if "stratified" in preds_json_name:
            k_fold_splits_dict_to_use = k_fold_splits_name_dict["stratified_combined"]
        elif "random" in preds_json_name:
            k_fold_splits_dict_to_use = k_fold_splits_name_dict["random_combined"]
        elif "mr_scaffold" in preds_json_name:
            k_fold_splits_dict_to_use = k_fold_splits_name_dict["mr_scaffold_combined"]
    else:
        if "stratified" in preds_json_name:
            k_fold_splits_dict_to_use = k_fold_splits_name_dict["stratified"]
        elif "random" in preds_json_name:
            k_fold_splits_dict_to_use = k_fold_splits_name_dict["random"]
        elif "mr_scaffold" in preds_json_name:
            k_fold_splits_dict_to_use = k_fold_splits_name_dict["mr_scaffold"]
    return mr_ar_ea_dict_to_use, k_fold_splits_dict_to_use


def get_train_val_test_mr_ar_ea_ea_preds_lists(mr_ar_ea_dict, k_fold_splits_dict, k_fold_preds_dict=None, k_fold_preds_dict_path=None):
    assert k_fold_preds_dict is not None or k_fold_preds_dict_path is not None, "k_fold_preds_dict or k_fold_preds_dict_path should be provided."
    
    if k_fold_preds_dict is None:
        k_fold_preds_dict = json.load(open(k_fold_preds_dict_path, "r"))
    
    train_mr_ar_list_full = []
    train_ea_list = []
    train_ea_pred_list = []
    
    val_mr_ar_list_full = []
    val_ea_list = []
    val_ea_pred_list = []
    
    test_mr_ar_list_full = []
    test_ea_list = []
    test_ea_pred_list = []
    
    for fold, split in k_fold_splits_dict.items():
        train_mr_ar_list = split[0][-1]
        val_mr_ar_list = split[1]
        test_mr_ar_list = split[2]
        mr_ar_pred_dict = {k: v for k, v in zip(k_fold_preds_dict[fold][-1][0], k_fold_preds_dict[fold][-1][1])}  ## take the preds for models trained with 100% training data.
        
        if k_fold_preds_dict_path is not None and "combined" in k_fold_preds_dict_path:
            # only consider fwd preds for combined models for plotting
            train_mr_ar_list = [i for i in train_mr_ar_list if "_rev" not in i]
            val_mr_ar_list = [i for i in val_mr_ar_list if "_rev" not in i]
            test_mr_ar_list = [i for i in test_mr_ar_list if "_rev" not in i]
            mr_ar_pred_dict = {k: v for k, v in mr_ar_pred_dict.items() if "_rev" not in k}
        
        for mr_ar in train_mr_ar_list:
            if mr_ar in mr_ar_pred_dict:
                train_mr_ar_list_full.append(mr_ar)
                train_ea_list.append(mr_ar_ea_dict[mr_ar])
                train_ea_pred_list.append(mr_ar_pred_dict[mr_ar])
        
        for mr_ar in val_mr_ar_list:
            if mr_ar in mr_ar_pred_dict:
                val_mr_ar_list_full.append(mr_ar)
                val_ea_list.append(mr_ar_ea_dict[mr_ar])
                val_ea_pred_list.append(mr_ar_pred_dict[mr_ar])
        
        for mr_ar in test_mr_ar_list:
            if mr_ar in mr_ar_pred_dict:
                test_mr_ar_list_full.append(mr_ar)
                test_ea_list.append(mr_ar_ea_dict[mr_ar])
                test_ea_pred_list.append(mr_ar_pred_dict[mr_ar])
                
    return train_mr_ar_list_full, train_ea_list, train_ea_pred_list, val_mr_ar_list_full, val_ea_list, val_ea_pred_list, test_mr_ar_list_full, test_ea_list, test_ea_pred_list


def rxn_features_fn_grps_accuracy_box(mr_ar_ea_dict, k_fold_splits_dict, k_fold_preds_dict_path, ar_smi_dict, ar_fn_groups_added_dict, save_dir, save_name):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # train_mr_ar_list, train_ea_list, train_ea_pred_list, val_mr_ar_list, val_ea_list, val_ea_pred_list, test_mr_ar_list, test_ea_list, test_ea_pred_list = get_train_val_test_mr_ar_ea_ea_preds_lists(mr_ar_ea_dict, k_fold_splits_dict, k_fold_preds_dict_path=k_fold_preds_dict_path)
    _, _, _, _, _, _, test_mr_ar_list, test_ea_list, test_ea_pred_list = get_train_val_test_mr_ar_ea_ea_preds_lists(mr_ar_ea_dict, k_fold_splits_dict, k_fold_preds_dict_path=k_fold_preds_dict_path)
    
    test_ea_list = [test_ea_list[i] for i in range(len(test_mr_ar_list)) if test_mr_ar_list[i].count("_") not in (2, 3)]  ## remove MRs
    test_ea_pred_list = [test_ea_pred_list[i] for i in range(len(test_mr_ar_list)) if test_mr_ar_list[i].count("_") not in (2, 3)]  ## remove MRs
    test_mr_ar_list = [test_mr_ar_list[i] for i in range(len(test_mr_ar_list)) if test_mr_ar_list[i].count("_") not in (2, 3)]  ## remove MRs
    
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
    
    features_accuracy_list_dict_ml_pred = {
        "Unimolec": [],
        "Bimolec": [],
        "Trimolec": [],
        "i": [],
        "ii": [],
        "iii": [],
        "iv": [],
        "v": [],
        "vi": [],
        "vii": [],
        "viii": []
    }
    
    features_accuracy_list_dict_mr_surrogate = deepcopy(features_accuracy_list_dict_ml_pred)
    
    for i, mr_ar in enumerate(test_mr_ar_list):
        abs_err_ml_pred = abs(test_ea_list[i] - test_ea_pred_list[i])
        abs_err_mr_surrogate = abs(test_ea_list[i] - mr_ar_ea_dict[ar_name_to_mr_name(mr_ar)])
        
        react_smi, prod_smi = ar_smi_dict[mr_ar].split(">>")
        react_molec = react_smi.count(".") + 1
        prod_molec = prod_smi.count(".") + 1
        if "preds_rev" in k_fold_preds_dict_path or ("_rev" in k_fold_preds_dict_path and "preds_fwd" not in k_fold_preds_dict_path):
            if prod_molec == 1:
                features_accuracy_list_dict_ml_pred["Unimolec"].append(abs_err_ml_pred)
                features_accuracy_list_dict_mr_surrogate["Unimolec"].append(abs_err_mr_surrogate)
            elif prod_molec == 2:
                features_accuracy_list_dict_ml_pred["Bimolec"].append(abs_err_ml_pred)
                features_accuracy_list_dict_mr_surrogate["Bimolec"].append(abs_err_mr_surrogate)
            elif prod_molec == 3:
                features_accuracy_list_dict_ml_pred["Trimolec"].append(abs_err_ml_pred)
                features_accuracy_list_dict_mr_surrogate["Trimolec"].append(abs_err_mr_surrogate)
        else:
            if react_molec == 1:
                features_accuracy_list_dict_ml_pred["Unimolec"].append(abs_err_ml_pred)
                features_accuracy_list_dict_mr_surrogate["Unimolec"].append(abs_err_mr_surrogate)
            elif react_molec == 2:
                features_accuracy_list_dict_ml_pred["Bimolec"].append(abs_err_ml_pred)
                features_accuracy_list_dict_mr_surrogate["Bimolec"].append(abs_err_mr_surrogate)
            elif react_molec == 3:
                features_accuracy_list_dict_ml_pred["Trimolec"].append(abs_err_ml_pred)
                features_accuracy_list_dict_mr_surrogate["Trimolec"].append(abs_err_mr_surrogate)
        
        for fn_added in ar_fn_groups_added_dict[mr_ar]:
            fn_class_int = int(fn_added.split("_")[0])
            fn_class_roman = fn_grp_class_roman_dict[fn_class_int]
            features_accuracy_list_dict_ml_pred[fn_class_roman].append(abs_err_ml_pred)
            features_accuracy_list_dict_mr_surrogate[fn_class_roman].append(abs_err_mr_surrogate)
    
    features_order = ["Unimolec", "Bimolec", "Trimolec", "i", "ii", "iii", "iv", "v", "vi", "vii", "viii"]
    
    plt.clf()
    fig = plt.figure(figsize=(5, 3))
    gs = fig.add_gridspec(1, 1, hspace=0.0, wspace=0)
    ax1 = fig.add_subplot(gs[0, 0])
    
    n_boxes = len(features_accuracy_list_dict_ml_pred)
    max_error = 25
    min_error = 0
    
    box_positions = np.arange(1, n_boxes+1)
    box_widths = 0.3
    
    ax1.boxplot([features_accuracy_list_dict_mr_surrogate[k] for k in features_order], positions=box_positions-box_widths/1.5, widths=box_widths, patch_artist=True, showfliers=False, boxprops=dict(facecolor=vs_colors["orange"][0], edgecolor='none', alpha=0.5, zorder=1), medianprops=dict(color='none'), whiskerprops=dict(color='none'), capprops=dict(color='none'), showmeans=True, meanline=True, meanprops=dict(color='green', linewidth=1, linestyle='-', zorder=2))
    ax1.boxplot([features_accuracy_list_dict_ml_pred[k] for k in features_order], positions=box_positions+box_widths/1.5, widths=box_widths, patch_artist=True, showfliers=False, boxprops=dict(facecolor=vs_colors["purple"][0], edgecolor='none', alpha=0.5, zorder=1), medianprops=dict(color='none'), whiskerprops=dict(color='none'), capprops=dict(color='none'), showmeans=True, meanline=True, meanprops=dict(color='green', linewidth=1, linestyle='-', zorder=2))
    
    ax1.boxplot([features_accuracy_list_dict_mr_surrogate[k] for k in features_order], positions=box_positions-box_widths/1.5, widths=box_widths, patch_artist=False, showfliers=False, boxprops=dict(color='black', linewidth=1, zorder=3), whiskerprops=dict(color='black', linewidth=1), capprops=dict(color='black', linewidth=1), medianprops=dict(color='purple', linewidth=1, linestyle='-', zorder=2))
    ax1.boxplot([features_accuracy_list_dict_ml_pred[k] for k in features_order], positions=box_positions+box_widths/1.5, widths=box_widths, patch_artist=False, showfliers=False, boxprops=dict(color='black', linewidth=1, zorder=3), whiskerprops=dict(color='black', linewidth=1), capprops=dict(color='black', linewidth=1), medianprops=dict(color='purple', linewidth=1, linestyle='-', zorder=2))
    
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
    
    # mr_surrogate_patch = mpatches.Patch(color=vs_colors["orange"][0], alpha=0.5, label='MR Surr.')
    # ml_pred_patch = mpatches.Patch(color=vs_colors["purple"][0], alpha=0.5, label='ML Pred.')
    # medae_patch = mlines.Line2D([0], [0], color='purple', linewidth=1, linestyle='-', label='Median')
    # mae_patch = mlines.Line2D([0], [0], color='green', linewidth=1, linestyle='-', label='Mean')
    # ax1.legend(handles=[mr_surrogate_patch, ml_pred_patch, medae_patch, mae_patch], fontsize=11, loc='upper right',
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
    
    plt.tight_layout()
    plt.savefig(transparent=True, fname=os.path.join(save_dir, save_name + ".pdf"), dpi=600, bbox_inches='tight', pad_inches=0.005)
    # plt.savefig(transparent=True, fname=os.path.join(save_dir, save_name + ".svg"), dpi=600, bbox_inches='tight', pad_inches=0.005)
    
    plt.close()
    
    return


def ar_ha_count_accuracy_box(mr_ar_ea_dict, k_fold_splits_dict, k_fold_preds_dict_path, ar_smi_dict, save_dir, save_name):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    _, _, _, _, _, _, test_mr_ar_list, test_ea_list, test_ea_pred_list = get_train_val_test_mr_ar_ea_ea_preds_lists(mr_ar_ea_dict, k_fold_splits_dict, k_fold_preds_dict_path=k_fold_preds_dict_path)
    
    test_ea_list = [test_ea_list[i] for i in range(len(test_mr_ar_list)) if test_mr_ar_list[i].count("_") not in (2, 3)]  ## remove MRs
    test_ea_pred_list = [test_ea_pred_list[i] for i in range(len(test_mr_ar_list)) if test_mr_ar_list[i].count("_") not in (2, 3)]  ## remove MRs
    test_mr_ar_list = [test_mr_ar_list[i] for i in range(len(test_mr_ar_list)) if test_mr_ar_list[i].count("_") not in (2, 3)]  ## remove MRs
    
    test_mr_ar_ea_dict = {test_mr_ar_list[i]: test_ea_list[i] for i in range(len(test_mr_ar_list))}
    test_mr_ar_ea_pred_dict = {test_mr_ar_list[i]: test_ea_pred_list[i] for i in range(len(test_mr_ar_list))}
    
    ha_ar_list_dict = {i:[] for i in range(5, 23)}
    
    for ar, ar_smi in ar_smi_dict.items():
        if ar not in test_mr_ar_list:
            continue
        ha_count = rxn_smi_to_ha_count(ar_smi)
        if ha_count in ha_ar_list_dict:
            ha_ar_list_dict[ha_count].append(ar)
        elif ha_count == 23:
            ha_ar_list_dict[22].append(ar)
        else:
            raise ValueError(f"Invalid ha count: {ha_count}")
    
    ha_accuracy_list_dict_ml_pred = {i: [abs(test_mr_ar_ea_dict[ar] - test_mr_ar_ea_pred_dict[ar]) for ar in ha_ar_list_dict[i]] for i in ha_ar_list_dict}
    ha_accuracy_list_dict_mr_surrogate = {i: [abs(test_mr_ar_ea_dict[ar] - mr_ar_ea_dict[ar_name_to_mr_name(ar)]) for ar in ha_ar_list_dict[i]] for i in ha_ar_list_dict}
    
    ha_order = list(range(5, 23))
    
    plt.clf()
    fig = plt.figure(figsize=(5, 1.75))
    gs = fig.add_gridspec(1, 1, hspace=0.0, wspace=0)
    
    ax1 = fig.add_subplot(gs[0, 0])
    
    n_boxes = len(ha_accuracy_list_dict_ml_pred)
    max_error = 25
    min_error = 0
    
    box_positions = np.arange(1, n_boxes+1)
    box_widths = 0.3
    
    ax1.boxplot([ha_accuracy_list_dict_mr_surrogate[i] for i in ha_order], positions=box_positions-box_widths/1.5, widths=box_widths, patch_artist=True, showfliers=False, boxprops=dict(facecolor=vs_colors["orange"][0], edgecolor='none', alpha=0.5, zorder=1), medianprops=dict(color='none'), whiskerprops=dict(color='none'), capprops=dict(color='none'), showmeans=True, meanline=True, meanprops=dict(color='green', linewidth=1, linestyle='-', zorder=2))
    ax1.boxplot([ha_accuracy_list_dict_ml_pred[i] for i in ha_order], positions=box_positions+box_widths/1.5, widths=box_widths, patch_artist=True, showfliers=False, boxprops=dict(facecolor=vs_colors["purple"][0], edgecolor='none', alpha=0.5, zorder=1), medianprops=dict(color='none'), whiskerprops=dict(color='none'), capprops=dict(color='none'), showmeans=True, meanline=True, meanprops=dict(color='green', linewidth=1, linestyle='-', zorder=2))
    
    ax1.boxplot([ha_accuracy_list_dict_mr_surrogate[i] for i in ha_order], positions=box_positions-box_widths/1.5, widths=box_widths, patch_artist=False, showfliers=False, boxprops=dict(color='black', linewidth=1, zorder=3), whiskerprops=dict(color='black', linewidth=1), capprops=dict(color='black', linewidth=1), medianprops=dict(color='purple', linewidth=1, linestyle='-', zorder=2))
    ax1.boxplot([ha_accuracy_list_dict_ml_pred[i] for i in ha_order], positions=box_positions+box_widths/1.5, widths=box_widths, patch_artist=False, showfliers=False, boxprops=dict(color='black', linewidth=1, zorder=3), whiskerprops=dict(color='black', linewidth=1), capprops=dict(color='black', linewidth=1), medianprops=dict(color='purple', linewidth=1, linestyle='-', zorder=2))
    
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
    
    mr_surrogate_patch = mpatches.Patch(color=vs_colors["orange"][0], alpha=0.5, label='MR Surr.')
    ml_pred_patch = mpatches.Patch(color=vs_colors["purple"][0], alpha=0.5, label='ML Pred.')
    medae_patch = mlines.Line2D([0], [0], color='purple', linewidth=1, linestyle='-', label='Median')
    mae_patch = mlines.Line2D([0], [0], color='green', linewidth=1, linestyle='-', label='Mean')
    ax1.legend(handles=[mr_surrogate_patch, ml_pred_patch, medae_patch, mae_patch], fontsize=12, loc='upper right', ncol=4,
                borderaxespad=0.2, columnspacing=0.75, handletextpad=0.25) #, handlelength=1.5) #, borderpad=0.25, labelspacing=0.25)
    
    plt.tight_layout()
    plt.savefig(transparent=True, fname=os.path.join(save_dir, save_name + ".pdf"), dpi=600, bbox_inches='tight', pad_inches=0.005)
    # plt.savefig(transparent=True, fname=os.path.join(save_dir, save_name + ".svg"), dpi=600, bbox_inches='tight', pad_inches=0.005)
    
    plt.close()
    
    return


def ar_reduced_ha_count_accuracy_box(mr_ar_ea_dict, k_fold_splits_dict, k_fold_preds_dict_path, ar_smi_dict, mr_smi_dict, save_dir, save_name):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    _, _, _, _, _, _, test_mr_ar_list, test_ea_list, test_ea_pred_list = get_train_val_test_mr_ar_ea_ea_preds_lists(mr_ar_ea_dict, k_fold_splits_dict, k_fold_preds_dict_path=k_fold_preds_dict_path)
    
    test_ea_list = [test_ea_list[i] for i in range(len(test_mr_ar_list)) if test_mr_ar_list[i].count("_") not in (2, 3)]  ## remove MRs
    test_ea_pred_list = [test_ea_pred_list[i] for i in range(len(test_mr_ar_list)) if test_mr_ar_list[i].count("_") not in (2, 3)]  ## remove MRs
    test_mr_ar_list = [test_mr_ar_list[i] for i in range(len(test_mr_ar_list)) if test_mr_ar_list[i].count("_") not in (2, 3)]  ## remove MRs
    
    test_mr_ar_ea_dict = {test_mr_ar_list[i]: test_ea_list[i] for i in range(len(test_mr_ar_list))}
    test_mr_ar_ea_pred_dict = {test_mr_ar_list[i]: test_ea_pred_list[i] for i in range(len(test_mr_ar_list))}
    
    ha_diff_ar_list_dict = {i:[] for i in range(1, 14)}
    
    for ar, ar_smi in ar_smi_dict.items():
        if ar not in test_mr_ar_list:
            continue
        ha_count = rxn_smi_to_ha_count(ar_smi)
        mr_ha_count = rxn_smi_to_ha_count(mr_smi_dict["_".join(ar.split("_")[:3])])
        ha_diff = ha_count - mr_ha_count
        if ha_diff in ha_diff_ar_list_dict:
            ha_diff_ar_list_dict[ha_diff].append(ar)
        elif ha_diff == 14:
            ha_diff_ar_list_dict[13].append(ar)
        else:
            raise ValueError(f"Invalid ha diff: {ha_diff}")
    
    ha_diff_accuracy_list_dict_ml_pred = {i: [abs(test_mr_ar_ea_dict[ar] - test_mr_ar_ea_pred_dict[ar]) for ar in ha_diff_ar_list_dict[i]] for i in ha_diff_ar_list_dict}
    ha_diff_accuracy_list_dict_mr_surrogate = {i: [abs(test_mr_ar_ea_dict[ar] - mr_ar_ea_dict[ar_name_to_mr_name(ar)]) for ar in ha_diff_ar_list_dict[i]] for i in ha_diff_ar_list_dict}
    
    ha_diff_order = list(range(1, 14))
    
    plt.clf()
    fig = plt.figure(figsize=(5, 1.75))
    gs = fig.add_gridspec(1, 1, hspace=0.0, wspace=0)
    
    ax1 = fig.add_subplot(gs[0, 0])
    
    n_boxes = len(ha_diff_accuracy_list_dict_ml_pred)
    max_error = 25
    min_error = 0
    
    box_positions = np.arange(1, n_boxes+1)
    box_widths = 0.3
    
    ax1.boxplot([ha_diff_accuracy_list_dict_mr_surrogate[i] for i in ha_diff_order], positions=box_positions-box_widths/1.5, widths=box_widths, patch_artist=True, showfliers=False, boxprops=dict(facecolor=vs_colors["orange"][0], edgecolor='none', alpha=0.5, zorder=1), medianprops=dict(color='none'), whiskerprops=dict(color='none'), capprops=dict(color='none'), showmeans=True, meanline=True, meanprops=dict(color='green', linewidth=1, linestyle='-', zorder=2))
    ax1.boxplot([ha_diff_accuracy_list_dict_ml_pred[i] for i in ha_diff_order], positions=box_positions+box_widths/1.5, widths=box_widths, patch_artist=True, showfliers=False, boxprops=dict(facecolor=vs_colors["purple"][0], edgecolor='none', alpha=0.5, zorder=1), medianprops=dict(color='none'), whiskerprops=dict(color='none'), capprops=dict(color='none'), showmeans=True, meanline=True, meanprops=dict(color='green', linewidth=1, linestyle='-', zorder=2))
    
    ax1.boxplot([ha_diff_accuracy_list_dict_mr_surrogate[i] for i in ha_diff_order], positions=box_positions-box_widths/1.5, widths=box_widths, patch_artist=False, showfliers=False, boxprops=dict(color='black', linewidth=1, zorder=3), whiskerprops=dict(color='black', linewidth=1), capprops=dict(color='black', linewidth=1), medianprops=dict(color='purple', linewidth=1, linestyle='-', zorder=2))
    ax1.boxplot([ha_diff_accuracy_list_dict_ml_pred[i] for i in ha_diff_order], positions=box_positions+box_widths/1.5, widths=box_widths, patch_artist=False, showfliers=False, boxprops=dict(color='black', linewidth=1, zorder=3), whiskerprops=dict(color='black', linewidth=1), capprops=dict(color='black', linewidth=1), medianprops=dict(color='purple', linewidth=1, linestyle='-', zorder=2))
    
    ax1.set_xlabel("AR-MR HA Diff", fontsize=14)
    ax1.set_xticks(range(1, len(ha_diff_order)+1))
    # ax1.set_xticklabels(ha_order) # , rotation=90) #, ha="right")
    ax1.set_xticklabels([str(i) if (i-1)%2==0 else "" for i in ha_diff_order], rotation=0) #, ha="right")
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
    
    mr_surrogate_patch = mpatches.Patch(color=vs_colors["orange"][0], alpha=0.5, label='MR Surr.')
    ml_pred_patch = mpatches.Patch(color=vs_colors["purple"][0], alpha=0.5, label='ML Pred.')
    medae_patch = mlines.Line2D([0], [0], color='purple', linewidth=1, linestyle='-', label='Median')
    mae_patch = mlines.Line2D([0], [0], color='green', linewidth=1, linestyle='-', label='Mean')
    ax1.legend(handles=[mr_surrogate_patch, ml_pred_patch, medae_patch, mae_patch], fontsize=12, loc='upper right', ncol=4,
                borderaxespad=0.2, columnspacing=0.75, handletextpad=0.25) #, handlelength=1.5) #, borderpad=0.25, labelspacing=0.25)
    
    plt.tight_layout()
    plt.savefig(transparent=True, fname=os.path.join(save_dir, save_name + ".pdf"), dpi=600, bbox_inches='tight', pad_inches=0.005)
    # plt.savefig(transparent=True, fname=os.path.join(save_dir, save_name + ".svg"), dpi=600, bbox_inches='tight', pad_inches=0.005)
    
    plt.close()
    
    return


def ar_count_per_mr_accuracy_box(mr_ar_ea_dict, k_fold_splits_dict, k_fold_preds_dict_path, mr_ar_list_dict, save_dir, save_name):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    _, _, _, _, _, _, test_mr_ar_list, test_ea_list, test_ea_pred_list = get_train_val_test_mr_ar_ea_ea_preds_lists(mr_ar_ea_dict, k_fold_splits_dict, k_fold_preds_dict_path=k_fold_preds_dict_path)
    test_ea_list = [test_ea_list[i] for i in range(len(test_mr_ar_list)) if test_mr_ar_list[i].count("_") not in (2, 3)]  ## remove MRs
    test_ea_pred_list = [test_ea_pred_list[i] for i in range(len(test_mr_ar_list)) if test_mr_ar_list[i].count("_") not in (2, 3)]  ## remove MRs
    test_mr_ar_list = [test_mr_ar_list[i] for i in range(len(test_mr_ar_list)) if test_mr_ar_list[i].count("_") not in (2, 3)]  ## remove MRs
    test_mr_ar_ea_dict = {test_mr_ar_list[i]: test_ea_list[i] for i in range(len(test_mr_ar_list))}
    test_mr_ar_ea_pred_dict = {test_mr_ar_list[i]: test_ea_pred_list[i] for i in range(len(test_mr_ar_list))}
    
    ar_count_order_list = ["[0,5)", "[5,10)", "[10,15)", "[15,20)", "[20,25)", "[25,30)", "[30,35)",
                            "[35,40)", "[40,45)", "[45,50)", "[50,55)", "[55,60)", "[60,65)", "[65,70]"]
    ar_count_ar_list_dict = {i:[] for i in ar_count_order_list}
    for mr, ar_list_in in mr_ar_list_dict.items():
        ar_count = len(ar_list_in)
        ar_list = [ar for ar in ar_list_in if ar in test_mr_ar_list]
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
        
    ar_count_accuracy_list_dict_ml_pred = {i: [abs(test_mr_ar_ea_dict[ar] - test_mr_ar_ea_pred_dict[ar]) for ar in ar_count_ar_list_dict[i]] for i in ar_count_ar_list_dict}
    ar_count_accuracy_list_dict_mr_surrogate = {i: [abs(test_mr_ar_ea_dict[ar] - mr_ar_ea_dict[ar_name_to_mr_name(ar)]) for ar in ar_count_ar_list_dict[i]] for i in ar_count_ar_list_dict}
    
    plt.clf()
    fig = plt.figure(figsize=(5, 2.75))
    gs = fig.add_gridspec(1, 1, hspace=0.0, wspace=0)
    
    ax1 = fig.add_subplot(gs[0, 0])
    
    n_boxes = len(ar_count_accuracy_list_dict_ml_pred)
    max_error = 40
    min_error = 0
    
    box_positions = np.arange(1, n_boxes+1)
    box_widths = 0.3
    
    ax1.boxplot([ar_count_accuracy_list_dict_mr_surrogate[i] for i in ar_count_order_list], positions=box_positions-box_widths/1.5, widths=box_widths, patch_artist=True, showfliers=False, boxprops=dict(facecolor=vs_colors["orange"][0], edgecolor='none', alpha=0.5, zorder=1), medianprops=dict(color='none'), whiskerprops=dict(color='none'), capprops=dict(color='none'), showmeans=True, meanline=True, meanprops=dict(color='green', linewidth=1, linestyle='-', zorder=2))
    ax1.boxplot([ar_count_accuracy_list_dict_ml_pred[i] for i in ar_count_order_list], positions=box_positions+box_widths/1.5, widths=box_widths, patch_artist=True, showfliers=False, boxprops=dict(facecolor=vs_colors["purple"][0], edgecolor='none', alpha=0.5, zorder=1), medianprops=dict(color='none'), whiskerprops=dict(color='none'), capprops=dict(color='none'), showmeans=True, meanline=True, meanprops=dict(color='green', linewidth=1, linestyle='-', zorder=2))
    
    ax1.boxplot([ar_count_accuracy_list_dict_mr_surrogate[i] for i in ar_count_order_list], positions=box_positions-box_widths/1.5, widths=box_widths, patch_artist=False, showfliers=False, boxprops=dict(color='black', linewidth=1, zorder=3), whiskerprops=dict(color='black', linewidth=1), capprops=dict(color='black', linewidth=1), medianprops=dict(color='purple', linewidth=1, linestyle='-', zorder=2))
    ax1.boxplot([ar_count_accuracy_list_dict_ml_pred[i] for i in ar_count_order_list], positions=box_positions+box_widths/1.5, widths=box_widths, patch_artist=False, showfliers=False, boxprops=dict(color='black', linewidth=1, zorder=3), whiskerprops=dict(color='black', linewidth=1), capprops=dict(color='black', linewidth=1), medianprops=dict(color='purple', linewidth=1, linestyle='-', zorder=2))
    
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
    
    # mr_surrogate_patch = mpatches.Patch(color=vs_colors["orange"][0], alpha=0.5, label='MR Surr.')
    # ml_pred_patch = mpatches.Patch(color=vs_colors["purple"][0], alpha=0.5, label='ML Pred.')
    # medae_patch = mlines.Line2D([0], [0], color='purple', linewidth=1, linestyle='-', label='Median')
    # mae_patch = mlines.Line2D([0], [0], color='green', linewidth=1, linestyle='-', label='Mean')
    # ax1.legend(handles=[mr_surrogate_patch, ml_pred_patch, medae_patch, mae_patch], fontsize=11, loc='upper right', ncol=2,
    #             borderaxespad=0.2, columnspacing=0.75, handletextpad=0.25) #, handlelength=1.5) #, borderpad=0.25, labelspacing=0.25)
    
    plt.tight_layout()
    plt.savefig(transparent=True, fname=os.path.join(save_dir, save_name + ".pdf"), dpi=600, bbox_inches='tight', pad_inches=0.005)
    # plt.savefig(transparent=True, fname=os.path.join(save_dir, save_name + ".svg"), dpi=600, bbox_inches='tight', pad_inches=0.005)
    
    plt.close()
    
    return


def check_create_dirs():
    if not os.path.exists(detailed_accuracy_plots_dir):
        os.makedirs(detailed_accuracy_plots_dir)
    return


def main():
    check_create_dirs()
    
    ar_fn_groups_added_dict = json.load(open(os.path.join(data_paper_data_dir, "ar_fn_groups_added_dict.json"), "r"))
    mr_smi_dict = json.load(open(os.path.join(data_paper_data_dir, "mr_smi_dict.json")))
    ar_smi_dict = json.load(open(os.path.join(data_paper_data_dir, "ar_smi_dict.json")))
    mr_ar_list_dict = json.load(open(os.path.join(data_paper_data_dir, "mr_ar_list_dict.json"), "r"))
    
    ## load ea dicts
    ar_ea_fwd_dict_path = os.path.join(data_paper_data_dir, "ar_ea_fwd_dict.json")
    ar_ea_fwd_dict = json.load(open(ar_ea_fwd_dict_path, "r"))
    
    ar_ea_rev_dict_path = os.path.join(data_paper_data_dir, "ar_ea_rev_dict.json")
    ar_ea_rev_dict = json.load(open(ar_ea_rev_dict_path, "r"))
    
    mr_ea_fwd_dict_path = os.path.join(data_paper_data_dir, "mr_ea_fwd_dict.json")
    mr_ea_fwd_dict = json.load(open(mr_ea_fwd_dict_path, "r"))
    
    mr_ea_rev_dict_path = os.path.join(data_paper_data_dir, "mr_ea_rev_dict.json")
    mr_ea_rev_dict = json.load(open(mr_ea_rev_dict_path, "r"))
    
    mr_ar_ea_fwd_dict = deepcopy(mr_ea_fwd_dict)
    mr_ar_ea_fwd_dict.update(ar_ea_fwd_dict)
    
    mr_ar_ea_rev_dict = deepcopy(mr_ea_rev_dict)
    mr_ar_ea_rev_dict.update(ar_ea_rev_dict)
    
    mr_ar_ea_combined_dict = deepcopy(mr_ar_ea_fwd_dict)
    mr_ar_ea_combined_dict.update({f"{k}_rev": v for k, v in mr_ar_ea_rev_dict.items()})
    
    mr_ar_ea_name_dict = {
        "fwd": mr_ar_ea_fwd_dict,
        "rev": mr_ar_ea_rev_dict,
        "combined": mr_ar_ea_combined_dict
    }
    
    mr_ar_ea_dict_wb97xd_refined = json.load(open(os.path.join(data_dir, "high_level_single_pt_refinement", "mr_ar_ea_combined_dict_wb97xd.json"), "r"))
    mr_ar_ea_dict_b2plypd3_refined = json.load(open(os.path.join(data_dir, "high_level_single_pt_refinement", "mr_ar_ea_combined_dict_b2plypd3.json"), "r"))
    
    ## load splits
    splits_dir = os.path.join(data_dir, "splits")
    stratified_k_fold_splits_dict = json.load(open(os.path.join(splits_dir, "stratified_k_fold_splits_dict.json"), "r"))
    random_k_fold_splits_dict = json.load(open(os.path.join(splits_dir, "random_k_fold_splits_dict.json"), "r"))
    mr_scaffold_k_fold_splits_dict = json.load(open(os.path.join(splits_dir, "mr_scaffold_k_fold_splits_dict.json"), "r"))
    
    stratified_k_fold_splits_combined_dict = splits_dict_to_combined_splits_dict(stratified_k_fold_splits_dict)
    random_k_fold_splits_combined_dict = splits_dict_to_combined_splits_dict(random_k_fold_splits_dict)
    mr_scaffold_k_fold_splits_combined_dict = splits_dict_to_combined_splits_dict(mr_scaffold_k_fold_splits_dict)
    
    k_fold_splits_name_dict = {
        "stratified": stratified_k_fold_splits_dict,
        "random": random_k_fold_splits_dict,
        "mr_scaffold": mr_scaffold_k_fold_splits_dict,
        "stratified_combined": stratified_k_fold_splits_combined_dict,
        "random_combined": random_k_fold_splits_combined_dict,
        "mr_scaffold_combined": mr_scaffold_k_fold_splits_combined_dict,
    }
    
    preds_dirs_list = [
        xgb_direct_preds_dir,
        xgb_delta_preds_dir,
        chemprop_direct_preds_dir,
        chemprop_delta_preds_dir,
        # chemprop_direct_transfer_preds_dir_1,
        # chemprop_direct_transfer_preds_dir_2,
        # chemprop_delta_transfer_preds_dir_1,
        # chemprop_delta_transfer_preds_dir_2
    ]
    
    # xgb_delta
    xgb_delta_preds_dicts_name_list = os.listdir(xgb_delta_preds_dir)
    xgb_delta_preds_dicts_name_list.sort()
    for xgb_delta_preds_dict_name in tqdm(xgb_delta_preds_dicts_name_list, desc="plotting xgb_delta plots"):
        print(xgb_delta_preds_dict_name)
        mr_ar_ea_dict_to_use, k_fold_splits_dict_to_use = return_ea_dict_splits_dict_to_use(xgb_delta_preds_dict_name, mr_ar_ea_name_dict, k_fold_splits_name_dict)
        xgb_delta_preds_dict_path = os.path.join(xgb_delta_preds_dir, xgb_delta_preds_dict_name)
        xgb_delta_save_name = xgb_delta_preds_dict_name.split("/")[-1].split(".")[0]
        if "combined" in xgb_delta_preds_dict_name:
            xgb_delta_save_name = f"{xgb_delta_save_name}_fwd"  ## only fwd preds parity will be plotted.
        # rxn_features_fn_grps_accuracy_box(mr_ar_ea_dict_to_use, k_fold_splits_dict_to_use, xgb_delta_preds_dict_path, ar_smi_dict, ar_fn_groups_added_dict, save_dir=os.path.join(detailed_accuracy_plots_dir, "xgb_delta", "rxn_features_fn_grps_accuracy_box"), save_name=xgb_delta_save_name)
        # ar_ha_count_accuracy_box(mr_ar_ea_dict_to_use, k_fold_splits_dict_to_use, xgb_delta_preds_dict_path, ar_smi_dict, save_dir=os.path.join(detailed_accuracy_plots_dir, "xgb_delta", "ar_ha_count_accuracy_box"), save_name=xgb_delta_save_name)
        ar_reduced_ha_count_accuracy_box(mr_ar_ea_dict_to_use, k_fold_splits_dict_to_use, xgb_delta_preds_dict_path, ar_smi_dict, mr_smi_dict, save_dir=os.path.join(detailed_accuracy_plots_dir, "xgb_delta", "ar_reduced_ha_count_accuracy_box"), save_name=xgb_delta_save_name)
        # ar_count_per_mr_accuracy_box(mr_ar_ea_dict_to_use, k_fold_splits_dict_to_use, xgb_delta_preds_dict_path, mr_ar_list_dict, save_dir=os.path.join(detailed_accuracy_plots_dir, "xgb_delta", "ar_count_per_mr_accuracy_box"), save_name=xgb_delta_save_name)
    
    # xgb_direct
    xgb_direct_preds_dicts_name_list = os.listdir(xgb_direct_preds_dir)
    xgb_direct_preds_dicts_name_list.sort()
    for xgb_direct_preds_dict_name in tqdm(xgb_direct_preds_dicts_name_list, desc="plotting xgb_direct plots"):
        print(xgb_direct_preds_dict_name)
        mr_ar_ea_dict_to_use, k_fold_splits_dict_to_use = return_ea_dict_splits_dict_to_use(xgb_direct_preds_dict_name, mr_ar_ea_name_dict, k_fold_splits_name_dict)
        xgb_direct_preds_dict_path = os.path.join(xgb_direct_preds_dir, xgb_direct_preds_dict_name)
        xgb_direct_save_name = xgb_direct_preds_dict_name.split("/")[-1].split(".")[0]
        if "combined" in xgb_direct_preds_dict_name:
            xgb_direct_save_name = f"{xgb_direct_save_name}_fwd"
        # rxn_features_fn_grps_accuracy_box(mr_ar_ea_dict_to_use, k_fold_splits_dict_to_use, xgb_direct_preds_dict_path, ar_smi_dict, ar_fn_groups_added_dict, save_dir=os.path.join(detailed_accuracy_plots_dir, "xgb_direct", "rxn_features_fn_grps_accuracy_box"), save_name=xgb_direct_save_name)
        # ar_ha_count_accuracy_box(mr_ar_ea_dict_to_use, k_fold_splits_dict_to_use, xgb_direct_preds_dict_path, ar_smi_dict, save_dir=os.path.join(detailed_accuracy_plots_dir, "xgb_direct", "ar_ha_count_accuracy_box"), save_name=xgb_direct_save_name)
        ar_reduced_ha_count_accuracy_box(mr_ar_ea_dict_to_use, k_fold_splits_dict_to_use, xgb_direct_preds_dict_path, ar_smi_dict, mr_smi_dict, save_dir=os.path.join(detailed_accuracy_plots_dir, "xgb_direct", "ar_reduced_ha_count_accuracy_box"), save_name=xgb_direct_save_name)
        # ar_count_per_mr_accuracy_box(mr_ar_ea_dict_to_use, k_fold_splits_dict_to_use, xgb_direct_preds_dict_path, mr_ar_list_dict, save_dir=os.path.join(detailed_accuracy_plots_dir, "xgb_direct", "ar_count_per_mr_accuracy_box"), save_name=xgb_direct_save_name)
        
    # chemprop_delta
    chemprop_delta_preds_dicts_name_list = os.listdir(chemprop_delta_preds_dir)
    chemprop_delta_preds_dicts_name_list.sort()
    for chemprop_delta_preds_dict_name in tqdm(chemprop_delta_preds_dicts_name_list, desc="plotting chemprop_delta plots"):
        print(chemprop_delta_preds_dict_name)
        mr_ar_ea_dict_to_use, k_fold_splits_dict_to_use = return_ea_dict_splits_dict_to_use(chemprop_delta_preds_dict_name, mr_ar_ea_name_dict, k_fold_splits_name_dict)
        chemprop_delta_preds_dict_path = os.path.join(chemprop_delta_preds_dir, chemprop_delta_preds_dict_name)
        chemprop_delta_save_name = chemprop_delta_preds_dict_name.split("/")[-1].split(".")[0]
        if "combined" in chemprop_delta_preds_dict_name:
            chemprop_delta_save_name = f"{chemprop_delta_save_name}_fwd"
        # rxn_features_fn_grps_accuracy_box(mr_ar_ea_dict_to_use, k_fold_splits_dict_to_use, chemprop_delta_preds_dict_path, ar_smi_dict, ar_fn_groups_added_dict, save_dir=os.path.join(detailed_accuracy_plots_dir, "chemprop_delta", "rxn_features_fn_grps_accuracy_box"), save_name=chemprop_delta_save_name)
        # ar_ha_count_accuracy_box(mr_ar_ea_dict_to_use, k_fold_splits_dict_to_use, chemprop_delta_preds_dict_path, ar_smi_dict, save_dir=os.path.join(detailed_accuracy_plots_dir, "chemprop_delta", "ar_ha_count_accuracy_box"), save_name=chemprop_delta_save_name)
        ar_reduced_ha_count_accuracy_box(mr_ar_ea_dict_to_use, k_fold_splits_dict_to_use, chemprop_delta_preds_dict_path, ar_smi_dict, mr_smi_dict, save_dir=os.path.join(detailed_accuracy_plots_dir, "chemprop_delta", "ar_reduced_ha_count_accuracy_box"), save_name=chemprop_delta_save_name)
        # ar_count_per_mr_accuracy_box(mr_ar_ea_dict_to_use, k_fold_splits_dict_to_use, chemprop_delta_preds_dict_path, mr_ar_list_dict, save_dir=os.path.join(detailed_accuracy_plots_dir, "chemprop_delta", "ar_count_per_mr_accuracy_box"), save_name=chemprop_delta_save_name)
    
    # chemprop_direct
    chemprop_direct_preds_dicts_name_list = os.listdir(chemprop_direct_preds_dir)
    chemprop_direct_preds_dicts_name_list.sort()
    for chemprop_direct_preds_dict_name in tqdm(chemprop_direct_preds_dicts_name_list, desc="plotting chemprop_direct parity"):
        print(chemprop_direct_preds_dict_name)
        mr_ar_ea_dict_to_use, k_fold_splits_dict_to_use = return_ea_dict_splits_dict_to_use(chemprop_direct_preds_dict_name, mr_ar_ea_name_dict, k_fold_splits_name_dict)
        chemprop_direct_preds_dict_path = os.path.join(chemprop_direct_preds_dir, chemprop_direct_preds_dict_name)
        chemprop_direct_save_name = chemprop_direct_preds_dict_name.split("/")[-1].split(".")[0]
        if "combined" in chemprop_direct_preds_dict_name:
            chemprop_direct_save_name = f"{chemprop_direct_save_name}_fwd"
        # rxn_features_fn_grps_accuracy_box(mr_ar_ea_dict_to_use, k_fold_splits_dict_to_use, chemprop_direct_preds_dict_path, ar_smi_dict, ar_fn_groups_added_dict, save_dir=os.path.join(detailed_accuracy_plots_dir, "chemprop_direct", "rxn_features_fn_grps_accuracy_box"), save_name=chemprop_direct_save_name)
        # ar_ha_count_accuracy_box(mr_ar_ea_dict_to_use, k_fold_splits_dict_to_use, chemprop_direct_preds_dict_path, ar_smi_dict, save_dir=os.path.join(detailed_accuracy_plots_dir, "chemprop_direct", "ar_ha_count_accuracy_box"), save_name=chemprop_direct_save_name)
        ar_reduced_ha_count_accuracy_box(mr_ar_ea_dict_to_use, k_fold_splits_dict_to_use, chemprop_direct_preds_dict_path, ar_smi_dict, mr_smi_dict, save_dir=os.path.join(detailed_accuracy_plots_dir, "chemprop_direct", "ar_reduced_ha_count_accuracy_box"), save_name=chemprop_direct_save_name)
        # ar_count_per_mr_accuracy_box(mr_ar_ea_dict_to_use, k_fold_splits_dict_to_use, chemprop_direct_preds_dict_path, mr_ar_list_dict, save_dir=os.path.join(detailed_accuracy_plots_dir, "chemprop_direct", "ar_count_per_mr_accuracy_box"), save_name=chemprop_direct_save_name)
    
    return


if __name__ == "__main__":
    main()
