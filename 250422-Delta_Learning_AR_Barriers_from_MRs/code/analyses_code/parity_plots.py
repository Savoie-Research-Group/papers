"""
    Last Modified: 2025/04/04
    Author: Veerupaksh (Veeru) Singla (singla2@purdue.edu)
    Description: 
"""

import os


this_script_dir = os.path.dirname(os.path.realpath(__file__))
os.chdir(this_script_dir)


import sys
sys.path.append(os.path.join(this_script_dir, ".."))


from utils import *


parity_plots_dir = os.path.join(analyses_dir, "parity_plots")
train_perc_error_plots_dir = os.path.join(analyses_dir, "train_perc_error_plots")

chemprop_delta_preds_dir = os.path.join(preds_dir, "chemprop_delta")
chemprop_direct_preds_dir = os.path.join(preds_dir, "chemprop_direct")
xgb_delta_preds_dir = os.path.join(preds_dir, "xgb_delta")
xgb_direct_preds_dir = os.path.join(preds_dir, "xgb_direct")


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


def plot_parity_b3lyp_single_pt_refine(mr_ar_ea_dict, mr_ar_ea_dict_sp_refined, params=None):
    print(len(mr_ar_ea_dict), len(mr_ar_ea_dict_sp_refined))
    
    ## plot parity plot for current ea data (b3lyp-d3/tzvp) vs ea data after single point refinement (wb97xd/def2-tzvp, and b2plypd3/cc-pvtz single point refinement)
    
    if params is None:
        params = {}
        
    mr_ar_list = list(mr_ar_ea_dict_sp_refined.keys())
    mr_ar_ea_list = [mr_ar_ea_dict[mr_ar] for mr_ar in mr_ar_list]
    mr_ar_ea_list_sp_refined = [mr_ar_ea_dict_sp_refined[mr_ar] for mr_ar in mr_ar_list]
    
    params_default = {
        "save_dir": parity_plots_dir,
        "save_name": "parity_sp_refined",
        "x_range": [0, 200],
        "y_range": [0, 200],
        "x_label": "B3LYP-D3 Ea (kcal/mol)",
        "y_label": "SP Refined Ea (kcal/mol)",
        "figsize": (3, 3),
        "aspect": 1,
        "cmap": "plasma",
    }
    
    params = {**params_default, **params}
    if not os.path.exists(params["save_dir"]):
        os.makedirs(params["save_dir"])
    
    r2 = np.corrcoef(mr_ar_ea_list, mr_ar_ea_list_sp_refined)[0, 1]**2
    median_ae = np.median([abs(x-y) for x, y in zip(mr_ar_ea_list, mr_ar_ea_list_sp_refined)])  ## kcal/mol
    mae = np.mean([abs(x-y) for x, y in zip(mr_ar_ea_list, mr_ar_ea_list_sp_refined)])  ## kcal/mol
    rmse = np.sqrt(np.mean([(x-y)**2 for x, y in zip(mr_ar_ea_list, mr_ar_ea_list_sp_refined)]))  ## kcal/mol
    
    err_list = [mr_ar_ea_list_sp_refined[i] - mr_ar_ea_list[i] for i in range(len(mr_ar_ea_list))]
    kde_values = gaussian_kde(err_list)(err_list)
    
    plt.clf()
    fig, ax = plt.subplots(figsize=params["figsize"])
    ax.set_aspect(params["aspect"])
    
    scatter = ax.scatter(mr_ar_ea_list, mr_ar_ea_list_sp_refined, c=kde_values, cmap=params["cmap"], s=1, alpha=0.8, edgecolors='none')
    ax.plot([params["x_range"][0], params["x_range"][1]], [params["y_range"][0], params["y_range"][1]], color='black', linestyle='--', linewidth=1, label="Parity")
    
    err_line = 30
    ax.plot([params["x_range"][0], params["x_range"][1] - err_line], [params["y_range"][0] + err_line, params["y_range"][1]], color='red', linestyle='--', linewidth=1)
    ax.plot([params["x_range"][0] + err_line, params["x_range"][1]], [params["y_range"][0], params["y_range"][1] - err_line], color='red', linestyle='--', linewidth=1, label="\u00B130")
    
    
    ax.grid(which='major', color='gray', linestyle='-', linewidth=0.5, alpha=0.5, zorder=0)
    ax.grid(which='minor', axis='both', color='gray', linestyle=':', linewidth=0.5, alpha=0.3, zorder=0)
    ax.set_axisbelow(True)
    ax.minorticks_on()
    # ax.set_title(params["save_name"], fontsize=8)
    
    ax.tick_params(axis='both', which='both', labelsize=14, direction='in')
    
    ax.set_xlim(params["x_range"][0], params["x_range"][1])
    ax.set_xlabel(params["x_label"], fontsize=14)
    ax.set_xticks(np.arange(params["x_range"][0], params["x_range"][1]+1, 50))
    ax.set_ylim(params["y_range"][0], params["y_range"][1])
    ax.set_ylabel(params["y_label"], fontsize=14)
    ax.set_yticks(np.arange(params["y_range"][0], params["y_range"][1]+1, 50))
    
    ax.text(0.96, 0.05,
            f"R\u00B2: {r2:.2f}\nMedAE: {median_ae:.2f}\nMAE: {mae:.2f}\nRMSE: {rmse:.2f}",
            transform=ax.transAxes, 
            verticalalignment='bottom', 
            horizontalalignment='right',
            fontsize=8,
            bbox=dict(facecolor='white', edgecolor='none', alpha=0.8))
    
    ax.legend(loc='upper left', fontsize=8, handlelength=1.5, labelspacing=0.15, handletextpad=0.2, borderpad=0.2, borderaxespad=0.2, columnspacing=0.5, ncols=1) #, frameon=False
    
    plt.tight_layout()
    plt.savefig(f"{os.path.join(params['save_dir'], params['save_name'])}.png", dpi=600, bbox_inches='tight', pad_inches=0.005)
    plt.savefig(f"{os.path.join(params['save_dir'], params['save_name'])}.pdf", dpi=600, bbox_inches='tight', pad_inches=0.005, transparent=True)
    plt.close()
    
    return


def plot_k_fold_parity(k_fold_splits_dict, mr_ar_ea_dict, k_fold_preds_dict=None, k_fold_preds_dict_path=None, params=None):
    assert k_fold_preds_dict is not None or k_fold_preds_dict_path is not None, "k_fold_preds_dict or k_fold_preds_dict_path should be provided."
    
    if k_fold_preds_dict is None:
        k_fold_preds_dict = json.load(open(k_fold_preds_dict_path, "r"))
    
    if params is None:
        params = {}
    
    ## will only plot test for now. creating train and val for later.
    train_ea_list = []
    train_ea_pred_list = []
    
    val_ea_list = []
    val_ea_pred_list = []
    
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
                train_ea_list.append(mr_ar_ea_dict[mr_ar])
                train_ea_pred_list.append(mr_ar_pred_dict[mr_ar])
        
        for mr_ar in val_mr_ar_list:
            if mr_ar in mr_ar_pred_dict:
                val_ea_list.append(mr_ar_ea_dict[mr_ar])
                val_ea_pred_list.append(mr_ar_pred_dict[mr_ar])
        
        for mr_ar in test_mr_ar_list:
            if mr_ar in mr_ar_pred_dict:
                test_ea_list.append(mr_ar_ea_dict[mr_ar])
                test_ea_pred_list.append(mr_ar_pred_dict[mr_ar])

    params_default = {
        "save_dir": parity_plots_dir,
        "save_name": "parity",
        "x_range": [0, 200],
        "y_range": [0, 200],
        "x_label": "Actual Ea (kcal/mol)",
        "y_label": "Predicted Ea (kcal/mol)",
        "figsize": (3, 3),
        "aspect": 1,
        "cmap": "plasma",
    }
    
    params = {**params_default, **params}
    
    if not os.path.exists(params["save_dir"]):
        os.makedirs(params["save_dir"])
    
    train_r2 = np.corrcoef(train_ea_list, train_ea_pred_list)[0, 1]**2
    train_median_ae = np.median([abs(x-y) for x, y in zip(train_ea_list, train_ea_pred_list)])  ## kcal/mol
    train_mae = np.mean([abs(x-y) for x, y in zip(train_ea_list, train_ea_pred_list)])  ## kcal/mol
    train_rmse = np.sqrt(np.mean([(x-y)**2 for x, y in zip(train_ea_list, train_ea_pred_list)]))  ## kcal/mol
    
    val_r2 = np.corrcoef(val_ea_list, val_ea_pred_list)[0, 1]**2
    val_median_ae = np.median([abs(x-y) for x, y in zip(val_ea_list, val_ea_pred_list)])  ## kcal/mol
    val_mae = np.mean([abs(x-y) for x, y in zip(val_ea_list, val_ea_pred_list)])  ## kcal/mol
    val_rmse = np.sqrt(np.mean([(x-y)**2 for x, y in zip(val_ea_list, val_ea_pred_list)]))  ## kcal/mol
    
    test_r2 = np.corrcoef(test_ea_list, test_ea_pred_list)[0, 1]**2
    test_median_ae = np.median([abs(x-y) for x, y in zip(test_ea_list, test_ea_pred_list)])  ## kcal/mol
    test_mae = np.mean([abs(x-y) for x, y in zip(test_ea_list, test_ea_pred_list)])  ## kcal/mol
    test_rmse = np.sqrt(np.mean([(x-y)**2 for x, y in zip(test_ea_list, test_ea_pred_list)]))  ## kcal/mol
    
    train_err_list = [train_ea_pred_list[i] - train_ea_list[i] for i in range(len(train_ea_list))]
    val_err_list = [val_ea_pred_list[i] - val_ea_list[i] for i in range(len(val_ea_list))]
    test_err_list = [test_ea_pred_list[i] - test_ea_list[i] for i in range(len(test_ea_list))]
    
    # train_kde_values = gaussian_kde(train_err_list)(train_err_list)
    # val_kde_values = gaussian_kde(val_err_list)(val_err_list)
    test_kde_values = gaussian_kde(test_err_list)(test_err_list)
    
    plt.clf()
    fig, ax = plt.subplots(figsize=params["figsize"])
    ax.set_aspect(params["aspect"])
    
    scatter = ax.scatter(test_ea_list, test_ea_pred_list, c=test_kde_values, cmap=params["cmap"], s=1, alpha=0.8, label="Test", edgecolors='none')
    # scatter = ax.scatter(train_ea_list, train_ea_pred_list, c=train_kde_values, cmap=params["cmap"], s=2, alpha=0.8, label="Train", edgecolors='none')
    # fig.colorbar(scatter, ax=ax, label="ErrorDensity")
    
    ax.plot([params["x_range"][0], params["x_range"][1]], [params["y_range"][0], params["y_range"][1]], color='black', linestyle='--', linewidth=1, label="Parity")
    
    err_line = 30
    ax.plot([params["x_range"][0], params["x_range"][1] - err_line], [params["y_range"][0] + err_line, params["y_range"][1]], color='red', linestyle='--', linewidth=1, label="Parity")
    ax.plot([params["x_range"][0] + err_line, params["x_range"][1]], [params["y_range"][0], params["y_range"][1] - err_line], color='red', linestyle='--', linewidth=1, label="Parity")
    
    ax.grid(which='major', color='gray', linestyle='-', linewidth=0.5, alpha=0.5, zorder=0)
    ax.grid(which='minor', axis='both', color='gray', linestyle=':', linewidth=0.5, alpha=0.3, zorder=0)
    ax.set_axisbelow(True)
    ax.minorticks_on()
    
    # ax.set_title(params["save_name"], fontsize=8)
    
    ax.tick_params(axis='both', which='both', labelsize=14, direction='in')
    
    ax.set_xlim(params["x_range"][0], params["x_range"][1])
    ax.set_xlabel(params["x_label"], fontsize=14)
    ax.set_xticks(np.arange(params["x_range"][0], params["x_range"][1]+1, 50))
    ax.set_ylim(params["y_range"][0], params["y_range"][1])
    ax.set_ylabel(params["y_label"], fontsize=14)
    ax.set_yticks(np.arange(params["y_range"][0], params["y_range"][1]+1, 50))
    
    # ax.legend(handles=[ax.get_lines()[0]], fontsize=10, loc='upper left')
    
    ## plot r2 and mae values
    
    ax.text(0.96, 0.05,
            f"R\u00B2: {test_r2:.2f}\nMedAE: {test_median_ae:.2f}\nMAE: {test_mae:.2f}\nRMSE: {test_rmse:.2f}",
            # f"R\u00B2: {train_r2:.2f}\nMedAE: {train_median_ae:.2f}\nMAE: {train_mae:.2f}\nRMSE: {train_rmse:.2f}",
            transform=ax.transAxes, 
            verticalalignment='bottom', 
            horizontalalignment='right',
            fontsize=8,
            bbox=dict(facecolor='white', edgecolor='none', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(f"{os.path.join(params['save_dir'], params['save_name'])}.png", dpi=600, bbox_inches='tight', pad_inches=0.005)
    plt.savefig(f"{os.path.join(params['save_dir'], params['save_name'])}.pdf", dpi=600, bbox_inches='tight', pad_inches=0.005, transparent=True)
    plt.close()
    
    return


def check_create_dirs():
    if not os.path.exists(parity_plots_dir):
        os.makedirs(parity_plots_dir)
    
    if not os.path.exists(train_perc_error_plots_dir):
        os.makedirs(train_perc_error_plots_dir)
    
    return


def main():
    check_create_dirs()
    
    ## load ea dicts
    ar_ea_fwd_dict_path = os.path.join(data_paper_data_dir, "ar_ea_fwd_dict.json")
    ar_ea_fwd_dict = json.load(open(ar_ea_fwd_dict_path, "r"))
    
    ar_ea_rev_dict_path = os.path.join(data_paper_data_dir, "ar_ea_rev_dict.json")
    ar_ea_rev_dict = json.load(open(ar_ea_rev_dict_path, "r"))
    
    ar_ea_combined_dict = deepcopy(ar_ea_fwd_dict)
    ar_ea_combined_dict.update({f"{k}_rev": v for k, v in ar_ea_rev_dict.items()})
    
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
    
    
    # ##################
    # ## parity plots ##
    # ##################
    
    # single pt refinement
    # plot_parity_b3lyp_single_pt_refine(mr_ar_ea_combined_dict, mr_ar_ea_dict_wb97xd_refined, params={"save_name": "parity_sp_refined_wb97xd", "y_label": "wB97XD SP Ea (kcal/mol)"})
    # plot_parity_b3lyp_single_pt_refine(mr_ar_ea_combined_dict, mr_ar_ea_dict_b2plypd3_refined, params={"save_name": "parity_sp_refined_b2plypd3", "y_label": "B2PLYPD3 SP Ea (kcal/mol)"})
    
    # plot_parity_b3lyp_single_pt_refine(create_delta_ea_dict(mr_ar_ea_combined_dict), create_delta_ea_dict(mr_ar_ea_dict_wb97xd_refined), params={"save_name": "parity_sp_refined_wb97xd_delta", "y_label": "wB97XD SP Delta Ea (kcal/mol)", "x_label": "B3LYP-D3 Delta Ea (kcal/mol)", "x_range": [0, 50], "y_range": [0, 50]})
    # plot_parity_b3lyp_single_pt_refine(create_delta_ea_dict(mr_ar_ea_combined_dict), create_delta_ea_dict(mr_ar_ea_dict_b2plypd3_refined), params={"save_name": "parity_sp_refined_b2plypd3_delta", "y_label": "B2PLYPD3 SP Delta Ea (kcal/mol)", "x_label": "B3LYP-D3 Delta Ea (kcal/mol)", "x_range": [0, 50], "y_range": [0, 50]})
    
    # # xgb_delta
    # xgb_delta_preds_dicts_name_list = os.listdir(xgb_delta_preds_dir)
    # xgb_delta_preds_dicts_name_list.sort()
    # for xgb_delta_preds_dict_name in tqdm(xgb_delta_preds_dicts_name_list, desc="plotting xgb_delta parity"):
    #     mr_ar_ea_dict_to_use, k_fold_splits_dict_to_use = return_ea_dict_splits_dict_to_use(xgb_delta_preds_dict_name, mr_ar_ea_name_dict, k_fold_splits_name_dict)
    #     xgb_delta_preds_dict_path = os.path.join(xgb_delta_preds_dir, xgb_delta_preds_dict_name)
    #     xgb_delta_save_name = xgb_delta_preds_dict_name.split("/")[-1].split(".")[0]
    #     if "combined" in xgb_delta_preds_dict_name:
    #         xgb_delta_save_name = f"{xgb_delta_save_name}_fwd"  ## only fwd preds parity will be plotted.
    #     plot_k_fold_parity(k_fold_splits_dict_to_use, mr_ar_ea_dict_to_use, k_fold_preds_dict_path=xgb_delta_preds_dict_path, params={"save_dir": os.path.join(parity_plots_dir,"xgb_delta"), "save_name": xgb_delta_save_name})
    
    # # xgb_direct
    # xgb_direct_preds_dicts_name_list = os.listdir(xgb_direct_preds_dir)
    # xgb_direct_preds_dicts_name_list.sort()
    # for xgb_direct_preds_dict_name in tqdm(xgb_direct_preds_dicts_name_list, desc="plotting xgb_direct parity"):
    #     mr_ar_ea_dict_to_use, k_fold_splits_dict_to_use = return_ea_dict_splits_dict_to_use(xgb_direct_preds_dict_name, mr_ar_ea_name_dict, k_fold_splits_name_dict)
    #     xgb_direct_preds_dict_path = os.path.join(xgb_direct_preds_dir, xgb_direct_preds_dict_name)
    #     xgb_direct_save_name = xgb_direct_preds_dict_name.split("/")[-1].split(".")[0]
    #     if "combined" in xgb_direct_preds_dict_name:
    #         xgb_direct_save_name = f"{xgb_direct_save_name}_fwd"
    #     plot_k_fold_parity(k_fold_splits_dict_to_use, mr_ar_ea_dict_to_use, k_fold_preds_dict_path=xgb_direct_preds_dict_path, params={"save_dir": os.path.join(parity_plots_dir,"xgb_direct"), "save_name": xgb_direct_save_name})
    
    # # chemprop_delta
    # chemprop_delta_preds_dicts_name_list = os.listdir(chemprop_delta_preds_dir)
    # chemprop_delta_preds_dicts_name_list.sort()
    # for chemprop_delta_preds_dict_name in tqdm(chemprop_delta_preds_dicts_name_list, desc="plotting chemprop_delta parity"):
    #     mr_ar_ea_dict_to_use, k_fold_splits_dict_to_use = return_ea_dict_splits_dict_to_use(chemprop_delta_preds_dict_name, mr_ar_ea_name_dict, k_fold_splits_name_dict)
    #     chemprop_delta_preds_dict_path = os.path.join(chemprop_delta_preds_dir, chemprop_delta_preds_dict_name)
    #     chemprop_delta_save_name = chemprop_delta_preds_dict_name.split("/")[-1].split(".")[0]
    #     if "combined" in chemprop_delta_preds_dict_name:
    #         chemprop_delta_save_name = f"{chemprop_delta_save_name}_fwd"
    #     plot_k_fold_parity(k_fold_splits_dict_to_use, mr_ar_ea_dict_to_use, k_fold_preds_dict_path=chemprop_delta_preds_dict_path, params={"save_dir": os.path.join(parity_plots_dir,"chemprop_delta"), "save_name": chemprop_delta_save_name})
    
    # # chemprop_direct
    # chemprop_direct_preds_dicts_name_list = os.listdir(chemprop_direct_preds_dir)
    # chemprop_direct_preds_dicts_name_list.sort()
    # for chemprop_direct_preds_dict_name in tqdm(chemprop_direct_preds_dicts_name_list, desc="plotting chemprop_direct parity"):
    #     mr_ar_ea_dict_to_use, k_fold_splits_dict_to_use = return_ea_dict_splits_dict_to_use(chemprop_direct_preds_dict_name, mr_ar_ea_name_dict, k_fold_splits_name_dict)
    #     chemprop_direct_preds_dict_path = os.path.join(chemprop_direct_preds_dir, chemprop_direct_preds_dict_name)
    #     chemprop_direct_save_name = chemprop_direct_preds_dict_name.split("/")[-1].split(".")[0]
    #     if "combined" in chemprop_direct_preds_dict_name:
    #         chemprop_direct_save_name = f"{chemprop_direct_save_name}_fwd"
    #     plot_k_fold_parity(k_fold_splits_dict_to_use, mr_ar_ea_dict_to_use, k_fold_preds_dict_path=chemprop_direct_preds_dict_path, params={"save_dir": os.path.join(parity_plots_dir,"chemprop_direct"), "save_name": chemprop_direct_save_name})
    
    
    ## mr surr
    ar_mr_surr_fwd_dict = {ar: mr_ar_ea_fwd_dict[ar_name_to_mr_name(ar)] for ar in ar_ea_fwd_dict.keys()}
    ar_mr_surr_rev_dict = {ar: mr_ar_ea_rev_dict[ar_name_to_mr_name(ar)] for ar in ar_ea_rev_dict.keys()}
    ar_mr_surr_combined_dict = {ar: mr_ar_ea_combined_dict[ar_name_to_mr_name(ar)] for ar in ar_ea_combined_dict.keys()}
    
    plot_parity_b3lyp_single_pt_refine(ar_ea_fwd_dict, ar_mr_surr_fwd_dict, params={"save_name": "parity_ar_mr_surr_fwd", "y_label": "MR Ea (kcal/mol)", "x_label": "AR Ea (kcal/mol)"})
    plot_parity_b3lyp_single_pt_refine(ar_ea_rev_dict, ar_mr_surr_rev_dict, params={"save_name": "parity_ar_mr_surr_rev", "y_label": "MR Ea (kcal/mol)", "x_label": "AR Ea (kcal/mol)"})
    plot_parity_b3lyp_single_pt_refine(ar_ea_combined_dict, ar_mr_surr_combined_dict, params={"save_name": "parity_ar_mr_surr_combined", "y_label": "MR Ea (kcal/mol)", "x_label": "AR Ea (kcal/mol)"})
    
    return


if __name__ == "__main__":
    main()
