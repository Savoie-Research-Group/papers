"""
    Last Modified: 2025/07/04
    Author: Veerupaksh (Veeru) Singla (singla2@purdue.edu)
    Description: make csv for each model predcition accuracy for alkane pyrolysis ers barrier prediction. testing extrapolation to radical species.
"""

import os


this_script_dir = os.path.dirname(os.path.realpath(__file__))
os.chdir(this_script_dir)


import sys
sys.path.append(os.path.join(this_script_dir, ".."))


from utils import *


alk_pyrolysis_analysis_dir = os.path.join(analyses_dir, "case_study_alkane_pyrolysis_analysis")

alk_pyrolysis_benchmark_dir = os.path.join(data_dir, "case_study_alkane_pyrolysis_mr_benchmark")
alk_pyrolysis_chemprop_delta_preds_dir = os.path.join(alk_pyrolysis_benchmark_dir, "chemprop_delta_preds")

refine_fn_1 = "wb97xd"
refine_fn_2 = "b2plypd3"
alk_pyrolysis_chemprop_delta_transfer_preds_dir_1 = os.path.join(alk_pyrolysis_benchmark_dir, f"chemprop_delta_transfer_preds_{refine_fn_1}")
alk_pyrolysis_chemprop_delta_transfer_preds_dir_2 = os.path.join(alk_pyrolysis_benchmark_dir, f"chemprop_delta_transfer_preds_{refine_fn_2}")


def get_mr_wise_k_fold_preds_list_dict(preds_list):
    ar_list, ar_preds_list_list, mr_list, mr_preds_list_list = preds_list[0], preds_list[1], preds_list[2], preds_list[3]
    
    ar_mr_k_fold_preds_list_dict = {}
    mr_mr_k_fold_preds_dict = {}
    for i in range(1, 60):
        mr_i = f"MR_{i}_0"
        
        if mr_i in mr_list:
            ar_idx_list_i = [j for j in range(len(ar_list)) if mr_i in ar_list[j]]
            mr_idx_i = [j for j in range(len(mr_list)) if mr_list[j] == mr_i][0]
            
            ar_list_i = [ar_list[j] for j in ar_idx_list_i]
            ar_k_fold_preds_list_list_i = [[ar_preds_list[j] for j in ar_idx_list_i] for ar_preds_list in ar_preds_list_list]
            ar_mr_k_fold_preds_list_dict[mr_i] = [ar_list_i, ar_k_fold_preds_list_list_i]
            
            mr_k_fold_preds_list_i = [mr_preds_list[mr_idx_i] for mr_preds_list in mr_preds_list_list]
            mr_mr_k_fold_preds_dict[mr_i] = mr_k_fold_preds_list_i
            
    return ar_mr_k_fold_preds_list_dict, mr_mr_k_fold_preds_dict


def preds_dir_to_k_fold_mr_wise_accuracy_csv_list(preds_dir, mr_ar_ea_dict):
    csv_header = "preds_json_name"
    mr_list = []
    for i in range(1, 60):
        mr_i = f"MR_{i}_0"
        if mr_i in mr_ar_ea_dict:
            csv_header += f",median_ae_{mr_i},mae_{mr_i},rmse_{mr_i}"
            mr_list.append(mr_i)
    csv_list = [csv_header]
            
    preds_json_list = os.listdir(preds_dir)
    preds_json_list.sort()
    for preds_json_name in tqdm(preds_json_list):
        csv_list.append(f"{preds_json_name.split('.json')[0]}")
        preds_list = json.load(open(os.path.join(preds_dir, preds_json_name), "r"))
        ar_mr_k_fold_preds_list_dict, mr_mr_k_fold_preds_dict = get_mr_wise_k_fold_preds_list_dict(preds_list)
        for mr_i in mr_list:
            ar_list_i = ar_mr_k_fold_preds_list_dict[mr_i][0]
            ar_k_fold_preds_list_list_i = ar_mr_k_fold_preds_list_dict[mr_i][1]
            ar_k_fold_preds_list_list_i = np.array(ar_k_fold_preds_list_list_i)
            
            ar_ea_list_i = np.array([mr_ar_ea_dict[ar_i] for ar_i in ar_list_i])
            ar_preds_list_i = np.mean(ar_k_fold_preds_list_list_i, axis=0)
            stats_dict_i = return_stats(ar_ea_list_i, ar_preds_list_i)
            
            csv_list[-1] += f",{stats_dict_i['median_ae']},{stats_dict_i['mae']},{stats_dict_i['rmse']}"
    
    csv_list.append(f"MR_surrogate_pred")  ## using mr ea as direct prediction for ar ea
    for mr_i in mr_list:
        ar_list_i = ar_mr_k_fold_preds_list_dict[mr_i][0]
        ar_ea_list_i = np.array([mr_ar_ea_dict[ar_i] for ar_i in ar_list_i])
        ar_preds_list_i = np.array([mr_ar_ea_dict[mr_i] for _ in ar_list_i])
        stats_dict_i = return_stats(ar_ea_list_i, ar_preds_list_i)
        csv_list[-1] += f",{stats_dict_i['median_ae']},{stats_dict_i['mae']},{stats_dict_i['rmse']}"
        
        # ## for manual analysis
        # if stats_dict_i['mae'] > 1:
        #     print(mr_i)
    
    return csv_list


def return_best_ar_mr_k_fold_errs_list_dict_from_accuracy_csv(preds_dir, mr_ar_ea_dict, mr_name, csv_path=None):
    if csv_path is None:
        csv_list = preds_dir_to_k_fold_mr_wise_accuracy_csv_list(preds_dir, mr_ar_ea_dict)
    else:
        csv_list = [line.strip() for line in open(csv_path, "r").readlines()]
    
    csv_header_list = csv_list[0].split(",")
    mr_i = f"mae_{mr_name}"
    mr_i_idx = csv_header_list.index(mr_i)
    
    csv_list = csv_list[1:-1]  ## remove header and mr surrogate row
    csv_list.sort(key=lambda x: float(x.split(",")[mr_i_idx]))
    best_csv_list = csv_list[0]
    best_csv_list = best_csv_list.split(",")
    best_preds_json_name = best_csv_list[0]
    print(mr_name, best_preds_json_name)
    best_preds_json_path = os.path.join(preds_dir, best_preds_json_name + ".json")
    
    best_preds_list = json.load(open(best_preds_json_path, "r"))
    ar_mr_k_fold_preds_list_dict, mr_mr_k_fold_preds_dict = get_mr_wise_k_fold_preds_list_dict(best_preds_list)
    
    ar_list_i = ar_mr_k_fold_preds_list_dict[mr_name][0]
    ar_k_fold_preds_list_list_i = ar_mr_k_fold_preds_list_dict[mr_name][1]
    ar_preds_list = np.mean(ar_k_fold_preds_list_list_i, axis=0)
    
    ar_ea_list_i = [mr_ar_ea_dict[ar_i] for ar_i in ar_list_i]
    ar_k_fold_abs_err_list_list_i = np.array([[abs(ar_ea_list_i[j] - ar_k_fold_preds_list_list_i[k][j]) for j in range(len(ar_list_i))] for k in range(len(ar_k_fold_preds_list_list_i))])
    
    ar_abs_err_list = np.array([abs(ar_ea_list_i[j] - ar_preds_list[j]) for j in range(len(ar_list_i))])
    ar_mr_surr_abs_err_list = [abs(ar_ea_list_i[j] - mr_ar_ea_dict[mr_name]) for j in range(len(ar_list_i))]
    
    return ar_list_i, ar_k_fold_abs_err_list_list_i.tolist(), ar_abs_err_list.tolist(), ar_mr_surr_abs_err_list


def check_create_dirs():
    if not os.path.exists(alk_pyrolysis_analysis_dir):
        os.makedirs(alk_pyrolysis_analysis_dir)
    return


def violin_plot(mr_to_plot_list, mr_surr_err_list_list, ml_pred_err_list_list, save_path):
    plt.clf()
    
    fig = plt.figure(figsize=(5, 2.5))  # , constrained_layout=True

    gs = fig.add_gridspec(1, hspace=0, wspace=0)
    axs = gs.subplots(sharex=True, sharey=True)

    positions = np.array(list(range(1, len(mr_surr_err_list_list) + 1)))
    axs_violinparts1 = axs.violinplot(mr_surr_err_list_list, positions=positions, widths=0.75, showextrema=False, showmeans=True, showmedians=True, side='low')
    axs_violinparts2 = axs.violinplot(ml_pred_err_list_list, positions=positions, widths=0.75, showextrema=False, showmeans=True, showmedians=True, side='high')
    
    for partname in ["cmedians", "cmeans"]:
        vp1 = axs_violinparts1[partname]
        vp2 = axs_violinparts2[partname]
        if partname == "cmedians":
            vp1.set_edgecolor(vs_colors["orange"][0])
            vp1.set_linewidth(1)
            vp2.set_edgecolor(vs_colors["purple"][0])
            vp2.set_linewidth(1)
        elif partname == "cmeans":
            vp1.set_edgecolor("white")
            vp1.set_linewidth(1)
            vp2.set_edgecolor("white")
            vp2.set_linewidth(1)
        
        for vp in [vp1, vp2]:
            paths = vp.get_paths()
            for path in paths:
                vertices = path.vertices
                if vp == vp1:
                    vertices[:, 0] -= 0.05
                elif vp == vp2:
                    vertices[:, 0] += 0.05
                # vertices[0, 0] -= 0.05
                # vertices[-1, 0] += 0.05
    
    for pc in axs_violinparts1["bodies"]:
        pc.set_facecolor(vs_colors["orange"][0])
        pc.set_alpha(0.5)
        pc.set_edgecolor('none')
    for pc in axs_violinparts2["bodies"]:
        pc.set_facecolor(vs_colors["purple"][0])
        pc.set_alpha(0.5)
        pc.set_edgecolor('none')
    
    q1_1 = [np.percentile(l, 25) for l in mr_surr_err_list_list]
    q3_1 = [np.percentile(l, 75) for l in mr_surr_err_list_list]
    q1_2 = [np.percentile(l, 25) for l in ml_pred_err_list_list]
    q3_2 = [np.percentile(l, 75) for l in ml_pred_err_list_list]
    
    inds = np.arange(1, len(q1_1) + 1)
    
    axs.vlines(inds - 0.05, q1_1, q3_1, color=vs_colors["orange"][0], linestyle='-', lw=1.5)
    axs.vlines(inds + 0.05, q1_2, q3_2, color=vs_colors["purple"][0], linestyle='-', lw=1.5)
    
    axs.grid(which='major', axis='y', color='gray', linestyle='-', linewidth=0.5, alpha=0.5, zorder=0)
    axs.grid(which='minor', axis='y', color='gray', linestyle=':', linewidth=0.5, alpha=0.3, zorder=0)
    axs.set_axisbelow(True)
    axs.minorticks_on()
    axs.tick_params(axis='both', which='both', labelsize=14, direction='in')
    axs.tick_params(axis='x', which='minor', bottom=False)  # Disable minor ticks on x-axis
    axs.tick_params(axis='y', which='minor', left=True)
    
    axs.set_xlim([positions[0]-0.5, positions[-1]+0.5])
    axs.set_xlabel("MR", fontsize=14)
    axs.set_xticks(positions)
    axs.set_xticklabels(mr_to_plot_list, rotation=90, fontsize=14)  # , ha="right")
    
    y_max = 5
    axs.set_ylim(0, y_max)
    axs.set_yticks(np.arange(0, y_max + 0.1, 1))
    axs.set_ylabel("Absolute Error", fontsize=14)
    
    axs.legend([axs_violinparts1["bodies"][0], axs_violinparts2["bodies"][0]], ["MR surrogate", "ML prediction"], loc="upper center", fontsize=10, ncols=2,
               handlelength=1.5, labelspacing=0.00, handletextpad=0.15, borderpad=0.15, borderaxespad=0.15, columnspacing=1.0) # , frameon=False)
    
    axs.label_outer()
    plt.tight_layout()
    plt.savefig(save_path + ".pdf", dpi=600, transparent=True, bbox_inches='tight', pad_inches=0.005)
    
    plt.close()
    
    return


def transfer_compare_scatter_plot(mr_list, mr_surr_mae_list, mr_surr_medae_list, ml_pred_mae_list_0, ml_pred_medae_list_0, ml_pred_mae_list_1, ml_pred_medae_list_1, ml_pred_mae_list_2, ml_pred_medae_list_2, save_path):
    plt.clf()
    fig, ax = plt.subplots(figsize=(2.75, 2.25))
    
    x_tick_pos_list = np.array(range(1, len(mr_list) + 1))
    x_tick_label_list = mr_list
    
    median_marker = "*"
    mean_marker = "."
    markersize=6
    
    alpha = 0.7
    
    ax.plot(x_tick_pos_list-0.15, mr_surr_mae_list, marker=mean_marker, linewidth=0.0, markersize=markersize, markeredgewidth=0, color=vs_colors["orange"][0], alpha=alpha, label="MR surrogate MAE")
    ax.plot(x_tick_pos_list+0.15, mr_surr_medae_list, marker=median_marker, linewidth=0.0, markersize=markersize, markeredgewidth=0, color=vs_colors["orange"][0], alpha=alpha, label="MR surrogate MedAE")
    
    ax.plot(x_tick_pos_list-0.15, ml_pred_mae_list_0, marker=mean_marker, linewidth=0.0, markersize=markersize, markeredgewidth=0, color=vs_colors["purple"][0], alpha=alpha, label="ML prediction MAE (original)")
    ax.plot(x_tick_pos_list+0.15, ml_pred_medae_list_0, marker=median_marker, linewidth=0.0, markersize=markersize, markeredgewidth=0, color=vs_colors["purple"][0], alpha=alpha, label="ML prediction MedAE (original)")
    
    ax.plot(x_tick_pos_list-0.15, ml_pred_mae_list_1, marker=mean_marker, linewidth=0.0, markersize=markersize, markeredgewidth=0, color=vs_colors["green"][0], alpha=alpha, label="ML prediction MAE (\u03C9B97XD)")
    ax.plot(x_tick_pos_list+0.15, ml_pred_medae_list_1, marker=median_marker, linewidth=0.0, markersize=markersize, markeredgewidth=0, color=vs_colors["green"][0], alpha=alpha, label="ML prediction MedAE (\u03C9B97XD)")
    
    ax.plot(x_tick_pos_list-0.15, ml_pred_mae_list_2, marker=mean_marker, linewidth=0.0, markersize=markersize, markeredgewidth=0, color=vs_colors["blue"][0], alpha=alpha, label="ML prediction MAE (B2PLYPD3)")
    ax.plot(x_tick_pos_list+0.15, ml_pred_medae_list_2, marker=median_marker, linewidth=0.0, markersize=markersize, markeredgewidth=0, color=vs_colors["blue"][0], alpha=alpha, label="ML prediction MedAE (B2PLYPD3)")
    
    ax.grid(which='major', axis='y', color='gray', linestyle='-', linewidth=0.5, alpha=0.5, zorder=0)
    ax.grid(which='minor', axis='y', color='gray', linestyle=':', linewidth=0.5, alpha=0.3, zorder=0)
    ax.set_axisbelow(True)
    ax.minorticks_on()
    ax.tick_params(axis='both', which='both', labelsize=14, direction='in')
    ax.tick_params(axis='x', which='minor', bottom=False)  # Disable minor ticks on x-axis
    ax.tick_params(axis='y', which='minor', left=True)
    
    ax.set_xlim(x_tick_pos_list[0]-0.5, x_tick_pos_list[-1]+0.5)
    ax.set_xlabel("MR", fontsize=14)
    ax.set_xticks(x_tick_pos_list)
    ax.set_xticklabels(x_tick_label_list, rotation=90, fontsize=14)
    
    y_max = 3
    ax.set_ylim(0, y_max)
    ax.set_ylabel("Absolute Error", fontsize=14)
    ax.set_yticks(np.arange(0, y_max + 0.1, 1))
    
    legend_handles = [
        Line2D([0], [0], marker=mean_marker, color='none', markeredgewidth=0, markerfacecolor="black", markersize=6, alpha=alpha, label="MAE"),
        Line2D([0], [0], marker=median_marker, color='none', markeredgewidth=0, markerfacecolor="black", markersize=6, alpha=alpha, label="MedAE"),
        mpatches.Patch(facecolor=vs_colors["orange"][0], alpha=alpha, edgecolor='none', label="MR Surr."),
        mpatches.Patch(facecolor=vs_colors["purple"][0], alpha=alpha, edgecolor='none', label="ML Pred."),
        mpatches.Patch(facecolor=vs_colors["green"][0], alpha=alpha, edgecolor='none', label="ML Tr. (\u03C9B97XD)"),
        mpatches.Patch(facecolor=vs_colors["blue"][0], alpha=alpha, edgecolor='none', label="ML Tr. (B2PLYPD3)")
    ]
    
    ax.legend(handles=legend_handles, loc="lower center", fontsize=5, ncols=3,
              handlelength=1, labelspacing=0.05, handletextpad=0.1, borderpad=0.1, borderaxespad=0.1, columnspacing=0.5) # , frameon=False)
    
    plt.tight_layout()
    plt.savefig(save_path + ".pdf", dpi=600, transparent=True, bbox_inches='tight', pad_inches=0.005)
    
    plt.close()
    
    return


def main():
    check_create_dirs()
    
    mr_ar_ea_dict = json.load(open(os.path.join(alk_pyrolysis_benchmark_dir, "mr_ar_ea_dict.json"), "r"))
    
    preds_dirs_list = [
        alk_pyrolysis_chemprop_delta_preds_dir,
        alk_pyrolysis_chemprop_delta_transfer_preds_dir_1,
        alk_pyrolysis_chemprop_delta_transfer_preds_dir_2
    ]
    
    out_csv_path_list = [os.path.join(alk_pyrolysis_analysis_dir, os.path.basename(preds_dir) + ".csv") for preds_dir in preds_dirs_list]
    
    # for preds_dir, out_csv_path in zip(preds_dirs_list, out_csv_path_list):
    #     csv_list = preds_dir_to_k_fold_mr_wise_accuracy_csv_list(preds_dir, mr_ar_ea_dict)
    #     with open(out_csv_path, "w") as f:
    #         f.write("\n".join(csv_list))
    #         f.close()
    
    ## manual mrs for plotting. ones where mr surrogate gives mae above 1.0 kcal/mol.
    mr_to_plot_list_0 = ["MR_2", "MR_5", "MR_7", "MR_11", "MR_12", "MR_13", "MR_19", "MR_21", "MR_25", "MR_26", "MR_29", "MR_31",
                       "MR_32", "MR_35", "MR_36", "MR_39", "MR_40", "MR_45", "MR_49", "MR_50", "MR_55", "MR_57", "MR_58"]
    mr_to_plot_list = [f"{mr_i}_0" for mr_i in mr_to_plot_list_0]
    
    # ## violin plot
    mr_surr_err_list_list = []
    ml_pred_err_list_list = []
    for mr_i in mr_to_plot_list:
        ar_list_i, ar_k_fold_abs_err_list_list_i, ar_abs_err_list_i, ar_mr_surr_abs_err_list = return_best_ar_mr_k_fold_errs_list_dict_from_accuracy_csv(alk_pyrolysis_chemprop_delta_preds_dir, mr_ar_ea_dict, mr_i)
        mr_surr_err_list_list.append(ar_mr_surr_abs_err_list)
        ml_pred_err_list_list.append(ar_abs_err_list_i)
    violin_plot(mr_to_plot_list_0, mr_surr_err_list_list, ml_pred_err_list_list, os.path.join(alk_pyrolysis_analysis_dir, "alk_pyrolysis_chemprop_delta_violin_plot"))
    
    mr_surr_mae_list = [np.mean(np.array(mr_surr_err_list)) for mr_surr_err_list in mr_surr_err_list_list]  ## will be same for all 3
    mr_surr_medae_list = [np.median(np.array(mr_surr_err_list)) for mr_surr_err_list in mr_surr_err_list_list]  ## will be same for all 3
    ml_pred_mae_list_0 = [np.mean(np.array(ml_pred_err_list)) for ml_pred_err_list in ml_pred_err_list_list]
    ml_pred_medae_list_0 = [np.median(np.array(ml_pred_err_list)) for ml_pred_err_list in ml_pred_err_list_list]
    
    ## violin plot wb97xd
    mr_surr_err_list_list = []
    ml_pred_err_list_list = []
    for mr_i in mr_to_plot_list:
        ar_list_i, ar_k_fold_abs_err_list_list_i, ar_abs_err_list_i, ar_mr_surr_abs_err_list = return_best_ar_mr_k_fold_errs_list_dict_from_accuracy_csv(alk_pyrolysis_chemprop_delta_transfer_preds_dir_1, mr_ar_ea_dict, mr_i)
        mr_surr_err_list_list.append(ar_mr_surr_abs_err_list)
        ml_pred_err_list_list.append(ar_abs_err_list_i)
    violin_plot(mr_to_plot_list_0, mr_surr_err_list_list, ml_pred_err_list_list, os.path.join(alk_pyrolysis_analysis_dir, "alk_pyrolysis_chemprop_delta_transfer_wb97xd_violin_plot"))
    
    mr_surr_mae_list = [np.mean(np.array(mr_surr_err_list)) for mr_surr_err_list in mr_surr_err_list_list]  ## will be same for all 3
    mr_surr_medae_list = [np.median(np.array(mr_surr_err_list)) for mr_surr_err_list in mr_surr_err_list_list]  ## will be same for all 3
    ml_pred_mae_list_1 = [np.mean(np.array(ml_pred_err_list)) for ml_pred_err_list in ml_pred_err_list_list]
    ml_pred_medae_list_1 = [np.median(np.array(ml_pred_err_list)) for ml_pred_err_list in ml_pred_err_list_list]
    
    ## violin plot b2plypd3
    mr_surr_err_list_list = []
    ml_pred_err_list_list = []
    for mr_i in mr_to_plot_list:
        ar_list_i, ar_k_fold_abs_err_list_list_i, ar_abs_err_list_i, ar_mr_surr_abs_err_list = return_best_ar_mr_k_fold_errs_list_dict_from_accuracy_csv(alk_pyrolysis_chemprop_delta_transfer_preds_dir_2, mr_ar_ea_dict, mr_i)
        mr_surr_err_list_list.append(ar_mr_surr_abs_err_list)
        ml_pred_err_list_list.append(ar_abs_err_list_i)
    violin_plot(mr_to_plot_list_0, mr_surr_err_list_list, ml_pred_err_list_list, os.path.join(alk_pyrolysis_analysis_dir, "alk_pyrolysis_chemprop_delta_transfer_b2plypd3_violin_plot"))
    
    mr_surr_mae_list = [np.mean(np.array(mr_surr_err_list)) for mr_surr_err_list in mr_surr_err_list_list]  ## will be same for all 3
    mr_surr_medae_list = [np.median(np.array(mr_surr_err_list)) for mr_surr_err_list in mr_surr_err_list_list]  ## will be same for all 3
    ml_pred_mae_list_2 = [np.mean(np.array(ml_pred_err_list)) for ml_pred_err_list in ml_pred_err_list_list]
    ml_pred_medae_list_2 = [np.median(np.array(ml_pred_err_list)) for ml_pred_err_list in ml_pred_err_list_list]
    
    ## for comparison of transfer learning. where original delta mae is still above 1.0 kcal/mol.
    mr_compare_list = ["MR_11", "MR_19", "MR_29", "MR_31", "MR_32", "MR_40", "MR_49", "MR_55", "MR_58"]
    # mr_compare_list = mr_to_plot_list_0
    mr_surr_mae_list = [mr_surr_mae_list[i] for i in range(len(mr_to_plot_list_0)) if mr_to_plot_list_0[i] in mr_compare_list]
    mr_surr_medae_list = [mr_surr_medae_list[i] for i in range(len(mr_to_plot_list_0)) if mr_to_plot_list_0[i] in mr_compare_list]
    ml_pred_mae_list_0 = [ml_pred_mae_list_0[i] for i in range(len(mr_to_plot_list_0)) if mr_to_plot_list_0[i] in mr_compare_list]
    ml_pred_medae_list_0 = [ml_pred_medae_list_0[i] for i in range(len(mr_to_plot_list_0)) if mr_to_plot_list_0[i] in mr_compare_list]
    ml_pred_mae_list_1 = [ml_pred_mae_list_1[i] for i in range(len(mr_to_plot_list_0)) if mr_to_plot_list_0[i] in mr_compare_list]
    ml_pred_medae_list_1 = [ml_pred_medae_list_1[i] for i in range(len(mr_to_plot_list_0)) if mr_to_plot_list_0[i] in mr_compare_list]
    ml_pred_mae_list_2 = [ml_pred_mae_list_2[i] for i in range(len(mr_to_plot_list_0)) if mr_to_plot_list_0[i] in mr_compare_list]
    ml_pred_medae_list_2 = [ml_pred_medae_list_2[i] for i in range(len(mr_to_plot_list_0)) if mr_to_plot_list_0[i] in mr_compare_list]
    
    transfer_compare_scatter_plot(mr_compare_list, mr_surr_mae_list, mr_surr_medae_list, ml_pred_mae_list_0, ml_pred_medae_list_0, ml_pred_mae_list_1, ml_pred_medae_list_1, ml_pred_mae_list_2, ml_pred_medae_list_2, os.path.join(alk_pyrolysis_analysis_dir, "alk_pyrolysis_transfer_compare_scatter_plot"))
    
    return


if __name__ == "__main__":
    main()
