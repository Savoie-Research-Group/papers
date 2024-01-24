import os
import pickle
import json
import numpy as np
from matplotlib import pyplot as plt


this_script_directory = os.path.dirname(os.path.realpath(__file__))
os.chdir(this_script_directory)


vs_colors = {"blue": ["#002E6B", [0, 46, 107]], "red": ["#A11D4B", [161, 29, 75]], "green": ["#155914", [21, 89, 20]], "orange": ["#D15019", [209, 80, 25]], "purple": ["#3C063D", [60, 6, 61]]}


def main():
    smi_hl_nn_dict_path = "../pretrained_models/mlp_models/till_c16_train_c17_test/alk_smi_pred_all_norm.json"
    with open(smi_hl_nn_dict_path) as f:
        smi_hl_nn_dict = json.load(f)
        f.close()
        
    smi_hl_chemprop_dict_path = "../pretrained_models/chemprop_models/till_c16_train_c17_test/alk_smi_pred_all_norm.json"
    with open(smi_hl_chemprop_dict_path) as f:
        smi_hl_chemprop_dict = json.load(f)
        f.close()
        
    with open("../data/genetic_mlp_run_all.p", "rb") as f:
        nn_gen_rn_all_list = pickle.load(f)
        f.close()
        
    with open("../data/genetic_chemprop_run_all.p", "rb") as f:
        chemprop_gen_rn_all_list = pickle.load(f)
        f.close()

    
    nn_best_alks_list, nn_worst_alks_list = [[j[0] for j in i] for i in nn_gen_rn_all_list], [[j[-1] for j in i] for i in nn_gen_rn_all_list]
    chemprop_best_alks_list, chemprop_worst_alks_list = [[j[0] for j in i] for i in chemprop_gen_rn_all_list], [[j[-1] for j in i] for i in chemprop_gen_rn_all_list]
    
    
    nn_best_stab_score_list, nn_worst_stab_score_list = [[smi_hl_nn_dict[j] for j in i] for i in nn_best_alks_list], [[smi_hl_nn_dict[j] for j in i] for i in nn_worst_alks_list]
    chemprop_best_stab_score_list, chemprop_worst_stab_score_list = [[smi_hl_chemprop_dict[j] for j in i] for i in chemprop_best_alks_list], [[smi_hl_chemprop_dict[j] for j in i] for i in chemprop_worst_alks_list]
    
    
    [nn_best_0, nn_best_25, nn_best_50, nn_best_75, nn_best_100] = np.quantile(nn_best_stab_score_list, [0, 0.25, 0.5, 0.75, 1], axis=0)
    [nn_worst_0, nn_worst_25, nn_worst_50, nn_worst_75, nn_worst_100] = np.quantile(nn_worst_stab_score_list, [0, 0.25, 0.5, 0.75, 1], axis=0)
    
    
    [chemprop_best_0, chemprop_best_25, chemprop_best_50, chemprop_best_75, chemprop_best_100] = np.quantile(chemprop_best_stab_score_list, [0, 0.25, 0.5, 0.75, 1], axis=0)
    [chemprop_worst_0, chemprop_worst_25, chemprop_worst_50, chemprop_worst_75, chemprop_worst_100] = np.quantile(chemprop_worst_stab_score_list, [0, 0.25, 0.5, 0.75, 1], axis=0)
    
    
    gens_list = list(range(len(nn_best_stab_score_list[0])))
    
    
    plt.clf()
    plt.rc('axes', axisbelow=True, labelsize=14)
    plt.rc('xtick', labelsize=14) 
    plt.rc('ytick', labelsize=14)
    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot()
    ax.set_box_aspect(0.5)
    # ax.set_yscale("log")
    plt.grid(which="both", linewidth=0.5)
    
    plt.fill_between(gens_list, nn_best_0, nn_best_100, color=vs_colors["blue"][0], alpha=0.5, edgecolor=None)
    plt.fill_between(gens_list, chemprop_best_0, chemprop_best_100, color=vs_colors["red"][0], alpha=0.5, edgecolor=None)
    plt.plot(gens_list, nn_best_50, color=vs_colors["blue"][0], linestyle="-", label="MLP Best")
    plt.plot(gens_list, chemprop_best_50, color=vs_colors["red"][0], linestyle="-", label="Chemprop Best")
    
    plt.fill_between(gens_list, nn_worst_0, nn_worst_100, color=vs_colors["blue"][0], alpha=0.5, edgecolor=None)
    plt.fill_between(gens_list, chemprop_worst_0, chemprop_worst_100, color=vs_colors["red"][0], alpha=0.5, edgecolor=None)
    plt.plot(gens_list, nn_worst_50, color=vs_colors["blue"][0], linestyle="--", label="MLP Worst")
    plt.plot(gens_list, chemprop_worst_50, color=vs_colors["red"][0], linestyle="--", label="Chemprop Worst")
    
    plt.legend(fancybox=False, framealpha=1, edgecolor="black", fontsize=12, loc='lower right')
    # plt.xticks(list(range(0, 61, 10)))
    plt.xlim([0, 60])
    plt.ylim([0, 120])
    plt.yticks(list(range(0, 101, 20)))
    
    plt.tick_params(which="both", direction="in")
    plt.xlabel('Generation')
    plt.ylabel('Scales Stability Scores')
    # plt.title("Median Ground Truth Half Lives of Moleculs Generated via different models")
    plt.tight_layout()
    plt.savefig("../data/genetic_c17_run.pdf", bbox_inches='tight', dpi=600)


if __name__ == "__main__":
    main()
