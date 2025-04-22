"""
    Date Modified: 2025/22/04
    Author: Veerupaksh (Veeru) Singla (singla2@purdue.edu)
    Description: Create umap using differential reaction fingerprint (drfp) (https://doi.org/10.1039/D1DD00006C).
"""

import os
import json
import numpy as np
import umap

this_script_dir = os.path.dirname(os.path.realpath(__file__))
os.chdir(this_script_dir)


from utils import *


def make_umap_plot_mr_ar(mr_umap_list, ar_umap_list, save_name):
    plt.clf()
    fig, ax = plt.subplots(figsize=(3.75, 3.75))
    ax.set_aspect(1)
    
    ax.scatter(ar_umap_list[:, 0], ar_umap_list[:, 1], c=vs_colors["red"][0], s=4, alpha=0.25, label="AR")
    ax.scatter(mr_umap_list[:, 0], mr_umap_list[:, 1], c=vs_colors["blue"][0], s=4, alpha=0.35, label="MR")
    
    ax.grid(which='major', color='gray', linestyle='-', linewidth=0.5, alpha=0.5, zorder=0)
    ax.grid(which='minor', axis='both', color='gray', linestyle=':', linewidth=0.5, alpha=0.3, zorder=0)
    ax.set_axisbelow(True)
    ax.minorticks_on()
    
    ax.set_xlabel("UMAP 1", fontsize=14)
    ax.set_ylabel("UMAP 2", fontsize=14)
    # ax.set_title("UMAP of MRs and ARs", fontsize=14)
    
    ax.tick_params(axis='both', which='both', labelsize=14, direction='in')
    
    legend_element1 = mlines.Line2D([], [], color=vs_colors["blue"][0], marker='o', linestyle='None', alpha=0.5,
                                    markersize=6, label="MR")
    legend_element2 = mlines.Line2D([], [], color=vs_colors["red"][0], marker='o', linestyle='None', alpha=0.5,
                                    markersize=6, label="AR")
    
    ax.legend(handles=[legend_element1, legend_element2], loc='upper left', fontsize=14, borderaxespad=0.25)
    
    plt.tight_layout()
    # plt.savefig(transparent=True, fname=save_name+".png", dpi=300, bbox_inches='tight')
    plt.savefig(transparent=True, fname=save_name+".pdf", dpi=300, bbox_inches='tight')
    plt.savefig(transparent=True, fname=save_name+".svg", dpi=300, bbox_inches='tight')
    
    return


def main():
    mr_smi_dict = json.load(open(os.path.join(data_path_main, "mr_smi_dict.json")))
    ar_smi_dict = json.load(open(os.path.join(data_path_main, "ar_smi_dict.json")))
    
    mr_smi_list = list(mr_smi_dict.values())
    ar_smi_list = list(ar_smi_dict.values())
    
    full_smi_list = mr_smi_list + ar_smi_list
    full_drfp_list = get_rxn_drfp(full_smi_list)
    full_drfp_list = np.array(full_drfp_list)
    umap_model = umap.UMAP(n_neighbors=50, min_dist=0.5, metric="jaccard", random_state=42, n_jobs=-1)
    full_drfp_list_umap = umap_model.fit_transform(full_drfp_list)
    full_smi_umap_dict = {}
    for i, smi in enumerate(full_smi_list):
        full_smi_umap_dict[smi] = full_drfp_list_umap[i].tolist()
    json.dump(full_smi_umap_dict, open(os.path.join(analyses_path_main, "full_smi_umap_dict.json"), "w"))
    
    full_smi_umap_dict = json.load(open(os.path.join(analyses_path_main, "full_smi_umap_dict.json")))
    
    mr_umap_list = np.array([full_smi_umap_dict[smi] for smi in mr_smi_list])
    ar_umap_list = np.array([full_smi_umap_dict[smi] for smi in ar_smi_list])
    
    save_name = os.path.join(analyses_path_main, "umap")
    make_umap_plot_mr_ar(mr_umap_list, ar_umap_list, save_name)
    
    return


if __name__ == "__main__":
    main()
