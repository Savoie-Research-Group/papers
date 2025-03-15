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


this_script_dir = os.path.dirname(os.path.realpath(__file__))
os.chdir(this_script_dir)


vs_colors = {"blue": ["#002E6B", [0, 46, 107]], "red": ["#A11D4B", [161, 29, 75]],
            "green": ["#155914", [21, 89, 20]], "orange": ["#D15019", [209, 80, 25]],
            "purple": ["#3C063D", [60, 6, 61]]}

## normalize vs_colors_rgb
for color in vs_colors:
    vs_colors[color][1] = [i/255 for i in vs_colors[color][1]]

data_path = os.path.join(this_script_dir, "../../data/expt_small_molecule_decomp_temp_data")
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


def plot_ha_decomp_temp_distribution(smi_ha_dict, smi_decomp_temp_dict, plots_path):
    import numpy as np
    from scipy import stats
    
    ha_list = list(smi_ha_dict.values())
    min_ha = round_down_to_base(min(ha_list), 5)
    max_ha = round_up_to_base(max(ha_list), 5)
    
    decomp_temp_list = list(smi_decomp_temp_dict.values())
    min_decomp_temp = int(round_down_to_base(min(decomp_temp_list), 50))
    max_decomp_temp = int(round_up_to_base(max(decomp_temp_list), 50))
    
    plt.clf()
    fig = plt.figure(figsize=(3.25, 7))
    gs = fig.add_gridspec(2, 1, hspace=0.21, wspace=0)
    (ax1, ax2) = gs.subplots(sharex=False, sharey=False)
    
    # Calculate KDEs
    ha_kde = stats.gaussian_kde(ha_list)
    temp_kde = stats.gaussian_kde(decomp_temp_list)
    
    x_ha = np.linspace(min_ha, max_ha, 200)
    x_temp = np.linspace(min_decomp_temp, max_decomp_temp, 200)
    
    kde_ha = ha_kde(x_ha)
    kde_temp = temp_kde(x_temp)
    
    n1, bins1, patches1 = ax1.hist(ha_list,
                                bins=range(min_ha, max_ha+1, 5),
                                edgecolor="none",
                                linewidth=0.0,
                                facecolor="none"
    )
    kde_ha = kde_ha / kde_ha.max() * max(n1)
    ax1.fill_between(x_ha, 0.0, 0.0 + kde_ha,
                    alpha=0.5, color=vs_colors["green"][0])
    ax1.vlines(ha_list, -kde_ha.max()/10, -kde_ha.max()/8 + kde_ha.max()/10,
            color=vs_colors["green"][0], alpha=0.25, linewidth=0.5)
    bp = ax1.boxplot(ha_list, positions=[-kde_ha.max()/16], vert=False,
                    widths=kde_ha.max()/5 - kde_ha.max()/8,
                    patch_artist=True, showfliers=False,
                    medianprops=dict(linestyle='-', color='black', linewidth=1.0))
    for patch in bp['boxes']:
        patch.set_facecolor(vs_colors["green"][1] + [0.0])
        patch.set_edgecolor("black")
    ax1.hist(ha_list,
            bins=bins1,
            edgecolor="black",
            linewidth=1.0,
            facecolor="none"
    )
    
    ax1.set_xlabel("Heavy Atoms", fontsize=14)
    ax1.set_xlim(0, max_ha)
    ax1.set_xticks(range(0, max_ha+1, 10))
    ax1.xaxis.set_label_coords(0.5, -0.105)
    
    ax1.set_ylim(-kde_ha.max()/7, int(round_up_to_base(max(n1)+1, 100)))
    ax1.set_yticks(range(0, int(round_up_to_base(max(n1)+1, 250)), 250))
    ax1.set_yticklabels([str(i) for i in range(0, int(round_up_to_base(max(n1)+1, 250)), 250)])
    
    # Plot Decomposition Temperature (bottom subplot)
    n2, bins2, patches2 = ax2.hist(decomp_temp_list,
                                bins=range(min_decomp_temp, max_decomp_temp+1, 50),
                                edgecolor="none",
                                linewidth=0.0,
                                facecolor="none"
    )
    kde_temp = kde_temp / kde_temp.max() * max(n2)
    ax2.fill_between(x_temp, 0.0, 0.0 + kde_temp,
                    alpha=0.5, color=vs_colors["orange"][0])
    
    ax2.vlines(decomp_temp_list, -kde_temp.max()/10, -kde_temp.max()/8 + kde_temp.max()/10,
            color=vs_colors["orange"][0], alpha=0.1, linewidth=0.5)
    bp = ax2.boxplot(decomp_temp_list, positions=[-kde_temp.max()/16], vert=False,
                    widths=kde_temp.max()/5 - kde_temp.max()/8,
                    patch_artist=True, showfliers=False,
                    medianprops=dict(linestyle='-', color='black', linewidth=1.0))
    ax2.set_ylim(-kde_temp.max()/7, int(round_up_to_base(max(n2)+1, 250)))
    
    for patch in bp['boxes']:
        patch.set_facecolor(vs_colors["orange"][1] + [0.0])
        patch.set_edgecolor("black")
    ax2.hist(decomp_temp_list,
            bins=bins2,
            edgecolor="black",
            linewidth=1.0,
            facecolor="none"
    )
    
    ax2.set_xlabel("Decomp Temp (C)", fontsize=14)
    ax2.set_xlim(0, max_decomp_temp)
    ax2.set_xticks(range(0, max_decomp_temp+1, 100))
    ax2.xaxis.set_label_coords(0.5, -0.105)
    
    ax2.set_yticks(range(0, int(round_up_to_base(max(n2)+1, 500)), 500))
    ax2.set_yticklabels([str(i) for i in range(0, int(round_up_to_base(max(n2)+1, 500)), 500)])
    ax2.set_ylabel("Number of Molecules", fontsize=14)
    ax2.yaxis.set_label_coords(-0.225, 1.1)
    
    for ax in (ax1, ax2):
        ax.grid(which='major', axis='y', color='gray', linestyle='-', linewidth=0.5, alpha=0.5, zorder=0)
        ax.grid(which='minor', axis='y', color='gray', linestyle=':', linewidth=0.5, alpha=0.3, zorder=0)
        ax.set_axisbelow(True)
        
        ax.minorticks_on()
        ax.xaxis.set_minor_locator(plt.NullLocator())
        
        ax.tick_params(axis='both', which='both', labelsize=14, direction='in')
    
    plt.tight_layout()
    plt.savefig(transparent=True, dpi=600, bbox_inches='tight',
                fname=os.path.join(plots_path, "expt_small_molec_distributions.pdf"),
                pad_inches=0.005)
    # plt.savefig(transparent=True, dpi=600, bbox_inches='tight',
    #             fname=os.path.join(plots_path, "expt_small_molec_distributions.svg"),
    #             pad_inches=0.005)
    # plt.savefig(transparent=True, dpi=600, bbox_inches='tight',
    #             fname=os.path.join(plots_path, "expt_small_molec_distributions.png"),
    #             pad_inches=0.005)
    plt.close()


def main():
    smi_expt_decomp_temp_dict_chon_f_cl_path = os.path.join(data_path, "smi_expt_decomp_temp_dict_chon_f_cl.json")
    smi_decomp_temp_dict = json.load(open(smi_expt_decomp_temp_dict_chon_f_cl_path, "r"))
    
    smi_ha_dict = {}
    for smi in smi_decomp_temp_dict:
        smi_lc = smi.lower()
        
        cl = smi_lc.count("cl")
        f = smi_lc.count("f")
        o = smi_lc.count("o")
        n = smi_lc.count("n")
        c = smi_lc.count("c") - cl
        
        ha = c + o + n + f + cl
        
        smi_ha_dict[smi] = ha
    
    plot_ha_decomp_temp_distribution(smi_ha_dict, smi_decomp_temp_dict, plots_path)
    
    return


if __name__ == "__main__":
    main()
