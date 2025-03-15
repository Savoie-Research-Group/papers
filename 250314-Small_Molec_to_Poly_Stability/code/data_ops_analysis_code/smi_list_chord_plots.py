import os

import json

from rdkit import Chem
from rdkit.Chem import rdMolDescriptors

import pandas as pd

from tqdm import tqdm

from d3blocks import D3Blocks


this_script_dir = os.path.dirname(os.path.realpath(__file__))
os.chdir(this_script_dir)


def get_stats_smi_list(smi_list):
    # smi_list_canon = [Chem.MolToSmiles(Chem.MolFromSmiles(smi)) for smi in smi_list]
    smi_list_canon = smi_list
    
    node_stats_dict = {
        'H': 0,
        'F': 0,
        'Cl': 0,
        'O': 0,
        'N': 0,
        'SP3 C': 0,
        'SP2 C': 0,
        'SP C': 0,
        'Ring C': 0,
        'Aromatic C': 0,
    }
    
    connectivity_stats_dict = {
        'SP3 C to H': 0,
        'SP3 C to F': 0,
        'SP3 C to Cl': 0,
        'SP3 C to O': 0,
        'SP3 C to N': 0,
        'SP3 C to SP2 C': 0,
        'SP3 C to SP C': 0,
        'SP3 C to Ring C': 0,
        'SP3 C to Aromatic C': 0,
        
        'SP2 C to H': 0,
        'SP2 C to F': 0,
        'SP2 C to Cl': 0,
        'SP2 C to O': 0,
        'SP2 C to N': 0,
        'SP2 C to SP C': 0,
        'SP2 C to Ring C': 0,
        'SP2 C to Aromatic C': 0,
        
        'SP C to H': 0,
        'SP C to F': 0,
        'SP C to Cl': 0,
        'SP C to O': 0,
        'SP C to N': 0,
        'SP C to Ring C': 0,
        'SP C to Aromatic C': 0,
        
        'Ring C to H': 0,
        'Ring C to F': 0,
        'Ring C to Cl': 0,
        'Ring C to O': 0,
        'Ring C to N': 0,
        'Ring C to Aromatic C': 0,
        
        'Aromatic C to H': 0,
        'Aromatic C to F': 0,
        'Aromatic C to Cl': 0,
        'Aromatic C to O': 0,
        'Aromatic C to N': 0,
        
        'O to H': 0,
        'O to F': 0,
        'O to Cl': 0,
        'O to N': 0,
        
        'N to H': 0,
        'N to F': 0,
        'N to Cl': 0,
    }

    
    for smi in tqdm(smi_list_canon):
        mol = Chem.MolFromSmiles(smi)

        if mol is None:
            print(f"Invalid SMILES: {smi}")
            continue

        mol = Chem.AddHs(mol)

        if mol is None:
            print(f"Invalid SMILES: {smi}")
            continue

        atom_types = {
            'H': [], 'F': [], 'Cl': [], 'O': [], 'N': [],
            'SP3 C': [], 'SP2 C': [], 'SP C': [],
            'Ring C': [], 'Aromatic C': []
        }

        connectivity = {}

        for atom in mol.GetAtoms():
            idx = atom.GetIdx()
            symbol = atom.GetSymbol()

            neighbors = [neighbor.GetIdx() for neighbor in atom.GetNeighbors()]
            connectivity[idx] = neighbors

            if symbol in ['H', 'F', 'Cl', 'O', 'N']:
                atom_types[symbol].append(idx)

            if symbol == 'C':
                hyb = atom.GetHybridization()
                if hyb == Chem.rdchem.HybridizationType.SP3:
                    atom_types['SP3 C'].append(idx)
                elif hyb == Chem.rdchem.HybridizationType.SP2:
                    atom_types['SP2 C'].append(idx)
                elif hyb == Chem.rdchem.HybridizationType.SP:
                    atom_types['SP C'].append(idx)

                if atom.IsInRing():
                    atom_types['Ring C'].append(idx)
                if atom.GetIsAromatic():
                    atom_types['Aromatic C'].append(idx)

        atom_idx_to_atom_type_list_dict = {}
        for key, value in atom_types.items():
            node_stats_dict[key] += len(value)
            for idx in value:
                if idx not in atom_idx_to_atom_type_list_dict:
                    atom_idx_to_atom_type_list_dict[idx] = set()
                atom_idx_to_atom_type_list_dict[idx].add(key)
        atom_idx_to_atom_type_list_dict = {k: list(v) for k, v in atom_idx_to_atom_type_list_dict.items()}

        for idx, neighbors in connectivity.items():
            atom_type = atom_idx_to_atom_type_list_dict[idx]
            for neighbor in neighbors:
                neighbor_type = atom_idx_to_atom_type_list_dict[neighbor]
                
                for at in atom_type:
                    for nt in neighbor_type:
                        key = f"{at} to {nt}"
                        if key in connectivity_stats_dict:
                            connectivity_stats_dict[key] += 1

    return node_stats_dict, connectivity_stats_dict


def links_df_from_node_connectivity_stats(node_stats, connectivity_stats):
    node_percentages = {k: v/sum(node_stats.values()) * 100 for k, v in node_stats.items()}
    node_percentages = {k: round(v, 3) for k, v in node_percentages.items()}
    source_color_dict = {
                    "SP3 C": "#17becf",
                    "SP2 C": "#d62728",
                    "SP C": "#2ca02c",
                    "Ring C": "#ff7f0e",
                    "O": "#8c564b",
                    "N": "#9467bd",
                    "H": "#bcbd22",
                    "F": "#7f7f7f",
                    "Cl": "#e377c2",
                    "Aromatic C": "#1f77b4"
    }
    links_df = []
    # min_count_nonzero = min([count for count in connectivity_stats.values() if count > 0])
    # sum_count = sum([count for count in connectivity_stats.values()])
    for conn, count in connectivity_stats.items():
        if count == 0:
            continue
        source, target = conn.split(' to ')
        # if target == 'H':
        #     continue
        source = f"{source} ({node_percentages[source]})"
        target = f"{target} ({node_percentages[target]})"
        links_df.append({
            'source': source,
            'target': target,
            'weight': count,
            # 'weight': int(count/min_count_nonzero),
            # 'weight': count/sum_count,
        })
        
    links_df = pd.DataFrame(links_df)
    print(links_df)
    link_color_list = []
    for source in links_df['source']:
        link_color_list.append(source_color_dict[str(' '.join(source.split(' ')[:-1]))])
    ## add link color to links_df as a column
    links_df['color'] = link_color_list
    print(links_df)
    return links_df, link_color_list


def plot_chord_d3blocks(links_df, plot_path, opacity=0.5, cmap='tab10_r', fontsize=18, arrowhead=0, title=None, figsize=[1000, 1000], conn_color_list=None):
    tab10 = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    tab10_r = ['#17becf', '#bcbd22', '#7f7f7f', '#e377c2', '#8c564b', '#9467bd', '#d62728', '#2ca02c', '#ff7f0e', '#1f77b4']
    print(conn_color_list)
    if title is None:
        title = plot_path.split('/')[-1].split('.')[0]
    d3 = D3Blocks()
    if conn_color_list is not None:
        d3.chord(links_df, filepath=plot_path,
            opacity=opacity,
            fontsize=fontsize,
            arrowhead=arrowhead,
            title=title,
            figsize=figsize)
    else:
        d3.chord(links_df, filepath=plot_path,
                opacity=opacity,
                cmap=cmap,
                fontsize=fontsize,
                arrowhead=arrowhead,
                title=title,
                figsize=figsize)
    return


def chord_plot_rmg_active_learning_data():
    plot_path = os.path.join(this_script_dir, "../../data/plots/chord_plot_rmg_active_learning_data.html")
    rmg_active_smi_list_path = os.path.join(this_script_dir, "../../data/rmg_active_learning_data/sampled_smi_list.txt")
    rmg_active_smi_list = []
    
    with open(rmg_active_smi_list_path, 'r') as f:
        for line in f:
            rmg_active_smi_list.append(line.strip())
    
    node_stats, connectivity_stats = get_stats_smi_list(rmg_active_smi_list)
    links_df, link_color_list = links_df_from_node_connectivity_stats(node_stats, connectivity_stats)
    plot_chord_d3blocks(links_df, plot_path, conn_color_list=link_color_list)
    return


def chord_plot_expt_small_molec_data():
    plot_path = os.path.join(this_script_dir, "../../data/plots/chord_plot_expt_small_molec_data.html")
    expt_small_molec_smi_temp_dict_path = os.path.join(this_script_dir, "../../data/expt_small_molecule_decomp_temp_data/smi_expt_decomp_temp_dict_chon_f_cl.json")
    expt_small_molec_smi_temp_dict = json.load(open(expt_small_molec_smi_temp_dict_path, 'r'))
    expt_small_molec_smi_list = list(expt_small_molec_smi_temp_dict.keys())
    
    node_stats, connectivity_stats = get_stats_smi_list(expt_small_molec_smi_list)
    links_df, _ = links_df_from_node_connectivity_stats(node_stats, connectivity_stats)
    plot_chord_d3blocks(links_df, plot_path)
    return


def main():
    chord_plot_rmg_active_learning_data()
    chord_plot_expt_small_molec_data()
    return


if __name__ == "__main__":
    main()
