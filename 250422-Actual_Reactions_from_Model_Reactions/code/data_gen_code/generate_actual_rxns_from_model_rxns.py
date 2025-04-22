"""
    Date: 2025/22/04
    Author: Veeru Singla (singla2@purdue.edu)
    Description: takes the mr xyz files and the dicts of functional groups to attach and generates the actual reactions.
"""

import os
import argparse
import numpy as np
import scipy as sp
import sys
from rdkit import Chem
import json
from tqdm import tqdm
import subprocess
from copy import deepcopy
import random
import shutil
from tqdm import tqdm
import multiprocessing as mp


this_script_directory = os.path.dirname(os.path.realpath(__file__))
os.chdir(this_script_directory)

from data_gen_utils import *


def gen_one_rxn(E_mr_in, RG_mr_in, adj_mat_1_mr_in, adj_mat_2_mr_in, bond_mat_1_mr_in, bond_mat_2_mr_in, atoms_to_remove_list_in, atoms_to_bond_to_list_in, fn_grps_to_add_dicts_list_in, rxn_xyz_write=False, rxn_xyz_path=None):
    assert len(atoms_to_bond_to_list_in) == len(fn_grps_to_add_dicts_list_in)
    
    ## no deepcopy needed for now since all used functions already do deepcopy.
    ## might need to do in future.
    
    # prepare mr for addition of fn group(s)
    old_to_new_atom_index_dict_mr, E_mr_1, RG_mr_1, PG_mr_1, adj_mat_1_mr_1, adj_mat_2_mr_1, bond_mat_1_mr_1, bond_mat_2_mr_1 = remove_atoms_from_mr(E_mr_in, RG_mr_in, RG_mr_in, adj_mat_1_mr_in, adj_mat_2_mr_in, bond_mat_1_mr_in, bond_mat_2_mr_in, atoms_to_remove_list_in=atoms_to_remove_list_in)
    
    # update atom numbers of modified mr
    atoms_to_bond_to_list = [old_to_new_atom_index_dict_mr[i] for i in atoms_to_bond_to_list_in]
    
    # add fn groups to the modified mr
    E_mr_2, RG_mr_2, PG_mr_2, adj_mat_1_mr_2, adj_mat_2_mr_2, bond_mat_1_mr_2, bond_mat_2_mr_2 = add_fn_grps_to_modified_mr(E_mr_1, RG_mr_1, PG_mr_1, adj_mat_1_mr_1, adj_mat_2_mr_1, bond_mat_1_mr_1, bond_mat_2_mr_1, atoms_to_bond_to_list=atoms_to_bond_to_list, fn_grps_to_add_dicts_list_in=fn_grps_to_add_dicts_list_in)
    
    # optimize new react and prod geos
    new_mol_1_temp = write_mol(E_mr_2, RG_mr_2, adj_mat_1_mr_2, bond_mat_1_mr_2, "temp_1.mol")
    opt_geo_mol(new_mol_1_temp)
    
    react_smi_obabel = mol_file_to_smi(new_mol_1_temp)
    try:
        react_smi = rdkit_kekulize_canonicalize(react_smi_obabel)
    except:
        return "", ""
    react_smi_atom_mapped = mol_file_to_atom_mapped_smi(new_mol_1_temp)
    
    new_mol_2_temp = write_mol(E_mr_2, PG_mr_2, adj_mat_2_mr_2, bond_mat_2_mr_2, "temp_2.mol")
    opt_geo_mol(new_mol_2_temp)
    prod_smi_obabel = mol_file_to_smi(new_mol_2_temp)
    try:
        prod_smi = rdkit_kekulize_canonicalize(prod_smi_obabel)
    except:
        return "", ""
    prod_smi_atom_mapped = mol_file_to_atom_mapped_smi(new_mol_2_temp)
    
    # parse new react and prod geos
    RE, RG, adj_mat_1, bond_mat_1, _ = parse_mol(new_mol_1_temp)
    PE, PG, adj_mat_2, bond_mat_2, _ = parse_mol(new_mol_2_temp)
    
    # remove temp files
    os.remove(new_mol_1_temp)
    os.remove(new_mol_2_temp)
    
    if RE != PE:
        print(RE, PE, rxn_xyz_path)
    
    assert RE == PE
    
    if rxn_xyz_write:
        
        if rxn_xyz_path is None:
            rxn_xyz_path = f"'./{random.randint(0, 10000000)}.xyz'"
            
        write_rxn_xyz(RE, RG, PG, rxn_xyz_path)
        
    return f"{react_smi}>>{prod_smi}", f"{react_smi_atom_mapped}>>{prod_smi_atom_mapped}"


# ## to handle polyamines, and polyols
# def handle_special_fn_groups():
    

def gen_rxns_for_one_mr(mr_xyz_path, fn_groups_ref_nums_to_smiles_dict, fn_groups_categories_to_ref_nums_dict, out_dir=None, scratch_dir=None, rxns_xyz_namespace=None):
    fn_grp_refs_needing_min_3_bonds = {"4_13", "5_12"}  # "[NH2].[NH2].[NH2]": Polyamine; "[OH].[OH].[OH]": Polyol
    fn_grp_refs_needing_min_2_bonds = {"4_12", "4_13", "5_11", "5_12"}  # "[NH2].[NH2]", "[NH2].[NH2].[NH2]": Polyamines; "[OH].[OH]", "[OH].[OH].[OH]": Polyols
    special_fn_grp_categories = ["amines_imines_nitriles", "acids_alcohols_aldehydes_esters_ethers_ketones_peroxides"]
    
    ## parse mr
    E_mr, RG_mr, PG_mr, adj_mat_1_mr, adj_mat_2_mr, bond_mat_1_mr, bond_mat_2_mr, primary_atoms_list_mr, first_neighbor_non_h_atoms_list_mr, first_neighbors_h_bonds_dict_mr = parse_mr_xyz(mr_xyz_path)
    
    num_fn_grps_categories = len(fn_groups_categories_to_ref_nums_dict)
    fn_grps_categories_list = list(fn_groups_categories_to_ref_nums_dict.keys())
    
    if out_dir is None or scratch_dir is None:
        dirs_num = random.randint(0, 10000000)
    
    if out_dir is None:
        out_dir = f"./rxns_{dirs_num}"
        ## if out_dir exists, delete it and create it again
        if os.path.exists(out_dir):
            shutil.rmtree(out_dir)
        os.mkdir(out_dir)
        out_dir = os.path.abspath(out_dir)
    else:
        os.makedirs(out_dir, exist_ok=True)
    
    if scratch_dir is None:
        scratch_dir = f"./.temp_{dirs_num}"
        ## if scratch_dir exists, delete it and create it again
        if os.path.exists(scratch_dir):
            shutil.rmtree(scratch_dir)
        os.mkdir(scratch_dir)
        scratch_dir = os.path.abspath(scratch_dir)
    else:
        os.makedirs(scratch_dir, exist_ok=True)
        
    if rxns_xyz_namespace is None:
        rxns_xyz_namespace = "rxn"
    
    current_dir = os.getcwd()
    os.chdir(scratch_dir)
    
    rxn_smi_to_xyz_name_dict = {}
    rxn_smi_to_atom_mapped_rxn_smi_dict = {}
    rxn_smi_to_fn_grps_added_dict = {}
    
    rxn_count = 0
    
    #  ## what if only 1 first neighbor is available with hydrogens to remove?
    #         # e.g. the other neighbor is a double bonded oxygen e.g. CC(=N)C=O>>CC(=C=O)N
    #         if len(first_neighbors_h_bonds_dict_mr) < 2:
    #             ## choose 1 first neighbor key from first_neighbors_h_bonds_dict_mr
    #             first_neighbor_atoms_list = random.sample(list(first_neighbors_h_bonds_dict_mr.keys()), 1)
    #             first_neighbor_atoms_bonded_h_atoms_list = [first_neighbors_h_bonds_dict_mr[first_neighbor_atom] for first_neighbor_atom in first_neighbor_atoms_list]
    
    # if no first neighbors are available with hydrogens to remove, then skip this mr.
    if len(first_neighbors_h_bonds_dict_mr) == 0:
        print(f"No first neighbors with hydrogens to remove. Skipping this mr {mr_xyz_path}")
    
    # if only one first neighbor is available with hydrogens to remove. then only add 1 fn group.
    # e.g. the other neighbor is a double bonded oxygen e.g. CC(=N)C=O>>CC(=C=O)N
    # go over fn group categories twice so that each fn group is represented twice.
    elif len(first_neighbors_h_bonds_dict_mr) == 1:
        for _, fn_grp_category_1 in tqdm(enumerate(fn_grps_categories_list + fn_grps_categories_list + fn_grps_categories_list + fn_grps_categories_list), total=4*num_fn_grps_categories):
            first_neighbor_atom = list(first_neighbors_h_bonds_dict_mr.keys())[0]
            first_neighbor_atoms_bonded_h_atoms = first_neighbors_h_bonds_dict_mr[first_neighbor_atom]
           
            if fn_grp_category_1 in special_fn_grp_categories:
                first_neighbor_atoms_bonded_h_atoms_len = len(first_neighbor_atoms_bonded_h_atoms)
                
                if first_neighbor_atoms_bonded_h_atoms_len == 3:
                    fn_grp_category_1_ref_nums_list = fn_groups_categories_to_ref_nums_dict[fn_grp_category_1]
                elif first_neighbor_atoms_bonded_h_atoms_len == 2:
                    fn_grp_category_1_ref_nums_list = list(set(fn_groups_categories_to_ref_nums_dict[fn_grp_category_1]) - fn_grp_refs_needing_min_3_bonds)
                elif first_neighbor_atoms_bonded_h_atoms_len == 1:
                    fn_grp_category_1_ref_nums_list = list(set(fn_groups_categories_to_ref_nums_dict[fn_grp_category_1]) - fn_grp_refs_needing_min_2_bonds)
                    
                fn_grp_1_ref_num = random.choice(fn_grp_category_1_ref_nums_list)
                fn_grp_1_smi = fn_groups_ref_nums_to_smiles_dict[fn_grp_1_ref_num]
                
                if fn_grp_1_ref_num in fn_grp_refs_needing_min_3_bonds:
                    atoms_to_remove_list_1 = first_neighbors_h_bonds_dict_mr[first_neighbor_atom]
                elif fn_grp_1_ref_num in fn_grp_refs_needing_min_2_bonds:
                    atoms_to_remove_list_1 = first_neighbors_h_bonds_dict_mr[first_neighbor_atom][:2]
                else:
                    atoms_to_remove_list_1 = first_neighbors_h_bonds_dict_mr[first_neighbor_atom][:1]
                    
            else:
                fn_grp_1_ref_num = random.choice(fn_groups_categories_to_ref_nums_dict[fn_grp_category_1])
                fn_grp_1_smi = fn_groups_ref_nums_to_smiles_dict[fn_grp_1_ref_num]
                atoms_to_remove_list_1 = first_neighbors_h_bonds_dict_mr[first_neighbor_atom][:1]
                
            atoms_to_remove_list = atoms_to_remove_list_1
            atoms_to_bond_to_list = [first_neighbor_atom]
            fn_grps_to_add_dicts_list = [parse_fn_grp_smi(fn_grp_1_smi)[1]]
            
            rxn_xyz_name = f"{rxns_xyz_namespace}_{rxn_count}_{fn_grp_1_ref_num}.xyz"
            rxn_xyz_path = os.path.join(out_dir, rxn_xyz_name)
            
            rxn_smi, rxn_smi_atom_mapped = gen_one_rxn(E_mr, RG_mr, adj_mat_1_mr, adj_mat_2_mr, bond_mat_1_mr, bond_mat_2_mr, atoms_to_remove_list, atoms_to_bond_to_list, fn_grps_to_add_dicts_list, rxn_xyz_write=True, rxn_xyz_path=rxn_xyz_path)
            
            if "+" in rxn_smi or "-" in rxn_smi:  ## removing charged/ionic species
                os.system(f"rm {rxn_xyz_path}")
            else:
                rxn_smi_to_fn_grps_added_dict[rxn_smi] = [fn_grp_1_ref_num]
                rxn_smi_to_xyz_name_dict[rxn_smi] = rxn_xyz_name
                rxn_smi_to_atom_mapped_rxn_smi_dict[rxn_smi] = rxn_smi_atom_mapped
                
                rxn_count += 1

    # when multiple first neighbors are available with hydrogens to remove, then add 2 fn groups.
    else:
        ## allowing repetions of the same category of fn group
        for i, fn_grp_category_1 in tqdm(enumerate(fn_grps_categories_list), total=num_fn_grps_categories):
            for _, fn_grp_category_2 in tqdm(enumerate(fn_grps_categories_list[i:] + fn_grps_categories_list[i:]), total=2*(num_fn_grps_categories-i)):
                
                ## choose 2 first neighbor keys from first_neighbors_h_bonds_dict_mr
                first_neighbor_atoms_list = random.sample(list(first_neighbors_h_bonds_dict_mr.keys()), 2)
                first_neighbor_atoms_bonded_h_atoms_list = [first_neighbors_h_bonds_dict_mr[first_neighbor_atom] for first_neighbor_atom in first_neighbor_atoms_list]
                
                if fn_grp_category_1 in special_fn_grp_categories or fn_grp_category_2 in special_fn_grp_categories:
                    ## max possible bonds possible on any of the first neighbor atoms. needed to determine polyamine and polyol cases.
                    
                    first_neighbor_atoms_bonded_h_atoms_len_list = [len(first_neighbor_atoms_bonded_h_atoms) for first_neighbor_atoms_bonded_h_atoms in first_neighbor_atoms_bonded_h_atoms_list]
                    max_bonds_possible_on_any_sampled_first_neighbor_atom = max(first_neighbor_atoms_bonded_h_atoms_len_list)
                    min_bonds_possible_on_any_sampled_first_neighbor_atom = min(first_neighbor_atoms_bonded_h_atoms_len_list)
                
                    if fn_grp_category_1 in special_fn_grp_categories:
                        if max_bonds_possible_on_any_sampled_first_neighbor_atom == 3:
                            fn_grp_category_1_ref_nums_list = fn_groups_categories_to_ref_nums_dict[fn_grp_category_1]
                        elif max_bonds_possible_on_any_sampled_first_neighbor_atom == 2:
                            fn_grp_category_1_ref_nums_list = list(set(fn_groups_categories_to_ref_nums_dict[fn_grp_category_1]) - fn_grp_refs_needing_min_3_bonds)
                        elif max_bonds_possible_on_any_sampled_first_neighbor_atom == 1:
                            fn_grp_category_1_ref_nums_list = list(set(fn_groups_categories_to_ref_nums_dict[fn_grp_category_1]) - fn_grp_refs_needing_min_2_bonds)
                            
                        fn_grp_1_ref_num = random.choice(fn_grp_category_1_ref_nums_list)
                        fn_grp_1_smi = fn_groups_ref_nums_to_smiles_dict[fn_grp_1_ref_num]
                        
                        if first_neighbor_atoms_bonded_h_atoms_len_list[0] == max_bonds_possible_on_any_sampled_first_neighbor_atom:
                            first_neighbor_atom_1 = first_neighbor_atoms_list[0]
                            first_neighbor_atom_2 = first_neighbor_atoms_list[1]
                        else:
                            first_neighbor_atom_1 = first_neighbor_atoms_list[1]
                            first_neighbor_atom_2 = first_neighbor_atoms_list[0]
                            
                        if fn_grp_1_ref_num in fn_grp_refs_needing_min_3_bonds:
                            atoms_to_remove_list_1 = first_neighbors_h_bonds_dict_mr[first_neighbor_atom_1]
                        elif fn_grp_1_ref_num in fn_grp_refs_needing_min_2_bonds:
                            atoms_to_remove_list_1 = first_neighbors_h_bonds_dict_mr[first_neighbor_atom_1][:2]
                        else:
                            atoms_to_remove_list_1 = first_neighbors_h_bonds_dict_mr[first_neighbor_atom_1][:1]
                        
                        ## what if both fn groups have polyamines or polyols?
                        if fn_grp_category_1 in special_fn_grp_categories and fn_grp_category_2 in special_fn_grp_categories:
                            if min_bonds_possible_on_any_sampled_first_neighbor_atom == 3:
                                fn_group_category_2_ref_nums_list = fn_groups_categories_to_ref_nums_dict[fn_grp_category_2]
                            elif min_bonds_possible_on_any_sampled_first_neighbor_atom == 2:
                                fn_group_category_2_ref_nums_list = list(set(fn_groups_categories_to_ref_nums_dict[fn_grp_category_2]) - fn_grp_refs_needing_min_3_bonds)
                            elif min_bonds_possible_on_any_sampled_first_neighbor_atom == 1:
                                fn_group_category_2_ref_nums_list = list(set(fn_groups_categories_to_ref_nums_dict[fn_grp_category_2]) - fn_grp_refs_needing_min_2_bonds)
                                
                            fn_grp_2_ref_num = random.choice(fn_group_category_2_ref_nums_list)
                            fn_grp_2_smi = fn_groups_ref_nums_to_smiles_dict[fn_grp_2_ref_num]
                            
                            if fn_grp_2_ref_num in fn_grp_refs_needing_min_3_bonds:
                                atoms_to_remove_list_2 = first_neighbors_h_bonds_dict_mr[first_neighbor_atom_2]
                            elif fn_grp_2_ref_num in fn_grp_refs_needing_min_2_bonds:
                                atoms_to_remove_list_2 = first_neighbors_h_bonds_dict_mr[first_neighbor_atom_2][:2]
                            else:
                                atoms_to_remove_list_2 = first_neighbors_h_bonds_dict_mr[first_neighbor_atom_2][:1]
                        else:
                            fn_grp_2_ref_num = random.choice(fn_groups_categories_to_ref_nums_dict[fn_grp_category_2])
                            fn_grp_2_smi = fn_groups_ref_nums_to_smiles_dict[fn_grp_2_ref_num]
                            atoms_to_remove_list_2 = first_neighbors_h_bonds_dict_mr[first_neighbor_atom_2][:1]

                    elif fn_grp_category_2 in special_fn_grp_categories:
                        if max_bonds_possible_on_any_sampled_first_neighbor_atom == 3:
                            fn_grp_category_2_ref_nums_list = fn_groups_categories_to_ref_nums_dict[fn_grp_category_2]
                        elif max_bonds_possible_on_any_sampled_first_neighbor_atom == 2:
                            fn_grp_category_2_ref_nums_list = list(set(fn_groups_categories_to_ref_nums_dict[fn_grp_category_2]) - fn_grp_refs_needing_min_3_bonds)
                        elif max_bonds_possible_on_any_sampled_first_neighbor_atom == 1:
                            fn_grp_category_2_ref_nums_list = list(set(fn_groups_categories_to_ref_nums_dict[fn_grp_category_2]) - fn_grp_refs_needing_min_2_bonds)
                            
                        fn_grp_2_ref_num = random.choice(fn_grp_category_2_ref_nums_list)
                        fn_grp_2_smi = fn_groups_ref_nums_to_smiles_dict[fn_grp_2_ref_num]
                        
                        if first_neighbor_atoms_bonded_h_atoms_len_list[0] == max_bonds_possible_on_any_sampled_first_neighbor_atom:
                            first_neighbor_atom_2 = first_neighbor_atoms_list[0]
                            first_neighbor_atom_1 = first_neighbor_atoms_list[1]
                        else:
                            first_neighbor_atom_2 = first_neighbor_atoms_list[1]
                            first_neighbor_atom_1 = first_neighbor_atoms_list[0]
                            
                        if fn_grp_2_ref_num in fn_grp_refs_needing_min_3_bonds:
                            atoms_to_remove_list_2 = first_neighbors_h_bonds_dict_mr[first_neighbor_atom_2]
                        elif fn_grp_2_ref_num in fn_grp_refs_needing_min_2_bonds:
                            atoms_to_remove_list_2 = first_neighbors_h_bonds_dict_mr[first_neighbor_atom_2][:2]
                        else:
                            atoms_to_remove_list_2 = first_neighbors_h_bonds_dict_mr[first_neighbor_atom_2][:1]
                            
                        ## what if both fn groups have polyamines or polyols?
                        if fn_grp_category_1 in special_fn_grp_categories and fn_grp_category_2 in special_fn_grp_categories:
                            if min_bonds_possible_on_any_sampled_first_neighbor_atom == 3:
                                fn_group_category_1_ref_nums_list = fn_groups_categories_to_ref_nums_dict[fn_grp_category_1]
                            elif min_bonds_possible_on_any_sampled_first_neighbor_atom == 2:
                                fn_group_category_1_ref_nums_list = list(set(fn_groups_categories_to_ref_nums_dict[fn_grp_category_1]) - fn_grp_refs_needing_min_3_bonds)
                            elif min_bonds_possible_on_any_sampled_first_neighbor_atom == 1:
                                fn_group_category_1_ref_nums_list = list(set(fn_groups_categories_to_ref_nums_dict[fn_grp_category_1]) - fn_grp_refs_needing_min_2_bonds)
                                
                            fn_grp_1_ref_num = random.choice(fn_group_category_1_ref_nums_list)
                            fn_grp_1_smi = fn_groups_ref_nums_to_smiles_dict[fn_grp_1_ref_num]
                            
                            if fn_grp_1_ref_num in fn_grp_refs_needing_min_3_bonds:
                                atoms_to_remove_list_1 = first_neighbors_h_bonds_dict_mr[first_neighbor_atom_1]
                            elif fn_grp_1_ref_num in fn_grp_refs_needing_min_2_bonds:
                                atoms_to_remove_list_1 = first_neighbors_h_bonds_dict_mr[first_neighbor_atom_1][:2]
                            else:
                                atoms_to_remove_list_1 = first_neighbors_h_bonds_dict_mr[first_neighbor_atom_1][:1]
                        else:
                            fn_grp_1_ref_num = random.choice(fn_groups_categories_to_ref_nums_dict[fn_grp_category_1])
                            fn_grp_1_smi = fn_groups_ref_nums_to_smiles_dict[fn_grp_1_ref_num]
                            atoms_to_remove_list_1 = first_neighbors_h_bonds_dict_mr[first_neighbor_atom_1][:1]
                            
                else:
                    first_neighbor_atom_1 = first_neighbor_atoms_list[0]
                    fn_grp_1_ref_num = random.choice(fn_groups_categories_to_ref_nums_dict[fn_grp_category_1])
                    fn_grp_1_smi = fn_groups_ref_nums_to_smiles_dict[fn_grp_1_ref_num]
                    atoms_to_remove_list_1 = first_neighbors_h_bonds_dict_mr[first_neighbor_atoms_list[0]][:1]
                    
                    first_neighbor_atom_2 = first_neighbor_atoms_list[1]
                    fn_grp_2_ref_num = random.choice(fn_groups_categories_to_ref_nums_dict[fn_grp_category_2])
                    fn_grp_2_smi = fn_groups_ref_nums_to_smiles_dict[fn_grp_2_ref_num]
                    atoms_to_remove_list_2 = first_neighbors_h_bonds_dict_mr[first_neighbor_atoms_list[1]][:1]
                    
                atoms_to_remove_list = atoms_to_remove_list_1 + atoms_to_remove_list_2
                atoms_to_bond_to_list = [first_neighbor_atom_1, first_neighbor_atom_2]
                fn_grps_to_add_dicts_list = [parse_fn_grp_smi(fn_grp_1_smi)[1], parse_fn_grp_smi(fn_grp_2_smi)[1]]
                
                rxn_xyz_name = f"{rxns_xyz_namespace}_{rxn_count}_{fn_grp_1_ref_num}_{fn_grp_2_ref_num}.xyz"
                rxn_xyz_path = os.path.join(out_dir, rxn_xyz_name)
                
                rxn_smi, rxn_smi_atom_mapped = gen_one_rxn(E_mr, RG_mr, adj_mat_1_mr, adj_mat_2_mr, bond_mat_1_mr, bond_mat_2_mr, atoms_to_remove_list, atoms_to_bond_to_list, fn_grps_to_add_dicts_list, rxn_xyz_write=True, rxn_xyz_path=rxn_xyz_path)
                
                if "+" in rxn_smi or "-" in rxn_smi:
                    os.system(f"rm {rxn_xyz_path}")
                else:
                    rxn_smi_to_fn_grps_added_dict[rxn_smi] = [fn_grp_1_ref_num, fn_grp_2_ref_num]
                    rxn_smi_to_xyz_name_dict[rxn_smi] = rxn_xyz_name
                    rxn_smi_to_atom_mapped_rxn_smi_dict[rxn_smi] = rxn_smi_atom_mapped
                    
                    rxn_count += 1
            
    os.chdir(current_dir)
    
    ## delete scratch_dir
    shutil.rmtree(scratch_dir)
    
    with open(os.path.join(out_dir, "rxn_smi_to_xyz_name_dict.json"), "w") as f:
        json.dump(rxn_smi_to_xyz_name_dict, f)
    with open(os.path.join(out_dir, "rxn_smi_to_atom_mapped_rxn_smi_dict.json"), "w") as f:
        json.dump(rxn_smi_to_atom_mapped_rxn_smi_dict, f)
    with open(os.path.join(out_dir, "rxn_smi_to_fn_grps_added_dict.json"), "w") as f:
        json.dump(rxn_smi_to_fn_grps_added_dict, f)
        
    return out_dir, rxn_smi_to_xyz_name_dict, rxn_smi_to_atom_mapped_rxn_smi_dict, rxn_smi_to_fn_grps_added_dict


def gen_rxns_multiple_mrs_multiprocess(mr_xyz_paths_list, fn_groups_ref_nums_to_smiles_dict, fn_groups_categories_to_ref_nums_dict, out_dir_list=[], scratch_dir_list=[], rxns_xyz_namespace_list=[], num_processes=len(os.sched_getaffinity(0))):
    assert len(out_dir_list) == len(scratch_dir_list) == len(rxns_xyz_namespace_list) == 0 or len(out_dir_list) == len(scratch_dir_list) == len(rxns_xyz_namespace_list) == len(mr_xyz_paths_list)
    
    if len(out_dir_list) == len(scratch_dir_list) == len(rxns_xyz_namespace_list) == 0:
        out_dir_list = [None] * len(mr_xyz_paths_list)
        scratch_dir_list = [None] * len(mr_xyz_paths_list)
        rxns_xyz_namespace_list = [None] * len(mr_xyz_paths_list)
    
    if len(out_dir_list) == len(scratch_dir_list) == len(rxns_xyz_namespace_list) == len(mr_xyz_paths_list):
        out_dir_list = [None if i == "" else i for i in out_dir_list]
        scratch_dir_list = [None if i == "" else i for i in scratch_dir_list]
        rxns_xyz_namespace_list = [None if i == "" else i for i in rxns_xyz_namespace_list]
    
    print(f"Generating reactions for {len(mr_xyz_paths_list)} model reactions.")
    print(f"Using {num_processes} processes.")
    
    with mp.Manager() as manager:
        ## using manager.dict() to share the dictionaries between processes and avoid race conditions. 
        # thought not useful here since dicts are not being updated. but still using for consistency.
        shared_fn_groups_ref_nums_to_smiles_dict = manager.dict(fn_groups_ref_nums_to_smiles_dict)
        shared_fn_groups_categories_to_ref_nums_dict = manager.dict(fn_groups_categories_to_ref_nums_dict)
        
        with mp.Pool(processes=num_processes) as pool:
            ## using starmap_async instead of apply_async to get the results in the order of mr_xyz_paths_list.
            results = pool.starmap_async(gen_rxns_for_one_mr, zip(mr_xyz_paths_list, [shared_fn_groups_ref_nums_to_smiles_dict] * len(mr_xyz_paths_list), [shared_fn_groups_categories_to_ref_nums_dict] * len(mr_xyz_paths_list), out_dir_list, scratch_dir_list, rxns_xyz_namespace_list))
            pool.close()
            pool.join()
            results = results.get()
    
    return results


def main():
    obabel = "/path/to/openbabel/openbabel_3.1.1/bin/obabel"
    mr_xyz_dir = "/path/to/input/mr/xyz/files"
    out_dir = "/path/to/output/ar/xyz/files"

    # fn_groups_ref_nums_to_smiles_dict = json.load(open(os.path.join(data_path_main, "fn_groups_ref_nums_to_smiles_dict.json"), "r"))
    # fn_groups_categories_to_ref_nums_dict = json.load(open(os.path.join(data_path_main, "fn_groups_categories_to_ref_nums_dict.json"), "r"))
    
    fn_groups_ref_nums_to_smiles_dict = json.load(open("fn_groups_ref_nums_to_smiles_dict.json", "r"))
    fn_groups_categories_to_ref_nums_dict = json.load(open("fn_groups_categories_to_ref_nums_dict.json", "r"))

    main_smi_inchikey_dict_path = "main_smi_inchikey_dict.json"
    main_smi_inchikey_dict = json.load(open(main_smi_inchikey_dict_path, "r"))

    mr_xyz_paths_list = []
    out_dir_list = []
    scratch_dir_list = []
    rxns_xyz_namespace_list = []

    for mr_xyz in os.listdir(mr_xyz_dir):
        mr_xyz_path = os.path.join(mr_xyz_dir, mr_xyz)
        mr_name = mr_xyz.split(".")[0]

        mr_xyz_paths_list.append(mr_xyz_path)
        out_dir_list.append(os.path.join(out_dir, mr_name))
        scratch_dir_list.append(os.path.join(out_dir, f".temp_{mr_name}"))
        rxns_xyz_namespace_list.append(mr_name)

    gen_rxns_multiple_mrs_multiprocess(mr_xyz_paths_list, fn_groups_ref_nums_to_smiles_dict, fn_groups_categories_to_ref_nums_dict, out_dir_list=out_dir_list, scratch_dir_list=scratch_dir_list, rxns_xyz_namespace_list=rxns_xyz_namespace_list)
    # gen_rxns_for_one_mr(mr_xyz_path, fn_groups_ref_nums_to_smiles_dict, fn_groups_categories_to_ref_nums_dict, out_dir=None, scratch_dir=None, rxns_xyz_namespace=None):
    
    ## update main_smi_inchikey_dict. useful when characterizing the ARs to prevent double calculations.
    for out_dir_i in tqdm(out_dir_list, desc="Updating main_smi_inchikey_dict"):
        try:
            rxn_smi_to_xyz_name_dict_i = json.load(open(os.path.join(out_dir_i, "rxn_smi_to_xyz_name_dict.json"), "r"))
        except:
            continue

        all_smi_set_i = set()
        for rxn_smi_i in rxn_smi_to_xyz_name_dict_i:
            if rxn_smi_i == "":
                print(f"Empty rxn_smi in {out_dir_i}")
                continue
            rxn_smi_split_i = rxn_smi_i.split(">>")
            all_smi_set_i.update(rxn_smi_split_i[0].split("."))
            all_smi_set_i.update(rxn_smi_split_i[1].split("."))

        if len(all_smi_set_i) == 0:
            continue
        for smi_i in all_smi_set_i:
            smi_can_i = Chem.MolToSmiles(Chem.MolFromSmiles(smi_i), canonical=True, isomericSmiles=False, kekuleSmiles=True)

            if smi_i not in main_smi_inchikey_dict and smi_can_i not in main_smi_inchikey_dict:
                inchikey_i = Chem.MolToInchiKey(Chem.MolFromSmiles(smi_can_i))
                main_smi_inchikey_dict[smi_i] = inchikey_i
                main_smi_inchikey_dict[smi_can_i] = inchikey_i
            else:
                if smi_can_i in main_smi_inchikey_dict:
                    inchikey_i = main_smi_inchikey_dict[smi_can_i]
                    main_smi_inchikey_dict[smi_i] = inchikey_i
                elif smi_i in main_smi_inchikey_dict:
                    inchikey_i = main_smi_inchikey_dict[smi_i]
                    main_smi_inchikey_dict[smi_can_i] = inchikey_i
    
    json.dump(main_smi_inchikey_dict, open(main_smi_inchikey_dict_path, "w"), indent=4)
    
    with open(main_smi_inchikey_dict_path, "w") as f:
        json.dump(main_smi_inchikey_dict, f, indent=4)

    ## clean out_dir. if no .xyz files are present, then delete the directory.
    for out_dir_i in out_dir_list:
        try:
            if len(os.listdir(out_dir_i)) == 0:
                os.rmdir(out_dir_i)
            else:
                for file_i in os.listdir(out_dir_i):
                    if file_i.endswith(".xyz"):
                        break
                else:
                    shutil.rmtree(out_dir_i)
        except:
            continue
    
    return


if __name__ == "__main__":
    main()
