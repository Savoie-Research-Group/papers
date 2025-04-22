"""
    Date: 2025/22/04
    Author(s): Veeru Singla (singla2@purdue.edu)
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


# yarp utils
this_script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(this_script_dir, "yarp_utils"))
from utility import *
from taffi_functions import *
from id_reaction_type import main as id_reaction_type_main


# def xyz_write_wrapper(name,elements,geo,append_opt=False,comment=''):
#     return xyz_write(name,elements,geo,append_opt=append_opt,comment=comment)


data_path_main = "../../data"


def reverse_xyz(in_xyz_path, out_xyz_path):
    E_in, RG_in, PG_in = parse_input(in_xyz_path, return_adj=False)
    E_out, RG_out, PG_out = E_in, PG_in, RG_in
    write_rxn_xyz(E_out, RG_out, PG_out, out_xyz_path)


def xyz_to_smi_yarp(xyz_path):
    E_parse, RG_parse, PG_parse, Radj_mat_parse, Padj_mat_parse = parse_input(xyz_path,return_adj=True)
    rsmi_ob = return_smi(E_parse, RG_parse, adj_mat=Radj_mat_parse)
    psmi_ob = return_smi(E_parse, PG_parse, adj_mat=Padj_mat_parse)
    
    rsmi_ob = rsmi_ob.replace("'\\'", "'\'")
    psmi_ob = psmi_ob.replace("'\\'", "'\'")
    
    try:
        rsmi_rdkit_kekule = rdkit_kekulize_canonicalize(rsmi_ob)
        psmi_rdkit_kekule = rdkit_kekulize_canonicalize(psmi_ob)
    except:
        print(f"WARNING: rdkit_kekulize_canonicalize failed for smiles: {rsmi_ob}>>{psmi_ob}, ", xyz_path)
        rsmi_rdkit_kekule = ""
        psmi_rdkit_kekule = ""
    
    return f"{rsmi_ob}>>{psmi_ob}", f"{rsmi_rdkit_kekule}>>{psmi_rdkit_kekule}"


## meant for when uniradical smiles are incorrectly imported as bi or tri radicals by chemdraw
def correct_radical_counts(fn_groups_ref_nums_to_smiles_dict):
    for num, smi in fn_groups_ref_nums_to_smiles_dict.items():
        rad_count = count_radicals(smi)
        corr_smi = smi
        if rad_count > 1:
            if "[C" in smi:
                if rad_count == 2:
                    corr_smi = corr_smi.replace("[C]", "[CH]")
                if rad_count == 3:
                    corr_smi = corr_smi.replace("[C]", "[CH2]") 
                
                fn_groups_ref_nums_to_smiles_dict[num] = corr_smi
                print(num, smi, rad_count, corr_smi)
            
            elif "[N" in smi:
                if rad_count == 2:
                    corr_smi = corr_smi.replace("[N]", "[NH]")
                    
                fn_groups_ref_nums_to_smiles_dict[num] = corr_smi
                print(num, smi, rad_count, corr_smi)
            
            else:
                print("radical not [C]", num, smi, rad_count)
    
    return fn_groups_ref_nums_to_smiles_dict


def rdkit_kekulize_canonicalize(smi):
    return Chem.MolToSmiles(Chem.MolFromSmiles(smi), canonical=True, isomericSmiles=False, kekuleSmiles=True)


## WARNING: only tested on closed shell and uni-radical neutral molecules. i.e. no open shell or multi-radical molecules, q_tot = 0
def smi_to_mol_xyz(smi, mol_path=None, gen_xyz=False, xyz_path=None, ff="mmff94", obabel="/depot/bsavoie/apps/openbabel_3.1.1/bin/obabel"):
    if mol_path is None:
        mol_path = f"'./{smi}.mol'"
    
    mol_gen_command = f"{obabel} -:'{smi}' -omol -O {mol_path} -xa -xv --gen3d best --minimize --sd --ff {ff}"
    subprocess.run(mol_gen_command, shell=True, check=True)
    
    if gen_xyz:
        if xyz_path is None:
            xyz_path = f"'./{smi}.xyz'"
        xyz_gen_command = f"{obabel} -imol {mol_path} -oxyz -O {xyz_path}"
        subprocess.run(xyz_gen_command, shell=True, check=True)
        return mol_path, xyz_path
    
    return mol_path, None


## WARNING: only tested on closed shell and uni-radical neutral molecules. i.e. no open shell or multi-radical molecules, q_tot = 0
def mol_file_to_smi(mol_path, explicit_h=False, obabel="/depot/bsavoie/apps/openbabel_3.1.1/bin/obabel"):
    smi_gen_command_1 = f"{obabel} -imol {mol_path} -osmi -xh -xi -xc -xk"
    smi_xh = subprocess.run(smi_gen_command_1, shell=True, check=True, capture_output=True).stdout.decode("utf-8").strip()
    
    if explicit_h:
        return smi_xh
    
    smi_gen_command_2 = f"{obabel} -:'{smi_xh}' -osmi -xi -xc -xk"
    smi = subprocess.run(smi_gen_command_2, shell=True, check=True, capture_output=True).stdout.decode("utf-8").strip()
    
    return smi


def mol_file_to_atom_mapped_smi(mol_path, obabel="/depot/bsavoie/apps/openbabel_3.1.1/bin/obabel"):
    smi_gen_command = f"{obabel} -imol {mol_path} -osmi -xh -xi -xa -xc -xk"
    smi = subprocess.run(smi_gen_command, shell=True, check=True, capture_output=True).stdout.decode("utf-8").strip()
    
    return smi


## WARNING: only tested on closed shell and uni-radical neutral molecules. i.e. no open shell or multi-radical molecules, q_tot = 0
def parse_mol(mol_path):
    ## have to modify to correctly parse atom_mapped_mol. i.e. mol file with atom class values as well. 
    # basically identify what atom number corresponds to what atom class value. 
    # and then make ensure that atom class values remain unchanged and atom numbers correspond to the correct atom class values.
    
    ## sample mol file starts below ##
        #
        #  OpenBabel01172423403D
        #
        #   8  8  0  0  0  0  0  0  0  0999 V2000
        #     1.1212    0.3740    0.0008 N   0  0  0  0  0  3  0  0  0  0  0  0
        #     0.6668   -0.8819    0.0008 C   0  0  0  0  0  4  0  0  0  0  0  0
        #    -0.7386   -0.9441   -0.0011 C   0  0  0  0  0  4  0  0  0  0  0  0
        #    -1.1464    0.3684   -0.0022 C   0  0  0  0  0  3  0  0  0  0  0  0
        #    -0.0022    1.1137   -0.0013 N   0  0  0  0  0  3  0  0  0  0  0  0
        #     1.3782   -1.6980    0.0020 H   0  0  0  0  0  1  0  0  0  0  0  0
        #    -1.3693   -1.8206   -0.0017 H   0  0  0  0  0  1  0  0  0  0  0  0
        #     0.0816    2.1240   -0.0024 H   0  0  0  0  0  1  0  0  0  0  0  0
        #   1  2  2  0  0  0  0
        #   1  5  1  0  0  0  0
        #   2  3  1  0  0  0  0
        #   2  6  1  0  0  0  0
        #   3  4  2  0  0  0  0
        #   3  7  1  0  0  0  0
        #   4  5  1  0  0  0  0
        #   5  8  1  0  0  0  0
        # M  END
    ## sample mol file ends above ##

    with open(mol_path, "r") as f:
        mol_lines = f.readlines()
        f.close()
        
    mol_lines = [i.strip().split() for i in mol_lines][3:]
    n_elems, n_bonds = int(mol_lines[0][0]), int(mol_lines[0][1])
    
    geo_elems_lines = mol_lines[1:n_elems+1]
    bonds_lines = mol_lines[n_elems+1:n_elems+n_bonds+1]
    
    G = [[float(i) for i in j[0:3]] for j in geo_elems_lines]
    E = [i[3] for i in geo_elems_lines]
    valence_vector = [int(i[9]) for i in geo_elems_lines]
    
    G = np.array(G)
    # E = np.array(E)
    
    adj_mat = np.zeros((n_elems, n_elems), dtype=int)
    bond_mat = np.zeros((n_elems, n_elems), dtype=int)
    
    for bond_line in bonds_lines:
        i, j, bond_type = int(bond_line[0])-1, int(bond_line[1])-1, int(bond_line[2])
        adj_mat[i][j] = 1
        adj_mat[j][i] = 1
        bond_mat[i][j] = bond_type
        bond_mat[j][i] = bond_type
    
    return E, G, adj_mat, bond_mat, valence_vector


## WARNING: only tested on closed shell and uni-radical neutral molecules. i.e. no open shell or multi-radical molecules, q_tot = 0    
def write_mol(E, G, adj_mat, bond_mat, mol_path=None, valence_vector=None):
    ## sample mol file starts below ##
        #
        #  OpenBabel01172423403D
        #
        #   8  8  0  0  0  0  0  0  0  0999 V2000
        #     1.1212    0.3740    0.0008 N   0  0  0  0  0  3  0  0  0  0  0  0
        #     0.6668   -0.8819    0.0008 C   0  0  0  0  0  4  0  0  0  0  0  0
        #    -0.7386   -0.9441   -0.0011 C   0  0  0  0  0  4  0  0  0  0  0  0
        #    -1.1464    0.3684   -0.0022 C   0  0  0  0  0  3  0  0  0  0  0  0
        #    -0.0022    1.1137   -0.0013 N   0  0  0  0  0  3  0  0  0  0  0  0
        #     1.3782   -1.6980    0.0020 H   0  0  0  0  0  1  0  0  0  0  0  0
        #    -1.3693   -1.8206   -0.0017 H   0  0  0  0  0  1  0  0  0  0  0  0
        #     0.0816    2.1240   -0.0024 H   0  0  0  0  0  1  0  0  0  0  0  0
        #   1  2  2  0  0  0  0
        #   1  5  1  0  0  0  0
        #   2  3  1  0  0  0  0
        #   2  6  1  0  0  0  0
        #   3  4  2  0  0  0  0
        #   3  7  1  0  0  0  0
        #   4  5  1  0  0  0  0
        #   5  8  1  0  0  0  0
        # M  END
    ## sample mol file ends above ##
    
    if mol_path is None:
        mol_path = f"'./{random.randint(0, 10000000)}.mol'"
    
    with open(mol_path, "w") as f:
        f.write(f"\n write_mol_vs\n\n")
        f.write("{:>3d}{:>3d}  0  0  0  0  0  0  0  0999 V2000\n".format(len(E),int(np.sum(adj_mat/2.0))))
        
        for i in range(len(E)):
            if valence_vector is None:
                f.write(" {:> 9.4f} {:> 9.4f} {:> 9.4f} {:<3s} 0  0  0  0  0  0  0  0  0{:>3d}  0  0\n".format(G[i][0],G[i][1],G[i][2],E[i],i+1))  # the last {:>3d} = i+1 is atom class/index to get atom mapped smiles
            else:
                f.write(" {:> 9.4f} {:> 9.4f} {:> 9.4f} {:<3s} 0  0  0  0  0{:>3d}  0  0  0{:>3d}  0  0\n".format(G[i][0],G[i][1],G[i][2],E[i],valence_vector[i],i+1))  # the last {:>3d} = i+1 is atom class/index to get atom mapped smiles
        
        for i in range(len(adj_mat)):
            for j in range(len(adj_mat))[i+1:]:
                if adj_mat[i][j] == 1:
                    f.write("{:>3d}{:>3d}{:>3d}  0  0  0  0\n".format(i+1,j+1,int(bond_mat[i][j])))
        
        f.write("M  END\n")
        f.close()
    
    return mol_path


def write_xyz(E, G, xyz_path=None):
    # with open(xyz_path, "w") as f:
    #     f.write(f"{len(E)}\n\n")
    #     for i in range(len(E)):
    #         f.write(f"{E[i]} {G[i][0]} {G[i][1]} {G[i][2]}\n")
    #     f.close()
    
    if xyz_path is None:
        xyz_path = f"'./{random.randint(0, 10000000)}.xyz'"
    
    with open(xyz_path, "w") as f:
        f.write(f"{len(E)}\n\n")
        for i in range(len(E)):
            f.write(f"{E[i]:<3s}{G[i][0]:>15.5f}{G[i][1]:>15.5f}{G[i][2]:>15.5f}\n")
        f.close()
    
    return xyz_path


def write_rxn_xyz(E, G_1, G_2, xyz_path=None):
    ## two xyz in same file one after the other
    
    if xyz_path is None:
        xyz_path = f"'./{random.randint(0, 10000000)}.xyz'"
    
    with open(xyz_path, "w") as f:
        f.write(f"{len(E)}\n\n")
        for i in range(len(E)):
            f.write(f"{E[i]:<3s}{G_1[i][0]:>15.5f}{G_1[i][1]:>15.5f}{G_1[i][2]:>15.5f}\n")
        f.write(f"{len(E)}\n\n")
        for i in range(len(E)):
            f.write(f"{E[i]:<3s}{G_2[i][0]:>15.5f}{G_2[i][1]:>15.5f}{G_2[i][2]:>15.5f}\n")
        f.close()
    
    return xyz_path


def opt_geo_mol(mol_path, ff="mmff94", obabel="/depot/bsavoie/apps/openbabel_3.1.1/bin/obabel"):
    # mol_opt_command = f"{obabel} -imol {mol_path} -omol -O {mol_path} -xa -xv --gen3d best --minimize --sd --ff {ff}"
    mol_opt_command = f"{obabel} -imol {mol_path} -omol -O {mol_path} -xa -xv --minimize --sd --ff {ff}"
    subprocess.run(mol_opt_command, shell=True, check=True)
    
    return mol_path


## WARNING: only for uniradical neutral smiles. there may be multiple smiles though, just that each smiles should be uniradical neutral.
def parse_fn_grp_smi(smi, ff="mmff94", obabel="/depot/bsavoie/apps/openbabel_3.1.1/bin/obabel"):
    max_valence_dict = {"C": 4, "N": 3, "O": 2, "H": 1}
    
    ## generate temp mol file. parse it. delete it.
    temp_mol, _ = smi_to_mol_xyz(smi, mol_path="./.temp.mol", ff=ff, obabel=obabel)
    
    rec_smi = mol_file_to_smi(temp_mol)
    smi_can = rdkit_kekulize_canonicalize(smi)
    rec_smi_can = rdkit_kekulize_canonicalize(rec_smi)
    
    if smi_can != rec_smi_can:
        print("WARNING: smiles to mol to smiles failed. recreated smiles is different from original smiles.")
        print("smi: ", smi)
        print("rec_smi: ", rec_smi)
        print("smi_can: ", smi_can)
        print("rec_smi_can: ", rec_smi_can)
        return False, {}
    
    E, G, adj_mat, bond_mat, valence_vector = parse_mol(temp_mol)
    os.remove(temp_mol)
    
    # ## write temp mol 2 file and parse it again to check if the parsing is correct
    # temp_mol_2 = "./.temp_2.mol"
    # write_mol(E, G, adj_mat, bond_mat, temp_mol_2, valence_vector=valence_vector)
    # rec_smi_2 = mol_file_to_smi(temp_mol_2)
    # subprocess.run(f"rm {temp_mol_2}", shell=True, check=True)
    # rec_smi_can_2 = Chem.MolToSmiles(Chem.MolFromSmiles(rec_smi_2), canonical=True, isomericSmiles=False, kekuleSmiles=True)
    
    # if smi_can != rec_smi_can_2:
    #     print("WARNING: parsed smiles to mol to smiles failed. recreated mol object is wrong.")
    #     print("smi: ", smi)
    #     print("rec_smi_2: ", rec_smi_2)
    #     print("smi_can: ", smi_can)
    #     print("rec_smi_can_2: ", rec_smi_can_2)
    #     return False, {}
    
    ## find uniradical atom using E and bond_mat. multiple uniradical molecules are possible.
    radical_atoms = []
    for i in range(len(E)):
        if np.sum(bond_mat[i]) < max_valence_dict[E[i]]:
            radical_atoms += [i]
    
    return True, {"E": E, "G": G, "adj_mat": adj_mat, "bond_mat": bond_mat, "valence_vector": valence_vector, "radical_atoms": radical_atoms}


def parse_mr_xyz(xyz_path):
    
    ## from yarp model_rxn code ##
    # Extract Element list and Coord list from the file
    canonical = False
    E, RG, PG, adj_mat_1, adj_mat_2 = parse_input(xyz_path, return_adj=True)    
    
    # apply find lewis
    lone_1,_,_,bond_mat_1,fc_1 = find_lewis(E,adj_mat_1,q_tot=0,keep_lone=[],return_pref=False,return_FC=True)
    lone_2,_,_,bond_mat_2,fc_2 = find_lewis(E,adj_mat_2,q_tot=0,keep_lone=[],return_pref=False,return_FC=True)
    
    # locate radical positions
    keep_lone_1  = [ [ count_i for count_i,i in enumerate(lone_electron) if i%2 != 0] for lone_electron in lone_1]
    keep_lone_2  = [ [ count_i for count_i,i in enumerate(lone_electron) if i%2 != 0] for lone_electron in lone_2]
    
    # contruct BE matrix
    BE_1   = np.diag(lone_1[0])+bond_mat_1[0]
    
    # loop over possible BE matrix of product 
    diff_list = []
    for ind in range(len(bond_mat_2)):
        BE_2   = np.diag(lone_2[ind])+bond_mat_2[ind]
        BE_change = BE_2 - BE_1
        diff_list += [np.abs(BE_change).sum()]
        
    # determine the BE matrix leads to the smallest change
    ind = diff_list.index(min(diff_list))
    BE_2   = np.diag(lone_2[ind])+bond_mat_2[ind]
    BE_change = BE_2 - BE_1
    
    # determine bonds break and bonds form from Reaction matrix
    bond_break = []
    bond_form  = []
    for i in range(len(E)):
        for j in range(i+1,len(E)):
            if BE_change[i][j] == -1:
                bond_break += [(i,j)]
            if BE_change[i][j] == 1:
                bond_form += [(i,j)]
    ## yarp model rxn code ends here ##
    
    ## my code starts here ##
    primary_atoms_list = []
    for bond in bond_break:
        primary_atoms_list += [bond[0], bond[1]]
    for bond in bond_form:
        primary_atoms_list += [bond[0], bond[1]]
    primary_atoms_list = list(set(primary_atoms_list))
    
    first_neighbor_non_h_atoms_list_1 = set()
    first_neighbor_non_h_atoms_list_2 = set()
    for atom in primary_atoms_list:
        first_neighbor_non_h_atoms_list_1.update({i for i in np.where(adj_mat_1[atom] == 1)[0] if E[i] != "H"})
        first_neighbor_non_h_atoms_list_2.update({i for i in np.where(adj_mat_2[atom] == 1)[0] if E[i] != "H"})
    first_neighbor_non_h_atoms_list_1 = first_neighbor_non_h_atoms_list_1 - set(primary_atoms_list)
    first_neighbor_non_h_atoms_list_2 = first_neighbor_non_h_atoms_list_2 - set(primary_atoms_list)
    assert first_neighbor_non_h_atoms_list_1 == first_neighbor_non_h_atoms_list_2
    first_neighbor_non_h_atoms_list_1 = list(first_neighbor_non_h_atoms_list_1)
    first_neighbor_non_h_atoms_list_2 = list(first_neighbor_non_h_atoms_list_2)
    
    first_neighbors_h_bonds_dict_1 = {}
    first_neighbors_h_bonds_dict_2 = {}
    for atom in first_neighbor_non_h_atoms_list_1:
        first_neighbors_h_bonds_dict_1[atom] = {i for i in np.where(adj_mat_1[atom] == 1)[0] if E[i] == "H"}
    for atom in first_neighbor_non_h_atoms_list_2:
        first_neighbors_h_bonds_dict_2[atom] = {i for i in np.where(adj_mat_2[atom] == 1)[0] if E[i] == "H"}
    assert first_neighbors_h_bonds_dict_1 == first_neighbors_h_bonds_dict_2
    first_neighbors_h_bonds_dict_1 = {k: list(v) for k, v in first_neighbors_h_bonds_dict_1.items()}
    # first_neighbors_h_bonds_dict_2 = {k: list(v) for k, v in first_neighbors_h_bonds_dict_2.items()}
    
    ## modify first_neighbor_non_h_atoms_list_1 and first_neighbors_h_bonds_dict_1 to remove empty lists from first_neighbors_h_bonds_dict_1
    first_neighbor_non_h_atoms_list_1 = [i for i in first_neighbor_non_h_atoms_list_1 if len(first_neighbors_h_bonds_dict_1[i]) > 0]
    first_neighbors_h_bonds_dict_1 = {k: v for k, v in first_neighbors_h_bonds_dict_1.items() if len(v) > 0}
    
    # lone_1[0], lone_2[0], fc_1[0], fc_2[0], bond_break, bond_form,
    return E, RG, PG, adj_mat_1, adj_mat_2, bond_mat_1[0], bond_mat_2[ind], primary_atoms_list, first_neighbor_non_h_atoms_list_1, first_neighbors_h_bonds_dict_1
    
    
## to prepare mr for addition of fn group(s)
def remove_atoms_from_mr(E_in, RG_in, PG_in, adj_mat_1_in, adj_mat_2_in, bond_mat_1_in, bond_mat_2_in, atoms_to_remove_list_in=[]):
    E = deepcopy(E_in)
    RG = deepcopy(RG_in)
    PG = deepcopy(PG_in)
    adj_mat_1 = deepcopy(adj_mat_1_in)
    adj_mat_2 = deepcopy(adj_mat_2_in)
    bond_mat_1 = deepcopy(bond_mat_1_in)
    bond_mat_2 = deepcopy(bond_mat_2_in)
    atoms_to_remove_list = deepcopy(atoms_to_remove_list_in)
    atoms_to_remove_list.sort(reverse=True)
    # old_to_new_atom_index_dict = {old_index: old_index for old_index in range(len(E)) if old_index not in atoms_to_remove_list}
    # for i in old_to_new_atom_index_dict.keys():
    #     for j in atoms_to_remove_list:
    #         if i > j:
    #             old_to_new_atom_index_dict[i] -= 1
    old_to_new_atom_index_dict = {old_index: old_index - sum([1 for i in atoms_to_remove_list if i < old_index]) for old_index in range(len(E)) if old_index not in atoms_to_remove_list}
    
    E = [i for count_i, i in enumerate(E) if count_i not in atoms_to_remove_list]
    RG = np.delete(RG, atoms_to_remove_list, axis=0)
    PG = np.delete(PG, atoms_to_remove_list, axis=0)
    adj_mat_1 = np.delete(np.delete(adj_mat_1, atoms_to_remove_list, axis=0), atoms_to_remove_list, axis=1)
    bond_mat_1 = np.delete(np.delete(bond_mat_1, atoms_to_remove_list, axis=0), atoms_to_remove_list, axis=1)
    adj_mat_2 = np.delete(np.delete(adj_mat_2, atoms_to_remove_list, axis=0), atoms_to_remove_list, axis=1)
    bond_mat_2 = np.delete(np.delete(bond_mat_2, atoms_to_remove_list, axis=0), atoms_to_remove_list, axis=1)
    
    return old_to_new_atom_index_dict, E, RG, PG, adj_mat_1, adj_mat_2, bond_mat_1, bond_mat_2


def separate_geos(G_1_in, G_2_in, min_dist=1.5):
    ## move both G_1 and G_2 to the origin.
    ## rotate G_2 about its centroid such that the average distance between any two atoms in G_1 and G_2 is the lowest possible of all possible rotations.
    ## then translate G_2 such that the minimum distance between any two atoms in G_1 and G_2 is at least min_dist.
    ## min_dist is a heuristic in angstroms.
    
    G_1 = deepcopy(G_1_in)
    G_2 = deepcopy(G_2_in)
    
    ## calculate centroid of G_1 and G_2
    centroid_1 = np.mean(G_1, axis=0)
    centroid_2 = np.mean(G_2, axis=0)
    
    ## move both G_1 and G_2 to the origin
    G_1_origin = G_1 - centroid_1
    G_2_origin = G_2 - centroid_2
    
    ## find which has less points and add rows of zeros to it to make it equal to the other. then before returning at the end, remove the rows of zeros.
    len_initial_G_1 = len(G_1_origin)
    len_initial_G_2 = len(G_2_origin)
    
    if len_initial_G_1 < len_initial_G_2:
        G_1_origin = np.concatenate((G_1_origin, np.zeros((len_initial_G_2 - len_initial_G_1, 3))), axis=0)
    elif len_initial_G_2 < len_initial_G_1:
        G_2_origin = np.concatenate((G_2_origin, np.zeros((len_initial_G_1 - len_initial_G_2, 3))), axis=0)
    
    ## find rotation matrix for G_2 using SVD
    U, _, Vt = np.linalg.svd(np.matmul(G_1_origin.T, G_2_origin))
    R = np.matmul(U, Vt)
    
    ## rotate G_2 using R
    G_2_rotated = np.matmul(G_2_origin, R)
    G_2_origin_rotated = G_2_rotated - np.mean(G_2_rotated, axis=0)
    
    ## find translation vector which will be the normal to the least squares plane of all the atoms in G_1_origin and G_2_origin_rotated
    G_all_origin = np.concatenate((G_1_origin, G_2_origin_rotated), axis=0)
    
    # fit plane to points and find normal
    A = np.c_[G_all_origin[:, 0], G_all_origin[:, 1], np.ones(G_all_origin.shape[0])]
    C, _, _, _ = sp.linalg.lstsq(A, G_all_origin[:, 2])    # coefficients of plane equation
    normal = np.array([C[0], C[1], -1.])
    normal = normal / np.linalg.norm(normal)
    
    ## translate G_2_origin_rotated along the normal such that the minimum distance between any two atoms in G_1_origin and G_2_origin_rotated is at least min_dist
    G_2_origin_rotated_translated = G_2_origin_rotated + normal * (min_dist + np.max(np.linalg.norm(G_1_origin[:, None, :] - G_2_origin_rotated, axis=2)))
    
    ## remove the rows of zeros from G_1_origin and G_2_origin_rotated_translated
    G_1_origin = G_1_origin[:len_initial_G_1]
    G_2_origin_rotated_translated = G_2_origin_rotated_translated[:len_initial_G_2]
    
    return G_1_origin, G_2_origin_rotated_translated


## use the modified (atom removed) mr and add fn group(s).
def add_fn_grps_to_modified_mr(E_in, RG_in, PG_in, adj_mat_1_in, adj_mat_2_in, bond_mat_1_in, bond_mat_2_in, atoms_to_bond_to_list=[], fn_grps_to_add_dicts_list_in=[]):
    E = deepcopy(E_in)
    RG = deepcopy(RG_in)
    PG = deepcopy(PG_in)
    adj_mat_1 = deepcopy(adj_mat_1_in)
    adj_mat_2 = deepcopy(adj_mat_2_in)
    bond_mat_1 = deepcopy(bond_mat_1_in)
    bond_mat_2 = deepcopy(bond_mat_2_in)
    fn_grps_to_add_dicts_list = deepcopy(fn_grps_to_add_dicts_list_in)
    
    assert len(atoms_to_bond_to_list) == len(fn_grps_to_add_dicts_list)
    
    def initialize_joint_adj_or_bond_mats(mat_1, mat_2):
        ## can be done in a single step. doing in multiple for clarity and readability.
        upper_left_mat = mat_1
        upper_right_mat = np.zeros((len(mat_1), len(mat_2)), dtype=int)
        upper_mat = np.concatenate((upper_left_mat, upper_right_mat), axis=1)
        
        lower_left_mat = np.zeros((len(mat_2), len(mat_1)), dtype=int)
        lower_right_mat = mat_2
        lower_mat = np.concatenate((lower_left_mat, lower_right_mat), axis=1)
        
        return np.concatenate((upper_mat, lower_mat), axis=0)
    
    updated_fn_grps_radical_atoms_list = []
    ## add fn groups to the modified mr
    for fn_grp_to_add in fn_grps_to_add_dicts_list:
        updated_fn_grps_radical_atoms_list.append([i+len(E) for i in fn_grp_to_add["radical_atoms"]])
        
        E += fn_grp_to_add["E"]
        RG, fn_grp_to_add_G_1 = separate_geos(RG, fn_grp_to_add["G"])
        PG, fn_grp_to_add_G_2 = separate_geos(PG, fn_grp_to_add["G"])
        RG = np.concatenate((RG, fn_grp_to_add_G_1), axis=0)
        PG = np.concatenate((PG, fn_grp_to_add_G_2), axis=0)
        
        adj_mat_1 = initialize_joint_adj_or_bond_mats(adj_mat_1, fn_grp_to_add["adj_mat"])
        bond_mat_1 = initialize_joint_adj_or_bond_mats(bond_mat_1, fn_grp_to_add["bond_mat"])
        adj_mat_2 = initialize_joint_adj_or_bond_mats(adj_mat_2, fn_grp_to_add["adj_mat"])
        bond_mat_2 = initialize_joint_adj_or_bond_mats(bond_mat_2, fn_grp_to_add["bond_mat"])
        
    ## add bonds between atoms_to_bond_to_list and the fn group atoms
    for atom_to_bond_to, fn_grp_radical_atoms in zip(atoms_to_bond_to_list, updated_fn_grps_radical_atoms_list):
        for fn_grp_radical_atom in fn_grp_radical_atoms:
            adj_mat_1[atom_to_bond_to][fn_grp_radical_atom] = 1
            adj_mat_1[fn_grp_radical_atom][atom_to_bond_to] = 1
            
            bond_mat_1[atom_to_bond_to][fn_grp_radical_atom] = 1
            bond_mat_1[fn_grp_radical_atom][atom_to_bond_to] = 1
            
            adj_mat_2[atom_to_bond_to][fn_grp_radical_atom] = 1
            adj_mat_2[fn_grp_radical_atom][atom_to_bond_to] = 1
            
            bond_mat_2[atom_to_bond_to][fn_grp_radical_atom] = 1
            bond_mat_2[fn_grp_radical_atom][atom_to_bond_to] = 1
            
    return E, RG, PG, adj_mat_1, adj_mat_2, bond_mat_1, bond_mat_2
