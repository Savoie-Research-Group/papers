import logging
logging.getLogger().setLevel(logging.ERROR)

import warnings
warnings.filterwarnings('ignore')

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # filter info and warnings. still prints errors

from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

import rdkit.Chem as Chem
import rdkit.Chem.AllChem as AllChem
import json
from matplotlib import pyplot as plt
from copy import deepcopy
import pickle
import numpy as np
import math
from scipy.special import expit

from numpy.random import default_rng
np_rng_ = default_rng()

import multiprocessing as mp


this_script_directory = os.path.dirname(os.path.realpath(__file__))
os.chdir(this_script_directory)


def dice_similarity(a, b):
    a = np.array(a, dtype=np.int32)
    b = np.array(b, dtype=np.int32)
    return 2 * np.dot(a, b) / (np.dot(a, a) + np.dot(b, b))


def flatten_2d_list(list_2d_):
    return list({jk for sub in list_2d_ for jk in sub})


def check_smi_valid(smi_in):
    m = Chem.MolFromSmiles(smi_in, sanitize=False)
    if m is None:
        return False
    else:
        try:
            Chem.SanitizeMol(m)
            return True
        except:
            return False


def list_non_h_atoms_and_bonds_in_rdmol(smi_in_, rdmol_in_):
    if smi_in_ == "[H]":
        non_h_atoms_1_ = set()
        non_h_bonds_1_ = []
        h_bonds_1_ = []
        heavy_atom_total_bonds_ = dict()
    else:
        # using union of all non-H paths separated by 1 distance
        rdmol_1_ = deepcopy(rdmol_in_)

        non_h_atoms_1_ = []
        heavy_atom_total_bonds_ = {}

        adj_mat_1_ = Chem.GetAdjacencyMatrix(rdmol_1_, useBO=True, force=True)

        for i, row__ in enumerate(adj_mat_1_):
            if int(sum(row__)) > 1:
                non_h_atoms_1_.append(i)
                heavy_atom_total_bonds_[i] = int(sum(row__))

        non_h_atoms_1_ = set(non_h_atoms_1_)

        # sets with bonding details in the format [[bond], bond_order]
        non_h_bonds_1_ = []
        h_bonds_1_ = []

        for i in range(len(adj_mat_1_)):
            for j in range(i, len(adj_mat_1_)):
                if adj_mat_1_[i][j] != 0:
                    bond_detail_1_ = [[i, j], int(adj_mat_1_[i][j])]
                    if i in non_h_atoms_1_ and j in non_h_atoms_1_:
                        non_h_bonds_1_.append(bond_detail_1_)
                    else:
                        h_bonds_1_.append(bond_detail_1_)

    return non_h_atoms_1_, non_h_bonds_1_, h_bonds_1_, heavy_atom_total_bonds_


def grow_alk(smi_in_rdmol_h, c_h_bonds):

    bonds_to_remove = {}
    for bond in c_h_bonds:
        if bond[0][0] not in bonds_to_remove:
            bonds_to_remove[bond[0][0]] = bond[0]

    alkyl_set = set()

    for bond_nums in bonds_to_remove.values():
        smi_in_rdmol_h_copy = deepcopy(smi_in_rdmol_h)
        Chem.Kekulize(smi_in_rdmol_h_copy, clearAromaticFlags=True)
        smi_in_rdmol_h_copy.RemoveBond(bond_nums[0], bond_nums[1])
        smi_in_rdmol_h_copy.GetAtomWithIdx(bond_nums[0]).SetNoImplicit(True)
        smi_in_rdmol_h_copy.GetAtomWithIdx(bond_nums[1]).SetNoImplicit(True)
        Chem.SanitizeMol(smi_in_rdmol_h_copy)

        for prod_smi in Chem.CanonSmiles(Chem.MolToSmiles(smi_in_rdmol_h_copy)).split("."):
            if prod_smi != "[H]":
                alkyl_set.add(prod_smi)

    all_growth_set = set()

    for alkyl_smi in alkyl_set:
        try:
            growth_out_smi = Chem.CanonSmiles(f"{alkyl_smi}.[CH3]")
            growth_out_rdmol = Chem.AddHs(Chem.MolFromSmiles(growth_out_smi))
            _, _, _, c_total_bonds = list_non_h_atoms_and_bonds_in_rdmol(growth_out_smi, growth_out_rdmol)
            radical_cs = [k for k, v in c_total_bonds.items() if v == 3]
            growth_out_rdmol_h = Chem.RWMol(growth_out_rdmol)
            Chem.Kekulize(growth_out_rdmol_h, clearAromaticFlags=True)
            growth_out_rdmol_h.AddBond(radical_cs[0], radical_cs[1], Chem.rdchem.BondType.SINGLE)
            growth_out_rdmol_h.GetAtomWithIdx(radical_cs[0]).SetNoImplicit(True)
            growth_out_rdmol_h.GetAtomWithIdx(radical_cs[1]).SetNoImplicit(True)
            Chem.SanitizeMol(growth_out_rdmol_h)
            all_growth_set.add(Chem.CanonSmiles(Chem.MolToSmiles(growth_out_rdmol_h)))
        except:
            pass

    return all_growth_set


def mut_del_alk(smi_in_rdmol_h, c_c_bonds):

    alkyl_pair_set = set()

    c_atom_bond_list_dict = {}

    for bond_ in c_c_bonds:
        for i, bn in enumerate(bond_[0]):
            if bn in c_atom_bond_list_dict:
                try:
                    c_atom_bond_list_dict[bn].add(bond_[0][i+1])
                except:
                    c_atom_bond_list_dict[bn].add(bond_[0][i-1])
            else:
                try:
                    c_atom_bond_list_dict[bn] = {bond_[0][i+1]}
                except:
                    c_atom_bond_list_dict[bn] = {bond_[0][i-1]}

        smi_in_rdmol_h_copy = deepcopy(smi_in_rdmol_h)
        Chem.Kekulize(smi_in_rdmol_h_copy, clearAromaticFlags=True)
        smi_in_rdmol_h_copy.RemoveBond(bond_[0][0], bond_[0][1])
        smi_in_rdmol_h_copy.GetAtomWithIdx(bond_[0][0]).SetNoImplicit(True)
        smi_in_rdmol_h_copy.GetAtomWithIdx(bond_[0][1]).SetNoImplicit(True)
        Chem.SanitizeMol(smi_in_rdmol_h_copy)

        alkyl_pair_set.add(frozenset(Chem.CanonSmiles(Chem.MolToSmiles(smi_in_rdmol_h_copy)).split(".")))

    all_mut_set = set()

    ##################
    #### deletion ####
    ##################

    for c_at in c_atom_bond_list_dict:
        try:
            c_atom_bond_list_dict[c_at] = list(c_atom_bond_list_dict[c_at])
            smi_in_rdmol_h_copy = deepcopy(smi_in_rdmol_h)
            Chem.Kekulize(smi_in_rdmol_h_copy, clearAromaticFlags=True)

            if len(c_atom_bond_list_dict[c_at]) == 1:
                smi_in_rdmol_h_copy.RemoveBond(c_at, c_atom_bond_list_dict[c_at][0])
                smi_in_rdmol_h_copy.GetAtomWithIdx(c_at).SetNoImplicit(True)
                smi_in_rdmol_h_copy.GetAtomWithIdx(c_atom_bond_list_dict[c_at][0]).SetNoImplicit(True)
                Chem.SanitizeMol(smi_in_rdmol_h_copy)
                for smi in Chem.CanonSmiles(Chem.MolToSmiles(smi_in_rdmol_h_copy)).split("."):
                    if smi != "[CH3]":
                        all_mut_set.add(smi.replace("[CH2]", "C").replace("[CH]", "C").replace("[C]", "C"))

            elif len(c_atom_bond_list_dict[c_at]) == 2:
                smi_in_rdmol_h_copy.RemoveBond(c_at, c_atom_bond_list_dict[c_at][0])
                smi_in_rdmol_h_copy.RemoveBond(c_at, c_atom_bond_list_dict[c_at][1])
                smi_in_rdmol_h_copy.AddBond(c_atom_bond_list_dict[c_at][0], c_atom_bond_list_dict[c_at][1], Chem.rdchem.BondType.SINGLE)
                smi_in_rdmol_h_copy.GetAtomWithIdx(c_at).SetNoImplicit(True)
                smi_in_rdmol_h_copy.GetAtomWithIdx(c_atom_bond_list_dict[c_at][0]).SetNoImplicit(True)
                smi_in_rdmol_h_copy.GetAtomWithIdx(c_atom_bond_list_dict[c_at][1]).SetNoImplicit(True)
                Chem.SanitizeMol(smi_in_rdmol_h_copy)
                for smi in Chem.CanonSmiles(Chem.MolToSmiles(smi_in_rdmol_h_copy)).split("."):
                    if "[" not in smi:
                        all_mut_set.add(smi)
        except:
            pass

    return all_mut_set


def mut_ins_alk(smi_in_rdmol_h, c_c_bonds):
    alkyl_pair_set = set()

    c_atom_bond_list_dict = {}

    for bond_ in c_c_bonds:
        for i, bn in enumerate(bond_[0]):
            if bn in c_atom_bond_list_dict:
                try:
                    c_atom_bond_list_dict[bn].add(bond_[0][i+1])
                except:
                    c_atom_bond_list_dict[bn].add(bond_[0][i-1])
            else:
                try:
                    c_atom_bond_list_dict[bn] = {bond_[0][i+1]}
                except:
                    c_atom_bond_list_dict[bn] = {bond_[0][i-1]}

        smi_in_rdmol_h_copy = deepcopy(smi_in_rdmol_h)
        Chem.Kekulize(smi_in_rdmol_h_copy, clearAromaticFlags=True)
        smi_in_rdmol_h_copy.RemoveBond(bond_[0][0], bond_[0][1])
        smi_in_rdmol_h_copy.GetAtomWithIdx(bond_[0][0]).SetNoImplicit(True)
        smi_in_rdmol_h_copy.GetAtomWithIdx(bond_[0][1]).SetNoImplicit(True)
        Chem.SanitizeMol(smi_in_rdmol_h_copy)

        alkyl_pair_set.add(frozenset(Chem.CanonSmiles(Chem.MolToSmiles(smi_in_rdmol_h_copy)).split(".")))

    all_mut_set = set()

    ###################
    #### insertion ####
    ###################

    for alkyl_pair in alkyl_pair_set:

        try:
            if len(alkyl_pair) == 1:
                alkyl_pair = list(alkyl_pair) * 2
            mut_ins_smi = ".[CH2].".join(alkyl_pair)
            mut_ins_smi_rdmol = Chem.AddHs(Chem.MolFromSmiles(mut_ins_smi))
            _, _, _, c_total_bonds = list_non_h_atoms_and_bonds_in_rdmol(mut_ins_smi, mut_ins_smi_rdmol)
            radical_alkyl_cs = [k for k, v in c_total_bonds.items() if v == 3]
            biradical_alkyl_cs = [k for k, v in c_total_bonds.items() if v == 2]
            mut_ins_smi_rdmol_h = Chem.RWMol(mut_ins_smi_rdmol)
            Chem.Kekulize(mut_ins_smi_rdmol_h, clearAromaticFlags=True)
            mut_ins_smi_rdmol_h.AddBond(radical_alkyl_cs[0], biradical_alkyl_cs[0], Chem.rdchem.BondType.SINGLE)
            mut_ins_smi_rdmol_h.AddBond(radical_alkyl_cs[1], biradical_alkyl_cs[0], Chem.rdchem.BondType.SINGLE)
            mut_ins_smi_rdmol_h.GetAtomWithIdx(radical_alkyl_cs[0]).SetNoImplicit(True)
            mut_ins_smi_rdmol_h.GetAtomWithIdx(biradical_alkyl_cs[0]).SetNoImplicit(True)
            mut_ins_smi_rdmol_h.GetAtomWithIdx(radical_alkyl_cs[1]).SetNoImplicit(True)
            Chem.SanitizeMol(mut_ins_smi_rdmol_h)
            all_mut_set.add(Chem.CanonSmiles(Chem.MolToSmiles(mut_ins_smi_rdmol_h)))
        except:
            pass

    return all_mut_set


smi_in_prods_master_record = {}

def mut_grow_smi_fixed_len(smi_in):
    if smi_in in smi_in_prods_master_record:
        return smi_in_prods_master_record[smi_in]
    
    len_sm_in = smi_in.count("C")
    smi_in_rdmol = Chem.AddHs(Chem.MolFromSmiles(smi_in))
    _, c_c_bonds, c_h_bonds, _ = list_non_h_atoms_and_bonds_in_rdmol(smi_in, smi_in_rdmol)
    smi_in_rdmol_h = Chem.RWMol(smi_in_rdmol)

    const_len_mut_grow_set = set()

    ## find all del and pass them to ins and grow
    del_set = {i for i in mut_del_alk(smi_in_rdmol_h, c_c_bonds) if i.count("C") == len_sm_in - 1}

    for del_smi in del_set:
        del_smi_rdmol = Chem.AddHs(Chem.MolFromSmiles(del_smi))
        _, c_c_bonds, c_h_bonds, _ = list_non_h_atoms_and_bonds_in_rdmol(del_smi, del_smi_rdmol)
        del_smi_rdmol_h = Chem.RWMol(del_smi_rdmol)
        
        const_len_mut_grow_set.update(mut_ins_alk(del_smi_rdmol_h, c_c_bonds).union(grow_alk(del_smi_rdmol_h, c_h_bonds)))

    smi_in_prods_master_record[smi_in] = const_len_mut_grow_set

    return const_len_mut_grow_set


def run_single_gen(start_pop_list_in, start_pop_fitness_list_in, fitness_val_dict, constraint_set, fr_to_keep, fr_parents):
    start_pop_fitness_list_in, start_pop_list_in = (list(t) for t in zip(*sorted(zip(start_pop_fitness_list_in, start_pop_list_in), reverse=True)))

    keep_len = math.ceil(len(start_pop_list_in) * fr_to_keep)
    parents_len = math.ceil(len(start_pop_list_in) * fr_parents)

    children_smi_set = set()
    for smi_ in start_pop_list_in[:parents_len]:
        children_smi_set.update(mut_grow_smi_fixed_len(smi_))

    children_smi_list = []
    children_fitness_list = []
    for pot_smi in children_smi_set:
        if pot_smi in constraint_set and pot_smi not in start_pop_list_in:
            children_smi_list.append(pot_smi)
            children_fitness_list.append(fitness_val_dict[pot_smi])
    children_fitness_list = expit(children_fitness_list)
    children_fitness_list = children_fitness_list/sum(children_fitness_list)

    replace_list = list(np_rng_.choice(children_smi_list, size=len(start_pop_list_in) - keep_len, replace=False, p=children_fitness_list))
    gen_pop_out_list = start_pop_list_in[:keep_len] + replace_list
    gen_pop_fitness_out_list = [fitness_val_dict[i] for i in gen_pop_out_list]

    gen_pop_fitness_out_list, gen_pop_out_list = (list(t) for t in zip(*sorted(zip(gen_pop_fitness_out_list, gen_pop_out_list), reverse=True)))

    return gen_pop_out_list, gen_pop_fitness_out_list


def run_genetic_n_gens(start_pop_list_in, fitness_val_dict, constraint_set, fr_to_keep, fr_parents, max_gens):
    start_pop_fitness_list_in = [fitness_val_dict[i] for i in start_pop_list_in]
    start_pop_fitness_list_in, start_pop_list_in = (list(t) for t in zip(*sorted(zip(start_pop_fitness_list_in, start_pop_list_in), reverse=True)))

    gen_molecs_list = [start_pop_list_in]
    gen_molecs_fitness_list = [start_pop_fitness_list_in]

    for gen_num in range(max_gens):
        print(f"Generation {gen_num}")
        curr_gen_molecs_list, curr_gen_fitness_list = run_single_gen(gen_molecs_list[gen_num], gen_molecs_fitness_list[gen_num], fitness_val_dict, constraint_set, fr_to_keep, fr_parents)
        gen_molecs_list.append(curr_gen_molecs_list)
        gen_molecs_fitness_list.append(curr_gen_fitness_list)

    return gen_molecs_list, gen_molecs_fitness_list


n_gens = 150
fr_keep_per_gen = 0.2  ## elitist selection: fraction of population to keep per generation
fr_parents_per_gen = 0.5  ## fraction of population to use as parents per generation
n_runs = 20 ## independent runs to compare consistency

smi_fp_dict_path = "../data/alkanes_till_c17_canon_fp_2048_int8.p"
with open(smi_fp_dict_path, "rb") as f:
    smi_fp_dict = pickle.load(f)
    f.close()

smi_hl_nn_dict_path = "../pretrained_models/mlp_models/till_c16_train_c17_test/alk_smi_pred_all_norm.json"  ## normalized predictions to make the genetic algo plots more human readable
with open(smi_hl_nn_dict_path) as f:
    smi_hl_nn_dict = json.load(f)
    f.close()

smi_hl_chemprop_dict_path = "../pretrained_models/chemprop_models/till_c16_train_c17_test/alk_smi_pred_all_norm.json"  ## normalized predictions to make the genetic algo plots more human readable
with open(smi_hl_chemprop_dict_path) as f:
    smi_hl_chemprop_dict = json.load(f)
    f.close()


smi_all_list_in = []
with open("../data/alkanes_till_c17_canon.txt") as f:
    for line in f:
        smi_ = line.strip()
        if smi_.lower().count("c") == 17:
            smi_all_list_in.append(Chem.CanonSmiles(smi_))
    f.close()

smi_hl_ct_dict = set(smi_all_list_in)

### find the least stable 1000 c17 alkanes from nn and cp dicts. then take intersection to find common least stable.
# done to show that stability metric can be combined with genetic algo irrespective of initial populations' stability to find more stable molecules
nn_least_stable = set([k for k, _ in sorted(smi_hl_nn_dict.items(), key=lambda item: item[1]) if k.count("C") == 17][:2352])
cp_least_stable = set([k for k, _ in sorted(smi_hl_chemprop_dict.items(), key=lambda item: item[1]) if k.count("C") == 17][:2352])

least_stable = list(nn_least_stable.intersection(cp_least_stable))

init_alk_list_list = [list(np_rng_.choice(least_stable, size=100, replace=False)) for _ in range(n_runs)]


#  make random lists equal to runs
def run_nn_genetic(run_, init_alk_list_list, smi_hl_nn_dict, smi_hl_ct_dict, fr_keep_per_gen, fr_parents_per_gen, n_gens):
    print(f"NN Run Number {run_}")
    init_alk_list = init_alk_list_list[run_]
    gens_alks_list, gens_alks_nn_stability = run_genetic_n_gens(init_alk_list, smi_hl_nn_dict, smi_hl_ct_dict, fr_keep_per_gen, fr_parents_per_gen, n_gens)
    return [[i[0] for i in gens_alks_list], gens_alks_list]


def run_chemprop_genetic(run_, init_alk_list_list, smi_hl_chemprop_dict, smi_hl_ct_dict, fr_keep_per_gen, fr_parents_per_gen, n_gens):
    print(f"chemprop Run Number {run_}")
    init_alk_list = init_alk_list_list[run_]
    gens_alks_list, gens_alks_chemprop_stability = run_genetic_n_gens(init_alk_list, smi_hl_chemprop_dict, smi_hl_ct_dict, fr_keep_per_gen, fr_parents_per_gen, n_gens)
    return [[i[0] for i in gens_alks_list], gens_alks_list]


### parallel run: each loop runs parallel
mp_cores = len(os.sched_getaffinity(0))
print(f"Using {mp_cores} cores")
pool = mp.Pool(mp_cores)

nn_all_run_gen_best_result = pool.starmap_async(run_nn_genetic, [(run_, init_alk_list_list, smi_hl_nn_dict, smi_hl_ct_dict, fr_keep_per_gen, fr_parents_per_gen, n_gens) for run_ in range(n_runs)])
chemprop_all_run_gen_best_result = pool.starmap_async(run_chemprop_genetic, [(run_, init_alk_list_list, smi_hl_chemprop_dict, smi_hl_ct_dict, fr_keep_per_gen, fr_parents_per_gen, n_gens) for run_ in range(n_runs)])

nn_run_results = nn_all_run_gen_best_result.get()
chemprop_run_results = chemprop_all_run_gen_best_result.get()
nn_all_run_gen_best_list = [i[0] for i in nn_run_results]
nn_all_run_gen_list = [i[1] for i in nn_run_results]
chemprop_all_run_gen_best_list = [i[0] for i in chemprop_run_results]
chemprop_all_run_gen_list = [i[1] for i in chemprop_run_results]
pool.close()
pool.join()

with open("../data/genetic_mlp_run_best.p", "wb") as f:
    pickle.dump(nn_all_run_gen_best_list, f)

with open("../data/genetic_mlp_run_all.p", "wb") as f:
    pickle.dump(nn_all_run_gen_list, f)

with open("../data/genetic_chemprop_run_best.p", "wb") as f:
    pickle.dump(chemprop_all_run_gen_best_list, f)

with open("../data/genetic_chemprop_run_all.p", "wb") as f:
    pickle.dump(chemprop_all_run_gen_list, f)
