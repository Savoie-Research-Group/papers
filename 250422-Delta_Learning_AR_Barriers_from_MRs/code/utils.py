import os
import json
import numpy as np
from scipy.spatial import distance_matrix
from scipy.stats import linregress, gaussian_kde
import pickle
from copy import deepcopy
from tqdm import tqdm
import random
random.seed(42)
import shutil
import gc
from itertools import combinations, permutations
import hashlib

from drfp import DrfpEncoder

from dscribe.descriptors import MBTR
from ase.io import read as ase_read
from ase.constraints import FixAtoms

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import rdDetermineBonds
from e3fp.pipeline import fprints_from_mol

import matplotlib as mpl
from matplotlib import pyplot as plt
from matplotlib.legend_handler import HandlerPathCollection
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

import cupy as cp

import xgboost as xgb
from xgboost import XGBRegressor
from sklearn.preprocessing import StandardScaler

import torch
torch.set_float32_matmul_precision('high')
from lightning import pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from chemprop import data, featurizers, models, nn
from chemprop.data import ReactionDatapoint, ReactionDataset, build_dataloader, MulticomponentDataset
from chemprop.featurizers import CondensedGraphOfReactionFeaturizer


this_script_dir = os.path.dirname(os.path.realpath(__file__))
os.chdir(this_script_dir)


## path to data dir for the data paper for this work.
scratch_path = os.environ.get('SCRATCH')
home_path = os.environ.get('HOME')

data_paper_data_dir = os.path.join(this_script_dir, "../../250422-Actual_Reactions_from_Model_Reactions/data")  ## from savoiegr github

data_dir = os.path.join(this_script_dir, "../data")
analyses_dir = os.path.join(this_script_dir, "../analyses_and_plots")
models_dir = os.path.join(data_dir, "models")
preds_dir = os.path.join(data_dir, "preds")

random_seed = 42

## custom colors for plotting
vs_colors = {"blue": ["#002E6B", [0, 46, 107]], "red": ["#A11D4B", [161, 29, 75]],
             "green": ["#155914", [21, 89, 20]], "orange": ["#D15019", [209, 80, 25]],
             "purple": ["#3C063D", [60, 6, 61]]}


class AlphaHandler(HandlerPathCollection):
    def __init__(self, alpha=1.0, **kw):
        HandlerPathCollection.__init__(self, **kw)
        self.alpha = alpha
    def create_collection(self, orig_handle, sizes, offsets, transOffset):
        collection = HandlerPathCollection.create_collection(self, orig_handle, sizes, offsets, transOffset)
        collection.set_alpha(self.alpha)
        return collection


def ar_name_to_mr_name(ar_name):
    mr_name = "_".join(ar_name.split("_")[:3])
    if "rev" in ar_name:
        mr_name += "_rev"
    return mr_name


def splits_dict_to_combined_splits_dict(splits_dict):
    combined_splits_dict = {}
    
    for split, [train_list, val_list, test_list] in splits_dict.items():
        if all(isinstance(item, str) for item in train_list):
            ## single train list
            train_list_combined = train_list + [f"{mr_ar}_rev" for mr_ar in train_list]
        elif all(isinstance(item, list) for item in train_list):
            ## when we have multiple train lists (increasing percent of train data)
            train_list_combined = [train_list_i + [f"{mr_ar}_rev" for mr_ar in train_list_i] for train_list_i in train_list]
        else:
            raise ValueError("function splits_dict_to_combined_splits_dict: train_list in splits_dict should be a list of lists or a list of strings")
        val_list_combined = val_list + [f"{mr_ar}_rev" for mr_ar in val_list]
        test_list_combined = test_list + [f"{mr_ar}_rev" for mr_ar in test_list]
        random.shuffle(train_list_combined)
        random.shuffle(val_list_combined)
        random.shuffle(test_list_combined)
        combined_splits_dict[split] = [train_list_combined, val_list_combined, test_list_combined]
    
    return combined_splits_dict


def round_up_to_base(x, base=10):
    """
    Rounds a number up to the nearest multiple of a specified base. (https://stackoverflow.com/a/65725123)

    Parameters:
    - x (int or float): The number to be rounded.
    - base (int): The base to which the number will be rounded. Default is 10.

    Returns:
    - int or float: The smallest multiple of base that is greater than or equal to x.
    """
    # Calculate the remainder of x divided by base and subtract it from base
    # Add the result to x to round up to the nearest multiple of base
    return x + (-x % base)


def round_down_to_base(x, base=10):
    """
    Rounds a number down to the nearest multiple of a specified base. (https://stackoverflow.com/a/65725123)

    Parameters:
    - x (int or float): The number to be rounded.
    - base (int): The base to which the number will be rounded. Default is 10.

    Returns:
    - int or float: The largest multiple of base that is less than or equal to x.
    """
    # Calculate the remainder of x divided by base and subtract it from x
    # This rounds down x to the nearest multiple of base
    return x - (x % base)


def rxn_smi_to_ha_count(rxn_smi):
    """
    Counts the number of heavy atoms (HA) in a reaction SMILES string. Currently only works for CHON reactions.

    Parameters:
    - rxn_smi (str): The reaction SMILES string.

    Returns:
    - int: The number of heavy atoms in the reaction SMILES string.
    """
    ## only for chon and same number of atoms on both sides
    react_smi = rxn_smi.split(">>")[0]
    ha_count = react_smi.count("c") + react_smi.count("C") + react_smi.count("n") + react_smi.count("N") + react_smi.count("o") + react_smi.count("O")
    return ha_count


def e_csv_to_dict(e_csv):
    e_dict = {}
    with open(e_csv, "r") as f:
        for line in f:
            ssl = line.strip().split(",")
            try:
                e_dict[ssl[0]] = [float(x) for x in ssl[1:]]
            except:
                e_dict[ssl[0]] = ssl[1:]
    return e_dict


def get_default_g_mr_ar_dicts():
    mr_smi_dict = json.load(open(os.path.join(data_paper_data_dir, "mr_smi_dict.json"), "r"))
    ar_smi_dict = json.load(open(os.path.join(data_paper_data_dir, "ar_smi_dict.json"), "r"))
    
    react_prod_smi_e_dict = e_csv_to_dict(os.path.join(data_paper_data_dir, "react_prod_smi_energy_list.csv"))
    
    ar_hrxn_fwd_dict, ar_hrxn_rev_dict, ar_grxn_fwd_dict, ar_grxn_rev_dict = create_mr_ar_erxn_dicts(ar_smi_dict, react_prod_smi_e_dict)
    mr_hrxn_fwd_dict, mr_hrxn_rev_dict, mr_grxn_fwd_dict, mr_grxn_rev_dict = create_mr_ar_erxn_dicts(mr_smi_dict, react_prod_smi_e_dict)
    
    mr_ar_hrxn_fwd_dict = deepcopy(mr_hrxn_fwd_dict)
    mr_ar_hrxn_fwd_dict.update(ar_hrxn_fwd_dict)
    mr_ar_hrxn_rev_dict = deepcopy(mr_hrxn_rev_dict)
    mr_ar_hrxn_rev_dict.update(ar_hrxn_rev_dict)
    mr_ar_hrxn_combined_dict = deepcopy(mr_ar_hrxn_fwd_dict)
    mr_ar_hrxn_combined_dict.update({f"{k}_rev": v for k, v in mr_ar_hrxn_rev_dict.items()})
    
    mr_ar_grxn_fwd_dict = deepcopy(mr_grxn_fwd_dict)
    mr_ar_grxn_fwd_dict.update(ar_grxn_fwd_dict)
    mr_ar_grxn_rev_dict = deepcopy(mr_grxn_rev_dict)
    mr_ar_grxn_rev_dict.update(ar_grxn_rev_dict)
    mr_ar_grxn_combined_dict = deepcopy(mr_ar_grxn_fwd_dict)
    mr_ar_grxn_combined_dict.update({f"{k}_rev": v for k, v in mr_ar_grxn_rev_dict.items()})
    
    return mr_ar_hrxn_fwd_dict, mr_ar_hrxn_rev_dict, mr_ar_hrxn_combined_dict, mr_ar_grxn_fwd_dict, mr_ar_grxn_rev_dict, mr_ar_grxn_combined_dict


def get_rxn_drfp(rxn_smi_list):
    """
    Generates Differential Reaction Fingerprints (DRFP) for a list of reaction SMILES strings.

    DRFP is a method to encode reaction information into a numerical fingerprint, as described in the article:
    https://doi.org/10.1039/D1DD00006C

    Parameters:
    - rxn_smi_list (list of str): A list of reaction SMILES strings. These can be either atom-mapped or regular SMILES.

    Returns:
    - list of np.ndarray: A list of DRFPs for each reaction SMILES in the input list.
    """
    # Encode the reaction SMILES into DRFPs, with progress displayed and hydrogens included
    drfp_list = DrfpEncoder.encode(rxn_smi_list, show_progress_bar=True, include_hydrogens=True)
    drfp_list = [i.astype('float32') for i in drfp_list]
    
    return drfp_list


def get_mbtr_from_xyz(xyz_path, degree_list=[1, 2, 3], species=["H", "C", "N", "O"]):
    # Many Body Tensor Representation (MBTR) descriptor from the Dscribe package.
    # package link: https://singroup.github.io/dscribe/stable/tutorials/descriptors/mbtr.html
    # original paper link: https://doi.org/10.1088/2632-2153/aca005
    
    # degree_list=[2]
    
    degree_set = set([int(i) for i in degree_list])
    mbtr_list = []
    ase_system = ase_read(xyz_path)
    ase_system.set_constraint(FixAtoms(indices=range(len(ase_system))))
    
    if 1 in degree_set:
        desc_1 = MBTR(
            species=species,
            geometry={"function": "atomic_number"},
            grid={"min": 0, "max": 8, "n": 100, "sigma": 0.01},
            weighting={"function": "unity"},
            normalize_gaussians=True,
            normalization="l2",
            periodic=False,
            sparse=False,
            dtype='float32'
            )
        mbtr_1 = desc_1.create(ase_system, n_jobs=-1, only_physical_cores=True, verbose=True)
        mbtr_list.append(mbtr_1.astype('float32'))
        
    if 2 in degree_set:
        desc_2 = MBTR(
            species=species,
            geometry={"function": "inverse_distance"},
            grid={"min": 0.0, "max": 1.5, "n": 200, "sigma": 0.01},
            weighting={"function": "exp", "scale": 0.5, "threshold": 1e-4},
            normalize_gaussians=True,
            normalization="l2",
            periodic=False,
            sparse=False,
            dtype='float32'
            )
        mbtr_2 = desc_2.create(ase_system, n_jobs=-1, only_physical_cores=True, verbose=True)
        mbtr_list.append(mbtr_2.astype('float32'))
    
    if 3 in degree_set:
        desc_3 = MBTR(
            species=species,
            geometry={"function": "cosine"},
            grid={"min": -1.0, "max": 1.0, "n": 200, "sigma": 0.01},
            weighting={"function": "exp", "scale": 0.5, "threshold": 1e-6},
            normalize_gaussians=True,
            normalization="l2",
            periodic=False,
            sparse=False,
            dtype='float32'
            )
        mbtr_3 = desc_3.create(ase_system, n_jobs=-1, only_physical_cores=True, verbose=True)
        mbtr_list.append(mbtr_3.astype('float32'))
        
    return np.concatenate(mbtr_list).astype('float32')


def get_e3fp_from_xyz(xyz_path):
    """
    Generates an extended three-dimensional (E3FP) fingerprint from an XYZ file.

    This function reads a molecular structure from an XYZ file, processes it to 
    determine connectivity and update properties, and then generates an E3FP 
    fingerprint vector using specified parameters.

    Args:
        xyz_path (str): The file path to the XYZ file containing the molecular structure.

    Returns:
        np.ndarray: A dense NumPy array representing the E3FP fingerprint with dtype 'float32'.

    References:
        - [E3FP Paper](https://doi.org/10.1021/acs.jmedchem.7b00696)
        - [E3FP GitHub repository](https://github.com/keiserlab/e3fp)
    """

    mol = Chem.MolFromXYZFile(xyz_path)
    mol.SetProp("_Name", "mol")
    rdDetermineBonds.DetermineConnectivity(mol)
    mol.UpdatePropertyCache(strict=False)
    
    fprint_params = {'bits': 2048,\
                    'level': 5,\
                    'first': 1,\
                    'radius_multiplier': 1.718,\
                    'stereo': True,\
                    'counts': False,\
                    'include_disconnected': True,\
                    'rdkit_invariants': False,\
                    'remove_duplicate_substructs': True,\
                    'exclude_floating': False}
    
    e3fp = fprints_from_mol(mol, fprint_params=fprint_params)[0].to_vector(sparse=False, dtype='float32')
    
    return e3fp


class GeometricFingerprint:
    ## custom geo fp class
    
    def __init__(self, size=4096, dtype=bool, max_neighbors=4):
        """
        Initializes a GeometricFingerprint object.

        Args:
            size (int, optional): The size of the fingerprint. Defaults to 4096.
            dtype (type, optional): The data type of the fingerprint. Defaults to bool.
            max_neighbors (int, optional): The maximum number of neighbors to consider. Defaults to 4.
        """
        
        self.size = size
        self.dtype = dtype
        self.max_neighbors = max_neighbors
    
    
    @staticmethod
    def read_xyz(xyz_file):
        """
        Reads an XYZ file and processes the molecular structure to ensure 
        invariance to translation, rotation, and permutation.

        This method performs the following steps on the molecule:
        1. Centers the molecule's coordinates to the origin.
        2. Aligns the molecule to its principal axes.
        3. Rearranges atoms in order of increasing distance from the origin.

        Args:
            xyz_file (str): Path to the XYZ file containing the molecular structure.

        Returns:
            tuple: A tuple containing:
                - coords (np.ndarray): The processed coordinates of the atoms.
                - atomic_nums (list of int): The atomic numbers of the atoms.
                - atoms (list of str): The symbols of the atoms.
        """

        mol = Chem.MolFromXYZFile(xyz_file)
        coords = mol.GetConformer().GetPositions()
        
        # center to origin
        coords -= np.mean(coords, axis=0)
        
        # align to principle axes
        inertia = np.dot(coords.T, coords)
        eigenvals, eigenvecs = np.linalg.eigh(inertia)
        eigenvecs = eigenvecs[:, np.argsort(eigenvals)[::-1]]
        coords = np.dot(coords, eigenvecs)
        
        # ensure right-handed coordinate system
        if np.linalg.det(eigenvecs) < 0:
            eigenvecs[:, 2] *= -1
        
        # rearrange atoms in increasing distance from the origin
        dists = np.linalg.norm(coords, axis=1)
        dists_order = np.argsort(dists)
        coords = coords[dists_order]
        atoms = [mol.GetAtomWithIdx(int(i)).GetSymbol() for i in dists_order]
        atomic_nums = [mol.GetAtomWithIdx(int(i)).GetAtomicNum() for i in dists_order]
        
        return coords, atomic_nums, atoms
    
    
    @staticmethod
    def get_hash(value, size):
        ## use sha256 hash of value to get a deterministic hash value in [0, size)
        """
        Gets a deterministic hash value for the given value in the range [0, size).

        Args:
            value (any): The value to hash.
            size (int): The range of the hash value.

        Returns:
            int: The hashed value.
        """
        
        return int.from_bytes(hashlib.sha256(str(value).encode('utf8')).digest(), 'big') % size
    
    
    @staticmethod
    def bin_value(value, bins, min_val, max_val):
        """
        Bins a given value into one of the given bins.

        Args:
            value (float): The value to bin.
            bins (int): The number of bins.
            min_val (float): The minimum value of the range.
            max_val (float): The maximum value of the range.

        Returns:
            int: The index of the bin that the value falls into.
        """
        
        if value <= min_val:
            return 0
        if value >= max_val:
            return bins - 1
        bin_width = (max_val - min_val) / bins
        return int((value - min_val) / bin_width)
    
    
    def compute_fp(self, coords, atomic_nums):
        """
        Computes the geometric fingerprint of the given molecule.

        This method computes a geometric fingerprint of the given molecule based on
        the atomic positions and types. The fingerprint is a binary vector of length
        self.size, where each entry corresponds to a particular feature of the molecule.
        The features are encoded as follows:

        - The atomic number of the atom.
        - The inverse distance of the atom to its closest neighbor.
        - The cosine of the angle between the atom and its closest two neighbors.
        - The cosine of the dihedral angle between the atom and its closest three neighbors.

        The fingerprint is computed by iterating over all atoms in the molecule and
        computing the above features for each atom. The features are then hashed to
        obtain a unique index in the fingerprint vector, and the corresponding entry
        in the vector is set to 1.

        Args:
            coords (np.ndarray): The 3D coordinates of the atoms in the molecule.
            atomic_nums (list of int): The atomic numbers of the atoms in the molecule.

        Returns:
            np.ndarray: The geometric fingerprint of the molecule.
        """
        
        if self.dtype == bool:
            fp = np.zeros(self.size, dtype=self.dtype)
            update_fp = lambda fp, idx: fp.__setitem__(idx, True)
        else:
            fp = np.zeros(self.size, dtype=self.dtype)
            update_fp = lambda fp, idx: fp.__setitem__(idx, fp[idx] + 1)
        
        dist_mat = distance_matrix(coords, coords)
        
        n_neighbors = min(self.max_neighbors, len(atomic_nums) - 1)
        
        for i, (coord, atomic_n) in enumerate(zip(coords, atomic_nums)):
            # include info about position and atomic number
            update_fp(fp, self.get_hash((i, atomic_n), self.size))  # encoding atomic potion to atomic number (i.e. atom type)
            update_fp(fp, self.get_hash((atomic_n), self.size))
            
            # get the indices of the closest atoms
            neighbor_indices = np.argsort(dist_mat[i])[1:n_neighbors+1]
            
            # distance features (inverse distance since it is self normalized)
            for j_idx, j in enumerate(neighbor_indices):
                dist = dist_mat[i, j]
                inv_dist = 1.0 / dist  # inverse distance. assuming min distance to be 0.65 angstrom. max inverse distance is 1.5.
                
                ## bin inverse distance. min = 0, max = 1.5. bins = 100 (assuming a max variation of ~0.1 angstroms at a distance of ~2.5 angstroms)
                inv_dist = self.bin_value(inv_dist, 100, 0, 1.5)
                
                # update_fp(fp, self.get_hash((i, j, inv_dist), self.size))  # encoding pairwise atomic positions and inverse distance. i.e. a type of global feature.
                update_fp(fp, self.get_hash((atomic_n, atomic_nums[j], inv_dist), self.size))  # encoding atom type pairs with inverse distance. i.e. a type of local substructural feature
                
                # bond angle features (cosine of angle since it is self normalized.)
                for k_idx, k in enumerate(neighbor_indices[j_idx+1:]):
                    vec1 = coords[j] - coords[i]
                    vec2 = coords[k] - coords[i]
                    cos_angle = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))  ## differentiate between acute and obtuse. no abs.
                    
                    ## bin cosine angle. min = -1, max = 1. bins = 140 (assuming a max variation of ~10 degrees at ~cosine(180 degrees))
                    cos_angle = self.bin_value(cos_angle, 140, -1, 1)
                    
                    # update_fp(fp, self.get_hash((i, j, k, cos_angle), self.size))  # encoding location and cosine of bond angle. i.e. a type of global feature.
                    update_fp(fp, self.get_hash((atomic_n, min(atomic_nums[j], atomic_nums[k]), max(atomic_nums[j], atomic_nums[k]), cos_angle), self.size))  # encoding atom type pairs with cosine of bond angle. i.e. a type of local substructural feature
                    
                    # dihedral angle features (cosine of angle since it is self normalized. taking abs of cos to ensure the smallest angle is always used)
                    for l_idx, l in enumerate(neighbor_indices[j_idx+1:][k_idx+1:]):
                        vec1_0 = coords[j] - coords[i]
                        vec2_0 = coords[i] - coords[l]
                        vec3_0 = coords[l] - coords[k]
                        
                        vec1_1 = coords[k] - coords[i]
                        vec2_1 = coords[i] - coords[j]
                        vec3_1 = coords[j] - coords[l]
                        
                        vec1_2 = coords[l] - coords[i]
                        vec2_2 = coords[i] - coords[k]
                        vec3_2 = coords[k] - coords[j]
                        
                        normal1_0 = np.cross(vec1_0, vec2_0)
                        normal2_0 = np.cross(vec2_0, vec3_0)
                        
                        normal1_1 = np.cross(vec1_1, vec2_1)
                        normal2_1 = np.cross(vec2_1, vec3_1)
                        
                        normal1_2 = np.cross(vec1_2, vec2_2)
                        normal2_2 = np.cross(vec2_2, vec3_2)
                        
                        cos_angle_0 = abs(np.dot(normal1_0, normal2_0) / (np.linalg.norm(normal1_0) * np.linalg.norm(normal2_0)))
                        cos_angle_1 = abs(np.dot(normal1_1, normal2_1) / (np.linalg.norm(normal1_1) * np.linalg.norm(normal2_1)))
                        cos_angle_2 = abs(np.dot(normal1_2, normal2_2) / (np.linalg.norm(normal1_2) * np.linalg.norm(normal2_2)))
                        
                        ## bin abs cosine angle. min = 0, max = 1. bins = 70 (assuming a max variation of ~10 degrees at ~cosine(0 degrees))
                        cos_angle_0 = self.bin_value(cos_angle_0, 70, 0, 1)
                        cos_angle_1 = self.bin_value(cos_angle_1, 70, 0, 1)
                        cos_angle_2 = self.bin_value(cos_angle_2, 70, 0, 1)
                        
                        cos_angle_0, cos_angle_1, cos_angle_2 = sorted([cos_angle_0, cos_angle_1, cos_angle_2], reverse=True)
                        an1, an2, an3 = sorted([atomic_nums[j], atomic_nums[k], atomic_nums[l]])
                        
                        # update_fp(fp, self.get_hash((i, j, k, l, cos_angle_0, cos_angle_1, cos_angle_2), self.size))  # encoding location and cosine of dihedral angle. i.e. a type of global feature
                        update_fp(fp, self.get_hash((atomic_n, an1, an2, an3, cos_angle_0, cos_angle_1, cos_angle_2), self.size))  # encoding atom type pairs with cosine of dihedral angle. i.e. a type of local substructural feature
        
        return fp
    
    def compute_fp_from_xyz(self, xyz_file):
        """
        Computes the geometric fingerprint from a .xyz file.

        Parameters:
            xyz_file (str): The path to the .xyz file containing the molecule to be fingerprinted.

        Returns:
            np.ndarray: The computed geometric fingerprint.
        """

        coords, atomic_nums, _ = self.read_xyz(xyz_file)
        return self.compute_fp(coords, atomic_nums)


def filter_mr_ar_list(mr_ar_list):
    """
    Filters a list of model reactions and actual reactions to generate a list of only MR and another list of only AR.

    Parameters:
    - mr_ar_list (list of str): A list of strings, each representing a model reaction (MR) or actual reaction (AR).

    Returns:
    - filtered_mr_list (list of str): A list of strings, each representing a model reaction (MR).
    - filtered_ar_list (list of str): A list of strings, each representing an actual reaction (AR).
    """
    
    filtered_mr_list = []
    filtered_ar_list = []
    for mr_ar in mr_ar_list:
        if "rev" in mr_ar:
            if mr_ar.count("_") == 3:
                ## mr_abcd_e_rev
                filtered_mr_list.append(mr_ar)
            elif mr_ar.count("_") == 6 or mr_ar.count("_") == 8:
                ## again similar since a _rev is added
                filtered_ar_list.append(mr_ar)
            else:
                # Should not happen. Just a sanity check.
                print(f"Error: mr_ar {mr_ar} has unexpected number of underscores")
        else:    
            if mr_ar.count("_") == 2:
                # Model reactions should have 2 underscores.
                filtered_mr_list.append(mr_ar)
            elif mr_ar.count("_") == 5 or mr_ar.count("_") == 7:
                # Actual reactions should have 5 or 7 underscores depending on whetehr one or two fn grps have been added.
                filtered_ar_list.append(mr_ar)
            else:
                # Should not happen. Just a sanity check.
                print(f"Error: mr_ar {mr_ar} has unexpected number of underscores")
    return filtered_mr_list, filtered_ar_list


def create_mr_ar_ea_dicts(mr_smi_dict, mr_ts_e_dict, ar_smi_dict, ar_ts_e_dict, react_prod_smi_e_dict):
    # mr: model reaction
    # ar: actual reaction
    # ts: transition state
    
    """
    Creates dictionaries of free energies of activation for model reactions and actual reactions.

    Parameters:
    - mr_smi_dict (dict): A dictionary where the keys are model reaction IDs and the values are strings of reactant and product SMILES. The reactant and product SMILES are separated by a ">>".
    - mr_ts_e_dict (dict): A dictionary where the keys are model reaction IDs and the values are lists of [h, g, spe] in Hartree.
    - ar_smi_dict (dict): A dictionary where the keys are actual reaction IDs and the values are strings of reactant and product SMILES. The reactant and product SMILES are separated by a ">>".
    - ar_ts_e_dict (dict): A dictionary where the keys are actual reaction IDs and the values are lists of [h, g, spe] in Hartree.
    - react_prod_smi_e_dict (dict): A dictionary where the keys are reactant/product SMILES and the values are lists of [h, g, spe] in Hartree.

    Returns:
    - mr_ea_fwd_dict (dict): A dictionary where the keys are model reaction IDs and the values are forward free energies of activation in kcal/mol.
    - mr_ea_rev_dict (dict): A dictionary where the keys are model reaction IDs and the values are reverse free energies of activation in kcal/mol.
    - ar_ea_fwd_dict (dict): A dictionary where the keys are actual reaction IDs and the values are forward free energies of activation in kcal/mol.
    - ar_ea_rev_dict (dict): A dictionary where the keys are actual reaction IDs and the values are reverse free energies of activation in kcal/mol.
    """
    
    hartree2kcalpermol = 627.5  ## Conversion factor from Hartree to kcal/mol
    
    mr_ea_fwd_dict = {}
    mr_ea_rev_dict = {}
    
    ar_ea_fwd_dict = {}
    ar_ea_rev_dict = {}
    
    for mr, mr_smi in mr_smi_dict.items():
        ts_f = mr_ts_e_dict[mr][1]  ## free energy of transition state, Hartree
        
        r_f = 0  ## free energy of reactants, Hartree
        p_f = 0  ## free energy of products, Hartree
        r_smi, p_smi = mr_smi.split(">>")
        for smi in r_smi.split("."):
            r_f += react_prod_smi_e_dict[smi][1]
        for smi in p_smi.split("."):
            p_f += react_prod_smi_e_dict[smi][1]
        
        ea_fwd = (ts_f - r_f) * hartree2kcalpermol  ## forward free energy of activation, kcal/mol
        ea_rev = (ts_f - p_f) * hartree2kcalpermol  ## reverse free energy of activation, kcal/mol
        mr_ea_fwd_dict[mr] = ea_fwd
        mr_ea_rev_dict[mr] = ea_rev
    
    for ar, ar_smi in ar_smi_dict.items():
        ts_f = ar_ts_e_dict[ar][1]
        
        r_f = 0
        p_f = 0
        r_smi, p_smi = ar_smi.split(">>")
        for smi in r_smi.split("."):
            r_f += react_prod_smi_e_dict[smi][1]
        for smi in p_smi.split("."):
            p_f += react_prod_smi_e_dict[smi][1]
        
        ea_fwd = (ts_f - r_f) * hartree2kcalpermol
        ea_rev = (ts_f - p_f) * hartree2kcalpermol
        ar_ea_fwd_dict[ar] = ea_fwd
        ar_ea_rev_dict[ar] = ea_rev
    
    return mr_ea_fwd_dict, mr_ea_rev_dict, ar_ea_fwd_dict, ar_ea_rev_dict


def create_mr_ar_erxn_dicts(mr_ar_smi_dict, react_prod_smi_e_dict):
    """
    Calculate the forward and reverse heats and free energies of reaction for 
    both model and actual reactions.
    
    Parameters
    ----------
    mr_ar_smi_dict: dict
        Dictionary where keys are model reaction or actual reaction (mr_ar) 
        names and values are the corresponding SMILES strings.
        
    react_prod_smi_e_dict: dict
        Dictionary where keys are reactant or product SMILES strings and values 
        are tuples of their Enthalpy, Gibbs free energy, and Single Point Energy, in Hartree.
        
    Returns
    -------
    mr_ar_hrxn_fwd_dict, mr_ar_hrxn_rev_dict, mr_ar_grxn_fwd_dict, mr_ar_grxn_rev_dict: dict
        Dictionaries where keys are model reaction or actual reaction (mr_ar) 
        names and values are the forward and reverse heats and free energies of 
        reaction, in kcal/mol.
    """
    
    hartree2kcalpermol = 627.5  ## Conversion factor from Hartree to kcal/mol
    
    mr_ar_hrxn_fwd_dict = {}
    mr_ar_grxn_fwd_dict = {}
    
    for mr_ar, mr_ar_smi in mr_ar_smi_dict.items():
        r_h, p_h, r_g, p_g = 0.0, 0.0, 0.0, 0.0
        r_smi, p_smi = mr_ar_smi.split(">>")
        for smi in r_smi.split("."):
            r_h += react_prod_smi_e_dict[smi][0]
            r_g += react_prod_smi_e_dict[smi][1]
        for smi in p_smi.split("."):
            p_h += react_prod_smi_e_dict[smi][0]
            p_g += react_prod_smi_e_dict[smi][1]
        
        mr_ar_hrxn_fwd_dict[mr_ar] = (p_h - r_h) * hartree2kcalpermol
        mr_ar_grxn_fwd_dict[mr_ar] = (p_g - r_g) * hartree2kcalpermol
        
    mr_ar_hrxn_rev_dict = {k: -v for k, v in mr_ar_hrxn_fwd_dict.items()}
    mr_ar_grxn_rev_dict = {k: -v for k, v in mr_ar_grxn_fwd_dict.items()}
        
    return mr_ar_hrxn_fwd_dict, mr_ar_hrxn_rev_dict, mr_ar_grxn_fwd_dict, mr_ar_grxn_rev_dict


def create_smi_ea_csv_list(mr_ar_ea_dict, mr_ar_smi_dict):
    """
    Creates a list of strings in CSV format representing the activation energies
    for model and actual reactions.

    Parameters:
    - mr_ar_ea_dict (dict): A dictionary where keys are reaction identifiers (AR or MR) 
      and values are their corresponding activation energies.
    - mr_ar_smi_dict (dict): A dictionary where keys are reaction identifiers (AR or MR) 
      and values are their corresponding SMILES strings.

    Returns:
    - list of str: A list containing strings in the format "smi,ea", where 'smi' is 
      the SMILES representation of the reaction and 'ea' is the activation energy.
    """

    smi_ea_csv_list = ["smi,ea"]
    
    for mr_ar, ea in mr_ar_ea_dict.items():
        smi = mr_ar_smi_dict[mr_ar]
        smi_ea_csv_list.append(f"{smi},{ea}")
    
    return smi_ea_csv_list


def create_delta_ea_dict(mr_ar_ea_dict):
    """
    Creates a dictionary of delta activation energies (EA) for reactions.
    In general, may be used to create delta energy dicts.

    The delta EA is calculated as the difference between the actual reaction EA (ar_ea) 
    and the model reaction EA (mr_ea). For model reactions themselves, the delta EA is zero.

    Parameters:
    - mr_ar_ea_dict (dict): A dictionary where keys are reaction identifiers (AR or MR) 
      and values are their corresponding activation energies.

    Returns:
    - dict: A dictionary with the same reaction identifiers as keys and their delta 
      activation energies as values.
    """
    mr_ar_delta_ea_dict = {}  # Initialize the dictionary to store delta EAs. delta_ea = ar_ea - mr_ea. ar_ea = mr_ea + delta_ea
    mr_miss_list = []
    for mr_ar, ea in mr_ar_ea_dict.items():
        mr = ar_name_to_mr_name(mr_ar)
        try:
            mr_ar_delta_ea_dict[mr_ar] = ea - mr_ar_ea_dict[mr]  # Calculate delta EA
        except:
            # if "_rev" not in mr:
            #     mr_miss_list.append(mr)
            #     print(f"Error: {mr} not found in mr_ar_ea_dict")
            pass
    # print(mr_miss_list)
    
    return mr_ar_delta_ea_dict


def delta_pred_to_ea_dict(mr_ar_delta_pred_dict, mr_ar_ea_dict):
    """
    Converts a dictionary of predicted delta EA values to a dictionary of predicted EA values.
    In general, may be used to convert delta energy dicts to energy dicts.

    Parameters:
    - mr_ar_delta_pred_dict (dict): A dictionary where keys are reaction identifiers (AR or MR) 
      and values are their corresponding predicted delta EA values.
    - mr_ar_ea_dict (dict): A dictionary where keys are reaction identifiers (AR or MR) 
      and values are their corresponding activation energies.

    Returns:
    - dict: A dictionary with the same reaction identifiers as keys and their predicted activation energies as values.
    """
    
    ea_pred_dict = {}
    
    for mr_ar, delta_ea in mr_ar_delta_pred_dict.items():
        mr = ar_name_to_mr_name(mr_ar)  ## will also work if mr_ar is mr.
        # Calculate EA as the sum of the predicted delta EA and the EA of the corresponding model reaction
        ea_pred_dict[mr_ar] = mr_ar_ea_dict[mr] + delta_ea
    
    return ea_pred_dict


def create_fp1_mr_fp2_concat_dict(mr_ar_fp1_dict, mr_ar_fp2_dict):
    ## concat fp2 of the corresponding mr to the fp1 of each rxn.
    
    mr_fp2_concat_fp1_dict = {}
    for mr_ar, fp1 in mr_ar_fp1_dict.items():
        mr = ar_name_to_mr_name(mr_ar)
        mr_fp2 = mr_ar_fp2_dict[mr]
        
        try:
            mr_fp2_concat_fp1_dict[mr_ar] = np.concatenate((fp1.astype('float32'), mr_fp2.astype('float32')))
        except:
            # mr_fp2_concat_fp1_dict[mr_ar] = fp1 + mr_fp2
            raise ValueError("fp dict values should be numpy ndarrays")
    
    return mr_fp2_concat_fp1_dict


def create_mr_concat_fp_dict(mr_ar_fp_dict):
    return create_fp1_mr_fp2_concat_dict(mr_ar_fp_dict, mr_ar_fp_dict)


def create_mr_ea_concat_fp_dict(mr_ar_fp_dict, mr_ar_ea_dict):
    return create_fp1_mr_fp2_concat_dict(mr_ar_fp_dict, {k:np.array([v]) for k, v in mr_ar_ea_dict.items()})


def return_stats(y, y_pred):
    abs_err_list = [abs(x-y) for x, y in zip(y, y_pred)]
    stats_dict = {
        "r2": np.corrcoef(y, y_pred)[0, 1]**2,
        "median_ae": np.median(abs_err_list),
        "mae": np.mean(abs_err_list),
        "rmse": np.sqrt(np.mean([abs_err**2 for abs_err in abs_err_list])),
    }
    return stats_dict


def print_train_test_stats(y_train, y_train_pred, y_test, y_test_pred):
    stats_dict_train = return_stats(y_train, y_train_pred)
    stats_dict_test = return_stats(y_test, y_test_pred)
    
    r2_train = stats_dict_train["r2"]
    median_ae_train = stats_dict_train["median_ae"]
    mae_train = stats_dict_train["mae"]
    rmse_train = stats_dict_train["rmse"]
    
    r2_test = stats_dict_test["r2"]
    median_ae_test = stats_dict_test["median_ae"]
    mae_test = stats_dict_test["mae"]
    rmse_test = stats_dict_test["rmse"]
    
    print(f"R^2 | train: {r2_train} | test: {r2_test}")
    print(f"Median AE | train: {median_ae_train} | test: {median_ae_test}")
    print(f"MAE | train: {mae_train} | test: {mae_test}")
    print(f"RMSE | train: {rmse_train} | test: {rmse_test}")
    print("\n")
    return


def train_pred_xgb_regressor(X_train, y_train, X_test, y_test, random_state=random_seed, params=None):
    ## return model, y_train_pred, y_test_pred
    
    nproc = len(os.sched_getaffinity(0)) - 1
    
    if torch.cuda.is_available():
        device = "cuda"
        print("Training XGB on GPU...")
    else:
        device = "cpu"
        print("Training XGB on CPU...")
    
    if params is None:
        params = {
            "objective": "reg:squarederror",
            "eval_metric": "rmse",
            "n_estimators": 500,  # 100–1000
            "verbosity": 2,
            "max_depth": 10,  # 3–10
            "eta": 0.1,  # 0.01–0.3
            "subsample": 0.8,  # 0.7–1.0
            "colsample_bytree": 0.8,  # 0.5–1.0
            "nthread": nproc,
            "reg_alpha": 0.1,  # 0-1
            "reg_lambda": 1,  # 0–10
            "min_child_weight": 10,  # 1–10
            "seed": random_state,
            
            "tree_method": "hist",
            "device": device,
            "max_bin": 256,
        }
    else:
        if 'seed' not in params:
            params['seed'] = random_state
        params['nthread'] = nproc
        params['verbosity'] = 2 ## force verbosity to 2.
        params['device'] = device
        
    print(params)
    
    if device == "cuda":
        X_train = cp.asarray(X_train)
        y_train = cp.asarray(y_train)
        X_test = cp.asarray(X_test)
        y_test = cp.asarray(y_test)
    
    xgb_reg = XGBRegressor(**params)
    xgb_reg.fit(X_train, y_train)
    
    y_train_pred = xgb_reg.predict(X_train)
    y_test_pred = xgb_reg.predict(X_test)
    
    if device == "cuda":
        X_train = cp.asnumpy(X_train)
        y_train = cp.asnumpy(y_train)
        X_test = cp.asnumpy(X_test)
        y_test = cp.asnumpy(y_test)
        y_train_pred = cp.asnumpy(y_train_pred)
        y_test_pred = cp.asnumpy(y_test_pred)
    
    return xgb_reg, y_train_pred, y_test_pred


def pred_xgb_regressor(X, xgb_reg=None, xgb_reg_path=None):
    if xgb_reg is None and xgb_reg_path is None:
        raise ValueError("Either xgb_reg or xgb_reg_path must be provided.")
    
    nproc = len(os.sched_getaffinity(0))
    if torch.cuda.is_available():
        device = "cuda"
        print("Predicting XGB on GPU...")
    else:
        device = "cpu"
        print("Predicting XGB on CPU...")
    
    if xgb_reg is None:
        xgb_reg = XGBRegressor()
        xgb_reg.load_model(xgb_reg_path)
    
    xgb_reg.set_params(nthread=nproc, device=device)
    
    if device == "cuda":
        X = cp.asarray(X)
        
    y_pred = xgb_reg.predict(X)
    
    if device == "cuda":
        X = cp.asnumpy(X)
        y_pred = cp.asnumpy(y_pred)
        
    return y_pred


def create_chemprop_reaction_datapoint_list(
    mr_ar_list, mr_ar_smis_dict, mr_ar_ea_dict, mr_e_rxn_dict=None,
    concat_mr_feature=False, append_mr_ea_feature=False, append_mr_e_rxn_feature=False, delta_ea=False,
    mr_ar_ea_dict_extra_for_append_mr_ea_feature=None
):
    """
    Creates a list of chemprop ReactionDatapoint objects from the given inputs.
    
    Parameters
    ----------
    mr_ar_list : list
        A list of reaction names for which to create datapoints.
    mr_ar_smis_dict : dict
        A dictionary mapping the reaction atom mapped smiles to the reaction
        smiles.
    mr_ar_ea_dict : dict
        A dictionary mapping the reaction smiles to the reaction EA.
    concat_mr_feature : bool, optional
        If True, the model reaction smiles are concatenated to the
        reaction smiles as a feature for the delta models. Default is False.
    append_mr_ea_feature : bool, optional
        If True, the model reaction EA is appended as a feature to the reaction
        smiles. Default is False.
    delta_ea : bool, optional
        If True, the y values are the delta EA. If False, the y values are the
        EA. Default is False.
    
    Returns
    -------
    reaction_datapoint_list : list
        A list of chemprop ReactionDatapoint objects.
    """
    
    if append_mr_e_rxn_feature:  # (heat or free anergy of reaction)
        assert mr_e_rxn_dict is not None, "mr_e_rxn_dict should be provided if append_mr_e_rxn_feature is True."
    
    mr_ar_smi_list = [mr_ar_smis_dict[mr_ar] for mr_ar in mr_ar_list]
    
    mr_ar_ea_list = [mr_ar_ea_dict[mr_ar] for mr_ar in mr_ar_list]
    
    x_d_list = []
    if append_mr_e_rxn_feature and append_mr_ea_feature:
        for mr_ar in mr_ar_list:
            mr = ar_name_to_mr_name(mr_ar)
            if mr_ar_ea_dict_extra_for_append_mr_ea_feature is not None:
                x_d_list.append(np.concatenate((np.array([mr_ar_ea_dict_extra_for_append_mr_ea_feature[mr]]), np.array([mr_e_rxn_dict[mr]]))))
            else:
                x_d_list.append(np.concatenate((np.array([mr_ar_ea_dict[mr]]), np.array([mr_e_rxn_dict[mr]]))))
    elif append_mr_e_rxn_feature:
        for mr_ar in mr_ar_list:
            mr = ar_name_to_mr_name(mr)
            x_d_list.append(np.array([mr_e_rxn_dict[mr]]))
    elif append_mr_ea_feature:
        for mr_ar in mr_ar_list:
            mr = ar_name_to_mr_name(mr_ar)
            if mr_ar_ea_dict_extra_for_append_mr_ea_feature is not None:
                x_d_list.append(np.array([mr_ar_ea_dict_extra_for_append_mr_ea_feature[mr]]))
            else:
                x_d_list.append(np.array([mr_ar_ea_dict[mr]]))
    else:
        pass
    
    if delta_ea:
        mr_ar_delta_ea_dict = create_delta_ea_dict(mr_ar_ea_dict)
        mr_ar_delta_ea_list = [mr_ar_delta_ea_dict[mr_ar] for mr_ar in mr_ar_list]
        y_list = mr_ar_delta_ea_list
    else:
        y_list = mr_ar_ea_list
    
    if concat_mr_feature:
        mr_ar_mr_smi_list = []
        for mr_ar in mr_ar_list:
            mr = ar_name_to_mr_name(mr_ar)
            mr_ar_mr_smi_list.append(mr_ar_smis_dict[mr])
        
        if len(x_d_list) > 0:
            reaction_datapoint_list = [
                [ReactionDatapoint.from_smi(smi, np.array([y]), x_d=x_d, keep_h=True) for smi, y, x_d in zip(mr_ar_smi_list, y_list, x_d_list)],  ## from chemprop examples: The target is stored in the 0th component.
                [ReactionDatapoint.from_smi(mr_smi, x_d=x_d, keep_h=True) for mr_smi, x_d in zip(mr_ar_mr_smi_list, x_d_list)]
                ]
        else:
            reaction_datapoint_list = [
                [ReactionDatapoint.from_smi(smi, np.array([y]), keep_h=True) for smi, y in zip(mr_ar_smi_list, y_list)],  ## from chemprop examples: The target is stored in the 0th component.
                [ReactionDatapoint.from_smi(mr_smi, keep_h=True) for mr_smi in mr_ar_mr_smi_list]
                ] 
    else:
        if len(x_d_list) > 0:
            reaction_datapoint_list = [ReactionDatapoint.from_smi(smi, np.array([y]), x_d=x_d, keep_h=True) for smi, y, x_d in zip(mr_ar_smi_list, y_list, x_d_list)]
        else:
            reaction_datapoint_list = [ReactionDatapoint.from_smi(smi, np.array([y]), keep_h=True) for smi, y in zip(mr_ar_smi_list, y_list)]
        
    return reaction_datapoint_list


def train_chemprop(train_datapoint_list, val_datapoint_list, test_datapoint_list, checkpoint_dir, kw, multicomponent=False, random_state=random_seed, warmup_epochs=2, patience=15, max_epochs=50, batch_size=128, extra_feature_len=0, params=None):
    DEFAULT_HIDDEN_DIM = 300
    if params is None:
        mp_depth = 4
        mp_d_h = 2400
        mp_dropout = 0.0
        ffn_hidden_dim = 2200
        ffn_n_layers = 2
        ffn_dropout = 0.2
    elif params.lower() == "default":
        mp_depth = 3
        mp_d_h = DEFAULT_HIDDEN_DIM
        mp_dropout = 0.0
        ffn_hidden_dim = 300
        ffn_n_layers = 1
        ffn_dropout = 0.0
    else:
        mp_depth = params["mp_depth"]
        mp_d_h = params["mp_d_h"]
        mp_dropout = params["mp_dropout"]
        ffn_hidden_dim = params["ffn_hidden_dim"]
        ffn_n_layers = params["ffn_n_layers"]
        ffn_dropout = params["ffn_dropout"]
    
    print(f"mp_depth: {mp_depth}, mp_d_h: {mp_d_h}, mp_dropout: {mp_dropout}, ffn_hidden_dim: {ffn_hidden_dim}, ffn_n_layers: {ffn_n_layers}, ffn_dropout: {ffn_dropout}")
    
    num_workers = len(os.sched_getaffinity(0)) - 1
    
    # Early stopping callback
    early_stop_callback = EarlyStopping(
        monitor='val_loss',
        min_delta=0.00,
        patience=patience,
        verbose=False,
        mode='min'
    )
    # Model checkpoint callback
    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        dirpath=checkpoint_dir,
        # filename='best-{epoch:02d}-{val_loss:.2f}'+f"_{split_name}",
        filename='best'+f"_{kw}",
        save_top_k=1,
        mode='min',
    )
    ## trainer from pytorch lightening
    trainer = pl.Trainer(
        logger=False,
        enable_checkpointing=True,  # Use `True` if you want to save model checkpoints. The checkpoints will be saved in the `checkpoints` folder.
        callbacks=[early_stop_callback, checkpoint_callback],
        enable_progress_bar=True,
        # accelerator="cpu",
        accelerator="auto",
        # accelerator="gpu",
        # devices=2,  # Matches the number of GPUs requested in SLURM
        # strategy="ddp",
        # num_nodes=1,
        max_epochs=max_epochs,  # number of epochs to train for
    )
    
    featurizer = CondensedGraphOfReactionFeaturizer(mode_="REAC_DIFF")  ## REAC_PROD, REAC_DIFF (default), PROD_DIFF
    
    if multicomponent:
        train_datasets = [ReactionDataset(train_datapoint_list[i], featurizer) for i in range(len(train_datapoint_list))]
        val_datasets = [ReactionDataset(val_datapoint_list[i], featurizer) for i in range(len(val_datapoint_list))]
        test_datasets = [ReactionDataset(test_datapoint_list[i], featurizer) for i in range(len(test_datapoint_list))]
        
        train_dataset_unscaled = MulticomponentDataset(train_datasets)
        train_dataset = MulticomponentDataset(train_datasets)
        scaler = train_dataset.normalize_targets()
        
        val_dataset_unscaled = MulticomponentDataset(val_datasets)
        val_dataset = MulticomponentDataset(val_datasets)
        val_dataset.normalize_targets(scaler)
        
        test_dataset_unscaled = MulticomponentDataset(test_datasets)
        
        fdims = featurizer.shape
        mp = nn.MulticomponentMessagePassing(
            blocks=[nn.BondMessagePassing(*fdims, d_h=mp_d_h, depth=mp_depth, dropout=mp_dropout) for _ in range(len(train_datasets))],
            n_components=len(train_datasets),
        )
    else:
        train_dataset_unscaled = ReactionDataset(train_datapoint_list, featurizer)
        train_dataset = ReactionDataset(train_datapoint_list, featurizer)
        scaler = train_dataset.normalize_targets()
        
        val_dataset_unscaled = ReactionDataset(val_datapoint_list, featurizer)
        val_dataset = ReactionDataset(val_datapoint_list, featurizer)
        val_dataset.normalize_targets(scaler)
        
        test_dataset_unscaled = ReactionDataset(test_datapoint_list, featurizer)
        
        fdims = featurizer.shape
        mp = nn.BondMessagePassing(*fdims, d_h=mp_d_h, depth=mp_depth, dropout=mp_dropout)
    
    train_dataloader = build_dataloader(train_dataset, seed=random_state, batch_size=batch_size, num_workers=num_workers, shuffle=True)
    val_dataloader = build_dataloader(val_dataset, seed=random_state, batch_size=batch_size, num_workers=num_workers, shuffle=False)
    
    agg = nn.NormAggregation()  ## NormAggregation() (default), MeanAggregation()
    output_transform = nn.UnscaleTransform.from_standard_scaler(scaler)
    
    ## input_dim changed for cases where mr_ea is additional feature.
    ffn = nn.RegressionFFN(n_tasks=1, input_dim=mp.output_dim+extra_feature_len, hidden_dim=ffn_hidden_dim, n_layers=ffn_n_layers, dropout=ffn_dropout, activation='relu', criterion=None, task_weights=None, threshold=None, output_transform=output_transform)  ## all args
    metric_list = [nn.metrics.RMSE(), nn.metrics.MAE()]  ## only rmse used for training
    if multicomponent:
        mpnn = models.multi.MulticomponentMPNN(mp, agg, ffn, batch_norm=True, metrics=metric_list, warmup_epochs=warmup_epochs, init_lr=0.0001, max_lr=0.001, final_lr=0.0001, X_d_transform=None)  ## all args defaults
    else:
        mpnn = models.MPNN(mp, agg, ffn, batch_norm=True, metrics=metric_list, warmup_epochs=warmup_epochs, init_lr=0.0001, max_lr=0.001, final_lr=0.0001, X_d_transform=None)  ## all args defaults
    
    ## train
    trainer.fit(mpnn, train_dataloader, val_dataloader)
    
    ## predict
    mpnn_best_checkpoint = os.path.join(checkpoint_dir, "best"+f"_{kw}.ckpt")
    if multicomponent:
        mpnn_best = models.multi.MulticomponentMPNN.load_from_checkpoint(mpnn_best_checkpoint)
    else:
        mpnn_best = models.MPNN.load_from_checkpoint(mpnn_best_checkpoint)
        
    train_pred_dataloader = build_dataloader(train_dataset_unscaled, num_workers=num_workers, shuffle=False)
    val_pred_dataloader = build_dataloader(val_dataset_unscaled, num_workers=num_workers, shuffle=False)
    test_pred_dataloader = build_dataloader(test_dataset_unscaled, num_workers=num_workers, shuffle=False)
    
    with torch.inference_mode():
        trainer = pl.Trainer(
            logger=None,
            enable_progress_bar=True,
            accelerator="auto",
            # accelerator="gpu",
            # devices=2,  # Matches the number of GPUs requested in SLURM
            # strategy="ddp",
        )
        train_preds = trainer.predict(mpnn_best, train_pred_dataloader)
        train_preds = np.concatenate(train_preds, axis=0).flatten().tolist()
        val_preds = trainer.predict(mpnn_best, val_pred_dataloader)
        val_preds = np.concatenate(val_preds, axis=0).flatten().tolist()
        test_preds = trainer.predict(mpnn_best, test_pred_dataloader)
        test_preds = np.concatenate(test_preds, axis=0).flatten().tolist()
            
    return mpnn_best_checkpoint, train_preds, val_preds, test_preds


def train_chemprop_transfer(pretrained_checkpoint, ## addition to train_chemprop
                            train_datapoint_list, val_datapoint_list, test_datapoint_list, checkpoint_dir, kw, multicomponent=False, random_state=random_seed, patience=15, max_epochs=50, batch_size=64, extra_feature_len=0):
    ## transfer learning using a pretrained model with all but the last layer frozen.
    num_workers = len(os.sched_getaffinity(0)) - 1
    
    if multicomponent:
        mpnn = models.multi.MulticomponentMPNN.load_from_checkpoint(pretrained_checkpoint)
    else:
        mpnn = models.MPNN.load_from_file(pretrained_checkpoint)
    
    pretraining_scaler = StandardScaler()
    pretraining_scaler.mean_ = mpnn.predictor.output_transform.mean.cpu().numpy()
    pretraining_scaler.scale_ = mpnn.predictor.output_transform.scale.cpu().numpy()
    
    early_stop_callback = EarlyStopping(
        monitor='val_loss',
        min_delta=0.00,
        patience=patience,
        verbose=False,
        mode='min'
    )
    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        dirpath=checkpoint_dir,
        filename='best'+f"_{kw}",
        save_top_k=1,
        mode='min'
    )
    trainer = pl.Trainer(
        logger=False,
        enable_checkpointing=True,
        callbacks=[early_stop_callback, checkpoint_callback],
        enable_progress_bar=True,
        accelerator="auto",
        max_epochs=max_epochs
    )
    
    featurizer = CondensedGraphOfReactionFeaturizer(mode_="REAC_DIFF")
    if multicomponent:
        train_datasets = [ReactionDataset(train_datapoint_list[i], featurizer) for i in range(len(train_datapoint_list))]
        val_datasets = [ReactionDataset(val_datapoint_list[i], featurizer) for i in range(len(val_datapoint_list))]
        test_datasets = [ReactionDataset(test_datapoint_list[i], featurizer) for i in range(len(test_datapoint_list))]
        
        train_dataset_unscaled = MulticomponentDataset(train_datasets)
        train_dataset = MulticomponentDataset(train_datasets)
        train_dataset.normalize_targets(pretraining_scaler)
        
        val_dataset_unscaled = MulticomponentDataset(val_datasets)
        val_dataset = MulticomponentDataset(val_datasets)
        val_dataset.normalize_targets(pretraining_scaler)
        
        test_dataset_unscaled = MulticomponentDataset(test_datasets)
    else:
        train_dataset_unscaled = ReactionDataset(train_datapoint_list, featurizer)
        train_dataset = ReactionDataset(train_datapoint_list, featurizer)
        train_dataset.normalize_targets(pretraining_scaler)
        
        val_dataset_unscaled = ReactionDataset(val_datapoint_list, featurizer)
        val_dataset = ReactionDataset(val_datapoint_list, featurizer)
        val_dataset.normalize_targets(pretraining_scaler)
        
        test_dataset_unscaled = ReactionDataset(test_datapoint_list, featurizer)
    
    train_dataloader = build_dataloader(train_dataset, seed=random_state, batch_size=batch_size, num_workers=num_workers, shuffle=True)
    val_dataloader = build_dataloader(val_dataset, seed=random_state, batch_size=batch_size, num_workers=num_workers, shuffle=False)
    
    # freeze mpnn
    if multicomponent:
        for mp_block in mpnn.message_passing.blocks:
            mp_block.apply(lambda module: module.requires_grad_(False))
            mp_block.eval()
    mpnn.message_passing.apply(lambda module: module.requires_grad_(False))
    mpnn.message_passing.eval()
    mpnn.bn.apply(lambda module: module.requires_grad_(False))
    mpnn.bn.eval()
    
    ## freeze ffn
    for ffn_idx in range(len(mpnn.predictor.ffn)-1):
        mpnn.predictor.ffn[ffn_idx].requires_grad_(False)
        mpnn.predictor.ffn[ffn_idx + 1].eval()
    
    ## train
    trainer.fit(mpnn, train_dataloader, val_dataloader)
    
    ## predict
    mpnn_best_checkpoint = os.path.join(checkpoint_dir, "best"+f"_{kw}.ckpt")
    if multicomponent:
        mpnn_best = models.multi.MulticomponentMPNN.load_from_checkpoint(mpnn_best_checkpoint)
    else:
        mpnn_best = models.MPNN.load_from_checkpoint(mpnn_best_checkpoint)
    
    train_pred_dataloader = build_dataloader(train_dataset_unscaled, num_workers=num_workers, shuffle=False)
    val_pred_dataloader = build_dataloader(val_dataset_unscaled, num_workers=num_workers, shuffle=False)
    test_pred_dataloader = build_dataloader(test_dataset_unscaled, num_workers=num_workers, shuffle=False)
    
    with torch.inference_mode():
        trainer = pl.Trainer(
            logger=None,
            enable_progress_bar=True,
            accelerator="auto"
        )
        train_preds = trainer.predict(mpnn_best, train_pred_dataloader)
        train_preds = np.concatenate(train_preds, axis=0).flatten().tolist()
        val_preds = trainer.predict(mpnn_best, val_pred_dataloader)
        val_preds = np.concatenate(val_preds, axis=0).flatten().tolist()
        test_preds = trainer.predict(mpnn_best, test_pred_dataloader)
        test_preds = np.concatenate(test_preds, axis=0).flatten().tolist()
    
    return mpnn_best_checkpoint, train_preds, val_preds, test_preds


def pred_chemprop(datapoint_list, mpnn_best_checkpoint, multicomponent=False):
    num_workers = len(os.sched_getaffinity(0)) - 1
    
    featurizer = CondensedGraphOfReactionFeaturizer(mode_="REAC_DIFF")
    
    if multicomponent:
        dataset = MulticomponentDataset([ReactionDataset(datapoint_list[i], featurizer) for i in range(len(datapoint_list))])
    else:
        dataset = ReactionDataset(datapoint_list, featurizer)
        
    pred_dataloader = build_dataloader(dataset, num_workers=num_workers, shuffle=False)
    
    if multicomponent:
        mpnn_best = models.multi.MulticomponentMPNN.load_from_checkpoint(mpnn_best_checkpoint)
    else:
        mpnn_best = models.MPNN.load_from_checkpoint(mpnn_best_checkpoint)
        
    with torch.inference_mode():
        trainer = pl.Trainer(
            logger=None,
            enable_progress_bar=True,
            accelerator="auto",
            # accelerator="gpu",
            # devices=2,  # Matches the number of GPUs requested in SLURM
            # strategy="ddp",
        )
        
        preds = trainer.predict(mpnn_best, pred_dataloader)
        preds = np.concatenate(preds, axis=0).flatten().tolist()
        
    return preds
