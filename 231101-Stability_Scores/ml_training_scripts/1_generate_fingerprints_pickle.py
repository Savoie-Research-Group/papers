"""
    Date Modified: 2024/01/23
    Author: Veerupaksh (Veeru) Singla (singla2@purdue.edu)
    Description: creating 2048 int8 figerprints of all acyclic alkanes till C17.
"""


import rdkit.Chem as Chem 
import rdkit.Chem.AllChem as AllChem
import numpy as np
from tqdm import tqdm
import pickle


smi_in_path = "../data/alkanes_till_c17_canon.txt"
p_out_path = "../data/alkanes_till_c17_canon_fp_2048_int8.p"


###############
## binary fp ##
###############

# fp_rad = 2
# fp_size = 4096  # 2048

# def mol_to_fp(mol, radius=fp_rad, nBits=fp_size):
#     # if mol is None:
#     #     return np.zeros((nBits,), dtype=np.bool)
#     return np.array(AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nBits, useChirality=False), dtype=int)


# def smi_to_fp(smi, radius=fp_rad, nBits=fp_size):
#     # if not smi:
#     #     return np.zeros((nBits,), dtype=np.bool)
#     return mol_to_fp(Chem.MolFromSmiles(smi), radius, nBits)


##############
## uint8 fp ##
##############

fp_len_ = 2048
fp_rad_ = 2
convFunc_ = np.array
dtype_ = np.uint8

def mol_to_fp_(mol_, radius_=fp_rad_, nBits_=fp_len_, convFunc__=convFunc_):
    fp = AllChem.GetMorganFingerprint(mol_, radius_, useChirality=False) # uitnsparsevect
    fp_folded = np.zeros((nBits_,), dtype=dtype_)
    for k, v in fp.GetNonzeroElements().items():
        fp_folded[k % nBits_] += v
    return convFunc__(fp_folded)

def smi_to_fp_(smi_, radius_=fp_rad_, nBits_=fp_len_):
    return mol_to_fp_(Chem.MolFromSmiles(smi_), radius_, nBits_)

canon_alkane_list = []

with open(smi_in_path) as f:
    for line in f:
        canon_alkane_list.append(line.strip())
    f.close()

canon_alk_fp_dict = {}

for c_alk_ in tqdm(canon_alkane_list):
    canon_alk_fp_dict[c_alk_] = smi_to_fp_(c_alk_)  # .tolist()

with open(p_out_path, "wb") as f:
    pickle.dump(canon_alk_fp_dict, f)
    f.close()
