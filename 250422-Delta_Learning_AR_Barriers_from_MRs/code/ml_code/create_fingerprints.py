"""
    Last Modified: 2025/04/04
    Author: Veerupaksh (Veeru) Singla (singla2@purdue.edu)
    Description: create fingerprints for ARs and MRs
"""

import os


this_script_dir = os.path.dirname(os.path.realpath(__file__))
os.chdir(this_script_dir)


import sys
sys.path.append(os.path.join(this_script_dir, ".."))


from utils import *


def main():
    
    # Check if fps dir in data_dir. Else create it.
    fps_dir = os.path.join(data_dir, "fps")
    if not os.path.exists(fps_dir):
        os.makedirs(fps_dir)
    else:
        print("fps dir already exists.")
    
    ar_atom_mapped_smi_dict_path = os.path.join(data_paper_data_dir, "ar_atom_mapped_smi_dict.json")
    ar_atom_mapped_smi_dict = json.load(open(ar_atom_mapped_smi_dict_path, "r"))
    
    mr_atom_mapped_smi_dict_path = os.path.join(data_paper_data_dir, "mr_atom_mapped_smi_dict.json")
    mr_atom_mapped_smi_dict = json.load(open(mr_atom_mapped_smi_dict_path, "r"))
    
    mr_ts_dft_opt_geo_dir = os.path.join(data_paper_data_dir, "mr_transition_state_geometries_dft_optimized")
    
    ar_list = list(ar_atom_mapped_smi_dict.keys())
    ar_atom_mapped_smi_list = [ar_atom_mapped_smi_dict[ar] for ar in ar_list]
    
    mr_list = list(mr_atom_mapped_smi_dict.keys())
    mr_atom_mapped_smi_list = [mr_atom_mapped_smi_dict[mr] for mr in mr_list]
    
    mr_ar_list = deepcopy(mr_list)
    mr_ar_list.extend(ar_list)
    
    mr_ar_atom_mapped_smi_list = deepcopy(mr_atom_mapped_smi_list)
    mr_ar_atom_mapped_smi_list.extend(ar_atom_mapped_smi_list)
    
    mr_ar_atom_mapped_smi_list_rev = []
    for smi in mr_ar_atom_mapped_smi_list:
        p, r = smi.split(">>")
        mr_ar_atom_mapped_smi_list_rev.append(r + ">>" + p)
    
    # ## create and save drfp
    # mr_ar_drfp_list = get_rxn_drfp(mr_ar_atom_mapped_smi_list)
    # mr_ar_drfp_dict = {mr_ar_list[i]: mr_ar_drfp_list[i].astype('float32') for i in range(len(mr_ar_list))}
    
    # mr_ar_drfp_list_rev = get_rxn_drfp(mr_ar_atom_mapped_smi_list_rev)
    # mr_ar_drfp_dict_rev = {mr_ar_list[i]: mr_ar_drfp_list_rev[i].astype('float32') for i in range(len(mr_ar_list))}
    
    # mr_ar_drfp_dict_path = os.path.join(fps_dir, "mr_ar_fwd_drfp_dict.pkl")
    # pickle.dump(mr_ar_drfp_dict, open(mr_ar_drfp_dict_path, "wb"))
    
    # mr_ar_drfp_dict_path_rev = os.path.join(fps_dir, "mr_ar_rev_drfp_dict.pkl")
    # pickle.dump(mr_ar_drfp_dict_rev, open(mr_ar_drfp_dict_path_rev, "wb"))
    
    
    # ## create and save mbtr
    # mr_ts_mbtr_dict = {}
    # mr_ts_geo_list = os.listdir(mr_ts_dft_opt_geo_dir)
    # for mr_ts_geo in tqdm(mr_ts_geo_list, desc="mr_ts mbtr"):
    #     mr_ts_xyz_path = os.path.join(mr_ts_dft_opt_geo_dir, mr_ts_geo)
    #     mr_ts_mbtr = get_mbtr_from_xyz(mr_ts_xyz_path)
    #     mr_ts_mbtr_dict[mr_ts_geo.split("-")[0]] = mr_ts_mbtr
    #     print(mr_ts_mbtr.shape, np.count_nonzero(mr_ts_mbtr))
    
    # mr_ts_mbtr_dict_path = os.path.join(fps_dir, "mr_ts_mbtr_dict.pkl")
    # print(mr_ts_mbtr_dict.keys())
    # pickle.dump(mr_ts_mbtr_dict, open(mr_ts_mbtr_dict_path, "wb"))
    
    
    # ## create and save e3fp
    # mr_ts_e3fp_dict = {}
    # mr_ts_geo_list = os.listdir(mr_ts_dft_opt_geo_dir)
    # for mr_ts_geo in tqdm(mr_ts_geo_list, desc="mr_ts e3fp"):
    #     mr_ts_xyz_path = os.path.join(mr_ts_dft_opt_geo_dir, mr_ts_geo)
    #     mr_ts_e3fp = get_e3fp_from_xyz(mr_ts_xyz_path)
    #     mr_ts_e3fp_dict[mr_ts_geo.split("-")[0]] = mr_ts_e3fp
    #     print(mr_ts_e3fp.shape, np.count_nonzero(mr_ts_e3fp), np.nonzero(mr_ts_e3fp))
        
    # mr_ts_e3fp_dict_path = os.path.join(fps_dir, "mr_ts_e3fp_dict.pkl")
    # print(mr_ts_e3fp_dict.keys())
    # pickle.dump(mr_ts_e3fp_dict, open(mr_ts_e3fp_dict_path, "wb"))
    
    
    # ## create and save custom geo fp
    # mr_ts_geofp_dict = {}
    # mr_ts_geo_list = os.listdir(mr_ts_dft_opt_geo_dir)
    # for mr_ts_geo in tqdm(mr_ts_geo_list, desc="mr_ts geofp"):
    #     print(mr_ts_geo)
    #     mr_ts_xyz_path = os.path.join(mr_ts_dft_opt_geo_dir, mr_ts_geo)
    #     mr_ts_geofp = GeometricFingerprint(size=4096, dtype=bool, max_neighbors=4).compute_fp_from_xyz(mr_ts_xyz_path)
    #     mr_ts_geofp_dict[mr_ts_geo.split("-")[0]] = mr_ts_geofp.astype('float32')
    #     print(mr_ts_geofp.shape, np.count_nonzero(mr_ts_geofp), np.nonzero(mr_ts_geofp))
        
    # mr_ts_geofp_dict_path = os.path.join(fps_dir, "mr_ts_geofp_dict.pkl")
    # print(mr_ts_geofp_dict.keys())
    # pickle.dump(mr_ts_geofp_dict, open(mr_ts_geofp_dict_path, "wb"))
    
    return


if __name__ == "__main__":
    main()
