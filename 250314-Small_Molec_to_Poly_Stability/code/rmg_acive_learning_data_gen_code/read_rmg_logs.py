import os
import argparse
import json
import multiprocessing as mp


this_script_dir = os.path.dirname(os.path.realpath(__file__))
os.chdir(this_script_dir)


def read_rmg_log(smi, working_dir, smi_dir_dict, working_dir_list=None):
    if working_dir_list is None:
        working_dir_list = os.listdir(working_dir)
    
    if smi not in smi_dir_dict:
        print("SMILES {} not in smi_dir_dict".format(smi))
        return -1
    
    if smi_dir_dict[smi] not in working_dir_list:
        print("SMILES {} not in working directory {}".format(smi, working_dir))
        return -1
    
    rmg_log_path = os.path.join(working_dir, smi_dir_dict[smi], "RMG.log")
    
    try:
        with open(rmg_log_path, "r") as f:
            f_read = f.read()
            f.close()
    except:
        print("RMG.log not found for SMILES {}".format(smi))
        return -1
    
    if "MODEL GENERATION COMPLETED" not in f_read:
        print("Model generation NOT completed for SMILES {}".format(smi))
        return -1
    
    f_read = f_read.split("\n")
    f_read.reverse()
    
    for line in f_read:
        if "At time" in line:
            if "reached target termination conversion" not in line:
                print("SMILES {} did not reach target termination conversion".format(smi))
                return -1
            else:
                try:
                    ## check for collision_rate_violators.log. Using max violation factor of 5 based on:
                    ##     Dongping Chen, Kun Wang, Hai Wang. Violation of collision limit in recently published reaction models. https://doi.org/10.1016/j.combustflame.2017.08.005
                    
                    collision_rate_violators_log_path = os.path.join(working_dir, smi_dir_dict[smi], "collision_rate_violators.log")
                    violation_factors_list = []
                    with open(collision_rate_violators_log_path, "r") as collision_rate_violators_log_read:
                        for collision_rate_violators_log_line in collision_rate_violators_log_read:
                            if "Violation factor:" in collision_rate_violators_log_line:
                                violation_factors_list.append(float(collision_rate_violators_log_line.strip().split("Violation factor:")[-1]))
                        collision_rate_violators_log_read.close()
                    
                    if max(violation_factors_list) > 5:
                        print("SMILES {} has collision rate violation factor > 5".format(smi))
                        return -1
                    time = float(line.split(" ")[2])
                    return time
                except:
                    time = float(line.split(" ")[2])
                    return time
    return -1


def main():
    parser = argparse.ArgumentParser("read RMG half life job outputs")
    parser.add_argument("--smiles_list_path", type=str, required=True, help="Path to smiles list file")
    parser.add_argument("--working_dir", type=str, required=True, help="Path to working directory")
    args = parser.parse_args()
    
    smiles_list_path = args.smiles_list_path
    working_dir = args.working_dir
    
    smiles_list = []
    with open(smiles_list_path, "r") as f:
        for line in f:
            smiles_list.append(line.strip())
        f.close()
        
    smi_dir_dict = json.load(open(os.path.join(working_dir, "smi_dir_dict.json"), "r"))
    # smi_dir_dict = {}

    list_working_dir = os.listdir(working_dir)
    
    smi_hl_dict = {}
    
    mp_cores = len(os.sched_getaffinity(0))
    print("Using {} cores".format(mp_cores))
    
    with mp.Pool(mp_cores) as pool:
        results_ = pool.starmap(read_rmg_log, [(smi, working_dir, smi_dir_dict, list_working_dir) for smi in smiles_list])
        
    for smi, hl in zip(smiles_list, results_):
        smi_hl_dict[smi] = hl
        
    with open(os.path.join(working_dir, "smi_hl_dict.json"), "w") as f:
        json.dump(smi_hl_dict, f, indent=4)
        f.close()
        
    os.chdir(this_script_dir)


if __name__ == "__main__":
    main()
