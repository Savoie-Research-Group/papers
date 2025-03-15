import os
import argparse
import time
import subprocess
from tqdm import tqdm
import json


this_script_dir = os.path.dirname(os.path.realpath(__file__))
os.chdir(this_script_dir)


def gen_rmg_hl_input_from_template(template_path, output_path, smi, keyword="<smi>"):
    with open(template_path, "r") as f:
        template = f.read()
        f.close()
        
    template = template.replace(keyword, smi)
    
    with open(output_path, "w") as f:
        f.write(template)
        f.close()
        
    return output_path


def gen_submit_file_from_template(template_path, i, smi, rmg_input_path, sub_output_path, n_cores=8, rmg_path="<rmg_path>/RMG-Py/rmg.py"):
    with open(template_path, "r") as f:
        template = f.read()
        f.close()
        
    template = template.replace("<n_cores>", str(n_cores))
    template = template.replace("<i>", i)
    template = template.replace("<smi>", smi)
    template = template.replace("<input_file>", rmg_input_path)
    template = template.replace("<rmg_path>", rmg_path)
    
    with open(sub_output_path, "w") as f:
        f.write(template)
        f.close()
        
    return sub_output_path


def main():
    parser = argparse.ArgumentParser("generate and submit RMG half life jobs")
    parser.add_argument("--smiles_list_path", type=str, required=True, help="Path to smiles list file")
    parser.add_argument("--working_dir", type=str, required=True, help="Path to working directory")
    parser.add_argument("--rmg_input_template_path", type=str, required=True, help="Path to RMG input template file")
    parser.add_argument("--submit_file_template_path", type=str, required=True, help="Path to submit file template file")
    parser.add_argument("--n_cores", type=int, default=8, help="Number of cores to use. Default: 8")
    parser.add_argument("--rmg_path", type=str, default="<rmg_path>/RMG-Py/rmg.py", help="Path to RMG executable. Default: '<rmg_path>/RMG-Py/rmg.py'")  ## CHANGE RMG PATH ACCORDING TO YOUR SYSTEM
    args = parser.parse_args()
    
    smiles_list_path = args.smiles_list_path
    working_dir = args.working_dir
    rmg_input_template_path = args.rmg_input_template_path
    submit_file_template_path = args.submit_file_template_path
    n_cores = args.n_cores
    rmg_path = args.rmg_path
    
    smiles_ran_list = os.listdir(working_dir)
    
    smiles_list = []
    with open(smiles_list_path, "r") as f:
        for line in f:
            smi = line.strip()
            if smi not in smiles_ran_list:
                smiles_list.append(smi)
        f.close()
        
    os.chdir(working_dir)
    # os.system("rm -rf *")
    
    smi_dir_dict = {}
    for i, smi in tqdm(enumerate(smiles_list), desc="Creating smi-dir mapping"):
        smi_dir_dict[smi] = str(i)
        
    with open("smi_dir_dict.json", "w") as f:
        json.dump(smi_dir_dict, f, indent=4)
        f.close()
    
    for smi in tqdm(smiles_list, desc="Submitting jobs"):
        i = smi_dir_dict[smi]
        
        os.system(f"mkdir {i}")
        os.chdir(i)
        
        gen_rmg_hl_input_from_template(rmg_input_template_path, "input_hl.py", smi)
        gen_submit_file_from_template(submit_file_template_path, i, smi, "input_hl.py", "job_submit.sub", n_cores=n_cores, rmg_path=rmg_path)
        
        ## MODIFY TO RUN ON YOUR QUEUE MANAGEMENT SYSTEM. THIS IS FOR SLURM
        while True:
            n_jobs = subprocess.check_output("squeue -u <USER> | wc -l", shell=True).decode("utf-8").strip()
            n_jobs = int(n_jobs) - 1
            if n_jobs < 4501:
                break
            else:
                time.sleep(10)
        os.system("sbatch job_submit.sub")
        os.chdir("..")
    
    os.chdir(this_script_dir)


if __name__ == "__main__":
    main()
