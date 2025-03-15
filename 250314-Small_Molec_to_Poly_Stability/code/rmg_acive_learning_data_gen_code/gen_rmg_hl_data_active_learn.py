import os
import pickle
import json
import math
from copy import deepcopy
import subprocess
import time
import multiprocessing as mp

from rdkit import Chem
import rdkit.Chem.AllChem as AllChem

import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils

this_script_dir = os.path.dirname(os.path.realpath(__file__))
os.chdir(this_script_dir)

from submit_rmg_hl_jobs import gen_rmg_hl_input_from_template, gen_submit_file_from_template
from read_rmg_logs import read_rmg_log


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")
print("Device:", device)


def read_smi_list(smi_list_path):
    smi_list = []
    with open(smi_list_path, "r") as f:
        for line in f:
            smi_list.append(line.strip())
        f.close()
    return smi_list


def write_smi_list(smi_list, smi_list_path):
    with open(smi_list_path, "w") as f:
        f.write("\n".join(smi_list))
        f.close()


def read_pickle(pickle_path):
    return pickle.load(open(pickle_path, "rb"))


def write_pickle(obj, pickle_path):
    pickle.dump(obj, open(pickle_path, "wb"))


def submit_job_get_job_id(submit_file_path):
    return subprocess.check_output(f"sbatch {submit_file_path}", shell=True, text=True).strip().split()[-1]


def get_job_ids_squeue_command(squeue_command):
    return [jl.split()[0] for jl in subprocess.check_output(squeue_command, shell=True, text=True).split("\n")[1:-1]]


def gen_unit8_fp_from_smiles(smi, fp_radius=3, fp_size=2048):
    convFunc = np.array
    dtype = np.uint8
    def mol_to_fp(mol, radius=fp_radius, nBits=fp_size, convFunc=convFunc):
        fp = AllChem.GetMorganFingerprint(mol, radius, useChirality=False)
        fp_folded = np.zeros((nBits,), dtype=dtype)
        for k, v in fp.GetNonzeroElements().items():
            fp_folded[k % nBits] += v
        return convFunc(fp_folded)
    def smi_to_fp(smi, radius=fp_radius, nBits=fp_size):
        return mol_to_fp(Chem.MolFromSmiles(smi), radius, nBits)
    return smi_to_fp(smi)


def custom_hinge_loss(preds, targets):
    targets_diff = targets.unsqueeze(0) - targets.unsqueeze(1)
    preds_diff = preds.unsqueeze(0) - preds.unsqueeze(1)
    # epsilon = 1.0
    epsilon = 0.1
    scaled_diff = (epsilon * targets_diff - preds_diff) * torch.sign(targets_diff)
    n_pairs = targets.size(0) * (targets.size(0) - 1)
    hinge_loss = torch.sum(F.relu(scaled_diff)) / n_pairs
    return hinge_loss


class Net1_2048fp(nn.Module):
    def __init__(self):
        super(Net1_2048fp, self).__init__()
        self.fc1 = nn.Linear(2048, 512)
        self.dropout1 = nn.Dropout(0.35)
        self.fc2 = nn.Linear(512, 512)
        self.dropout2 = nn.Dropout(0.35)
        self.fc3 = nn.Linear(512, 256)
        self.dropout3 = nn.Dropout(0.35)
        self.fc4 = nn.Linear(256, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        x = F.relu(self.fc3(x))
        x = self.dropout3(x)
        x = self.fc4(x)
        return x


def prepare_data_for_training(smi_list, smi_hl_dict, smi_fp_dict, train_fr=0.8, val_fr=0.1, test_fr=0.1, bool_test=True):
    X = np.array([smi_fp_dict[i] for i in smi_list])
    y = np.array([smi_hl_dict[i] for i in smi_list])
    
    if bool_test:
        fr_scale = 1 / (train_fr + val_fr + test_fr)
        train_fr = train_fr * fr_scale
        val_fr = val_fr * fr_scale
        test_fr = 1 - train_fr - val_fr
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_fr)
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=val_fr / (train_fr + val_fr))
        y_train_mean = np.mean(y_train)
        y_train_std = np.std(y_train)
        y_train = (y_train - y_train_mean) / y_train_std
        y_val = (y_val - y_train_mean) / y_train_std
        y_test = (y_test - y_train_mean) / y_train_std
        return X_train, X_val, X_test, y_train, y_val, y_test, y_train_mean, y_train_std
    
    fr_scale = 1 / (train_fr + val_fr)
    train_fr = train_fr * fr_scale
    val_fr = 1 - train_fr
    
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=val_fr)
    y_train_mean = np.mean(y_train)
    y_train_std = np.std(y_train)
    y_train = (y_train - y_train_mean) / y_train_std
    y_val = (y_val - y_train_mean) / y_train_std
    return X_train, X_val, None, y_train, y_val, None, y_train_mean, y_train_std


def train_1(NetClass, X_train, X_val, y_train, y_val, criterion="mse", max_epochs=100, tolerance_epochs=25, batch_size=128, lr=0.0001, device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
    net = NetClass().to(device)
    
    if criterion == "mse":
        criterion = nn.MSELoss()
    elif criterion == "custom_hinge":
        criterion = custom_hinge_loss
    
    optimizer = optim.Adam(net.parameters(), lr=lr)
    
    X_train = torch.from_numpy(X_train).float()
    y_train = torch.from_numpy(y_train).float().view(-1, 1)
    X_val = torch.from_numpy(X_val).float()
    y_val = torch.from_numpy(y_val).float().view(-1, 1)
    
    train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataset = torch.utils.data.TensorDataset(X_val, y_val)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
    
    train_losses = []
    val_losses = []
    
    best_val_loss = float("inf")
    best_model = deepcopy(net)
    best_epoch = 0
    for epoch in tqdm(range(max_epochs), desc="Training Epochs"):
        net.train()
        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        train_losses.append(running_loss / len(train_loader))
        
        net.eval()
        running_loss = 0.0
        with torch.no_grad():
            for i, data in enumerate(val_loader, 0):
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = net(inputs)
                loss = criterion(outputs, labels)
                running_loss += loss.item()
            val_losses.append(running_loss / len(val_loader))
            if val_losses[-1] < best_val_loss:
                best_val_loss = val_losses[-1]
                best_model = deepcopy(net)
                best_epoch = epoch
            elif epoch - best_epoch >= tolerance_epochs:
                print(f"Early stopping at epoch {epoch + 1}")
                break
        if epoch % 5 == 0:
            print(f"Epoch {epoch + 1}/{max_epochs} | Train Loss: {train_losses[-1]:.4f} | Val Loss: {val_losses[-1]:.4f} | Best Epoch: {best_epoch + 1} | Best Val Loss: {best_val_loss:.4f}")
    return best_model, best_model.state_dict(), train_losses, val_losses, best_epoch, best_val_loss


def train_single_model(X_train, X_val, X_test, y_train, y_val, y_test, y_train_mean, y_train_std, NetClass=Net1_2048fp, criterion="mse", max_epochs=300, tolerance_epochs=50, batch_size=128, lr=0.0001, device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
    best_model, best_model_state_dict, train_losses, val_losses, best_epoch, best_val_loss = train_1(NetClass, X_train, X_val, y_train, y_val, criterion, max_epochs, tolerance_epochs, batch_size, lr, device)
    return best_model, best_model_state_dict, train_losses, val_losses, best_epoch, best_val_loss


def train_ensemble_models(smi_list, smi_hl_dict, smi_fp_dict, train_fr=0.8, val_fr=0.1, test_fr=0.1, bool_test=False, re_split=True, NetClass=Net1_2048fp, criterion="mse", max_epochs=300, tolerance_epochs=50, batch_size=128, lr=0.0001, device=torch.device("cuda" if torch.cuda.is_available() else "cpu"), n_models=30):
    X_train, X_val, X_test, y_train, y_val, y_test, y_train_mean, y_train_std = prepare_data_for_training(smi_list, smi_hl_dict, smi_fp_dict, train_fr, val_fr, test_fr, bool_test)
    # print(f"X_train: {X_train.shape} | X_val: {X_val.shape} | X_test: {X_test.shape if X_test is not None else 'None'} | y_train: {y_train.shape} | y_val: {y_val.shape} | y_test: {y_test.shape if y_test is not None else 'None'} | y_train_mean: {y_train_mean} | y_train_std: {y_train_std}")
    models = []
    model_state_dicts = []
    train_losses = []
    val_losses = []
    best_epochs = []
    best_val_losses = []
    print("Training ensemble models...")
    for i in range(n_models):
        print(f"Training Model {i + 1}/{n_models}")
        if re_split:
            X_train, X_val, X_test, y_train, y_val, y_test, y_train_mean, y_train_std = prepare_data_for_training(smi_list, smi_hl_dict, smi_fp_dict, train_fr, val_fr, test_fr, bool_test)
        best_model, best_model_state_dict, train_loss, val_loss, best_epoch, best_val_loss = train_single_model(X_train, X_val, X_test, y_train, y_val, y_test, y_train_mean, y_train_std, NetClass, criterion, max_epochs, tolerance_epochs, batch_size, lr, device)
        models.append(best_model)
        model_state_dicts.append(best_model_state_dict)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        best_epochs.append(best_epoch)
        best_val_losses.append(best_val_loss)
    return models, model_state_dicts, train_losses, val_losses, best_epochs, best_val_losses, X_train, X_val, X_test, y_train, y_val, y_test, y_train_mean, y_train_std


def train_single_mse_model(X_train, X_val, X_test, y_train, y_val, y_test, y_train_mean, y_train_std, NetClass=Net1_2048fp, max_epochs=300, tolerance_epochs=50, batch_size=128, lr=0.0001, device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
    return train_single_model(X_train, X_val, X_test, y_train, y_val, y_test, y_train_mean, y_train_std, NetClass, "mse", max_epochs, tolerance_epochs, batch_size, lr, device)


def train_ensemble_mse_models(smi_list, smi_hl_dict, smi_fp_dict, train_fr=0.8, val_fr=0.1, test_fr=0.1, bool_test=False, re_split=True, NetClass=Net1_2048fp, max_epochs=300, tolerance_epochs=50, batch_size=128, lr=0.0001, device=torch.device("cuda" if torch.cuda.is_available() else "cpu"), n_models=30):
    return train_ensemble_models(smi_list, smi_hl_dict, smi_fp_dict, train_fr, val_fr, test_fr, bool_test, re_split, NetClass, "mse", max_epochs, tolerance_epochs, batch_size, lr, device, n_models)


def train_single_hinge_model(X_train, X_val, X_test, y_train, y_val, y_test, y_train_mean, y_train_std, NetClass=Net1_2048fp, max_epochs=300, tolerance_epochs=50, batch_size=128, lr=0.0001, device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
    return train_single_model(X_train, X_val, X_test, y_train, y_val, y_test, y_train_mean, y_train_std, NetClass, "custom_hinge", max_epochs, tolerance_epochs, batch_size, lr, device)


def train_ensemble_hinge_models(smi_list, smi_hl_dict, smi_fp_dict, train_fr=0.8, val_fr=0.1, test_fr=0.1, bool_test=False, re_split=True, NetClass=Net1_2048fp, max_epochs=300, tolerance_epochs=50, batch_size=128, lr=0.0001, device=torch.device("cuda" if torch.cuda.is_available() else "cpu"), n_models=30):
    return train_ensemble_models(smi_list, smi_hl_dict, smi_fp_dict, train_fr, val_fr, test_fr, bool_test, re_split, NetClass, "custom_hinge", max_epochs, tolerance_epochs, batch_size, lr, device, n_models)


def predict_single_model(model, X, batch_size=100000, device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
    model.eval()
    if not isinstance(X, torch.Tensor):
        X = torch.from_numpy(X).float()
    preds = torch.zeros(X.size(0), device=device)
    with torch.no_grad():
        for i in tqdm(range(0, X.size(0), batch_size), desc="Predicting"):
            preds[i:i + batch_size] = model(X[i:i + batch_size].to(device)).view(-1)
    return preds.cpu().detach().numpy()


def predict_ensemble_models(models, X, mask="linear", batch_size=100000, device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
    preds = np.zeros((len(models), X.shape[0]))
    for i, model in tqdm(enumerate(models), desc="Predicting Ensemble Models"):
        if mask == "linear":
            preds[i] = predict_single_model(model, X, batch_size, device)
        elif mask == "sigmoid":
            preds[i] = torch.sigmoid(torch.from_numpy(predict_single_model(model, X, batch_size, device)).float()).cpu().detach().numpy()
    preds_mean = np.mean(preds, axis=0)
    preds_std = np.std(preds, axis=0)
    preds_median = np.median(preds, axis=0)
    preds_q1 = np.quantile(preds, 0.25, axis=0)
    preds_q3 = np.quantile(preds, 0.75, axis=0)
    return preds_mean, preds_std, preds_median, preds_q1, preds_q3, preds


def get_smiles_uncertainty(smi_list, smi_fp_dict, models, batch_size=100000, device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
    smi_list = np.array(smi_list)
    mean_arr = np.empty(len(smi_list))
    std_arr = np.empty(len(smi_list))
    for i in tqdm(range(0, len(smi_list), batch_size), desc="Getting batch uncertainties"):
        smi_fp = np.array([smi_fp_dict[i] for i in smi_list[i:i + batch_size]])
        smi_fp = torch.from_numpy(smi_fp).float().to(device)
        preds = torch.zeros((len(models), smi_fp.size(0)), device=device)
        for j, model in enumerate(models):
            model.eval()
            with torch.no_grad():
                preds[j] = model(smi_fp).view(-1)
        preds = preds.cpu().detach().numpy()
        preds_mean = np.mean(preds, axis=0)
        preds_std = np.std(preds, axis=0)
        mean_arr[i:i + batch_size] = preds_mean
        std_arr[i:i + batch_size] = preds_std
    # reverse sort by standard deviation. i.e. most uncertain first
    return smi_list, mean_arr, std_arr


def gen_one_it_smi_list_active_learn(main_smi_hl_dict, iteration_smi_lists_path_list, main_smi_fp_dicts_path_list, n_smi_to_get=4000, batch_size=100000, device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
    main_smi_list = list(main_smi_hl_dict.keys())
    main_smi_fp_dict = {i: gen_unit8_fp_from_smiles(i, fp_radius=3, fp_size=2048) for i in tqdm(main_smi_list, desc="Generating 2048 unit8 rad3 fingerprints")}
    smi_done_list = []
    for iter_smi_list_path in iteration_smi_lists_path_list:
        smi_done_list.extend(read_smi_list(iter_smi_list_path))
    models, _, _, _, _, _, _, _, _, _, _, _, _, _ = train_ensemble_mse_models(main_smi_list, main_smi_hl_dict, main_smi_fp_dict, train_fr=0.8, val_fr=0.1, test_fr=0.1, bool_test=False, re_split=True,
                                                                              NetClass=Net1_2048fp, max_epochs=30, tolerance_epochs=15, batch_size=128, lr=0.0005, device=device, n_models=30)
    smi_list_list = []
    mean_arr_list = []
    std_arr_list = []
    for smi_fp_dict_path in tqdm(main_smi_fp_dicts_path_list, desc="Getting uncertainties for smi_fp_dicts"):
        smi_fp_dict = read_pickle(smi_fp_dict_path)
        smi_list = [i for i in smi_fp_dict.keys() if i not in smi_done_list]
        smi_list, mean_arr, std_arr = get_smiles_uncertainty(smi_list, smi_fp_dict, models, batch_size, device)
        smi_list_list.append(smi_list)
        mean_arr_list.append(mean_arr)
        std_arr_list.append(std_arr)
    smi_list = np.concatenate(smi_list_list)
    mean_arr = np.concatenate(mean_arr_list)
    std_arr = np.concatenate(std_arr_list)
    print("\nSelecting most uncertain smiles...")
    sort_idx = np.argsort(-std_arr)
    smi_selected = smi_list[sort_idx[:n_smi_to_get]]
    return smi_selected, std_arr[sort_idx[:n_smi_to_get]]


def submit_rmg_hl_jobs(working_dir, smiles_list_path=None, smiles_list=None,
                       rmg_input_template_path=os.path.join(this_script_dir, "rmg_input_template.py"), 
                       submit_file_template_path=os.path.join(this_script_dir, "job_submit_template_slurm.sub"), n_cores=8, 
                       rmg_path="<rmg_path>/RMG-Py/rmg.py"):  ### UPDATE RMG PATH
    if smiles_list is None:
        smiles_list = read_smi_list(smiles_list_path)
    this_script_dir = os.path.dirname(os.path.realpath(__file__))
    os.chdir(working_dir)
    smi_dir_dict = {smi: str(i) for i, smi in enumerate(smiles_list)}
    json.dump(smi_dir_dict, open("smi_dir_dict.json", "w"))
    submitted_jid_list = []
    for smi in tqdm(smiles_list, desc="Submitting RMG jobs"):
        i = smi_dir_dict[smi]
        os.system(f"mkdir {i}")
        os.chdir(i)
        gen_rmg_hl_input_from_template(rmg_input_template_path, "input_hl.py", smi)
        gen_submit_file_from_template(submit_file_template_path, i, smi, "input_hl.py", "job_submit.sub", n_cores=n_cores, rmg_path=rmg_path)
        while True:
            n_jobs = subprocess.check_output("squeue -u singla2 | wc -l", shell=True).decode("utf-8").strip()
            n_jobs = int(n_jobs) - 1
            if n_jobs < 4501:
                break
            else:
                time.sleep(10)
        submit_jid = submit_job_get_job_id("job_submit.sub")
        submitted_jid_list.append(submit_jid)
        os.chdir("..")
    os.chdir(this_script_dir)
    submitted_jid_set = set(submitted_jid_list)
    print("Waiting for RMG jobs to finish...")
    while len(submitted_jid_set - set(get_job_ids_squeue_command("squeue -u singla2"))) < len(submitted_jid_set):
        time.sleep(10)
    print("RMG jobs finished")
    return


def read_rmg_logs(working_dir, smiles_list_path=None, smiles_list=None):
    if smiles_list is None:
        smiles_list = read_smi_list(smiles_list_path)
    smi_dir_dict = json.load(open(os.path.join(working_dir, "smi_dir_dict.json"), "r"))
    list_working_dir = os.listdir(working_dir)
    smi_hl_dict = {}
    mp_cores = len(os.sched_getaffinity(0))
    print("\nUsing {} cores to read RMG results".format(mp_cores))
    with mp.Pool(mp_cores) as pool:
        results_ = pool.starmap(read_rmg_log, [(smi, working_dir, smi_dir_dict, list_working_dir) for smi in smiles_list])
    for smi, hl in zip(smiles_list, results_):
        smi_hl_dict[smi] = hl
    return smi_hl_dict


def smi_fp_dict_gen(smi_list_path, smi_fp_dicts_path):
    smi_list = []
    with open(smi_list_path, "r") as f:
        for line in f:
            smi_list.append(line.strip())
        f.close()
    smi_dict_num = 0
    for i in range(0, len(smi_list), 500000):
        smi_fp_dict = {i: gen_unit8_fp_from_smiles(i, fp_radius=3, fp_size=2048) for i in tqdm(smi_list[i:i + 500000], desc="Generating 2048 unit8 rad3 fingerprints")}
        pickle.dump(smi_fp_dict, open(f"{smi_fp_dicts_path}/main_smi_2048_uint8_rad3_fp_dict_{smi_dict_num}.p", "wb"))
        smi_dict_num += 1


def main():
    this_script_dir = os.path.dirname(os.path.realpath(__file__))
    os.chdir(this_script_dir)
    
    #---------------------------------------------------------------------------------------------------------------------------------------#
    #######################
    ## initializing code ##
    #######################
    smi_fp_dict_gen(os.path.join(this_script_dir, "../../data/rmg_active_learning_data/sampling_space_pubchem_chon_f_cl_max_15_ha_clean_acyclic.txt"), "<YOUR_SCRATCH_PATH>/active_learn_data/smi_fp_dicts")
    #---------------------------------------------------------------------------------------------------------------------------------------#
    
    #---------------------------------------------------------------------------------------------------------------------------------------#
    ##################
    ## data loading ##
    ##################
    active_learning_data_path = "<YOUR_SCRATCH_PATH>/active_learn_data"  ## PATH WHERE ACTIVE SAMPLING DATA WILL BE STORED
    rmg_working_dir = "<YOUR_SCRATCH_PATH>/rmg_run"  ## PATH WHERE RMG JOBS WILL BE SUBMITTED AND OUTPUT DATA WILL BE STORED
    rmg_input_template_path = os.path.join(this_script_dir, "rmg_input_template.py")
    rmg_submit_file_template_path = os.path.join(this_script_dir, "job_submit_template_slurm.sub")
    
    num_active_learn_iterations = 10
    
    main_smi_hl_dict_path = os.path.join(active_learning_data_path, "main_smi_hl_dict.json")
    main_smi_log_hl_dict_path = os.path.join(active_learning_data_path, "main_smi_log_hl_dict.json")
    all_smi_list_path = os.path.join(this_script_dir, "../../data/rmg_active_learning_data/sampling_space_pubchem_chon_f_cl_max_15_ha_clean_acyclic.txt")
    iteration_smi_lists_path = os.path.join(active_learning_data_path, "iterations_smi_lists")
    iteration_smi_lists_path_list = [os.path.join(iteration_smi_lists_path, i) for i in os.listdir(iteration_smi_lists_path)]
    main_smi_fp_dicts_path = os.path.join(active_learning_data_path, "smi_fp_dicts")
    main_smi_fp_dicts_path_list = [os.path.join(main_smi_fp_dicts_path, i) for i in os.listdir(main_smi_fp_dicts_path)]
    main_smi_hl_dict = json.load(open(main_smi_hl_dict_path, "r"))
    main_smi_log_hl_dict = json.load(open(main_smi_log_hl_dict_path, "r"))
    
    #---------------------------------------------------------------------------------------------------------------------------------------#
    
    #----------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
    ########################
    ## Do active learning ##
    ########################
    active_learn_iter_smi_list_num = len(iteration_smi_lists_path_list)
    already_done_iter_smi_lists_path = os.path.join(active_learning_data_path, "iterations_smi_lists")  ## WHERE ALREADY RAN SMILES WILL BE STORED to filter during re-runs
    already_done_smi_hl_dict_path = os.path.join(active_learning_data_path, "main_smi_hl_dict.json")
    already_done_smi_log_hl_dict_path = os.path.join(active_learning_data_path, "main_smi_log_hl_dict.json")
    
    already_done_smi_list = []
    for smi_l_i in os.listdir(already_done_iter_smi_lists_path):
        already_done_smi_list.extend(read_smi_list(os.path.join(already_done_iter_smi_lists_path, smi_l_i)))
    already_done_smi_hl_dict = json.load(open(already_done_smi_hl_dict_path, "r"))
    already_done_smi_log_hl_dict = json.load(open(already_done_smi_log_hl_dict_path, "r"))
    print(f"Already Done Smi List: {len(already_done_smi_list)}")
    print(f"Already Done Smi HL Dict: {len(already_done_smi_hl_dict)}")
    print(f"Already Done Smi Log HL Dict: {len(already_done_smi_log_hl_dict)}")
    
    for active_learn_iter in range(num_active_learn_iterations):
        print(f"\nActive Learning Iteration {active_learn_iter + 1}/{num_active_learn_iterations}")
        ## run ensemble mse models to get smi uncertainty
        smi_to_get, smi_uncertainty = gen_one_it_smi_list_active_learn(main_smi_log_hl_dict, iteration_smi_lists_path_list, main_smi_fp_dicts_path_list,
                                                                       n_smi_to_get=8000, batch_size=100000, device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        iter_smi_list_path = os.path.join(iteration_smi_lists_path, f"iteration_{active_learn_iter_smi_list_num}_smi_list.txt")
        write_smi_list(smi_to_get, iter_smi_list_path)
        iteration_smi_lists_path_list.append(iter_smi_list_path)
        active_learn_iter_smi_list_num += 1
        os.system(f"rm -rf {rmg_working_dir}; mkdir {rmg_working_dir}")
        # run rmg with rmg_input_template.py
        smi_to_get_unique = [smi for smi in smi_to_get if smi not in already_done_smi_list]
        print(f"Unique Smi to get: {len(smi_to_get_unique)}")
        submit_rmg_hl_jobs(rmg_working_dir, smiles_list=smi_to_get_unique, rmg_input_template_path=rmg_input_template_path,
                           submit_file_template_path=rmg_submit_file_template_path, n_cores=8)
        iter_smi_hl_dict = read_rmg_logs(rmg_working_dir, smiles_list=smi_to_get_unique)
        iter_smi_hl_dict = {k: v for k, v in iter_smi_hl_dict.items() if v > 0}
        iter_smi_log_hl_dict = {k: math.log(v) for k, v in tqdm(iter_smi_hl_dict.items(), desc="Generating log half life dict")}
        iter_smi_hl_dict_already_done = {k: already_done_smi_hl_dict[k] for k in smi_to_get if k in already_done_smi_hl_dict}
        iter_smi_log_hl_dict_already_done = {k: already_done_smi_log_hl_dict[k] for k in smi_to_get if k in already_done_smi_log_hl_dict}
        print(f"Already Done Smi HL Dict: {len(iter_smi_hl_dict_already_done)}")
        print(f"Already Done Smi Log HL Dict: {len(iter_smi_log_hl_dict_already_done)}")
        iter_smi_hl_dict.update(iter_smi_hl_dict_already_done)
        iter_smi_log_hl_dict.update(iter_smi_log_hl_dict_already_done)
        main_smi_hl_dict.update(iter_smi_hl_dict)
        main_smi_log_hl_dict.update(iter_smi_log_hl_dict)
        with open(main_smi_hl_dict_path, "w") as f:
            json.dump(main_smi_hl_dict, f, indent=4)
        with open(main_smi_log_hl_dict_path, "w") as f:
            json.dump(main_smi_log_hl_dict, f, indent=4)
        print(f"\nActive Learning Iteration {active_learn_iter + 1}/{num_active_learn_iterations} Done")
        
    print("\nActive Learning Done")
    #----------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
    return


if __name__ == "__main__":
    main()
