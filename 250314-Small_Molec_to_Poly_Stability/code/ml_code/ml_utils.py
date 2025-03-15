"""
    Date Modified: 2024/10/24
    Author: Veerupaksh (Veeru) Singla (singla2@purdue.edu)
    Corresponding Author: Brett M Savoie (bsavoie2@nd.edu)
"""


import os
import multiprocessing as mp
import random
import json
from copy import deepcopy

import numpy as np
from tqdm import tqdm

from rdkit import Chem
import rdkit.Chem.AllChem as AllChem

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torchinfo import summary

import sklearn
from sklearn.model_selection import train_test_split


this_script_dir = os.path.dirname(os.path.realpath(__file__))
os.chdir(this_script_dir)


## global variables
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

random_state = 41
np.random.seed(random_state)
torch.manual_seed(random_state)
random.seed(random_state)
sklearn.utils.check_random_state(random_state)


def get_ha_from_smi(smi):
    mol = Chem.MolFromSmiles(smi)
    return mol.GetNumHeavyAtoms()


def gen_uint8_fp_from_smiles(smi, fp_radius=2, fp_size=2048):
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


def gen_uint8_fp_from_smiles_list_parallel(smi_list, fp_radius=2, fp_size=2048):
    # mp_cores = mp.cpu_count()
    mp_cores = len(os.sched_getaffinity(0))
    
    with mp.Pool(mp_cores) as pool:
        fps = pool.starmap(gen_uint8_fp_from_smiles, [(smi, fp_radius, fp_size) for smi in smi_list])
    
    return dict(zip(smi_list, fps))


def my_hinge_loss_v1(preds, targets):
    # same as my_hinge_loss_v2 but using for loop instead of torch.triu
    
    torch_device = preds.device
    n = targets.size(dim=0)
    
    f_0 = torch.empty(n * (n - 1) // 2, device=torch_device, dtype=torch.float32)
    f_1 = torch.empty(n * (n - 1) // 2, device=torch_device, dtype=torch.float32)
    zero = torch.zeros(n * (n - 1) // 2, device=torch_device, dtype=torch.float32)
    
    idx = 0
    for i in range(n-1):
        f_0[idx:idx+n-i-1] = targets[i+1:, 0] - targets[i, 0]
        f_1[idx:idx+n-i-1] = preds[i+1:, 0] - preds[i, 0]
        idx += n - i - 1
    
    return torch.sum(torch.maximum(torch.mul(f_0 - f_1, torch.sign(f_0)), zero))


def my_hinge_loss_v2(preds, targets):
    # same as my_hinge_loss_v1 but using torch.triu instead of for loop. might be faster
    
    torch_device = preds.device
    n = targets.size(dim=0)
    
    f_0 = targets.unsqueeze(0) - targets.unsqueeze(1)
    f_1 = preds.unsqueeze(0) - preds.unsqueeze(1)
    
    mask = torch.triu(torch.ones(n, n, device=torch_device, dtype=torch.bool), 1)
    f_0 = f_0[mask]
    f_1 = f_1[mask]
    
    zero = torch.zeros_like(f_0)
    return torch.sum(torch.maximum(torch.mul(f_0 - f_1, torch.sign(f_0)), zero))


def get_pairwise_accuracy(y_true, y_pred, y_true_thresh=0.0):
    y_true = np.array(y_true, dtype=np.float32).flatten()
    y_pred = np.array(y_pred, dtype=np.float32).flatten()
    
    n1 = len(y_true)
    n2 = len(y_pred)
    
    assert n1 == n2, "Lengths of y_true and y_pred should be equal"
    
    n = n1
    total_pairs = n * (n - 1) // 2
    
    f_true = np.empty(total_pairs, dtype=np.float32).flatten()
    f_pred = np.empty(total_pairs, dtype=np.float32).flatten()
    
    idx = 0
    for i in range(n-1):
        f_true[idx:idx+n-i-1] = y_true[i+1:] - y_true[i]
        f_pred[idx:idx+n-i-1] = y_pred[i+1:] - y_pred[i]
        idx += n - i - 1
    
    acc_arr = np.multiply(np.sign(f_true), np.sign(f_pred))
    acc_arr[np.abs(f_true) <= y_true_thresh] = 1.0
    
    return np.sum(acc_arr >= 0), total_pairs


class FFNN1(nn.Module):
    def __init__(self, input_size=2048, hidden_sizes=[300, 300, 300, 300, 300], output_size=1):
        super(FFNN1, self).__init__()
        
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size
        
        self.layers = nn.ModuleList()
        for i, hidden_size in enumerate(self.hidden_sizes):
            if i == 0:
                self.layers.append(nn.Linear(self.input_size, hidden_size))
            else:
                self.layers.append(nn.Linear(self.hidden_sizes[i - 1], hidden_size))
            self.layers.append(nn.ReLU())
        self.layers.append(nn.Linear(self.hidden_sizes[-1], self.output_size))
        
        self.model = nn.Sequential(*self.layers)
    
    def forward(self, x):
        return self.model(x)
    

## reset all pytorch weights before training
def weight_reset(m):
    reset_parameters = getattr(m, "reset_parameters", None)
    if callable(reset_parameters):
        m.reset_parameters()


def train_save_single_model(model, save_path_kwd, X_train, X_val, y_train, y_val, batch_size=128, learning_rate=0.001, max_epochs=300, patience=300, warmup_epochs=0):
    # model.apply(weight_reset)
    summary(model)
    model = model.to(device)
    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
    X_val = torch.tensor(X_val, dtype=torch.float32)
    y_val = torch.tensor(y_val, dtype=torch.float32).view(-1, 1)
    
    y_train_mean = torch.mean(y_train)
    y_train_std = torch.std(y_train)
    y_train = (y_train - y_train_mean) / y_train_std
    y_val = (y_val - y_train_mean) / y_train_std
    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    criterion = my_hinge_loss_v2
    # criterion = my_hinge_loss_v1
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    best_val_loss = np.inf
    best_model_state_dict = None
    best_epoch = 0
    patience_counter = 0
    train_loss_list = []
    val_loss_list = []
    for epoch in tqdm(range(max_epochs)):
        model.train()
        train_loss = 0.0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            y_pred = model(X_batch)
            loss = criterion(y_pred, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * X_batch.size(0)
        train_loss /= len(train_loader.dataset)
        train_loss_list.append(train_loss)
        
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)
                y_pred = model(X_batch)
                loss = criterion(y_pred, y_batch)
                val_loss += loss.item() * X_batch.size(0)
            val_loss /= len(val_loader.dataset)
            val_loss_list.append(val_loss)
        
        if val_loss < best_val_loss and epoch >= warmup_epochs:
            best_val_loss = val_loss
            best_model_state_dict = deepcopy(model.cpu().state_dict())
            model.to(device)
            best_epoch = epoch
            patience_counter = 0
        else:
            patience_counter += 1
        if patience_counter >= patience and epoch >= warmup_epochs:
            print(f"Early stopping: epoch {epoch}, best epoch {best_epoch}, best val loss {best_val_loss}")
            break
        if epoch % 10 == 0:
            print(f"Epoch: [{epoch}/{max_epochs}], train loss {train_loss}, val loss {val_loss}")
            
    print(f"Best epoch: {best_epoch}, best val loss {best_val_loss}")
    model_info_dict = {
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "train_loss_list": train_loss_list,
        "val_loss_list": val_loss_list,
        "y_train_mean": y_train_mean.item(),
        "y_train_std": y_train_std.item()
    }
    model_state_dict_path = save_path_kwd + "_best_state_dict.pth"
    model_info_dict_path = save_path_kwd + "_info.json"
    torch.save(best_model_state_dict, model_state_dict_path)
    json.dump(model_info_dict, open(model_info_dict_path, "w"))
    model = model.cpu()
    return model_state_dict_path, model_info_dict_path


def train_save_k_fold_model(model, save_dir, X, y, k=10, batch_size=128, learning_rate=0.001, max_epochs=300, patience=300, warmup_epochs=0, smi_list=None):
    ## k_max = 10
    assert k <= 10, "k should be less than or equal to 10"
    model_init_state_dict = deepcopy(model.state_dict())
    
    X = np.array(X)
    y = np.array(y).flatten()
    if smi_list is not None:
        smi_list_loc = deepcopy(smi_list)
        smi_list_loc = np.array(smi_list_loc).flatten()
    ## shuffle data
    n = len(y)
    indices = np.arange(n)
    np.random.shuffle(indices)
    
    X = X[indices]
    y = y[indices]
    if smi_list is not None:
        smi_list_loc = smi_list_loc[indices]
    
    indices = np.arange(n)
    if smi_list is not None:
        k_fold_train_smi_dict = {}
        k_fold_val_smi_dict = {}
    
    for i in range(k):
        print("Fold:", i+1)
        valid_indices = indices[i::10]
        train_indices = np.setdiff1d(indices, valid_indices)
        X_train = X[train_indices]
        y_train = y[train_indices]
        X_val = X[valid_indices]
        y_val = y[valid_indices]
        if smi_list is not None:
            k_fold_train_smi_dict[i] = list(smi_list_loc[train_indices])
            k_fold_val_smi_dict[i] = list(smi_list_loc[valid_indices])
        save_path_kwd = os.path.join(save_dir, f"model_fold_{i}")
        model_i = model.load_state_dict(deepcopy(model_init_state_dict))
        model_i = model.to(device)
        train_save_single_model(
            model=model_i,
            save_path_kwd=save_path_kwd,
            X_train=X_train,
            X_val=X_val,
            y_train=y_train,
            y_val=y_val,
            batch_size=batch_size,
            learning_rate=learning_rate,
            max_epochs=max_epochs,
            patience=patience,
            warmup_epochs=warmup_epochs
        )
    if smi_list is not None:
        return k_fold_train_smi_dict, k_fold_val_smi_dict
    return


def load_eval_single_model(model, model_state_dict_path, X):
    model.load_state_dict(torch.load(model_state_dict_path))
    model = model.to(device)
    model.eval()
    X = torch.tensor(np.array(X), dtype=torch.float32).to(device)
    y_pred = model(X).detach().cpu().numpy().flatten()
    return y_pred


def load_eval_k_fold_model(model, model_save_dir, X, k=10):
    ## k_max = 10
    assert k <= 10, "k should be less than or equal to 10"
    # model = model.to(device)
    # model.eval()
    fold_y_pred_dict = {}
    for i in range(k):
        model_state_dict_path = os.path.join(model_save_dir, f"model_fold_{i}_best_state_dict.pth")
        y_pred = load_eval_single_model(model, model_state_dict_path, X)
        fold_y_pred_dict[i] = y_pred
    return fold_y_pred_dict

