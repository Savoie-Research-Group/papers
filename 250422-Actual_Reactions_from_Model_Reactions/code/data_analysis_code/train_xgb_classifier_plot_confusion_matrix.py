"""
    Date Modified: 2025/22/04
    Author: Veerupaksh (Veeru) Singla (singla2@purdue.edu)
    Description: train xgboost classifier on drfp (https://doi.org/10.1039/D1DD00006C) for multiclass classification from ar to mr and plot confusion matrix. hyperparameter optimization using hyperopt. plot train/test confusion matrix.
"""


import os
import json
import numpy as np
import pickle
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (accuracy_score, classification_report, precision_score,
                             confusion_matrix, ConfusionMatrixDisplay, f1_score, recall_score,
                             roc_auc_score)
from imblearn.over_sampling import SMOTE


this_script_dir = os.path.dirname(os.path.realpath(__file__))
os.chdir(this_script_dir)


from utils import *


def make_stratified_splits(mr_ar_drfp_dict, x_0_train, y_0_train, x_1, y_1, n_splits=1, train_valid_test_fr=[0.8,0.1,0.1], random_state=42):
    assert len(train_valid_test_fr) == 3
    tvt_sum = sum(train_valid_test_fr)
    train_valid_test_fr = [i/tvt_sum for i in train_valid_test_fr]  ## normalize to sum to 1.
    
    sss1 = StratifiedShuffleSplit(n_splits=n_splits, test_size=train_valid_test_fr[1]+train_valid_test_fr[2], random_state=random_state) ## split into train and test. n_splits may be increased for cv.
    stratified_train_idx_list = []
    stratified_test_idx_list_temp = []
    stratified_test_idx_list = []
    stratified_valid_idx_list = []
    for i, (train_idx, test_idx) in enumerate(sss1.split(x_1, y_1)):
        stratified_train_idx_list.append([j for j in train_idx])
        stratified_test_idx_list_temp.append([j for j in test_idx])
    sss2 = StratifiedShuffleSplit(n_splits=1, test_size=train_valid_test_fr[2]/(train_valid_test_fr[1]+train_valid_test_fr[2]), random_state=random_state) ## split earlier test into valid and test. n_splits=1 always.
    for i, test_idx_temp in enumerate(stratified_test_idx_list_temp):
        for j, (valid_idx, test_idx) in enumerate(sss2.split([x_1[k] for k in test_idx_temp], [y_1[k] for k in test_idx_temp])):
            stratified_valid_idx_list.append([test_idx_temp[k] for k in valid_idx])
            stratified_test_idx_list.append([test_idx_temp[k] for k in test_idx])
            
    stratified_train_x = []
    stratified_train_y = []
    stratified_valid_x = []
    stratified_valid_y = []
    stratified_test_x = []
    stratified_test_y = []
    for i, (train_idx, valid_idx, test_idx) in enumerate(zip(stratified_train_idx_list, stratified_valid_idx_list, stratified_test_idx_list)):
        stratified_train_x.append([j for j in x_0_train] + [x_1[j] for j in train_idx])
        stratified_train_y.append([j for j in y_0_train] + [y_1[j] for j in train_idx])
        stratified_valid_x.append([x_1[j] for j in valid_idx])
        stratified_valid_y.append([y_1[j] for j in valid_idx])
        stratified_test_x.append([x_1[j] for j in test_idx])
        stratified_test_y.append([y_1[j] for j in test_idx])
        
    stratified_train_X = []
    stratified_valid_X = []
    stratified_test_X = []
    for i, (train_x, valid_x, test_x) in enumerate(zip(stratified_train_x, stratified_valid_x, stratified_test_x)):
        stratified_train_X.append([mr_ar_drfp_dict[rxn] for rxn in train_x])
        stratified_valid_X.append([mr_ar_drfp_dict[rxn] for rxn in valid_x])
        stratified_test_X.append([mr_ar_drfp_dict[rxn] for rxn in test_x])
        
    return stratified_train_x, stratified_train_X, stratified_train_y, stratified_valid_x, stratified_valid_X, stratified_valid_y, stratified_test_x, stratified_test_X, stratified_test_y


def train_xgb_classifier(X_train, y_train, X_valid, y_valid, X_test, y_test, random_state=42, params=None):
    X_train = np.array([np.array(x) for x in list(X_train) + list(X_valid)])
    y_train = np.array(list(y_train) + list(y_valid))
    
    X_test = np.array([np.array(x) for x in X_test])
    y_test = np.array(y_test)
    
    # X_train, y_train = SMOTE(random_state=random_state).fit_resample(X_train, y_train)
    
    print(X_train.shape, y_train.shape)
    
    nproc = len(os.sched_getaffinity(0))
    if params is None:
        class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
        class_weight_dict = dict(zip(np.unique(y_train), class_weights))
        
        params = {
            'objective': 'multi:softprob',
            'num_class': len(np.unique(y_train)),
            'eval_metric': 'mlogloss',
            'verbosity': 2,
            'eta': 0.1,
            'max_depth': 6,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'nthread': nproc,
            'seed': random_state,
            'scale_pos_weight': class_weight_dict,
            'max_delta_step': 1,
            'use_label_encoder': False
        }
    else:
        if 'seed' not in params:
            params['seed'] = random_state
        
        params['nthread'] = nproc
        params['verbosity'] = 2 ## force verbosity to 2.
            
    xgb_clf = XGBClassifier(**params)
    xgb_clf.fit(X_train, y_train)
    
    return xgb_clf


def eval_xgb_classifier(xgb_clf, X_train, y_train, X_valid, y_valid, X_test, y_test):
    X_train = np.array([np.array(x) for x in list(X_train) + list(X_valid)])
    y_train = np.array(list(y_train) + list(y_valid))
    
    X_test = np.array([np.array(x) for x in X_test])
    y_test = np.array(y_test)
    
    y_train_pred = xgb_clf.predict(X_train)
    y_test_pred = xgb_clf.predict(X_test)
    print(y_train_pred.shape, y_test_pred.shape)
    
    train_report = classification_report(y_train, y_train_pred, target_names=mr_list)
    test_report = classification_report(y_test, y_test_pred, target_names=mr_list)
    
    train_acc = accuracy_score(y_train, y_train_pred)
    test_acc = accuracy_score(y_test, y_test_pred)
    
    train_precision = precision_score(y_train, y_train_pred, average='macro')
    test_precision = precision_score(y_test, y_test_pred, average='macro')
    
    train_conf_mat = confusion_matrix(y_train, y_train_pred, normalize='true', labels=np.arange(len(mr_list)))
    test_conf_mat = confusion_matrix(y_test, y_test_pred, normalize='true', labels=np.arange(len(mr_list)))
    
    train_f1 = f1_score(y_train, y_train_pred, average='macro')
    test_f1 = f1_score(y_test, y_test_pred, average='macro')
    
    train_recall = recall_score(y_train, y_train_pred, average='macro')
    test_recall = recall_score(y_test, y_test_pred, average='macro')
    
    train_roc_auc = roc_auc_score(y_train, xgb_clf.predict_proba(X_train), average='macro', multi_class='ovr')
    test_roc_auc = roc_auc_score(y_test, xgb_clf.predict_proba(X_test), average='macro', multi_class='ovr')
    
    return train_report, test_report, train_acc, test_acc, train_precision, test_precision, train_conf_mat, test_conf_mat, train_f1, test_f1, train_recall, test_recall, train_roc_auc, test_roc_auc


def plot_confusion_matrix(conf_mat, save_name):
    ## since we have too many classes, we will not plot the class names. only the confusion matrix.
    plt.clf()
    fig = plt.figure(figsize=(12, 12))
    gs = fig.add_gridspec(1, 1, hspace=0.0, wspace=0)
    
    ax1 = fig.add_subplot(gs[0, 0])
    
    disp = ConfusionMatrixDisplay(confusion_matrix=conf_mat)
    disp.plot(ax=ax1, include_values=False, cmap='Purples')
    ax1.set_xlabel("Predicted", fontsize=14)
    ax1.set_ylabel("True", fontsize=14)
    ax1.set_xticks([])
    ax1.set_yticks([])
    plt.tight_layout()
    plt.savefig(transparent=True, fname=save_name + ".pdf", dpi=300, bbox_inches='tight', pad_inches=0.005)
    plt.close()


def main():
    mr_smi_dict = json.load(open(os.path.join(data_path_main, "mr_smi_dict.json")))
    ar_smi_dict = json.load(open(os.path.join(data_path_main, "ar_smi_dict.json")))
    mr_atom_mapped_smi_dict = json.load(open(os.path.join(data_path_main, "mr_atom_mapped_smi_dict.json")))
    ar_atom_mapped_smi_dict = json.load(open(os.path.join(data_path_main, "ar_atom_mapped_smi_dict.json")))
    ar_mr_dict = json.load(open(os.path.join(data_path_main, "ar_mr_dict.json")))
    mr_ar_list_dict = json.load(open(os.path.join(data_path_main, "mr_ar_list_dict.json")))
    
    mr_ar_count_dict = {mr: len(ar_list) for mr, ar_list in mr_ar_list_dict.items()}
    
    global mr_list
    
    mr_list = list(mr_smi_dict.keys())
    ar_list = list(ar_smi_dict.keys())
    
    mr_smi_list = [mr_smi_dict[mr] for mr in mr_list]
    ar_smi_list = [ar_smi_dict[ar] for ar in ar_list]
    
    mr_atom_mapped_smi_list = [mr_atom_mapped_smi_dict[mr] for mr in mr_list]
    ar_atom_mapped_smi_list = [ar_atom_mapped_smi_dict[ar] for ar in ar_list]
    
    # ## generate drfp and save. uncomment if have to be made again.
    # mr_drfp_list = get_rxn_drfp(mr_atom_mapped_smi_list)
    # ar_drfp_list = get_rxn_drfp(ar_atom_mapped_smi_list)
    # mr_drfp_dict = {mr: drfp for mr, drfp in zip(mr_list, mr_drfp_list)}
    # ar_drfp_dict = {ar: drfp for ar, drfp in zip(ar_list, ar_drfp_list)}
    # pickle.dump(mr_drfp_dict, open(os.path.join(analyses_path_main, "mr_drfp_dict.p"), "wb"))
    # pickle.dump(ar_drfp_dict, open(os.path.join(analyses_path_main, "ar_drfp_dict.p"), "wb"))
    
    mr_drfp_dict = pickle.load(open(os.path.join(analyses_path_main, "mr_drfp_dict.p"), "rb"))
    ar_drfp_dict = pickle.load(open(os.path.join(analyses_path_main, "ar_drfp_dict.p"), "rb"))
    mr_ar_drfp_dict = {**mr_drfp_dict, **ar_drfp_dict}
    
    ## keep only mr with at least 4 ar for stratified splits. so that test and valid have at least 1 ar for an 60:20:20 split.
    mr_list = [mr for mr in mr_list if mr_ar_count_dict[mr] >= 4]
    ar_list = [ar for ar in ar_list if "_".join(ar.split("_")[:3]) in mr_list]
    
    mr_smi_list = [mr_smi_dict[mr] for mr in mr_list]
    ar_smi_list = [ar_smi_dict[ar] for ar in ar_list]
    
    ## mr names are the output classes for multiclass classification. mrs will always be in training. ars to be split in train, valid, test.
    mr_i_dict = {mr: i for i, mr in enumerate(mr_list)} # mr to index. index is the class label.
    i_mr_dict = {i: mr for i, mr in enumerate(mr_list)} # index to mr. index is the class label.
    
    x_0_train = [mr for mr in mr_list]
    y_0_train = [mr_i_dict[mr] for mr in mr_list]
    
    x_1 = [ar for ar in ar_list]
    y_1 = [mr_i_dict[ar_mr_dict[ar]] for ar in ar_list]
    
    ## make stratified splits.
    stratified_train_x, stratified_train_X, stratified_train_y, stratified_valid_x, stratified_valid_X, stratified_valid_y, stratified_test_x, stratified_test_X, stratified_test_y = make_stratified_splits(
        mr_ar_drfp_dict, x_0_train, y_0_train, x_1, y_1, n_splits=1, train_valid_test_fr=[0.6,0.2,0.2], random_state=42)
    
    
    # ## train xgboost classifier.
    # xgb_clf  = train_xgb_classifier(stratified_train_X[0], stratified_train_y[0], stratified_valid_X[0], stratified_valid_y[0],
    #                                  stratified_test_X[0], stratified_test_y[0], random_state=42)
    
    # ## save xgboost classifier.
    # xgb_clf.save_model(os.path.join(analyses_path_main, "best_xgb_model.json"))
    
    ## load xgboost classifier.
    xgb_clf = XGBClassifier()
    xgb_clf.load_model(os.path.join(analyses_path_main, "best_xgb_model.json"))
    
    ## evaluate xgboost classifier.
    train_report, test_report, train_acc, test_acc, train_precision, test_precision, train_conf_mat, test_conf_mat, train_f1, test_f1, train_recall, test_recall, train_roc_auc, test_roc_auc = eval_xgb_classifier(
        xgb_clf, stratified_train_X[0], stratified_train_y[0], stratified_valid_X[0], stratified_valid_y[0], stratified_test_X[0], stratified_test_y[0])
    
    ## save train and test reports.
    with open(os.path.join(analyses_path_main, "xgb_train_report.txt"), "w") as f:
        f.write(train_report)
    with open(os.path.join(analyses_path_main, "xgb_test_report.txt"), "w") as f:
        f.write(test_report)
    
    # ## these error lists by themselves aren't conclusive since some values while not being 1 are very close.
    # ## we will compare the common labels between train and test reports only. sorted manually once auto lists are generated.
    # with open(os.path.join(analyses_path_main, "xgb_train_report_errors.txt"), "w") as f:
    #     for line in train_report.split("\n"):
    #         if "1.00      1.00      1.00" not in line:
    #             f.write(line + "\n")
    # with open(os.path.join(analyses_path_main, "xgb_test_report_errors.txt"), "w") as f:
    #     for line in test_report.split("\n"):
    #         if "1.00      1.00      1.00" not in line:
    #             f.write(line + "\n")
    
    print("Train Accuracy: ", train_acc)
    print("Test Accuracy: ", test_acc)
    print("Train Precision: ", train_precision)
    print("Test Precision: ", test_precision)
    print("Train F1: ", train_f1)
    print("Test F1: ", test_f1)
    print("Train Recall: ", train_recall)
    print("Test Recall: ", test_recall)
    print("Train ROC AUC: ", train_roc_auc)
    print("Test ROC AUC: ", test_roc_auc)
    
    ## plot train and test confusion matrices.
    plot_confusion_matrix(train_conf_mat, os.path.join(analyses_path_main, "xgb_train_confusion_matrix"))
    plot_confusion_matrix(test_conf_mat, os.path.join(analyses_path_main, "xgb_test_confusion_matrix"))
    
    return


if __name__ == "__main__":
    main()
