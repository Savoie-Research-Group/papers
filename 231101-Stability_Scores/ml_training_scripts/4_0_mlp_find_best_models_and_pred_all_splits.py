"""
    Date Modified: 2024/01/23
    Author: Veerupaksh (Veeru) Singla (singla2@purdue.edu)
    Description: finds the best mlp model with lowest val loss from the all splits training log files. then loads the best model pickel and predicts on all splits
"""


import os
import json
import numpy as np
import tensorflow as tf
from tensorflow import keras
import keras.backend as K
from keras.models import load_model


this_script_directory = os.path.dirname(os.path.realpath(__file__))
os.chdir(this_script_directory)


def read_nn_log(nn_log_file):
    with open(nn_log_file, "r") as f:
        lines = f.readlines()
    lines = [line.strip() for line in lines]
    return lines


def get_best_model(lines):
    epoch_list = [0]
    val_loss_list = []
    best_model_val_loss = None
    best_model_epoch = None
    
    for line in lines:
        if "val_loss:" in line:
            val_loss_list.append(float(line.split("val_loss: ")[-1]))
            epoch_list.append(epoch_list[-1] + 1)
    
    epoch_list = epoch_list[1:]
            
    for i in range(len(val_loss_list)):
        if best_model_val_loss is None or val_loss_list[i] < best_model_val_loss:
            best_model_val_loss = val_loss_list[i]
            best_model_epoch = epoch_list[i]
    
    return best_model_val_loss, best_model_epoch


batch_size_ = 128
def hinge_loss_(y_true_, y_pred_):
    y_pred_ = tf.convert_to_tensor(y_pred_)
    y_true_ = tf.cast(y_true_, y_pred_.dtype)

    f_0_ = y_true_ - tf.roll(y_true_, -1 * 1, axis=0)
    f_1_ = y_pred_ - tf.roll(y_pred_, -1 * 1, axis=0)
    for i_ in range(2, int(batch_size_ / 2) + 1):
        f_0_ = K.concatenate([f_0_, y_true_ - tf.roll(y_true_, -1 * i_, axis=0)], axis=0)
        f_1_ = K.concatenate([f_1_, y_pred_ - tf.roll(y_pred_, -1 * i_, axis=0)], axis=0)

    # return K.sum(tf.maximum(f_0_ - f_1_, 0.), axis=-1)
    return K.sum(tf.maximum(tf.multiply(f_0_ - f_1_, tf.sign(f_0_)), 0.), axis=-1)


def main():

    all_data_paths = [
        "../data/splits/random_90-10_train-test_split",
        "../data/splits/4_or_less_core_branches_train_rest_test",
        "../data/splits/6_or_less_total_branches_train_rest_test",
        "../data/splits/backbone_smaller_than_equal_10_train_rest_test",
        "../data/splits/till_c16_train_c17_test"
    ]
    
    ## update with where the pickles are saved
    all_train_paths = [
        "../pretrained_models/mlp_models/random_90-10_train-test_split",
        "../pretrained_models/mlp_models/4_or_less_core_branches_train_rest_test",
        "../pretrained_models/mlp_models/6_or_less_total_branches_train_rest_test",
        "../pretrained_models/mlp_models/backbone_smaller_than_equal_10_train_rest_test",
        "../pretrained_models/mlp_models/till_c16_train_c17_test"
    ]
    
    for i in range(len(all_data_paths)):
        data_path = all_data_paths[i]
        train_path = all_train_paths[i]
        
        ## change log.txt to whatever the log file is called
        nn_log_file = os.path.join(train_path, "log.txt")
        lines = read_nn_log(nn_log_file)
        best_model_val_loss, best_model_epoch = get_best_model(lines)
        
        with open(os.path.join(train_path, "best_model_epoch_val_loss.json"), "w") as f:
            json.dump({f"model_{best_model_epoch:03d}.h5": [best_model_epoch, best_model_val_loss]}, f)
        
        best_model = load_model(os.path.join(train_path, f"model_{best_model_epoch:03d}.h5"), custom_objects={'hinge_loss_': hinge_loss_})
        
        x_fp_train = np.load(os.path.join(data_path, "x_fp_train.npy"))
        x_fp_test = np.load(os.path.join(data_path, "x_fp_test.npy"))
        
        y_pred_train = best_model.predict(x_fp_train)
        y_pred_test = best_model.predict(x_fp_test)
        
        np.save(os.path.join(train_path, "y_pred_train.npy"), y_pred_train)
        np.save(os.path.join(train_path, "y_pred_test.npy"), y_pred_test)
        

if __name__ == "__main__":
    main()
