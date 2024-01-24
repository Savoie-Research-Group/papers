"""
    Date Modified: 2024/01/23
    Author: Veerupaksh (Veeru) Singla (singla2@purdue.edu)
    Description: Custom hinge loss function for tensorflow (for our MLP model) and pytorch (for Chemprop).
"""

import tensorflow as tf
from tensorflow import keras
import keras.backend as K
import torch


## tensorflow. needs batch_size_ to be defined. natch_size_ may be replaced by tf.shape(y_true_)[0]
def hinge_loss_(y_true_, y_pred_):
    y_pred_ = tf.convert_to_tensor(y_pred_)
    y_true_ = tf.cast(y_true_, y_pred_.dtype)

    f_0_ = y_true_ - tf.roll(y_true_, -1 * 1, axis=0)
    f_1_ = y_pred_ - tf.roll(y_pred_, -1 * 1, axis=0)
    for i_ in range(2, int(batch_size_ / 2) + 1):
        f_0_ = K.concatenate([f_0_, y_true_ - tf.roll(y_true_, -1 * i_, axis=0)], axis=0)
        f_1_ = K.concatenate([f_1_, y_pred_ - tf.roll(y_pred_, -1 * i_, axis=0)], axis=0)

    return K.sum(tf.maximum(tf.multiply(f_0_ - f_1_, tf.sign(f_0_)), 0.), axis=-1)


## pytorch (for Chemprop). all changes on "442a1602b670f173de166f987eab64396571ee98" hash of Chemprop (https://github.com/chemprop/chemprop)

# my_hinge_loss needs to be in chemprop/train/loss_functions.py. 
# the "regression" subdict in the supported_loss_functions dict in chemprop/train/loss_functions.py needs to be updated too to include my_hinge_loss: "my_hinge": my_hinge_loss,
def my_hinge_loss(preds_, targets_):
    
    torch_device = preds_.device

    f_0_ = targets_ - torch.roll(targets_, -1, dims=0)
    f_1_ = preds_ - torch.roll(preds_, -1, dims=0)
    
    for i_ in range(2, int(targets_.size(dim=0) / 2) + 1):
        f_0_ = torch.cat((f_0_, targets_ - torch.roll(targets_, -1 * i_, dims=0)), dim=0)
        f_1_ = torch.cat((f_1_, preds_ - torch.roll(preds_, -1 * i_, dims=0)), dim=0)
    
    return torch.sum(torch.maximum(torch.mul(f_0_ - f_1_, torch.sign(f_0_)), torch.zeros(f_0_.size(), device=torch_device)))

