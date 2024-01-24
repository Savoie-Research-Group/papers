"""
    Date Modified: 2024/01/23
    Author: Veerupaksh (Veeru) Singla (singla2@purdue.edu)
    Description: training mlp model on a given train-test split. this script saves checkpoint files for all epochs. nn_find_best_models_and_pred_all_splits.py finds the best model with lowest val loss from the log file
"""


import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
import keras.backend as K
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import argparse


this_script_directory = os.path.dirname(os.path.realpath(__file__))
os.chdir(this_script_directory)


## all settings default in argparse except for split-dir and model-dir. e.g. command to run this script and save log to find best model:
# python train_mlp.py --split-dir '../data/splits/random_90-10_train-test_split' --model-dir '../pretrained_models/mlp_models/random_90-10_train-test_split' > '../pretrained_models/mlp_models/random_90-10_train-test_split/log.txt'

parser = argparse.ArgumentParser(description='Train and test a neural network for stability score prediction.')
parser.add_argument('--fp-size', type=int, default=2048, help='Fingerprint size')
parser.add_argument('--batch-size', type=int, default=128, help='Batch size for training')
parser.add_argument('--learning-rate', type=float, default=0.001, help='Learning rate for optimizer')
parser.add_argument('--epochs', type=int, default=300, help='Number of epochs for training')
parser.add_argument('--split-dir', type=str, help='Directory containing train-test split data')
parser.add_argument('--model-dir', default="", type=str, help='Directory to save model and training logs')
parser.add_argument('--num-layers', type=int, default=5, help='Number of hidden layers')
parser.add_argument('--num-units', type=int, default=300, help='Number of units in each hidden layer')
args = parser.parse_args()

fp_size = args.fp_size
batch_size_ = args.batch_size
learning_rate_ = args.learning_rate
epochs_ = args.epochs
splits_path = args.split_dir
if args.model_dir == "":
    saving_folder_path = splits_path
else:
    saving_folder_path = args.model_dir

num_layers_ = args.num_layers
num_units_ = args.num_units

x_fp_train = np.load(splits_path + "/x_fp_train.npy")
x_fp_test = np.load(splits_path + "/x_fp_test.npy")
y_hl_train = np.load(splits_path + "/y_hl_train.npy")
y_hl_test = np.load(splits_path + "/y_hl_test.npy")


def hinge_loss_(y_true_, y_pred_):
    y_pred_ = tf.convert_to_tensor(y_pred_)
    y_true_ = tf.cast(y_true_, y_pred_.dtype)

    f_0_ = y_true_ - tf.roll(y_true_, -1 * 1, axis=0)
    f_1_ = y_pred_ - tf.roll(y_pred_, -1 * 1, axis=0)
    for i_ in range(2, int(batch_size_ / 2) + 1):
        f_0_ = K.concatenate([f_0_, y_true_ - tf.roll(y_true_, -1 * i_, axis=0)], axis=0)
        f_1_ = K.concatenate([f_1_, y_pred_ - tf.roll(y_pred_, -1 * i_, axis=0)], axis=0)

    return K.sum(tf.maximum(tf.multiply(f_0_ - f_1_, tf.sign(f_0_)), 0.), axis=-1)


def build_model():
    inputs = keras.Input(shape=(fp_size, ))

    dense = inputs
    for _ in range(num_layers_):
        dense = layers.Dense(num_units_, activation="relu", kernel_regularizer=tf.keras.regularizers.L1L2(l1=0.0, l2=0.0))(dense)

    outputs = layers.Dense(1)(dense)
    model = keras.Model(inputs=inputs, outputs=outputs, name="half_life_pred")
    
    # print model architecture to file
    keras.utils.plot_model(model, to_file="network_arch_.png", show_shapes=True, show_dtype=False, expand_nested=False, show_layer_names=False, dpi=600, show_layer_activations=True)
    
    model.compile(
        loss=hinge_loss_,
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate_, amsgrad=False)
    )

    return model
    
    
model = build_model()
checkpoint = keras.callbacks.ModelCheckpoint(os.path.join(saving_folder_path, 'model_{epoch:03d}.h5'), period=1)
history = model.fit(x_fp_train, y_hl_train, batch_size=batch_size_, epochs=epochs_, validation_split=0.1111, verbose=1, callbacks=[checkpoint])

# final model may not necessarily be the best model.
model.save(os.path.join(saving_folder_path, "model_final.h5"))

# plot learning curve: train and val loss vs epoch
plt.clf()
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validate'], loc='best')
plt.savefig(os.path.join(saving_folder_path, "train_val_loss.png"), bbox_inches='tight', dpi=600)
