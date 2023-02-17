import os

import pandas as pd
from tensorflow import keras
import tensorflow.keras.backend as K
import tensorflow as tf
from tensorflow.keras.callbacks import Callback

from datetime import datetime


def fit_model(self, cnn, dg_train, dg_valid, call_back, delayed_early_stop, cprofile=False):

    # customize optimizer
    if self.optimizer_str == 'adam':  # learn_rate = 0.001
        optimizer = keras.optimizers.Adam(self.learn_rate, decay=self.decay_rate)
        print('adam', self.learn_rate, self.decay_rate)
    elif self.optimizer_str == 'SGD':  # learn_rate = 0.1
        optimizer = keras.optimizers.SGD(learning_rate=self.learn_rate, decay=self.decay_rate)
        print('SGD', self.learn_rate, self.decay_rate)
    else:
        optimizer = keras.optimizers.Adam(self.learn_rate)

    # delayed early stopping
    if delayed_early_stop == True:
        print('delayed early stopping')
        callback = CustomStopper(monitor='val_loss', patience=self.patience,
                                 restore_best_weights=True, min_delta=0,
                                 start_epoch=self.start_epoch)
    else:
        callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=self.patience,
                                                    restore_best_weights=True, min_delta=0.001)

    # compile models
    cnn.compile(loss=keras.losses.CategoricalCrossentropy(),
                metrics=['accuracy'], optimizer=optimizer)

    # fit models
    if call_back:
        if cprofile:
            # Create a TensorBoard callback
            logs = "logs/fit/" + datetime.now().strftime("%Y%m%d-%H%M%S")
            tboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logs,
                                                             histogram_freq=1,
                                                             profile_batch='5,10',
                                                             write_images=True)
            self.hist = cnn.fit(dg_train, epochs=self.ep, validation_data=dg_valid, callbacks=[callback, tboard_callback])
        else:
            self.hist = cnn.fit(dg_train, epochs=self.ep, validation_data=dg_valid, callbacks=[callback, PrintLearningRate()])
    else:
        self.hist = cnn.fit(dg_train, epochs=self.ep, validation_data=dg_valid,
                       callbacks=[PrintLearningRate()])


class CustomStopper(keras.callbacks.EarlyStopping):
    def __init__(self, monitor='val_loss', min_delta=0, patience=0,
                 verbose=0, mode='auto', baseline=None, restore_best_weights=False,
                 start_epoch=5):  # add argument for starting epoch
        super(CustomStopper, self).__init__()

        # hand over values from the arguments
        self.monitor = monitor
        self.min_delta = min_delta
        self.patience = patience
        self.verbose = verbose
        self.mode = mode
        self.baseline = baseline
        self.restore_best_weights = restore_best_weights
        self.start_epoch = start_epoch

    def on_epoch_end(self, epoch, logs=None):
        if epoch >= self.start_epoch:  # since epoch count starts from 0.
            # print("lagged early stop")
            super().on_epoch_end(epoch, logs)


# print learning rate after every epoch to monitor how the learning rate decay works
class PrintLearningRate(Callback):
    def __init__(self):
        pass

    def on_epoch_begin(self, epoch, logs=None):
        lr = K.eval(self.model.optimizer._decayed_lr(tf.float64))
        print("\nLearning rate at epoch {} is {}".format(epoch, lr))


def save_model_info(model, model_name, v, lead_time, dg_train_len, fold_no, fct_train_keys, time, folder=''):

    if model.call_back == True:
        if model.delayed_early_stop == False:
            early_stop = 'normal'
        else:
            early_stop = 'delayed'
    else:
        early_stop = False
    
    setup = pd.DataFrame(
                 columns=['model_name', 'target_variable', 'lead_time', 'model_architecture',
                          'train_patches', 'fold_no', 'train_time', 'epochs', 'batch_size',
                          'n_trainbatches', 'callback', 'features',
                           'output_dims', 'input_dims', 'patch_stride', 'patch_na', 'region',
                          'optimizer', 'learn_rate', 'learn_decay', 'early_stopping',
                          'filters', 'hidden_nodes', 'n_blocks', 'dropout_rate', 'batch_norm',
                          'radius_basis_func'])

    # params saved as class params
    d = model.__dict__ 
    
    # some cols got new names, so dataframe column names do not match the keys
    d1 = {'radius_basis_func': 'basis_rad', 'batch_size': 'bs', 'epochs':'ep',
          'batch_norm': 'bn', 'optimizer': 'optimizer_str',
          'learn_decay': 'decay_rate'}
    
    # some additional info that needs to be saved
    add_params = {'model_name': model_name, 'target_variable': v, 
                  'lead_time': lead_time, 'n_trainbatches': dg_train_len,
                  'callback': len(model.hist.history.get('accuracy')) - model.patience,
                  'fold_no': fold_no, 'early_stopping': early_stop,
                  'features': fct_train_keys, 'train_time': time}
    
    for k in setup.columns:
        if k in d.keys():
            setup.at[0, k] = d[k]
        elif k in d1.keys():
            if d1[k] in d.keys():
                setup.at[0, k] = d[d1[k]]
        elif k in add_params.keys():
            setup.at[0, k] = add_params[k]
    setup = setup.fillna('-')  
        
    results_ = pd.DataFrame({'accuracy': [model.hist.history.get('accuracy')],
                             'val_accuracy': [model.hist.history.get('val_accuracy')],
                             'loss': [model.hist.history.get('loss')],
                             'val_loss': [model.hist.history.get('val_loss')]})

    results = pd.concat([setup, results_], axis=1)

    if os.path.isfile(f'{folder}/results.csv'):
        results.to_csv(f'{folder}/results.csv', sep=';', index=False, mode='a', header=None)
    else:
        results.to_csv(f'{folder}/results.csv', sep=';', index=False, mode='a')
