"""
Script for model training using cross-validation

- available model architectures: CNN with basis functions, CNN with transposed
    convolutions, Unet
-  Unet architecture, can decide whether to train on small patches or global,
    the CNN architectures train patch-wise
- hyper-parameters for the models are set as class attributes

@author: horatn
"""

import tensorflow as tf
import numpy as np
import xarray as xr
xr.set_options(display_style='text')
from datetime import datetime
import time

from utils.helper_load_data import get_data, get_basis
from utils.helper_preprocess import preprocess_input
from utils.helper_datagenerator import DataGeneratorGlobal, DataGeneratorMultipatch

from utils.helper_train import fit_model, save_model_info

from models.unet import Unet
from models.standard_cnn import StandardCnn

import warnings
warnings.simplefilter("ignore")
import os
import random as rn

path_data = 'local'  # 'server'
folder = 'main'

# %%
# =============================================================================
# target and features
# =============================================================================
# =============================================================================
# for v in ['t2m', 'tp']:
#     for lead_time in [0,1]:
# =============================================================================

v = 't2m'  # 'tp'  #
lead_time = 0  # 1  #

# feature variables
if v == 't2m':
    ls_var_list = [[v, 'msl', 'gh850', 'gh500']]
else:
    ls_var_list = [[v, 'tcw', 'msl', 'gh850', 'gh500']]

var_list = ls_var_list[0]

# %%
# =============================================================================
# models architecture and hyper-parameters
# =============================================================================

model_architecture = 'unet'  # 'basis_func'# 'conv_trans'  #
train_patches = False  # True  #
if model_architecture != 'unet':
    train_patches = True
print(f'Architecture: {model_architecture} \npatch-wise: {train_patches}')

# load models and associated params
if model_architecture == 'unet':
    model = Unet(v, train_patches)
else:
    model = StandardCnn(v, model_architecture)

# set env random seeds
os.environ['TF_CUDNN_DETERMINISTIC'] = 'true'
os.environ['TF_DETERMINISTIC_OPS'] = 'true'
os.environ['PYTHONHASHSEED'] = '0'
seed = 10
tf.random.set_seed(seed)
np.random.seed(seed)
rn.seed(seed)

# %%
# =============================================================================
# prepare data
# =============================================================================

# load training data
hind_2000_2019, obs_2000_2019, obs_2000_2019_terciled, mask = get_data(var_list, path_data)
print(var_list)

# create datasets
fct_ = hind_2000_2019.isel(lead_time=lead_time)

obs_ = obs_2000_2019_terciled[v].isel(lead_time=lead_time)
obs_ = obs_.where(mask[v].isel(lead_time=lead_time).notnull())

if model_architecture == 'basis_func':
    # compute basis
    basis, lats, lons, n_xy, n_basis = get_basis(obs_.isel(category=0,
                                                             latitude=slice(0, model.output_dims),
                                                             longitude=slice(0, model.output_dims)), model.basis_rad)
    # climatology
    clim_probs = np.log(1 / 3) * np.ones((n_xy, model.n_bins))
    model.basis = basis
    model.clim_probs = clim_probs

# %%
# =============================================================================
# K-fold Cross Validation
# =============================================================================

years_per_fold = 2  # 5 --> 4-fold
for fold_no in range(0, 10):
    print(f'fold {fold_no}')
    start = time.time()
    #%%

    # train and validation split
    valid_indices = range(fold_no * years_per_fold * 53,
                          fold_no * years_per_fold * 53 + 53 * years_per_fold)
    train_indices = [i for i in range(53 * 20) if i not in valid_indices]

    # train
    fct_train = fct_.isel(forecast_time=train_indices)
    obs_train = obs_.isel(forecast_time=train_indices).compute()

    # validation
    fct_valid = fct_.isel(forecast_time=valid_indices)
    obs_valid = obs_.isel(forecast_time=valid_indices).compute()
    # .compute() should speed up data generation and training

    # preprocess input: compute and standardize features
    fct_train, fct_valid = preprocess_input(fct_train, v, path_data, lead_time, fct_valid)

    fct_train = fct_train.compute()
    fct_valid = fct_valid.compute()
    # up to here is preprocessing of global data

    # %%
    # create batches
    if train_patches == True:  # data augmentation using several patches
        print('create batches')
        dg_train = DataGeneratorMultipatch(fct_train, obs_train, model=model, input_dims=model.input_dims,
                                           output_dims=model.output_dims, mask_v=mask[v].isel(lead_time=0),
                                           region=model.region, batch_size=model.bs, load=True, reduce_sample_size=None,
                                           patch_stride=model.patch_stride, fraction=model.patch_na)

        dg_valid = DataGeneratorMultipatch(fct_valid, obs_valid, model=model, input_dims=model.input_dims,
                                           output_dims=model.output_dims, mask_v=mask[v].isel(lead_time=0),
                                           region=model.region, batch_size=model.bs, load=True, reduce_sample_size=None,
                                           patch_stride=model.patch_stride, fraction=model.patch_na)
        print('finished creating batches')
        if model_architecture == 'basis_func':  # drop since these attributes are very large
            model.basis = 0
            model.clim_probs = 0
    else:
        dg_train = DataGeneratorGlobal(fct_train, obs_train, region=model.region,
                                        batch_size=model.bs, load=True)
        dg_valid = DataGeneratorGlobal(fct_valid, obs_valid, region=model.region,
                                        batch_size=model.bs, load=True)


    #%%
    # =============================================================================
    # build and fit model
    # =============================================================================

    cnn = model.build_model(dg_train[0][0][0].shape)
    fit_model(model, cnn, dg_train, dg_valid, model.call_back, model.delayed_early_stop)

    end = time.time()
    # %%

    # =============================================================================
    # save model
    # =============================================================================

    # parameters for model naming
    dateobj = datetime.now().date()
    timeobj = datetime.now().time()
    var_list_name = var_list.copy()
    vars_name = '_'.join(var_list_name)

    # save model does not work if name is too long
    model_name = f'{v}_{lead_time}_{vars_name}_{dateobj.day}{dateobj.month}{dateobj.year}_{timeobj.hour}{timeobj.minute}'
    cnn.save(f'../results/trained_models/{folder}/{model_name}')

    # save additional information to model and training
    save_model_info(model, model_name, v, lead_time, len(dg_train), fold_no, list(fct_train.keys()), round(end - start),
                    folder='../results')

    if fold_no == 0:
        from tensorflow.keras.utils import plot_model
        if model.train_patches:
            train_mode = 'patchwise'
        else:
            train_mode = 'global'

        # save model architecture plot
        plot_model(cnn,
                   to_file=f'../results/architecture_{model.model_architecture}_{train_mode}.png')

        # save model summary to file
        with open(f'../results/architecture_{model.model_architecture}_{train_mode}.txt', 'w') as f:
            # Pass the file handle in as a lambda function to make it callable
            cnn.summary(print_fn=lambda x: f.write(x + '\n'))
