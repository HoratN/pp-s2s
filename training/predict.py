"""
Script for creating predictions from trained models

@author: horatn
"""

import tensorflow.keras as keras
import tensorflow as tf

import numpy as np
import pandas as pd

import xarray as xr
xr.set_options(display_style='text')

from utils.helper_load_data import get_data, load_data, clean_coords
from utils.helper_preprocess import preprocess_input
from utils.helper_predict import slide_predict, global_predict

from utils.paths import get_paths

import os

import warnings
warnings.simplefilter("ignore")

#%%
# set env random seeds
import random as rn
os.environ['TF_CUDNN_DETERMINISTIC'] = 'true'
os.environ['TF_DETERMINISTIC_OPS'] = 'true'
os.environ['PYTHONHASHSEED'] = '0'
seed = 10
tf.random.set_seed(seed)
np.random.seed(seed)
rn.seed(seed)

#%%

# =============================================================================
# directories
# =============================================================================

# folder where models are, and where the submissions should be saved
folder = 'main'

# file system
path = 'local'  # 'server'
cache_path, path_add_vars, path_model, path_pred, path_results = get_paths(path)

#%%

# =============================================================================
# create predictions (load model and setup specs, load input data, load model,
# predict, save predictions)
# =============================================================================

print('create predictions')
# run through saved models and create predictions if necessary
for filename in os.listdir(f'{path_model}{folder}'):
    path_to_model = os.path.join(path_model, folder, filename)
    name = filename

    if os.path.isfile(f'{path_pred}{folder}/global_pred_{name}_allyears_raw.nc') and  \
            os.path.isfile(f'{path_pred}{folder}/global_pred_{name}_2020_raw.nc'):
        print("Predictions exist")
        continue
    if (filename == '.ipynb_checkpoints') or (filename in ['best_models', 'experiments', 'embeddings',
                                                           'feature_selection', 'unet_param_opt', 'early_models']):
        continue

    print(name)
#%%

# =============================================================================
# initialize setup/model parameters
# =============================================================================

    # read params from filename
    print(filename)
    f_ = filename.split('_')

    var_list = f_[2:-2]
    print(var_list)

    v = f_[0]
    lead_time = int(f_[1])
    print(v, lead_time)

    # load additional settings from results.csv:
    params_all = pd.read_csv(f'{path_results}results.csv', delimiter=';')
    params = params_all.loc[params_all['model_name'] == filename]

    output_dims = params['output_dims'].values[0]
    input_dims = params['input_dims'].values[0]
    basis_rad = params['radius_basis_func'].values[0]
    train_patches = params['train_patches'].values[0]
    region = params['region'].values[0]
    model_architecture = params['model_architecture'].values[0]

    # standard param:
    n_bins = 3

#%%
# =============================================================================
# load and prepare data
# =============================================================================

    # load training data
    hind_2000_2019, obs_2000_2019, obs_2000_2019_terciled, mask = get_data(var_list, path)
    fct_train = hind_2000_2019.isel(lead_time=lead_time)

    # load 2020 data
    fct_2020 = load_data(data='forecast_2020', aggregation='biweekly',
                         path=path, var_list=var_list).isel(lead_time=lead_time)
    fct_2020 = clean_coords(fct_2020)[var_list]

    # preprocess input: compute and standardize features
    fct_train, fct_2020 = preprocess_input(fct_train, v, path, lead_time, valid=fct_2020)

    fct_train = fct_train.compute()
    fct_2020 = fct_2020.compute()

    # up to here preprocessing of global data

#%%
# =============================================================================
# load model
# =============================================================================

    cnn = keras.models.load_model(path_to_model, compile=False)
#%%
# =============================================================================
# predict for training data
# =============================================================================

    if train_patches == True:
        # predict globally using the trained local model
        da = slide_predict(fct_train, input_dims, output_dims, cnn, model_architecture, basis_rad, n_bins)
    else:
        da = global_predict(fct_train, cnn, region=region)

    # if folder does not yet exist, create one
    if os.path.isdir(f'{path_pred}{folder}') is False:
        os.mkdir(f'{path_pred}{folder}')
    da.to_dataset(name=v).to_netcdf(f'{path_pred}{folder}/global_pred_{name}_allyears_raw.nc')

#%%

# =============================================================================
# predict for 2020 data
# =============================================================================

    if train_patches == True:
        da_2020 = slide_predict(fct_2020, input_dims, output_dims, cnn, model_architecture, basis_rad, n_bins)
    else:
        da_2020 = global_predict(fct_2020, cnn, region=region)

    da_2020.to_dataset(name=v).to_netcdf(f'{path_pred}{folder}/global_pred_{name}_2020_raw.nc')
