import tensorflow as tf
import xarray as xr
import pandas as pd
import os
from datetime import datetime

from utils.paths import get_paths
from utils.helper_average_pred import get_mean, save_mean

path_data = 'server'  #'local'
folder = 'main'
cache_path, path_add_vars, path_model, path_pred, path_results = get_paths(path_data)


res = pd.read_csv(f'{path_results}results.csv', delimiter=';')
res = res.drop_duplicates()
res['model'] = res['model_architecture']
res.loc[(res['model_architecture'] == 'unet') & (res['train_patches'] == False), 'model'] = 'global UNet'
res.loc[(res['model_architecture'] == 'unet') & (res['train_patches'] == True), 'model'] = 'Patch-wise UNet'
res.loc[(res['model_architecture'] == 'basis_func'), 'model'] = 'CNN w. basis'
res.loc[(res['model_architecture'] == 'conv_trans'), 'model'] = 'CNN w. conv.'

models_all = res[['model_name', 'target_variable', 'lead_time', 'model_architecture',
                  'train_patches', 'fold_no', 'model', 'weighted_loss']]


for v in ['t2m', 'tp']:
    print(v)
    for lead in [0, 1]: # lead: int, lead time, 0 represents the shorter lead time of 14 days, 1 a lead time of 28 days
        print(lead)
        for weighted in [False, True]: # weighted: bool, indicating whether models where trained with or without weighted loss
            if weighted == False:
                folder_pred = folder
            else:
                folder_pred = f'weighted_{folder}'
            print(weighted)
            
            model_unique_names = ['global UNet', 'Patch-wise UNet', 'CNN w. basis',
                                  'CNN w. conv.']  # np.unique(models_all['model'])

            for model in model_unique_names:
                # select relevant predictions
                submodels = models_all.loc[(models_all['model'] == model)].reset_index(drop=True)
                if len(submodels) == 0:
                    continue
                df_models = submodels.loc[(submodels['target_variable'] == v) & (submodels['lead_time'] == lead)
                        & (submodels['weighted_loss'] == weighted)].reset_index(drop=True)
                            
                ds_mean = get_mean(df_models, model, path_pred, folder_pred)
                if ds_mean is None:
                    continue
                save_mean(ds_mean, v, lead, model, weighted, submodels['model_architecture'][0],
                          submodels['train_patches'][0], path_pred, f'{folder_pred}_mean', path_results)
