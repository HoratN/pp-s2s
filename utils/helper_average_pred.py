import tensorflow as tf
import xarray as xr
import pandas as pd
import os
from datetime import datetime


def get_mean(df_models, model, path_pred, folder):
    """
    average over predictions from the 10 models obtained by 10-fold CV

    Parameters
    ----------
    df_models: dataframe, obtained from results.csv
    model: string, model name
    path_pred: string, path to predictions
    folder: string, folder name of folder with predictions
    
    Returns
    -------
    Xarray DataSet, contains mean prediction

    """

    if len(df_models) == 0:
        return None

    ls_pred_inner = []
    ls_filenames = []
    for i in range(0, len(df_models)):
        filename = df_models['model_name'][i]
        fold_no = df_models['fold_no'][i]
        ls_pred_inner.append(xr.open_dataset(f'{path_pred}/{folder}/global_pred_{filename}_2020_raw.nc'
                                             ).assign_coords({'fold': fold_no, 'model': model,
                                                              'filename': filename}))
        ls_filenames.append(filename)

    string_filenames = ', '.join(ls_filenames)

    preds = xr.concat(ls_pred_inner, 'fold'
                        ).assign_coords({'filename': string_filenames}
                                        ).expand_dims('lead_time')

    return preds.mean('fold').assign_coords({'ens_size': len(preds.fold)})


def save_mean(ds, v, lead, model, weighted, model_architecture, train_patches, path_pred, folder, path_results):
    """
    save average prediction
    
    Parameters
    ----------
    ds: xarray DataSet, contains mean prediction
    v: string, target variable
    lead: int, lead time, 0 represents the shorter lead time of 14 days, 1 a lead time of 28 days
    model: string, model name
    weighted: bool, indicating whether models where trained with or without weighted loss
    model_architectures: string, model name used in the filename
    train_patches: string, either False or True indicating whether patch-wise training was used or not
    path_pred: string, path to predictions
    folder: string, folder name of folder with predictions
    path_results: string, path to csv
    
    """

    # create unique filename
    dateobj = datetime.now().date()
    mean_name = f'{v}_{lead}_{model_architecture}_{train_patches}_w{weighted}_{dateobj.day}{dateobj.month}{dateobj.year}'

    # save to file
    ds.to_netcdf(f'{path_pred}{folder}/mean_{mean_name}_2020_raw.nc')

    # save corresponding metadata
    params = []
    params.append([mean_name, v, lead, model,
                   model_architecture, train_patches,
                   weighted, ds.ens_size.values, ds.filename.values])
    df_mean_results = pd.DataFrame(params,
                                   columns=['model_name', 'target_variable', 'lead_time', 'model',
                                            'model_architecture', 'train_patches',
                                            'weighted_loss', 'number_of_folds', 'preds'])

    if len(df_mean_results) > 0:
        # append to file
        if os.path.isfile(f'{path_results}results_mean.csv'):
            df_mean_results.to_csv(f'{path_results}results_mean.csv', sep=';', index=False, mode='a', header=None)
        else:
            df_mean_results.to_csv(f'{path_results}results_mean.csv', sep=';', index=False, mode='a')
    else:
        print('no predictions available')
