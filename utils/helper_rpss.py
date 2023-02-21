import numpy as np
import pandas as pd
import xarray as xr
xr.set_options(display_style='text')
import xskillscore as xs

import warnings
warnings.simplefilter("ignore")


def compute_rpss(name, agg_domain='global', cache_path='../template/data', folder='', path_pred='../submissions/'):
    """
    computes global average rpss for predictions of the model with the given name

    Parameters
    ----------
    name: string, name of model (of format {v}_{lead time}_{predictor vars}_{date}_{time}
    agg_domain: string, one of global, midlats, europe
    cache_path: string, path to observations / ground truth
    folder: string, folder where model and predictions are saved
    path_pred: string, path to the predictions

    Returns
    -------
    rpss: pandas dataframe, contains yearly rpss values (global average rpss)
    """

    dfs_rpss = []

    # train
    for data in ['train', 'test']:

        if data == 'train':
            ds_pred = xr.open_dataset(f'{path_pred}{folder}/global_pred_{name}_allyears_raw.nc')
        else:
            ds_pred = xr.open_dataset(f'{path_pred}{folder}/global_pred_{name}_2020_raw.nc')

        if agg_domain == 'midlats':
            ds_pred = ds_pred.sel(latitude=slice(60, 40))
        if agg_domain == 'europe':
            ds_pred = ds_pred.where((ds_pred['longitude'] <= 60) | (ds_pred['longitude'] > 301), drop=True)
            ds_pred = ds_pred.sel(latitude=slice(79, 19))

        df = skill_by_year_single(ds_pred, adapt=True, cache_path=cache_path)
        df['model_name'] = name
        df['data'] = data
        df['aggregation_domain'] = agg_domain

        dfs_rpss.append(df)
        print(f'finished {data}')

    rpss = pd.concat(dfs_rpss, axis=0)
    rpss['model_name'] = name

    return rpss


def skill_by_year_single(preds, cache_path='../template/data', adapt=False, spatial=False, clim_baseline=False):
    """
    based on skill_by_year method in utils.scripts,
    returns pd.Dataframe with yearly rpss.

    Parameters
    ----------
    preds: DataSet, predictions from ML models with at least one variable and one lead time
    cache_path: string, path to observations / ground truth
    adapt: bool, restrict evaluation to coordinates for which predictions are provided
    spatial: bool, whether to compute spatial rpss fields, if False compute spatially averaged rpss
    clim_baseline: bool or DataSet, if True, use ecmwf baseline as reference (only for 2020), if False use climatology
                   as reference, if DataSet, this data set is used as reference

    Returns
    -------
    if spatial False, pandas dataframe with yearly global averages of rpss per lead time and variable
    if spatial True, xarray DataSet with yearly averages of rpss per lead time and variable per grid cell
    """

    xr.set_options(keep_attrs=True)

    # obs
    if 2020 in preds.forecast_time.dt.year:
        obs_p = xr.open_dataset(f'{cache_path}/forecast-like-observations_2020_biweekly_terciled.nc'
                                ).sel(forecast_time=preds.forecast_time.values).compute()
    else:
        obs_p = xr.open_dataset(f'{cache_path}/hindcast-like-observations_2000-2019_biweekly_terciled.zarr',
                                engine='zarr')
        obs_p = obs_p.sel(forecast_time=preds.forecast_time.values).compute()

    # climatology
    if (2020 in preds.forecast_time.dt.year) and (clim_baseline == True):
        clim_p = xr.open_dataset(f'{cache_path}/ecmwf_recalibrated_benchmark_2020_biweekly_terciled.nc')
    elif (clim_baseline != False) and (clim_baseline != True):
        clim_p = clim_baseline
    else:  # clim_baseline == False:
        clim_p = xr.ones_like(obs_p) * 1 / 3

    # ML probabilities
    fct_p = preds

    if adapt:
        # select only obs_p where fct_p forecasts provided
        for c in ['longitude', 'latitude', 'forecast_time', 'lead_time']:
            obs_p = obs_p.sel({c: fct_p[c].values})
            clim_p = clim_p.sel({c: fct_p[c].values})
        obs_p = obs_p[list(fct_p.data_vars)]
        clim_p = clim_p[list(fct_p.data_vars)]

    print('subset selection done')

    # rps_ML
    rps_ML = xs.rps(obs_p, fct_p, category_edges=None, dim=[], input_distributions='p').compute()
    # rps_clim
    rps_clim = xs.rps(obs_p, clim_p, category_edges=None, dim=[], input_distributions='p').compute()

    print('rps done')

    # RPSS
    # penalize # https://renkulab.io/gitlab/aaron.spring/s2s-ai-challenge-template/-/issues/7
    expect = obs_p.sum('category')
    expect = expect.where(expect > 0.98).where(expect < 1.02)  # should be True if not all NaN

    # https://renkulab.io/gitlab/aaron.spring/s2s-ai-challenge-template/-/issues/50
    rps_ML = rps_ML.where(expect, other=2)  # assign RPS=2 where value was expected but NaN found

    # following Weigel 2007: https://doi.org/10.1175/MWR3280.1
    rpss = 1 - (rps_ML.groupby('forecast_time.year').mean() / rps_clim.groupby('forecast_time.year').mean())
    # clip
    rpss = rpss.clip(-10, 1)

    if spatial == False:
        # weighted area mean
        weights = np.cos(np.deg2rad(np.abs(rpss.latitude)))
        # spatially weighted score averaged over lead_times and variables to one single value
        scores = rpss.sel(latitude=slice(None, -60)).weighted(weights).mean('latitude').mean('longitude')
        return scores.to_array().to_dataframe('RPSS')
    else:
        return rpss


def get_spatial_rpss(name, cache_path='../template/data', folder='', path_pred='../submissions/'):
    """
    compute and save rpss on grid-point level

    Parameters
    ----------
    name: string, name of model (of format {v}_{lead time}_{predictor vars}_{date}_{time}
    cache_path: string, path to observations / ground truth
    folder: string, folder where model and predictions are saved
    path_pred: string, path to the predictions

    Returns
    -------
    Nothing, since rpss is saved to file directly
    """

    ls_data = []
    for data in ['train', 'test']:

        if data == 'train':
            ds_pred = xr.open_dataset(f'{path_pred}{folder}/global_pred_{name}_allyears_raw.nc')
        else:
            ds_pred = xr.open_dataset(f'{path_pred}{folder}/global_pred_{name}_2020_raw.nc')

        ls_data.append(skill_by_year_single(ds_pred, adapt=True, cache_path=cache_path, spatial=True))

    ds_rpss = xr.concat(ls_data, 'year')
    ds_rpss.to_netcdf(f'{path_pred}{folder}/rpss_{name}_raw.nc')
    return
