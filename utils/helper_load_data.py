import numpy as np
import xarray as xr
xr.set_options(display_style='text')

from utils.paths import get_paths


def get_data(varlist, path_data, drop_negative_tp='zero'):
    """
    wraps load_data to load hindcasts and observations for a given set of variables.
    also creates a mask for the missing values (grid-cells with at least one missing
    value in 2000-2019 are set to nan.)

    Parameters
    ----------
    var_list: list, list of variables
    path_data: string or 2-element list, if string, it has to be one of {'local', 'server'},
               if 2-element list, it is expected to be the user-defined path to the data
    drop_negative_tp: either True or 'zero', if 'zero' negative precipitation values are set to zero,
                      if True, negative precipitation values are replaced by NA.

    Returns
    -------
    hind: DataSet, contains hindcasts for the specified hindcasts
    obs: DataSet, observations for the hindcast period
    obs_terciled: DataSet, one-hot representation of observations split in terciles, for hindcast period
    mask: DataSet, mask for the missing values (grid-cells with at least one missing value in 2000-2019
                   are set to NA)
    """
    # hindcasts
    hind = get_data_single(data='hind_2000-2019', aggregation='biweekly',
                           path_data=path_data, var_list=varlist, drop_negative_tp=drop_negative_tp)

    # observations corresponding to hindcasts
    obs = load_data(data='obs_2000-2019', aggregation='biweekly', path=path_data)
    # terciled
    obs_terciled = load_data(data='obs_terciled_2000-2019', aggregation='biweekly',
                             path=path_data)

    # mask: same missing values at all forecast_times, only used for label data
    # notnull=True --> set to 1
    mask = xr.where(obs.notnull(), 1, np.nan).mean('forecast_time', skipna=False)

    return hind, obs, obs_terciled, mask


def get_data_single(data, path_data, var_list=['tp', 't2m'],
                    drop_negative_tp='zero', aggregation='biweekly'):
    """
    wraps load_data to load single data files and drops superfluous coordinates and deals with negative
    precipitation amounts.

    Parameters
    ----------
    data: string,  one of {'hind_2000-2019', 'obs_2000-2019', 'obs_terciled_2000-2019',
          'obs_tercile_edges_2000-2019', 'forecast_2020', 'obs_2020', 'obs_terciled_2020'}
    path_data: string or 2-element list, if string, it has to be one of {'local', 'server'},
               if 2-element list, it is expected to be the user-defined path to the data
    var_list: list, list of variables
    drop_negative_tp: either True or 'zero', if 'zero' negative precipitation values are set to zero,
                      if True, negative precipitation values are replaced by NA.
    aggregation: string, one of {'biweekly','weekly'}

    Returns
    -------
    dat: DataSet, with some processing already done
    """
    dat = load_data(data=data, aggregation=aggregation,
                    path=path_data, var_list=var_list)

    # hind and forecasts
    if 'realization' in dat.coords:
        dat = dat[var_list]
        dat = clean_coords(dat)
        dat = clean_data(dat, drop_negative_tp=drop_negative_tp)

    return dat


def load_data(data='hind_2000-2019', aggregation='biweekly', path='server',
              var_list=['tp', 't2m']):
    """
    loads .nc and .zarr files

    Parameters
    ----------
    data: string, one of {'hind_2000-2019', 'obs_2000-2019', 'obs_terciled_2000-2019',
          'obs_tercile_edges_2000-2019', 'forecast_2020', 'obs_2020', 'obs_terciled_2020'}
    aggregation: string, one of {'biweekly','weekly'}
    path: string or 2-element list, if string, it has to be one of {'local', 'server'},
          if 2-element list, it is expected to be the user-defined path to the data
    var_list: list, list of variables

    Returns
    -------
    dat: DataSet, unmodified

    """

    cache_path, path_add_vars, path_model, path_pred, path_results = get_paths(path)

    if data == 'hind_2000-2019':
        dat_list = []
        for var in var_list:
            if (var == 'tp') or (var == 't2m'):
                if aggregation == 'biweekly':
                    dat_item = xr.open_zarr(
                        '{}/ecmwf_hindcast-input_2000-2019_{}_deterministic.zarr'.format(cache_path, aggregation),
                        consolidated=True)
                else:
                    dat_item = xr.open_zarr(
                        '{}/ecmwf_hindcast-input_2000-2019_{}_deterministic.zarr'.format(path_add_vars, aggregation),
                        consolidated=True)
                var_list = [i for i in var_list if i not in ['tp', 't2m']]
            else:
                dat_item = xr.open_zarr(
                    '{}/ecmwf_hindcast-input_2000-2019_{}_deterministic_{}.zarr'.format(path_add_vars, aggregation,
                                                                                        var), consolidated=True)
                if (var == 'gh500') or (var == 'gh850'):
                    dat_item = dat_item.reset_coords('plev', drop=True)

            dat_list.append(dat_item)
        dat = xr.merge(dat_list)

    elif data == 'obs_2000-2019':
        dat = xr.open_zarr(f'{cache_path}/hindcast-like-observations_2000-2019_biweekly_deterministic.zarr',
                           consolidated=True)

    elif data == 'obs_terciled_2000-2019':
        dat = xr.open_zarr(f'{cache_path}/hindcast-like-observations_2000-2019_biweekly_terciled.zarr',
                           consolidated=True)

    elif data == 'obs_tercile_edges_2000-2019':
        dat = xr.open_dataset(f'{cache_path}/hindcast-like-observations_2000-2019_biweekly_tercile-edges.nc')

    elif data == 'forecast_2020':
        dat_list = []
        for var in var_list:
            if (var == 'tp') or (var == 't2m'):
                if aggregation == 'biweekly':
                    dat_item = xr.open_zarr(
                        '{}/ecmwf_forecast-input_2020_{}_deterministic.zarr'.format(cache_path, aggregation),
                        consolidated=True)
                else:
                    dat_item = xr.open_zarr(
                        '{}/ecmwf_forecast-input_2020_{}_deterministic.zarr'.format(path_add_vars, aggregation),
                        consolidated=True)
                var_list = [i for i in var_list if i not in ['tp', 't2m']]  # needed since tp and t2m are loaded jointly
            else:
                dat_item = xr.open_zarr(
                    '{}/ecmwf_forecast-input_2020_{}_deterministic_{}.zarr'.format(path_add_vars, aggregation, var),
                    consolidated=True)
                if (var == 'gh500') or (var == 'gh850'):
                    dat_item = dat_item.reset_coords('plev', drop=True)
            dat_list.append(dat_item)
        dat = xr.merge(dat_list)

    elif data == 'obs_2020':
        dat = xr.open_zarr(f'{cache_path}/forecast-like-observations_2020_biweekly_deterministic.zarr',
                           consolidated=True)

    elif data == 'obs_terciled_2020':
        dat = xr.open_dataset(f'{cache_path}/forecast-like-observations_2020_biweekly_terciled.nc')

    elif data == 'ecmwf_baseline':
        dat = xr.open_dataset(f'{cache_path}/ecmwf_recalibrated_benchmark_2020_biweekly_terciled.nc')
    else:
        print("specified data name is not valid")

    return dat


def clean_coords(dat):
    """
    remove superfluous coordinates

    Parameters
    -------
    dat: DataSet
    """

    if 'sm20' in dat.keys():
        dat = dat.isel(depth_below_and_layer=0).reset_coords('depth_below_and_layer', drop=True)
    if 'msl' in dat.keys():
        dat = dat.isel(meanSea=0).reset_coords('meanSea', drop=True)
    if 'tcw' in dat.keys():
        dat = dat.isel(entireAtmosphere=0).reset_coords('entireAtmosphere', drop=True)
    return dat


def clean_data(dat, drop_negative_tp='zero'):
    """

    Parameters
    ----------
    dat: DataSet
    drop_negative_tp: either True or 'zero', if 'zero' negative precipitation values are set to zero,
                      if True, negative precipitation values are replaced by NA.

    """
    print('drop negative tp values: ', drop_negative_tp)
    # set negative tp values to nan
    if ('tp' in dat.keys()) and (drop_negative_tp == True):
        dat['tp'] = xr.where(dat.tp < 0, np.nan, dat.tp)
    # set to zero
    elif ('tp' in dat.keys()) and (drop_negative_tp == 'zero'):
        dat['tp'] = xr.where(dat.tp < 0, 0, dat.tp)
    return dat


def get_basis(out_field, r_basis):
    """ returns a set of basis functions for the input field, adapted from Scheuerer et al. (2020)

    Parameters
    ----------
    out_field: DataArray, basis functions for these lat lon coordinates will be created
    r_basis: int, radius of support of basis functions, the distance between centers of
             basis functions is half this radius, should be chosen depending on input field size.

    Returns
    -------
    basis: values for basis functions over out_field
    lats: lats of input field
    lons: lons of input field
    n_xy: number of grid points in input field
    n_basis: number of basis functions
    """

    # distance between centers of basis functions
    dist_basis = r_basis / 2
    lats = out_field.latitude
    lons = out_field.longitude

    # number of basis functions
    n_basis = int(np.ceil((lats[0] - lats[-1]) / dist_basis + 1) * np.ceil((lons[-1] - lons[0]) / dist_basis + 1))

    # grid coords
    lon_np = lons
    lat_np = lats

    length_lon = len(lon_np)
    length_lat = len(lat_np)

    lon_np = np.outer(lon_np, np.ones(length_lat)).reshape(int(length_lon * length_lat))
    lat_np = np.outer(lat_np, np.ones(length_lon)).reshape(int(length_lon * length_lat))

    # number of grid points
    n_xy = int(length_lon * length_lat)

    # centers of basis functions
    lon_ctr = np.arange(lons[0], lons[-1] + dist_basis, dist_basis)
    length_lon_ctr = len(lon_ctr)  # number of center points in lon direction

    lat_ctr = np.arange(lats[0], lats[-1] - dist_basis, - dist_basis)
    length_lat_ctr = len(lat_ctr)  # number of center points in lat direction

    lon_ctr = np.outer(lon_ctr, np.ones(length_lat_ctr)).reshape(int(n_basis))
    lat_ctr = np.outer(np.ones(length_lon_ctr), lat_ctr).reshape(int(n_basis))

    # compute distances between fct grid and basis function centers
    dst_lon = np.abs(np.subtract.outer(lon_np, lon_ctr).reshape(len(lons), len(lats), n_basis))  # 10,14
    dst_lon = np.swapaxes(dst_lon, 0, 1)
    dst_lat = np.abs(np.subtract.outer(lat_np, lat_ctr).reshape(len(lats), len(lons), n_basis))  # 14,10

    dst = np.sqrt(dst_lon ** 2 + dst_lat ** 2)
    dst = np.swapaxes(dst, 0, 1).reshape(n_xy, n_basis)

    # define basis functions
    basis = np.where(dst > r_basis, 0., (1. - (dst / r_basis) ** 3) ** 3)  # main step, zero outside,
    basis = basis / np.sum(basis, axis=1)[:, None]  # normalization at each grid point
    # nbs = basis.shape[1]

    return basis, lats, lons, n_xy, n_basis
