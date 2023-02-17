import numpy as np
import xarray as xr
xr.set_options(display_style='text')

import tensorflow as tf
from tqdm import tqdm

from utils.helper_datagenerator import pad_earth, flip_antimeridian
from utils.helper_load_data import get_basis


def slide_predict(fct, input_dims, output_dims, cnn, model_architecture, basis_rad, n_bins):
    """"slide the model that was trained on one specific patch over the earth
    to create predictions everywhere

    Parameters
    ----------
    model_architecture
    fct: DataSet, preprocessed forecasts without padding
    input_dims: int
    output_dims: int

    Returns
    -------
    prediction: numpy array, global prediction
    """

    # pad the earth by the required amount
    fct_padded = pad_earth(fct, [input_dims, output_dims]).compute()

    # initialize 1/3 matrix to save predictions
    prediction = np.ones(shape=(len(fct_padded.forecast_time),
                                len(fct_padded.longitude),
                                len(fct_padded.latitude),  # int(360/1.5), int(180/1.5) + 10
                                3)) * 1 / 3

    # iterate over global and create predictions patch-wise
    for lat_i in tqdm(range(0, int(180 / 1.5) + output_dims - 1, output_dims), desc='slide over globe'):
        for lon_i in range(0, int(360 / 1.5) + output_dims - 1, output_dims):
            # select patch
            patch = fct_padded.isel(longitude=slice(lon_i, lon_i + input_dims),
                                    latitude=slice(lat_i, lat_i + input_dims))

            # prepare input to cnn.predict
            if model_architecture == 'basis_func':
                # compute basis
                basis, lats, lons, n_xy, n_basis = get_basis(fct.isel(latitude=slice(0, output_dims),
                                                                      longitude=slice(0, output_dims)), basis_rad)
                # climatology
                clim_probs = np.log(1 / 3) * np.ones((n_xy, n_bins))

                input_list = [
                    patch.fillna(0.).to_array().transpose('forecast_time', ..., 'longitude', 'latitude', 'variable').values,
                    # field info
                    np.repeat(basis[np.newaxis, :, :], len(patch.forecast_time), axis=0),  # basis
                    np.repeat(clim_probs[np.newaxis, :, :], len(patch.forecast_time), axis=0)]  # clim
            else:
                input_list = [
                    patch.fillna(0.).to_array().transpose('forecast_time', ..., 'longitude', 'latitude',
                                                          'variable').values]

            preds = cnn.predict(input_list).squeeze()

            prediction[:, lon_i:(lon_i + output_dims), lat_i:(lat_i + output_dims), :] = preds

    prediction = prediction[:, output_dims:(int(360 / 1.5) + output_dims),  # depends on padding
                            output_dims:(int(180 / 1.5) + 1 + output_dims), :]

    return add_coords(prediction, fct.forecast_time, fct.latitude, fct.longitude, fct.lead_time)


def global_predict(fct, cnn, region='global'):
    """"
    Parameters:
    ----------
    fct: DataSet, preprocessed forecasts without padding

    Returns:
    -------
    prediction: numpy array, global prediction
    """

    if region == 'europe':
        fct = flip_antimeridian(fct, to='Pacific', lonn='longitude')

        # European domain that is divisible by 8 in lat lon direction
        fct_padded = fct.sel(latitude=slice(90, 7), longitude=slice(-71, 73))
        fct = fct.sel(latitude=slice(79, 19), longitude=slice(-59, 60))
    else:  # global
        # pad the earth by the required amount
        fct_padded = pad_earth(fct, 8).compute()
        fct_padded = fct_padded.isel(latitude=slice(4, -5))

    input_list = [fct_padded.fillna(0.).to_array().transpose('forecast_time',
                                                             ..., 'longitude', 'latitude',
                                                             'variable').values]

    prediction = cnn.predict(input_list).squeeze()
    prediction_ = add_coords(prediction, fct.forecast_time, fct.latitude, fct.longitude, fct.lead_time)

    if region == 'europe':  # convert back to europe/not pacific. o.w. can't run the standard evaluation methods
        prediction_ = flip_antimeridian(prediction_.to_dataset(name='v'), to='Europe', lonn='longitude'
                                        )
        prediction_ = prediction_.v.compute()

    return prediction_


def add_coords(pred, fcst_time, global_lats, global_lons, lead_output_coords):
    """ add proper coordinates to numpy array predictions"""
    da = xr.DataArray(
        tf.transpose(pred, [0, 3, 1, 2]),
        dims=['forecast_time', 'category', 'longitude', 'latitude'],
        coords={'forecast_time': fcst_time, 'category': ['below normal', 'near normal', 'above normal'],
                'latitude': global_lats,
                'longitude': global_lons
                }
    )
    da = da.transpose('category', 'forecast_time', 'latitude', ...)
    da = da.assign_coords(lead_time=lead_output_coords)
    return da
