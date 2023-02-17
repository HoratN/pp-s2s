import numpy as np
import pandas as pd
import xarray as xr
xr.set_options(display_style='text')

import tensorflow.keras as keras


class DataGeneratorGlobal(keras.utils.Sequence):
    """
    data generator used for global training
    input_lats=128, input_lons=256,
    """

    def __init__(self, fct, verif, region='global',
                 batch_size=32, shuffle=True, load=False):
        """
        Parameters
        ----------
        fct: DataSet, global preprocessed fields, coordinates: latitude, longitude, forecast_time
        verif: DataSet, global preprocessed fields, coordinates: latitude, longitude, forecast_time
        region: string, domain the model should be trained on, either global or europe
        batch_size: int
        shuffle: bool, if True, data is shuffled
        load: bool, if True, data is loaded into RAM.

        Returns
        -------
        data generator object
        """

        self.batch_size = batch_size
        self.shuffle = shuffle

        if region == 'europe':
            fct = flip_antimeridian(fct, to='Pacific', lonn='longitude')
            verif = flip_antimeridian(verif, to='Pacific', lonn='longitude')

            # European domain that is divisible by 8 in lat lon direction
            fct = fct.sel(latitude=slice(90, 7), longitude=slice(-71, 73))
            verif = verif.sel(latitude=slice(79, 19), longitude=slice(-59, 60))
        else:  # global
            fct = pad_earth(fct, pad_args=8)
            fct = fct.isel(latitude=slice(4, -5))

        # create self. ... data
        self.fct_data = fct.transpose('forecast_time', ...)
        self.verif_data = verif.transpose('forecast_time', ...)

        self.n_samples = self.fct_data.forecast_time.shape[0]

        self.on_epoch_end()

        if load: self.fct_data.load()

    def __len__(self):
        """Denotes the number of batches per epoch"""
        return int(np.ceil(self.n_samples / self.batch_size))

    def __getitem__(self, i):
        """Generate one batch of data"""

        idxs = self.idxs[i * self.batch_size:(i + 1) * self.batch_size]

        X_x_list = []
        y_list = []
        for j in idxs:
            X_x_list.append(np.expand_dims(self.fct_data.isel(forecast_time=j
                                                            ).fillna(0.).to_array(
                            ).transpose(..., 'longitude', 'latitude', 'variable'
                                        ).values, axis=0))

            y_list.append(np.expand_dims(self.verif_data.isel(forecast_time=j
                                                              ).fillna(0.
                                         ).transpose(..., 'longitude', 'latitude', 'category'
                                                     ).values, axis=0))

        X_x = np.concatenate(X_x_list)
        X = [X_x]
        y = np.concatenate(y_list)

        return X, y

    def on_epoch_end(self):
        """Updates indexes after each epoch"""

        self.idxs = np.arange(self.n_samples)

        if self.shuffle == True:
            np.random.shuffle(self.idxs)  # in place
            print('reshuffled idxs')


class DataGeneratorMultipatch(keras.utils.Sequence):
    """ data generator used for patch-wise training"""

    def __init__(self, fct, verif, model, input_dims, output_dims, mask_v, region=None, batch_size=32, shuffle=True,
                 load=False, reduce_sample_size=None, patch_stride=2, used_members=11, fraction=0 / 8, weighted=False):
        """
        data generator for using data from all valid patches of the selected region

        Parameters
        ----------
        model
        fct: DataSet, global preprocessed fields, coordinates: latitude, longitude, forecast_time
        verif: DataSet, global preprocessed fields, coordinates: latitude, longitude, forecast_time
        input_dims: int, size of input domain
        output_dims: int, size of output domain
        mask_v: DataArray, no lead time coordinates, target variable already selected
        region: string, if None, all valid  patches are used, ow only valid patches from the
                respective region are used (poles, midlats, subtropics, tropics)
        batch_size: int
        shuffle: bool, if True, data is shuffled
        load: bool, if True, data is loaded into RAM.
        reduce_sample_size: int or None, if not None, provide integer that indicates
                            the fraction of used patches from the list of valid patches
        used_members: int, only used if fct has realization coordinates,
                      specifies how many ensemble members should be used
        fraction: float, fraction of grid cell that can be NA for a valid patch
        weighted: bool, if True weights for each grid cell are returned for every sample

        Returns
        -------
        data generator object
        """

        self.batch_size = batch_size
        self.shuffle = shuffle
        self.model = model
        if model.model_architecture == 'basis_func':
            self.basis = model.basis
            self.clim_probs = model.clim_probs

        self.input_dims = input_dims
        self.output_dims = output_dims

        self.weighted = weighted

        if 'realization' in fct:
            # select members that should be used
            # do this as early as possible, since padding is very memory-inefficient
            if used_members > 11:
                print('invalid value in used_members, set to 11')
                use_members = 11
            fct = fct.isel(realization=slice(0, used_members))

        # pad
        pad_args = [input_dims, output_dims]
        fct = pad_earth(fct, pad_args)
        verif = pad_earth(verif, pad_args)
        # mask gets padded in get_all_valid_patches

        # create self. ... data
        self.fct_data = fct.transpose('forecast_time', ...)
        self.verif_data = verif.transpose('forecast_time', ...)

        # create reference df with coordinates of all valid patches
        # patch is valid if it contains less NA than indicated by 'fraction'
        valid_coords_raw = get_all_valid_patches(mask_v, output_dims, pad_args=pad_args, fraction=fraction,
                                                 patch_stride=patch_stride, region=region)

        if reduce_sample_size != None:
            valid_coords_raw = valid_coords_raw.iloc[
                               0:int(valid_coords_raw.shape[0] / reduce_sample_size)]  # e.g. 10 or 5

        # expand valid_coords_raw by forecast_time dimension
        n_samples_time = self.fct_data.forecast_time.size
        self.valid_coords = pd.concat([valid_coords_raw] * n_samples_time, ignore_index=True)  # 2'976'480 rows
        new_col = np.repeat(np.arange(n_samples_time), valid_coords_raw.shape[0])
        self.valid_coords['time'] = new_col

        if 'realization' in self.fct_data:
            valid_coords = self.valid_coords.copy()
            n_samples_realization = self.fct_data.realization.size
            self.valid_coords = pd.concat([valid_coords] * n_samples_realization, ignore_index=True)
            new_col = np.repeat(np.arange(n_samples_realization), valid_coords.shape[0])
            self.valid_coords['realization'] = new_col

        self.n_samples = self.valid_coords.shape[0]

        self.on_epoch_end()

        if self.weighted == True:
            weights = np.cos(np.deg2rad(np.abs(mask_v.latitude)))
            self.mask_weighted = (mask_v.transpose('longitude', 'latitude').fillna(0) * weights)
            self.mask_weighted = pad_earth(self.mask_weighted, pad_args)
            self.mask_weighted.load()

        if load: self.fct_data.load()

    def __len__(self):
        """Denotes the number of batches per epoch"""
        return int(np.ceil(self.n_samples / self.batch_size))

    def __getitem__(self, i):
        """Generate one batch of data"""
        idxs = self.idxs[i * self.batch_size:(i + 1) * self.batch_size]

        X_x_list = []
        y_list = []
        weight_list = []

        for j in idxs:

            time = self.valid_coords['time'].iloc[j]
            lat = self.valid_coords['latitude'].iloc[j]
            lon = self.valid_coords['longitude'].iloc[j]

            if 'realization' in self.fct_data:
                realization = self.valid_coords['realization'].iloc[j]
                X_x_list.append(np.expand_dims(self.fct_data.isel(forecast_time=time,
                                                                  latitude=self.get_patch_slices(lat, lon, 'input_lats',
                                                                                                 self.input_dims,
                                                                                                 self.output_dims),
                                                                  longitude=self.get_patch_slices(lat, lon,
                                                                                                  'input_lons',
                                                                                                  self.input_dims,
                                                                                                  self.output_dims),
                                                                  realization=realization
                                                                  ).fillna(0.).to_array().transpose(..., 'longitude',
                                                                                                    'latitude',
                                                                                                    'variable').values,
                                               axis=0))
            else:
                X_x_list.append(np.expand_dims(self.fct_data.isel(forecast_time=time,
                                                                  latitude=self.get_patch_slices(lat, lon, 'input_lats',
                                                                                                 self.input_dims,
                                                                                                 self.output_dims),
                                                                  longitude=self.get_patch_slices(lat, lon,
                                                                                                  'input_lons',
                                                                                                  self.input_dims,
                                                                                                  self.output_dims)
                                                                  ).fillna(0.).to_array().transpose(..., 'longitude',
                                                                                                    'latitude',
                                                                                                    'variable').values,
                                               axis=0))

            if self.weighted:
                weight_list.append(np.expand_dims(self.mask_weighted.isel(
                    latitude=self.get_patch_slices(lat, lon, 'output_lats', self.input_dims, self.output_dims),
                    longitude=self.get_patch_slices(lat, lon, 'output_lons', self.input_dims, self.output_dims)
                    ).transpose('longitude', 'latitude').values, axis=0))

            y_list.append(np.expand_dims(self.verif_data.isel(forecast_time=time,
                                                              latitude=self.get_patch_slices(lat, lon, 'output_lats',
                                                                                             self.input_dims,
                                                                                             self.output_dims),
                                                              longitude=self.get_patch_slices(lat, lon, 'output_lons',
                                                                                              self.input_dims,
                                                                                              self.output_dims)
                                                              ).fillna(0.).transpose(..., 'longitude', 'latitude',
                                                                                     'category').values,
                                         axis=0))

        X_x = np.concatenate(X_x_list)
        if self.model.model_architecture == 'basis_func':
            X_basis = np.repeat(self.basis[np.newaxis, :, :], len(idxs), axis=0)
            X_clim = np.repeat(self.clim_probs[np.newaxis, :, :], len(idxs), axis=0)
            X = [X_x, X_basis, X_clim]
        else:
            X = [X_x]
        y = np.concatenate(y_list)

        if self.weighted:
            weights = np.concatenate(weight_list)
            X = X + [weights, y]

        return X, y

    def on_epoch_end(self):
        """Updates indexes after each epoch"""

        self.idxs = np.arange(self.n_samples)

        if self.shuffle == True:
            np.random.shuffle(self.idxs)  # in place
            print('reshuffled idxs')

    def get_patch_slices(self, lat, lon, kind, input_dims, output_dims):
        """
        yield slices for latitude and longitude coordinates based on the coordinates
        of a patch corner
        """
        if kind == 'input_lats':
            input_lats = slice(lat - output_dims - int((input_dims - output_dims) / 2) + 1,
                               lat + int((input_dims - output_dims) / 2) + 1)
            return input_lats
        elif kind == 'input_lons':
            input_lons = slice(lon - output_dims - int((input_dims - output_dims) / 2) + 1,
                               lon + int((input_dims - output_dims) / 2) + 1)
            return input_lons
        elif kind == 'output_lats':
            output_lats = slice(lat - output_dims + 1, lat + 1)
            return output_lats
        elif kind == 'output_lons':
            output_lons = slice(lon - output_dims + 1, lon + 1)
            return output_lons
        else:
            print('no valid "kind" argument provided')


def flip_antimeridian(ds, to='Pacific', lonn='lon'):
    """
    # https://git.iac.ethz.ch/utility_functions/utility_functions_python/-/blob/regionmask/xarray.py

    Flip the antimeridian (i.e. longitude discontinuity) between Europe
    (i.e., [0, 360)) and the Pacific (i.e., [-180, 180)).
    Parameters:
    - ds (xarray.Dataset or .DataArray): Has to contain a single longitude
      dimension.
    - to='Pacific' (str, optional): Flip antimeridian to one of
      * 'Europe': Longitude will be in [0, 360)
      * 'Pacific': Longitude will be in [-180, 180)
    - lonn=None (str, optional): Name of the longitude dimension. If None it
      will be inferred by the CF convention standard longitude unit.
    Returns:
    same type as input ds
    """

    attrs = ds[lonn].attrs

    if to.lower() == 'europe' and not antimeridian_pacific(ds):
        return ds  # already correct, do nothing
    elif to.lower() == 'pacific' and antimeridian_pacific(ds):
        return ds  # already correct, do nothing
    elif to.lower() == 'europe':
        ds = ds.assign_coords(**{lonn: (ds[lonn] % 360)})
    elif to.lower() == 'pacific':
        ds = ds.assign_coords(**{lonn: (((ds[lonn] + 180) % 360) - 180)})
    else:
        errmsg = 'to has to be one of [Europe | Pacific] not {}'.format(to)
        raise ValueError(errmsg)

    was_da = isinstance(ds, xr.core.dataarray.DataArray)
    if was_da:
        da_varn = ds.name
        if da_varn is None:
            da_varn = 'data'
        enc = ds.encoding
        ds = ds.to_dataset(name=da_varn)

    idx = np.argmin(ds[lonn].data)
    varns = [varn for varn in ds.variables if lonn in ds[varn].dims]
    for varn in varns:
        if xr.__version__ > '0.10.8':
            ds[varn] = ds[varn].roll(**{lonn: -idx}, roll_coords=False)
        else:
            ds[varn] = ds[varn].roll(**{lonn: -idx})

    ds[lonn].attrs = attrs
    if was_da:
        da = ds[da_varn]
        da.encoding = enc
        return da
    return ds


def antimeridian_pacific(ds, lonn=None):
    """Returns True if the antimeridian is in the Pacific (i.e. longitude runs
    from -180 to 180."""
    if lonn is None:
        lonn = get_longitude_name(ds)
    if ds[lonn].min() < 0 or ds[lonn].max() < 180:
        return True
    return False


def get_longitude_name(ds):
    """Get the name of the longitude dimension by CF unit"""
    lonn = []
    if isinstance(ds, xr.core.dataarray.DataArray):
        ds = ds.to_dataset()
    for dimn in ds.dims.keys():
        if 'units' in ds[dimn].attrs and ds[dimn].attrs['units'] in ['degree_east', 'degrees_east']:
            lonn.append(dimn)
    if len(lonn) == 1:
        return lonn[0]
    elif len(lonn) > 1:
        errmsg = 'More than one longitude coordinate found by unit.'
    else:
        errmsg = 'Longitude could not be identified by unit.'
    raise ValueError(errmsg)


def pad_earth(fct, pad_args):
    """
    pad global input field on all sides
    data resolution is 1.5°

    Parameters:
    ----------
    fct: DataSet
    pad_args: int or list, if int, it is the pad, if list, should be [input_dims, output_dims],
              will be used to compute the pad
    """

    if type(pad_args) == list:
        input_dims = pad_args[0]
        output_dims = pad_args[1]
        pad = int((input_dims - output_dims) / 2) + output_dims
    else:
        pad = pad_args

    # create padding for north pole and south pole
    fct_shift = fct.pad(pad_width={'longitude': (0, 120)}, mode='wrap'
                        ).shift({'longitude': 120}
                                ).isel(longitude=slice(120, 120 + int(360 / 1.5)))  # 240/2 = 120

    fct_shift_pad = fct_shift.pad(pad_width={'latitude': (pad, pad)}, mode='reflect')
    shift_pad_south = fct_shift_pad.isel(latitude=slice(len(fct_shift_pad.latitude) - pad,
                                                        len(fct_shift_pad.latitude)))

    shift_pad_north = fct_shift_pad.isel(latitude=slice(0, pad))

    # add pole padding to ftc_train
    fct_lat_pad = xr.concat([shift_pad_north, fct, shift_pad_south], dim='latitude')

    # pad in east-west direction
    fct_padded = fct_lat_pad.pad(pad_width={'longitude': (pad, pad)}, mode='wrap')
    return fct_padded


def get_all_valid_patches(mask, output_dims, pad_args, fraction=1 / 8, patch_stride=2, region=None):
    """
    Parameters
    ----------
    mask: DataArray no lead time coords, targed variable already selected
    output_dims: int, size of output domain
    fraction: float, fraction of grid cell that can be NA for a valid patch.
    pad_args: int, with how many grid cells global field should be padded
    region: string, if None, all valid  patches are used, ow only valid
                    patches from the respective region are used (poles, midlats, subtropics, tropics)
    Returns
    -------
    possible_coords: dataframe with isel coords of valid patches for padded dataarray
                     coords corresponds to bottom right corner of the valid patch.
    """

    if type(pad_args) == list:
        input_dims = pad_args[0]
        output_dims = pad_args[1]
        pad = int((input_dims - output_dims) / 2) + output_dims
    else:
        pad = pad_args

    print('patch stride', patch_stride)
    mask_pad = pad_earth(mask, pad_args)  # make sure that all pixel in the final domain are completely rolled

    mask_pad_new_coords = mask_pad.assign_coords(longitude=np.arange(0, len(mask_pad.longitude)),
                                                 latitude=np.arange(0, len(mask_pad.latitude)))
    rolling = mask_pad_new_coords.rolling(latitude=output_dims, longitude=output_dims, min_periods=1)
    use_patch_out = rolling.construct(latitude='latitude_win',
                                      longitude='longitude_win',
                                      stride=patch_stride).sum(('latitude_win',
                                                                'longitude_win')).compute()
    # the first element of the rolling is the sum of the first element in mask_pad_new_coords and win-1 na values.
    # so, if 16 elements are padded, then 16 rolling elements are created for this region.

    keep_final_new = xr.where(use_patch_out > output_dims ** 2 * (1 - fraction) - 1, 1, 0
                              ).compute()  # -1 since now 1/8 missing is fine

    coords = keep_final_new.to_dataframe().reset_index()
    possible_coords = coords.loc[coords[coords.columns[3]] == 1]

    # ensure that output is only in domain + output_dims -1
    possible_coords = possible_coords.loc[(possible_coords['latitude'] > pad - 1) &
                                          (possible_coords['latitude'] < pad + 121)]
    possible_coords = possible_coords.loc[(possible_coords['longitude'] > pad - 1) &
                                          (possible_coords['longitude'] < pad + 240)]

    if region == 'poles':  # >60
        possible_coords = possible_coords.loc[
            (possible_coords.latitude < pad + 20) | (possible_coords.latitude > pad + 100)]
    elif region == 'midlats':  # >40 <=60
        possible_coords = possible_coords.loc[
            ((possible_coords.latitude >= pad + 20) & (possible_coords.latitude < pad + 34)) |
            ((possible_coords.latitude <= pad + 100) & (possible_coords.latitude > pad + 86))]
    elif region == 'subtropics':  # >23.5 <=40
        possible_coords = possible_coords.loc[
            ((possible_coords.latitude >= pad + 34) & (possible_coords.latitude < pad + 45)) |
            ((possible_coords.latitude <= pad + 86) & (possible_coords.latitude > pad + 75))]
    elif region == 'tropics':  # <=23.5
        possible_coords = possible_coords.loc[
            (possible_coords.latitude >= pad + 45) & (possible_coords.latitude <= pad + 75)]

    return possible_coords
