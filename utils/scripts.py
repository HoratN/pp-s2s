""" taken from the S2S AI Challenge
    https://renkulab.io/gitlab/aaron.spring/s2s-ai-challenge-template/-/tree/master/notebooks
    Outcomes of the WMO Prize Challenge to Improve Sub-Seasonal
    to Seasonal Predictions Using Artificial Intelligence, Vitart et al. 2022
"""

import xarray as xr
import pandas as pd
import numpy as np
import climetlab_s2s_ai_challenge
import climetlab as cml

cache_path = '../template/data'


def download(varlist_forecast=['tp','t2m'],
             center_list=['ecwmf'],
             forecast_dataset_labels=['hindcast-input','forecast-input'],
             obs_dataset_labels=['hindcast-like-observations','forecast-like-observations'],
             varlist_observations=['t2m','tp'],
             benchmark=True,
             format='netcdf'
            ):
    """Download files with climetlab to cache_path. Set cache_path:
    cml.settings.set("cache-directory", cache_path)
    """
    if isinstance(center_list, str):
        center_list = [center_list]
    if isinstance(varlist_forecast, str):
        varlist_forecast = [varlist_forecast]

    dates = xr.cftime_range(start='20200102',freq='7D', periods=53).strftime('%Y%m%d').to_list()
    
    if forecast_dataset_labels:
        print(f'Downloads variables {varlist_forecast} from datasets {forecast_dataset_labels} from center {center_list} in {format} format.')
        for center in center_list:
            for ds in forecast_dataset_labels:
                for parameter in varlist_forecast: 
                    try:
                        cml.load_dataset(f"s2s-ai-challenge-{ds}", origin=center, parameter=varlist_forecast, format=format).to_xarray()
                    except:
                        pass
    if obs_dataset_labels:
        print(f'Downloads variables tp and t2m from datasets {obs_dataset_labels} netcdf format. Additionally downloads raw t2m and pr observations with a time dimension.')
        try:
            for ds in obs_dataset_labels:
                for parameter in varlist_observations:
                    cml.load_dataset(f"s2s-ai-challenge-{ds}", date=dates, parameter=parameter).to_xarray()
        except:
            pass
        # raw
        cml.load_dataset(f"s2s-ai-challenge-observations", parameter=varlist_observations).to_xarray()
    if benchmark:
        cml.load_dataset("s2s-ai-challenge-test-output-benchmark", parameter=['tp','t2m']).to_xarray()
    print('finished')
    return


def add_valid_time_from_forecast_reference_time_and_lead_time(forecast, init_dim='forecast_time'):
    """Creates valid_time(forecast_time, lead_time).
    
    lead_time: pd.Timedelta
    forecast_time: datetime
    """
    times = xr.concat(
        [
            xr.DataArray(
                forecast[init_dim] + lead,
                dims=init_dim,
                coords={init_dim: forecast[init_dim]},
            )
            for lead in forecast.lead_time
        ],
        dim="lead_time",
        join="inner",
        compat="broadcast_equals",
    )
    forecast = forecast.assign_coords(valid_time=times)
    return forecast


def aggregate_biweekly(da):
    """
    Aggregate initialized S2S forecasts biweekly for xr.DataArrays.
    Use ds.map(aggregate_biweekly) for xr.Datasets.
    
    Applies to the ECMWF S2S data model: https://confluence.ecmwf.int/display/S2S/Parameters
    """
    # biweekly averaging
    w34 = [pd.Timedelta(f'{i} d') for i in range(14,28)]
    w34 = xr.DataArray(w34,dims='lead_time', coords={'lead_time':w34})
    
    w56 = [pd.Timedelta(f'{i} d') for i in range(28,42)]
    w56 = xr.DataArray(w56,dims='lead_time', coords={'lead_time':w56})
    
    biweekly_lead = [pd.Timedelta(f"{i} d") for i in [14, 28]] # take first day of biweekly average as new coordinate

    v = da.name
    if climetlab_s2s_ai_challenge.CF_CELL_METHODS[v] == 'sum': # biweekly difference for sum variables: tp and ttr
        d34 = da.sel(lead_time=pd.Timedelta("28 d")) - da.sel(lead_time=pd.Timedelta("14 d")) # tp from day 14 to day 27
        d56 = da.sel(lead_time=pd.Timedelta("42 d")) - da.sel(lead_time=pd.Timedelta("28 d")) # tp from day 28 to day 42
        da_biweekly = xr.concat([d34,d56],'lead_time').assign_coords(lead_time=biweekly_lead)
    else: # t2m, see climetlab_s2s_ai_challenge.CF_CELL_METHODS # biweekly: mean [day 14, day 27]
        d34 = da.sel(lead_time=w34).mean('lead_time')
        d56 = da.sel(lead_time=w56).mean('lead_time')
        da_biweekly = xr.concat([d34,d56],'lead_time').assign_coords(lead_time=biweekly_lead)
    
    da_biweekly = add_valid_time_from_forecast_reference_time_and_lead_time(da_biweekly)
    da_biweekly['lead_time'].attrs = {'long_name':'forecast_period', 'description': 'Forecast period is the time interval between the forecast reference time and the validity time.',
                         'aggregate': 'The pd.Timedelta corresponds to the first day of a biweekly aggregate.',
                         'week34_t2m': 'mean[day 14, 27]',
                         'week56_t2m': 'mean[day 28, 41]',
                         'week34_tp': 'day 28 minus day 14',
                         'week56_tp': 'day 42 minus day 28'}
    return da_biweekly


def ensure_attributes(da, biweekly=False):
    """Ensure that coordinates and variables have proper attributes. Set biweekly==True to set special comments for the biweely aggregates."""
    template = cml.load_dataset('s2s-ai-challenge-test-input',parameter='t2m', origin='ecmwf', format='netcdf', date='20200102').to_xarray()
    for c in da.coords:
        if c in template.coords:
            da.coords[c].attrs.update(template.coords[c].attrs)
    
    if 'valid_time' in da.coords:
        da['valid_time'].attrs.update({'long_name': 'validity time',
                                     'standard_name': 'time',
                                     'description': 'time for which the forecast is valid',
                                     'calculate':'forecast_time + lead_time'})
    if 'forecast_time' in da.coords:
        da['forecast_time'].attrs.update({'long_name' : 'initial time of forecast', 'standard_name': 'forecast_reference_time',
                                      'description':'The forecast reference time in NWP is the "data time", the time of the analysis from which the forecast was made. It is not the time for which the forecast is valid.'})
    # fix tp
    if da.name == 'tp':
        da.attrs['units'] = 'kg m-2'
    if biweekly:
        da['lead_time'].attrs.update({'standard_name':'forecast_period', 'long_name': 'lead time',
                                      'description': 'Forecast period is the time interval between the forecast reference time and the validity time.',
                         'aggregate': 'The pd.Timedelta corresponds to the first day of a biweekly aggregate.',
                         'week34_t2m': 'mean[14 days, 27 days]',
                         'week56_t2m': 'mean[28 days, 41 days]',
                         'week34_tp': '28 days minus 14 days',
                         'week56_tp': '42 days minus 28 days'})
        if da.name == 'tp':
            da.attrs.update({'aggregate_week34': '28 days minus 14 days',
                      'aggregate_week56': '42 days minus 28 days',
                      'description': 'https://confluence.ecmwf.int/display/S2S/S2S+Total+Precipitation'})
        if da.name == 't2m':
            da.attrs.update({'aggregate_week34': 'mean[14 days, 27 days]',
                      'aggregate_week56': 'mean[28 days, 41 days]',
                      'variable_before_categorization': 'https://confluence.ecmwf.int/display/S2S/S2S+Surface+Air+Temperature'})
    return da


def add_year_week_coords(ds):
    import numpy as np
    if 'week' not in ds.coords and 'year' not in ds.coords:
        year = ds.forecast_time.dt.year.to_index().unique()
        week = (list(np.arange(1,54)))
        weeks = week * len(year)
        years = np.repeat(year,len(week))
        ds.coords["week"] = ("forecast_time", weeks)
        ds.coords['week'].attrs['description'] = "This week represents the number of forecast_time starting from 1 to 53. Note: This week is different from the ISO week from groupby('forecast_time.weekofyear'), see https://en.wikipedia.org/wiki/ISO_week_date and https://renkulab.io/gitlab/aaron.spring/s2s-ai-challenge/-/issues/29"
        ds.coords["year"] = ("forecast_time", years)
        ds.coords['year'].attrs['long_name'] = "calendar year"
    return ds


def make_probabilistic(ds, tercile_edges, member_dim='realization', mask=None, groupby_coord='week'):
    """Compute probabilities from ds (observations or forecasts) based on tercile_edges."""
    # broadcast
    ds = add_year_week_coords(ds)
    tercile_edges = tercile_edges.sel({groupby_coord: ds.coords[groupby_coord]})
    bn = ds < tercile_edges.isel(category_edge=0, drop=True)  # below normal
    n = (ds >= tercile_edges.isel(category_edge=0, drop=True)) & (ds < tercile_edges.isel(category_edge=1, drop=True))  # normal
    an = ds >= tercile_edges.isel(category_edge=1, drop=True)  # above normal

    if member_dim in ds.dims:# using this, the function can deal with nans correctly
        denominator = ds.notnull().sum(member_dim)
        bn = bn.sum(member_dim)/denominator
        an = an.sum(member_dim)/denominator
        n = n.sum(member_dim)/denominator
# =============================================================================
#     if member_dim in ds.dims:
#         bn = bn.mean(member_dim)
#         an = an.mean(member_dim)
#         n = n.mean(member_dim)
# =============================================================================
    ds_p = xr.concat([bn, n, an],'category').assign_coords(category=['below normal', 'near normal', 'above normal'])
    if mask is not None:
        ds_p = ds_p.where(mask)
    if 'tp' in ds_p.data_vars:
        # mask arid grid cells where category_edge are too close to 0
        # we are using a dry mask as in https://doi.org/10.1175/MWR-D-17-0092.1
        tp_arid_mask = tercile_edges.tp.isel(category_edge=0, drop=True) > 0.01#lead_time=0,
        ds_p['tp'] = ds_p['tp'].where(tp_arid_mask)
    ds_p['category'].attrs = {'long_name': 'tercile category probabilities', 'units': '1',
                        'description': 'Probabilities for three tercile categories. All three tercile category probabilities must add up to 1.'}
    if 'tp' in ds_p.data_vars:
        ds_p['tp'].attrs = {'long_name': 'Probability of total precipitation in tercile categories', 'units': '1',
                          'comment': 'All three tercile category probabilities must add up to 1.',
                          'variable_before_categorization': 'https://confluence.ecmwf.int/display/S2S/S2S+Total+Precipitation'
                         }
    if 't2m' in ds_p.data_vars:
        ds_p['t2m'].attrs = {'long_name': 'Probability of 2m temperature in tercile categories', 'units': '1',
                          'comment': 'All three tercile category probabilities must add up to 1.',
                          'variable_before_categorization': 'https://confluence.ecmwf.int/display/S2S/S2S+Surface+Air+Temperature'
                          }
    if 'year' in ds_p.coords:
        del ds_p.coords['year']
    if groupby_coord in ds_p.coords:
        ds_p = ds_p.drop(groupby_coord)
    return ds_p


def skill_by_year(preds, cache_path = '../template/data', adapt=False):
    """Returns pd.Dataframe of RPSS per year."""
    # similar verification_RPSS.ipynb
    # as scorer bot but returns a score for each year
    import xarray as xr
    import xskillscore as xs
    import pandas as pd
    import numpy as np
    xr.set_options(keep_attrs=True)
    
    # from root
    #renku storage pull data/forecast-like-observations_2020_biweekly_terciled.nc
    #renku storage pull data/hindcast-like-observations_2000-2019_biweekly_terciled.nc
    #cache_path = '../template/data'
    if 2020 in preds.forecast_time.dt.year:
        obs_p = xr.open_dataset(f'{cache_path}/forecast-like-observations_2020_biweekly_terciled.nc').sel(forecast_time=preds.forecast_time)
    else:
        obs_p = xr.open_dataset(f'{cache_path}/hindcast-like-observations_2000-2019_biweekly_terciled.zarr', engine='zarr').sel(forecast_time=preds.forecast_time)
    
    # ML probabilities
    fct_p = preds

    
    # climatology
    clim_p = xr.DataArray([1/3, 1/3, 1/3], dims='category', coords={'category':['below normal', 'near normal', 'above normal']}).to_dataset(name='tp')
    clim_p['t2m'] = clim_p['tp']
    
    if adapt:
        # select only obs_p where fct_p forecasts provided
        for c in ['longitude', 'latitude', 'forecast_time', 'lead_time']:
            obs_p = obs_p.sel({c:fct_p[c]})
        obs_p = obs_p[list(fct_p.data_vars)]
        clim_p = clim_p[list(fct_p.data_vars)]
    
    else:
        # check inputs
        assert_predictions_2020(obs_p)
        assert_predictions_2020(fct_p)
        
    # rps_ML
    rps_ML = xs.rps(obs_p, fct_p, category_edges=None, dim=[], input_distributions='p').compute()
    # rps_clim
    rps_clim = xs.rps(obs_p, clim_p, category_edges=None, dim=[], input_distributions='p').compute()

    ## RPSS
    # penalize # https://renkulab.io/gitlab/aaron.spring/s2s-ai-challenge-template/-/issues/7
    expect = obs_p.sum('category')
    expect = expect.where(expect > 0.98).where(expect < 1.02)  # should be True if not all NaN

    # https://renkulab.io/gitlab/aaron.spring/s2s-ai-challenge-template/-/issues/50
    rps_ML = rps_ML.where(expect, other=2)  # assign RPS=2 where value was expected but NaN found

    # following Weigel 2007: https://doi.org/10.1175/MWR3280.1
    rpss = 1 - (rps_ML.groupby('forecast_time.year').mean() / rps_clim.groupby('forecast_time.year').mean())
    # clip
    rpss = rpss.clip(-10, 1)
    
    # weighted area mean
    weights = np.cos(np.deg2rad(np.abs(rpss.latitude)))
    # spatially weighted score averaged over lead_times and variables to one single value
    scores = rpss.sel(latitude=slice(None, -60)).weighted(weights).mean('latitude').mean('longitude')
    scores = scores.to_array().mean(['lead_time', 'variable'])
    return scores.to_dataframe('RPSS')


def assert_predictions_2020(preds_test, exclude='weekofyear'):
    """Check the variables, coordinates and dimensions of 2020 predictions."""
    from xarray.testing import assert_equal # doesnt care about attrs but checks coords
    
    # is dataset
    assert isinstance(preds_test, xr.Dataset)

    # has both vars: tp and t2m
    if 'data_vars' in exclude:
        assert 'tp' in preds_test.data_vars
        assert 't2m' in preds_test.data_vars
    
    ## coords
    # ignore weekofyear coord if not dim
    if 'weekofyear' in exclude and 'weekofyear' in preds_test.coords and 'weekofyear' not in preds_test.dims:
        preds_test = preds_test.drop('weekofyear')
    
    # forecast_time
    if 'forecast_time' in exclude:
        d = pd.date_range(start='2020-01-02', freq='7D', periods=53)
        forecast_time = xr.DataArray(d, dims='forecast_time', coords={'forecast_time':d}, name='forecast_time')
        assert_equal(forecast_time,  preds_test['forecast_time'])

    # longitude
    if 'longitude' in exclude:
        lon = np.arange(0., 360., 1.5)
        longitude = xr.DataArray(lon, dims='longitude', coords={'longitude': lon}, name='longitude')
        assert_equal(longitude, preds_test['longitude'])

    # latitude
    if 'latitude' in exclude:
        lat = np.arange(-90., 90.1, 1.5)[::-1]
        latitude = xr.DataArray(lat, dims='latitude', coords={'latitude': lat}, name='latitude')
        assert_equal(latitude, preds_test['latitude'])
    
    # lead_time
    if 'lead_time' in exclude:
        lead = [pd.Timedelta(f'{i} d') for i in [14, 28]]
        lead_time = xr.DataArray(lead, dims='lead_time', coords={'lead_time': lead}, name='lead_time')
        assert_equal(lead_time, preds_test['lead_time'])
    
    # category
    if 'category' in exclude:
        cat = np.array(['below normal', 'near normal', 'above normal'], dtype='<U12')
        category = xr.DataArray(cat, dims='category', coords={'category': cat}, name='category')
        assert_equal(category, preds_test['category'])
    
    # size
    if 'size' in exclude:
        from dask.utils import format_bytes
        size_in_MB = float(format_bytes(preds_test.nbytes).split(' ')[0])
        # todo: refine for dtypes
        assert size_in_MB > 50
        assert size_in_MB < 250
    
    # no other dims
    if 'dims' in exclude:
        assert set(preds_test.dims) - {'category', 'forecast_time', 'latitude', 'lead_time', 'longitude'} == set()
    
