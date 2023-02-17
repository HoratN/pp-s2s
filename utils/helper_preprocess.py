import xarray as xr
xr.set_options(display_style='text')

from utils.scripts import add_year_week_coords
from utils.helper_load_data import load_data


def preprocess_input(train, v, path_data, lead_time, valid=None, ens_mean=True, spatial_std=True):
    """
    preprocess ensemble forecast input:
        - compute ensemble mean
        - remove annual cycle from non-target features
        - subtract tercile edges from target variable features
        - standardize using standard deviation

    Parameters
    ----------
    train: DataSet, contains ensemble forecasts for one lead time from training data
    v: string, target variable
    path_data: string, path used for load_data
    lead_time: int, lead time in fct (for isel)
    valid: DataSet, if you also want to preprocess validation data
    ens_mean: bool, whether to only use th ensemble mean
    spatial_std: bool, standard deviation used to normalize all features,
                       if False, standard deviation across lat/lon and average over forecast time,
                       if True, std is a spatial field,

    Returns
    -------
    train: DataSet, preprocessed
    valid: DataSet, if given, also preprocessed

    """
    print('use ensemble mean: ', ens_mean)
    if valid != None:
        assert (train.keys() == valid.keys()), f'train and validation set do not contain the same variables: train {list(train.keys())}, val {list(valid.keys())}'

    ls_target_vars = [i for i in train.keys() if i in [v]]  # list of target var features
    ls_rm_annual = [i for i in train.keys() if i not in ['lsm', v]]  # list of vars for which annual cycle should be removed

    # use ensemble mean
    if ens_mean == True:
        train = train.mean('realization')

    if valid != None:
        reference = train.copy(deep=True)

    # remove annual cycle from non-target features
    # don't remove annual cycle from land-sea mask and target variable
    train.update(rm_annualcycle(train[ls_rm_annual], train[ls_rm_annual]))

    # preprocessing for target var: subtract tercile edges from target variable
    if len(ls_target_vars) > 0:
        target_features = target_features_tercile_distance(v, train[ls_target_vars],
                                                           path_data, lead_time)
        # merge all features and drop original target variable feature
        train = xr.merge([train, target_features])
        train = train.drop_vars(ls_target_vars)

    # standardization
    if spatial_std == False:  # standard deviation across lat/lon of the ensemble mean
        # todo: weighted std
        train_std = train.std(('latitude', 'longitude')).mean('forecast_time')
    else:
        train_std = train.std('forecast_time')
    train = train / train_std

    # =============================================================================
    # do the same for valid using train whenever necessary
    # =============================================================================
    if valid != None:

        # use ensemble mean
        if ens_mean == True:
            valid = valid.mean('realization')

        # remove annual cycle from non-target features
        valid.update(rm_annualcycle(valid[ls_rm_annual], reference[ls_rm_annual]))

        # preprocessing for target var: subtract tercile edges from target variable
        if len(ls_target_vars) > 0:
            target_features_valid = target_features_tercile_distance(v, valid[ls_target_vars],
                                                                     path_data, lead_time)
            # merge all features and drop original target variable feature
            valid = xr.merge([valid, target_features_valid])
            valid = valid.drop_vars(ls_target_vars)

        # standardization
        valid = valid / train_std

    if valid != None:
        return train, valid
    else:
        return train


def rm_annualcycle(ds, ds_train):
    """
    remove annual cycle for each location

    Parameters
    ----------
    ds: DataSet, this is the dataset from which the annual cycle should be removed
    ds_train: DataSet, this is the dataset from which the annual cycle is computed

    Returns
    -------
    ds_stand: ds with removed annual cycle
    """

    ds = add_year_week_coords(ds)
    ds_train = add_year_week_coords(ds_train)

    if 'realization' in ds_train.coords:  # always use train data to compute the annual cycle
        ann_cycle = ds_train.mean('realization').groupby('week').mean(['forecast_time'])
    else:
        ann_cycle = ds_train.groupby('week').mean(['forecast_time'])

    ds_stand = ds - ann_cycle

    ds_stand = ds_stand.sel({'week': ds.coords['week']})
    ds_stand = ds_stand.drop(['week', 'year'])

    return ds_stand


def target_features_tercile_distance(v, fct, path_data, lead_time):
    """
    Compute distance from ensemble mean to the tercile edges,
    wrapper to rm_tercile_edges that saves distances under new
    variable names lower_{v}, upper_{v}

    Parameters
    ----------
    v: string, target variable
    fct: DataSet, only contains target variable v
    path_data: string, path used for load-data to load the tercile edges
    lead_time: int, lead time in fct (for isel)

    Returns
    -------
    dist_renamed: DataSet
    """
    tercile_edges = load_data(data='obs_tercile_edges_2000-2019',
                              aggregation='biweekly', path=path_data
                              ).isel(lead_time=lead_time)[v]
    dist = rm_tercile_edges(fct, tercile_edges)

    # rename the new tercile features
    ls_dist = []
    for var in list(dist.keys()):
        ls_dist.append(
            dist[var].assign_coords(category_edge=['lower_{}'.format(var),
                                                   'upper_{}'.format(var)]
                                    ).to_dataset(dim='category_edge')
        )
    dist_renamed = xr.merge(ls_dist)

    return dist_renamed


def rm_tercile_edges(ds, tercile_edges):
    """
    compute distance to tercile edges

    Parameters
    ----------
    ds: DataSet
    tercile_edges: DataSet, loaded tercile edges

    Returns
    -------
    ds_stand: distance to tercile edges
    """

    ds = add_year_week_coords(ds)

    ds_stand = tercile_edges - ds

    ds_stand = ds_stand.sel({'week': ds.coords['week']})
    ds_stand = ds_stand.drop(['week', 'year'])

    return ds_stand
