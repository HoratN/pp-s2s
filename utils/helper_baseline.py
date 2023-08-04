import xarray as xr
xr.set_options(display_style='text')
from utils.scripts import make_probabilistic
from utils.paths import get_paths
from utils.helper_load_data import load_data


def compute_corrected_baseline(path_data, time_period='test', impute=True):
    """
    compute probabilistic forecasts analogous to the ecmwf baseline
    (i.e. w.r.t. model tercile edges, but with cumulative probability = 1 everywhere)

    Parameters
    ----------
    path_data: string, one of {local, server}
    time_period: sting, one of {test, train}, indicates for what time period the baseline should be computed
    impute: bool, can choose whether to fill dry locations with missing values with 1/3.

    Returns
    -------
    corr_baseline: DataSet
    """

    cache_path, path_add_vars, path_model, path_pred, path_results = get_paths(path_data)
    if time_period == 'test':
        ds = load_data(data='forecast_2020', path=path_data)
    else:
        ds = load_data(data='hind_2000-2019', path=path_data)

    tercile_edges_model = xr.open_dataset(f'{path_add_vars}/ecmwf_tercile_edges_2000-2019.nc')

    corr_baseline = make_probabilistic(ds, tercile_edges_model)
    if impute == True:
        corr_baseline = corr_baseline.fillna(1 / 3)

    return corr_baseline