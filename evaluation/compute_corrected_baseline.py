import tensorflow.keras as keras

import xarray as xr
xr.set_options(display_style='text')

from utils.helper_load_data import get_data, load_data
from utils.scripts import add_year_week_coords, make_probabilistic
from utils.paths import get_paths
from utils.helper_plot import plot_save_prediction
from utils.helper_baseline import compute_corrected_baseline

import seaborn as sns
import matplotlib.pyplot as plt
import cartopy.crs as ccrs

import os
import warnings

warnings.simplefilter("ignore")

folder = 'corrected_baseline'
path_data = 'local'  # 'server'
cache_path, path_add_vars, path_model, path_pred, path_results = get_paths(path_data)

#%%
# load data
hind_2000_2019, obs_2000_2019, obs_2000_2019_terciled, mask = get_data(['tp', 't2m'], path_data)
print(list(hind_2000_2019.data_vars))

# %%
# =============================================================================
# compute and save tercile edges for model climatology
# =============================================================================

if os.path.isfile(f'{path_add_vars}/ecmwf_tercile_edges_2000-2019.nc'):
    print("ecmwf model tercile edges exist")
else:
    hind = add_year_week_coords(hind_2000_2019)
    # 1h on CPU-Server:
    tercile_edges_model = hind.chunk({'forecast_time': -1, 'longitude': 'auto'}
                                     ).groupby('week'
                                               ).quantile(q=[1./3., 2./3.], dim=('forecast_time', 'realization')
                                                          ).rename({'quantile': 'category_edge'}).compute()

    tercile_edges_model.astype('float32').to_netcdf(f'{path_add_vars}/ecmwf_tercile_edges_2000-2019.nc')

# %%
# =============================================================================
# compute corrected baseline
# =============================================================================

# create probabilistic forecasts
corr_baseline_hind = compute_corrected_baseline(path_data=path_data, time_period='train')
corr_baseline_2020 = compute_corrected_baseline(path_data=path_data, time_period='test')

# check that corrected baseline has cumulative probability = 1:
print(corr_baseline_2020.sum('category').tp.max().values)
print(corr_baseline_2020.sum('category').tp.min().values)

# %%
# =============================================================================
# plotting
# =============================================================================
baseline = xr.open_dataset(f'{cache_path}/ecmwf_recalibrated_benchmark_2020_biweekly_terciled.nc')

# cumulative probability of original ECMWF baseline
sns.set_context("paper", font_scale=0.85)
plt.figure()
proj = ccrs.Robinson()
p = baseline.sum('category', skipna=False).mean('forecast_time').tp.plot(
            col='lead_time', vmin=1, vmax=1.3,
            cmap='hot_r', subplot_kws={'projection': proj}, transform=ccrs.PlateCarree(),
            add_labels=True, add_colorbar=True, levels=[1, 1.01, 1.05, 1.1, 1.2, 1.3],
            cbar_kwargs={'orientation': 'vertical', 'anchor': (-0.2, 0.55),
                         'shrink': 0.5, 'label': 'cumulative probability'})
plt.subplots_adjust(wspace=0.05, hspace=0.05, right=0.75)

for ax in p.axes.flat:
    ax.coastlines()
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.set_title(ax.get_title().replace('lead_time = ', 'lead time: '))
    ax.set_title(ax.get_title().replace('00:00:00', ''))

plt.savefig(f'{path_results}plots/{folder}/ecmwf_baseline_cumulative_probability.png', dpi=1200, bbox_inches='tight')

#%%

# average number of nans per (11-)member ensemble in the hindcast data
# used to compute the tercile edges

sns.set_context("paper", font_scale=0.85)
plt.figure()
proj = ccrs.Robinson()
p = hind_2000_2019.tp.isnull().sum('realization'
                                   ).mean('forecast_time'
                                          ).plot(
            col='lead_time', cmap='hot_r', subplot_kws={'projection': proj},
            transform=ccrs.PlateCarree(), add_labels=True, add_colorbar=True,
            levels=[0, 0.5, 1, 2, 3, 5],
            cbar_kwargs={'orientation': 'vertical', 'anchor': (-0.2, 0.55),
                         'shrink': 0.5, 'label': 'number of missing values'})
plt.subplots_adjust(wspace=0.05, hspace=0.05, right=0.75)

for ax in p.axes.flat:
    ax.coastlines()
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.set_title(ax.get_title().replace('lead_time = ', 'lead time: '))
    ax.set_title(ax.get_title().replace('00:00:00', ''))

#%%

# example predictions
plot_save_prediction(corr_baseline_2020.tp.isel(lead_time=1), 
                     figname='tp_corrected', folder='', save=False,
                     title='corrected ECMWF baseline for precipitation with 28 days lead time',
                     fcst_time=0, vertical=False)

plot_save_prediction(baseline.tp.isel(lead_time=1), 
                     figname='tp_ecmwf', folder='', save=False,
                     title='original ECMWF baseline for precipitation with 28 days lead time',
                     fcst_time=0, vertical=False)

# differences corrected baseline and baseline
sns.set_context("paper")
plt.figure()
proj = ccrs.Robinson()
p = (corr_baseline_2020 - baseline).tp.isel(lead_time=1).mean('forecast_time'
                    ).plot(col='category',
                            vmin=-0.3, vmax=0.3, cmap='coolwarm',
                            subplot_kws={'projection': proj}, transform=ccrs.PlateCarree(),
                            add_labels=True, add_colorbar=True,
                            cbar_kwargs={'orientation': 'vertical', 'anchor': (-0.1, 0.6),
                                         'shrink': 0.5, 'label': 'probability'})
plt.subplots_adjust(wspace=0.05, hspace=0.05, right=0.75)
plt.suptitle('corrected - original ECMWF baseline for precipitation with 28 days lead time', y=0.825, x=0.45)

for ax in p.axes.flat:
    ax.coastlines()
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.set_title(ax.get_title().replace('category = ', ''))
