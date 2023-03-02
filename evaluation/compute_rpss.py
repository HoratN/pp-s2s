import tensorflow as tf
import pandas as pd
import xarray as xr
xr.set_options(display_style='text')

from utils.helper_rpss import compute_rpss
from utils.paths import get_paths

import os
import warnings
warnings.simplefilter("ignore")

path = 'server'  # 'local'  #
cache_path, path_add_vars, path_model, path_pred, path_results = get_paths(path)
folder = 'main'

#%%
# =============================================================================
# compute globally aggregated RPSS and save to csv
# =============================================================================
compute_global_rpss = False
if compute_global_rpss == True:
    ls_rpss = []

    # go through models
    for filename in os.listdir(f'{path_model}{folder}'):
        name = filename

        # if predictions exist, compute skill
        if os.path.isfile(f'{path_pred}{folder}/global_pred_{name}_allyears_raw.nc') and  \
                os.path.isfile(f'{path_pred}{folder}/global_pred_{name}_2020_raw.nc'):

            if name.find('2023') != -1:  # use this to only compute rpss for new predictions
                print(name)

                v = filename.split('_')[0]
                for agg_domain in ['global']:  # 'europe'  # 'midlats'
                    rpss = compute_rpss(name, agg_domain, folder=folder, cache_path=cache_path, path_pred=path_pred)
                    ls_rpss.append(rpss)

    if len(ls_rpss) > 0:  # if computed rpss for at least one prediction
        # append rpss to file
        if os.path.isfile('rpss.csv'):
            pd.concat(ls_rpss, axis=0).to_csv(f'{path_results}rpss.csv', sep=';', index=True, mode='a', header=None)
        else:
            pd.concat(ls_rpss, axis=0).to_csv(f'{path_results}rpss.csv', sep=';', index=True, mode='a')
    else:
        print('no predictions available')

compute_regional_rpss = False
if compute_regional_rpss == True:
    ls_rpss = []

    # go through models
    for filename in os.listdir(f'{path_model}{folder}'):
        name = filename

        # if predictions exist, compute skill
        if os.path.isfile(f'{path_pred}{folder}/global_pred_{name}_allyears_raw.nc') and  \
                os.path.isfile(f'{path_pred}{folder}/global_pred_{name}_2020_raw.nc'):

            if name.find('2023') != -1:  # use this to only compute rpss for new predictions
                print(name)

                v = filename.split('_')[0]
                for agg_domain in ['NH','tropics','SH']:  # 'europe'  # 'midlats'
                    rpss = compute_rpss(name, agg_domain, folder=folder, cache_path=cache_path, path_pred=path_pred)
                    ls_rpss.append(rpss)

    if len(ls_rpss) > 0:  # if computed rpss for at least one prediction
        # append rpss to file
        if os.path.isfile('rpss_regional.csv'):
            pd.concat(ls_rpss, axis=0).to_csv(f'{path_results}rpss_regional.csv', sep=';', index=True, mode='a', header=None)
        else:
            pd.concat(ls_rpss, axis=0).to_csv(f'{path_results}rpss_regional.csv', sep=';', index=True, mode='a')
    else:
        print('no predictions available')

#%%

# =============================================================================
# compute spatial RPSS and save to nc
# =============================================================================
compute_spatial_rpss = False  # True  #
if compute_spatial_rpss == True:
    from utils.helper_rpss import get_spatial_rpss
    print('compute spatial distribution of RPSS')

    # go through models
    for filename in os.listdir(f'{path_model}{folder}'):
        name = filename

        # if predictions exist, compute skill
        if os.path.isfile(f'{path_pred}{folder}/global_pred_{name}_allyears_raw.nc') and  \
                os.path.isfile(f'{path_pred}{folder}/global_pred_{name}_2020_raw.nc'):

            if name.find('2023') != -1:  # use this to only compute for new predictions
                print(name)
                v = filename.split('_')[0]
                get_spatial_rpss(name, cache_path=cache_path, folder=folder, path_pred=path_pred)
