import seaborn as sns
import matplotlib.pyplot as plt
import cartopy.crs as ccrs


def plot_save_prediction(da_pred, figname, folder='', save=False, title=None,
                         fcst_time=0, vertical=True, path='', round_projection = True):
    """Plots a probabilistic forecast.

    PARAMETERS:
    da_pred: data array of probabilistic forecast for one lead time and variable, with forecast_time axis
    figname: string, name of figure
    folder: string, name of folder where figure should be saved
    save: bool, wether to save plot or not. If string, filetype of saved plot
          e.g. 'png', 'svg', 'pdf'.

    """

    sns.set_context("paper")#"notebook")

    plt.figure() #figsize=[12, 8]
    
    if round_projection == True:
        proj =  ccrs.Robinson()
    else:
        proj = ccrs.PlateCarree(central_longitude=0) #

    from matplotlib import colors
    divnorm = colors.TwoSlopeNorm(vmin=0., vcenter=1/3, vmax=1)
    if vertical == True:
        p = da_pred.isel(forecast_time=fcst_time).plot(col='category',
                         vmin=0, vmax=1, col_wrap=1,  # this makes it vertical
                         cmap='coolwarm', norm=divnorm,
                         subplot_kws={'projection': proj},
                         transform=ccrs.PlateCarree(),
                         add_labels=True, add_colorbar=True,
                         cbar_kwargs={'orientation': 'horizontal',
                                      'pad': .1, 'shrink': 0.8, 'label': 'probability'})
        plt.subplots_adjust(wspace=0.05, hspace=0.05, bottom=0.25)
        if title is not None: plt.suptitle(f'{title}', y=0.9)
    else: #horizontal
        p = da_pred.isel(forecast_time = fcst_time).plot(col='category',
                         vmin=0, vmax=1,
                         cmap='coolwarm', norm=divnorm,
                         subplot_kws={'projection': proj},
                         transform=ccrs.PlateCarree(),
                         add_labels=True, add_colorbar=True,
                         cbar_kwargs={'orientation': 'vertical', 'anchor': (-0.1, 0.6),
                                      'shrink': 0.5, 'label': 'probability'})
        plt.subplots_adjust(wspace=0.05, hspace=0.05, right=0.75)
        if title is not None: plt.suptitle(f'{title}', y=0.825, x=0.45)

    for ax in p.axes.flat:
        ax.coastlines()
        ax.set_xlabel('')
        ax.set_ylabel('')
        ax.set_title(ax.get_title().replace('category = ', ''))

    if save == True:
        plt.savefig(f'{path}/plots/{folder}/{figname}.png', dpi=1200, bbox_inches='tight')
    elif save != False:
        plt.savefig(f'{path}/plots/{folder}/{figname}.{save}', dpi=1200, bbox_inches='tight')
    return

