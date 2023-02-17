def get_paths(path):
    """
    get paths for data, predictions, trained models and result.csv based on the current folder structure
    adapt the paths below according to your folder structure

    Parameters
    ----------
    path: string, indicates what folder structure is present

    Returns
    -------
    cache_path: string, path to data provided by S2S AI Challenge
    path_add_vars: string, path to additional data
    path_model: string, path to folder where trained models should be saved
    path_pred: string, path to folder where predictions should be saved
    path_results: string, path to folder where all results including results.csv are saved
    """

    # paths to load forecasts and observations
    if path == 'server':
        cache_path = '../../../../Data/s2s_ai/data'
        path_add_vars = '../../../../Data/s2s_ai'
    elif path == 'local':
        cache_path = '../../s2s-ai-challenge-my-fork/template/data'
        path_add_vars = '../../s2s-ai-challenge-my-fork/data'

    # path to models
    if path == 'local':
        path_model = '../results/trained_models/'
    else:
        path_model = './models/'

    # path to save predictions
    if path == 'server':
        path_pred = '../../../../Data/s2s_ai/predictions/'
    else:
        path_pred = '../results/predictions/'

    # path to results.csv
    if path == 'local':
        path_results = '../results/'
    else:
        path_results = ''

    return cache_path, path_add_vars, path_model, path_pred, path_results
