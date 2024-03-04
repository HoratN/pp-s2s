# Deep Learning for Postprocessing Global Probabilistic Forecasts on Subseasonal Time Scales

This repository provides Python code accompanying the paper
> Horat, N. and Lerch, S. (2024). Deep Learning for Postprocessing Global Probabilistic Forecasts on Subseasonal Time Scales. Monthly Weather Review, 152, 667–687, https://doi.org/10.1175/MWR-D-23-0150.1.
     

# Data
The data is available through the [S2S AI Challenge repository](https://renkulab.io/gitlab/aaron.spring/s2s-ai-challenge-template/-/tree/master/data) and a dedicated [CliMetLab plugin](https://github.com/ecmwf-lab/climetlab-s2s-ai-challenge) that provides forecast data for additional variables and from other forecasting centers.
## (Re-)Forecasts from ECMWF
-	Variables: 
    - temperature (t2m), averaged over two weeks
    - precipitation (tp), accumulated over two weeks
    - additional predictors (slp, gh850, gh500, tcw), averaged over two weeks
-	Lead time: 14 days (predictions for weeks 3-4), 28 days (weeks 5-6)
-	Time period:  2000 - 2019 (reforecasts), 2020 (forecasts)
-	Ensemble size: 11 members (reforecasts), 51 members (forecasts)
-	Data coverage: global, gridded at 1.5°

## Observations from NOAA Climate Prediction Center (CPC)
-	Variables: temperature and precipitation
-	Time period: 2000 – 2020
-	Data coverage: global over land, gridded at 1.5°

# Post-processing
-	Benchmarks: climatology and (corrected) calibrated ECMWF baseline
-	Standard CNN methods adapted to global probabilistic forecasting: BF-CNN and TC-CNN
-	UNet methods: UNet models trained on patches (UNet patch-wise) and on the global forecast fields (UNet global).

# Code
The *models* folder contains the classes for the different model architectures, whose model instances can be trained and used for prediction with the scripts in the *training* folder. The *evaluation* folder contains scripts to compute average predictions and scores, as well as baselines not provided by the S2S AI Challenge. All helper functions are gathered in the folder *utils*.

|Folder|File| Description|
|----|-------------|---------------|
|**evaluation**|compute_average.py|Script to average over predictions from 10-fold cross-validation.
||compute_corrected_baseline.py|Script to compute the corrected ECMWF baseline with cumulative probabilities summing up to one.
||compute_rpss.py|Script to compute the RPSS of the predictions after inference.|
|**models**|standard_cnn.py|Class of the standard CNN methods adapted for post-processing global forecast fields  (BF-CNN and TC-CNN).|
||unet.py|Class of the UNet models (UNet global and UNet patch-wise) for post-processing global forecast fields.|
|**results**||Place holder for saving the trained models and the model predictions.|
|**training**|predict.py|Python script for inference.|
||train_model|Python script for training all post-processing architectures and saving the model weights.|
|**utils**|helper_average_pred.py|Helper functions to compute and save average prediction from 10-fold cross-validation.|
||helper_baseline.py|Helper function to compute the corrected ECMWF baseline.|
||helper_datagenerator.py|Contains the data generator classes for global and patch-wise training and helper functions related to padding and patch creation.|
||helper_load_data.py|Helper functions to load the data and consolidate the data format. Function to create the basis functions.|
||helper_plot.py|Function to plot tercile predictions.|
||helper_predict.py|Functions used for inference.|
||helper_preprocess.py|Functions for computing and standardizing the features.|
||helper_rpss.py|Functions to compute RPSS values based on the spatial predictions.|
||helper_train.py|Functions used to fit the model, including weighted loss and early stopping. Function to save information on model hyperparameters.|
||paths.py|Function to get paths to data and results folder. You can adapt the paths here to your folder structure. The code is written to work on two different folder structures, server and local.|
||scripts.py|[From S2S AI Challenge](https://renkulab.io/gitlab/aaron.spring/s2s-ai-challenge-template/-/tree/master/notebooks).|
