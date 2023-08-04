# Deep learning for post-processing global probabilistic forecasts on sub-seasonal time scales

This repository provides Python code for post-processing global probabilistic forecasts for temperature and precipitation on sub-seasonal time scales.

# Data
The data is available through the S2S AI Challenge repository (https://renkulab.io/gitlab/aaron.spring/s2s-ai-challenge-template/-/tree/master/data) and a dedicated CliMetLab plugin that provides forecast data for additional variables and from other forecasting centers (https://github.com/ecmwf-lab/climetlab-s2s-ai-challenge).
## (Re-)Forecasts from ECMWF
-	Variables: 
    - temperature (t2m), averaged over two weeks
    - precipitation (tp), accumulated over two weeks
    - additional predictors, averaged over two weeks
-	Lead time: 14 days (predictions for weeks 3-4), 28 days (weeks 5-6)
-	Time period:  2000 - 2019 (reforecasts), 2020 (forecasts)
-	Ensemble size: 11 members (reforecasts), 51 members (forecasts)
-	Data coverage: global, gridded at 1.5°

## Observations from NOAA Climate Prediction Center (CPC)
-	Variables: temperature and precipitation
-	Time period: 2000 – 2020
-	Data coverage: global over land, gridded at 1.5°

# Post-processing
-	Benchmarks: climatology and calibrated ECMWF baseline
-	Standard CNN methods adapted to global probabilistic forecasting: BF-CNN and TC-CNN
-	UNet methods: UNet models trained on patches (UNet patch-wise) and on the global forecast fields (UNet global).

# Code
|Folder|File| Description|
|----|-------------|---------------|
|evaluation|compute_rpss.py|Script to compute the RPSS of the predictions after inference.|
|models|standard_cnn.py|Class of the standard CNN methods adapted for post-processing global forecast fields  (BF-CNN and TC-CNN).|
|models|unet.py|Class of the UNet models (UNet global and UNet patch-wise) for post-processing global forecast fields.|
|results||Place holder for saving the trained models and the model predictions.|
|training|predict.py|Python script for inference.|
|training|train_model|Python script for training all post-processing architectures and saving the model weights.| 
