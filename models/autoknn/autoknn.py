#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#
# AutoKNN: Locally linear regression with fixed and nearest neighbor lags
# See https://arxiv.org/abs/1809.07394 for more details
#
import os, sys
from subseasonal_toolkit.utils.notebook_util import isnotebook

if isnotebook():
    # Autoreload packages that are modified
    get_ipython().run_line_magic('load_ext', 'autoreload')
    get_ipython().run_line_magic('autoreload', '2')
else:
    from argparse import ArgumentParser

import numpy as np
import pandas as pd
import sys
import time
from datetime import datetime, timedelta
from multiprocessing import cpu_count
from functools import partial
from ttictoc import tic, toc
from subseasonal_data.utils import get_measurement_variable
from subseasonal_toolkit.utils.experiments_util import get_first_year, get_start_delta, clim_merge, month_day_subset
from subseasonal_toolkit.utils.fit_and_predict import apply_parallel
from subseasonal_toolkit.utils.skill import *
from subseasonal_toolkit.utils.eval_util import get_target_dates, mean_rmse_to_score, save_metric
from subseasonal_toolkit.utils.models_util import (get_submodel_name, start_logger, log_params, get_forecast_filename,
                                                   save_forecasts)
from subseasonal_toolkit.models.autoknn.autoknn_util import *
from subseasonal_toolkit.models.perpp.perpp_util import fit_and_predict
from subseasonal_data import data_loaders


# # Model Parameters

# In[ ]:


#
# Specify model parameters
#

if not isnotebook():
    # If notebook run as a script, parse command-line arguments
    parser = ArgumentParser()
    parser.add_argument("pos_vars",nargs="*",
                        help="specify gt_id (e.g., \"contest_tmp2m\") "
                             "and horizon (e.g., \"34w\")")
    parser.add_argument('--history', default=60,
                       help="Number of past days that should contribute to measure of similarity")
    parser.add_argument('--lag', '-l', default=365,
                       help="Number of days between target date and first date considered as neighbor")
    parser.add_argument('--num_neighbors', '-n', default=20,
                       help="Number of date neighbors to consider")
    parser.add_argument('--target_dates', '-t', default='std_contest')
    parser.add_argument('--metric', default='cos',
                       help="Similarity metric ('rmse' or 'cos')")
    parser.add_argument('--margin_in_days', '-m', default="None",
                       help="number of month-day combinations on either side of the target combination "
                            "to include when training; set to 0 include only target month-day combo; "
                            "set to None to include entire year")
    
    args = parser.parse_args()
    
    # Assign variables
    gt_id = args.pos_vars[0]                                                                         
    horizon = args.pos_vars[1]
    history = int(args.history)
    lag = int(args.lag)
    num_neighbors = int(args.num_neighbors)
    target_dates = args.target_dates
    metric = args.metric
    if args.margin_in_days == "None":
        margin_in_days = None
    else:
        margin_in_days = int(args.margin_in_days)
else:
    # Otherwise, specify arguments interactively 
    gt_id = "contest_tmp2m"
    horizon = "34w"
    history = 60
    lag = 365
    num_neighbors = 20 if gt_id.endswith("tmp2m") else 1
    target_dates = 'std_contest'
    metric = 'rmse'
    margin_in_days = None if gt_id.endswith("tmp2m") else 56
    
#
# Process arguments
#

# Create list of target dates corresponding to submission dates in YYYYMMDD format
target_date_objs = pd.Series(get_target_dates(date_str=target_dates, horizon=horizon))
# Sort target_date_objs by day of week
target_date_objs = target_date_objs[target_date_objs.dt.weekday.argsort(kind='stable')]

# Choose the name of this model
model_name = "autoknn"

# Name of cache directory for storing non-submission-date specific
# intermediate files
cache_dir = os.path.join('models', model_name, 'cache')
# if cache_dir doesn't exist, create it
if not os.path.isdir(cache_dir):
    os.makedirs(cache_dir)

submodel_name = get_submodel_name(model_name, 
    history = history, lag = lag, num_neighbors = num_neighbors, 
    margin_in_days = margin_in_days, metric = metric)
printf(f"Submodel name {submodel_name}")

if not isnotebook():
    logger = start_logger(model=model_name,submodel=submodel_name,gt_id=gt_id,
                          horizon=horizon,target_dates=target_dates)
    # Store parameter values in log
    params_names = ['gt_id', 'horizon', 'target_dates',
                    'metric', 'history', 'lag',
                    'num_neighbors', 'margin_in_days']
    
    params_values = [eval(param) for param in params_names]
    log_params(params_names, params_values)


# # Compute KNN
# 
# Computes similarities between each pair of dates based on how skillfully the history of one date predicts the history of the other.

# In[ ]:


# Ensure the predictions of the top neighbors for each target date have been saved
preds_file = get_knn_preds_file_name(gt_id, horizon, history, 
                                     lag, num_neighbors, metric, model_name)
regen_preds = True
regen_viable = True
if not regen_preds and os.path.isfile(preds_file):
    printf("KNN Predictions already saved in "+preds_file)
else:
    # Load the similarities between each target date and each candidate date
    viable_similarities_file = get_viable_similarities_file_name(
        gt_id, horizon, history, lag, metric, model_name)
    if not regen_viable and os.path.isfile(viable_similarities_file):
        printf("Reading viable similarities from "+viable_similarities_file)
        tic();viable_similarities = pd.read_hdf(viable_similarities_file);toc()
    else:
        # Compute similarities
        gt_sim = compute_gt_similarity(gt_id, metric=metric, hindcast_year=None)
        viable_similarities = compute_viable_similarities(gt_sim, gt_id, horizon, 
                                                          history, lag, metric, model_name)
    # Compute and save KNN predictions
    get_knn_preds(viable_similarities, gt_id, horizon, 
                  history, lag, num_neighbors, metric, model_name)
    del(viable_similarities)


# # Regression

# In[ ]:


#
# Choose regression parameters
#
measurement_variable = get_measurement_variable(gt_id) # 'tmp2m' or 'precip'

# column names for gt_col, clim_col and anom_col
gt_col = measurement_variable
clim_col = measurement_variable+"_clim"
anom_col = measurement_variable+"_anom" # 'tmp2m_anom' or 'precip_anom'
# anom_inv_std_col: column name of inverse standard deviation of anomalies for each start_date
anom_inv_std_col = anom_col+"_inv_std"
# Name of knn columns
knn_cols = ["knn"+str(ii) for ii in range(1,num_neighbors+1)]

# Construct fixed lag anomaly variable names
lags = (['43', '86'] if horizon == '56w' else ['29', '58']) + ['365']
if metric == 'cos':
    # Column to subtract away from target prior to prediction
    base_col = clim_col
    lag_cols = [measurement_variable+'_shift'+lag+'_anom' for lag in lags]
    # Columns to group by when fitting regressions (a separate regression
    # is fit for each group); use ['ones'] to fit a single regression to all points
    group_by_cols = ['lat', 'lon']
    # anom_scale_col: multiply anom_col by this amount prior to prediction
    # (e.g., 'ones' or anom_inv_std_col)
    anom_scale_col = anom_inv_std_col
    # pred_anom_scale_col: multiply predicted anomalies by this amount
    # (e.g., 'ones' or anom_inv_std_col)
    pred_anom_scale_col = anom_scale_col
    # Regressors
    x_cols = knn_cols + lag_cols + ['ones']
else: # RMSE metric
    base_col = 'zeros'
    lag_cols = [measurement_variable+'_shift'+lag for lag in lags]
    group_by_cols = ['lat', 'lon']
    anom_scale_col = 'ones'
    pred_anom_scale_col = anom_scale_col
    # Regressors
    x_cols = knn_cols + lag_cols + ['ones',clim_col]

# Columns to load from lat_lon_date dataframe
load_cols = lag_cols+[gt_col,anom_inv_std_col,'start_date','lat','lon']


# In[ ]:


#
# Load data
#

# Load fixed lags and target variables
printf("\nLoading fixed lags and target data")
data = data_loaders.load_combined_data("lat_lon_date_data", gt_id, horizon, columns=load_cols)

printf(f"\nDropping rows with missing {lag_cols} values")
tic()
data.dropna(subset = lag_cols, inplace = True)
toc()

if clim_col in x_cols+[base_col]:
    printf("Merging in climatology")
    tic()
    data = clim_merge(data, data_loaders.get_climatology(gt_id))
    toc()

# Load AutoKNN data
printf(f"\nLoading autoknn data from {preds_file}")
tic()
knn_data = pd.read_hdf(preds_file)
if metric == "cos":
    knn_data[knn_cols] /= knn_data.groupby(["start_date"])[knn_cols].transform('std')
toc()

# Restrict data to relevant columns
printf("\nMerging datasets")
tic()
data = pd.merge(data, knn_data, on=["start_date","lat","lon"], how="left")
del(knn_data)
toc()

# Add supplementary columns
printf("\nAdding supplementary columns")
tic()
data['ones'] = 1.0
data['zeros'] = 0.0
# Square root of weight assigned to each training datapoint
data['sqrt_sample_weight'] = data[pred_anom_scale_col]
# If undefined, assign value of 1
data.loc[data['sqrt_sample_weight'].isna(),'sqrt_sample_weight'] = 1
toc()

# Drop rows with missing values for any relevant column
printf("\nDropping irrelevant columns and rows with missing values")
relevant_cols = list(set(x_cols+[base_col,'sqrt_sample_weight',gt_col,
                    'start_date','lat','lon']+group_by_cols))
tic()
data = data[relevant_cols].dropna(subset = x_cols+[base_col])
toc()

# Scale regressors and target by square root sample weight
printf("\nScaling regressors and target by square root sample weight")
tic()
data.loc[:,x_cols+[base_col,gt_col]] = data.loc[:,x_cols+[base_col,gt_col]].mul(
       data['sqrt_sample_weight'], axis = 'index')
toc()


# In[ ]:


# Specify regression model
# Exclude intercept as explicit 'ones' column is included
fit_intercept = False
model = linear_model.LinearRegression(fit_intercept=fit_intercept)

# Form predictions for each grid point (in parallel) using train / test split
# and the selected model
prediction_func = partial(fit_and_predict, model=model)
num_cores = cpu_count()

# Store rmses
rmses = pd.Series(index=target_date_objs, dtype='float64')


# In[ ]:


# Number of days between target date and last viable training date
start_delta = timedelta(get_start_delta(horizon, gt_id))
for target_date_obj in target_date_objs:
    if not any(data.start_date.isin([target_date_obj])):
        printf(f"warning: some features unavailable for target={target_date_obj}; skipping")
        continue    
        
    target_date_str = datetime.strftime(target_date_obj, '%Y%m%d')

    # Skip if forecast already produced for this target
    forecast_file = get_forecast_filename(
        model=model_name, submodel=submodel_name, 
        gt_id=gt_id, horizon=horizon, 
        target_date_str=target_date_str)
    print(forecast_file)

    if os.path.isfile(forecast_file):
        printf(f"prior forecast exists for target={target_date_obj}; loading")
        tic()
        preds = pd.read_hdf(forecast_file)
        
        # Add ground truth for later evaluation
        preds = pd.merge(preds, data.loc[data.start_date==target_date_obj,['lat','lon',gt_col]], 
                         on=['lat','lon'])
        
        preds.rename(columns={gt_col:'truth'}, inplace=True)
        toc()
    else:
        printf(f'target={target_date_str}')
        print("CREATING FORECAST PREDICTIONS")
        # Subset data based on margin
        if margin_in_days is not None:
            printf(f"subsetting based on margin {margin_in_days}")
            tic()
            sub_data = month_day_subset(data, target_date_obj, margin_in_days)
            toc()
        else:
            sub_data = data
            
        # Find the last observable training date for this target
        last_train_date = target_date_obj - start_delta
            
        tic()
        preds = apply_parallel(
            sub_data.groupby(group_by_cols),
            prediction_func, 
            num_cores=num_cores,
            gt_col=gt_col,
            x_cols=x_cols, 
            base_col=base_col, 
            last_train_date=last_train_date,
            test_dates=[target_date_obj],
            return_cols=['sqrt_sample_weight'])
        
        # Undo test point sample weighting
        preds = preds.loc[:,['pred','truth']].div(
            preds['sqrt_sample_weight'], axis='index')
        
        # Ensure raw precipitation predictions are never less than zero
        if gt_id.endswith("precip"):
            tic()
            preds['pred'] = np.maximum(preds['pred'],0)
            toc()
            
        preds = preds.reset_index()
        
        # Save prediction to file in standard format
        save_forecasts(preds.drop(columns=['truth']),
            model=model_name, submodel=submodel_name, 
            gt_id=gt_id, horizon=horizon, 
            target_date_str=target_date_str)
        
        print("")
        toc()
        
    # Evaluate and store error
    rmse = np.sqrt(np.square(preds.pred - preds.truth).mean())
    rmses.loc[target_date_obj] = rmse
    print("-rmse: {}, score: {}".format(rmse, mean_rmse_to_score(rmse)))
    mean_rmse = rmses.mean()
    print("-mean rmse: {}, running score: {}".format(mean_rmse, mean_rmse_to_score(mean_rmse)))

# Save rmses in standard format
rmses = rmses.reset_index()
rmses.columns = ['start_date','rmse']
save_metric(rmses, model=model_name, submodel=submodel_name, gt_id=gt_id, horizon=horizon, target_dates=target_dates, metric="rmse")


# In[ ]:





# In[ ]:




