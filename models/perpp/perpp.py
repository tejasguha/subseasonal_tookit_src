#!/usr/bin/env python
# coding: utf-8

# # Persistence++
# 
# Locally linear combination of dynamical model forecasts, lagged measurements, and climatology.

# In[ ]:


import os, sys
from subseasonal_toolkit.utils.notebook_util import isnotebook
if isnotebook():
    # Autoreload packages that are modified
    get_ipython().run_line_magic('load_ext', 'autoreload')
    get_ipython().run_line_magic('autoreload', '2')
    #%cd "~/forecast_rodeo_ii"
else:
    from argparse import ArgumentParser

# Imports 
import numpy as np
import pandas as pd
from sklearn import *
import sys
import json
import subprocess
from datetime import datetime, timedelta
from functools import partial
from multiprocessing import cpu_count
from subseasonal_data.utils import get_measurement_variable, df_merge, shift_df
from subseasonal_toolkit.utils.general_util import printf, tic, toc
from subseasonal_toolkit.utils.experiments_util import (get_first_year, 
                                                        get_start_delta, clim_merge)
from subseasonal_toolkit.utils.eval_util import get_target_dates, mean_rmse_to_score, save_metric
from subseasonal_toolkit.utils.fit_and_predict import apply_parallel
from subseasonal_toolkit.utils.models_util import (get_submodel_name, start_logger, log_params, 
                                                   get_forecast_filename, save_forecasts)
from subseasonal_toolkit.models.perpp.perpp_util import fit_and_predict, years_ago
from subseasonal_data import data_loaders

import re


# In[ ]:


#
# Specify model parameters
#
model_name = "perpp"
if not isnotebook():
    # If notebook run as a script, parse command-line arguments
    parser = ArgumentParser()
    parser.add_argument("pos_vars",nargs="*")  # gt_id and horizon                                                                                  
    parser.add_argument('--target_dates', '-t', default="std_contest")
    parser.add_argument('--train_years', '-y', default="all",
                        help="Number of years to use in training (\"all\" or integer)")
    parser.add_argument('--margin_in_days', '-m', default="None",
                        help="number of month-day combinations on either side of the target combination "
                            "to include when training; set to 0 include only target month-day combo; "
                            "set to None to include entire year")
    parser.add_argument('--forecast', '-f', default="cfsv2", 
                        help="include the forecasts of this dynamical model as features")
    args, opt = parser.parse_known_args()
    
    # Assign variables                                                                                                                                     
    gt_id = args.pos_vars[0] # "contest_precip" or "contest_tmp2m"                                                                            
    horizon = args.pos_vars[1] # "34w" or "56w"                                                                                        
    target_dates = args.target_dates
    train_years = args.train_years
    if train_years != "all":
        train_years = int(train_years)
    if args.margin_in_days == "None":
        margin_in_days = None
    else:
        margin_in_days = int(args.margin_in_days)
    forecast = args.forecast
    
else:
    # Otherwise, specify arguments interactively 
    gt_id = "us_tmp2m_1.5x1.5"
    horizon = "34w"
    target_dates = "std_paper_forecast"
    train_years = "all"
    margin_in_days = None
    forecast = "cfsv2"

#
# Process model parameters
#

# Get list of target date objects
target_date_objs = pd.Series(get_target_dates(date_str=target_dates,horizon=horizon))

# Sort target_date_objs by day of week
target_date_objs = target_date_objs[target_date_objs.dt.weekday.argsort(kind='stable')]

# Identify measurement variable name
measurement_variable = get_measurement_variable(gt_id) # 'tmp2m' or 'precip'

# Column names for gt_col, clim_col and anom_col 
gt_col = measurement_variable
clim_col = measurement_variable+"_clim"
anom_col = get_measurement_variable(gt_id)+"_anom" # 'tmp2m_anom' or 'precip_anom'

# For a given target date, the last observable training date is target date - gt_delta
# as gt_delta is the gap between the start of the target date and the start of the
# last ground truth period that's fully observable at the time of forecast issuance
gt_delta = timedelta(days=get_start_delta(horizon, gt_id))


# In[ ]:


#
# Choose regression parameters
#
# Record standard settings of these parameters
base_col = "zeros"    
if (gt_id.endswith("tmp2m")) and (horizon == "12w"):
    forecast_col = f'subx_{forecast}_tmp2m'
    x_cols = [
    'tmp2m_shift15',
    'tmp2m_shift30',
    forecast_col,
    clim_col
    ] 
elif (gt_id.endswith("precip")) and (horizon == "12w"):
    forecast_col = f'subx_{forecast}_precip'
    x_cols = [
    'precip_shift15',
    'precip_shift30',
    forecast_col,
    clim_col
    ] 
elif (gt_id.endswith("tmp2m")) and (horizon == "34w"):
    forecast_col = f'subx_{forecast}_tmp2m'
    x_cols = [
    'tmp2m_shift29',
    'tmp2m_shift58',
    forecast_col,
    clim_col
    ] 
elif (gt_id.endswith("precip")) and (horizon == "34w"):
    forecast_col = f'subx_{forecast}_precip'
    x_cols = [
    'precip_shift29',
    'precip_shift58',
    forecast_col,
    clim_col
    ] 
elif (gt_id.endswith("tmp2m")) and (horizon == "56w"):
    forecast_col = f'subx_{forecast}_tmp2m'
    x_cols = [
    'tmp2m_shift43',
    'tmp2m_shift86',
    forecast_col,
    clim_col
    ] 
elif (gt_id.endswith("precip")) and (horizon == "56w"):
    forecast_col = f'subx_{forecast}_precip'
    x_cols = [
    'precip_shift43',
    'precip_shift86',
    forecast_col,
    clim_col
    ]
elif (gt_id.endswith("tmp2m_1.5x1.5")) and (horizon == "12w"):
    forecast_col = f'iri_{forecast}_tmp2m'
    x_cols = [
    'tmp2m_shift15',
    'tmp2m_shift30',
    forecast_col,
    clim_col
    ] 
elif (gt_id.endswith("precip_1.5x1.5")) and (horizon == "12w"):
    forecast_col = f'iri_{forecast}_precip'
    x_cols = [
    'precip_shift15',
    'precip_shift30',
    forecast_col,
    clim_col
    ] 
elif (gt_id.endswith("tmp2m_1.5x1.5")) and (horizon == "34w"):
    forecast_col = f'iri_{forecast}_tmp2m'
    x_cols = [
    'tmp2m_shift29',
    'tmp2m_shift58',
    forecast_col,
    clim_col
    ] 
elif (gt_id.endswith("precip_1.5x1.5")) and (horizon == "34w"):
    forecast_col = f'iri_{forecast}_precip'
    x_cols = [
    'precip_shift29',
    'precip_shift58',
    forecast_col,
    clim_col
    ] 
elif (gt_id.endswith("tmp2m_1.5x1.5")) and (horizon == "56w"):
    forecast_col = f'iri_{forecast}_tmp2m'
    x_cols = [
    'tmp2m_shift43',
    'tmp2m_shift86',
    forecast_col,
    clim_col
    ] 
elif (gt_id.endswith("precip_1.5x1.5")) and (horizon == "56w"):
    forecast_col = f'iri_{forecast}_precip'
    x_cols = [
    'precip_shift43',
    'precip_shift86',
    forecast_col,
    clim_col
    ]
group_by_cols = ['lat', 'lon']

# Record submodel names for perpp model
submodel_name = get_submodel_name(
    model_name, train_years=train_years, margin_in_days=margin_in_days,
    forecast=forecast)

printf(f"Submodel name {submodel_name}")

if not isnotebook():
    # Save output to log file
    logger = start_logger(model=model_name,submodel=submodel_name,gt_id=gt_id,
                          horizon=horizon,target_dates=target_dates)
    # Store parameter values in log                                                                                                                        
    params_names = ['gt_id', 'horizon', 'target_dates',
                    'train_years', 'margin_in_days',
                    'base_col', 'x_cols', 'group_by_cols', 'forecast'
                   ]
    params_values = [eval(param) for param in params_names]
    log_params(params_names, params_values)


# In[ ]:


#
# Load ground truth data
#
printf("Loading ground truth data")
tic()
gt = data_loaders.get_ground_truth(gt_id)[['lat','lon','start_date',gt_col]]
toc()

#
# Added shifted ground truth features
#
printf("Adding shifted ground truth features")
lld_data = gt
shifts = [int(re.search(r'\d+$', col).group()) for col in x_cols if col.startswith(gt_col+"_shift")]
tic()
for shift in shifts:
    gt_shift = shift_df(gt, shift)
    lld_data = df_merge(lld_data, gt_shift, how="right")
toc()

#
# Drop rows with empty pred_cols
#
pred_cols = x_cols+[base_col]
exclude_cols = set([clim_col, forecast_col, 'zeros']) 
lld_data = lld_data.dropna(subset=set(pred_cols) - exclude_cols)

# Add climatology
if clim_col in pred_cols:
    printf("Merging in climatology")
    tic()
    lld_data = clim_merge(lld_data, data_loaders.get_climatology(gt_id))
    toc()

# Add zeros
if 'zeros' in pred_cols:
    lld_data['zeros'] = 0


# In[ ]:


#
# Add ensemble forecast as feature
#
printf(f"Forming {forecast} ensemble forecast...")
shift = 15 if horizon == "34w" else 29 if horizon == "56w" else 1
first_lead = 14 if horizon == "34w" else 28 if horizon == "56w" else 0
suffix = "-us" if gt_id.startswith("us_") else ""

if gt_id.endswith("1.5x1.5"):
    prefix = f"iri_{forecast}"
    suffix += "1_5"
else:
    prefix = f"subx_{forecast}"

if forecast == "subx_mean":
    forecast_id = prefix+"-"+gt_id.split("_")[1]+"_"+horizon+suffix
else:
    forecast_id = prefix+"-"+gt_id.split("_")[1]+suffix

tic(); data = data_loaders.get_forecast(forecast_id=forecast_id, shift=shift); toc()

# Select last lead to include in ensemble
if horizon == "12w":
    last_lead = first_lead + 1
else:
    # Find the largest available lead
    max_lead = 0
    for col in data.columns:
        try:
            max_lead = max(max_lead, 
                           int(re.search(f'{prefix}_{gt_col}-(.*).5d_shift{shift}', 
                                         col).group(1)))
        except AttributeError:
            continue
    last_lead = min(max_lead, 29)
printf(f"Aggregating lead {first_lead} with shift {shift}")
tic()
feature = data[['lat','lon','start_date',f'{prefix}_{gt_col}-{first_lead}.5d_shift{shift}']].set_index(
    ['lat','lon','start_date']).squeeze().unstack(['lat','lon']).copy()
initial_feature_index = feature.index
# Count the number of non-nan leads values contributing to each feature entry
num_leads = pd.DataFrame(data=1., index=feature.index, columns=feature.columns, dtype=float)
toc()
for lead in range(first_lead+1,last_lead+1):
    printf(f"Aggregating lead {lead} with shift {shift+lead-first_lead}")
    tic()
    new_feat = data[['lat','lon','start_date',f'{prefix}_{gt_col}-{lead}.5d_shift{shift}']].set_index(
        ['lat','lon','start_date']).squeeze().unstack(['lat','lon']).shift(lead-first_lead,freq="D")
    # Add contribution of new feature if it exists and 0 otherwise
    feature = feature.add(new_feat, fill_value=0.)
    # Count contribution of new feature if it exists
    num_leads = num_leads.add(new_feat.notna().astype(float),  fill_value=0.)
    toc()
del data
# Normalize feature sums by the number of contributing leads
feature /= num_leads
del num_leads
# Restrict to initial feature index dates
feature = feature.loc[initial_feature_index]

# Drop dates with no forecasts and reshape
feature = feature.dropna().unstack().rename(forecast_col)
# Merge feature forecast with lld_data
printf(f"Merging {forecast_col} with lld_data")
tic()
lld_data = pd.merge(lld_data, feature, left_on=['lat','lon','start_date'], 
                    right_index=True)
toc()
del feature


# In[ ]:


# specify regression model
fit_intercept = True
model = linear_model.LinearRegression(fit_intercept=fit_intercept)

# Form predictions for each grid point (in parallel) using train / test split
# and the selected model
prediction_func = partial(fit_and_predict, model=model)
num_cores = cpu_count()

# Store rmses
rmses = pd.Series(index=target_date_objs, dtype='float64')

# Restrict data to relevant columns and rows for which predictions can be made
relevant_cols = set(
    ['start_date','lat','lon',gt_col,base_col]+x_cols).intersection(lld_data.columns)
lld_data = lld_data[relevant_cols].dropna(subset=x_cols+[base_col])


# In[ ]:


for target_date_obj in target_date_objs:
    if not any(lld_data.start_date.isin([target_date_obj])):
        printf(f"warning: some features unavailable for target={target_date_obj}; skipping")
        continue    

    target_date_str = datetime.strftime(target_date_obj, '%Y%m%d')

    # Skip if forecast already produced for this target
    forecast_file = get_forecast_filename(
        model=model_name, submodel=submodel_name, 
        gt_id=gt_id, horizon=horizon, 
        target_date_str=target_date_str)

    if True and os.path.isfile(forecast_file):
        printf(f"prior forecast exists for target={target_date_obj}; loading")
        tic()
        preds = pd.read_hdf(forecast_file)

        # Add ground truth for later evaluation
        preds = pd.merge(preds, lld_data.loc[lld_data.start_date==target_date_obj,['lat','lon',gt_col]], 
                         on=['lat','lon'])

        preds.rename(columns={gt_col:'truth'}, inplace=True)
        toc()
    else:
        printf(f'target={target_date_str}')

        # Subset data based on margin
        if margin_in_days is not None:
            tic()
            sub_data = month_day_subset(lld_data, target_date_obj, margin_in_days)
            toc()
        else:
            sub_data = lld_data

        # Find the last observable training date for this target
        last_train_date = target_date_obj - gt_delta 

        # Only train on train_years worth of data
        if train_years != "all":
            tic()
            sub_data = sub_data.loc[sub_data.start_date >= years_ago(last_train_date, train_years)]
            toc()

        tic()
        preds = apply_parallel(
            sub_data.groupby(group_by_cols),
            prediction_func, 
            num_cores=num_cores,
            gt_col=gt_col,
            x_cols=x_cols, 
            base_col=base_col, 
            last_train_date=last_train_date,
            test_dates=[target_date_obj])  

        # Ensure raw precipitation predictions are never less than zero
        if gt_id.endswith("precip"):
            tic()
            preds['pred'] = np.maximum(preds['pred'],0)
            toc()

        preds = preds.reset_index()

        if True:
            # Save prediction to file in standard format
            save_forecasts(preds.drop(columns=['truth']),
                model=model_name, submodel=submodel_name, 
                gt_id=gt_id, horizon=horizon, 
                target_date_str=target_date_str)
        toc()

    # Evaluate and store error
    rmse = np.sqrt(np.square(preds.pred - preds.truth).mean())
    rmses.loc[target_date_obj] = rmse
    printf("-rmse: {}, score: {}".format(rmse, mean_rmse_to_score(rmse)))
    mean_rmse = rmses.mean()
    printf("-mean rmse: {}, running score: {}".format(mean_rmse, mean_rmse_to_score(mean_rmse)))

if True:
    # Save rmses in standard format
    rmses = rmses.reset_index()
    rmses.columns = ['start_date','rmse']
    save_metric(rmses, model=model_name, submodel=submodel_name, gt_id=gt_id, horizon=horizon, target_dates=target_dates, metric="rmse")
    save_metric(rmses, model=f'perpp_{forecast}', submodel=submodel_name, gt_id=gt_id, horizon=horizon, target_dates=target_dates, metric="rmse")


# In[ ]:




