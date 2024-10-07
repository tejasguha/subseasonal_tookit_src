#!/usr/bin/env python
# coding: utf-8

# # SubX++
# 
# Learned correction for SubX ensemble forecasts

# In[ ]:


import os, sys
from subseasonal_toolkit.utils.notebook_util import isnotebook
if isnotebook():
    # Autoreload packages that are modified
    get_ipython().run_line_magic('load_ext', 'autoreload')
    get_ipython().run_line_magic('autoreload', '2')
    #%cd "~/forecast_rodeo_ii"
    #%pwd
else:
    from argparse import ArgumentParser
import pandas as pd
import numpy as np
from scipy.spatial.distance import cdist, euclidean
from datetime import datetime, timedelta
from filelock import FileLock
from subseasonal_data.utils import get_measurement_variable
from subseasonal_toolkit.utils.general_util import printf, tic, toc
from subseasonal_toolkit.utils.experiments_util import (get_first_year, get_start_delta,
                                                        get_forecast_delta)
from subseasonal_toolkit.utils.models_util import (get_submodel_name, start_logger, log_params, get_forecast_filename,
                                                   save_forecasts, get_selected_submodel_name)
from subseasonal_toolkit.utils.eval_util import get_target_dates, mean_rmse_to_score, save_metric
from sklearn.linear_model import *

from subseasonal_data import data_loaders


# In[ ]:


#
# Specify model parameters
#
model_name = "subxpp"
if not isnotebook():
    # If notebook run as a script, parse command-line arguments
    parser = ArgumentParser()
    parser.add_argument("pos_vars",nargs="*")  # gt_id and horizon                                                                                  
    parser.add_argument('--target_dates', '-t', default="std_test")
    # Fit intercept parameter if and only if this flag is specified
    parser.add_argument('--forecast', '-f', default="cfsv2",
                        help="include the forecasts of this dynamical model as features")
    parser.add_argument('--fit_intercept', '-i', default="False",
                        choices=['True', 'False'],
                        help="Fit intercept parameter if \"True\"; do not if \"False\"")
    parser.add_argument('--train_years', '-y', default="all",
                       help="Number of years to use in training (\"all\" or integer)")
    parser.add_argument('--margin_in_days', '-m', default="None",
                       help="number of month-day combinations on either side of the target combination "
                            "to include when training; set to 0 include only target month-day combo; "
                            "set to None to include entire year")
    parser.add_argument('--first_day', '-fd', default=1,
                        help="first available daily subx forecast (1 or greater) to average")
    parser.add_argument('--last_day', '-ld', default=1,
                        help="last available daily subx forecast (first_day or greater) to average")
    parser.add_argument('--loss', '-l', default="mse", 
                        help="loss function: mse, rmse, skill, or ssm")
    parser.add_argument('--first_lead', '-fl', default=0, 
                        help="first subx lead to average into forecast (0-29)")
    parser.add_argument('--last_lead', '-ll', default=29, 
                        help="last subx lead to average into forecast (0-29)")
    parser.add_argument('--mei', default=False, action='store_true', help="Whether to condition on MEI")
    parser.add_argument('--mjo', default=False, action='store_true', help="Whether to condition on MJO")
    args, opt = parser.parse_known_args()
    
    # Assign variables                                                                                                                                     
    gt_id = args.pos_vars[0] # "contest_precip" or "contest_tmp2m"                                                                            
    horizon = args.pos_vars[1] # "12w", "34w", or "56w"                                                                                        
    target_dates = args.target_dates
    forecast = args.forecast
    fit_intercept = args.fit_intercept
    mei = args.mei
    mjo = args.mjo
    if fit_intercept == "False":
        fit_intercept = False
    elif fit_intercept == "True":
        fit_intercept = True
    else:
        raise ValueError(f"unrecognized value {fit_intercept} for fit_intercept")
    train_years = args.train_years
    if train_years != "all":
        train_years = int(train_years)
    if args.margin_in_days == "None":
        margin_in_days = None
    else:
        margin_in_days = int(args.margin_in_days)
    first_day = int(args.first_day)
    last_day = int(args.last_day)
    loss = args.loss
    first_lead = int(args.first_lead)
    last_lead = int(args.last_lead)
else:
    # Otherwise, specify arguments interactively 
    gt_id = "us_precip_1.5x1.5"
    horizon = "34w"
    target_dates = "20200101"#"std_paper_forecast"
    forecast = "subx_mean"
    fit_intercept = True
    loss = "mse"
    train_years = 12
    margin_in_days = 28
    mei = False
    mjo = False
    if "tmp2m" in gt_id and (horizon == "12w"):
        first_day = 1
        last_day = 1
        first_lead = 0
        last_lead = 0
    elif "precip" in gt_id and (horizon == "12w"):
        first_day = 1
        last_day = 1
        first_lead = 0
        last_lead = 0
    elif "tmp2m" in gt_id and (horizon == "34w"):
        first_day = 1
        last_day = 7
        first_lead = 15
        last_lead = 22
    elif "precip" in gt_id and (horizon == "34w"):
        first_day = 1
        last_day = 35
        first_lead = 0
        last_lead = 29
    elif "tmp2m" in gt_id and (horizon == "56w"):
        first_day = 1
        last_day = 14 
        first_lead = 29
        last_lead = 29
    elif "precip" in gt_id and (horizon == "56w"):
        first_day = 1
        last_day = 21
        first_lead = 0
        last_lead = 29
        
#
# Choose regression parameters
#
# Record standard settings of these parameters
x_cols = ['zeros']
if gt_id.endswith("1.5x1.5"):
    prefix = f"iri_{forecast}"
else:
    prefix = f"subx_{forecast}" 
if "tmp2m" in gt_id:
    base_col = prefix+'_tmp2m'
elif "precip" in gt_id:
    base_col = prefix+'_precip'
group_by_cols = ['lat', 'lon']    

#
# Process model parameters
#

# Get list of target date objects
target_date_objs = pd.Series(get_target_dates(date_str=target_dates, horizon=horizon))

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

LAST_SAVE_YEAR = get_first_year(prefix) # Don't save forecasts for years earlier than LAST_SAVE_YEAR

# Record model and submodel names
submodel_name = get_submodel_name(
    model_name, forecast=forecast, fit_intercept=fit_intercept,
    train_years=train_years, margin_in_days=margin_in_days,
    first_day=first_day, last_day=last_day, loss=loss, 
    first_lead=first_lead, last_lead=last_lead, mei=mei, mjo=mjo)

if not isnotebook():
    # Save output to log file
    logger = start_logger(model=model_name,submodel=submodel_name,gt_id=gt_id,
                          horizon=horizon,target_dates=target_dates)
    # Store parameter values in log                                                                                                                        
    params_names = ['gt_id', 'horizon', 'target_dates', 'forecast',
                    'fit_intercept', 'train_years', 'margin_in_days',
                    'first_day', 'last_day', 'loss', 
                    'first_lead', 'last_lead',
                    'base_col', 'x_cols', 'group_by_cols'
                   ]
    params_values = [eval(param) for param in params_names]
    log_params(params_names, params_values)
    
def geometric_median(X, eps=1e-5):
    """Computes the geometric median of the columns of X, up to a tolerance epsilon.
    The geometric median is the vector that minimizes the mean Euclidean norm to
    each column of X.
    """
    y = np.mean(X, 0)

    while True:
        D = cdist(X, [y])
        nonzeros = (D != 0)[:, 0]

        Dinv = 1 / D[nonzeros]
        Dinvs = np.sum(Dinv)
        W = Dinv / Dinvs
        T = np.sum(W * X[nonzeros], 0)

        num_zeros = len(X) - np.sum(nonzeros)
        if num_zeros == 0:
            y1 = T
        elif num_zeros == len(X):
            return y
        else:
            R = (T - y) * Dinvs
            r = np.linalg.norm(R)
            rinv = 0 if r == 0 else num_zeros/r
            y1 = max(0, 1-rinv)*T + min(1, rinv)*y

        if euclidean(y, y1) < eps:
            return y1

        y = y1

def ssm(X, alpha=1):
    """Computes stabilized sample mean (Orenstein, 2019) of each column of X
    
    Args:
        alpha: if infinity, recovers the mean; if 0 approximates median
    """
    # Compute first, second, and third uncentered moments
    mu = np.mean(X,0)
    mu2 = np.mean(np.square(X),0)
    mu3 = np.mean(np.power(X,3),0)
    # Return mean - (third central moment)/(3*(2+numrows(X))*variance)
    return mu - (mu3 - 3*mu*mu2+2*np.power(mu,3)).div(3*(2+alpha*X.shape[0])*(mu2 - np.square(mu)))
    
# Select estimator based on loss
if loss == "rmse":
    estimator = geometric_median 
elif loss == "ssm":
    estimator = ssm
else: 
    estimator = np.mean


# In[ ]:


# Load and process data
printf("Loading subx data and averaging leads")
# Choose data shift based on horizon and first day to be averaged
base_shift = get_forecast_delta(horizon) + first_day - 1
tic()
mask = None
if gt_id.startswith("us_"):
    suffix = "-us"  
else:
    suffix = ""
if gt_id.endswith("1.5x1.5"):
    suffix += "1_5"
else:
    mask = data_loaders.get_us_mask()

if forecast == "subx_mean":
    forecast_id = prefix+"-"+measurement_variable+"_"+horizon+suffix
else:
    forecast_id = prefix+"-"+measurement_variable+suffix

tic(); data = data_loaders.get_forecast(forecast_id=forecast_id, mask_df=mask, shift=base_shift); toc()

cols = [prefix+"_"+gt_id.split("_")[1]+"-{}.5d_shift{}".format(col,base_shift) 
        for col in range(first_lead, last_lead+1)]
data[base_col] = data[cols].mean(axis=1)
toc()

printf('Pivoting dataframe to have one row per start_date')
tic()
data = data[['lat','lon','start_date',base_col]].set_index(['lat','lon','start_date']).unstack(['lat','lon'])
toc()

printf(f"Computing rolling mean over days {first_day}-{last_day}")
days = last_day - first_day + 1
tic()
data = data.rolling(f"{days}d").mean().dropna(how='any')
toc()



# In[ ]:


# Load ground truth
tic()
# gt = get_ground_truth(gt_id).loc[:,['lat','lon','start_date',gt_col]]
gt = data_loaders.get_ground_truth(gt_id).loc[:,['lat','lon','start_date',gt_col]]
toc()
printf('Pivoting ground truth to have one row per start_date')
tic()
gt = gt.loc[gt.start_date.isin(data.index),['lat','lon','start_date',gt_col]].set_index(['lat','lon','start_date']).unstack(['lat','lon'])
toc()


# In[ ]:


printf("Merging ground truth")
tic()
data = data.join(gt, how="left") 
# del gt
toc()


# In[ ]:


printf('Extracting target variable (ground-truth - base prediction) and dropping NAs')
tic()
target = (data[gt_col] - data[base_col]).dropna(how='any')
toc()


# In[ ]:


# Conditioning
if mei or mjo:
    conditioning_data = data_loaders.load_combined_data('date_data', gt_id, horizon)
    conditioning_columns = get_conditioning_cols(gt_id, horizon, mei=mei, mjo=mjo)
    # Combined data start dates and gt start dates don't fully overlap
    conditioned_targets = pd.DataFrame(gt.index).merge(conditioning_data[["start_date"] + conditioning_columns], on="start_date", how="left")


# In[ ]:


# Make predictions for each target date
printf('Creating dataframes to store performance and date-based covariates')
tic()
rmses = pd.Series(index=target_date_objs, dtype=np.float64)
X = pd.DataFrame(index=target.index, columns = ["delta", "dividend", "remainder"], 
                 dtype=np.float64)
toc()
printf('Initializing target date predictions to base column')
tic()
# Only form predictions for target dates in data matrix
valid_targets = data.index.intersection(target_date_objs)
preds = data.loc[valid_targets, base_col]
preds.index.name = "start_date"
# Order valid targets by day of week
valid_targets = valid_targets[valid_targets.weekday.argsort(kind='stable')]
toc()
days_per_year = 365.242199


# In[ ]:


for target_date_obj in valid_targets:
    # Skip if forecast already produced for this target
    target_date_str = datetime.strftime(target_date_obj, '%Y%m%d')
    forecast_file = get_forecast_filename(
        model=model_name, submodel=submodel_name, 
        gt_id=gt_id, horizon=horizon, 
        target_date_str=target_date_str)
    if os.path.isfile(forecast_file):
        printf(f"prior forecast exists for target={target_date_obj}")
        pred = pd.read_hdf(forecast_file).set_index(['lat','lon']).pred - preds.loc[target_date_obj,:]
    else:
        tic()
        printf(f"Preparing covariates for {target_date_str}")
        # Compute days from target date
        X['delta'] = (target_date_obj - target.index).days
        # Extract the dividend and remainder when delta is divided by the number of days per year
        # The dividend is analogous to the year
        # (Negative values will ultimately be excluded)
        X['dividend'] = np.floor(X.delta / days_per_year)
        # The remainder is analogous to the day of the year
        X['remainder'] = np.floor(X.delta % days_per_year)
        # Find the last observable training date for this target
        last_train_date = target_date_obj - gt_delta
        # Restrict data based on training date, dividend, and remainder
        if mei or mjo:
            target_conditioning_val = conditioning_data[conditioning_data.start_date == target_date_obj][conditioning_columns].values[0]
            indic = cond_indices(conditioned_targets, conditioning_columns, target_conditioning_val)
            indic &= (X.index <= last_train_date)
        else:
            indic = (X.index <= last_train_date)
        if margin_in_days is not None:
            indic &= ((X.remainder <= margin_in_days) | (X.remainder >= 365-margin_in_days))
        if train_years != "all":
            indic = indic & (X.dividend < train_years)
        toc()
        printf(f'Fitting {model_name} model with loss {loss} for {target_date_obj}')
        tic()
        if fit_intercept and not indic.any():
            printf(f'-Warning: no training data for {target_date_str}; using base prediction')
            # Do not adjust base prediction
            pred = 0
        elif fit_intercept:
            # Add learned prediction to base prediction
            pred = estimator(target.loc[indic,:])
            preds.loc[target_date_obj,:] += pred
        else:
            # Do not adjust base prediction
            pred = 0
        # Save prediction to file in standard format
        if target_date_obj.year >= LAST_SAVE_YEAR:
            save_forecasts(
                preds.loc[[target_date_obj],:].unstack().rename("pred").reset_index(),
                model=model_name, submodel=submodel_name, 
                gt_id=gt_id, horizon=horizon, 
                target_date_str=target_date_str)
        toc()
    # Evaluate and store error if we have ground truth data
    tic()
    if target_date_obj in target.index:
        rmse = np.sqrt(np.square(pred - target.loc[target_date_obj,:]).mean())
        rmses.loc[target_date_obj] = rmse
        print("-rmse: {}, score: {}".format(rmse, mean_rmse_to_score(rmse)))
        mean_rmse = rmses.mean()
        print("-mean rmse: {}, running score: {}".format(mean_rmse, mean_rmse_to_score(mean_rmse)))
    toc()
printf("Save rmses in standard format")
rmses = rmses.sort_index().reset_index()
rmses.columns = ['start_date','rmse']
save_metric(rmses, model=model_name, submodel=submodel_name, gt_id=gt_id, horizon=horizon, target_dates=target_dates, metric="rmse")
save_metric(rmses, model=f'{forecast}pp', submodel=submodel_name, gt_id=gt_id, horizon=horizon, target_dates=target_dates, metric="rmse")



# In[ ]:




