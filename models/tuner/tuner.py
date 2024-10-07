#!/usr/bin/env python
# coding: utf-8

# # Model tuner
# 
# ### Selects a best sub-model of a model for each target date

# In[ ]:


# Ensure notebook is being run from base repository directory
import os, sys
from subseasonal_toolkit.utils.notebook_util import isnotebook
if isnotebook():
    # Autoreload packages that are modified
    get_ipython().run_line_magic('load_ext', 'autoreload')
    get_ipython().run_line_magic('autoreload', '2')
else:
    from argparse import ArgumentParser
import pandas as pd
import numpy as np
import shutil
from datetime import datetime, timedelta
from filelock import FileLock
from pkg_resources import resource_filename
from ttictoc import tic, toc
from subseasonal_data.utils import get_measurement_variable
from subseasonal_toolkit.utils.general_util import printf, make_directories, symlink
from subseasonal_toolkit.utils.experiments_util import get_first_year, get_start_delta
from subseasonal_toolkit.utils.models_util import (get_submodel_name, start_logger, log_params, get_forecast_filename,
                                                   save_forecasts)
from subseasonal_toolkit.utils.eval_util import get_target_dates, mean_rmse_to_score
from subseasonal_toolkit.models.tuner.util import *
pd.set_option('display.max_rows', None)

from subseasonal_data import data_loaders


# In[ ]:


#
# Specify model parameters
#
if not isnotebook():
    # If notebook run as a script, parse command-line arguments
    parser = ArgumentParser()
    parser.add_argument("pos_vars",nargs="*")  # gt_id and horizon 
    parser.add_argument('--model_name', '-mn', default="climpp")                                                                                 
    parser.add_argument('--target_dates', '-t', default="std_test")
    parser.add_argument('--num_years', '-y', default="all",
                       help="Number of years to use in training (all or integer)")
    parser.add_argument('--margin_in_days', '-m', default="None", 
                       help="Number of month-day combinations on either side of the target combination to include; "
                            "set to 0 to include only target month-day combo; set to None to include entire year; "
                            "None by default")
    args = parser.parse_args()
    
    # Assign variables                                                                                                                                     
    gt_id = args.pos_vars[0] # e.g., "contest_precip" or "contest_tmp2m"                                                                            
    horizon = args.pos_vars[1] # e.g., "12w", "34w", or "56w"    
    model_name = args.model_name
    target_dates = args.target_dates
    num_years = args.num_years
    if num_years != "all":
        num_years = int(num_years)
    margin_in_days = args.margin_in_days
    if margin_in_days == "None":
        margin_in_days = None
    else:
        margin_in_days = int(args.margin_in_days)
else:
    # Otherwise, specify arguments interactively
    gt_id = "global_tmp2m_p1_1.5x1.5" 
    horizon = "34w"
    model_name = "climpp" 
    target_dates = "s2s"
    num_years = 3
    margin_in_days = None


#
# Process model parameters
#
# Record output model name and submodel name
output_model_name = f"tuned_{model_name}"
task = f"{gt_id}_{horizon}"

submodel_name = get_tuner_submodel_name(
    output_model_name=output_model_name, num_years=num_years, 
    margin_in_days=margin_in_days)


# Prepare a directory to store tuned model attributes and configuration files
src_dir = resource_filename(__name__, os.path.join("."))
dst_dir = resource_filename(__name__, os.path.join("..","..","models", output_model_name))
if not os.path.exists(dst_dir):
    tic()
    printf(f'\nCreating {dst_dir}')
    make_directories(dst_dir)
    # copy attributes and selected submodel files to output model folder
    shutil.copy(os.path.join(src_dir, "attributes.py"), os.path.join(dst_dir, "attributes.py"))
    shutil.copy(os.path.join(src_dir, "selected_submodel.json"), os.path.join(dst_dir, "selected_submodel.json"))
    # update MODEL_NAME in the attribute file
    filename = os.path.join(dst_dir, "attributes.py")
    with open(filename, "r") as f:
        newText=f.read().replace("tuner", output_model_name)
    with open(filename, "w") as f:
        f.write(newText)
    toc()

# Create directory for storing forecasts if one does not already exist
out_dir = os.path.join("models", output_model_name, "submodel_forecasts", 
                       submodel_name, f"{gt_id}_{horizon}")
if not os.path.exists(out_dir):
    make_directories(out_dir)
    

if not isnotebook():
    # Save output to log file
    logger = start_logger(model=output_model_name,submodel=submodel_name,gt_id=gt_id,
                          horizon=horizon,target_dates=target_dates)
    # Store parameter values in log
    params_names = ['gt_id', 'horizon', 'model_name', 
                    'target_dates', 'num_years', 'margin_in_days']
    params_values = [eval(param) for param in params_names]
    log_params(params_names, params_values)

# Select target dates and restrict to dates with available ground truth data
target_date_objs = get_target_dates(target_dates, horizon=horizon)
#print(target_date_objs)


# In[ ]:


printf(f'\nLoading metrics of {model_name} submodels')
if gt_id in ["global_precip_p1_1.5x1.5", "global_precip_p3_1.5x1.5",
             "global_tmp2m_p1_1.5x1.5", "global_tmp2m_p3_1.5x1.5"]:
    # Use s2s_eval dates for S2S gt_id's
    tic()
    metric_df = load_metric_df(gt_id=gt_id, target_horizon=horizon, model_name=model_name, metric="wtd_mse",
                               metric_file_regex="*s2s_eval*", first_year=2017)
    toc()    
else:
    tic()
    metric_df = load_metric_df(gt_id=gt_id, target_horizon=horizon, model_name=model_name)
    toc()


# In[ ]:


printf(f"\n Loading ground truth")
tic()
var = get_measurement_variable(gt_id)
gt = data_loaders.get_ground_truth(gt_id).loc[:,["start_date","lat","lon",var]]
gt = gt.set_index(['start_date'])
gt = gt.dropna(how='any')
toc()


# In[ ]:


#
# Generate predictions
#
# Template for selected submodel forecast file
src_file_template = os.path.join("models", model_name, "submodel_forecasts", "{}", 
                                 f"{gt_id}_{horizon}", f"{gt_id}_{horizon}"+"-{}.h5")
# Auxiliary dataframe used to identify tuning dates
X = pd.DataFrame(index=metric_df.index, columns = ["delta", "dividend", "remainder"], 
                 dtype=np.float64)
rmses = pd.Series(index=target_date_objs, dtype=np.float64)
for target_date_obj in target_date_objs:
    tic()
    target_date_str = datetime.strftime(target_date_obj, '%Y%m%d')
    # Determine which dates will be used to assess submodels
    printf(f"\n\nObtaining tuning dates for target date {target_date_str}")
    tic()
    tuning_dates = get_tuning_dates(gt_id, horizon, target_date_obj, 
                                    num_years, margin_in_days, X)
    
    #print(tuning_dates)
    if not tuning_dates.any():
        printf(f"Warning: No tuning dates have RMSEs for target date {target_date_str}; skipping")
        continue
    toc()
    printf(f"Selecting most accurate submodel")
    tic()
    # Select most accurate submodel across these dates
    model_selected_submodel = metric_df[tuning_dates].mean().idxmin()
    # Form forecast by softlinking to selected submodel forecast
    src_file = src_file_template.format(model_selected_submodel, target_date_str) 
    toc()
    printf(f"Selected predictions -- {model_selected_submodel} for target_date {target_date_str}")
    if os.path.isfile(src_file):
        tic()
        dst_file = os.path.join(out_dir, f"{gt_id}_{horizon}-{target_date_str}.h5")
        #if target_date_obj > last_train_date:
        printf(f"Saving predictions")
        symlink(src_file, dst_file, use_abs_path=True)
        toc()
        if target_date_obj in gt.index:
            printf(f"Calculating rmse")
            tic()
            preds = pd.read_hdf(src_file)
            rmse = np.sqrt(np.square(preds.set_index(['start_date'])["pred"] - gt.loc[target_date_obj,:][var]).mean())
            rmses.loc[target_date_obj] = rmse
            toc()
    else:
        printf(f"Warning: Missing file:\n{src_file}")
    printf(f"Total processing time")
    toc()


# In[ ]:




