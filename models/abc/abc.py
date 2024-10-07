#!/usr/bin/env python
# coding: utf-8

# # ABC
# 
# Adaptive Bias Correction combining dynamical model forecasts, lagged measurements, and climatology.
# 
# **Note:** This script will call `batch_predict` and `batch_metrics` for any ABC ensemble member 
#   (perpp, tuned_climpp, or tuned_{forecast}pp) that is missing RMSE metrics for the 
#   given target_dates. However, it does not call `bulk_batch_predict` for climpp or
#   {forecast}pp, so please ensure that all submodels have been generated for climpp
#   {forecast}pp prior to calling this script, for example, by using the commands:  
#   
#   `python -m subseasonal_toolkit.generate_predictions -t std_paper_eval -e -u -b -m climpp`
#   
#   `python -m subseasonal_toolkit.generate_predictions -t std_paper_eval -e -u -b -m ecmwfpp`

# In[ ]:


import os, sys
from subseasonal_toolkit.utils.notebook_util import isnotebook
if isnotebook():
    # Autoreload packages that are modified
    get_ipython().run_line_magic('load_ext', 'autoreload')
    get_ipython().run_line_magic('autoreload', '2')
else:
    from argparse import ArgumentParser

# Imports 
import numpy as np
import pandas as pd
from sklearn import *
import sys
import json
import subprocess
import pandas as pd
import numpy as np
import shutil
from datetime import datetime, timedelta
from filelock import FileLock
from pkg_resources import resource_filename
from ttictoc import tic, toc
from subseasonal_toolkit.utils.general_util import printf
from subseasonal_toolkit.utils.eval_util import get_target_dates, get_named_targets
from subseasonal_toolkit.utils.models_util import get_selected_submodel_name
from subseasonal_toolkit.models.linear_ensemble.attributes import get_submodel_name as get_linear_ensemble_sn
from subseasonal_toolkit.models.abc.abc_util import *
pd.set_option('display.max_rows', None)


# In[ ]:


#
# Specify model parameters
#
if not isnotebook():
    # If notebook run as a script, parse command-line arguments
    parser = ArgumentParser()
    parser.add_argument("pos_vars",nargs="*")  # gt_id and horizon                                                                                  
    parser.add_argument('--target_dates', '-t', default="std_contest")
    parser.add_argument('--forecast', '-f', default="cfsv2", 
                        help="include the forecasts of this dynamical model as features")
    parser.add_argument('--ensemble', '-e', nargs='+', default=['linear_ensemble'],
                        help="ensembling algorithm, either 'linear_ensemble', 'online_learning' or both")
    parser.add_argument('--cmd_prefix', '-c', default="python")
    args, opt = parser.parse_known_args()
    
    # Assign variables                                                                                                                                     
    gt_id = args.pos_vars[0] # "contest_precip" or "contest_tmp2m"                                                                            
    horizon = args.pos_vars[1] # "34w" or "56w"                                                                                        
    target_dates = args.target_dates
    forecast = args.forecast
    ensemble_models = args.ensemble
    # Strip whitespace from inputted cmd_prefix
    cmd_prefix = args.cmd_prefix.strip()
else:
    # Otherwise, specify arguments interactively 
    gt_id = "us_precip_1.5x1.5"
    horizon = "34w"
    target_dates = "std_paper_forecast"
    forecast = "ecmwf"
    ensemble_models = ["linear_ensemble"]#, "online_learning"]
    cmd_prefix = "python"

# Get forecasting task
task = f"{gt_id}_{horizon}"
# Get list of ensemble member model names
pp_name = f'tuned_{forecast}pp'
perpp_name = 'perpp_' + forecast
if "ecmwf:" in forecast:
    # ECMWF submodel requested; skip individual ensemble member steps
    ensemble_members = []
elif horizon == '12w':
    ensemble_members = [pp_name, perpp_name]
else:
    ensemble_members = ['tuned_climpp', pp_name, perpp_name]

# Get list of target date objects
target_date_objs = pd.Series(get_target_dates(date_str=target_dates,horizon=horizon))
    
# Process command-line arguments
metrics_prefix = cmd_prefix
if cmd_prefix != "python":
    # Add slurm resource requirements
    metrics_prefix += " --memory 8 --cores 1 --hours 0 --minutes 10"
metrics_script = resource_filename(__name__, os.path.join("..","..","batch_metrics.py"))
job_dependency_model = ""
job_dependency_metric = ""
job_dependency_ensemble = ""
cmd_suffix = ""
cluster_str = ""


# In[ ]:


# Generate ensemble members forecasts
for i, m in enumerate(ensemble_members):
    printf(f"\nGenerating forecasts for {m}:")
    model = m.replace('tuned_','') if m.startswith('tuned_') else m
        
    # Set command prefix and suffix
    if cmd_prefix != "python":
        cluster_str = get_cluster_params(model, gt_id, horizon, target_dates)
        # Keep track of job ID to specify job dependencies
        cmd_suffix = "| tail -n 1 | awk '{print $NF}'"
    
    
    # Skip if model metrics already exist
    if metric_file_exists(m, task, target_dates):
        printf(f"Skipping {m} -- metrics already exist for {target_dates}\n")
        continue
    elif m.startswith('tuned_'):
        predict_script = resource_filename(__name__, os.path.join("..","..","models","tuner","batch_predict.py"))
        cmd = f"{cmd_prefix} {cluster_str} \"{predict_script}\" {gt_id} {horizon} -t {target_dates} -mn {model} -y 3 -m None {cmd_suffix}"
    else:
        #args_str = get_sn_params(model, task, target_dates)
        predict_script = resource_filename(__name__, os.path.join("..","..","models",model,"batch_predict.py"))
        cmd = f"{cmd_prefix} {cluster_str} \"{predict_script}\" {gt_id} {horizon} -t {target_dates} -y all -m None {cmd_suffix}"

    printf(f"Running batch predict: \n{cmd}")
    if cmd_prefix == "python":
        subprocess.call(cmd, shell=True)
    else:
        # Store job ID to ensure batch metric call runs afterwards
        process = subprocess.run(cmd, check=True, stdout=subprocess.PIPE, universal_newlines=True, shell=True)
        job_id = process.stdout.rstrip()
        job_dependency_model=f"-d {job_id}" 

    
    #Run dependent job for metric generation on named target_date ranges
    if True:#target_dates in get_named_targets():
        metrics = "rmse score skill lat_lon_rmse"
        metrics_args = f"{gt_id} {horizon} -mn {m} -t {target_dates} -m {metrics}"
        metrics_cmd=f"{metrics_prefix} {job_dependency_model} {metrics_script} {metrics_args} {cmd_suffix}"
        printf(f"Running batch metrics: \n{metrics_cmd}\n")
        if cmd_prefix == "python":
            subprocess.call(metrics_cmd, shell=True)
        else:
            # Store job ID to ensure batch metric call runs afterwards
            process = subprocess.run(metrics_cmd, check=True, stdout=subprocess.PIPE, universal_newlines=True, shell=True)
            job_id = process.stdout.rstrip()
            job_dependency_metric=f"-d {job_id}" if i==0 else f"{job_dependency_model},{job_id}"


# In[ ]:


# Run ensembling model
for model in ensemble_models:
    printf(f"Running ensembling via {model}:")

    # Set command prefix and suffix
    if cmd_prefix != "python":
        cluster_str = get_cluster_params(model, gt_id, horizon, target_dates)
        # Keep track of job ID to specify job dependencies
        cmd_suffix = "| tail -n 1 | awk '{print $NF}'"

    predict_script = resource_filename(__name__, os.path.join("..", "..", "models",model,"batch_predict.py"))
    predict_args = f"{gt_id} {horizon} -t {target_dates} -f {forecast}"
    cmd = f"{cmd_prefix} {cluster_str} {job_dependency_metric} \"{predict_script}\" {predict_args} {cmd_suffix}"

    printf(f"Running ensemble batch predict: \n{cmd}")
    if cmd_prefix == "python":
        subprocess.call(cmd, shell=True)
    else:
        # Store job ID to ensure batch metric call runs afterwards
        process = subprocess.run(cmd, check=True, stdout=subprocess.PIPE, universal_newlines=True, shell=True)
        job_id = process.stdout.rstrip()
        job_dependency_ensemble=f"-d {job_id}"

    #Run dependent job for metric generation on named target_date ranges
    if ensemble_members:#target_dates in get_named_targets():
        metrics = "rmse score skill lat_lon_rmse"
        sn = get_linear_ensemble_sn(model_names=','.join(ensemble_members))
        metrics_args = f"{gt_id} {horizon} -mn abc_{forecast} -sn {sn} -t {target_dates} -m {metrics}"
        metrics_cmd=f"{metrics_prefix} {job_dependency_ensemble} {metrics_script} {metrics_args} {cmd_suffix}"
        printf(f"Running ensemble batch metrics: \n{metrics_cmd}\n\n")
        subprocess.call(metrics_cmd, shell=True)

    

