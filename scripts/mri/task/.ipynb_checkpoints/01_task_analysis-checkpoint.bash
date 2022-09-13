#!/bin/bash

echo $ARG_SESSION_ID
echo $ARG_TASK_NUM
echo $ARG_TASK_TYPE
echo $ARG_SMOOTH_FWHM

echo "Start task analysis for session id ${ARG_SESSION_ID}"
SESSION_ID="${ARG_SESSION_ID}"
TASK_NUM="${ARG_TASK_NUM}"
TASK_TYPE="${ARG_TASK_TYPE}"
SMOOTH_FWHM="${ARG_SMOOTH_FWHM}"

sessions_dir=/data/research/imaging/datasets/embarc/processed_data/pf-pipelines/qunex-nbridge/studies/embarc-20220301-EuKEp5Gw/sessions

# ---------
# Step 1
# ---------
# create batch file required for task processing
qunex create_batch     \
    sourcefiles="session_hcp.txt"     \
    paramfile="${sessions_dir}/specs/batch_parameters.txt" \
    sessionsfolder=${sessions_dir}     \
    targetfile="${sessions_dir}/${SESSION_ID}/fmri_task_batch.txt"     \
    sessions=$SESSION_ID \
    overwrite=yes
    
    
