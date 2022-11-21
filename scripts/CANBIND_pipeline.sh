#!/bin/bash

# Build Docker Image
# cd to imaging features repo
cd /home/ubuntu/Projects/canbind


# Update below AWS credential variables
docker build  \
--build-arg AWS_ACCESS_KEY_ID=$AWS_ACCESS_KEY_ID \
--build-arg AWS_SECRET_ACCESS_KEY=$AWS_SECRET_ACCESS_KEY \
--build-arg AWS_SESSION_TOKEN=$AWS_SESSION_TOKEN \
 -f Dockerfile -t imaging_features .


# Set environment variables
# ---------------------
FSX_PATH=/home/ubuntu/fsx
DOCKER_FSX_PATH=/data

# ------------------------------------------
# Run Docker interactive 
# ------------------------------------------

sudo docker run -it -v \
  ${FSX_PATH}:${DOCKER_FSX_PATH} \
  -v /home/ubuntu/Projects/canbind/utils:/imaging-features/utils \
  -d imaging_features:latest


# jump into interactive session 
sudo docker exec -it ce351c7e6dfc bash



# run within the qunex container

#pip3 install kedro

#source activate imaging_features
export PYTHONPATH=/opt/env/qunex/bin:$PYTHONPATH

REPO_PATH=/imaging-features
qunex_dir=/data/research/imaging/datasets/CANBIND/processed_data/pf-pipelines/qunex-nbridge/studies/CANBIND-20220818-mCcU5pi4/sessions
feature_dir=/data/research/imaging/datasets/CANBIND/imaging-features


# ---------------------------------
# Freesurfer / Anatomical Features
# ---------------------------------
#### Desikan 

python3 $REPO_PATH/utils/neumora_struct_pipeline.py \
--sessions-dir ${qunex_dir} \
--output-dir ${feature_dir} \
--regex-session-filter '(.{7}_(01))' \
--study-name 'CANBIND' \
--n-processes 90


#### CABNP
python3 $REPO_PATH/utils/neumora_struct_pipeline.py \
--sessions-dir ${qunex_dir} \
--output-dir ${feature_dir} \
--regex-session-filter '(.{7}_(01))' \
--study-name 'CANBIND' \
--cifti-atlas /imaging-features/utils/reference/parcellation/ColeAnticevic/raw/CortexSubcortex_ColeAnticevic_NetPartition_wSubcorGSR_parcels_LR.dlabel.nii \
--cifti-atlas-name 'CABNP' \
--overwrite False \
--n-processes 90


#### YeoPlus
python /imaging-features/utils/neumora_struct_pipeline.py \
  --sessions-dir ${qunex_dir} \
  --output-dir ${feature_dir} \
  --regex-session-filter '(.{7}_(01))' \
  --study-name 'CANBIND' \
  --cifti-atlas /imaging-features/utils/reference/parcellation/YeoPlus/YeoPlus_Schaefer17Net200_CBL17Net_TianS3.dlabel.nii  \
  --cifti-atlas-name 'YeoPlus' \
  --overwrite False \
  --n-processes 40

# ---------------------------------
# Resting State / Functional Features
# ---------------------------------
# CABNP atlas

/opt/env/qunex/bin/python3 /imaging-features/utils/neumora_rest_pipeline.py \
  --sessions-dir ${qunex_dir} \
  --output-dir ${feature_dir} \
  --regex-session-filter '(.{7}_(01))' \
  --scans 'bold faces run-01' --scans 'bold faces run-02' --scans 'bold rest run-01' --scans 'bold anhedonia run-01' --scans 'bold gonogo run-01'  \
  --input-scan-pattern '_Atlas_s_hpss_res-mVWMWB1d_lpss.dtseries.nii' \
  --concatenate-scans \
  --concatenate-scan-name 'rest_concatenated' \
  --study-name 'CANBIND' \
  --cifti-atlas /imaging-features/utils/reference/parcellation/ColeAnticevic/raw/CortexSubcortex_ColeAnticevic_NetPartition_wSubcorGSR_parcels_LR.dlabel.nii \
  --cifti-atlas-name 'CABNP' \
  --n-processes 80 \
  --make-refined True \
  --make-production True


# make-production fails at task features 
#   Working on: task
# 100%|███████████████████████████████████████████████████████████████████████████| 298/298 [00:00<00:00, 14882.45it/s]
# Traceback (most recent call last):
#   File "/imaging-features/utils/neumora_rest_pipeline.py", line 797, in <module>
#     main()
#   File "/usr/local/lib/python3.9/site-packages/click/core.py", line 1137, in __call__
#     return self.main(*args, **kwargs)
#   File "/usr/local/lib/python3.9/site-packages/click/core.py", line 1062, in main
#     rv = self.invoke(ctx)
#   File "/usr/local/lib/python3.9/site-packages/click/core.py", line 1404, in invoke
#     return ctx.invoke(self.callback, **ctx.params)
#   File "/usr/local/lib/python3.9/site-packages/click/core.py", line 763, in invoke
#     return __callback(*args, **kwargs)
#   File "/imaging-features/utils/neumora_rest_pipeline.py", line 273, in main
#     feature_df = pd.concat(feature_df_list)
#   File "/usr/local/lib/python3.9/site-packages/pandas/util/_decorators.py", line 311, in wrapper
#     return func(*args, **kwargs)
#   File "/usr/local/lib/python3.9/site-packages/pandas/core/reshape/concat.py", line 294, in concat
#     op = _Concatenator(
#   File "/usr/local/lib/python3.9/site-packages/pandas/core/reshape/concat.py", line 374, in __init__
#     raise ValueError("All objects passed were None")
# ValueError: All objects passed were None


# YeoPlus atlas

/opt/env/qunex/bin/python3 /imaging-features/utils/neumora_rest_pipeline.py \
  --sessions-dir ${qunex_dir} \
  --output-dir ${feature_dir} \
  --regex-session-filter '(.{7}_(01))' \
  --scans 'bold faces run-01' --scans 'bold faces run-02' --scans 'bold rest run-01' --scans 'bold anhedonia run-01' --scans 'bold gonogo run-01'  \
  --input-scan-pattern '_Atlas_s_hpss_res-mVWMWB1d_lpss.dtseries.nii' \
  --study-name 'CANBIND' \
  --cifti-atlas /imaging-features/utils/reference/parcellation/YeoPlus/YeoPlus_Schaefer17Net200_CBL17Net_TianS3.dlabel.nii  \
  --cifti-atlas-name 'YeoPlus' \
  --n-processes 50 \
  --make-refined True \
  --make-production False

# make-production fails at funccon/covariance
#   ------------------------------
# Compiling "production" stats files
# ------------------------------
# Working on: funccon/covariance
# 100%|████████████████████████████████████████████████████████████████████████████| 298/298 [00:00<00:00, 6742.24it/s]
# Traceback (most recent call last):
#   File "/imaging-features/utils/neumora_rest_pipeline.py", line 797, in <module>
#     main()
#   File "/usr/local/lib/python3.9/site-packages/click/core.py", line 1137, in __call__
#     return self.main(*args, **kwargs)
#   File "/usr/local/lib/python3.9/site-packages/click/core.py", line 1062, in main
#     rv = self.invoke(ctx)
#   File "/usr/local/lib/python3.9/site-packages/click/core.py", line 1404, in invoke
#     return ctx.invoke(self.callback, **ctx.params)
#   File "/usr/local/lib/python3.9/site-packages/click/core.py", line 763, in invoke
#     return __callback(*args, **kwargs)
#   File "/imaging-features/utils/neumora_rest_pipeline.py", line 210, in main
#     mat_stack   = np.concatenate(matrix_list)
#   File "<__array_function__ internals>", line 180, in concatenate
# ValueError: need at least one array to concatenate


# ---------------------------------
# BOLD Motion/Mask QC Features
# ---------------------------------

sudo python3 ${REPO_PATH}/utils/neumora_qc_pipeline.py \
  --sessions-dir ${qunex_dir} \
  --output-dir ${feature_dir} \
  --regex-session-filter '(.{7}_(01))' \
  --scans 'bold faces run-01' --scans 'bold faces run-02' --scans 'bold rest run-01' --scans 'bold anhedonia run-01' --scans 'bold gonogo run-01'  \
  --study-name 'CANBIND' \
  --freesurfer-home '/opt/freesurfer/freesurfer-6.0/bin/freesurfer' \
  --n-processes 80 \
  --overwrite False
