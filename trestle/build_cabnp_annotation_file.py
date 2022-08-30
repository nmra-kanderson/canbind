"""
THIS SCRIPT INTENDED TO BE CALLED by scripts/build_cabnp_annotation_file.sh

Do not run it on its own -- it will likely break.

Written by jburt on Oct 14, 2021.

"""

import nibabel as nib
import numpy as np
import pandas as pd

# # first build a mapping from parcel to (network, structure, hemisphere)

# network_id[network_name] = network_id
network_id = dict()
parcel_ids = list()
parcel_names = list()

i = 0
with open('Network_Labels.txt', 'r') as f:
    lines = f.readlines()
    for line in lines:
        if not (i % 2):  # even
            name = line.rstrip('\n')
        else:
            integer = int(line.rstrip('\n').split(' ')[0])
            network_id[name] = integer
        i += 1

# invert such that network_name[id] = name
network_name = {nid: name for name, nid in network_id.items()}

i = 0
with open('Parcel_Labels.txt', 'r') as f:
    lines = f.readlines()
    for line in lines:
        if not (i % 2):  # even
            parcel_names.append(line.rstrip('\n'))
        else:  # odd
            parcel_ids.append(int(line.rstrip('\n').split(' ')[0]))
        i += 1

network_labels = np.array(
    nib.load('Parcel_To_Network_Mapping.plabel.nii').get_fdata(),
    dtype=int).squeeze()

vals = list()

for pid, pn, nid in zip(parcel_ids, parcel_names, network_labels):
    structure = pn.split('-')[-1]
    hemi = pn.rstrip(f'-{structure}').split('_')[-1]
    line = [pid, pn, nid, network_name[nid], hemi, structure]
    vals.append(line)

df = pd.DataFrame(
    index=np.arange(len(vals)),
    columns=['parcel_id', 'parcel_name', 'network_id', 'network_name', 'hemisphere',
             'structure'],
    data=vals
)
df.index.name = 'parcel_index'
df.to_csv('cabnp_parcel_to_network.csv')

# -------------------------------------------------

# next, build a simple mapping from voxel to parcel

# TODO obtain this filename programmatically?
of = nib.load(
    "CortexSubcortex_ColeAnticevic_NetPartition_wSubcorGSR_parcels_LR.dlabel.nii")
dlabels = np.asarray(of.get_fdata()).squeeze()
assert dlabels.size == 91282
voxel_names = np.array(list(map(lambda x: f"voxel_{x}", range(91282))))
df2 = pd.DataFrame(
    data={'voxel_name': voxel_names,
          'parcel_id': dlabels,
          'voxel_id': np.arange(91282).astype(int)}
)

# map from parcel_id to parcel_name
mapping = df.reset_index()[
    ['parcel_id', 'parcel_name', 'parcel_index']].set_index('parcel_id')
df2['parcel_id'] = df2['parcel_id'].astype(int)
df2 = df2.merge(mapping, left_on='parcel_id', right_on='parcel_id')
assert df2.parcel_name.nunique() == df2.parcel_id.nunique() == 718
assert len(set(df2.index.values)) == 91282

df2.set_index('voxel_id').sort_index().drop('parcel_id', axis=1).to_csv(
    'cabnp_voxel_to_parcel.csv', index=False
)

# -------------------------------------------------

# also build a mapping from voxel to network

of = nib.load(
    "CortexSubcortex_ColeAnticevic_NetPartition_wSubcorGSR_netassignments_LR.dlabel.nii")
dlabels = np.asarray(of.get_fdata()).squeeze()
assert dlabels.size == 91282
voxel_names = np.array(list(map(lambda x: f"voxel_{x}", range(91282))))
df3 = pd.DataFrame(
    data={'voxel_name': voxel_names,
          'network_id': dlabels,
          'voxel_id': np.arange(91282).astype(int)}
)
df3['network_id'] = df3['network_id'].astype(int)

# map from network_id to network_name
mapping = df.reset_index()[['network_id', 'network_name']].sort_values(
    by='network_id', axis=0).drop_duplicates()
mapping['network_index'] = np.arange(mapping.shape[0], dtype=int)
mapping.set_index('network_id', inplace=True)

df3 = df3.merge(mapping, left_on='network_id', right_on='network_id')
assert df3.network_name.nunique() == df3.network_id.nunique() == 12
assert len(set(df3.index.values)) == 91282

df3.set_index('voxel_id').sort_index().drop('network_id', axis=1).to_csv(
    'cabnp_voxel_to_network.csv', index=False
)
