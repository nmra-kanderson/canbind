#!/bin/python

import numpy as np
import pandas as pd
from pathlib import Path

base_dir = '/home/ubuntu/canbind-fsx'
sessions_dir = Path(base_dir, 'research/imaging/datasets/SRPBS/processed_data/pf-pipelines/qunex-nbridge/studies/CANBIND-20220818-mCcU5pi4/sessions')


# list session subdirs
session_dirs = list(Path(sessions_dir).glob('*_01'))
manifest_df = pd.DataFrame({'session_id': [x.stem for x in session_dirs]})

session_list = []
for session in session_dirs:
    print(session)
    session_hcp = Path(session, 'session_hcp.txt')
    if session_hcp.exists():
        hcp_map = parse_session_hcp(session_hcp, ['T1w', 'bold faces run-01', 'bold anhedonia run-01', 'bold gonogo run-01','bold faces run-02', 'bold rest run-01'])
        session_df = pd.DataFrame({'scan':['T1w'] + list(hcp_map.keys()), 'scan_num': ['T1w'] + list(hcp_map.values())})
        session_df.insert(0, 'session_id', session.stem)
        session_list.append(session_df)

manifest_df = pd.concat(session_list)
manifest_df.to_csv('/home/ubuntu/canbind-fsx/research/imaging/datasets/SRPBS/imaging_features/manifest_df_base.csv', index=None)


def parse_session_hcp(file_path, scan_types):
    """
    Finds the BOLD ID associated with each named file from
    the session_hcp.txt file of a given session

    Parameters
    ----------
    file_path: str
        Path to the sessions_hcp.txt file
        Example:
            ~/fsx-mount/embarc-20201122-LHzJPHi4/sessions/CU0011_baseline/session_hcp.txt
    scan_types: list[str]
        The names of scans to process. The names must exactly match
        what appears in the session_hcp.txt file. Not all scan types
        need to appear in all of the session_hcp.txt files from every
        session, but "ert" won't pick up "bold ert", if "bold ert"
        is what appears in session_hcp.txt. Within a given study,
        some sessions will assign different IDs to the same scan types
        depending on scan availability within a session. QuNex doesn't
        like gaps in BOLD ID numbers within a session
        Example: ['bold ert', 'bold rest run-1', 'bold rest run-2',
            'bold reward' ]

    Returns
    -------
    mapping: dict:
        The names and their associated BOLD IDs
        Example: {
            'bold ert': '1', 'bold rest run-1': '2',
            'bold rest run-2': '3', 'bold reward': '4'
        }
    """

    # Build the mapping from rest/task to the bold scan, e.g., bold1, bold2, etc.
    with open(file_path, 'r') as f:
        info = f.read()
    mapping = dict()

    # The regex | means "or", i.e., allowed to match any of the scan_types
    scan_type_regex = '|'.join(scan_types)

    # Find all of the matches
    matches = re.findall(f'bold([0-9]+).+:({scan_type_regex})\n', info)
    for bold_id, bold_name in matches:
        mapping[bold_name] = bold_id

    # Try the first known HCP format if no matches found after standard mapping
    if len(mapping) == 0:
        matches = re.findall(f'bold([0-9]+):.+\s:\s({scan_type_regex}).+\n', info)
        for bold_id, bold_name in matches:
            mapping[bold_name] = bold_id

    return mapping

