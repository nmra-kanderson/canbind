import os
import numpy as np
import pandas as pd
import pathlib

# Files & directories
phenotype_dir = (
    '/embarc-data/research/imaging/datasets/embarc/processed_data/'
    'pf-pipelines/qunex-nbridge/studies/embarc-20201122-LHzJPHi4/info/bids/'
    'embarc-20201122-LHzJPHi4/phenotype/'
)
output_dir = pathlib.Path(
    "/home/ubuntu/Documents/mml/processed-data/phenotype/")

# Master dictionary containing info about files to process and scales to retain
table_dict = {}

# EMB Demographics
table_name = 'EMBDEM01'
table_dict[table_name] = {}
tmp = pd.read_csv(
    os.path.join(phenotype_dir, f'{table_name}.tsv'), sep='\t',
    index_col=0).dropna(axis=1, how='all')
columns = tmp.columns.tolist()
columns.remove('WEEK')
table_dict[table_name]['val_type_dict'] = {'DEMO': columns}
table_dict[table_name]['drop_on'] = None

# Randomization
table_name = 'randomization'
table_dict[table_name] = {}
table_dict[table_name]['val_type_dict'] = {
    'RAND': ['Stage1TX1', 'Stage2TX', 'CurrentStatus']
}
table_dict[table_name]['drop_on'] = None

# HAMD
table_name = 'HRSD01'
table_dict[table_name] = {}
table_dict[table_name]['val_type_dict'] = {
    'IS': ['HSOIN', 'HMNIN', 'HEMIN', 'HMDSD', 'HPANX', 'HINSG', 'HWL',
           'HSANX', 'HHYPC', 'HVWSF', 'HSUIC', 'HINTR', 'HSLOW', 'HAGIT',
           'HSEX', 'HAMD_02', 'HAMD_04',
           'HAMD_05', 'HAMD_09', 'HAMD_10', 'HAMD_11', 'HAMD_12', 'HAMD_13',
           'HAMD_15', 'HAMD_16', 'HAMD_17', 'HAMD_18', 'HAMD_19', 'HAMD_22',
           'HAMD_25', 'HAMD_31', 'HAMD_32', 'HAMD_33', 'HAMD_34', 'HAMD_35', ],
    'SS': ['HAMD_36', 'HAMD_SCORE_24'],
    'TS': ['HRSD_TOTAL']}
table_dict[table_name]['drop_on'] = None

# QIDS
table_name = 'QIDS01'
table_dict[table_name] = {}
table_dict[table_name]['val_type_dict'] = {
    'IS': ['VSOIN', 'VMNIN',
           'VEMIN', 'VHYSM', 'VMDSD', 'VAPDC', 'VAPIN', 'VWTDC', 'VWTIN',
           'VCNTR', 'VVWSF', 'VSUIC', 'VINTR', 'VENGY', 'VSLOW', 'VAGIT', ],
    'TS': ['QIDS_EVAL_TOTAL']}
table_dict[table_name]['drop_on'] = ['QIDS_EVAL_TOTAL']

# STAI
table_name = 'STAI01'
table_dict[table_name] = {}
table_dict[table_name]['val_type_dict'] = {
    'IS': ['STAI1', 'STAI2',
           'STAI3', 'STAI4', 'STAI5', 'STAI6', 'STAI7', 'STAI8', 'STAI9',
           'STAI10', 'STAI11', 'STAI12', 'STAI13', 'STAI14', 'STAI15',
           'STAI16', 'STAI17', 'STAI18', 'STAI19', 'STAI20', ],
    'SS': [],
    'TS': ['STAI_POST_FINAL_SCORE']}
table_dict[table_name]['drop_on'] = ['STAI_POST_FINAL_SCORE']

# SHAPS
table_name = 'SHAPS01'
table_dict[table_name] = {}
table_dict[table_name]['val_type_dict'] = {
    'IS': ['SHAPS1', 'SHAPS2',
           'SHAPS3', 'SHAPS4', 'SHAPS5', 'SHAPS6', 'SHAPS7', 'SHAPS8',
           'SHAPS9', 'SHAPS10', 'SHAPS11', 'SHAPS12', 'SHAPS13', 'SHAPS14', ],
    'SS': [],
    'TS': ['SHAPS_TOTAL_CONTINUOUS', 'SHAPS_TOTAL_DICHOTOMOUS']}
table_dict[table_name]['drop_on'] = None

# MASQ
table_name = 'MASQ01'
table_dict[table_name] = {}
table_dict[table_name]['val_type_dict'] = {
    'IS': ['MASQ_01', 'MASQ_02',
           'MASQ_03', 'MASQ_04', 'MASQ_05', 'MASQ_06', 'MASQ_07', 'MASQ_08',
           'MASQ_09', 'MASQ_10', 'MASQ_11', 'MASQ_12', 'MASQ_13', 'MASQ_14',
           'MASQ_15', 'MASQ_16', 'MASQ_17', 'MASQ_18', 'MASQ_19', 'MASQ_20',
           'MASQ_21', 'MASQ_22', 'MASQ_23', 'MASQ_24', 'MASQ_25', 'MASQ_26',
           'MASQ_27', 'MASQ_28', 'MASQ_29', 'MASQ_30'],
    'SS': ['MASQ2_SCORE_AA',
           'MASQ2_SCORE_AD', 'MASQ2_SCORE_GD', 'MASQ2_SCORE_GDD'],
}
table_dict[table_name]['drop_on'] = None

# NEO
table_name = 'NEO_FFI_FORM_S_ADULT_200301'
table_dict[table_name] = {}
table_dict[table_name]['val_type_dict'] = {
    'IS': ['NEO_Q1', 'NEO_Q2', 'NEO_Q3', 'NEO_Q4', 'NEO_Q5', 'NEO_Q6',
           'NEO_Q7', 'NEO_Q8', 'NEO_Q9', 'NEO_Q10', 'NEO_Q11', 'NEO_Q12',
           'NEO_Q13', 'NEO_Q14', 'NEO_Q15', 'NEO_Q16', 'NEO_Q17', 'NEO_Q18',
           'NEO_Q19', 'NEO_Q20', 'NEO_Q21', 'NEO_Q22', 'NEO_Q23', 'NEO_Q24',
           'NEO_Q25', 'NEO_Q26', 'NEO_Q27', 'NEO_Q28', 'NEO_Q29', 'NEO_Q30',
           'NEO_Q31', 'NEO_Q32', 'NEO_Q33', 'NEO_Q34', 'NEO_Q35', 'NEO_Q36',
           'NEO_Q37', 'NEO_Q38', 'NEO_Q39', 'NEO_Q40', 'NEO_Q41', 'NEO_Q42',
           'NEO_Q43', 'NEO_Q44', 'NEO_Q45', 'NEO_Q46', 'NEO_Q47', 'NEO_Q48',
           'NEO_Q49', 'NEO_Q50', 'NEO_Q51', 'NEO_Q52', 'NEO_Q53', 'NEO_Q54',
           'NEO_Q55', 'NEO_Q56', 'NEO_Q57', 'NEO_Q58', 'NEO_Q59', 'NEO_Q60', ],
    'SS': ['NEO_N', 'NEO_E', 'NEO_O', 'NEO_A', 'NEO_C'],
}
table_dict[table_name]['drop_on'] = None

# CGI
table_name = 'CGI01'
table_dict[table_name] = {}
table_dict[table_name]['val_type_dict'] = {
    'SS': ['CGI_SI', 'CGI_SII'],
}
table_dict[table_name]['drop_on'] = None

# VAMS
table_name = 'VAMS01'
table_dict[table_name] = {}
table_dict[table_name]['val_type_dict'] = {
    'SS': ['VAS1', 'VAMS_HAP1',
           'VAMS_WIT1', 'VAMS_REL1', 'VAMS_SOC1'],
}
table_dict[table_name]['drop_on'] = None

# MDQ01
table_name = 'MDQ01'
table_dict[table_name] = {}
table_dict[table_name]['val_type_dict'] = {
    'IS': ['MDQ1A1', 'MDQ1A2', 'MDQ1A3', 'MDQ1A4', 'MDQ1A5', 'MDQ1A6',
           'MDQ1A7', 'MDQ1A8', 'MDQ1A9', 'MDQ1A10', 'MDQ1A11', 'MDQ1A12',
           'MDQ1A13'],
    'TS': ['MDQSCORE_TOTAL']
}
table_dict[table_name]['drop_on'] = None

# ASRM01
table_name = 'ASRM01'
table_dict[table_name] = {}
table_dict[table_name]['val_type_dict'] = {
    'IS': ['ASRM_1', 'ASRM_2', 'ASRM_3', 'ASRM_4', 'ASRM_5'],
    'SS': [],
    'TS': ['ASRM_SCORE2']
}
table_dict[table_name]['drop_on'] = None

# SAPAS
table_name = 'SAPAS01'
table_dict[table_name] = {}
table_dict[table_name]['val_type_dict'] = {
    'IS': ['SAPAS_01',
           'SAPAS_02', 'SAPAS_03', 'SAPAS_04', 'SAPAS_05', 'SAPAS_06',
           'SAPAS_07', 'SAPAS_08'],
    'SS': [],
    'TS': ['SAPAS_SCORE']
}
table_dict[table_name]['drop_on'] = None

# AAQ
table_name = 'AAQ01'
table_dict[table_name] = {}
table_dict[table_name]['val_type_dict'] = {
    'IS': ['AAQ_1', 'AAQ_2',
           'AAQ_3', 'AAQ_4', 'AAQ_5A', 'AAQ_5B', 'AAQ_5C', 'AAQ_5D', 'AAQ_5E',
           'AAQ_5F', 'AAQ_5G', 'AAQ_5H', 'AAQ_5I', 'AAQ_5J', 'AAQ_5K',
           'AAQ_5L', 'AAQ_5M', 'AAQ_6', 'AAQ_7', ],
    'TS': ['AAQ_SCORE_RESULT']
}
table_dict[table_name]['drop_on'] = None

# CTQ01
table_name = 'CTQ01'
table_dict[table_name] = {}
table_dict[table_name]['val_type_dict'] = {
    'IS': ['CTQ_01', 'CTQ_02',
           'CTQ_03', 'CTQ_04', 'CTQ_05', 'CTQ_06', 'CTQ_07', 'CTQ_08',
           'CTQ_09', 'CTQ_10', 'CTQ_11', 'CTQ_12', 'CTQ_13', 'CTQ_14',
           'CTQ_15', 'CTQ_16', 'CTQ_17', 'CTQ_18', 'CTQ_19', 'CTQ_20',
           'CTQ_21', 'CTQ_22', 'CTQ_23', 'CTQ_24', 'CTQ_25', 'CTQ_26',
           'CTQ_27', 'CTQ_28'],
    'SS': ['CTQSCORE_EA', 'CTQSCORE_EN', 'CTQSCORE_PA',
           'CTQSCORE_PN', 'CTQSCORE_SA', 'CTQSCORE_VAL'],
}
table_dict[table_name]['drop_on'] = None

# CASTS01
table_name = 'CASTS01'
table_dict[table_name] = {}
table_dict[table_name]['val_type_dict'] = {
    'IS': ['CAST_01', 'CAST_02',
           'CAST_03', 'CAST_04', 'CAST_05', 'CAST_06', 'CAST_07', 'CAST_08',
           'CAST_09', 'CAST_10', 'CAST_11', 'CAST_12', 'CAST_13', 'CAST_14',
           'CAST_15', 'CAST_16', 'CAST_17'],
    #                 'SS':[],
    'TS': ['CAST_SCORE_TOTAL']
}
table_dict[table_name]['drop_on'] = None

# Social Adjustment
table_name = 'SOADJS01'
table_dict[table_name] = {}
table_dict[table_name]['val_type_dict'] = {
    'IS': ['SAS_SR_01',
           'SAS_SR_02', 'SAS_SR_03', 'SAS_SR_04', 'SAS_SR_05', 'SAS_SR_06',
           'SAS_SR_07', 'SAS_SR_08', 'SAS_SR_09', 'SAS_SR_10', 'SAS_SR_11',
           'SAS_SR_12', 'SAS_SR_13', 'SAS_SR_14', 'SAS_SR_15', 'SAS_SR_16',
           'SAS_SR_17', 'SAS_SR_18', 'SAS_SR_19', 'SAS_SR_1A', 'SAS_SR_20',
           'SAS_SR_21', 'SAS_SR_22', 'SAS_SR_23', 'SAS_SR_24', 'SAS_SR_2A',
           'SAS_SR_3A', 'SAS_SR_3C', 'SAS_SR_4A', 'SAS_SR_5A', 'SAS_SR_7B',
           'SAS_SR_8B', 'SAS_SR_9B', ],
    'SS': ['SAS_OVERALL_FACTOR', 'SAS_OVERALL_MEAN',
           'SAS_OVERALL_NUMQUES'],
    #                 'TS':[]
}
table_dict[table_name]['drop_on'] = None

# Concise Health Risk Tracking
table_name = 'CHRT01'
table_dict[table_name] = {}
table_dict[table_name]['val_type_dict'] = {
    'SS': [
        'CHRTP_PROPENSITY_SCORE',
        'CHRTP_RISK_SCORE']
}
table_dict[table_name]['drop_on'] = None

# Childhood Trauma Questionnaire
table_name = 'CTQ01'
table_dict[table_name] = {}
table_dict[table_name]['val_type_dict'] = {
    'SS': [
        'CTQSCORE_EA',  # emotional abuse
        'CTQSCORE_EN',  # emotional neglect
        'CTQSCORE_PA',  # physical abuse
        'CTQSCORE_PN',  # physical neglect
        'CTQSCORE_SA',  # sexual abuse
        'CTQSCORE_VAL',  # validity
    ]
}
table_dict[table_name]['drop_on'] = None

# Fagerstrom Test for Nicotine Dependence
table_name = 'FAGERSTROM01'
table_dict[table_name] = {}
table_dict[table_name]['val_type_dict'] = {
    'DEMO': [
        'FTND_7',  # Do you currently smoke any cigarettes? Y/N
    ]
}
table_dict[table_name]['drop_on'] = None

# Behavioral Phenotyping

# 'ANOTB',  # a not b
# 'CHOICERT',  # choice reaction time
# 'FLKR',  # flanker task
# 'PRT',  # probabilistic reward task
# 'WF'  # word fluency

tasks = ['ANOTB', 'CHOICERT', 'FLKR', 'PRT', 'WF']
table_name = 'BP_SCALAR01'
table_dict[table_name] = {}
table_dict[table_name]['val_type_dict'] = {}
df = pd.read_csv(os.path.join(phenotype_dir, f'{table_name}.tsv'),
                 sep='\t', index_col=0).dropna(axis=1, how='all')
bp_cols = list()
for prefix in tasks:
    bp_cols += list(
        df.columns.values[np.where(df.columns.str.contains(prefix))[0]]
    )
table_dict[table_name]['val_type_dict']['BP'] = bp_cols
table_dict[table_name]['drop_on'] = None

# Screening Test Results
table_name = 'STR01'
table_dict[table_name] = {}
table_dict[table_name]['val_type_dict'] = {
    'LAB': [
        'WEIGHT_STD',
        'HEIGHT_STD',
        'BMI',
        'STRF_CHOLESTEROL_HDL',
        'STRF_CHOLESTEROL_LDL',
        'STRF_CHOLESTEROL_TOTAL',
        'STRF_CHOLESTEROL_TRIGLYCERIDES',
        'GLUVAL',  # glucose value
        'TSH',  # thyroid stimulating hormone
        'STRF_CRP'  # c-reactive protein
    ]
}
table_dict[table_name]['drop_on'] = None

# SCID
table_name = 'SCID01'
table_dict[table_name] = {}
table_dict[table_name]['val_type_dict'] = {
    'SS': ['Q021_MDD_LIFE', 'Q022_MDD_EP', 'Q026_MDD_CURRENT', 'Q027_MDD_SEV',
           'Q096_PTSD_LIFE', 'Q097_PTSD_MON'],
}
table_dict[table_name]['drop_on'] = None

# ------------------------------------------------


delim = '-'

all_columns = list()
metadata = list()

type2desc = {
    'DEMO': 'demographic',
    'RAND': 'randomization',
    'IS': 'item_score',
    'SS': 'subscale_score',
    'TS': 'total_score',
    'BP': 'behavior',
    'LAB': 'lab_test'
}

for table_name in table_dict:
    val_type_dict = table_dict[table_name]['val_type_dict']
    for val_type in val_type_dict:
        # all_columns += list(map(
        #     lambda x: x + f'{delim}{val_type}{delim}{table_name}',
        #     val_type_dict[val_type]
        # ))
        cols = list(val_type_dict[val_type])
        ncols = len(cols)
        all_columns += cols
        metadata += list(
            zip(cols, [type2desc[val_type]]*ncols, [table_name]*ncols)
        )

# save metadata containing type and origin of each column
metadf = pd.DataFrame(
    columns=['Column', 'Type', 'Source'],
    data=np.array(metadata)
)
metadf.to_csv(
    output_dir.joinpath('embarc.trestle_project.phenotype.processed_data.agg.csv'),
    index=False
)


timepoints = [0, 1, 2, 4, 6, 8, 10, 12, 14, 16]

# determine unique subjects
ndar_subject = pd.read_csv(
    os.path.join(phenotype_dir, 'NDAR_SUBJECT01.tsv'), sep='\t')
subjects = ndar_subject.drop_duplicates('participant_id').participant_id
nsub = len(subjects)


# initialize master dataframe with all subjects at all timepoints
def template(_timepoint):
    return pd.DataFrame(
        data={
            'participant_id': subjects,
            'WEEK': [_timepoint] * nsub,
        }
    )


master = pd.DataFrame(columns=['participant_id', 'WEEK'])
for timepoint in timepoints:
    master = master.append(template(timepoint))
print(master.shape)
assert master.shape == (nsub * len(timepoints), 2)
master.head()

# add all columns
master[all_columns] = np.nan
master = master.copy()  # de-fragment in memory
print(master.shape)

# set multiindex to be (participant_id, week)
master.set_index(['participant_id', 'WEEK'], inplace=True)
print(master.shape)

# loop over files
for table_name in table_dict:

    print(f'\n------- {table_name} -------')

    val_type_dict = table_dict[table_name]['val_type_dict']
    drop_on = table_dict[table_name]['drop_on']

    file = os.path.join(phenotype_dir, f'{table_name}.tsv')
    df = pd.read_csv(file, sep='\t', index_col=0).dropna(axis=1, how='all')

    if table_name == 'BP_SCALAR01':
        # each task is on a different row; need to aggregate so that each
        # row corresponds to (subject, week, all_tasks)
        temp = df.copy().reset_index()
        df = temp.groupby(['participant_id', 'WEEK']
                          ).first().reset_index().set_index('participant_id')

    if drop_on:
        df = df.dropna(subset=drop_on)

    if table_name != 'randomization':
        timepoints = df['WEEK'].unique()
    else:
        timepoints = np.array([0])

    # loop over column types (item score, subscore, total score)
    for val_type in val_type_dict:

        # loop over timepoints
        for timepoint in timepoints:

            cols = val_type_dict[val_type]

            # get data for this timepoint (week) for desired columns
            dfw = df.copy()
            if table_name != 'randomization':
                dfw = dfw[dfw['WEEK'] == timepoint][cols]
            else:  # treat all randomization data as week 0 data
                pass

            if dfw.isna().all().all():
                continue

            # check for duplicate rows
            dup_ids = dfw.loc[dfw.index.duplicated()].index.unique().values
            if len(dup_ids):

                print(
                    f'{len(dup_ids)} subjects have duplicated rows at week {timepoint}!')

                for p_id in dup_ids:

                    dfw.reset_index(inplace=True)

                    # subset of rows with same participant id
                    p_id_df = dfw[dfw['participant_id'] == p_id]
                    num_dup = p_id_df.shape[1]
                    assert num_dup > 0

                    # case where all the rows are identical -- select first one
                    if p_id_df.duplicated().sum() == (p_id_df.shape[0] - 1):
                        drop_index = p_id_df.loc[~p_id_df.duplicated()].index.values
                        dfw.drop(drop_index, axis=0, inplace=True)

                    # case where rows are different -- compute mean
                    else:

                        # compute mean value
                        # means = dfw[dfw['participant_id'] == p_id].iloc[-1]
                        means = dfw[dfw['participant_id'] == p_id].mean(numeric_only=True)

                        # drop all but 1 row for this subject
                        insert_idx = dfw[dfw['participant_id'] == p_id].index[-1]
                        drop_index = dfw[dfw['participant_id'] == p_id].index[:-1]
                        dfw.drop(drop_index, axis=0, inplace=True)

                        # update the non-dropped row with the mean values
                        nonnumeric_data = \
                            dfw[dfw['participant_id'] == p_id].select_dtypes(
                                include=object).loc[insert_idx]
                        # dfw.loc[insert_idx] = means
                        dfw.loc[insert_idx] = means.append(nonnumeric_data)

                    dfw.set_index('participant_id', inplace=True)

            dfw.reset_index(inplace=True)
            dfw['WEEK'] = timepoint
            dfw.set_index(['participant_id', 'WEEK'], inplace=True)
            # dfw.rename(columns=lambda x: x + f'{delim}{val_type}{delim}{table_name}',
            #            inplace=True)

            master.update(dfw, overwrite=True, errors='raise')


# drop "sub-" prefix from subject ID
master.reset_index(inplace=True)
master['participant_id'] = master['participant_id'].apply(
    lambda x: x.lstrip('sub-'))

# write to file
master['WEEK'] = master['WEEK'].astype(str)
master['Index'] = master[['participant_id', 'WEEK']].agg('_'.join, axis=1)
master = master.set_index('Index')
master.to_csv(output_dir.joinpath('phenotype_long.csv'), index=False)
