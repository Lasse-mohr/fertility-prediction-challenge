from os import makedirs
from os.path import isdir, isfile
import json
import pandas as pd
# import re

PID_col = 'nomem_encr'

def to_sequences(df, codebook, use_codebook=True, custom_pairs=None,
                 save_inter_path='data_processing/codebook_false/to_sequences/'):
    """
        Arguments:
            use_codebook (bool): determines if the codebook is used to group columns
                that correspond to the same questions and use these groups to create
                variable-name indices or to read this information from file
                created during an earlier call to the function
    """
    #   paths for storing intermediary results to avoid using codebook again
    generic_name_path = save_inter_path + 'generic_names.csv'
    var_name_index_path = save_inter_path + 'var_name_index.json'

    #  use codebook to infer columns corresponding to the same question
    if use_codebook:
        codebook = codebook[codebook.year.notna()]
        codebook['year'] = codebook['year'].astype(int)

        # Get all question pairs
        codebook["pairs"] = codebook['var_name'].apply(get_generic_name)

        if custom_pairs is not None:
            codebook = codebook[codebook["pairs"].isin(custom_pairs)]

        var_name_index = {}
        for i, (_, x) in enumerate(codebook.groupby("pairs", sort=False)):
            for _, row in x.iterrows():
                var_name_index[row['var_name']] = (row.year, i)

        if not isdir(save_inter_path):
            makedirs(save_inter_path)

        with open(var_name_index_path, 'w') as file:
            json.dump(var_name_index, file)

        pairs = pd.Series(codebook['pairs'].unique())
        pairs.to_csv(generic_name_path, index=False)

    else:
        #  Read from files which columns correspond to the same question

        if not isdir(save_inter_path):
            makedirs(save_inter_path)

        if isfile(generic_name_path):
            pairs = pd.read_csv(generic_name_path).squeeze()
        else:
            print(f'File with pair names not found at: {generic_name_path}')
            print('Call to_sequence with use_codebook=True to create file')

        if isfile(var_name_index_path):
            with open(var_name_index_path, 'r') as file:
                var_name_index = json.load(file)
        else:
            print(f'File with variable indices not found at: {var_name_index_path}')
            print('Call to_sequences with use_codebook=True to create file')

    pids = df[PID_col]
    # Create dict of {pid: {year: sequences}}
    N = len(pairs)
    years = [2007 + year for year in range(14)]
    seq = {pid: {year: [101]*N for year in years}
           for pid in pids}    # 101 is UNK

    for column, (year, idx) in var_name_index.items():
        # If column isn't present, we skip it (defaulting to 101)
        if column not in df:
            continue
        for pid, val in zip(pids, df[column]):
            seq[pid][year][idx] = val

    return seq


def get_generic_name(var_name: str):
    """
    Returns standardized name of the column if possible: XXNNN, but only if the string starts with 'c'.
    """
    if var_name.startswith('c'):
        if var_name.endswith('_m'):
            return var_name[:2] + var_name[-1]
        else:
            return var_name[:2] + var_name[-3:]
    else:
        return var_name.split("_")[0]

# DEPRECATED


def get_pairs(var_name):
    """
    Return standardized names of columns in the form CANNN or CAM.

    Parameters:
    var_name (str): The original column name.

    Returns:
    str: The standardized column name.
    """
    if var_name.startswith('c'):
        if var_name.endswith('_m'):
            return var_name[:2] + var_name[-1]
        else:
            return var_name[:2] + var_name[-3:]
    else:
        return var_name.split("_")[0]
