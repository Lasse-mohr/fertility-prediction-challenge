from os import makedirs
from os.path import isdir, isfile
import pickle
import pandas as pd

def to_sequences(df, codebook, use_codebook=True, custom_pairs=None,
                save_inter_path='data/codebook_false/to_sequences/',):
    """
        Arguments:
            use_codebook (bool): determines if the codebook is used to group columns
                that correspond to the same questions and use these groups to create
                variable-name indeces or wether to read this information from file
                created during an earlier call to the function
    """
    #   paths for storing intermediary results to avoid using codebook again
    pair_path = save_inter_path + 'pairs.csv'
    var_name_index_path = save_inter_path + 'var_name_index.p'

    #  use codebook to infer columns corresponding to the same question
    if use_codebook:
        codebook = codebook[codebook.year.notna()]
        codebook['year'] = codebook['year'].astype(int)

        # Get all question pairs
        codebook["pairs"] = codebook['var_name'].apply(get_pairs)
        if custom_pairs is not None:
            codebook = codebook[codebook["pairs"].isin(custom_pairs)]
        var_name_index = {}
        for i, (_, x) in enumerate(codebook.groupby("pairs", sort=False)):
            for _, row in x.iterrows():
                var_name_index[row['var_name']] = (row.year, i)

        if not isdir(save_inter_path):
            makedirs(save_inter_path)

        with open(var_name_index_path, 'wb') as file:
            pickle.dump(var_name_index, file, protocol=pickle.HIGHEST_PROTOCOL)

    #  Read from files which columns correspond to the same question
    else:

        if isfile(pair_path):
            pairs = pd.read_csv(pair_path)

        else:
            print(f'File with pair names not found at: {pair_path}')
            print('Call to_sequence with use_codebook=True to create file')

        if isdir(var_name_index_path):
            with open(var_name_index_path, 'rb') as file:
                var_name_index = pickle.load(file)
        else:
            print(f'File with variable indices not found at: {var_name_index_path}')
            print('Call to_sequences with use_codebook=True to create file')

    pids = df['nomem_encr']
    # Create dict of {pid: {year: sequences}}
    N = len(codebook['pairs'].unique())
    seq = {pid: {year: [101]*N for year in codebook['year'].unique()} for pid in pids}    # 101 is UNK
    for column, (year, idx) in var_name_index.items():
        if column not in df:    # If column isn't present, we skip it (defaulting to 101)
            continue
        for pid, val in zip(pids, df[column]):
            seq[pid][year][idx] = val

    return seq

def get_pairs(var_name):
    if var_name.startswith('c'):
        if var_name.endswith('_m'):
            return var_name[:2] + var_name[-1]
        else:
            return var_name[:2] + var_name[-3:]
    else:
        return var_name.split("_")[0]
