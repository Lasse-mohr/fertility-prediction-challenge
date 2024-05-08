def to_sequences(df, codebook):
    codebook = codebook[codebook.year.notna()]
    codebook['year'] = codebook['year'].astype(int)

    # Get all question pairs
    pairs = codebook['var_name'].apply(get_pairs)
    var_name_index = {}
    for i, (_, x) in enumerate(codebook.groupby(pairs, sort=False)):
        for _, row in x.iterrows():
            var_name_index[row['var_name']] = (row.year, i)

    pids = df['nomem_encr'] 
    # Create dict of {pid: {year: sequences}}
    seq = {pid: {year: [101]*(len(set(pairs))) for year in codebook['year'].unique()} for pid in pids}    # 101 is UNK
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
    
