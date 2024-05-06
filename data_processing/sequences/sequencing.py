import pandas as pd


def to_sequences(df, summary):
    core_questions = summary[summary.var_names.str.startswith('c')].reset_index()

    q_to_index = {}
    for sequence_index, question in core_questions.iterrows():
        var_names = question.var_names.split(";")
        which_waves = question.which_waves.split(",") if pd.notna(question.which_waves) else []
        for var_name, year in zip(var_names, which_waves):
            q_to_index[var_name.strip()] = sequence_index

    pids = df['nomem_encr'] 
    seq = {pid: {str.zfill(str(i), 2): [101]*len(core_questions) for i in range(7, 21)} for pid in pids}    # 101 is UNK
    for column, idx in q_to_index.items():
        year = column[2:4]
        question = df[column]
        for pid, val in zip(pids, question):
            seq[pid][year][idx] = val

    return seq
