import pandas as pd
import numpy as np

from encoding.numeric_and_date import ToQuantileTransformer

#filter data for baselines

def with_outcome(df_x, df_y):
    mask = (df_y['new_child'].values == 1) + (df_y['new_child'].values == 0)
    return df_x[mask], df_y[mask]

def make_categorical(data):
    # Convert object columns to 'category' dtype
    for col in data.select_dtypes(include=['object']).columns:
        data[col] = data[col].astype('category')
    return data

def drop_columns(data, threshold=0.1):
    """
    Drop columns with missing values exceeding the threshold
    """
    missing_percentage = data.isnull().sum() / len(data) 
    columns_to_keep = missing_percentage[missing_percentage < threshold].index
    data_cleaned = data[columns_to_keep]
    return data_cleaned


def quantile_dates_and_numeric(
    df_x,
    path_for_PreFer_folder,
      ):
    """
    Uses the codebook to approximate which columns to transform into quantiles.
    Later, we might want to infer dtypes ourselves, but this will work for now.
    """

    # infer which cols to transform
    codebook = pd.read_csv(path_for_PreFer_folder +
                           'codebooks/PreFer_codebook.csv')

    # find cols with right dtype which are in training data
    datetime_bool = codebook['type_var'] == 'date or time'
    in_training_bool = codebook['dataset'] == 'PreFer_train_data.csv'
    columns_bool = (datetime_bool ) & in_training_bool

    # get their variable names
    columns = codebook.loc[columns_bool, 'var_name']

    # transform or drop the selected columns while leaving the rest unaltered
    quantile_trans = ToQuantileTransformer(columns=columns)
    quantile_trans.fit(df_x)
    df_x = quantile_trans.transform(df_x)

    return df_x


def remove_question_type(df_x):
    """
    Remove columns with data type 'date or time'
    """
    codebook = pd.read_csv('codebooks/PreFer_codebook.csv')
    date_cols = codebook[codebook['type_var'] == 'date or time2']['var_name']
    open_ended_cols = codebook[codebook['type_var'] == 'response to open-ended question']['var_name']
    remove_question_type = set(date_cols).union(set(open_ended_cols))
    intersection = set(remove_question_type).intersection(set(df_x.columns))
    #remove nomem_encr
    if 'nomem_encr' in intersection:
        intersection.remove('nomem_encr')
    return df_x.drop(columns=intersection)


def save_data(df_x,df_y, path='training_data/',fname='louis'):

    print(np.sum(df_x['nomem_encr'] == df_y['nomem_encr']) == len(df_x))
    #df_x.drop('nomem_encr', axis=1, inplace=True)
    df_y.drop('nomem_encr', axis=1, inplace=True)
    df_x.drop('outcome_available', axis=1, inplace=True)

    df_x.to_csv(path+f'PreFer_train_data_{fname}.csv', index=False)
    df_y.to_csv(path+f'PreFer_train_outcome_{fname}.csv', index=False)
    return None


if __name__ == '__main__':
    path_for_PreFer_folder = ''
    threshold = 0.8
    output_name = 'louis'
    df_x_ini = pd.read_csv(
        path_for_PreFer_folder + 'training_data/PreFer_train_data.csv',
        low_memory=False,
    )

    df_y_ini = pd.read_csv(
        path_for_PreFer_folder + 'training_data/PreFer_train_outcome.csv',
        low_memory=False,
    )

    for threshold in np.linspace(0.005, 0.5, 20):
        df_x, df_y = with_outcome(df_x_ini, df_y_ini)
        df_x = make_categorical(df_x)
        df_x = quantile_dates_and_numeric(df_x, path_for_PreFer_folder)
        df_x = drop_columns(df_x, threshold=threshold)
        df_x = remove_question_type(df_x)
        print(df_x.columns)
        save_data(df_x,df_y, path='training_data/',fname=f'{output_name}_{threshold:.2f}')

