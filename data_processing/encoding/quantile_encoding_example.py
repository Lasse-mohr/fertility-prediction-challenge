import pandas as pd
from to_quantile import ToQuantileTransformer


def quantile_dates_and_numeric(
    df,
    path_for_PreFer_folder,
    static_columns=[
                    'nomem_encr',
                    'new_child',
                    'outcome_available',
                     ]
      ):
    """
    Uses the codebook to approximate which columns to transform into quantiles.
    Later, we might want to infer dtypes ourselves, but this will work for now.
    """

    # infer which cols to transform
    codebook = pd.read_csv(path_for_PreFer_folder +
                           '/codebooks/PreFer_codebook.csv')

    # find cols with right dtype which are in training data
    datetime_bool = codebook['type_var'] == 'date or time'
    numeric_bool = codebook['type_var'] == 'numeric'
    in_training_bool = codebook['dataset'] == 'PreFer_train_data.csv'
    columns_bool = (datetime_bool | numeric_bool) & in_training_bool

    # get their variable names
    columns = codebook.loc[columns_bool, 'var_name']

    # remove static columns
    static_columns_index = columns.isin(static_columns)
    columns = columns[~static_columns_index]

    # transform or drop the selected columns while leaving the rest unaltered
    quantile_trans = ToQuantileTransformer(columns=columns)
    quantile_trans.fit(df)
    df = quantile_trans.transform(df)

    return df


if __name__ == '__main__':
    # load the data

    path_for_PreFer_folder = ''

    df = pd.read_csv(path_for_PreFer_folder +
                     '/training_data/PreFer_train_data.csv',
                     low_memory=False)
    # transform all date and numeric columns to quantiles
    # (drop cols that could not be transformed)
    df = quantile_dates_and_numeric(
        df=df,
        path_for_PreFer_folder=path_for_PreFer_folder
    )
