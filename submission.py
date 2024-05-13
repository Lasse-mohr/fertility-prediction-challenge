"""
This is an example script to generate the outcome variable given the input dataset.

This script should be modified to prepare your own submission that predicts
the outcome for the benchmark challenge by changing the clean_df and predict_outcomes function.

The predict_outcomes function takes a Pandas data frame. The return value must
be a data frame with two columns: nomem_encr and outcome. The nomem_encr column
should contain the nomem_encr column from the input data frame. The outcome
column should contain the predicted outcome for each nomem_encr. The outcome
should be 0 (no child) or 1 (having a child).

clean_df should be used to clean (preprocess) the data.

run.py can be used to test your submission.
"""

# List your libraries and modules here. Don't forget to update environment.yml!
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
import joblib

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer
from sklearn.base import BaseEstimator, TransformerMixin


from data_processing.encoding.numeric_and_date import ToQuantileTransformer

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
    
    
# Function to clean the dataframe
def clean_df(df, background_df=None):
    """
    arguments:
        df (pd.DataFrame): the questionaire feature dataframe
    """
    path_for_PreFer_folder = ''
    threshold = 0.4
    #df_x, df_y = with_outcome(df_x_ini, df_y_ini)
    df_x = make_categorical(df)
    df_x = quantile_dates_and_numeric(df_x, path_for_PreFer_folder)
    #df_x = drop_columns(df_x, threshold=threshold)
    #columms = df_x.columns
    df_x = remove_question_type(df_x)
    #print(df_x.columns)
    #save_data(df_x, path='training_data/',fname=f'{output_name}_{threshold:.2f}')

    return df_x


def predict_outcomes(df, background_df=None, model_path="model.joblib"):
    """Generate predictions using the saved model and the input dataframe.

    The predict_outcomes function accepts a Pandas DataFrame as an argument
    and returns a new DataFrame with two columns: nomem_encr and
    prediction. The nomem_encr column in the new DataFrame replicates the
    corresponding column from the input DataFrame. The prediction
    column contains predictions for each corresponding nomem_encr. Each
    prediction is represented as a binary value: '0' indicates that the
    individual did not have a child during 2021-2023, while '1' implies that
    they did.

    Parameters:
    df (pd.DataFrame): The input dataframe for which predictions are to be made.
    background_df (pd.DataFrame): The background dataframe for which predictions are to be made.
    model_path (str): The path to the saved model file (which is the output of training.py).

    Returns:
    pd.DataFrame: A dataframe containing the identifiers and their corresponding predictions.
    """

    ## This script contains a bare minimum working example
    if "nomem_encr" not in df.columns:
        print("The identifier variable 'nomem_encr' should be in the dataset")

    # Load the model
    model = joblib.load(model_path)

    # Preprocess the fake / holdout data
    df = clean_df(df, background_df)

    # Exclude the variable nomem_encr if this variable is NOT in your model
    vars_without_id = df.columns[df.columns != 'nomem_encr']

    # Generate predictions from model, should be 0 (no child) or 1 (had child)
    predictions = model.predict(df[vars_without_id])

    # Output file should be DataFrame with two columns, nomem_encr and predictions
    df_predict = pd.DataFrame(
        {"nomem_encr": df["nomem_encr"], "prediction": predictions}
    )

    # Return only dataset with predictions and identifier
    return df_predict
