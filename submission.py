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


from sklearn.base import BaseEstimator, TransformerMixin
from dateutil.parser import parse, ParserError
import warnings

class ToQuantileTransformer(BaseEstimator, TransformerMixin):

    """
    Transformer for converting specified DataFrame columns into quantiles.

    This transformer handles numeric and datetime columns, transforming them based
    on their own distribution into a specified number of quantiles. Columns with a
    single unique value are transformed to quantile 0. The class can identify and
    ignore columns that are neither numeric nor datetime.

    NaN values are encoded as -1. 

    Parameters:
    -----------
    columns : list of str
        List of column names in the DataFrame to be transformed.
    n_bins : int, default=100
        Number of quantiles to use for transforming the data.

    Attributes:
    -----------
    cols : list of str
        Stores the column names to be transformed.
    n_bins : int
        Stores the number of bins for quantization.
    quantile_bins : dict
        Dictionary to hold the quantile boundaries for each column.
    dtypes : dict
        Dictionary to store the detected data type of each column ('datetime' or 'numeric').
    bad_cols : set
        Set of column names that cannot be classified as either numeric or datetime.

    Methods:
    --------
    fit(X, y=None):
        Fits the transformer to the data, identifying column types and calculating quantiles.
    transform(X):
        Transforms the DataFrame's specified columns into quantiles.

    """

    def __init__(self, columns, n_bins = 100):
        # List of column names to be transformed
        self.cols = list(columns)
        self.n_bins = n_bins

        # the following dictionaries and sets are populated during training.
        self.quantile_bins = {} # the bins that define quantile boundaries
        self.dtypes = {} # To store infered dtypes of columns (either 'datetime' or 'numeric')
        self.bad_cols = set() # Keeps track of columns that were neither datetime nor numeric.

    def is_datetime(self, ser: pd.Series):
        """
        Check if the Series is of datetime type.

        Parameters:
        -----------
        ser : pd.Series
            Pandas Series to check for datetime type.

        Returns:
        --------
        bool: True if series is datetime, otherwise False.
        """
        is_datetime = False
        try:
            with warnings.catch_warnings():
                ser.apply(parse)
            is_datetime = True
        except (ValueError, ParserError, TypeError):
            pass
        return is_datetime

    def is_numeric(self, ser: pd.Series):
        """
        Check if the Series is numeric.

        Parameters:
        -----------
        ser : pd.Series
            Pandas Series to check for numeric type.

        Returns:
        --------
        bool
            True if series is numeric, otherwise False.
        """
        is_numeric = False
        try:
            with warnings.catch_warnings():
                ser.astype('float')
            is_numeric = True
        except (ValueError, TypeError):
            pass
        return is_numeric

    def check_dtypes(self, X: pd.DataFrame):
        """
        Determine and store the data types of the specified columns in the DataFrame.

        Parameters:
        -----------
        X : pd.DataFrame
            DataFrame containing the columns to check.
        """
        self.check_dtypes = {}

        for col in self.cols:
            ser = X[col]
            ser = ser.dropna()

            if self.is_datetime(ser):
                self.dtypes[col] = 'datetime'
            elif self.is_numeric(ser):
                self.dtypes[col] = 'numeric'
            else:
                self.bad_cols.add(col) #the column will be dropped from the dataframe

    def fit(self, X, y=None):
        """
        Fit the transformer to the data by determining data types and calculating quantiles.
        Does not return anything.

        Parameters:
        -----------
        X : pd.DataFrame
            DataFrame containing the data to fit.
        y : Ignored
            This parameter is not used but is included for compatibility with scikit-learn.

        """
        self.check_dtypes(X) #assigns col dtype to self.dtypes

        for col, dtype in self.dtypes.items():
            series = X[col]
            series = series[series.notna()]

            if series.shape[0] > 0:
                if dtype == 'datetime': # convert to numpy compatible date format
                    series = series.apply(parse)

                elif dtype == 'numeric': # if dtype is numeric no conversion is needed
                    pass

                self.quantile_bins[col] = np.quantile(
                                        series,
                                        np.linspace(0, 1, num=self.n_bins + 1)
                                        )


    def transform(self, X):
        """
        Transform the DataFrame's columns into quantiles based on the fitted data.

        Parameters:
        -----------
        X : pd.DataFrame
            DataFrame to transform.

        Returns:
        --------
        X_transformed : pd.DataFrame
            DataFrame with specified columns transformed to their quantile indices.
        """
        # subselect columns to transform
        X_transformed = X.drop(columns=list(self.bad_cols)).copy()
        # exclude columns where dtype could not be inferred
        X_cols = set(X_transformed.columns)

        # Apply the quantile bins to transform to quantile indices
        for col, quantile_map in self.quantile_bins.items():
            if col in X_cols:
                series = X_transformed[col]
                not_na_mask = ~series.isna()
                series = series[not_na_mask]

                dtype = self.dtypes[col]

                # check if all current data is na
                if series.shape[0] > 0:
                    if dtype == 'datetime':
                        series = series.apply(parse)

                    quantile_values = np.searchsorted(
                                            quantile_map,
                                            series,
                                            side='left') / (self.n_bins)

                    X_transformed[col] = np.nan
                    X_transformed.loc[not_na_mask, col] = quantile_values

                else:
                    X_transformed[col] = pd.Series(
                        [np.nan]*X_transformed.shape[0], dtype='float'
                    )

        return X_transformed
    
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
