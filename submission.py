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
import joblib
from model.utils import get_device
from model.submission_utils import DataProcessor, PreFerPredictor
from data_processing.pipeline import get_generic_name
from data_processing.encoding.categorical import CategoricalTransformer
from data_processing.encoding.numeric_and_date import ToQuantileTransformer
import warnings

# Function to clean the dataframe


def clean_df(df: pd.DataFrame,
             col_importance: pd.DataFrame = None,
             codebook: pd.DataFrame = None):
    """
    arguments:
        df (pd.DataFrame): the questionaire feature dataframe
        background_df (pd.DataFrame): background dataframe
        codebook (pd.DataFrame): preloaded codebook 
    """

    data = df.copy()

    # Encode categorical columns
    categorical_transformer = CategoricalTransformer()
    categorical_transformer.fit(codebook, use_codebook=codebook is not None)

    categorical_columns = pd.read_csv('data_processing/codebook_false/encoding_pipeline/categorical_columns.csv').var_name.tolist()
    quantile_columns = pd.read_csv('data_processing/codebook_false/encoding_pipeline/quantile_columns.csv').var_name.tolist()

    data[categorical_columns] = categorical_transformer.transform(
        data[categorical_columns])

    # Encode numeric and date columns
    quantile_transformer = ToQuantileTransformer(quantile_columns)
    quantile_transformer.fit(data)
    data = quantile_transformer.transform(data)

    # Fill any nans
    #########################
    # df = data.fillna(101)
    # GS Temporary Fix
    data = data.fillna(
        {col: 101 for col in data.columns[data.dtypes.eq(float)]})
    #########################
    data = data.astype(int, errors='ignore')
    # Drop object columns (automatically filled with 101 in to_sequences)
    cleaned_data = data[data.columns[data.dtypes != 'object']]

    # we do not produce a cleaned df, instead we make all the preprocessing while converting pd.DataFrame to Torch Tensor.
    # these can be found in model.submission_utils.DataProcessor

    return cleaned_data


def predict_outcomes(df: pd.DataFrame, background_df: pd.DataFrame = None,  model_path: str = "model.joblib",
                     codebook_path: str = "codebooks/PreFer_codebook.csv",
                     importance_path: str = "features_importance_all.csv"):
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

    # This script contains a bare minimum working example
    if "nomem_encr" not in df.columns:
        print("The identifier variable 'nomem_encr' should be in the dataset")

    # Load the model
    device = get_device()
    model = joblib.load(model_path).to(device)
    model.eval()
    print("(SUBMISSION) Trained model is loaded!")
    # Preprocess the fake / holdout data
    #################################
    # GS Temporary Fix
    codebook = None#pd.read_csv(codebook_path)
    col_importance = pd.read_csv(importance_path)
    cleaned_df = clean_df(df=df, codebook=codebook,
                          col_importance=col_importance)
    #################################

    data_processor = DataProcessor(
        cleaned_df=None,
        codebook=codebook,
        col_importance=col_importance,
        n_cols=150)

    data_processor.prepare_predictdata(
        df=cleaned_df, batch_size=16, use_codebook=codebook is not None)
    print("(SUBMISSION) Data Processor is done!")

    # Generate predictions from model, should be 0 (no child) or 1 (had child)
    predictions = model.predict(data_processor.prediction_dataloader, device)
    predictions = (predictions > 0.5).astype(int)
    print("(SUBMISSION) Predictions are ready")

    # Output file should be DataFrame with two columns, nomem_encr and predictions
    df_predict = pd.DataFrame(
        {"nomem_encr": df["nomem_encr"], "prediction": predictions}
    )
    # Return only dataset with predictions and identifier
    return df_predict
