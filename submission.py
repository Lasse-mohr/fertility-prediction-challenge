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
from data_processing.pipeline import encoding_pipeline, get_generic_name
from data_processing.encoding.categorical import CategoricalTransformer
from data_processing.encoding.numeric_and_date import ToQuantileTransformer


# Sorted IDs of Questions (based on the xboost impargtance measure)
QUESTIONS_TO_USE = ['cf130' 'cf393' 'cf394' 'ca072' 'cfm' 'cf029' 'cf031' 'cr172' 'cf396'
                    'belbezig' 'cs103' 'cp199' 'cf388' 'cf390' 'cp018' 'cs527' 'cs523'
                    'cp025' 'cf025' 'cv153' 'cs387' 'cw600' 'cs176' 'cr116' 'cv308' 'cs571'
                    'cf128' 'cf456' 'cf012' 'cp059' 'cs474' 'cf457' 'cr093' 'cs524'
                    'birthyear' 'cp105' 'cf251' 'ch148' 'cs277' 'cf166' 'ca066' 'cw377'
                    'cf028' 'cr178' 'ca075' 'cr133' 'cf249' 'cr142' 'cw052' 'cf026'
                    'nettoink' 'cd028' 'ci002' 'cw428' 'cw412' 'cf397' 'cd014' 'cs415'
                    'cv117' 'cp052' 'cd034' 'cv165' 'cs268' 'cs557' 'cr134' 'cr042' 'cp042'
                    'cw045' 'brutoink' 'cp036' 'cw003' 'ch249' 'cp160' 'cs001' 'cv114'
                    'burgstat' 'ci005' 'cv024' 'cp019' 'cv258' 'cp111' 'cs337' 'cp149'
                    'cs199' 'cs288' 'ca001' 'cv148' 'cr081' 'cw611' 'cv303' 'cs495' 'cw576'
                    'cs253' 'netinc' 'cd033' 'cp039' 'cr038' 'cs252' 'ch206' 'cv160' 'cp037'
                    'cd010' 'cp076' 'cs102' 'cv279' 'cf163' 'ci020' 'cv129' 'cv101' 'ch219'
                    'cs581' 'cp072' 'ci326' 'cr117' 'cw504' 'cv015' 'cs227' 'cf004' 'oplcat'
                    'ch130' 'cs425' 'oplzon' 'cv230' 'cr083' 'ci337' 'cs231' 'ch107' 'cv247'
                    'cs485' 'cp103' 'cs386' 'cv115' 'cv041' 'cp047' 'cw517' 'brutohh' 'cp027'
                    'cp016' 'cf143' 'cf002' 'ch256' 'cv123' 'ci006' 'cd079' 'ch165' 'cp165'
                    'cw002' 'ch002' 'cv289' 'cr179']


# Function to clean the dataframe
def clean_df(df: pd.DataFrame,
             background_df: pd.DataFrame = None,
             col_importances: pd.DataFrame = None,
             codebook: pd.DataFrame = None):
    """
    arguments:
        df (pd.DataFrame): the questionaire feature dataframe
        background_df (pd.DataFrame): background dataframe
        codebook (pd.DataFrame): preloaded codebook 
    """

    data = df.copy()

    # Process According to Type
    if codebook is None:
        raise NotImplementedError("Provide path to codebook")
    else:
        use_codebook = True
        # Select only questions with yearly component
        codebook = codebook[codebook.year.notna()]
        # Get all question pairs
        if QUESTIONS_TO_USE is not None:
            codebook["pairs"] = codebook['var_name'].apply(get_generic_name)
            codebook = codebook[codebook["pairs"].isin(QUESTIONS_TO_USE)]

        # Get relevant columns
        categorical_columns = codebook[codebook.type_var ==
                                       'categorical'].var_name
        quantile_columns = codebook[((codebook.type_var == 'numeric') | (
            codebook.type_var == 'date or time'))].var_name

    # Encode categorical columns
    categorical_transformer = CategoricalTransformer()
    categorical_transformer.fit(codebook, use_codebook=use_codebook)

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


def predict_outcomes(df, background_df=None,  model_path="model.joblib",
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
    codebook = pd.read_csv(codebook_path)
    col_importance = pd.read_csv(importance_path)
    cleaned_df = clean_df(df=df, codebook=codebook)
    #################################

    data_processor = DataProcessor(
        cleaned_df=None,
        codebook=codebook,
        col_importance=col_importance,
        n_cols=150)
    data_processor.prepare_predictdata(
        df=cleaned_df, batch_size=16, use_codebook=True)
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
