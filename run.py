"""
This script calls submission.py. Add your method to submission.py to run your
prediction method.

To test your submission use the following command:

python run.py

For example:

python run.py PreFer_fake_data.csv PreFer_fake_background_data.csv

Optionally, you can use the score function to calculate evaluation scores given 
your predictions and the ground truth within the training dataset.

"""

import sys
import argparse
import pandas as pd
import submission

parser = argparse.ArgumentParser(description="Process data.")

parser.add_argument("data_path", help="Path to data data CSV file.", default='training_data/PreFer_fake_data.csv')
parser.add_argument(
    "background_data_path", help="Path to background data data CSV file.", default='training_data/PreFer_fake_background_data.csv')
parser.add_argument(
    "--output", help="Path to prediction output CSV file.",default='training_data/PreFer_fake_outcome_.csv')

args = parser.parse_args()


def predict(data_path, background_data_path, output):
    """Predict Score (evaluate) the predictions and write the metrics.

    This function takes the path to an data CSV file containing the data data.
    It calls submission.py clean_df and predict_outcomes writes the predictions
    to a new output CSV file.

    This function should not be modified.
    data_path: str: path to the data CSV file
    background_data_path: str: path to the background data CSV file
    output: str: path to the output CSV file
    data_path = 'training_data/PreFer_fake_data.csv'
    background_data_path = 'training_data/PreFer_fake_background_data.csv'
    output = 'training_data/PreFer_fake_outcome_.csv'
    
    """

    if output is None:
        output = sys.stdout
    data_df = pd.read_csv(
        data_path, encoding="latin-1", encoding_errors="replace", low_memory=False
    )
    background_data_df = pd.read_csv(
        background_data_path,
        encoding="latin-1",
        encoding_errors="replace",
        low_memory=False,
    )

    predictions = submission.predict_outcomes(data_df, background_data_df, model_path="model_cpu.joblib")
    assert (
        predictions.shape[1] == 2
    ), "Predictions must have two columns: nomem_encr and prediction"
    # Check for the columns, order does not matter
    assert set(predictions.columns) == set(
        ["nomem_encr", "prediction"]
    ), "Predictions must have two columns: nomem_encr and prediction"

    predictions.to_csv(output, index=False)


if __name__ == "__main__":
    args = parser.parse_args()
    print(args)
    predict(args.data_path, args.background_data_path, args.output)

