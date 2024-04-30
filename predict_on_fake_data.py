import pandas as pd

import submission

fake_data = pd.read_csv(
    "/Users/lmmi/fertility-prediction-challenge/PreFer_fake_data.csv",
    low_memory=False)
# loading the outcome
#fake_outcomes = pd.read_csv(
#    "/Users/lmmi/fertility_prediction_challenge/PreFer_fake_outcome.csv"
#)

print(submission.predict_outcomes(df=fake_data))
