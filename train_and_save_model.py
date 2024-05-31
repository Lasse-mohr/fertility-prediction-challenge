import os
import pandas as pd


import submission
import training

# loading data (predictors)
train = pd.read_csv(
    "/Users/lmmi/PreFer/training_data/PreFer_train_data.csv", low_memory=False)
# loading the outcome
outcome = pd.read_csv(
    "/Users/lmmi/PreFer/training_data/PreFer_train_outcome.csv")

# preprocessing the data
train_cleaned = submission.clean_df(train)

# training and saving the model
training.train_save_model(train_cleaned, outcome)
