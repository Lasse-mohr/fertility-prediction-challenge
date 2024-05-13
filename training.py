"""
This is an example script to train your model given the (cleaned) input dataset.

This script will not be run on the holdout data,
but the resulting model model.joblib will be applied to the holdout data.

It is important to document your training steps here, including seed,
number of folds, model, et cetera
"""
import pandas as pd
from sklearn.linear_model import LogisticRegression
import joblib
import xgboost as xgb
import submission

def train_save_model(cleaned_df, outcome_df):
    """
    Trains a model using the cleaned dataframe and saves the model to a file.

    Parameters:
    cleaned_df (pd.DataFrame): The cleaned data from clean_df function to be used for training the model.
    outcome_df (pd.DataFrame): The data with the outcome variable (e.g., from PreFer_train_outcome.csv or PreFer_fake_outcome.csv).
    """

    ## This script contains a bare minimum working example

    # Combine cleaned_df and outcome_df
    model_df = pd.merge(cleaned_df, outcome_df, on="nomem_encr")

    # Filter cases for whom the outcome is not available
    model_df = model_df[~model_df['new_child'].isna()]

    # Logistic regression model
    scale_pos_weight =  model_df['new_child'].value_counts()[0] / model_df['new_child'].value_counts()[1]
    
    model = xgb.XGBClassifier(
            objective='binary:logistic', 
            use_label_encoder=False, 
            eval_metric='logloss',
            scale_pos_weight=scale_pos_weight, 
            max_delta_step=1,
            verbosity=0,
            tree_method='exact',
            learning_rate=0.1,
            subsample=0.8,
            reg_lambda = 5, #10
            reg_alpha = 0.1, #0.1
            )

    # Fit the model # model_df['new_child'] is the target variable
    model.fit(model_df.drop(['nomem_encr', 'new_child'], axis=1), model_df['new_child'])
  
    # print model fitting results
    print(model)
    
    # Save the model
    joblib.dump(model, "model.joblib")

if __name__ == "__main__":
    df = pd.read_csv("training_data/PreFer_train_data.csv")
    outcome_df = pd.read_csv("training_data/PreFer_train_outcome.csv")

    cleaned_df = submission.clean_df(df)
    train_save_model(cleaned_df, outcome_df)