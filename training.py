"""
This is an example script to train your model given the (cleaned) input dataset.

This script will not be run on the holdout data,
but the resulting model model.joblib will be applied to the holdout data.

It is important to document your training steps here, including seed,
number of folds, model, et cetera
"""
import torch
import torch.nn as nn
import torch.optim as optim
import random
import pandas as pd
import joblib
import numpy as np
import submission

from model.utils import get_device
from model.submission_utils import PreFerPredictor, DataProcessor


def train_save_model(cleaned_df: pd.DataFrame, outcome_df: pd.DataFrame, codebook_df: pd.DataFrame, importance_df: pd.DataFrame):
    """
    Trains a model using the cleaned dataframe and saves the model to a file.

    Parameters:
    cleaned_df (pd.DataFrame): The cleaned data from clean_df function to be used for training the model.
    outcome_df (pd.DataFrame): The data with the outcome variable (e.g., from PreFer_train_outcome.csv or PreFer_fake_outcome.csv).
    """
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)

    # This script contains a bare minimum working example
    device = get_device()
    # 1. Convert Data
    data_processor = DataProcessor(cleaned_df=cleaned_df,
                                   codebook=codebook_df,
                                   col_importance=importance_df,
                                   n_cols=150,
                                   )
    data_processor.df_to_sequences(use_codebook=True)
    data_processor.prepare_traindata(batch_size=16, outcomes=outcome_df)
    print("(TRAINING) Data Ready")

    # 2. Setup model training
    model = PreFerPredictor().to(device)
    # Define the loss function
    NUM_EPOCHS = 13

    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.Tensor([5.]).to(device))
    optimizer = torch.optim.RAdam(
        model.parameters(), lr=1e-2, weight_decay=1e-2, decoupled_weight_decay=True)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=NUM_EPOCHS * len(data_processor.full_dataloader), eta_min=8e-4, last_epoch=-1)
    # EMA Weight Averaging
    avg_fn = optim.swa_utils.get_ema_avg_fn(0.99)
    avg_model = optim.swa_utils.AveragedModel(
        model, avg_fn=avg_fn, use_buffers=False)
    avg_start = 3

    print("(TRAINING) Model is initialized")

    # 3. Training loop
    model.train()
    avg_model.train()
    for epoch in range(NUM_EPOCHS):
        loss_per_epoch = []
        for batch in data_processor.full_dataloader:
            optimizer.zero_grad()
            inputs, labels = batch
            labels = labels.to(torch.float).to(device)
            input_year, input_seq = inputs
            # Model
            output = model(input_year=input_year,
                           input_seq=input_seq, labels=labels)
            # Loss
            loss = loss_fn(output, labels)
            loss_per_epoch.append(loss.detach().cpu().numpy())
            loss.backward()
            optimizer.step()
            scheduler.step()

            # EMA averaging starts only after 3 epochs
            if epoch > avg_start:
                avg_model.update_parameters(model)
        print(f"Epoch {epoch} -- loss: {np.mean(loss_per_epoch):.4f}")
    print("(TRAINING) Model training is finished!")

    avg_model.eval()
    # Save the averaged model
    joblib.dump(avg_model.get_submodule("module"), "model.joblib")
    print("(TRAINING) Model training is saved!")


if __name__ == "__main__":
    # df = pd.read_csv("training_data/PreFer_train_data.csv")
    # outcome_df = pd.read_csv("training_data/PreFer_train_outcome.csv")
    df = pd.read_csv(
        "training_data/PreFer_train_data.csv", low_memory=False)
    outcome_df = pd.read_csv("training_data/PreFer_train_outcome.csv")
    codebook_df = pd.read_csv("codebooks/PreFer_codebook.csv")
    importance_path = "features_importance_all.csv"

    importance_df = pd.read_csv(importance_path)
    ##########################################
    # GS Temporary fix
    cleaned_df = submission.clean_df(
        df=df, codebook=codebook_df, col_importance=importance_df)
    ###########################################
    train_save_model(cleaned_df, outcome_df, codebook_df, importance_df)
