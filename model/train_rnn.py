import pandas as pd

import torch
import torch.optim as optim

import torch.nn as nn

from torch.utils.data import DataLoader
from torch.utils.data import Dataset

from model.rnn import GRUDecoder

from model.utils import get_device


class SequencesWithTarget(Dataset):
    def __init__(self, sequences: dict, targets: pd.DataFrame):
        self.sequences = sequences
        self.target = targets.set_index(keys='nomem_encr').squeeze().to_dict()
        self.keys = list(sequences.keys())

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, index):
        person_id = self.keys[index]

        target = self.target[person_id]
        sequence = self.sequences[person_id]

        return target, sequence


def train_rnn(sequences: dict, targets: pd.Series,
              autoencoder: nn.Module, hidden_size=128,
              num_epochs=50, learning_rate=0.001, encoding_size=16,
              max_seq_len=14, batch_size=64,
              ):

    device = get_device()

    # format the data
    data = {person_id:
            autoencoder.encode(
                torch.Tensor(
                    list(wave_responses.values())
                ).to(device))
            for person_id, wave_responses in sequences.items()}

    # create the dataloader
    dataset = SequencesWithTarget(data, targets=targets)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = GRUDecoder(
        input_size=encoding_size,
        hidden_size=hidden_size,
        max_seq_len=max_seq_len
    ).to(device)

    # assume that all 14 years are observed for everyone
    single_mask = torch.BoolTensor([True]*14).to(device)

    # Define loss function and optimizer for RNN
    loss = torch.nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    # Training loop

    model.train()
    for epoch in range(num_epochs):
        print(epoch)
        running_loss = 0

        for batch in dataloader:
            labels, inputs = batch
            labels = labels.to(torch.float).to(device)
            inputs = inputs.to(device)

            optimizer.zero_grad()

            # not correct masking
            mask = torch.stack([single_mask]*len(labels))

            xx = model(inputs, mask)
            outputs = torch.nn.functional.sigmoid(xx)

            loss = loss(torch.flatten(outputs), labels)

            # loss.backward(retain_graph=True)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)

        # Calculate average loss for the epoch
        epoch_loss = running_loss / len(dataloader.dataset)

        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}")

    return model
