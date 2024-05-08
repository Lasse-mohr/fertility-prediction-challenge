""" This file contains everything needed to train the autoencoder """
import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader

from model.autoencoder import AutoEncoder

from model.utils import get_device


def train_autoencoder(sequences: dict, hidden_dim=512, encoding_size=64,
                      batch_size=64, num_epochs_autoencoder=50,
                      learning_rate_autoencoder=0.0001,
                      loss_fun='MSE') -> nn.Module:

    device = get_device()

    # compute the len of each survey wave (assuming they have uniform size)
    first_sequence = next(iter(sequences.values()))
    first_survey_wave = next(iter(first_sequence.values()))
    seq_len = len(first_survey_wave)

    # determine the vocabulary size
    vocab_size = len(set(
        [elem for _, sequence in sequences.items()
         for _, item in sequence.items() for elem in item]
    ))

    # set up the data for the autoencoder
    # this is a temporary way to do it. It will be replaced by a data subclass
    # that passes the year along with the survey response
    autoencoder_data = torch.Tensor([
                                wave_response
                                for _, wave_responses in sequences.items()
                                for _, wave_response in wave_responses.items()
                        ]).to(torch.int64)

    train_dataloader = DataLoader(autoencoder_data, batch_size=batch_size,
                                  shuffle=True)
    autoencoder = AutoEncoder(vocab_size=vocab_size, embedding_size=hidden_dim,
                              encoding_size=encoding_size,
                              sequence_len=seq_len).to(device)

    if loss_fun == 'MSE':
        error = nn.MSELoss()
    else:
        print(f'Unrecognized loss function: {loss_fun}')
        print('Continuing with MSELoss')
        error = nn.MSELoss()

    optimizer = optim.Adam(autoencoder.parameters())

    autoencoder.train()
    for epoch in range(num_epochs_autoencoder):
        for batch in train_dataloader:
            batch = batch.to(device)

            optimizer.zero_grad()
            xx = autoencoder(batch)
            loss = error(xx, autoencoder.embedding(batch))

            loss.backward()

            optimizer.step()
        print(f'epoch {epoch} \t Loss: {loss.item():.4g}')

    return autoencoder
