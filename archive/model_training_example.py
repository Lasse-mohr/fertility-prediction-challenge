
import pandas as pd
from data_processing.pipeline import encoding_pipeline
from model.train_rnn import train_rnn
from model.train_autoencoder import train_autoencoder



def main(data: pd.DataFrame, targets: pd.DataFrame, codebook: pd.DataFrame):
    sequences = encoding_pipeline(data, codebook)
    print('Successfully tokenized data')

    autoencoder = train_autoencoder(sequences=sequences, hidden_dim=512,
                                    encoding_size=64, batch_size=64,
                                    num_epochs_autoencoder=50,
                                    learning_rate_autoencoder=0.0001,
                                    loss_fun='MSE'
                                   )
    print('Autoencoder successfully trained')

    _ = train_rnn(sequences=sequences, targets=targets,
                    autoencoder=autoencoder, hidden_size=128,
                    num_epochs=50, learning_rate=0.001, encoding_size=16,
                    max_seq_len=14, batch_size=64
                 )
    print('RNN successfully trained')

    print('Hurra')

if __name__ == '__main__':

    # fake data
    data = pd.read_csv('data/other_data/PreFer_fake_data.csv')
    targets = pd.read_csv('data/other_data/PreFer_fake_outcome.csv')

    # real data
    #data = pd.read_csv('data/training/PreFer_train_data.csv', nrows = 200)
    #targets = pd.read_csv('data/training/PreFer_train_outcome.csv', nrows = 200)

    codebook = pd.read_csv('data/codebooks/PreFer_codebook.csv')

    main(data=data, targets=targets, codebook=codebook)
