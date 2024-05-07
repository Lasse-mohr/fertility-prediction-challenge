import torch


class AutoEncoder(torch.nn.Module):
    def __init__(self, num_embeddings, encoding_dim=16) -> None:
        super().__init__()
        self.encoding_dim = encoding_dim

        self.embed = torch.nn.Embedding(num_embeddings, 512)

        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(512, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, encoding_dim),
        )

        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(encoding_dim, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 512),
        )

    def forward(self, x):
        x = self.embed(x)
        x = self.encoder(x)
        x = self.decoder(x)
        return x

    def get_loss(self, x):
        x_hat = self.forward(x)
        return torch.nn.functional.mse_loss(x_hat, self.embed(x))

    def embed_and_encode(self, x):
        x = self.embed(x)
        return self.encode(x)

    def get_encoding_dim(self):
        return self.encoding_dim
