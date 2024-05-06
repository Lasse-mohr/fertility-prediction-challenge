import torch


class AutoEncoder(torch.nn.Module):
    def __init__(self, num_embeddings) -> None:
        super().__init__()

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
            torch.nn.Linear(32, 16),
        )

        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(16, 32),
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
