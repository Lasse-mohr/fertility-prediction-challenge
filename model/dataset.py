import torch
from torch.utils.data import Dataset


class PretrainingDataset(Dataset):
    def __init__(self, sequences: dict):
        self.years = self.split_to_years(sequences)

    def split_to_years(self, sequences):
        years = {str.zfill(str(i), 2): [] for i in range(7, 21)}
        for year in years:
            for pid, values in sequences.items():
                years[year].append(torch.tensor(values[year]))

        return years

    def __len__(self):
        return len(self.years), len(self.years['07'])

    def __getitem__(self, idx):
        year, idx = idx
        return self.years[year][idx]
    
    def __iter__(self):
        for year in self.years:
            yield self.years[year]