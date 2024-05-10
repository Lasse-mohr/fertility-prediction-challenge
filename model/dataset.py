import torch
from torch.utils.data import Dataset


class PretrainingDataset(Dataset):
    def __init__(self, sequences: dict):
        self.samples = []
        for person_id, years_data in sequences.items():
            for year, sequence in years_data.items():
                self.samples.append((year-2007, torch.tensor(sequence)))
                
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

    def get_seq_len(self):
        year, sequence = self.samples[0]
        return len(sequence)
    
    def get_vocab_size(self):
        current_max = 0
        for year, sequence in self.samples:
            current_max = max(current_max, sequence.max().item())
        return current_max + 1

