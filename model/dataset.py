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


class FinetuningDataset(Dataset):
    def __init__(self, sequences: dict, targets: dict):
        """ We expect sequences to be pre-encoded and structered accordingly here"""
        self.keys = list(sequences.keys())

        targets = targets.set_index(keys='nomem_encr').squeeze().to_dict()
        for person_id, target in targets.items():
            targets[person_id] = torch.tensor(target)

        self.sequences = sequences
        self.targets = targets

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        person_id = self.keys[idx]

        return self.sequences[person_id], self.targets[person_id]
    

class PredictionDataset(Dataset):
    def __init__(self, sequences: dict):
        """ We expect sequences to be pre-encoded and structered accordingly here"""
        self.keys = list(sequences.keys())

        self.sequences = sequences

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        person_id = self.keys[idx]

        return self.sequences[person_id], -1
    