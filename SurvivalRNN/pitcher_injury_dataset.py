import numpy as np
from torch.utils.data import Dataset

class PitcherInjuryDataset(Dataset):
    """
    Synthetic dataset for pitcher injury prediction.
    Each sample is a sequence (variable-length) of feature vectors.
    - features: a (seq_length, input_dim) array.
    - target: a (seq_length,) vector with a single "1" at the event time if injury occurred,
              or all zeros if the sample is censored.
    - mask: a (seq_length,) vector with 1 for valid time steps and 0 for padded steps.
    """
    def __init__(self, X, T, E):
        self.samples = []

        num_samples = len(X)
        for i in range(num_samples):
            seq_length = X[i].size()[0]
            features = X[i]
            event_occurred = E[i]

            target = np.zeros(seq_length, dtype=np.float32)
            if event_occurred:
                event_time = T[i]
                target[event_time - 1] = 1.0
            # Mask: all ones for valid time steps.
            mask = np.ones(seq_length, dtype=np.float32)
            self.samples.append((features, target, mask))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]