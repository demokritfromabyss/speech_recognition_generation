from pathlib import Path
from typing import Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchaudio.datasets import SPEECHCOMMANDS


LABELS = {"no": 0, "yes": 1}
TARGET_LEN = 16000


class YesNoSpeechCommands(Dataset):
    def __init__(self, root: str = "./data", subset: str = "training"):
        super().__init__()
        Path(root).mkdir(parents=True, exist_ok=True)
        self.ds = SPEECHCOMMANDS(root=root, subset=subset, download=True)
        self.items = []

        for i in range(len(self.ds)):
            waveform, sample_rate, label, speaker_id, utterance_number = self.ds[i]
            if sample_rate != 16000:
                raise ValueError(f"Expected sample_rate=16000, got {sample_rate}")
            if label in LABELS:
                self.items.append((waveform.squeeze(0), LABELS[label]))

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        waveform, label = self.items[idx]

        if waveform.size(0) < TARGET_LEN:
            waveform = F.pad(waveform, (0, TARGET_LEN - waveform.size(0)))
        else:
            waveform = waveform[:TARGET_LEN]

        return waveform, torch.tensor(label, dtype=torch.long)


def collate_fn(batch):
    xs, ys = zip(*batch)
    xs = torch.stack(xs, dim=0)
    ys = torch.stack(ys, dim=0)
    return xs, ys
