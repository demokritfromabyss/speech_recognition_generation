import torch
import torch.nn as nn

from melbanks import LogMelFilterBanks


class SmallKeywordCNN(nn.Module):
    def __init__(self, n_mels: int = 80, groups: int = 1):
        super().__init__()
        if n_mels % groups != 0:
            raise ValueError(f"n_mels ({n_mels}) must be divisible by groups ({groups})")

        self.features = LogMelFilterBanks(n_mels=n_mels)
        self.encoder = nn.Sequential(
            nn.Conv1d(n_mels, 64, kernel_size=5, padding=2, groups=groups),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.classifier = nn.Linear(128, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.encoder(x)
        x = x.squeeze(-1)
        return self.classifier(x)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
