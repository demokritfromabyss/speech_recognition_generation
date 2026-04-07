import torch
import torchaudio

from melbanks import LogMelFilterBanks


if __name__ == "__main__":
    signal = torch.randn(1, 16000)

    ref = torchaudio.transforms.MelSpectrogram(
        sample_rate=16000,
        n_fft=400,
        hop_length=160,
        n_mels=80,
        power=2.0,
    )(signal)
    ref = torch.log(ref + 1e-6)

    ours = LogMelFilterBanks(
        samplerate=16000,
        n_fft=400,
        hop_length=160,
        n_mels=80,
        power=2.0,
    )(signal)

    print("ref shape:", tuple(ref.shape))
    print("ours shape:", tuple(ours.shape))
    print("max abs diff:", (ref - ours).abs().max().item())
    print("allclose(atol=1e-5):", torch.allclose(ref, ours, atol=1e-5))
