import csv
import json
import os
import time
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import torch
import torchaudio
from thop import profile
from torch.nn.functional import cross_entropy
from torch.optim import Adam
from torch.utils.data import DataLoader

from dataset import YesNoSpeechCommands, collate_fn
from melbanks import LogMelFilterBanks
from model import SmallKeywordCNN, count_parameters


ROOT = Path(__file__).resolve().parent
FIGURES_DIR = ROOT / "figures"
FIGURES_DIR.mkdir(exist_ok=True)


def evaluate(model, loader, device: str) -> float:
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            logits = model(x)
            pred = logits.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.numel()
    return correct / max(total, 1)


def make_loaders(batch_size: int = 64, root: str = "./data"):
    train_ds = YesNoSpeechCommands(root=root, subset="training")
    val_ds = YesNoSpeechCommands(root=root, subset="validation")
    test_ds = YesNoSpeechCommands(root=root, subset="testing")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    return train_loader, val_loader, test_loader


def compute_flops_and_params(model, device: str = "cpu"):
    dummy = torch.randn(1, 16000).to(device)
    model = model.to(device)
    macs, params = profile(model, inputs=(dummy,), verbose=False)
    flops = 2 * macs
    return int(flops), int(params)


def train_one_experiment(
    n_mels: int = 80,
    groups: int = 1,
    epochs: int = 10,
    batch_size: int = 64,
    lr: float = 1e-3,
    data_root: str = "./data",
) -> Dict:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_loader, val_loader, test_loader = make_loaders(batch_size=batch_size, root=data_root)

    model = SmallKeywordCNN(n_mels=n_mels, groups=groups).to(device)
    optimizer = Adam(model.parameters(), lr=lr)

    history = {"train_loss": [], "val_acc": [], "epoch_time": []}

    for epoch in range(epochs):
        model.train()
        start = time.time()
        loss_sum = 0.0
        n_samples = 0

        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad()
            logits = model(x)
            loss = cross_entropy(logits, y)
            loss.backward()
            optimizer.step()

            loss_sum += loss.item() * x.size(0)
            n_samples += x.size(0)

        train_loss = loss_sum / max(n_samples, 1)
        val_acc = evaluate(model, val_loader, device)
        epoch_time = time.time() - start

        history["train_loss"].append(train_loss)
        history["val_acc"].append(val_acc)
        history["epoch_time"].append(epoch_time)

        print(
            f"[n_mels={n_mels}, groups={groups}] epoch {epoch + 1}/{epochs} | "
            f"train_loss={train_loss:.4f} | val_acc={val_acc:.4f} | time={epoch_time:.2f}s"
        )

    test_acc = evaluate(model, test_loader, device)
    flops, thop_params = compute_flops_and_params(SmallKeywordCNN(n_mels=n_mels, groups=groups), device=device)

    result = {
        "n_mels": n_mels,
        "groups": groups,
        "test_acc": test_acc,
        "params": count_parameters(model),
        "params_thop": thop_params,
        "flops": flops,
        "history": history,
    }
    return result


def save_json(obj: Dict, path: Path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def save_results_csv(rows: List[Dict], path: Path):
    if not rows:
        return
    keys = ["n_mels", "groups", "test_acc", "params", "params_thop", "flops", "avg_epoch_time"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def plot_mel_comparison(data_root: str = "./data"):
    ds = YesNoSpeechCommands(root=data_root, subset="validation")
    waveform, _ = ds[0]
    waveform = waveform.unsqueeze(0)

    ref = torchaudio.transforms.MelSpectrogram(
        sample_rate=16000,
        n_fft=400,
        hop_length=160,
        n_mels=80,
        power=2.0,
    )(waveform)
    ref = torch.log(ref + 1e-6)

    ours = LogMelFilterBanks(
        samplerate=16000,
        n_fft=400,
        hop_length=160,
        n_mels=80,
        power=2.0,
    )(waveform)

    fig, ax = plt.subplots(figsize=(8, 4))
    diff = (ref - ours).abs().squeeze(0).cpu()
    im = ax.imshow(diff, aspect="auto", origin="lower")
    ax.set_title("Absolute difference: log(MelSpectrogram) vs LogMelFilterBanks")
    ax.set_xlabel("Frame")
    ax.set_ylabel("Mel bin")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "mel_difference.png", dpi=150)
    plt.close(fig)


def plot_loss_curves(mel_results: List[Dict]):
    fig, ax = plt.subplots(figsize=(8, 4))
    for res in mel_results:
        epochs = range(1, len(res["history"]["train_loss"]) + 1)
        ax.plot(list(epochs), res["history"]["train_loss"], label=f'n_mels={res["n_mels"]}')
    ax.set_title("Train loss by n_mels")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "train_loss_by_mels.png", dpi=150)
    plt.close(fig)


def plot_test_acc_vs_mels(mel_results: List[Dict]):
    fig, ax = plt.subplots(figsize=(8, 4))
    xs = [r["n_mels"] for r in mel_results]
    ys = [r["test_acc"] for r in mel_results]
    ax.plot(xs, ys, marker="o")
    ax.set_title("Test accuracy vs n_mels")
    ax.set_xlabel("n_mels")
    ax.set_ylabel("Test accuracy")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "test_acc_vs_mels.png", dpi=150)
    plt.close(fig)


def plot_group_metrics(group_results: List[Dict]):
    xs = [r["groups"] for r in group_results]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(xs, [sum(r["history"]["epoch_time"]) / len(r["history"]["epoch_time"]) for r in group_results], marker="o")
    ax.set_title("Average epoch time vs groups")
    ax.set_xlabel("groups")
    ax.set_ylabel("Average epoch time, s")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "epoch_time_vs_groups.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(xs, [r["params"] for r in group_results], marker="o")
    ax.set_title("Parameter count vs groups")
    ax.set_xlabel("groups")
    ax.set_ylabel("Parameters")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "params_vs_groups.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(xs, [r["flops"] for r in group_results], marker="o")
    ax.set_title("FLOPs vs groups")
    ax.set_xlabel("groups")
    ax.set_ylabel("FLOPs")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "flops_vs_groups.png", dpi=150)
    plt.close(fig)


def run_all_experiments():
    plot_mel_comparison()

    mel_results = []
    for n_mels in [20, 40, 80]:
        res = train_one_experiment(n_mels=n_mels, groups=1)
        mel_results.append(res)
        save_json(res, ROOT / f"result_mels_{n_mels}.json")

    group_results = []
    for groups in [1, 2, 4, 8, 16]:
        res = train_one_experiment(n_mels=80, groups=groups)
        group_results.append(res)
        save_json(res, ROOT / f"result_groups_{groups}.json")

    plot_loss_curves(mel_results)
    plot_test_acc_vs_mels(mel_results)
    plot_group_metrics(group_results)

    rows = []
    for res in mel_results + group_results:
        rows.append({
            "n_mels": res["n_mels"],
            "groups": res["groups"],
            "test_acc": res["test_acc"],
            "params": res["params"],
            "params_thop": res["params_thop"],
            "flops": res["flops"],
            "avg_epoch_time": sum(res["history"]["epoch_time"]) / len(res["history"]["epoch_time"]),
        })
    save_results_csv(rows, ROOT / "results_summary.csv")


if __name__ == "__main__":
    run_all_experiments()
