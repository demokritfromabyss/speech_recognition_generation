import argparse
import json
import time
from pathlib import Path
from typing import Dict, List

import jiwer
import pandas as pd
import torchaudio

from wav2vec2decoder import Wav2Vec2Decoder


AUDIO_EXTENSIONS = {".wav", ".flac", ".mp3", ".m4a"}


def resolve_audio_path(data_dir: Path, raw_path: str | Path) -> Path:
    """
    Resolve audio paths robustly for manifests that may store:
    - just a filename: sample_0.wav
    - a relative path under data_dir: subdir/sample_0.wav
    - a relative path already including data_dir prefix:
      data/librispeech_test_other/sample_0.wav
    - an absolute path
    """
    raw = Path(str(raw_path))

    if raw.is_absolute():
        return raw

    # Case 1: path already works from current working directory
    if raw.exists():
        return raw.resolve()

    # Case 2: path should be resolved relative to data_dir
    candidate = (data_dir / raw).resolve()
    if candidate.exists():
        return candidate

    # Case 3: raw path already starts with data_dir, but code is called with data_dir too
    # Example:
    #   data_dir = data/librispeech_test_other
    #   raw_path = data/librispeech_test_other/sample_0.wav
    try:
        data_dir_parts = data_dir.parts
        raw_parts = raw.parts
        if len(raw_parts) >= len(data_dir_parts) and raw_parts[: len(data_dir_parts)] == data_dir_parts:
            candidate = Path(*raw_parts).resolve()
            if candidate.exists():
                return candidate
    except Exception:
        pass

    # Fallback: return normalized candidate so the caller gets a readable error path
    return candidate


def load_manifest(data_dir: Path) -> List[Dict[str, str]]:
    """
    Tries several common dataset layouts.
    Supported patterns:
      - manifest.csv / metadata.csv / transcripts.csv with columns like path/audio/transcript/text/reference
      - *.txt / *.trans.txt with lines: <utt_id> <text>
      - audio files with matching .txt files next to them
    """
    data_dir = data_dir.resolve()

    candidates = ["manifest.csv", "metadata.csv", "transcripts.csv", "labels.csv"]
    for name in candidates:
        path = data_dir / name
        if path.exists():
            df = pd.read_csv(path)
            audio_col = next((c for c in df.columns if c.lower() in {"path", "audio", "audio_path", "wav", "wav_path", "file"}), None)
            text_col = next((c for c in df.columns if c.lower() in {"text", "transcript", "reference", "label", "sentence"}), None)
            if audio_col and text_col:
                records = []
                for _, row in df.iterrows():
                    audio_path = resolve_audio_path(data_dir, row[audio_col])
                    records.append({
                        "audio_path": str(audio_path),
                        "reference": str(row[text_col]).strip().lower(),
                    })
                return records

    for txt_name in ["transcripts.txt", "manifest.txt", "references.txt"]:
        path = data_dir / txt_name
        if path.exists():
            records = []
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split(maxsplit=1)
                    if len(parts) != 2:
                        continue
                    utt_id, text = parts
                    audio_path = None
                    for ext in AUDIO_EXTENSIONS:
                        cand = data_dir / f"{utt_id}{ext}"
                        if cand.exists():
                            audio_path = cand
                            break
                    if audio_path is None:
                        continue
                    records.append({"audio_path": str(audio_path.resolve()), "reference": text.lower()})
                if records:
                    return records

    records = []
    for audio_path in sorted(p for p in data_dir.rglob("*") if p.suffix.lower() in AUDIO_EXTENSIONS):
        txt_path = audio_path.with_suffix(".txt")
        if txt_path.exists():
            records.append({
                "audio_path": str(audio_path.resolve()),
                "reference": txt_path.read_text(encoding="utf-8").strip().lower(),
            })
    if records:
        return records

    raise FileNotFoundError(
        f"Could not infer manifest format inside {data_dir}. Add a manifest.csv with columns path,text."
    )


def evaluate_dataset(
    decoder: Wav2Vec2Decoder,
    data_dir: Path,
    method: str,
    limit: int | None = None,
) -> Dict:
    data_dir = data_dir.resolve()
    records = load_manifest(data_dir)
    if limit is not None:
        records = records[:limit]

    rows = []
    t0 = time.time()
    for sample in records:
        audio_path = resolve_audio_path(data_dir, sample["audio_path"])
        audio, sr = torchaudio.load(str(audio_path))
        if sr != 16000:
            audio = torchaudio.functional.resample(audio, sr, 16000)
        hyp = decoder.decode(audio, method=method).lower().strip()
        ref = sample["reference"].lower().strip()
        rows.append({
            "audio_path": str(audio_path),
            "reference": ref,
            "hypothesis": hyp,
            "wer": jiwer.wer(ref, hyp),
            "cer": jiwer.cer(ref, hyp),
        })

    df = pd.DataFrame(rows)
    total_time = time.time() - t0
    return {
        "n_samples": len(df),
        "wer": float(jiwer.wer(df["reference"].tolist(), df["hypothesis"].tolist())),
        "cer": float(jiwer.cer(df["reference"].tolist(), df["hypothesis"].tolist())),
        "avg_sample_time_sec": total_time / max(len(df), 1),
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=Path, required=True)
    parser.add_argument("--method", type=str, default="greedy", choices=["greedy", "beam", "beam_lm", "beam_lm_rescore"])
    parser.add_argument("--beam_width", type=int, default=10)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--beta", type=float, default=0.5)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--lm_model_path", type=str, default="lm/3-gram.pruned.1e-7.arpa.gz")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output_json", type=Path, default=None)
    args = parser.parse_args()

    lm_path = args.lm_model_path if args.method in {"beam_lm", "beam_lm_rescore"} else None
    decoder = Wav2Vec2Decoder(
        beam_width=args.beam_width,
        alpha=args.alpha,
        beta=args.beta,
        temperature=args.temperature,
        lm_model_path=lm_path,
    )

    result = evaluate_dataset(decoder, args.data_dir, args.method, limit=args.limit)
    payload = {
        "data_dir": str(args.data_dir),
        "method": args.method,
        "beam_width": args.beam_width,
        "alpha": args.alpha,
        "beta": args.beta,
        "temperature": args.temperature,
        "lm_model_path": lm_path,
        **{k: v for k, v in result.items() if k != "rows"},
    }

    print(json.dumps(payload, indent=2, ensure_ascii=False))
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump({**payload, "rows": result["rows"]}, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
