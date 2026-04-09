import json
from pathlib import Path
import pandas as pd

RESULTS_DIR = Path("results")

def extract_json(text: str):
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("No JSON object found")
    return json.loads(text[start:end+1])

rows = []
for path in RESULTS_DIR.glob("*.json"):
    try:
        payload = extract_json(path.read_text(encoding="utf-8", errors="ignore"))
        payload["file"] = path.name
        rows.append(payload)
    except Exception:
        pass

df = pd.DataFrame(rows)
if df.empty:
    raise SystemExit("No results found in results/*.json")

print("=== Best beam_lm on LibriSpeech ===")
beam_lm = df[(df["data_dir"] == "data/librispeech_test_other") & (df["method"] == "beam_lm")]
if not beam_lm.empty:
    best = beam_lm.sort_values("wer").iloc[0]
    print(best[["file", "wer", "cer", "alpha", "beta", "beam_width", "temperature", "lm_model_path"]].to_string())
else:
    print("No beam_lm rows found.")

print("\n=== Best beam_lm_rescore on LibriSpeech ===")
rescore = df[(df["data_dir"] == "data/librispeech_test_other") & (df["method"] == "beam_lm_rescore")]
if not rescore.empty:
    best = rescore.sort_values("wer").iloc[0]
    print(best[["file", "wer", "cer", "alpha", "beta", "beam_width", "temperature", "lm_model_path"]].to_string())
else:
    print("No beam_lm_rescore rows found.")
