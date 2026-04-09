import json
import re
from pathlib import Path

import pandas as pd

RESULTS_DIR = Path("results")
OUT_CSV = RESULTS_DIR / "summary.csv"

def extract_json(text: str):
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("No JSON object found")
    return json.loads(text[start:end+1])

rows = []
for path in sorted(RESULTS_DIR.glob("*.json")):
    try:
        payload = extract_json(path.read_text(encoding="utf-8", errors="ignore"))
        payload["file"] = path.name
        rows.append(payload)
    except Exception as e:
        rows.append({"file": path.name, "parse_error": str(e)})

df = pd.DataFrame(rows)

sort_cols = [c for c in ["data_dir", "method", "wer", "cer", "beam_width", "alpha", "beta", "temperature"] if c in df.columns]
if sort_cols:
    df = df.sort_values(sort_cols, na_position="last")

OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(OUT_CSV, index=False)

print(df.to_string(index=False))
print(f"\nSaved to {OUT_CSV}")
