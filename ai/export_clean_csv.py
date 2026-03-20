import os
import pandas as pd
from preprocess import preprocess_comment

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)

INPUT_CSV = os.path.join(PROJECT_DIR, "data", "labeled_raw.csv")
OUTPUT_CSV = os.path.join(PROJECT_DIR, "data", "labeled_clean.csv")

TEXT_COL = "text"
LABEL_COL = "Label"

def main():
    print(f"Loading from: {INPUT_CSV}")
    
    if not os.path.exists(INPUT_CSV):
        raise FileNotFoundError(f"File tidak ditemukan: {INPUT_CSV}")
    
    df = pd.read_csv(INPUT_CSV)

    if TEXT_COL not in df.columns or LABEL_COL not in df.columns:
        raise ValueError(f"Kolom wajib tidak ada. Ditemukan: {list(df.columns)}")

    df = df.dropna(subset=[TEXT_COL, LABEL_COL]).copy()
    df[LABEL_COL] = df[LABEL_COL].astype(int)
    df = df[df[LABEL_COL].isin([0, 1])].copy()

    before = len(df)
    df["text_norm"] = df[TEXT_COL].astype(str).apply(preprocess_comment)
    df = df[df["text_norm"].str.len() > 0].copy()
    after = len(df)

    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8")

    print(f"OK: {INPUT_CSV} -> {OUTPUT_CSV}")
    print(f"Rows before: {before} | after: {after} | dropped: {before - after}")

if __name__ == "__main__":
    main()