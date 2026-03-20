"""
Aplikasi preprocessing ke dataset CSV:
- Load labeled csv
- Buat kolom text_norm
- Drop baris yang menjadi kosong setelah preprocessing
- Export ke file output
"""

import os
import pandas as pd

from preprocess import preprocess_comment


def main():
    input_path = "./data/labeled_raw.csv"
    output_path = "./data/labeled_clean.csv"

    df = pd.read_csv(input_path)
    df = df.dropna(subset=["text", "label"]).copy()
    df["label"] = df["label"].astype(int)
    df = df[df["label"].isin([0, 1])].copy()

    df["text_norm"] = df["text"].astype(str).apply(preprocess_comment)
    df = df[df["text_norm"].str.len() > 0].copy()

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    df.to_csv(output_path, index=False)

    print("=== Preprocessing done ===")
    print(f"Input : {input_path}")
    print(f"Output: {output_path}")
    print(f"Rows  : {len(df)}")


if __name__ == "__main__":
    main()