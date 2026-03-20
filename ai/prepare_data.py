import os
import pandas as pd
from sklearn.model_selection import train_test_split
from config import Config

def main():
    config = Config()
    input_path = "./data/labeled_clean.csv"
    split_dir = "./data/splits"

    df = pd.read_csv(input_path)
    df = df.dropna(subset=["text", "Label"])
    df["Label"] = df["Label"].astype(int)
    df = df[df["Label"].isin([0, 1])].copy()

    train_df, temp_df = train_test_split(
        df, test_size=0.2, random_state=config.seed, stratify=df["Label"]
    )
    val_df, test_df = train_test_split(
        temp_df, test_size=0.5, random_state=config.seed, stratify=temp_df["Label"]
    )

    os.makedirs(split_dir, exist_ok=True)
    train_df.to_csv(os.path.join(split_dir, "train.csv"), index=False)
    val_df.to_csv(os.path.join(split_dir, "val.csv"), index=False)
    test_df.to_csv(os.path.join(split_dir, "test.csv"), index=False)

    print("=== Data Split Complete ===")
    print(f"Total  : {len(df)}")
    print(f"Train  : {len(train_df)}")
    print(f"Val    : {len(val_df)}")
    print(f"Test   : {len(test_df)}")

if __name__ == "__main__":
    main()