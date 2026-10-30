"""
Data Ingestion and Dataset Preparation Module.
Loads the UCI SMS Spam Collection dataset along with custom domain-specific scam samples,
standardizes schemas, deduplicates entries, and creates reproducible stratified splits.
"""

import os
import urllib.request
import zipfile
from pathlib import Path
from typing import Tuple, Optional
import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import (
    DATA_RAW_DIR,
    DATA_PROCESSED_DIR,
    TRAIN_FILE,
    VAL_FILE,
    TEST_FILE,
    RANDOM_SEED,
    TRAIN_RATIO,
    VAL_RATIO,
    TEST_RATIO,
    LABEL_MAP,
    UCI_DATASET_FILENAME,
    CUSTOM_DATASET_FILENAME,
)
from src.preprocess import clean_text

UCI_SPAM_ZIP_URL: str = "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"


def download_uci_dataset_if_missing(raw_dir: Path = DATA_RAW_DIR) -> Path:
    """
    Check if the UCI SMS Spam Collection dataset exists locally.
    If missing, attempts to download and extract the official archive from UCI Machine Learning Repository.

    Args:
        raw_dir (Path): Directory where raw datasets reside.

    Returns:
        Path: Path to the raw UCI dataset text file.
    """
    raw_dir.mkdir(parents=True, exist_ok=True)
    potential_targets = [
        raw_dir / UCI_DATASET_FILENAME,
        raw_dir / f"{UCI_DATASET_FILENAME}.txt",
        raw_dir / "SMSSpamCollection",
        raw_dir / "SMSSpamCollection.txt",
        raw_dir / "spam.csv",
    ]

    for target in potential_targets:
        if target.exists() and target.stat().st_size > 0:
            return target

    # If missing, attempt download
    target_path = raw_dir / "SMSSpamCollection"
    zip_path = raw_dir / "sms_spam_collection.zip"

    print(f"[INFO] UCI SMS Spam Collection not found in '{raw_dir}'.")
    print(f"[INFO] Downloading official dataset from: {UCI_SPAM_ZIP_URL} ...")

    try:
        urllib.request.urlretrieve(UCI_SPAM_ZIP_URL, zip_path)
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(raw_dir)
        if zip_path.exists():
            zip_path.unlink()  # Cleanup zip archive
        print(f"[SUCCESS] UCI dataset downloaded and extracted to: {target_path}")
        return target_path
    except Exception as exc:
        instructions = (
            "\n" + "=" * 70 + "\n"
            "[DATASET DOWNLOAD INSTRUCTION]\n"
            "Could not automatically download the UCI SMS Spam dataset.\n"
            "Please manually download the zip archive from:\n"
            f"  {UCI_SPAM_ZIP_URL}\n"
            "Extract the file 'SMSSpamCollection' and place it into:\n"
            f"  {raw_dir.resolve()}\n"
            "=" * 70
        )
        print(instructions)
        raise FileNotFoundError(f"Missing UCI dataset: {exc}") from exc


def load_raw_uci_dataset(file_path: Path) -> pd.DataFrame:
    """
    Parse the raw UCI tab-delimited text file into a standardized DataFrame.

    Args:
        file_path (Path): Path to the UCI SMS Spam Collection file.

    Returns:
        pd.DataFrame: Standardized DataFrame with columns ['text', 'label', 'source'].
    """
    try:
        # Standard format is tab-separated: 'ham'/'spam' followed by message text
        df = pd.read_csv(
            file_path,
            sep="\t",
            names=["raw_label", "text"],
            quoting=3,  # QUOTE_NONE to prevent quote character truncation
            encoding="utf-8",
            on_bad_lines="skip",
        )
    except Exception:
        # Fallback for latin-1 encoding if utf-8 fails
        df = pd.read_csv(
            file_path,
            sep="\t",
            names=["raw_label", "text"],
            quoting=3,
            encoding="latin-1",
            on_bad_lines="skip",
        )

    df["raw_label"] = df["raw_label"].astype(str).str.strip().str.lower()
    df["label"] = df["raw_label"].map(LABEL_MAP)
    df["source"] = "uci_sms_collection"
    df["language"] = "en"

    # Filter invalid rows
    df = df.dropna(subset=["text", "label"]).copy()
    df["label"] = df["label"].astype(int)
    return df[["text", "label", "language", "source"]]


def load_custom_scams_dataset(file_path: Path) -> pd.DataFrame:
    """
    Load custom domain-specific scam and legitimate message examples.

    Args:
        file_path (Path): Path to the custom CSV file.

    Returns:
        pd.DataFrame: Standardized DataFrame.
    """
    if not file_path.exists():
        print(f"[WARN] Custom scams file '{file_path}' not found. Skipping.")
        return pd.DataFrame(columns=["text", "label", "language", "source"])

    df = pd.read_csv(file_path, encoding="utf-8")
    if "label" in df.columns:
        df["label"] = df["label"].astype(str).str.strip().str.lower().map(LABEL_MAP)
    if "source" not in df.columns:
        df["source"] = "custom_dataset"
    if "language" not in df.columns:
        df["language"] = "en"

    df = df.dropna(subset=["text", "label"]).copy()
    df["label"] = df["label"].astype(int)
    return df[["text", "label", "language", "source"]]


def load_and_merge_datasets(raw_dir: Path = DATA_RAW_DIR) -> pd.DataFrame:
    """
    Download/load UCI dataset, merge with custom CSVs, clean and deduplicate.

    Args:
        raw_dir (Path): Raw data directory.

    Returns:
        pd.DataFrame: Combined, cleaned DataFrame.
    """
    uci_path = download_uci_dataset_if_missing(raw_dir)
    df_uci = load_raw_uci_dataset(uci_path)

    custom_path = raw_dir / CUSTOM_DATASET_FILENAME
    df_custom = load_custom_scams_dataset(custom_path)

    combined_df = pd.concat([df_uci, df_custom], ignore_index=True)

    # Clean whitespace and strip empty messages
    combined_df["text"] = combined_df["text"].astype(str).str.strip()
    combined_df = combined_df[combined_df["text"] != ""].copy()

    # Preprocessed text column for models
    combined_df["cleaned_text"] = combined_df["text"].apply(clean_text)

    # Remove exact duplicate texts to prevent train/test leakage
    initial_len = len(combined_df)
    combined_df = combined_df.drop_duplicates(subset=["cleaned_text"]).reset_index(drop=True)
    dedup_count = initial_len - len(combined_df)

    print(f"[INFO] Merged total: {len(combined_df)} samples (Deduplicated {dedup_count} rows).")
    print(f"[INFO] Class Distribution:\n{combined_df['label'].value_counts(normalize=True).to_dict()}")

    return combined_df


def create_stratified_splits(
    df: Optional[pd.DataFrame] = None,
    save_splits: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Partition dataset into stratified Train (70%), Validation (15%), and Test (15%) sets.

    Args:
        df (pd.DataFrame, optional): Data to split. If None, loads and merges raw datasets.
        save_splits (bool): Whether to persist split CSVs into data/processed/.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: train_df, val_df, test_df
    """
    if df is None:
        df = load_and_merge_datasets()

    # First split: Train (70%) vs Temp (30%)
    temp_ratio = VAL_RATIO + TEST_RATIO  # 0.30
    train_df, temp_df = train_test_split(
        df,
        test_size=temp_ratio,
        random_state=RANDOM_SEED,
        stratify=df["label"],
    )

    # Second split: Temp (30%) split equally into Val (15%) and Test (15%)
    val_rel_ratio = VAL_RATIO / temp_ratio  # 0.50
    val_df, test_df = train_test_split(
        temp_df,
        test_size=(1.0 - val_rel_ratio),
        random_state=RANDOM_SEED,
        stratify=temp_df["label"],
    )

    print(
        f"[INFO] Stratified Splits Created -> Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}"
    )

    if save_splits:
        DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        train_df.to_csv(TRAIN_FILE, index=False)
        val_df.to_csv(VAL_FILE, index=False)
        test_df.to_csv(TEST_FILE, index=False)
        print(f"[INFO] Saved splits to: {DATA_PROCESSED_DIR}")

    return train_df, val_df, test_df


if __name__ == "__main__":
    create_stratified_splits()
