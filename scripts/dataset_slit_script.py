import os
import pandas as pd
from sklearn.model_selection import train_test_split

# ===============================
# CONFIG
# ===============================
DATASET_PATH = r"C:\Users\fisar\Desktop\Diplomka\pytorch-template-master\data\CICIoV2024"
OUTPUT_PATH = r"C:\Users\fisar\Desktop\Diplomka\pytorch-template-master\data\CICIoV2024_split"

VAL_SPLIT = 0.1
TEST_SPLIT = 0.2
TRAIN_SPLIT = 1 - (VAL_SPLIT + TEST_SPLIT)
RANDOM_STATE = 42

# ===============================
# HELPERS
# ===============================
def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def process_mode(mode):
    """Combine all CSVs from one mode, shuffle, split, and save as train/val/test CSV files."""
    mode_dir = os.path.join(DATASET_PATH, mode)
    if not os.path.isdir(mode_dir):
        print(f"Missing folder: {mode_dir}")
        return

    all_files = [os.path.join(mode_dir, f) for f in os.listdir(mode_dir) if f.endswith(".csv")]
    if not all_files:
        print(f"No CSV files found in {mode_dir}")
        return

    print(f"Processing mode: {mode} ({len(all_files)} files)")

    # Načtení a sloučení všech CSV
    df_list = []
    for fpath in all_files:
        try:
            df = pd.read_csv(fpath, low_memory=False)
            df_list.append(df)
        except Exception as e:
            print(f"Error reading {fpath}: {e}")

    combined_df = pd.concat(df_list, ignore_index=True)
    print(f"Combined shape: {combined_df.shape}")

    # Náhodné promíchání všech řádků
    combined_df = combined_df.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)

    # Rozdělení datasetu
    trainval_df, test_df = train_test_split(
        combined_df, test_size=TEST_SPLIT, random_state=RANDOM_STATE, shuffle=True
    )
    relative_val_split = VAL_SPLIT / (TRAIN_SPLIT + VAL_SPLIT)
    train_df, val_df = train_test_split(
        trainval_df, test_size=relative_val_split, random_state=RANDOM_STATE, shuffle=True
    )

    # Uložení do výstupních CSV
    out_dir = os.path.join(OUTPUT_PATH, mode)
    ensure_dir(out_dir)

    train_df.to_csv(os.path.join(out_dir, "train.csv"), index=False)
    val_df.to_csv(os.path.join(out_dir, "val.csv"), index=False)
    test_df.to_csv(os.path.join(out_dir, "test.csv"), index=False)

    print(f"Saved: train ({len(train_df)}), val ({len(val_df)}), test ({len(test_df)})")

# ===============================
# MAIN
# ===============================
for mode in ["binary", "decimal", "hexadecimal"]:
    process_mode(mode)

print("\nAll splits completed and saved to:", OUTPUT_PATH)