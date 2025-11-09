import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision import datasets, transforms
from base import BaseDataLoader
from enum import Enum

# =====================================================
# =========       Constants and options       =========
# =====================================================

class Mode(Enum):
    BINARY = "binary"
    DECIMAL = "decimal"
    HEXADECIMAL = "hexadecimal"

DEFAULT_MODE = Mode.HEXADECIMAL 



# ======================================================
# =========             Dataset              ===========
# ======================================================

class CICIoV2024_Dataset(Dataset):
    """
    Loads all CSVs from the given CICIoV2024 mode folder (binary/decimal/hexadecimal).
    Each CSV must contain DATA_0 ... DATA_7 columns and 'label'.
    """
    def __init__(self, data_dir, mode=DEFAULT_MODE):
        self.mode = mode
        self.folder = os.path.join(data_dir, mode)
        if not os.path.isdir(self.folder):
            raise FileNotFoundError(f"Folder not found: {self.folder}")

        self.data = self._load_all_csvs(self.folder)
        self.data_no_labels = self._remove_labels(self.data)
        self.labels = self._map_labels(self.data["label"])

    def _load_all_csvs(self, folder_path):
        all_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith(".csv")]
        if not all_files:
            raise FileNotFoundError(f"No CSV files found in {folder_path}")

        df_list = []
        for f in all_files:
            df = pd.read_csv(f, low_memory=False)
            df_list.append(df)
        df = pd.concat(df_list, ignore_index=True)
        df.columns = [c.strip() for c in df.columns] # normalize column names
        # ensure numeric conversion for data columns
        for col in df.columns:
            if col.startswith("DATA_") or col in ["ID", "DLC"]:
                df[col] = pd.to_numeric(df[col], errors="coerce") # TODO: Overit funkcnost pro Binary/Hexadec
        df = df.fillna(0) # vyplnit pripadne prazdne hodnoty nulou
        return df

    def _remove_labels(self, df):
        """Return only numeric feature columns."""
        feature_cols = [c for c in df.columns if c.startswith("DATA_")]
        if not feature_cols:
            raise ValueError("No DATA_ columns found in dataset.")
        return df[feature_cols].values

    def _map_labels(self, labels):
        """(0=BENIGN, 1=ATTACK)."""
        mapped = labels.str.upper().map(lambda x: 0 if "BENIGN" in x else 1)
        return mapped.values

    def __len__(self):
        return len(self.data_no_labels)

    def __getitem__(self, idx):
        x = self.data_no_labels[idx].copy()

        def safe_int(val, base=10):
            return int(str(val), base) # Pandas často čte hodnoty jako float → přetypujeme na string

        # Převod podle módu
        if self.mode in [Mode.BINARY, Mode.BINARY.value]:
            x = [safe_int(v, base=2) for v in x]
        elif self.mode in [Mode.HEXADECIMAL, Mode.HEXADECIMAL.value]:
            x = [safe_int(v, base=16) for v in x]
        elif self.mode in [Mode.DECIMAL, Mode.DECIMAL.value]:
            x = [float(v) if not pd.isna(v) else 0.0 for v in x]
        else:
            raise ValueError(f"Unsupported mode: {self.mode}")

        x = torch.tensor(x, dtype=torch.float32)
        y = torch.tensor(self.labels[idx], dtype=torch.long)
        return x, y

# ========================================================
# =========             DataLoader              ==========
# ========================================================
class CICIoV2024_DataLoader(BaseDataLoader):
    """
    DataLoader for CICIoV2024 datasets.
    """
    def __init__(self, data_dir, batch_size, mode=DEFAULT_MODE,
                 shuffle=True, validation_split=0.1, num_workers=2):
        dataset = CICIoV2024_Dataset(data_dir=data_dir, mode=mode)
        super().__init__(dataset, batch_size, shuffle, validation_split, num_workers)