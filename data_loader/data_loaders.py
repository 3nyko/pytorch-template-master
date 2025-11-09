from torchvision import datasets, transforms
import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from base import BaseDataLoader 

DEFAULT_MODE = "hexadecimal" # binary, decimal, hexadecimal

class CICIoV2024_dataset(Dataset):
    def __init__(self, data_dir, mode=DEFAULT_MODE):
        """
        Dataset loader for CICIoV2024.
        Args:
            data_dir (str): path to root 'data/CICIoV2024'
            mode (str): one of ['binary', 'decimal', 'hexadecimal']
        """
        self.mode = mode
        self.data_path = os.path.join(data_dir, mode)
        self.data = self._load_all_csvs(self.data_path)

        # vstupní data a labely
        self.data_no_label = self.data.drop(columns=["Interface", "label", "category", "specific_class"]).values # data bez labelu
        self.labels = self.data["label"].map(self._map_label).values # volání _map_label nad každou hodnout label

    def _map_label(self, label_value):
        if label_value.upper() == "BENIGN":
            return 0
        return 1

    def _load_all_csvs(self, folder_path):
        """
        Load and concatenate all CSV files in the given folder.
        """
        all_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith(".csv")]
        df_list = []
        for f in all_files:
            df = pd.read_csv(f)
            df_list.append(df)
        combined = pd.concat(df_list, ignore_index=True)
        return combined

    def __len__(self):
        return len(self.data_no_label)

    def __getitem__(self, idx):
        # načti jeden řádek dat jako float, nahrad NaN nulami
        x = torch.tensor(pd.to_numeric(self.data_no_label[idx], errors='coerce').fillna(0).astype(float), dtype=torch.float32)
        y = torch.tensor(int(self.labels[idx]), dtype=torch.long)
        return x, y


class CICIoV2024_data_loader(BaseDataLoader):
    """
    CICIoV2024 data loading class using BaseDataLoader
    """
    def __init__(self, data_dir, batch_size, mode=DEFAULT_MODE,
                 shuffle=True, validation_split=0.0, num_workers=1):
        dataset = CICIoV2024_dataset(data_dir=data_dir, mode=mode)
        super().__init__(dataset, batch_size, shuffle, validation_split, num_workers)




class MnistDataLoader(BaseDataLoader):
    """
    MNIST data loading demo using BaseDataLoader
    """
    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, num_workers=1, training=True):
        trsfm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        self.data_dir = data_dir
        self.dataset = datasets.MNIST(self.data_dir, train=training, download=True, transform=trsfm)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)
