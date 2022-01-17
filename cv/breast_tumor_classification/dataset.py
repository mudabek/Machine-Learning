import pandas as pd
import numpy as np
from skimage import io

from torch.utils.data import Dataset
import torch



class BreastCancerDataset(Dataset):

    def __init__(self, csv_file, transform=None):
        self.data_info = pd.read_csv(csv_file)
        self.transform = transform

    def __len__(self):
        return len(self.data_info)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        image = io.imread(self.data_info.iloc[idx, 2], as_gray=True).astype(np.float32)
        label = self.data_info.iloc[idx, 3]
        table_idx = self.data_info.iloc[idx, 1]


        if label == 'normal':
            label_id = 0
        elif label == 'benign':
            label_id = 1
        else:
            label_id = 2

        if self.transform:
            img = self.transform(image)

        sample = {'image': img, 'label_id': label_id, 'label': label, 'idx': table_idx}

        

        return sample