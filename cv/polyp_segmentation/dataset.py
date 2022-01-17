import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from skimage import io


class PolypDataset(Dataset):

    def __init__(self, root_path='', transforms=None, mode='train'):
        self.root_path = root_path
        self.transforms = transforms

        if mode not in ['train', 'test']:
            raise ValueError(f"Argument 'mode' must be 'train' or 'test'. Received {mode}")
        self.mode = mode

        if mode == 'train':
            self.data_info = pd.read_csv('train_data.csv')
        else:
            self.data_info = pd.read_csv('test_data.csv')        


    def __len__(self):
        return len(self.data_info)


    def __getitem__(self, index):
        sample = dict()

        image = io.imread(self.data_info.iloc[index, 2]).astype(np.uint8)
        gt_mask = io.imread(self.data_info.iloc[index, 3]).astype(np.uint8)

        sample['input'] = image
        sample['target'] = gt_mask

        if self.transforms:
            sample = self.transforms(sample)

        return sample