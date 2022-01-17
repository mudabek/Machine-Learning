import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision.utils import save_image


class FacadesDataset(Dataset):
    def __init__(self, root_dir, is_cyclegan=False):
        self.root_dir = root_dir
        self.list_files = os.listdir(self.root_dir)
        self.is_cyclegan = is_cyclegan

    def __len__(self):
        return len(self.list_files)

    def __getitem__(self, index):
        img_file = self.list_files[index]
        img_path = os.path.join(self.root_dir, img_file)
        image = np.array(Image.open(img_path))
        target_image = image[:, :256, :]
        input_image = image[:, 256:, :]

        if not self.is_cyclegan:
            augmentations = both_transform(image=input_image, image0=target_image)
            input_image = augmentations["image"]
            target_image = augmentations["image0"]

            input_image = transform_only_input(image=input_image)["image"]
            target_image = transform_only_mask(image=target_image)["image"]
        else:
            augmentations = cgan_transforms(image=input_image, image0=target_image)
            input_image = augmentations["image"]
            target_image = augmentations["image0"]

        return input_image, target_image


both_transform = A.Compose(
    [A.Resize(width=256, height=256),], additional_targets={"image0": "image"},
)

transform_only_input = A.Compose(
    [
        A.HorizontalFlip(p=0.5),
        A.ColorJitter(p=0.2),
        A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], max_pixel_value=255.0,),
        ToTensorV2(),
    ]
)

transform_only_mask = A.Compose(
    [
        A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], max_pixel_value=255.0,),
        ToTensorV2(),
    ]
)

cgan_transforms = A.Compose(
    [
        A.Resize(width=256, height=256),
        A.HorizontalFlip(p=0.5),
        A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], max_pixel_value=255),
        ToTensorV2(),
     ],
    additional_targets={"image0": "image"},
)


