import copy
import json
import cv2
import random
import numpy as np
import pandas as pd
import torchvision.transforms as transforms
import albumentations as A
from PIL import Image
from torch.utils.data import Dataset
import tifffile


def GetImage_Mask_Transform_SpatialLevel():
    image_mask_transform_spatiallevel = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Transpose(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.25, scale_limit=0.5,
                           rotate_limit=90, border_mode=0, value=0, p=0.5),
    ], additional_targets={"image1": "image", "mask1": "mask"}, p=1)
    return image_mask_transform_spatiallevel


def GetImage_Mask_Transform_RandomCrop(image_size):
    image_mask_transform_randomCrop = A.Compose([
        A.OneOf([
            A.RandomCrop(height=image_size[0], width=image_size[1], p=1),
        ], p=1)], additional_targets={"image1": "image", "mask1": "mask"}, p=1)
    return image_mask_transform_randomCrop


def GetImage_Transform_PixelLevel():
    image_transform_pixellevel = A.Compose([
        # A.RandomBrightnessContrast(p=0.5),
    ], p=1)
    return image_transform_pixellevel


def OpenTiffImageAndConvertToPilImage(path):
    im = tifffile.imread(path)
    im = Image.fromarray(im)
    return im


class CVCClinicDBDataset(Dataset):

    def __init__(self, mode="train", normalization=True, augmentation=False):
        super().__init__()
        self.normalization = normalization
        self.augmentation = augmentation

        self.imagenet_mean = [0.419, 0.278, 0.186]
        # 0.41894006729125977, 0.2778775095939636, 0.18646252155303955
        self.imagenet_std = [0.295, 0.204, 0.139]
        # 0.2950814664363861, 0.20460145175457, 0.13856109976768494
        self.mode = mode

        if self.mode == 'train':
            self.image_size = (256, 256)
        else:
            self.image_size = (256, 256)

        self.image_mask_transform_spatiallevel = GetImage_Mask_Transform_SpatialLevel()
        self.image_mask_transform_randomCrop = GetImage_Mask_Transform_RandomCrop(
            self.image_size)
        self.image_transform_pixellevel = GetImage_Transform_PixelLevel()

        # Define list of image transformations
        label_transformation = [transforms.ToTensor()]
        image_transformation = [transforms.ToTensor()]
        if self.normalization:
            image_transformation.append(transforms.Normalize(
                self.imagenet_mean, self.imagenet_std))
        self.label_transformation = transforms.Compose(label_transformation)
        self.image_transformation = transforms.Compose(image_transformation)

        # Get all image paths and label paths from data
        if mode not in ['train', 'test']:
            raise ValueError(f"Argument 'mode' must be 'train' or 'test'. Received {mode}")\

        if mode == 'train':
            self.data_info = pd.read_csv('train_data.csv')
        else:
            self.data_info = pd.read_csv('test_polyp_segm.csv')

    def __len__(self):
        return len(self.data_info)

    def __getitem__(self, index):
        # Read image
        image_path = self.data_info.iloc[index, 2]
        image_data = OpenTiffImageAndConvertToPilImage(
            image_path)  # Image.open(image_path).convert("RGB")

        # Read label
        label_path = self.data_info.iloc[index, 3]
        mask_data = OpenTiffImageAndConvertToPilImage(
            label_path)  # Image.open(label_path).convert("RGB")

        if self.mode == "train":

            image_data = np.array(image_data)
            mask_data = np.array(mask_data)

            if self.augmentation is True:
                transformed = self.image_mask_transform_spatiallevel(
                    image=image_data, mask=mask_data)
                image_data = transformed["image"]
                mask_data = transformed["mask"]
                image_data = self.image_transform_pixellevel(image=image_data)[
                    "image"]

            if random.uniform(0, 1) > 0.1 and self.augmentation is True:
                image_data = A.Resize(height=288, width=384, interpolation=cv2.INTER_LINEAR, p=1)(
                    image=image_data)["image"]
                mask_data = A.Resize(height=288, width=384, interpolation=cv2.INTER_NEAREST, p=1)(
                    image=mask_data)["image"]
                transformed = self.image_mask_transform_randomCrop(
                    image=image_data, mask=mask_data)
                image_data = transformed["image"]
                mask_data = transformed["mask"]
            else:
                image_data = A.Resize(height=self.image_size[0], width=self.image_size[1], interpolation=cv2.INTER_LINEAR, p=1)(
                    image=image_data)["image"]
                mask_data = A.Resize(height=self.image_size[0], width=self.image_size[1], interpolation=cv2.INTER_NEAREST, p=1)(
                    image=mask_data)["image"]

            image_data = Image.fromarray(image_data)
            mask_data = Image.fromarray(mask_data)

            image_data = self.image_transformation(image_data)
            mask_data = self.label_transformation(
                mask_data)[0, :, :][None, :, :]
            return image_data, mask_data, index

        elif self.mode == "val" or self.mode == "test":

            image_data = np.array(image_data)
            mask_data = np.array(mask_data)
            image_data = A.Resize(height=self.image_size[0], width=self.image_size[1], interpolation=cv2.INTER_LINEAR, p=1)(
                image=image_data)["image"]
            mask_data = A.Resize(height=self.image_size[0], width=self.image_size[1], interpolation=cv2.INTER_NEAREST, p=1)(
                image=mask_data)["image"]
            image_data = Image.fromarray(image_data)
            mask_data = Image.fromarray(mask_data)

            image_data = self.image_transformation(image_data)
            mask_data = self.label_transformation(
                mask_data)[0, :, :][None, :, :]
            return image_data, mask_data, index

        else:
            raise Exception("Mode setting is not valid")
