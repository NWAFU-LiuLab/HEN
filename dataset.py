from config import config
import random
import os
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
from config import Config as config
from torchvision.transforms import ToTensor

class FloodDataset(Dataset):
    def __init__(self, labeled_data_path, unlabeled_data_path, split="train", fold_idx=0, transform=None):
        self.labeled_data_path = labeled_data_path
        self.unlabeled_data_path = unlabeled_data_path
        self.transform = transform
        self.split = split
        self.to_tensor = ToTensor()

        if self.split in ["train", "val", "test"]:
            self.labeled_image_list = os.listdir(os.path.join(labeled_data_path, "Image"))
            self.mask_list = sorted(os.listdir(os.path.join(self.labeled_data_path, "Mask")))
            assert len(self.labeled_image_list) == len(self.mask_list)

            random.seed(7)
            random.shuffle(self.labeled_image_list)
            random.shuffle(self.mask_list)

            num_folds = 5
            fold_size = len(self.labeled_image_list) // num_folds

            if self.split == "test":
                self.image_list = self.labeled_image_list[-100:]
                self.mask_list = self.mask_list[-100:]
            else:
                # Here we slice the dataset into training and validation based on fold index
                val_start = fold_idx * fold_size
                val_end = (fold_idx + 1) * fold_size

                if self.split == "train":
                    self.image_list = self.labeled_image_list[:val_start] + self.labeled_image_list[val_end:-100]
                    self.mask_list = self.mask_list[:val_start] + self.mask_list[val_end:-100]
                elif self.split == "val":
                    self.image_list = self.labeled_image_list[val_start:val_end]
                    self.mask_list = self.mask_list[val_start:val_end]
        # Process unlabeled data
        elif self.split == "unlabeled":
            self.image_list = os.listdir(unlabeled_data_path)
            self.mask_list = [None] * len(self.image_list)  # Use None as placeholders
        else:
            raise ValueError("Invalid split argument. It should be 'train', 'val', or 'unlabeled'.")

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        img_name = self.image_list[idx]
        if isinstance(img_name, torch.Tensor):
            image = img_name
            mask = self.mask_list[idx]
        elif self.split == "unlabeled":
            image = Image.open(os.path.join(self.unlabeled_data_path, img_name)).convert("RGB")
            # Convert PIL images to PyTorch tensors
            image = self.preprocess_unlabeled(image)
            mask = None  # There's no mask for unlabeled data
        else:
            image = Image.open(os.path.join(self.labeled_data_path, "Image", img_name)).convert("RGB")
            # mask = Image.open(os.path.join(self.labeled_data_path, "Mask", img_name)).convert("L")
            mask = Image.open(os.path.join(self.labeled_data_path, "Mask", img_name.split('.')[0]+'.png')).convert("L")

        if self.transform:
            image, mask = self.transform(image, mask)
        elif mask is not None:  # Don't preprocess mask if it's None
            image, mask = self.preprocess(image, mask)

        return image, mask if mask is not None else image

    def preprocess(self, image, mask):
        if isinstance(image, torch.Tensor) and isinstance(mask, torch.Tensor):
            image = image.squeeze(0) if image.size(0) == 1 else image
            mask = mask.squeeze(0) if mask.size(0) == 1 else mask
            return image, mask
        image = image.resize((config.IMG_WIDTH, config.IMG_HEIGHT), Image.BILINEAR)
        image = np.array(image, dtype=np.float32)
        image /= 255.0
        image = np.transpose(image, (2, 0, 1))

        mask = mask.resize((config.IMG_WIDTH, config.IMG_HEIGHT), Image.NEAREST)
        mask = np.array(mask, dtype=np.float32)
        mask /= 255.0

        image = torch.from_numpy(image).float()
        mask = torch.from_numpy(mask).float()

        return image, mask

    def preprocess_unlabeled(self,image):

        image = image.resize((config.IMG_WIDTH, config.IMG_HEIGHT), Image.BILINEAR)
        image = np.array(image, dtype=np.float32)
        image /= 255.0
        image = np.transpose(image, (2, 0, 1))
        image = torch.from_numpy(image).float()
        return image

    def extend(self, images, masks):
        self.image_list.extend(images)
        self.mask_list.extend(masks)

