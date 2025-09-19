import torch
import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from typing import List


class TextImageFilesDataset(Dataset):
    def __init__(self, image_folder, caption_file, image_paths: List[str] = None, captions: List[str] = None, n_digits = 5, transform=None, img_format='png'):
        """
        Args:
            image_folder (string): Directory with all the images.
            caption_file (string): Path to the text file with captions.
            image_pahts List[str]: List of image paths
            captions: All captions
            transform (callable, optional): Optional transform to be applied on an image.
        """
        self.image_folder = image_folder
        self.image_paths = image_paths
        self.transform = transform
        self.n_digits = n_digits
        self.captions = captions
        self.img_format = img_format


        # Read captions from the text file
        if self.captions is None:
            with open(caption_file, 'r') as file:
                self.captions = file.readlines()

    def __len__(self):
        return len(self.captions)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        if self.image_paths is None:
            image_name = f'{idx :0{self.n_digits}d}.{self.img_format}'
            image_path = os.path.join(self.image_folder, image_name)
        else:
            image_path = self.image_paths[idx]

        try:
            image = Image.open(image_path).convert('RGB')
        except (IOError, SyntaxError) as e:
            print(f"Error reading image {image_path}: {e}")
            return None

        caption = self.captions[idx].strip()

        if self.transform:
            image = self.transform(image)

        return image, caption
