'''
    This code is adopted from https://github.com/marcojira/fld
'''

import torch
import torchvision.transforms as transforms
from FKEA.features.ImageFeatureExtractor import ImageFeatureExtractor


class MonoPixelFeatureExtractor(ImageFeatureExtractor):
    def __init__(self, save_path=None, logger=None):
        self.name = "pixel"
        super().__init__(save_path, logger)

        self.features_size = 784
        self.preprocess = transforms.Compose(
            [
                transforms.ToTensor(),
            ]
        )
        self.model = None
    
    def get_feature_batch(self, img_batch: torch.Tensor):
        
        batch_size = img_batch.shape[0]
        
        return img_batch.view(batch_size, -1)

class ColorPixelFeatureExtractor(ImageFeatureExtractor):
    def __init__(self, save_path=None, logger=None):
        self.name = "pixel"
        super().__init__(save_path, logger)
        
        self.features_size = 784*3
        self.preprocess = transforms.Compose(
            [
                transforms.ToTensor(),
            ]
        )
        self.model = None

    
    def get_feature_batch(self, img_batch: torch.Tensor):
        
        batch_size = img_batch.shape[0]
        
        return img_batch.view(batch_size, -1)
