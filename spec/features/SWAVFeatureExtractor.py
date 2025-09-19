import torch
import torchvision.transforms as transforms
from features.ImageFeatureExtractor import ImageFeatureExtractor


class SWAVFeatureExtractor(ImageFeatureExtractor):
    def __init__(self, save_path=None, logger=None):
        self.name = f"swav"
        super().__init__(save_path, logger)

        self.features_size = 1000
        self.preprocess = transforms.Compose(
            [
                transforms.Resize(
                    (224, 224), interpolation=transforms.InterpolationMode.BICUBIC
                ),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
                ),
            ]
        )

        self.model = torch.hub.load('facebookresearch/swav:main', 'resnet50')
        self.model.eval()
        self.model.to("cuda")
    
    def get_feature_batch(self, img_batch: torch.Tensor):
        return self.model(img_batch)