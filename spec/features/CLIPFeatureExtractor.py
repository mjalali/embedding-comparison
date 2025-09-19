'''
    This code is adopted from https://github.com/marcojira/fld
'''


import torch
import torchvision.transforms as transforms
import clip
import open_clip
from features.ImageFeatureExtractor import ImageFeatureExtractor
from features.TextFeatureExtractor import TextFeatureExtractor


class CLIPImageFeatureExtractor(ImageFeatureExtractor):
    def __init__(self, save_path=None, logger=None, clip_download_root=None):
        self.name = "clip-image"

        super().__init__(save_path, logger)

        self.features_size = 512
        self.preprocess = transforms.Compose(
            [
                transforms.Resize(
                    224, interpolation=transforms.InterpolationMode.BICUBIC
                ),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.48145466, 0.4578275, 0.40821073),
                    (0.26862954, 0.26130258, 0.27577711),
                ),
            ]
        )
        self.model, _ = clip.load("ViT-B/32", device="cuda", download_root=clip_download_root)
        self.model.eval()
    
    def get_feature_batch(self, img_batch):
        with torch.no_grad():
            features = self.model.encode_image(img_batch)
        return features


class CLIPTextFeatureExtractor(TextFeatureExtractor):
    def __init__(self, save_path=None, logger=None, clip_download_root=None):
        self.name = "clip-text"

        super().__init__(save_path, logger)

        self.features_size = 512
        
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.model, _ = clip.load("ViT-B/32", device=self.device, download_root=clip_download_root)
        self.model.eval()

    def get_feature_batch(self, text_batch):
        with torch.no_grad():
            tokenized_text = clip.tokenize(text_batch, truncate=True).to(self.device)
            text_feats = self.model.encode_text(tokenized_text)
        return text_feats.to(self.device)


class OpenCLIPImageFeatureExtractor(ImageFeatureExtractor):
    def __init__(self, save_path=None, logger=None, clip_download_root=None, pretrained=None):
        self.name = "open_clip-image"

        super().__init__(save_path, logger)

        self.features_size = 512
        self.model, _, self.preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained=pretrained, device="cuda", cache_dir=clip_download_root)
        self.model.eval()

    def get_feature_batch(self, img_batch):
        with torch.no_grad():
            features = self.model.encode_image(img_batch)
        return features
