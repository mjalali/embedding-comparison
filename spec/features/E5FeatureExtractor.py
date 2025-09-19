import torch
import torchvision.transforms as transforms
import clip
import open_clip
from features.ImageFeatureExtractor import ImageFeatureExtractor
from features.TextFeatureExtractor import TextFeatureExtractor

import torch.nn.functional as F

from torch import Tensor
from transformers import AutoTokenizer, AutoModel


def average_pool(last_hidden_states: Tensor,
                 attention_mask: Tensor) -> Tensor:
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]


class E5TextFeatureExtractor(TextFeatureExtractor):
    def __init__(self, save_path=None, logger=None):
        self.name = "E5-large-v2"

        super().__init__(save_path, logger)

        self.features_size = 1024
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained('intfloat/e5-large-v2')
        self.model = AutoModel.from_pretrained('intfloat/e5-large-v2')
        # self.model.eval()

    def get_feature_batch(self, text_batch):
        prefixed_text_batch = [f"query: {text}" for text in text_batch]  # It is neccessary for E5 input to be like "query: {text}"
        with torch.no_grad():
            batch_dict = self.tokenizer(prefixed_text_batch, max_length=512, padding=True, truncation=True, return_tensors='pt')
            outputs = self.model(**batch_dict)
            embeddings = average_pool(outputs.last_hidden_state, batch_dict['attention_mask'])
            embeddings = F.normalize(embeddings, p=2, dim=1)
            print(embeddings.shape)
            return embeddings
