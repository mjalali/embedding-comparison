import torch
from features.TextFeatureExtractor import TextFeatureExtractor
from openai import OpenAI
import numpy as np


class GPTFeatureExtractor(TextFeatureExtractor):
    def __init__(self, save_path=None, logger=None, API_KEY = None):
        self.name = "gpt-large"

        super().__init__(save_path, logger)

        self.features_size = 3072
        
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.model = 'text-embedding-3-large'
        self.api_key = API_KEY
    
    
    def get_feature_batch(self, text_batch):
        client = OpenAI(api_key=self.api_key)
        embeddings = client.embeddings.create(input = text_batch, model=self.model).data
        embeddings = [e.embedding for e in embeddings]
        
        return torch.tensor(embeddings).to(self.device)