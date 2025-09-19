import torch
from features.TextFeatureExtractor import TextFeatureExtractor
from transformers import RobertaModel, RobertaTokenizer

class RoBERTaFeatureExtractor(TextFeatureExtractor):
    def __init__(self, save_path=None, logger=None, API_KEY='your_api_key'):
        self.name = "roberta"
        super().__init__(save_path, logger)

        self.features_size = 768  # RoBERTa output size
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"

        self.model = RobertaModel.from_pretrained('roberta-base').to(self.device)
        self.tokenizer = RobertaTokenizer.from_pretrained('roberta-base')

    def preprocess(self, text):
        tokens = self.tokenizer.tokenize(text)

        if len(tokens) > 254:
            tokens = tokens[:254]
        tokens = ['<s>'] + tokens + ['</s>']  # RoBERTa uses different special tokens
        T = 256
        padded_tokens = tokens + ['<pad>' for _ in range(T - len(tokens))]
        attn_mask = [1 if token != '<pad>' else 0 for token in padded_tokens]

        # RoBERTa does not use token type IDs
        token_ids = self.tokenizer.convert_tokens_to_ids(padded_tokens)

        token_ids_tensor = torch.tensor(token_ids).unsqueeze(0)
        attn_mask_tensor = torch.tensor(attn_mask).unsqueeze(0)

        return token_ids_tensor, attn_mask_tensor

    def get_feature_batch(self, text_batch):
        token_ids_batch = None
        attn_mask_batch = None

        for text in text_batch:
            token_ids, attn_mask = self.preprocess(text)
            if token_ids_batch is None:
                token_ids_batch = token_ids
                attn_mask_batch = attn_mask
            else:
                token_ids_batch = torch.cat((token_ids_batch, token_ids), axis=0)
                attn_mask_batch = torch.cat((attn_mask_batch, attn_mask), axis=0)

        token_ids_batch = token_ids_batch.to(self.device)
        attn_mask_batch = attn_mask_batch.to(self.device)

        with torch.no_grad():
            output = self.model(token_ids_batch, attention_mask=attn_mask_batch)
            last_hidden_state = output.last_hidden_state  # Shape: (batch_size, sequence_length, hidden_size)
            pooler_output = last_hidden_state[:, 0]  # Use the first token (CLS token)

        return pooler_output
