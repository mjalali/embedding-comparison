import torch
from features.TextFeatureExtractor import TextFeatureExtractor
from transformers import MBartTokenizer, MBartModel

class MbartFeatureExtractor(TextFeatureExtractor):
    def __init__(self, save_path=None, logger=None, API_KEY='your_api_key'):
        self.name = "facebook/mbart-large-50-many-to-many-mmt"
        super().__init__(save_path, logger)

        self.features_size = 1024  # mBART output size (for mBART-large)
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"

        # Load the facebook/mbart-large-50-many-to-many-mmt model and tokenizer
        self.model = MBartModel.from_pretrained('facebook/mbart-large-50-many-to-many-mmt').to(self.device)
        self.tokenizer = MBartTokenizer.from_pretrained('facebook/mbart-large-50-many-to-many-mmt')

    def preprocess(self, text):
        # Tokenize the input text
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=256)

        # Move inputs to the correct device (GPU or CPU)
        input_ids = inputs['input_ids'].to(self.device)
        attention_mask = inputs['attention_mask'].to(self.device)

        return input_ids, attention_mask

    def get_feature_batch(self, text_batch):
        token_ids_batch = None
        attn_mask_batch = None

        # Preprocess each text in the batch
        for text in text_batch:
            token_ids, attn_mask = self.preprocess(text)
            if token_ids_batch is None:
                token_ids_batch = token_ids
                attn_mask_batch = attn_mask
            else:
                token_ids_batch = torch.cat((token_ids_batch, token_ids), axis=0)
                attn_mask_batch = torch.cat((attn_mask_batch, attn_mask), axis=0)

        # Move tensors to the correct device (GPU or CPU)
        token_ids_batch = token_ids_batch.to(self.device)
        attn_mask_batch = attn_mask_batch.to(self.device)

        # Get embeddings from the model
        with torch.no_grad():
            output = self.model(input_ids=token_ids_batch, attention_mask=attn_mask_batch)
            last_hidden_state = output.last_hidden_state  # Shape: (batch_size, seq_len, hidden_size)

            # Extract embeddings for the [EOS] token (last token in the sequence)
            eos_embeddings = last_hidden_state[:, -1, :]  # Use the last token (EOS) embeddings

        return eos_embeddings
