import torch
from features.TextFeatureExtractor import TextFeatureExtractor
from transformers import BertModel, BertTokenizer


class BERTFeatureExtractor(TextFeatureExtractor):
    def __init__(self, save_path=None, logger=None, API_KEY = 'your_api_key'):
        self.name = "bert"

        super().__init__(save_path, logger)

        self.features_size = 768
        
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        
        self.model = BertModel.from_pretrained('bert-base-uncased').to(self.device)
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        
    def preprocess(self, text, bert_model, tokenizer):
        
        tokens = tokenizer.tokenize(text)
        
        if len(tokens) > 254:
            tokens = tokens[:254]
        tokens = ['[CLS]'] + tokens + ['[SEP]']
        T=256
        padded_tokens=tokens +['[PAD]' for _ in range(T-len(tokens))]
        attn_mask=[ 1 if token != '[PAD]' else 0 for token in padded_tokens  ]
        
        seg_ids=[0 for _ in range(len(padded_tokens))]
        
        sent_ids=tokenizer.convert_tokens_to_ids(padded_tokens)
        
        token_ids = torch.tensor(sent_ids).unsqueeze(0)
        attn_mask = torch.tensor(attn_mask).unsqueeze(0)
        seg_ids   = torch.tensor(seg_ids).unsqueeze(0)
        
        return token_ids, attn_mask, seg_ids
    
    def get_feature_batch(self, text_batch):
        token_ids_batch = None
        attn_mask_batch = None
        seg_ids_batch = None
        for text in text_batch:
            token_ids, attn_mask, seg_ids = self.preprocess(text, self.model, self.tokenizer)
            # print(token_ids.shape, attn_mask.shape, seg_ids.shape)
            if token_ids_batch is None:
                token_ids_batch = token_ids
                attn_mask_batch = attn_mask
                seg_ids_batch = seg_ids
            else:
                token_ids_batch = torch.cat((token_ids_batch, token_ids), axis=0)
                attn_mask_batch = torch.cat((attn_mask_batch, attn_mask), axis=0)
                seg_ids_batch = torch.cat((seg_ids_batch, seg_ids), axis=0)
        
        token_ids_batch = token_ids_batch.to(self.device)
        attn_mask_batch = attn_mask_batch.to(self.device)
        seg_ids_batch = seg_ids_batch.to(self.device)
        
        with torch.no_grad():
            output = self.model(token_ids_batch, attention_mask=attn_mask_batch,token_type_ids=seg_ids_batch)
            last_hidden_state, pooler_output = output[0], output[1]

        return pooler_output