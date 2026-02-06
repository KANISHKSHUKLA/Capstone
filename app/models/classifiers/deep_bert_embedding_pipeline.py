

import torch
import numpy as np
from typing import List
from transformers import BertTokenizer, BertModel


class DeepBERTEmbeddingPipeline:

    def __init__(self, model_name: str = "bert-base-uncased"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertModel.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()

    
    # Tokenization Logic
    

    def tokenize_text(self, text: str):
        encoded = self.tokenizer(
            text,
            max_length=512,
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        )

        return {
            "input_ids": encoded["input_ids"].to(self.device),
            "attention_mask": encoded["attention_mask"].to(self.device),
        }


    # Embedding Extraction
   

    def forward_pass(self, tokenized_input):
        with torch.no_grad():
            outputs = self.model(
                input_ids=tokenized_input["input_ids"],
                attention_mask=tokenized_input["attention_mask"],
                output_hidden_states=True
            )
        return outputs

    def extract_cls_embedding(self, outputs):
        cls_embedding = outputs.last_hidden_state[:, 0, :]
        return cls_embedding.squeeze().cpu().numpy()

    def extract_layerwise_mean(self, outputs):
        """
        Aggregate information from multiple layers.
        """
        hidden_states = outputs.hidden_states
        selected_layers = hidden_states[-4:]

        stacked = torch.stack(selected_layers)
        mean_embedding = torch.mean(stacked, dim=0)
        cls_mean = mean_embedding[:, 0, :]
        return cls_mean.squeeze().cpu().numpy()

   
    # Public Interface
    

    def generate_embedding(self, text: str, strategy: str = "cls"):
        tokens = self.tokenize_text(text)
        outputs = self.forward_pass(tokens)

        if strategy == "cls":
            return self.extract_cls_embedding(outputs)
        elif strategy == "layer_mean":
            return self.extract_layerwise_mean(outputs)
        else:
            raise ValueError("Unsupported embedding strategy")
