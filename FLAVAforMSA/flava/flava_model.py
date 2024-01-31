import torch
from torch import nn
from transformers import FlavaModel, FlavaConfig

class FlavaForClassification(nn.Module):
    def __init__(self, num_labels=3, dropout=0.3, freeze_layers = 1):
        super().__init__()
        self.flava = FlavaModel.from_pretrained("facebook/flava-full")

        if freeze_layers > 0:
            self.freeze_flava_layers(freeze_layers)

        embedding_size = self.flava.config.hidden_size

        self.classifier = nn.Sequential(
            nn.Linear(embedding_size, embedding_size), # First layer
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embedding_size, embedding_size // 2), # Second layer
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embedding_size // 2, num_labels) # Output layer
        )

    def freeze_flava_layers(self, freeze=1):
        for param in self.flava.text_model.parameters():
            param.requires_grad = False
        
        for param in self.flava.image_model.parameters():
            param.requires_grad = False

    def create_attention_layer(self, embedding_size, dropout):
        return nn.Sequential(
            nn.Linear(embedding_size, 1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Softmax(dim=1)
        )

    def forward(self, inputs):
        outputs = self.flava(**inputs)
        multimodal_embeddings = outputs.multimodal_embeddings
        global_token_embeddings = multimodal_embeddings[:, 0, :]
        # Pass the global token embeddings through the classifier
        logits = self.classifier(global_token_embeddings)
        return logits

