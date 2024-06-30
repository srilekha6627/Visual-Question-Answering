import torch
import torch.nn as nn
import pickle 
from typing import Dict, List, Optional, Tuple
from transformers import AutoTokenizer
from PIL import Image
from transformers import (
    # Preprocessing / Common
    AutoTokenizer, AutoFeatureExtractor,
    # Text & Image Models (Now, image transformers like ViTModel, DeiTModel, BEiT can also be loaded using AutoModel)
    AutoModel,            
    # Training / Evaluation
    TrainingArguments, Trainer,
    # Misc
    logging
)
import os
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"


with open('ans_space.pkl', 'rb') as f:
    answer_space = pickle.load(f)

class MultimodalVQAModel(nn.Module):
    def __init__(
            self,
            num_labels: int = len(answer_space),
            intermediate_dim: int = 512,
            pretrained_text_name: str = 'bert-base-uncased',
            pretrained_image_name: str = 'google/vit-base-patch16-224-in21k'):
     
        super(MultimodalVQAModel, self).__init__()
        self.num_labels = num_labels
        self.pretrained_text_name = pretrained_text_name
        self.pretrained_image_name = pretrained_image_name
        
        self.text_encoder = AutoModel.from_pretrained(
            self.pretrained_text_name,
        )
        self.image_encoder = AutoModel.from_pretrained(
            self.pretrained_image_name,
        )
        self.fusion = nn.Sequential(
            nn.Linear(self.text_encoder.config.hidden_size + self.image_encoder.config.hidden_size, intermediate_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
        )
        
        self.classifier = nn.Linear(intermediate_dim, self.num_labels)
        
        self.criterion = nn.CrossEntropyLoss()
    
    def forward(
            self,
            input_ids: torch.LongTensor,
            pixel_values: torch.FloatTensor,
            attention_mask: Optional[torch.LongTensor] = None,
            token_type_ids: Optional[torch.LongTensor] = None,
            labels: Optional[torch.LongTensor] = None):
        
        encoded_text = self.text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_dict=True,
        )
        encoded_image = self.image_encoder(
            pixel_values=pixel_values,
            return_dict=True,
        )
        fused_output = self.fusion(
            torch.cat([encoded_text['pooler_output'], encoded_image['pooler_output'],],dim=1)
        )
        logits = self.classifier(fused_output)
        
        out = {
            "logits": logits
        }
        if labels is not None:
            loss = self.criterion(logits, labels)
            out["loss"] = loss
        
        return out

unexpected_keys = ['text_encoder.embeddings.position_ids']
for key in unexpected_keys:
    del torch.load('MM10_model.pth', map_location=torch.device('cpu'))[key]

# Load the modified state dictionary
model = MultimodalVQAModel()
model.load_state_dict(torch.load('MM10_model.pth', map_location=torch.device('cpu')), strict=False)

# model = MultimodalVQAModel()  # Assuming MultimodalVQAModel is your model class
# model.load_state_dict(torch.load('MM10_model.pth', map_location=torch.device('cpu')))

texts = "What is on the left side of the nightstand"
# preprocess text
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
encoded_text = tokenizer(
            text=texts,
            padding='longest',
            max_length=24,
            truncation=True,
            return_tensors='pt',
            return_token_type_ids=True,
            return_attention_mask=True,
        )
input_ids = encoded_text['input_ids']
token_type_ids = encoded_text['token_type_ids']
attention_mask = encoded_text['attention_mask']
print("-------------------------------------------------------------")
print(input_ids)
print("-------------------------------------------------------------")
print(token_type_ids)
print("-------------------------------------------------------------")
print(attention_mask)
print("-------------------------------------------------------------")
processor = AutoFeatureExtractor.from_pretrained('google/vit-base-patch16-224-in21k')
processed_images = processor(
            images=Image.open(os.path.join( "image1074" + ".png")).convert('RGB'),
            return_tensors="pt",
        )
pixel_values = processed_images['pixel_values'].squeeze()
print(pixel_values)
print("-------------------------------------------------------------")

# Ensure input tensor has the correct shape
if len(pixel_values.shape) == 3:  # If it's a grayscale image
    # Add a singleton dimension for num_channels
    pixel_values = pixel_values.unsqueeze(0)  # Assuming batch_size is 1
elif len(pixel_values.shape) == 4:  # If it's an RGB image
    pass  # No modification needed
else:
    raise ValueError("Unexpected shape for pixel_values tensor")

model.eval()
output = model(input_ids, pixel_values, attention_mask, token_type_ids)

print(output)

# Get the top 5 predicted classes and their corresponding logits
top5_preds = torch.topk(output["logits"], k=5, dim=1)
top5_indices = top5_preds.indices.cpu().numpy()[0]
#top5_logits = top5_preds.values.cpu().numpy()[0]
top5_logits = top5_preds.values.detach().cpu().numpy()[0]
# Apply softmax to logits to get probabilities
probabilities = torch.nn.functional.softmax(torch.tensor(top5_logits), dim=0)

# Convert indices to answer space and print along with confidence
print("Top 5 Predicted Answers with Confidence:")
for j, (pred_idx, probability) in enumerate(zip(top5_indices, probabilities)):
    answer = answer_space[pred_idx]
    confidence = probability.item() * 100
    print(f"{j+1}. {answer} (Confidence: {confidence:.2f}%)")
