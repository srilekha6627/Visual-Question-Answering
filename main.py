from flask import Flask, request, render_template
import base64
from PIL import Image
from io import BytesIO
from modell import MultimodalVQAModel, answer_space
import torch
import torch.nn as nn
import pickle 
from typing import Dict, List, Optional, Tuple
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

app = Flask(__name__)

import os
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"


# Load the model and tokenizer
model = MultimodalVQAModel()
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
processor = AutoFeatureExtractor.from_pretrained('google/vit-base-patch16-224-in21k')

def convert_image_to_base64(image):
    pil_img = Image.open(image)
    # Convert RGBA to RGB if the image has an alpha channel
    if pil_img.mode == 'RGBA':
        pil_img = pil_img.convert('RGB')

    img_buffer = BytesIO()
    pil_img.save(img_buffer, format='JPEG')
    img_buffer.seek(0)

    image_base64 = base64.b64encode(img_buffer.getvalue()).decode()
    return f"data:image/jpeg;base64,{image_base64}"

@app.route('/', methods=['GET', 'POST'])
def index():
    answer = None
    question = ""
    image_data = None

    if request.method == 'POST':
        image = request.files['image']
        question = request.form['question']

        image_data = convert_image_to_base64(image)

        # Process the image and question to get an answer
        answer = process_vqa(image, question)
    
    return render_template('index.html', answer=answer, question=question, image_data=image_data)

@app.route('/answer', methods=['POST'])
def answer():
    image = request.files['image']
    question = request.form['question']

    # Process the image and question to get an answer
    answer = process_vqa(image, question)
    
    return render_template('result.html', answer=answer)

def process_vqa(image, question):
    # Preprocess the image
    processed_images = processor(
        images=Image.open(image).convert('RGB'),
        return_tensors="pt",
    )
    pixel_values = processed_images['pixel_values'].squeeze()
    # Ensure input tensor has the correct shape
    if len(pixel_values.shape) == 3:  # If it's a grayscale image
        # Add a singleton dimension for num_channels
        pixel_values = pixel_values.unsqueeze(0)  # Assuming batch_size is 1
    elif len(pixel_values.shape) == 4:  # If it's an RGB image
        pass  # No modification needed
    else:
        raise ValueError("Unexpected shape for pixel_values tensor")
    # Preprocess the question
    encoded_text = tokenizer(
        text=question,
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

    # Run the model
    output = model(input_ids, pixel_values, attention_mask, token_type_ids)

    # Get the top 5 predicted classes and their corresponding logits
    top5_preds = torch.topk(output["logits"], k=5, dim=1)
    top5_indices = top5_preds.indices.cpu().numpy()[0]
    top5_logits = top5_preds.values.detach().cpu().numpy()[0]

    # Apply softmax to logits to get probabilities
    probabilities = torch.nn.functional.softmax(torch.tensor(top5_logits), dim=0)

    # Convert indices to answer space and print along with confidence
    top5_answers = []
    for j, (pred_idx, probability) in enumerate(zip(top5_indices, probabilities)):
        answer = answer_space[pred_idx]
        confidence = probability.item() * 100
        top5_answers.append((answer, confidence))

    return top5_answers

if __name__ == '__main__':
    app.run(debug=True)