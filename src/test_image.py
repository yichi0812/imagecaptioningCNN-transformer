#!/usr/bin/env python3
"""
Simple Image Captioning Test Script
Usage: python test_image.py <image_path> [--epoch EPOCH_NUM]
"""

import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import pickle
import argparse
import os

from src.model import CNNTransformerCaptioning

def load_model(checkpoint_path, vocab_size, device):
    """Load model from checkpoint."""
    model = CNNTransformerCaptioning(
        vocab_size=vocab_size,
        d_model=512,
        nhead=8,
        num_layers=3,
        dim_feedforward=2048,
        max_len=5000
    ).to(device)
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model

def generate_caption(model, image_tensor, word2idx, idx2word, device, max_length=50):
    """Generate caption for an image."""
    model.eval()
    with torch.no_grad():
        image_tensor = image_tensor.unsqueeze(0).to(device)
        caption = [word2idx.get('<start>', word2idx.get('<START>', 1))]
        
        for _ in range(max_length):
            caption_tensor = torch.LongTensor(caption).unsqueeze(0).to(device)
            outputs = model(image_tensor, caption_tensor)
            predicted_idx = outputs[0, -1, :].argmax().item()
            caption.append(predicted_idx)
            
            if predicted_idx == word2idx.get('<end>', word2idx.get('<END>', 2)):
                break
        
        words = []
        for idx in caption[1:]:
            word = idx2word.get(idx, '<unk>')
            if word in ['<end>', '<END>']:
                break
            if word not in ['<start>', '<START>', '<pad>', '<PAD>', '<unk>', '<UNK>']:
                words.append(word)
        
        return ' '.join(words)

def main():
    parser = argparse.ArgumentParser(description='Generate caption for an image')
    parser.add_argument('image_path', type=str, help='Path to the image file')
    parser.add_argument('--epoch', type=int, default=12, help='Epoch number to use (default: 12, the best model)')
    args = parser.parse_args()
    
    # Check if image exists
    if not os.path.exists(args.image_path):
        print(f"Error: Image not found at {args.image_path}")
        return
    
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load vocabulary
    print("Loading vocabulary...")
    with open('vocabulary.pkl', 'rb') as f:
        vocab = pickle.load(f)
        word2idx = vocab['word2idx']
        idx2word = vocab['idx2word']
        vocab_size = len(word2idx)
    
    # Load model
    checkpoint_path = f'checkpoints/epoch_{args.epoch}.pth'
    if not os.path.exists(checkpoint_path):
        print(f"Error: Checkpoint not found at {checkpoint_path}")
        return
    
    print(f"Loading model from epoch {args.epoch}...")
    model = load_model(checkpoint_path, vocab_size, device)
    
    # Image preprocessing
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    # Load and process image
    print(f"Processing image: {args.image_path}")
    image = Image.open(args.image_path).convert('RGB')
    image_tensor = transform(image)
    
    # Generate caption
    print("Generating caption...")
    caption = generate_caption(model, image_tensor, word2idx, idx2word, device)
    
    print("\n" + "="*60)
    print(f"Image: {args.image_path}")
    print(f"Model: Epoch {args.epoch}")
    print(f"Caption: {caption}")
    print("="*60)

if __name__ == "__main__":
    main()
