#!/usr/bin/env python3
"""
CNN+Transformer Image Captioning - Inference Script
Generate captions for images using the trained model
"""

import torch
import pickle
import argparse
from model import CNNTransformerCaptioning

def load_model(checkpoint_path, vocab_path, device='cuda'):
    """Load the trained model and vocabulary"""
    
    # Load vocabulary
    with open(vocab_path, 'rb') as f:
        vocab = pickle.load(f)
    
    word2idx = vocab['word2idx']
    idx2word = vocab['idx2word']
    vocab_size = len(word2idx)
    
    # Initialize model
    model = CNNTransformerCaptioning(
        embed_size=512,
        vocab_size=vocab_size,
        num_heads=8,
        num_layers=6,
        dropout=0.1
    )
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Move to device
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    
    return model, word2idx, idx2word

def generate_caption(model, image_path, word2idx, idx2word, max_len=20):
    """Generate caption for an image"""
    caption = model.generate_caption(image_path, word2idx, idx2word, max_len)
    return caption

def main():
    parser = argparse.ArgumentParser(description='Generate image captions')
    parser.add_argument('--image', type=str, required=True, help='Path to input image')
    parser.add_argument('--checkpoint', type=str, default='checkpoints/epoch_20.pth', 
                       help='Path to model checkpoint')
    parser.add_argument('--vocab', type=str, default='vocabulary.pkl',
                       help='Path to vocabulary file')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use (cuda or cpu)')
    parser.add_argument('--max-len', type=int, default=20,
                       help='Maximum caption length')
    
    args = parser.parse_args()
    
    # Load model
    print(f"Loading model from {args.checkpoint}...")
    model, word2idx, idx2word = load_model(args.checkpoint, args.vocab, args.device)
    print(f"Model loaded successfully! Vocabulary size: {len(word2idx)}")
    
    # Generate caption
    print(f"\nGenerating caption for {args.image}...")
    caption = generate_caption(model, args.image, word2idx, idx2word, args.max_len)
    
    # Print result
    print(f"\n{'='*60}")
    print(f"Image: {args.image}")
    print(f"Caption: {caption}")
    print(f"{'='*60}\n")

if __name__ == '__main__':
    main()
