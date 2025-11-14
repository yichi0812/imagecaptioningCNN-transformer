#!/usr/bin/env python3


import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import json
import pickle
import argparse
import os
from datetime import datetime
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer

try:
    from pycocoevalcap.cider.cider import Cider
except ImportError:
    print("Warning: pycocoevalcap not installed. CIDEr metric will not be available.")
    Cider = None

from model import CNNTransformerCaptioning


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
    return model, checkpoint


def generate_caption(model, image_tensor, word2idx, idx2word, device, max_length=50):
    """Generate caption for an image."""
    model.eval()
    with torch.no_grad():
        image_tensor = image_tensor.unsqueeze(0).to(device)
        start_token = word2idx.get('<start>', word2idx.get('<START>', 1))
        caption = [start_token]
        
        for _ in range(max_length):
            caption_tensor = torch.LongTensor(caption).unsqueeze(0).to(device)
            output = model(image_tensor, caption_tensor)
            next_token = output[0, -1].argmax().item()
            
            if next_token == word2idx.get('<end>', word2idx.get('<END>', 2)):
                break
            caption.append(next_token)
        
        # Convert to words
        words = []
        for idx in caption[1:]:  # Skip start token
            word = idx2word.get(idx, '<UNK>')
            if word not in ['<start>', '<START>', '<end>', '<END>', '<pad>', '<PAD>']:
                words.append(word)
        
        return ' '.join(words)


def calculate_metrics(generated, references):
    """Calculate BLEU, METEOR, ROUGE-L metrics."""
    metrics = {}
    
    # Tokenize
    gen_tokens = generated.lower().split()
    ref_tokens_list = [ref.lower().split() for ref in references]
    
    # BLEU scores
    smoothing = SmoothingFunction().method1
    metrics['bleu1'] = sentence_bleu(ref_tokens_list, gen_tokens, weights=(1, 0, 0, 0), smoothing_function=smoothing)
    metrics['bleu2'] = sentence_bleu(ref_tokens_list, gen_tokens, weights=(0.5, 0.5, 0, 0), smoothing_function=smoothing)
    metrics['bleu3'] = sentence_bleu(ref_tokens_list, gen_tokens, weights=(0.33, 0.33, 0.33, 0), smoothing_function=smoothing)
    metrics['bleu4'] = sentence_bleu(ref_tokens_list, gen_tokens, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smoothing)
    
    # METEOR
    try:
        meteor_scores = [meteor_score([ref], generated) for ref in references]
        metrics['meteor'] = sum(meteor_scores) / len(meteor_scores)
    except:
        metrics['meteor'] = 0.0
    
    # ROUGE-L
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    rouge_scores = [scorer.score(ref, generated)['rougeL'].fmeasure for ref in references]
    metrics['rouge_l'] = sum(rouge_scores) / len(rouge_scores)
    
    # Accuracy (word overlap >= 50%)
    gen_set = set(gen_tokens)
    ref_sets = [set(ref.split()) for ref in ref_tokens_list]
    max_overlap = max([len(gen_set & ref_set) / max(len(gen_set), 1) for ref_set in ref_sets])
    metrics['accuracy'] = 1 if max_overlap >= 0.5 else 0
    
    return metrics


def calculate_cider(generated_captions, reference_captions):
    """Calculate CIDEr score using pycocoevalcap."""
    if Cider is None:
        return 0.0
    
    # Format for CIDEr calculation
    gts = {}
    res = {}
    for i, (gen, refs) in enumerate(zip(generated_captions, reference_captions)):
        gts[i] = [{'caption': ref} for ref in refs]
        res[i] = [{'caption': gen}]
    
    cider_scorer = Cider()
    score, _ = cider_scorer.compute_score(gts, res)
    return score


def evaluate_epoch(epoch, args, word2idx, idx2word, image_ids, annotations, device):
    """Evaluate a single epoch."""
    checkpoint_path = os.path.join(args.checkpoints, f'epoch_{epoch}.pth')
    
    if not os.path.exists(checkpoint_path):
        print(f"Checkpoint not found: {checkpoint_path}")
        return None
    
    print(f"\n{'='*60}")
    print(f"Evaluating Epoch {epoch}")
    print(f"{'='*60}")
    print(f"Loading checkpoint: {checkpoint_path}")
    
    model, checkpoint = load_model(checkpoint_path, len(word2idx), device)
    
    # Image preprocessing
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    # Evaluate
    all_metrics = []
    generated_captions = []
    reference_captions = []
    correct = 0
    
    for idx, img_id in enumerate(image_ids[:args.num_images]):
        img_filename = f"COCO_val2014_{img_id:012d}.jpg"
        img_path = os.path.join(args.images, img_filename)
        
        if not os.path.exists(img_path):
            continue
        
        # Load and process image
        image = Image.open(img_path).convert('RGB')
        image_tensor = transform(image)
        
        # Generate caption
        generated = generate_caption(model, image_tensor, word2idx, idx2word, device)
        references = annotations[img_id]
        
        # Calculate metrics
        metrics = calculate_metrics(generated, references)
        all_metrics.append(metrics)
        generated_captions.append(generated)
        reference_captions.append(references)
        correct += metrics['accuracy']
        
        if (idx + 1) % 100 == 0:
            avg_bleu4 = sum(m['bleu4'] for m in all_metrics) / len(all_metrics)
            print(f"Progress: {idx + 1}/{args.num_images} images, Avg BLEU-4: {avg_bleu4:.4f}")
    
    # Calculate CIDEr
    cider_score = calculate_cider(generated_captions, reference_captions)
    
    # Average metrics
    result = {
        'epoch': epoch,
        'bleu1': sum(m['bleu1'] for m in all_metrics) / len(all_metrics),
        'bleu2': sum(m['bleu2'] for m in all_metrics) / len(all_metrics),
        'bleu3': sum(m['bleu3'] for m in all_metrics) / len(all_metrics),
        'bleu4': sum(m['bleu4'] for m in all_metrics) / len(all_metrics),
        'meteor': sum(m['meteor'] for m in all_metrics) / len(all_metrics),
        'rouge_l': sum(m['rouge_l'] for m in all_metrics) / len(all_metrics),
        'cider': cider_score,
        'accuracy': (correct / len(all_metrics)) * 100,
        'correct': correct,
        'total': len(all_metrics),
        'train_loss': checkpoint.get('train_loss', 'N/A'),
        'val_loss': checkpoint.get('val_loss', 'N/A')
    }
    
    print(f"\nEpoch {epoch} Results:")
    print(f"  BLEU-1: {result['bleu1']:.4f}")
    print(f"  BLEU-2: {result['bleu2']:.4f}")
    print(f"  BLEU-3: {result['bleu3']:.4f}")
    print(f"  BLEU-4: {result['bleu4']:.4f}")
    print(f"  METEOR: {result['meteor']:.4f}")
    print(f"  ROUGE-L: {result['rouge_l']:.4f}")
    print(f"  CIDEr: {result['cider']:.4f}")
    print(f"  Accuracy: {result['accuracy']:.2f}%")
    
    return result


def main():
    parser = argparse.ArgumentParser(description='Calculate comprehensive metrics for image captioning')
    parser.add_argument('--annotations', required=True, help='Path to COCO annotations JSON file')
    parser.add_argument('--images', required=True, help='Path to COCO validation images directory')
    parser.add_argument('--vocab', default='../vocabulary.pkl', help='Path to vocabulary file')
    parser.add_argument('--checkpoints', default='../checkpoints/', help='Path to checkpoints directory')
    parser.add_argument('--output', default='comprehensive_metrics.txt', help='Output file path')
    parser.add_argument('--num-images', type=int, default=1000, help='Number of images to evaluate')
    parser.add_argument('--epochs', nargs='+', type=int, default=list(range(1, 21)), help='Epochs to evaluate')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu', help='Device to use')
    args = parser.parse_args()
    
    device = torch.device(args.device)
    print(f"Using device: {device}")
    
    # Load vocabulary
    print("Loading vocabulary...")
    with open(args.vocab, 'rb') as f:
        vocab = pickle.load(f)
        word2idx = vocab['word2idx']
        idx2word = vocab['idx2word']
    print(f"Vocabulary size: {len(word2idx)}")
    
    # Load annotations
    print("Loading COCO annotations...")
    with open(args.annotations, 'r') as f:
        coco_data = json.load(f)
    
    annotations = {}
    for ann in coco_data['annotations']:
        img_id = ann['image_id']
        caption = ann['caption'].strip().lower()
        if img_id not in annotations:
            annotations[img_id] = []
        annotations[img_id].append(caption)
    
    image_ids = list(annotations.keys())
    print(f"Testing on {min(args.num_images, len(image_ids))} images")
    
    # Evaluate all epochs
    print(f"\n{'='*60}")
    print("Image Captioning Model - Comprehensive Metrics Calculation")
    print(f"{'='*60}")
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Device: {device}")
    print(f"Number of images: {args.num_images}")
    print(f"Vocabulary size: {len(word2idx)}")
    print(f"{'='*60}\n")
    
    results = []
    for epoch in args.epochs:
        result = evaluate_epoch(epoch, args, word2idx, idx2word, image_ids, annotations, device)
        if result:
            results.append(result)
    
    # Write results
    print(f"\n{'='*60}")
    print("Writing results to file...")
    print(f"{'='*60}\n")
    
    with open(args.output, 'w') as f:
        f.write("="*120 + "\n")
        f.write("Image Captioning Model - Comprehensive Metrics Results\n")
        f.write("="*120 + "\n")
        f.write(f"Evaluation Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Number of Images: {args.num_images}\n")
        f.write(f"Vocabulary Size: {len(word2idx)}\n")
        f.write(f"Device: {device}\n")
        f.write("="*120 + "\n\n")
        
        # Table header
        f.write(f"{'Epoch':<8} {'BLEU-1':<10} {'BLEU-2':<10} {'BLEU-3':<10} {'BLEU-4':<10} {'METEOR':<10} {'ROUGE-L':<10} {'CIDEr':<10} {'Accuracy':<12} {'Train Loss':<12} {'Val Loss':<12}\n")
        f.write("-"*120 + "\n")
        
        for result in results:
            f.write(f"{result['epoch']:<8} ")
            f.write(f"{result['bleu1']:<10.4f} ")
            f.write(f"{result['bleu2']:<10.4f} ")
            f.write(f"{result['bleu3']:<10.4f} ")
            f.write(f"{result['bleu4']:<10.4f} ")
            f.write(f"{result['meteor']:<10.4f} ")
            f.write(f"{result['rouge_l']:<10.4f} ")
            f.write(f"{result['cider']:<10.4f} ")
            f.write(f"{result['accuracy']:<12.2f}% ")
            train_loss = f"{result['train_loss']:.4f}" if isinstance(result['train_loss'], float) else str(result['train_loss'])
            val_loss = f"{result['val_loss']:.4f}" if isinstance(result['val_loss'], float) else str(result['val_loss'])
            f.write(f"{train_loss:<12} ")
            f.write(f"{val_loss:<12}\n")
        
        f.write("\n" + "="*120 + "\n")
        f.write("Summary:\n")
        f.write("-"*120 + "\n")
        
        if results:
            best_bleu4 = max(results, key=lambda x: x['bleu4'])
            best_cider = max(results, key=lambda x: x['cider'])
            best_acc = max(results, key=lambda x: x['accuracy'])
            
            f.write(f"Best BLEU-4: {best_bleu4['bleu4']:.4f} (Epoch {best_bleu4['epoch']})\n")
            f.write(f"Best CIDEr: {best_cider['cider']:.4f} (Epoch {best_cider['epoch']})\n")
            f.write(f"Best Accuracy: {best_acc['accuracy']:.2f}% (Epoch {best_acc['epoch']})\n")
        
        f.write("="*120 + "\n")
    
    print(f"Results saved to: {args.output}")
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("\nDone!")


if __name__ == "__main__":
    main()
