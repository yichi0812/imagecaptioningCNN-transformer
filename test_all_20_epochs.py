import torch
import pickle
import sys
import os
from PIL import Image
import requests
from io import BytesIO

sys.path.insert(0, '/mnt/data/clean_cnn_transformer')
from model import CNNTransformerCaptioning

# Load vocabulary
with open('/mnt/data/image-captioning-neural-networks/data/vocabulary.pkl', 'rb') as f:
    vocab_dict = pickle.load(f)

word2idx = vocab_dict['word2idx']
idx2word = vocab_dict['idx2word']
vocab_size = len(word2idx)

print(f"Vocabulary size: {vocab_size}")
print("="*80)

# 8 test image URLs (diverse scenes from Pexels)
test_images = [
    ("Dog in park", "https://images.pexels.com/photos/1108099/pexels-photo-1108099.jpeg?w=400"),
    ("City street", "https://images.pexels.com/photos/378570/pexels-photo-378570.jpeg?w=400"),
    ("Beach sunset", "https://images.pexels.com/photos/189349/pexels-photo-189349.jpeg?w=400"),
    ("Cat on sofa", "https://images.pexels.com/photos/416160/pexels-photo-416160.jpeg?w=400"),
    ("Mountain landscape", "https://images.pexels.com/photos/417074/pexels-photo-417074.jpeg?w=400"),
    ("Food on table", "https://images.pexels.com/photos/376464/pexels-photo-376464.jpeg?w=400"),
    ("Person with umbrella", "https://images.pexels.com/photos/1529360/pexels-photo-1529360.jpeg?w=400"),
    ("Train station", "https://images.pexels.com/photos/1591373/pexels-photo-1591373.jpeg?w=400")
]

print("Test Images:")
for i, (desc, url) in enumerate(test_images, 1):
    print(f"{i}. {desc}")
print("\n" + "="*80)

# Download and save images to /dev/shm (RAM disk)
print("\nDownloading test images to /dev/shm...")
image_paths = []
for i, (desc, url) in enumerate(test_images, 1):
    try:
        response = requests.get(url, timeout=10)
        img = Image.open(BytesIO(response.content)).convert('RGB')
        temp_path = f'/dev/shm/test_img_{i}.jpg'
        img.save(temp_path)
        image_paths.append((desc, temp_path))
        print(f"  Downloaded: {desc}")
    except Exception as e:
        print(f"  ERROR downloading {desc}: {str(e)}")
        image_paths.append((desc, None))

print("\n" + "="*80)

# Test all 20 epochs
for epoch in range(1, 21):
    print(f"\nEPOCH {epoch}")
    print("="*80)
    
    checkpoint_path = f'/mnt/data/models/clean_cnn_transformer/epoch_{epoch}.pth'
    
    # Load model
    model = CNNTransformerCaptioning(
        vocab_size=vocab_size,
        d_model=512,
        nhead=8,
        num_layers=6,
        dim_feedforward=2048,
        max_len=5000
    )
    
    checkpoint = torch.load(checkpoint_path, map_location='cuda')
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to('cuda')
    model.eval()
    
    print(f"Train Loss: {checkpoint.get('train_loss', 'N/A'):.4f}")
    print(f"Val Loss: {checkpoint.get('val_loss', 'N/A'):.4f}\n")
    
    # Generate captions for all 8 images
    for i, (desc, img_path) in enumerate(image_paths, 1):
        if img_path and os.path.exists(img_path):
            try:
                caption = model.generate_caption(img_path, word2idx, idx2word, max_len=20)
                print(f"{i}. {desc}")
                print(f"   PRED-> {caption}\n")
            except Exception as e:
                print(f"{i}. {desc}")
                print(f"   ERROR: {str(e)}\n")
        else:
            print(f"{i}. {desc}")
            print(f"   SKIPPED (download failed)\n")
    
    del model
    torch.cuda.empty_cache()
    
    print(f"Epoch {epoch} completed\n")

# Cleanup
print("\nCleaning up temporary files...")
for desc, img_path in image_paths:
    if img_path and os.path.exists(img_path):
        os.remove(img_path)

print("\n" + "="*80)
print("ALL 20 EPOCHS TESTED SUCCESSFULLY!")
print("="*80)
