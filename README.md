# CNN+Transformer Image Captioning

A deep learning model for automatic image captioning using CNN (ResNet50) for image encoding and Transformer decoder for caption generation.

## Model Architecture

- **Encoder:** ResNet50 (pretrained on ImageNet)
- **Decoder:** Transformer with 6 layers, 8 attention heads
- **Embedding Size:** 512
- **Vocabulary Size:** 11,919 words
- **Training:** 20 epochs on COCO dataset

## Key Features

✅ **Causal Masking Fix:** Properly implemented causal masking in the Transformer decoder to prevent attention to future tokens  
✅ **Device Compatibility:** Automatic GPU/CPU detection and proper device handling  
✅ **Pretrained CNN:** Uses ResNet50 pretrained on ImageNet for robust feature extraction  
✅ **Positional Encoding:** Proper positional encoding for sequence understanding  

## Training Results

| Metric | Epoch 1 | Epoch 20 | Improvement |
|--------|---------|----------|-------------|
| **Train Loss** | 3.0322 | 1.9126 | **36.9%** ↓ |
| **Val Loss** | 2.6524 | 2.4578 | **7.3%** ↓ |

## Installation

```bash
# Clone the repository
git clone https://github.com/yichi0812/imagecaptioningCNN-transformer.git
cd imagecaptioningCNN-transformer

# Install dependencies
pip install -r requirements.txt
```

## Download Model Files

Due to file size limitations, the checkpoint files and vocabulary need to be downloaded separately:

### Required Files:
1. **vocabulary.pkl** (~500KB) - Word mappings
2. **Checkpoint files** (~390MB each):
   - `epoch_1.pth` through `epoch_20.pth`
   - Recommended: Download `epoch_20.pth` (final trained model)

### Download Links:
*[Files will be provided via Google Drive or similar service]*

Place the downloaded files in the following structure:
```
imagecaptioningCNN-transformer/
├── model.py
├── generate_caption.py
├── requirements.txt
├── README.md
├── vocabulary.pkl          # Download this
└── checkpoints/
    ├── epoch_1.pth         # Download these
    ├── epoch_2.pth
    ...
    └── epoch_20.pth
```

## Usage

### Basic Usage

```bash
python generate_caption.py --image path/to/your/image.jpg
```

### Advanced Options

```bash
python generate_caption.py \
    --image path/to/image.jpg \
    --checkpoint checkpoints/epoch_20.pth \
    --vocab vocabulary.pkl \
    --device cuda \
    --max-len 20
```

### Parameters:
- `--image`: Path to input image (required)
- `--checkpoint`: Path to model checkpoint (default: `checkpoints/epoch_20.pth`)
- `--vocab`: Path to vocabulary file (default: `vocabulary.pkl`)
- `--device`: Device to use - `cuda` or `cpu` (default: `cuda`)
- `--max-len`: Maximum caption length (default: 20)

## Example Output

```
Loading model from checkpoints/epoch_20.pth...
Model loaded successfully! Vocabulary size: 11919

Generating caption for test_image.jpg...

============================================================
Image: test_image.jpg
Caption: a dog walking on a sidewalk next to a building.
============================================================
```

## Python API Usage

```python
import torch
import pickle
from model import CNNTransformerCaptioning

# Load vocabulary
with open('vocabulary.pkl', 'rb') as f:
    vocab = pickle.load(f)

word2idx = vocab['word2idx']
idx2word = vocab['idx2word']

# Initialize model
model = CNNTransformerCaptioning(
    embed_size=512,
    vocab_size=len(word2idx),
    num_heads=8,
    num_layers=6,
    dropout=0.1
)

# Load checkpoint
checkpoint = torch.load('checkpoints/epoch_20.pth')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Generate caption
caption = model.generate_caption('image.jpg', word2idx, idx2word)
print(f"Caption: {caption}")
```

## Model Performance

### Qualitative Results (Epoch 20):

1. **Teddy bear scene:** "a stuffed bear is sitting in a car seat"
2. **Motorcycle:** "a motorcycle parked in a parking lot next to a building."
3. **Dog:** "a dog walking on a sidewalk next to a building."
4. **Furniture:** "a wooden bunk bed with a drawer underneath."
5. **Living room:** "a living room with a couch and a coffee table"
6. **Birds:** "a group of birds perched on a tree branch."

### Key Improvements:
- ✅ Accurate object recognition
- ✅ Proper spatial relationships ("next to", "underneath")
- ✅ Natural language generation
- ✅ Detailed descriptions
- ✅ No repetition or nonsensical output

## Technical Details

### Bug Fix: Causal Masking

The original model had a critical bug where the causal mask was not properly applied in the Transformer decoder. This caused the model to "see" future tokens during training, leading to nonsensical output with excessive `<UNK>` tokens.

**Fix implemented:**
```python
def generate_square_subsequent_mask(self, sz):
    """Generate a causal mask to prevent attention to future tokens"""
    mask = torch.triu(torch.ones(sz, sz), diagonal=1)
    mask = mask.masked_fill(mask == 1, float('-inf'))
    return mask
```

This ensures the model can only attend to previous tokens during caption generation, which is essential for autoregressive sequence generation.

### Device Handling

Proper device handling was implemented to ensure the model works on both GPU and CPU:
```python
device = next(self.parameters()).device
image = image.to(device)
```

## System Requirements

- **GPU:** NVIDIA GPU with CUDA support (recommended)
- **CPU:** Works on CPU but slower
- **RAM:** 8GB minimum, 16GB recommended
- **Storage:** ~8GB for all checkpoints, ~400MB for single checkpoint

## Training Details

- **Dataset:** COCO 2014
- **Optimizer:** Adam
- **Learning Rate:** 0.0001
- **Batch Size:** 64
- **Epochs:** 20
- **Training Time:** ~8 hours on single GPU

## Citation

If you use this model in your research, please cite:

```bibtex
@misc{cnn_transformer_captioning_2025,
  title={CNN+Transformer Image Captioning with Causal Masking Fix},
  author={Your Name},
  year={2025},
  howpublished={\url{https://github.com/yichi0812/imagecaptioningCNN-transformer}}
}
```

## License

MIT License

## Acknowledgments

- COCO Dataset: [https://cocodataset.org/](https://cocodataset.org/)
- PyTorch: [https://pytorch.org/](https://pytorch.org/)
- ResNet50: [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)
- Transformer: [Attention Is All You Need](https://arxiv.org/abs/1706.03762)

## Contact

For questions or issues, please open an issue on GitHub.
