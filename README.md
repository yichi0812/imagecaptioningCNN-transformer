# Image Captioning with CNN + Transformer

Deep learning model for automatic image captioning using ResNet50 encoder and Transformer decoder.

## Model Architecture
- Encoder: ResNet50 (pretrained on ImageNet )
- Decoder: Transformer (3 layers, 8 heads, 512 d_model)
- Vocabulary: 11,919 words
- Dataset: MS COCO 2014

## Best Performance (Epoch 12)
- BLEU-4: 0.2112
- CIDEr: 0.5423
- Accuracy: 45.30%

## Usage
python test_image.py your_image.jpg

See comprehensive_metrics_results.txt for complete results.
