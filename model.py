import torch
import torch.nn as nn
import torchvision.models as models
from PIL import Image
from torchvision import transforms
import math

class CNNTransformerCaptioning(nn.Module):
    def __init__(self, embed_size, vocab_size, num_heads=8, num_layers=6, dropout=0.1):
        super(CNNTransformerCaptioning, self).__init__()
        
        # CNN Encoder (ResNet50)
        resnet = models.resnet50(pretrained=True)
        modules = list(resnet.children())[:-2]  # Remove avgpool and fc
        self.cnn = nn.Sequential(*modules)
        
        # Freeze CNN parameters
        for param in self.cnn.parameters():
            param.requires_grad = False
            
        # Adaptive pooling to get fixed size output
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Linear layer to project CNN features to embedding size
        self.fc_encoder = nn.Linear(2048, embed_size)
        
        # Transformer Decoder
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.pos_encoder = PositionalEncoding(embed_size, dropout)
        
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_size,
            nhead=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        
        # Output layer
        self.fc_out = nn.Linear(embed_size, vocab_size)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, images, captions):
        # CNN encoding
        features = self.cnn(images)
        features = self.adaptive_pool(features)
        features = features.view(features.size(0), -1)
        memory = self.fc_encoder(features).unsqueeze(1)
        
        # Caption embedding
        tgt = self.embedding(captions)
        tgt = self.pos_encoder(tgt)
        
        # Create causal mask
        tgt_mask = self.generate_square_subsequent_mask(captions.size(1)).to(captions.device)
        
        # Transformer decoding
        output = self.transformer_decoder(tgt, memory, tgt_mask=tgt_mask)
        output = self.fc_out(output)
        
        return output
    
    def generate_square_subsequent_mask(self, sz):
        """Generate a causal mask to prevent attention to future tokens"""
        mask = torch.triu(torch.ones(sz, sz), diagonal=1)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        return mask
    
    def generate_caption(self, image_path, word2idx, idx2word, max_len=20):
        """Generate caption for a single image"""
        self.eval()
        
        # Load and transform image
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
        
        image = Image.open(image_path).convert('RGB')
        image = transform(image).unsqueeze(0)
        
        with torch.no_grad():
            # Move image to same device as model
            device = next(self.parameters()).device
            image = image.to(device)
            
            # Get CNN features
            features = self.cnn(image)
            features = self.adaptive_pool(features)
            features = features.view(features.size(0), -1)
            memory = self.fc_encoder(features).unsqueeze(1)
            
            # Start with <START> token
            caption = [word2idx.get('<START>', 1)]
            
            for _ in range(max_len):
                tgt = torch.LongTensor([caption]).to(device)
                tgt_embedded = self.embedding(tgt)
                tgt_embedded = self.pos_encoder(tgt_embedded)
                
                # Create causal mask
                tgt_mask = self.generate_square_subsequent_mask(len(caption)).to(device)
                
                # Decode
                output = self.transformer_decoder(tgt_embedded, memory, tgt_mask=tgt_mask)
                output = self.fc_out(output)
                
                # Get next word
                next_word_idx = output[0, -1].argmax().item()
                
                # Stop if <END> token
                if next_word_idx == word2idx.get('<END>', 2):
                    break
                    
                caption.append(next_word_idx)
            
            # Convert indices to words
            words = [idx2word.get(idx, '<UNK>') for idx in caption[1:]]  # Skip <START>
            return ' '.join(words)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)
