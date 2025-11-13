import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image


class CNNTransformerCaptioning(nn.Module):
    def __init__(self, vocab_size, d_model=512, nhead=8, num_layers=3, dim_feedforward=2048, max_len=5000):
        super(CNNTransformerCaptioning, self).__init__()
        resnet = models.resnet50(pretrained=True)
        modules = list(resnet.children())[:-2]
        self.cnn = nn.Sequential(*modules)
        for param in self.cnn.parameters():
            param.requires_grad = False
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc_encoder = nn.Linear(2048, d_model)
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_len)
        decoder_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, batch_first=True)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.fc_out = nn.Linear(d_model, vocab_size)
        self.d_model = d_model

    def generate_square_subsequent_mask(self, sz):
        mask = torch.triu(torch.ones(sz, sz), diagonal=1)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        return mask

    def forward(self, images, captions):
        features = self.cnn(images)
        features = self.adaptive_pool(features)
        features = features.view(features.size(0), -1)
        memory = self.fc_encoder(features).unsqueeze(1)
        tgt_embedded = self.embedding(captions) * torch.sqrt(torch.tensor(self.d_model, dtype=torch.float32))
        tgt_embedded = self.pos_encoder(tgt_embedded)
        tgt_len = captions.size(1)
        device = captions.device
        tgt_mask = self.generate_square_subsequent_mask(tgt_len).to(device)
        output = self.transformer_decoder(tgt_embedded, memory, tgt_mask=tgt_mask)
        output = self.fc_out(output)
        return output

    def generate_caption(self, image_path, word2idx, idx2word, max_len=20):
        self.eval()
        transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        image = Image.open(image_path).convert('RGB')
        image = transform(image).unsqueeze(0)
        with torch.no_grad():
            device = next(self.parameters()).device
            image = image.to(device)
            features = self.cnn(image)
            features = self.adaptive_pool(features)
            features = features.view(features.size(0), -1)
            memory = self.fc_encoder(features).unsqueeze(1)
            caption = [word2idx.get('<START>', 1)]
            for _ in range(max_len):
                tgt = torch.LongTensor([caption]).to(device)
                tgt_embedded = self.embedding(tgt) * torch.sqrt(torch.tensor(self.d_model, dtype=torch.float32, device=device))
                tgt_embedded = self.pos_encoder(tgt_embedded)
                tgt_len = tgt.size(1)
                tgt_mask = self.generate_square_subsequent_mask(tgt_len).to(device)
                output = self.transformer_decoder(tgt_embedded, memory, tgt_mask=tgt_mask)
                output = self.fc_out(output)
                next_word_idx = output[0, -1, :].argmax().item()
                if next_word_idx == word2idx.get('<END>', 2):
                    break
                caption.append(next_word_idx)
            caption_words = [idx2word.get(idx, '<UNK>') for idx in caption[1:]]
            return ' '.join(caption_words)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]
