import torch
import torch.nn as nn
from torchvision import models

class HybridCNNTransformer(nn.Module):
    def __init__(self, num_classes, pretrained=True, num_layers=4, nhead=8, hidden_dim=2048):
        super(HybridCNNTransformer, self).__init__()

        # --- CNN Backbone (ResNet50) ---
        resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT if pretrained else None)
        self.cnn_backbone = nn.Sequential(*list(resnet.children())[:-2])  # remove avgpool + fc → (B, 2048, 7, 7)

        self.feature_dim = hidden_dim  # typically 2048 for ResNet50
        self.patch_dim = 7 * 7  # 49 patches
        self.flatten = nn.Flatten(2)  # (B, 2048, 49)
        self.transpose = lambda x: x.transpose(1, 2)  # (B, 49, 2048)

        # --- Positional Encoding ---
        self.pos_embedding = nn.Parameter(torch.randn(1, self.patch_dim + 1, self.feature_dim))

        # --- CLS token ---
        self.cls_token = nn.Parameter(torch.randn(1, 1, self.feature_dim))

        # --- Transformer Encoder ---
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.feature_dim,
            nhead=nhead,
            dim_feedforward=4096,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # --- Classification Head ---
        self.fc = nn.Linear(self.feature_dim, num_classes)

    def forward(self, x):
        # 1. CNN feature extraction
        features = self.cnn_backbone(x)  # (B, 2048, 7, 7)

        # 2. Flatten → (B, 2048, 49)
        features = self.flatten(features)

        # 3. Transpose to sequence format → (B, 49, 2048)
        features = self.transpose(features)

        # 4. Add CLS token
        B = features.size(0)
        cls_tokens = self.cls_token.expand(B, -1, -1)  # (B, 1, 2048)
        tokens = torch.cat((cls_tokens, features), dim=1)  # (B, 50, 2048)

        # 5. Add positional encoding
        tokens = tokens + self.pos_embedding[:, :tokens.size(1), :]

        # 6. Transformer
        transformer_out = self.transformer(tokens)  # (B, 50, 2048)

        # 7. Classification using CLS token
        out = self.fc(transformer_out[:, 0, :])  # (B, num_classes)
        return out
