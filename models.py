import torch
import torch.nn as nn
from ncps.torch import LTC
from ncps.wirings import AutoNCP

# ==================== ATTENTION MODULE: CBAM ====================

class CBAM(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        
        # Channel Attention
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False)
        )
        
        # Spatial Attention
        self.conv = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        # Channel attention
        b, c, h, w = x.size()
        avg_out = self.fc(self.avg_pool(x).view(b, c))
        max_out = self.fc(self.max_pool(x).view(b, c))
        channel_att = self.sigmoid(avg_out + max_out).view(b, c, 1, 1)
        x = x * channel_att
        
        # Spatial attention
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        spatial_att = self.sigmoid(self.conv(torch.cat([avg_out, max_out], dim=1)))
        x = x * spatial_att
        
        return x


# ==================== MODEL 1: BASELINE CNN ====================

class BaselineCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.25),
            
            # Block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.25),
        )
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


# ==================== MODEL 2: PURE LNN ====================

class PureLNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        
        # Extract patches
        self.patch_embed = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=4, stride=4),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        
        # LNN processes patches as sequence
        wiring = AutoNCP(128, num_classes) 
        self.liquid = LTC(64, wiring, batch_first=True) 
    
    def forward(self, x):
        patches = self.patch_embed(x) 
        
        batch_size = patches.size(0)
        patches = patches.flatten(2).transpose(1, 2)
        
        # Process through LNN
        output, _ = self.liquid(patches)
        
        return output[:, -1, :]


# ==================== MODEL 3: LNN + CBAM ====================

class LNN_CBAM(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        
        self.patch_embed = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=4, stride=4),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        
        # CBAM Plugin
        self.attention = CBAM(channels=64, reduction=8)
        
        # LNN
        wiring = AutoNCP(128, num_classes)
        self.liquid = LTC(64, wiring, batch_first=True)
    
    def forward(self, x):
        patches = self.patch_embed(x)
        
        # attention before LNN
        patches = self.attention(patches)
        
        patches = patches.flatten(2).transpose(1, 2)
        output, _ = self.liquid(patches)
        return output[:, -1, :]


# ==================== MODEL 4: HYBRID CNN-LNN ====================

class HybridCNNLNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        
        # CNN backbone
        self.cnn_features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        
        # LNN classifier
        wiring = AutoNCP(128, num_classes)
        self.liquid = LTC(64, wiring, batch_first=True)
    
    def forward(self, x):
        features = self.cnn_features(x)
        
        # Treat spatial positions as sequence
        features = features.flatten(2).transpose(1, 2)
        
        output, _ = self.liquid(features)
        return output[:, -1, :]


# ==================== MODEL 5: HYBRID + CBAM ====================

class HybridCNNLNN_CBAM(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        
        self.conv2 = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        
        # Attention plugin
        self.attention = CBAM(channels=64, reduction=8)
        
        # LNN classifier
        wiring = AutoNCP(128, num_classes)
        self.liquid = LTC(64, wiring, batch_first=True)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.attention(x)
        
        features = x.flatten(2).transpose(1, 2)
        output, _ = self.liquid(features)
        return output[:, -1, :]


# ==================== UTILITY FUNCTIONS ====================

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_model(model_name, num_classes=10):
    models = {
        'baseline': BaselineCNN,
        'pure_lnn': PureLNN,
        'lnn_cbam': LNN_CBAM,
        'hybrid': HybridCNNLNN,
        'hybrid_cbam': HybridCNNLNN_CBAM,
    }
    
    if model_name not in models:
        raise ValueError(f"Unknown model: {model_name}. Choose from {list(models.keys())}")
    
    model = models[model_name](num_classes=num_classes)
    params = count_parameters(model)
    print(f"Created {model_name}: {params:,} parameters")
    
    return model


if __name__ == '__main__':
    # Test all models
    print("Testing all models...\n")
    
    batch_size = 4
    dummy_input = torch.randn(batch_size, 3, 32, 32)
    
    for name in ['baseline', 'pure_lnn', 'lnn_cbam', 'hybrid', 'hybrid_cbam']:
        print(f"Testing {name}:")
        model = get_model(name)
        output = model(dummy_input)
        print(f"  Output shape: {output.shape}")
        print(f"  Parameters: {count_parameters(model):,}\n")