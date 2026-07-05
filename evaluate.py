import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from models import get_model
import matplotlib as mpl

# Set global font sizes BEFORE any plotting
mpl.rcParams['font.size'] = 16           # Base
mpl.rcParams['axes.titlesize'] = 18      # Titles
mpl.rcParams['axes.labelsize'] = 17      # Axis labels
mpl.rcParams['xtick.labelsize'] = 14     # X ticks
mpl.rcParams['ytick.labelsize'] = 14     # Y ticks
mpl.rcParams['legend.fontsize'] = 14     # Legend
mpl.rcParams['figure.titlesize'] = 19    # Figure title
mpl.rcParams['font.weight'] = 'normal'
mpl.rcParams['axes.labelweight'] = 'bold'

classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
           'dog', 'frog', 'horse', 'ship', 'truck']

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                       std=[0.2023, 0.1994, 0.2010])
])

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = DataLoader(testset, batch_size=128, shuffle=False)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def extract_features(model_name, model_type):
    """Extract features from second-to-last layer"""
    model = get_model(model_type)
    model.load_state_dict(torch.load(f'{model_name}_best.pth', weights_only=False))
    model = model.to(device)
    model.eval()
    
    features = []
    labels = []
    
    with torch.no_grad():
        for inputs, batch_labels in testloader:
            inputs = inputs.to(device)
            
            if model_type == 'baseline':
                # Extract from before final classifier
                x = model.features(inputs)
                x = torch.flatten(x, 1)
                feats = x
            elif model_type == 'pure_lnn' or model_type == 'lnn_cbam':
                # Pure LTC or LTC+CBAM
                patches = model.patch_embed(inputs)
                
                if hasattr(model, 'attention'):
                    patches = model.attention(patches)
                
                patches = patches.flatten(2).transpose(1, 2)
                output, _ = model.liquid(patches)
                feats = output[:, -1, :]  # Last timestep
            else:
                # Hybrid models (hybrid or hybrid_cbam)
                x = model.conv1(inputs)
                x = model.conv2(x)
                
                if hasattr(model, 'attention'):
                    x = model.attention(x)
                
                features_cnn = x.flatten(2).transpose(1, 2)
                output, _ = model.liquid(features_cnn)
                feats = output[:, -1, :]  # Last timestep
            
            features.append(feats.cpu().numpy())
            labels.extend(batch_labels.numpy())
    
    features = np.vstack(features)
    labels = np.array(labels)
    
    # Sample 1000 points for faster t-SNE
    np.random.seed(42)
    idx = np.random.choice(len(labels), min(1000, len(labels)), replace=False)
    
    return features[idx], labels[idx]

# Extract features from key models
models_to_viz = [
    ('1_Baseline_CNN', 'baseline', 'Baseline CNN'),
    ('2_Pure_LNN', 'pure_lnn', 'Pure LTC'),
    ('5_Hybrid_CNN_LNN_CBAM', 'hybrid_cbam', 'Hybrid+CBAM')
]

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

colors = plt.cm.tab10(np.linspace(0, 1, 10))

for idx, (model_name, model_type, title) in enumerate(models_to_viz):
    print(f"Extracting features for {title}...")
    features, labels = extract_features(model_name, model_type)
    
    print(f"Running t-SNE for {title}...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=1000)
    features_2d = tsne.fit_transform(features)
    
    ax = axes[idx]
    for class_idx in range(10):
        mask = labels == class_idx
        ax.scatter(features_2d[mask, 0], features_2d[mask, 1],
                  c=[colors[class_idx]], label=classes[class_idx],
                  alpha=0.6, s=20, edgecolors='none')
    
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.set_xlabel('t-SNE Dimension 1', fontsize=11)
    ax.set_ylabel('t-SNE Dimension 2', fontsize=11)
    ax.legend(loc='best', fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('tsne_features.pdf', dpi=300, bbox_inches='tight')
plt.savefig('tsne_features.png', dpi=300, bbox_inches='tight')
print("Figure 8 saved!")