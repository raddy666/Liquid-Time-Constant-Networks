import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm
import time


def get_cifar10_loaders(batch_size=128, num_workers=2):
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.4914, 0.4822, 0.4465],
            std=[0.2023, 0.1994, 0.2010]
        ),
    ])
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.4914, 0.4822, 0.4465],
            std=[0.2023, 0.1994, 0.2010]
        ),
    ])
    
    trainset = torchvision.datasets.CIFAR10(
        root='./data',
        train=True,
        download=True, 
        transform=transform_train
    )
    trainloader = DataLoader(
        trainset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    testset = torchvision.datasets.CIFAR10(
        root='./data',
        train=False,
        download=True,
        transform=transform_test
    )
    testloader = DataLoader(
        testset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    print(f"Training samples: {len(trainset)}")
    print(f"Test samples: {len(testset)}")
    
    return trainloader, testloader


def train_one_epoch(model, trainloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(trainloader, desc='Training')
    for inputs, targets in pbar:
        inputs, targets = inputs.to(device), targets.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        
        pbar.set_postfix({
            'loss': running_loss / (pbar.n + 1),
            'acc': 100. * correct / total
        })
    
    epoch_loss = running_loss / len(trainloader)
    epoch_acc = 100. * correct / total
    
    return epoch_loss, epoch_acc


def test(model, testloader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, targets in testloader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    
    test_loss = running_loss / len(testloader)
    test_acc = 100. * correct / total
    
    return test_loss, test_acc

def train_model(model, model_name, device, epochs=20, batch_size=128, lr=0.001):
    
    print(f"\n{'='*60}")
    print(f"Training: {model_name}")
    print(f"{'='*60}")
    
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    trainloader, testloader = get_cifar10_loaders(batch_size=batch_size)
    
    history = {
        'train_loss': [],
        'train_acc': [],
        'test_loss': [],
        'test_acc': [],
        'epoch_time': []
    }
    
    best_acc = 0.0
    best_epoch = 0
    
    for epoch in range(epochs):
        start_time = time.time()
        
        train_loss, train_acc = train_one_epoch(
            model, trainloader, criterion, optimizer, device
        )
        
        test_loss, test_acc = test(model, testloader, criterion, device)
        
        if test_acc > best_acc:
            best_acc = test_acc
            best_epoch = epoch + 1
            torch.save(model.state_dict(), f"{model_name}_best.pth")
            print(f"  ✓ New best model saved! (Test Acc: {test_acc:.2f}%)")
        
        scheduler.step()
        
        epoch_time = time.time() - start_time
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['test_loss'].append(test_loss)
        history['test_acc'].append(test_acc)
        history['epoch_time'].append(epoch_time)
        
        print(f"Epoch {epoch+1}/{epochs}:")
        print(f"  Train - Loss: {train_loss:.4f}, Acc: {train_acc:.2f}%")
        print(f"  Test  - Loss: {test_loss:.4f}, Acc: {test_acc:.2f}%")
        print(f"  Time: {epoch_time:.1f}s\n")
    
    history['best_test_acc'] = best_acc
    history['best_epoch'] = best_epoch
    
    torch.save(model.state_dict(), f"{model_name}_final.pth")
    print(f"\n{'='*60}")
    print(f"Training Complete: {model_name}")
    print(f"Best model: Epoch {best_epoch} - {best_acc:.2f}%")
    print(f"Final model: Epoch {epochs} - {history['test_acc'][-1]:.2f}%")
    print(f"Best model saved to {model_name}_best.pth")
    print(f"Final model saved to {model_name}_final.pth")
    print(f"{'='*60}\n")
    
    return history