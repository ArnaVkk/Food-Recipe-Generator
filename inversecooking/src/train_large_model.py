# =============================================================================
# TRAIN LARGE COMBINED MODEL - OPTIMAL FOR BIG DATA
# =============================================================================

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
import os
import json
from PIL import Image
from tqdm import tqdm
from pathlib import Path
import time

print("=" * 60)
print("TRAINING ON LARGE DATASET (181 classes, 43K images)")
print("=" * 60)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\n🖥️  Device: {device}")
if torch.cuda.is_available():
    print(f"🎮 GPU: {torch.cuda.get_device_name(0)}")
    print(f"💾 GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

# =============================================================================
# CONFIGURATION - OPTIMIZED FOR LARGE DATASET
# =============================================================================
CONFIG = {
    'data_dir': '../data/combined_large',
    'output_dir': '../data/large_model',
    'batch_size': 32,  # Larger batch for more data
    'num_epochs': 25,  # Fewer epochs needed with more data
    'warmup_epochs': 3,
    'learning_rate': 0.001,  # Higher LR since more data
    'weight_decay': 0.0001,  # Less regularization needed
    'num_workers': 0,
    'image_size': 224,
    'patience': 5,
}

os.makedirs(CONFIG['output_dir'], exist_ok=True)

# =============================================================================
# DATASET
# =============================================================================
class LargeFoodDataset(Dataset):
    def __init__(self, root_dir, split='train', transform=None):
        self.root_dir = Path(root_dir) / split
        self.transform = transform
        self.samples = []
        self.class_to_idx = {}
        self.idx_to_class = {}
        self.cuisine_map = {}
        
        if self.root_dir.exists():
            classes = sorted([d.name for d in self.root_dir.iterdir() if d.is_dir()])
            
            for idx, class_name in enumerate(classes):
                self.class_to_idx[class_name] = idx
                self.idx_to_class[idx] = class_name
                self.cuisine_map[class_name] = 'indian' if class_name.startswith('indian_') else 'western'
                
                class_dir = self.root_dir / class_name
                for img_path in class_dir.glob('*'):
                    if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.webp']:
                        self.samples.append((str(img_path), idx))
        
        indian = sum(1 for c in self.cuisine_map.values() if c == 'indian')
        western = sum(1 for c in self.cuisine_map.values() if c == 'western')
        print(f"  {split}: {len(self.samples):,} images | {western} Western + {indian} Indian = {len(classes)} classes")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        try:
            image = Image.open(img_path).convert('RGB')
        except:
            image = Image.new('RGB', (224, 224), (128, 128, 128))
        
        if self.transform:
            image = self.transform(image)
        return image, label


# =============================================================================
# MODEL - STANDARD DROPOUT (LESS REGULARIZATION NEEDED)
# =============================================================================
class LargeFoodClassifier(nn.Module):
    def __init__(self, num_classes, dropout=0.25):
        super().__init__()
        
        self.backbone = models.efficientnet_b0(weights='IMAGENET1K_V1')
        
        # Freeze initially
        for param in self.backbone.parameters():
            param.requires_grad = False
        
        # Classifier
        num_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(dropout),
            nn.Linear(512, num_classes)
        )
    
    def unfreeze_partial(self):
        """Unfreeze last 3 blocks"""
        for name, param in self.backbone.named_parameters():
            if any(f'features.{i}' in name for i in [6, 7, 8]) or 'classifier' in name:
                param.requires_grad = True
    
    def unfreeze_all(self):
        for param in self.backbone.parameters():
            param.requires_grad = True
    
    def forward(self, x):
        return self.backbone(x)


# =============================================================================
# TRAINING FUNCTIONS
# =============================================================================
def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(loader, leave=False, ncols=100, desc="Training")
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        pbar.set_postfix({'loss': f'{loss.item():.3f}', 'acc': f'{100*correct/total:.1f}%'})
    
    return running_loss / len(loader), 100 * correct / total


def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in tqdm(loader, leave=False, ncols=100, desc="Validating"):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    return running_loss / len(loader), 100 * correct / total


def main():
    print("\n[1] Setting up data augmentation (moderate - more data = less augmentation needed)...")
    
    train_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    print("\n[2] Loading datasets...")
    train_dataset = LargeFoodDataset(CONFIG['data_dir'], 'train', train_transform)
    val_dataset = LargeFoodDataset(CONFIG['data_dir'], 'val', val_transform)
    
    train_loader = DataLoader(train_dataset, batch_size=CONFIG['batch_size'], 
                              shuffle=True, num_workers=CONFIG['num_workers'], pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=CONFIG['batch_size'],
                            shuffle=False, num_workers=CONFIG['num_workers'], pin_memory=True)
    
    num_classes = len(train_dataset.class_to_idx)
    print(f"\n  📊 {num_classes} classes, {len(train_dataset):,} training images")
    print(f"  📊 ~{len(train_dataset)//num_classes} images per class (was ~35 before)")
    
    print(f"\n[3] Creating model (dropout=0.25, less regularization needed)...")
    model = LargeFoodClassifier(num_classes=num_classes, dropout=0.25)
    model = model.to(device)
    
    # Light label smoothing
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    
    # Save mappings
    mappings = {
        'class_to_idx': train_dataset.class_to_idx,
        'idx_to_class': train_dataset.idx_to_class,
        'cuisine_map': train_dataset.cuisine_map
    }
    with open(os.path.join(CONFIG['output_dir'], 'class_mapping.json'), 'w') as f:
        json.dump(mappings, f, indent=2)
    
    print("\n" + "=" * 60)
    print("PHASE 1: Warmup (classifier only)")
    print("=" * 60)
    
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), 
                            lr=CONFIG['learning_rate'], weight_decay=CONFIG['weight_decay'])
    
    best_acc = 0.0
    best_epoch = 0
    best_gap = 0
    patience_counter = 0
    history = {'train_acc': [], 'val_acc': [], 'gap': []}
    start_time = time.time()
    
    for epoch in range(1, CONFIG['warmup_epochs'] + 1):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        gap = train_acc - val_acc
        
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        history['gap'].append(gap)
        
        status = ""
        if val_acc > best_acc:
            best_acc = val_acc
            best_epoch = epoch
            best_gap = gap
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'val_acc': val_acc,
                'train_acc': train_acc,
                'gap': gap,
                'class_to_idx': train_dataset.class_to_idx,
                'cuisine_map': train_dataset.cuisine_map,
            }, os.path.join(CONFIG['output_dir'], 'best_model.pth'))
            status = " ✅"
        
        print(f"[Warmup {epoch}] Train: {train_acc:.1f}% | Val: {val_acc:.1f}% | Gap: {gap:+.1f}%{status}")
    
    print("\n" + "=" * 60)
    print("PHASE 2: Fine-tuning (partial unfreezing)")
    print("=" * 60)
    
    model.unfreeze_partial()
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Unfrozen: {trainable:,} trainable parameters\n")
    
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), 
                            lr=CONFIG['learning_rate'] * 0.1, weight_decay=CONFIG['weight_decay'])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CONFIG['num_epochs'] - CONFIG['warmup_epochs'])
    
    for epoch in range(CONFIG['warmup_epochs'] + 1, CONFIG['num_epochs'] + 1):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        gap = train_acc - val_acc
        
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        history['gap'].append(gap)
        
        scheduler.step()
        lr = optimizer.param_groups[0]['lr']
        
        status = ""
        if val_acc > best_acc:
            best_acc = val_acc
            best_epoch = epoch
            best_gap = gap
            patience_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'val_acc': val_acc,
                'train_acc': train_acc,
                'gap': gap,
                'class_to_idx': train_dataset.class_to_idx,
                'cuisine_map': train_dataset.cuisine_map,
            }, os.path.join(CONFIG['output_dir'], 'best_model.pth'))
            status = " ✅"
        else:
            patience_counter += 1
        
        print(f"[{epoch:2d}/{CONFIG['num_epochs']}] Train: {train_acc:.1f}% | Val: {val_acc:.1f}% | Gap: {gap:+.1f}% | LR: {lr:.6f}{status}")
        
        if patience_counter >= CONFIG['patience']:
            print(f"\n⏹️  Early stopping (no improvement for {CONFIG['patience']} epochs)")
            break
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    total_time = time.time() - start_time
    
    # Save history
    with open(os.path.join(CONFIG['output_dir'], 'training_history.json'), 'w') as f:
        json.dump(history, f, indent=2)
    
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE!")
    print("=" * 60)
    print(f"\n⏱️  Time: {total_time/60:.1f} minutes")
    print(f"🏆 Best validation accuracy: {best_acc:.1f}% (epoch {best_epoch})")
    print(f"📊 Gap at best: {best_gap:+.1f}%")
    print(f"📁 Model: {CONFIG['output_dir']}/best_model.pth")
    
    print("\n📈 COMPARISON WITH PREVIOUS MODELS:")
    print(f"   Small data (3K images, 90 classes):")
    print(f"     - V1: 65.5% acc, +32% gap  ❌ Overfitting")
    print(f"     - V3: 63.6% acc, +15% gap  ⚠️ Slight overfitting")
    print(f"   Large data (43K images, 181 classes):")
    print(f"     - New: {best_acc:.1f}% acc, {best_gap:+.1f}% gap", end="")
    
    if 0 <= best_gap <= 10:
        print("  ✅ PERFECT!")
    elif best_gap < 0:
        print("  ⚠️ Underfitting")
    elif best_gap <= 15:
        print("  ✅ Good!")
    else:
        print("  ⚠️ Still overfitting")
    
    return model, history


if __name__ == '__main__':
    model, history = main()
