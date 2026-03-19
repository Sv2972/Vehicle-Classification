import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, WeightedRandomSampler
import numpy as np
from tqdm import tqdm
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt

# 1. Setup Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 2. Helper Functions
def get_sampler(dataset):
    targets = dataset.targets
    class_sample_count = np.array([len(np.where(targets == t)[0]) for t in np.unique(targets)])
    weight = 1. / class_sample_count
    samples_weight = np.array([weight[t] for t in targets])
    sampler = WeightedRandomSampler(torch.from_numpy(samples_weight), len(samples_weight))
    return sampler

def train_model(model, train_loader, criterion, optimizer, scheduler, num_epochs=10):
    history = {'loss': [], 'acc': []} 
    for epoch in range(num_epochs):
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        
        loop = tqdm(enumerate(train_loader), total=len(train_loader), leave=True)
        loop.set_description(f"Epoch [{epoch+1}/{num_epochs}]")

        for batch_idx, (inputs, labels) in loop:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            loop.set_postfix(loss=loss.item(), acc=f"{100.*correct/total:.2f}%")

        scheduler.step()
        history['loss'].append(running_loss / len(train_loader))
        history['acc'].append(100. * correct / total)
        print(f"--> Epoch {epoch+1} Complete: Avg Loss: {history['loss'][-1]:.4f} | Acc: {history['acc'][-1]:.2f}%")
        
        if (epoch + 1) % 5 == 0:
            torch.save(model.state_dict(), f'models/checkpoint_epoch_{epoch+1}.pth')
    return history

def plot_learning_curves(history):
    epochs = range(1, len(history['loss']) + 1)
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history['loss'], 'r-o', label='Training Loss')
    plt.title('Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history['acc'], 'b-o', label='Training Accuracy')
    plt.title('Training Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('models/learning_curves.png')

def evaluate_model(model, val_loader, class_names):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for inputs, labels in tqdm(val_loader, desc="Evaluating"):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Identify unique labels in the validation data
    present_classes = np.unique(all_labels)
    
    # SAFE FILTER: only map names if the index is within the training class list
    filtered_names = []
    for i in present_classes:
        if i < len(class_names):
            filtered_names.append(class_names[i])
        else:
            filtered_names.append(f"Unknown_Class_{i}")
    
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, 
                                labels=present_classes, 
                                target_names=filtered_names))

def export_to_onnx(model, save_path="models/vehicle_classifier.onnx"):
    model.eval()
    dummy_input = torch.randn(1, 3, 224, 224).to(device)
    torch.onnx.export(model, dummy_input, save_path, 
                      input_names=['input'], output_names=['output'],
                      dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}},
                      opset_version=11)

# 3. Main Execution Block
if __name__ == "__main__":
    os.makedirs('models', exist_ok=True)
    print(f"Training on: {torch.cuda.get_device_name(0)}")

    # Transforms
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.2, 0.2, 0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Datasets and Loaders
    DATA_DIR = 'data/vehicle_dataset_cleaned/train'
    train_dataset = datasets.ImageFolder(os.path.join(DATA_DIR, 'train'), transform=train_transform)
    val_dataset = datasets.ImageFolder('data/vehicle_dataset/val', transform=val_transform)

    train_loader = DataLoader(train_dataset, batch_size=32, sampler=get_sampler(train_dataset), num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)

    # Model Setup
    model = models.efficientnet_b0(weights='IMAGENET1K_V1')
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, len(train_dataset.classes))
    model = model.to(device)

    # Training Setup
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1) 
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)

    # Execute Pipeline
    history = train_model(model, train_loader, criterion, optimizer, scheduler, num_epochs=13)
    torch.save(model.state_dict(), 'models/vehicle_efficientnet_final.pth')
    plot_learning_curves(history)
    evaluate_model(model, val_loader, train_dataset.classes)
    export_to_onnx(model)
    
    with open('classes.txt', 'w') as f:
        f.write("\n".join(train_dataset.classes))

    print("\n[SUCCESS] All deliverables generated.")