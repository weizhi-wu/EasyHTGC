# --- START OF FILE src/runner.py ---

import os
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm

# Import local modules from the src/ directory
from models import ResNet20
from deepzero import DeepZero
from utils import parse_hyperparameters

def validate(model, testloader, device):
    """Helper function to calculate validation accuracy."""
    model.eval()  # Set the model to evaluation mode
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    return accuracy

def execute_experiment(args):
    """
    Contains the core scientific logic for setting up and running the experiment.
    """
    print(f"--- [runner.py] Core logic started for: {args.exp_id} ---")

    # 1. --- Setup and Hyperparameter Parsing ---
    params = parse_hyperparameters(args.exp_id)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"[runner.py] Using device: {device}")

    # 2. --- Data Loading ---
    # Construct the correct path to the dataset inside the scratch directory
    # This assumes the .tar.gz, when uncompressed, creates a 'cifar10_dataset' folder.
    # This path is critical and must match the structure of your dataset archive.
    data_root = os.path.join(args.data_path, 'cifar10_dataset')
    print(f"[runner.py] Loading dataset from staged path: {data_root}")

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    trainset = torchvision.datasets.CIFAR10(
        root=data_root, train=True, download=False, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=64, shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(
        root=data_root, train=False, download=False, transform=transform_test)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=100, shuffle=False, num_workers=2)

    # 3. --- Model, Optimizer, and Criterion Initialization ---
    model = ResNet20().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = DeepZero(model, model.parameters(), **params)

    # 4. --- Training and Validation Loop ---
    num_epochs = 5
    results_history = []
    print(f"[runner.py] Starting training for {num_epochs} epochs...")

    for epoch in range(num_epochs):
        model.train()  # Set the model to training mode
        optimizer.epoch_counter = epoch
        optimizer.iter_counter = 0
        
        running_loss = 0.0
        pbar = tqdm(enumerate(trainloader), total=len(trainloader), desc=f"Epoch {epoch + 1}/{num_epochs}")

        for i, data in pbar:
            inputs, labels = data[0].to(device), data[1].to(device)

            # The closure function is essential for the DeepZero optimizer
            def closure():
                model.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                return loss

            loss_val = optimizer.step(closure)
            running_loss += loss_val
            pbar.set_postfix({'loss': f'{running_loss / (i + 1):.4f}'})

        # --- Validation after each epoch ---
        epoch_train_loss = running_loss / len(trainloader)
        test_accuracy = validate(model, testloader, device)
        
        print(f"[runner.py] Epoch {epoch + 1} Summary: Train Loss={epoch_train_loss:.4f}, Test Acc={test_accuracy:.2f}%")

        # --- Record results for this epoch ---
        epoch_data = {
            'experiment_id': args.exp_id,
            'epoch': epoch + 1,
            'train_loss': round(epoch_train_loss, 4),
            'test_accuracy': round(test_accuracy, 2)
        }
        results_history.append(epoch_data)

    print("[runner.py] Training loop finished.")
    return results_history

# --- END OF FILE src/runner.py ---
