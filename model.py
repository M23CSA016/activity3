import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
from torchvision import models
import matplotlib.pyplot as plt
from tqdm import tqdm

class DataLoader:
    def __init__(self, root, transform, batch_size, train=True):
        self.dataset = torchvision.datasets.FashionMNIST(root=root, train=train, download=True, transform=transform)
        self.loader = torch.utils.data.DataLoader(self.dataset, batch_size=batch_size, shuffle=train)

class NeuralNetwork:
    def __init__(self, pretrained_model, num_classes):
        self.model = pretrained_model
        self.num_classes = num_classes
        self.device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')

    def freeze_model_params(self):
        for param in self.model.parameters():
            param.requires_grad = False

    def modify_output_layer(self):
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, self.num_classes)

    def train_model(self, train_loader, criterion, optimizer, num_epochs=1):
        self.model.to(self.device)
        self.model.train()
        train_losses = []
        train_accuracy = []

        for epoch in range(num_epochs):
            running_loss = 0.0
            correct = 0
            total = 0
            for images, labels in tqdm(train_loader, desc=f"Epoch [{epoch + 1}/{num_epochs}]"):
                images, labels = images.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                outputs = self.model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

            epoch_loss = running_loss / len(train_loader)
            epoch_accuracy = 100 * correct / total
            train_losses.append(epoch_loss)
            train_accuracy.append(epoch_accuracy)

            print(f"Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.2f}%")

        return train_losses, train_accuracy

    def evaluate_top5_accuracy(self, test_loader):
        self.model.eval()
        correct_top5 = 0
        total = 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                _, predicted = outputs.topk(5, dim=1)  # Get top-5 predictions
                correct_top5 += sum([1 for i in range(len(labels)) if labels[i] in predicted[i]])
                total += labels.size(0)
        top5_accuracy = (correct_top5 / total) * 100
        return top5_accuracy

def plot_curves(train_losses, train_accuracy):
    epochs = range(1, len(train_losses) + 1)

    plt.figure(figsize=(10, 5))
    plt.plot(epochs, train_losses, label='Training Loss', marker='o', linestyle='-')
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(10, 5))
    plt.plot(epochs, train_accuracy, label='Training Accuracy', marker='o', linestyle='-')
    plt.title('Training Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
    ])

    train_loader = DataLoader(root='./content', transform=transform, batch_size=64, train=True).loader
    test_loader = DataLoader(root='./content', transform=transform, batch_size=64, train=False).loader

    resnet = models.densenet121(pretrained=True)
    # resnet = models.resnet101(pretrained=True)
    neural_network = NeuralNetwork(pretrained_model=resnet, num_classes=10)
    neural_network.freeze_model_params()
    neural_network.modify_output_layer()

    criterion = nn.CrossEntropyLoss()
    # optimizers = [optim.Adam, optim.Adagrad, optim.RMSprop]

    optimizer = optim.Adam(neural_network.model.parameters())
    train_losses, train_accuracy = neural_network.train_model(train_loader, criterion, optimizer)
    plot_curves(train_losses, train_accuracy)
    
    # for optimizer_class in optimizers:
    #     print(f"Training with optimizer: {optimizer_class.__name__}")
    #     optimizer = optimizer_class(neural_network.model.parameters())
    #     train_losses, train_accuracy = neural_network.train_model(train_loader, criterion, optimizer)
    #     plot_curves(train_losses, train_accuracy)

    top5_accuracy = neural_network.evaluate_top5_accuracy(test_loader)
    print(f"Final Top-5 Test Accuracy: {top5_accuracy:.2f}%")