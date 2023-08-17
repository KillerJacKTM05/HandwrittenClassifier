import time
import torch
from torchvision import transforms, datasets
from torch.utils.data import random_split
import torch.nn as nn
import torch.nn.functional as F
import psutil
from torch.optim.lr_scheduler import StepLR

data_transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])

#Path
data_dir = '.\HandwrittenNum'
#User-defined parameters
n_epochs = int(input("Enter number of epochs: "))
batch_size = int(input("Enter batch size: "))

best_val_loss = float('inf')
start_time = time.time()
full_dataset = datasets.ImageFolder(root=data_dir, transform=data_transform)
train_size = int(0.8 * len(full_dataset))
test_size = len(full_dataset) - train_size
train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size)

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.fc1 = nn.Linear(64*32*32, 512)
        self.fc2 = nn.Linear(512, 10)  # Assuming 10 classes

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = x.view(-1, 64*32*32)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

model = SimpleCNN()

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()
scheduler = StepLR(optimizer, step_size=1, gamma=0.7)
best_val_loss = float('inf')

for epoch in range(n_epochs):
    epoch_start_time = time.time()
    print(f"Epoch {epoch+1}/{n_epochs}")
    print('-' * 10)

    # Training
    model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)
    
    scheduler.step()

    train_loss = running_loss / len(train_loader.dataset)
    
    # Validation
    model.eval()
    running_val_loss = 0.0
    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_val_loss += loss.item() * inputs.size(0)
    
    val_loss = running_val_loss / len(test_loader.dataset)
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), 'best_model.pth')
    
    print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

    # Monitoring System Resources
    print("CPU usage: {:.2f}%".format(psutil.cpu_percent()))
    if torch.cuda.is_available():
        print("GPU Memory Usage:", torch.cuda.memory_allocated())
        
    # Compute elapsed and remaining time
    epoch_end_time = time.time()
    elapsed_time = epoch_end_time - start_time
    avg_time_per_epoch = elapsed_time / (epoch + 1)
    remaining_epochs = n_epochs - (epoch + 1)
    remaining_time = avg_time_per_epoch * remaining_epochs
    
    # Convert seconds to minutes and seconds for a more readable format
    elapsed_min, elapsed_sec = divmod(elapsed_time, 60)
    remaining_min, remaining_sec = divmod(remaining_time, 60)

    print(f"Elapsed Time: {int(elapsed_min)}m {int(elapsed_sec)}s")
    print(f"Estimated Remaining Time: {int(remaining_min)}m {int(remaining_sec)}s")
    print()
        
correct = 0
total = 0

model.load_state_dict(torch.load('best_model.pth'))
model.eval()

with torch.no_grad():
    for inputs, labels in test_loader:
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy: {:.2f} %'.format(100 * correct / total))