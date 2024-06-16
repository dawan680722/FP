import os
import mne
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import matplotlib.pyplot as plt
from tqdm import tqdm
from datetime import datetime


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

data_folder = r'C:\Users\user\Desktop\Final project\ds002718\ds002718'

all_data = []
all_labels = []
event_orders = []

subjects = ['sub-002', 'sub-003', 'sub-004', 'sub-005', 'sub-006', 'sub-007', 
            'sub-008', 'sub-009', 'sub-010', 'sub-011', 'sub-012', 'sub-013', 
            'sub-014', 'sub-015', 'sub-016', 'sub-017', 'sub-018', 'sub-019']

event_types_to_include = [5, 6, 7, 13, 14, 15, 17, 18, 19]

for subject in subjects:
    file_path = os.path.join(data_folder, subject, 'eeg', f'{subject}_task-FaceRecognition_eeg.set')
    raw = mne.io.read_raw_eeglab(file_path)
    raw.filter(1., 40., fir_design='firwin')
    raw.set_eeg_reference('average', projection=True)
    raw.apply_proj()  
    events, event_id = mne.events_from_annotations(raw)
    # 過濾需要的事件類型
    selected_events = [event for event in events if event[2] in event_types_to_include]
    selected_event_ids = {key: value for key, value in event_id.items() if value in event_types_to_include}
    epochs = mne.Epochs(raw, np.array(selected_events), selected_event_ids, tmin=0.1, tmax=0.8, baseline=(None, 0))
    data = epochs.get_data()
    labels = epochs.events[:, -1]
    orders = epochs.events[:, 1]  # 獲取event_order
    all_data.append(data)
    all_labels.append(labels)
    event_orders.append(orders)


all_data = np.concatenate(all_data, axis=0)
all_labels = np.concatenate(all_labels, axis=0)
event_orders = np.concatenate(event_orders, axis=0)


unique_labels, counts = np.unique(all_labels, return_counts=True)
print("標籤分佈:")
for label, count in zip(unique_labels, counts):
    print(f"標籤 {label}: {count} 個樣本")


label_encoder = LabelEncoder()
y = label_encoder.fit_transform(all_labels)

X = all_data

scaler = StandardScaler()
for i in range(X.shape[0]):
    X[i] = scaler.fit_transform(X[i])


X_tensor = torch.tensor(X, dtype=torch.float32).unsqueeze(1).to(device)
y_tensor = torch.tensor(y, dtype=torch.long).to(device)
orders_tensor = torch.tensor(event_orders, dtype=torch.long).to(device)


dataset = TensorDataset(X_tensor, y_tensor, orders_tensor)
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])


train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)


class CNN(nn.Module):
    def __init__(self, num_classes):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=(1, 3), padding=(0, 1))
        self.conv2 = nn.Conv2d(16, 32, kernel_size=(1, 3), padding=(0, 1))
        self.pool = nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2))
        
      
        n_features = self._get_conv_output((1, X.shape[1], X.shape[2]))
        self.fc1 = nn.Linear(n_features, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def _get_conv_output(self, shape):
        bs = 1
        input = torch.rand(bs, *shape)
        output_feat = self._forward_features(input)
        n_size = output_feat.data.view(bs, -1).size(1)
        return n_size

    def _forward_features(self, x):
        x = torch.relu(self.conv1(x))
        x = self.pool(x)
        x = torch.relu(self.conv2(x))
        x = self.pool(x)
        return x

    def forward(self, x):
        x = self._forward_features(x)
        x = x.view(x.size(0), -1) 
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


num_classes = 9 
model = CNN(num_classes).to(device)


criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

train_losses = []
test_accuracies = []

num_epochs = 100
for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0
    for data, target, order in tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}'):
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    train_loss /= len(train_loader)
    train_losses.append(train_loss)

    model.eval()
    correct = 0
    total = 0
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for data, target, order in test_loader:
            output = model(data)
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(target.cpu().numpy())

    accuracy = 100 * correct / total
    test_accuracies.append(accuracy)
    print(f'Epoch {epoch+1}/{num_epochs}, Loss: {train_loss:.4f}, Accuracy: {accuracy:.2f}%')


timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
model_filename = f'model_{timestamp}.pth'
torch.save(model.state_dict(), model_filename)


plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(range(1, num_epochs+1), train_losses, marker='o')
plt.title('Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')

plt.subplot(1, 2, 2)
plt.plot(range(1, num_epochs+1), test_accuracies, marker='o')
plt.title('Test Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')


plot_filename = f'training_progress_{timestamp}.png'
plt.savefig(plot_filename)
plt.show()