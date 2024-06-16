import os
import mne
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler, LabelEncoder
import matplotlib.pyplot as plt
from datetime import datetime

# 定義CNN模型（這部分是從process_eeg_data_CNN.py中複製過來的）
class CNN(nn.Module):
    def __init__(self, num_classes):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=(1, 3), padding=(0, 1))
        self.conv2 = nn.Conv2d(16, 32, kernel_size=(1, 3), padding=(0, 1))
        self.pool = nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2))
        
        # 計算展平後的輸入大小
        n_features = self._get_conv_output((1, 74, 251))  # 這裡需要使用訓練時的輸入大小
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
        x = x.view(x.size(0), -1)  # 展平
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# 設置設備（CPU或CUDA）
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# 加載模型
model_path = 'model_20240616_143756.pth'  # 使用訓練時保存的模型文件名
model = CNN(num_classes=9)  # 替換為適當的num_classes
model.load_state_dict(torch.load(model_path))
model.to(device)
model.eval()

# 資料夾路徑
data_folder = r'C:\Users\user\Desktop\Final project\ds002718\ds002718'

# 初始化數據和標籤列表
inference_data = []
inference_labels = []

# 讀取多個受試者的數據
subjects = ['sub-018', 'sub-019']

# 需要分類的事件標記
event_types_to_include = ['famous_new', 'famous_second_early', 'famous_second_late', 
                          'scrambled_new', 'scrambled_second_early', 'scrambled_second_late', 
                          'unfamiliar_new', 'unfamiliar_second_early', 'unfamiliar_second_late']

for subject in subjects:
    file_path = os.path.join(data_folder, subject, 'eeg', f'{subject}_task-FaceRecognition_eeg.set')
    raw = mne.io.read_raw_eeglab(file_path)
    raw.filter(1., 40., fir_design='firwin')
    raw.set_eeg_reference('average', projection=True)
    raw.apply_proj()  # 應用投影
    events, event_id = mne.events_from_annotations(raw)
    print(f'Subject {subject}: Event IDs - {event_id}')  # 確認事件類型
    # 過濾需要的事件類型
    selected_events = np.array([event for event in events if event[2] in [2, 3, 4, 7, 8, 9, 10, 11, 12]], dtype=int)
    selected_event_ids = {key: value for key, value in event_id.items() if value in [2, 3, 4, 7, 8, 9, 10, 11, 12]}
    epochs = mne.Epochs(raw, selected_events, selected_event_ids, tmin=-0.2, tmax=0.8, baseline=(None, 0))
    data = epochs.get_data()
    labels = epochs.events[:, -1]
    inference_data.append(data)
    inference_labels.append(labels)

# 合併所有受試者的數據和標籤
inference_data = np.concatenate(inference_data, axis=0)
inference_labels = np.concatenate(inference_labels, axis=0)

# 將標籤轉換為從0開始的整數
label_encoder = LabelEncoder()
label_encoder.fit(inference_labels)  # 避免未見過的標籤錯誤
y_inference = label_encoder.transform(inference_labels)

# 將數據轉換為CNN格式 (樣本數, 通道數, 時間點數)
X_inference = inference_data

# 標準化數據
scaler = StandardScaler()
for i in range(X_inference.shape[0]):
    X_inference[i] = scaler.fit_transform(X_inference[i])

# 將數據轉換為tensor
X_inference_tensor = torch.tensor(X_inference, dtype=torch.float32).unsqueeze(1).to(device)
y_inference_tensor = torch.tensor(y_inference, dtype=torch.long).to(device)

# 建立數據集和數據加載器
inference_dataset = TensorDataset(X_inference_tensor, y_inference_tensor)
inference_loader = DataLoader(inference_dataset, batch_size=32, shuffle=False)

# 推論
all_preds = []
all_targets = []
with torch.no_grad():
    for data, target in inference_loader:
        output = model(data)
        _, predicted = torch.max(output.data, 1)
        all_preds.extend(predicted.cpu().numpy())
        all_targets.extend(target.cpu().numpy())

# 計算準確率
accuracy = 100 * np.sum(np.array(all_preds) == np.array(all_targets)) / len(all_targets)
print(f'Inference Accuracy: {accuracy:.2f}%')

# 每個類別的準確率
class_accuracies = {}
for label in np.unique(all_targets):
    label_indices = np.where(np.array(all_targets) == label)
    label_correct = np.sum(np.array(all_preds)[label_indices] == np.array(all_targets)[label_indices])
    label_total = len(label_indices[0])
    label_accuracy = 100 * label_correct / label_total if label_total > 0 else 0
    class_accuracies[label_encoder.inverse_transform([label])[0]] = label_accuracy

print("Class-wise Accuracy:")
for class_name, class_accuracy in class_accuracies.items():
    print(f'{class_name}: {class_accuracy:.2f}%')

# 可視化每個類別的準確率
plt.figure(figsize=(12, 8))
class_names = list(class_accuracies.keys())
accuracies = list(class_accuracies.values())
plt.bar(class_names, accuracies)
plt.xlabel('Class')
plt.ylabel('Accuracy (%)')
plt.title('Class-wise Accuracy')
plt.xticks(rotation=45)
plt.tight_layout()

# 保存圖表
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
inference_plot_filename = f'inference_class_accuracies_{timestamp}.png'
plt.savefig(inference_plot_filename)
plt.show()
