torch 從EHR 找出 ICD-11 編碼
在電子健康記錄（EHR）系統中，ICD（國際疾病分類）編碼用於標識診斷和程序。在 EHR 數據中，ICD-11 編碼的提取是醫療數據處理中的常見任務。為了從 EHR 數據中找出 ICD-11 編碼，可以釆用以下步驟：
步驟：
    1. 獲取 EHR 數據：EHR 數據通常以文本形式記錄病人的診斷、治療以及其他醫療信息。這些信息可以存儲在數據庫、CSV 文件或醫療管理系統中。
    2. ICD-11 代碼的庫：需要一個 ICD-11 的代碼映射，ICD-11 是最新的疾病分類系統，可以從ICD-11 官方網站下載最新的版本。
    3. 自然語言處理（NLP）：因為 EHR 數據可能包含非結搆化的文本，通常需要通過自然語言處理（NLP）技朮來提取與疾病相關的描述，并與 ICD-11 編碼匹配。
    4. 深度學習模型（可選）：你可以使用 PyTorch 來實現一個深度學習模型，通過模型對 EHR 文本進行訓練，以便自動識別 ICD-11 相關的描述。
    5. 匹配 ICD-11 代碼：根據文本中的症狀或診斷，查找并匹配相應的 ICD-11 代碼。
PyTorch 和 NLP 方法
我們可以使用 PyTorch 的 torchtext 模塊與其他 NLP 技朮（如 spaCy、transformers）結合，進行 EHR 文本的處理與 ICD-11 編碼的提取。這里提供了一個簡單的思路，展示如何使用詞嵌入和深度學習從 EHR 中提取 ICD-11 編碼。
示例流程：
    1. 加載 ICD-11 代碼表：將 ICD-11 代碼與對應的疾病描述加載到系統中，通常是一個 CSV 文件或數據庫。
    2. 處理 EHR 文本：通過 NLP 處理電子健康記錄中的文本信息，提取相關的診斷描述。
    3. 文本匹配與分類：使用模型來將 EHR 中的文本與 ICD-11 代碼進行匹配。
假設我們有 ICD-11 代碼的映射表（CSV 文件），并且 EHR 文本記錄存儲在另一個 CSV 文件中。我們將通過 NLP 提取 EHR 文本中的關鍵信息，并將其映射到 ICD-11 代碼。
import pandas as pd
import torch
import torch.nn as nn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import LabelEncoder

# 1. 加載 ICD-11 數據表和 EHR 數據
icd_data = pd.read_csv('icd11_codes.csv')  # ICD-11 代碼和描述
ehr_data = pd.read_csv('ehr_records.csv')  # EHR 記錄文本

# 2. 使用 TfidfVectorizer 對 EHR 文本進行特徵提取
vectorizer = TfidfVectorizer(max_features=5000)  # 將文本轉換為特徵向量
X = vectorizer.fit_transform(ehr_data['diagnosis_text']).toarray()

# 3. 標簽編碼，將 ICD-11 代碼轉換為數字標簽
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(ehr_data['icd_code'])

# 4. 拆分訓練和測試數據集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. 定義一個簡單的神經網絡模型來預測 ICD-11 代碼
class ICDClassifier(nn.Module):
    def __init__(self, input_size, num_classes):
        super(ICDClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, num_classes)
    
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

# 定義超參數
input_size = X_train.shape[1]
num_classes = len(label_encoder.classes_)
learning_rate = 0.001
batch_size = 64
num_epochs = 10

# 6. 創建 DataLoader
class EHRDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

train_dataset = EHRDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# 7. 定義模型、損失函數和優化器
model = ICDClassifier(input_size, num_classes)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# 8. 訓練模型
for epoch in range(num_epochs):
    for inputs, labels in train_loader:
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# 9. 評估模型
with torch.no_grad():
    test_inputs = torch.tensor(X_test, dtype=torch.float32)
    test_labels = torch.tensor(y_test, dtype=torch.long)
    
    outputs = model(test_inputs)
    _, predicted = torch.max(outputs, 1)
    
    accuracy = (predicted == test_labels).sum().item() / len(test_labels)
    print(f'Accuracy: {accuracy * 100:.2f}%')
說明：
    1. ICD-11 代碼表：通過 icd_data 加載 ICD-11 代碼表，可能包含 ICD 編碼及其描述。
    2. EHR 文本處理：我們使用 TfidfVectorizer 對 EHR 文本進行特徵提取，將文本轉換為向量表示，方便后續的模型處理。
    3. 簡單的神經網絡分類模型：我們定義了一個兩層的全連接神經網絡來預測 ICD-11 代碼。
    4. 訓練與測試：使用 PyTorch 進行訓練，最后通過模型預測并計算分類的准確性。
其他方法：
    1. 預訓練語言模型（如 BERT）：可以使用預訓練的語言模型（如 BERT）來對 EHR 文本進行處理，得到更好的特徵表示，再進行分類。
    2. 規則匹配：對於較為標准化的 EHR 文本，可能還可以使用正則表達式或關鍵字匹配 ICD-11 編碼。
總結
通過自然語言處理（NLP）技朮，可以從 EHR 文本中提取症狀、診斷信息并映射到相應的 ICD-11 編碼。PyTorch 提供了靈活的深度學習框架，可以幫助訓練模型，自動化處理復雜的 EHR 文本，提升編碼效率和准確性
