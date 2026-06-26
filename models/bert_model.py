# models/bert_model.py
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import os


class BERTClassifier(nn.Module):
    """BERT分类器"""

    def __init__(self, bert_model, num_classes=3, dropout=0.3):
        super(BERTClassifier, self).__init__()
        self.bert = bert_model
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return logits


class BERTDataset(Dataset):
    """BERT数据集"""

    def __init__(self, texts, labels, tokenizer, max_len=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_len,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(self.labels[idx], dtype=torch.long)
        }


class BERTClassifierModel:
    """BERT分类器封装类"""

    def __init__(self, model_path=None, batch_size=16, epochs=5, max_len=128):
        self.model_path = model_path or 'paraphrase-multilingual-MiniLM-L12-v2'
        self.batch_size = batch_size
        self.epochs = epochs
        self.max_len = max_len
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.label_map = {'正面': 0, '中性': 1, '负面': 2}
        self.reverse_label_map = {0: '正面', 1: '中性', 2: '负面'}

        self.tokenizer = None
        self.model = None
        self.is_trained = False

    def load_model(self):
        """加载预训练模型和分词器"""
        print(f"正在加载BERT模型从 {self.model_path}...")
        print(f"使用设备: {self.device}")

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        bert_model = AutoModel.from_pretrained(self.model_path)
        self.model = BERTClassifier(bert_model).to(self.device)

        print("BERT模型加载完成")

    def train(self, texts, labels):
        """训练模型"""
        if self.tokenizer is None or self.model is None:
            self.load_model()

        # 转换标签
        y = [self.label_map[label] for label in labels]

        # 分割数据
        X_train, X_val, y_train, y_val = train_test_split(
            texts, y, test_size=0.2, random_state=42, stratify=y
        )

        # 创建数据集
        train_dataset = BERTDataset(X_train, y_train, self.tokenizer, self.max_len)
        val_dataset = BERTDataset(X_val, y_val, self.tokenizer, self.max_len)

        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)

        # 优化器
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=2e-5)
        criterion = nn.CrossEntropyLoss()

        print("开始训练BERT模型...")
        best_accuracy = 0

        for epoch in range(self.epochs):
            # 训练
            self.model.train()
            total_loss = 0
            for batch in train_loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['label'].to(self.device)

                optimizer.zero_grad()
                outputs = self.model(input_ids, attention_mask)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            # 验证
            self.model.eval()
            predictions = []
            true_labels = []
            with torch.no_grad():
                for batch in val_loader:
                    input_ids = batch['input_ids'].to(self.device)
                    attention_mask = batch['attention_mask'].to(self.device)
                    labels = batch['label'].to(self.device)

                    outputs = self.model(input_ids, attention_mask)
                    _, preds = torch.max(outputs, 1)
                    predictions.extend(preds.cpu().numpy())
                    true_labels.extend(labels.cpu().numpy())

            accuracy = accuracy_score(true_labels, predictions)
            print(
                f"Epoch {epoch + 1}/{self.epochs}, Loss: {total_loss / len(train_loader):.4f}, Val Accuracy: {accuracy:.4f}")

            if accuracy > best_accuracy:
                best_accuracy = accuracy
                self.save('models/saved_models/bert_best.pth')

        print(f"\nBERT模型训练完成，最佳验证准确率: {best_accuracy:.4f}")
        self.is_trained = True
        return best_accuracy

    def predict(self, text):
        """预测单条文本"""
        if not self.is_trained or self.model is None:
            raise Exception("模型未训练，请先调用train方法")

        self.model.eval()

        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_len,
            return_tensors='pt'
        )

        input_ids = encoding['input_ids'].to(self.device)
        attention_mask = encoding['attention_mask'].to(self.device)

        with torch.no_grad():
            outputs = self.model(input_ids, attention_mask)
            probabilities = torch.softmax(outputs, dim=1)[0].cpu().numpy()
            pred_class = np.argmax(probabilities)

        return {
            'sentiment': self.reverse_label_map[pred_class],
            'probability': float(probabilities[pred_class]),
            'probabilities': {
                '正面': float(probabilities[0]),
                '中性': float(probabilities[1]),
                '负面': float(probabilities[2])
            }
        }

    def predict_batch(self, texts):
        """批量预测"""
        results = []
        for text in texts:
            results.append(self.predict(text))
        return results

    def save(self, filepath):
        """保存模型"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'label_map': self.label_map,
            'reverse_label_map': self.reverse_label_map
        }, filepath)
        print(f"BERT模型已保存到 {filepath}")

    def load(self, filepath):
        """加载模型"""
        if self.tokenizer is None:
            self.load_model()

        checkpoint = torch.load(filepath, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.label_map = checkpoint['label_map']
        self.reverse_label_map = checkpoint['reverse_label_map']
        self.is_trained = True
        print(f"BERT模型已从 {filepath} 加载")