# models/textcnn_model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import joblib
import os
from collections import Counter


class TextCNN(nn.Module):
    """TextCNN模型"""

    def __init__(self, vocab_size, embedding_dim, num_classes,
                 filter_sizes=[2, 3, 4], num_filters=100, dropout=0.5):
        super(TextCNN, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.convs = nn.ModuleList([
            nn.Conv2d(1, num_filters, (fs, embedding_dim))
            for fs in filter_sizes
        ])
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(len(filter_sizes) * num_filters, num_classes)

    def forward(self, x):
        # x: (batch_size, seq_len)
        x = self.embedding(x)  # (batch_size, seq_len, embedding_dim)
        x = x.unsqueeze(1)  # (batch_size, 1, seq_len, embedding_dim)

        # 卷积 + 池化
        conv_outputs = []
        for conv in self.convs:
            conv_out = F.relu(conv(x)).squeeze(3)  # (batch_size, num_filters, seq_len - filter_size + 1)
            pooled = F.max_pool1d(conv_out, conv_out.size(2)).squeeze(2)  # (batch_size, num_filters)
            conv_outputs.append(pooled)

        # 拼接
        x = torch.cat(conv_outputs, dim=1)  # (batch_size, num_filters * len(filter_sizes))
        x = self.dropout(x)
        x = self.fc(x)  # (batch_size, num_classes)

        return x


class TextCNNDataset(Dataset):
    """TextCNN数据集"""

    def __init__(self, texts, labels, word2idx, max_len=100):
        self.texts = texts
        self.labels = labels
        self.word2idx = word2idx
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        # 将文本转换为索引序列
        indices = [self.word2idx.get(word, self.word2idx.get('<UNK>', 0))
                   for word in text.split()[:self.max_len]]
        # 填充
        if len(indices) < self.max_len:
            indices += [self.word2idx.get('<PAD>', 0)] * (self.max_len - len(indices))

        return torch.tensor(indices, dtype=torch.long), torch.tensor(self.labels[idx], dtype=torch.long)


class TextCNNClassifier:
    """TextCNN分类器"""

    def __init__(self, embedding_dim=100, num_filters=100,
                 filter_sizes=[2, 3, 4], dropout=0.5, batch_size=32, epochs=20):
        self.embedding_dim = embedding_dim
        self.num_filters = num_filters
        self.filter_sizes = filter_sizes
        self.dropout = dropout
        self.batch_size = batch_size
        self.epochs = epochs
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.word2idx = {'<PAD>': 0, '<UNK>': 1}
        self.idx2word = {0: '<PAD>', 1: '<UNK>'}
        self.vocab_size = 2
        self.label_map = {'正面': 0, '中性': 1, '负面': 2}
        self.reverse_label_map = {0: '正面', 1: '中性', 2: '负面'}

        self.model = None
        self.is_trained = False

    def build_vocab(self, texts, max_vocab_size=10000):
        """构建词汇表"""
        word_counts = Counter()
        for text in texts:
            words = text.split()
            word_counts.update(words)

        # 保留最常见的词
        for word, _ in word_counts.most_common(max_vocab_size - 2):
            if word not in self.word2idx:
                self.word2idx[word] = self.vocab_size
                self.idx2word[self.vocab_size] = word
                self.vocab_size += 1

        print(f"词汇表大小: {self.vocab_size}")

    def train(self, texts, labels):
        """训练模型"""
        print(f"使用设备: {self.device}")
        print("开始构建词汇表...")
        self.build_vocab(texts)

        # 转换标签
        y = [self.label_map[label] for label in labels]

        # 分割数据
        X_train, X_val, y_train, y_val = train_test_split(
            texts, y, test_size=0.2, random_state=42, stratify=y
        )

        # 创建数据集
        train_dataset = TextCNNDataset(X_train, y_train, self.word2idx)
        val_dataset = TextCNNDataset(X_val, y_val, self.word2idx)

        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)

        # 创建模型
        self.model = TextCNN(
            vocab_size=self.vocab_size,
            embedding_dim=self.embedding_dim,
            num_classes=3,
            filter_sizes=self.filter_sizes,
            num_filters=self.num_filters,
            dropout=self.dropout
        ).to(self.device)

        # 优化器和损失函数
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()

        print("开始训练TextCNN模型...")
        best_accuracy = 0

        for epoch in range(self.epochs):
            # 训练
            self.model.train()
            total_loss = 0
            for batch_texts, batch_labels in train_loader:
                batch_texts = batch_texts.to(self.device)
                batch_labels = batch_labels.to(self.device)

                optimizer.zero_grad()
                outputs = self.model(batch_texts)
                loss = criterion(outputs, batch_labels)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            # 验证
            self.model.eval()
            predictions = []
            true_labels = []
            with torch.no_grad():
                for batch_texts, batch_labels in val_loader:
                    batch_texts = batch_texts.to(self.device)
                    outputs = self.model(batch_texts)
                    _, preds = torch.max(outputs, 1)
                    predictions.extend(preds.cpu().numpy())
                    true_labels.extend(batch_labels.numpy())

            accuracy = accuracy_score(true_labels, predictions)
            print(
                f"Epoch {epoch + 1}/{self.epochs}, Loss: {total_loss / len(train_loader):.4f}, Val Accuracy: {accuracy:.4f}")

            if accuracy > best_accuracy:
                best_accuracy = accuracy
                self.save('models/saved_models/textcnn_best.pth')

        print(f"\nTextCNN模型训练完成，最佳验证准确率: {best_accuracy:.4f}")
        self.is_trained = True
        return best_accuracy

    def predict(self, text):
        """预测单条文本"""
        if not self.is_trained or self.model is None:
            raise Exception("模型未训练，请先调用train方法")

        self.model.eval()

        # 处理文本
        words = text.split()[:100]
        indices = [self.word2idx.get(word, self.word2idx['<UNK>']) for word in words]
        if len(indices) < 100:
            indices += [self.word2idx['<PAD>']] * (100 - len(indices))

        input_tensor = torch.tensor([indices], dtype=torch.long).to(self.device)

        with torch.no_grad():
            outputs = self.model(input_tensor)
            probabilities = F.softmax(outputs, dim=1)[0].cpu().numpy()
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
            'word2idx': self.word2idx,
            'idx2word': self.idx2word,
            'vocab_size': self.vocab_size,
            'label_map': self.label_map,
            'reverse_label_map': self.reverse_label_map,
            'embedding_dim': self.embedding_dim,
            'num_filters': self.num_filters,
            'filter_sizes': self.filter_sizes
        }, filepath)
        print(f"TextCNN模型已保存到 {filepath}")

    def load(self, filepath):
        """加载模型"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.word2idx = checkpoint['word2idx']
        self.idx2word = checkpoint['idx2word']
        self.vocab_size = checkpoint['vocab_size']
        self.label_map = checkpoint['label_map']
        self.reverse_label_map = checkpoint['reverse_label_map']
        self.embedding_dim = checkpoint['embedding_dim']
        self.num_filters = checkpoint['num_filters']
        self.filter_sizes = checkpoint['filter_sizes']

        self.model = TextCNN(
            vocab_size=self.vocab_size,
            embedding_dim=self.embedding_dim,
            num_classes=3,
            filter_sizes=self.filter_sizes,
            num_filters=self.num_filters,
            dropout=self.dropout
        ).to(self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.is_trained = True
        print(f"TextCNN模型已从 {filepath} 加载")