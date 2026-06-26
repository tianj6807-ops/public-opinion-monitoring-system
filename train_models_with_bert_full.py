# train_models_with_bert_full.py - 完整三模型训练（含BERT）
import pandas as pd
import os
import sys
import warnings

warnings.filterwarnings('ignore')

sys.path.insert(0, r'D:\网络舆情\qimo')

from config import Config
from models.naive_bayes_model import NaiveBayesClassifier
from models.textcnn_model import TextCNNClassifier
from models.ensemble_model import EnsembleModel
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import torch

# 导入BERT微调模型
from transformers import AutoTokenizer, AutoModel
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


class BertSentimentClassifier(nn.Module):
    """使用本地BERT模型的情感分类器"""

    def __init__(self, bert_model, num_classes=3, dropout=0.3):
        super(BertSentimentClassifier, self).__init__()
        self.bert = bert_model
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return logits


class SentimentDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.label_map = {'正面': 0, '中性': 1, '负面': 2}

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
            'label': torch.tensor(self.label_map[self.labels[idx]], dtype=torch.long)
        }


class BertFineTuner:
    """BERT微调器"""

    def __init__(self, model_path, batch_size=8, epochs=5, max_len=128, lr=2e-5):
        self.model_path = model_path
        self.batch_size = batch_size
        self.epochs = epochs
        self.max_len = max_len
        self.lr = lr
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        print(f"使用设备: {self.device}")
        print(f"加载本地BERT模型: {model_path}")

        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.bert_model = AutoModel.from_pretrained(model_path)
        self.model = BertSentimentClassifier(self.bert_model).to(self.device)

        self.label_map = {'正面': 0, '中性': 1, '负面': 2}
        self.reverse_label_map = {0: '正面', 1: '中性', 2: '负面'}
        self.is_trained = False

    def train(self, texts, labels):
        print(f"\n开始微调BERT模型 (共{self.epochs}轮)...")
        print("注意：BERT训练较慢，请耐心等待...")

        X_train, X_val, y_train, y_val = train_test_split(
            texts, labels, test_size=0.2, random_state=42, stratify=labels
        )

        train_dataset = SentimentDataset(X_train, y_train, self.tokenizer, self.max_len)
        val_dataset = SentimentDataset(X_val, y_val, self.tokenizer, self.max_len)

        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)

        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr)
        criterion = nn.CrossEntropyLoss()

        best_accuracy = 0

        for epoch in range(self.epochs):
            self.model.train()
            total_loss = 0
            for batch in train_loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels_batch = batch['label'].to(self.device)

                optimizer.zero_grad()
                outputs = self.model(input_ids, attention_mask)
                loss = criterion(outputs, labels_batch)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            self.model.eval()
            predictions = []
            true_labels = []
            with torch.no_grad():
                for batch in val_loader:
                    input_ids = batch['input_ids'].to(self.device)
                    attention_mask = batch['attention_mask'].to(self.device)
                    labels_batch = batch['label'].to(self.device)

                    outputs = self.model(input_ids, attention_mask)
                    _, preds = torch.max(outputs, 1)
                    predictions.extend(preds.cpu().numpy())
                    true_labels.extend(labels_batch.cpu().numpy())

            accuracy = accuracy_score(true_labels, predictions)
            print(
                f"Epoch {epoch + 1}/{self.epochs}, Loss: {total_loss / len(train_loader):.4f}, Val Acc: {accuracy:.4f}")

            if accuracy > best_accuracy:
                best_accuracy = accuracy
                self.save('models/saved_models/bert_finetuned.pth')

        self.is_trained = True
        print(f"\n✅ BERT微调完成！最佳验证准确率: {best_accuracy:.4f}")
        return best_accuracy

    def predict(self, text):
        if not self.is_trained:
            raise Exception("BERT模型未训练")

        self.model.eval()

        encoding = self.tokenizer(
            text[:self.max_len],
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

    def save(self, filepath):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'label_map': self.label_map,
            'reverse_label_map': self.reverse_label_map
        }, filepath)

    def load(self, filepath):
        checkpoint = torch.load(filepath, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.label_map = checkpoint['label_map']
        self.reverse_label_map = checkpoint['reverse_label_map']
        self.is_trained = True
        print(f"BERT模型已从 {filepath} 加载")


def train_all_models():
    """训练所有三个模型"""
    print("=" * 70)
    print("完整三模型训练（朴素贝叶斯 + TextCNN + BERT）")
    print("=" * 70)

    # 加载平衡数据
    balanced_path = r'D:\网络舆情\qimo\data\balanced_data.csv'
    df = pd.read_csv(balanced_path)
    print(f"\n✅ 已加载 {len(df)} 条平衡数据")
    print(f"情感分布: {df['sentiment'].value_counts().to_dict()}")

    texts = df['processed_text'].tolist()
    labels = df['sentiment'].tolist()

    os.makedirs(Config.MODEL_SAVE_PATH, exist_ok=True)

    # 1. 训练朴素贝叶斯
    print("\n" + "=" * 50)
    print("[1/3] 训练朴素贝叶斯模型...")
    print("=" * 50)
    nb_model = NaiveBayesClassifier()
    nb_model.train(texts, labels)
    nb_model.save(os.path.join(Config.MODEL_SAVE_PATH, 'naive_bayes_balanced.pkl'))
    print("✅ 朴素贝叶斯模型保存完成")

    # 2. 训练TextCNN
    print("\n" + "=" * 50)
    print("[2/3] 训练TextCNN模型...")
    print("=" * 50)
    textcnn_model = TextCNNClassifier(epochs=15, batch_size=32)
    textcnn_model.train(texts, labels)
    textcnn_model.save(os.path.join(Config.MODEL_SAVE_PATH, 'textcnn_balanced.pth'))
    print("✅ TextCNN模型保存完成")

    # 3. 训练BERT（微调）
    print("\n" + "=" * 50)
    print("[3/3] 训练BERT模型（微调）...")
    print("=" * 50)
    print("这可能需要10-30分钟，请耐心等待...")

    bert_model = BertFineTuner(
        model_path=Config.BERT_MODEL_PATH,
        batch_size=8,
        epochs=5,
        max_len=128
    )
    bert_accuracy = bert_model.train(texts, labels)
    bert_model.save(os.path.join(Config.MODEL_SAVE_PATH, 'bert_finetuned.pth'))
    print(f"✅ BERT模型保存完成，准确率: {bert_accuracy:.4f}")

    # 4. 创建三模型融合
    print("\n" + "=" * 50)
    print("创建三模型融合投票系统...")
    print("=" * 50)

    ensemble_model = EnsembleModel({
        '朴素贝叶斯': nb_model,
        'TextCNN': textcnn_model,
        'BERT': bert_model
    })

    # 5. 评估三模型融合效果
    print("\n评估三模型融合性能...")
    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels, test_size=0.2, random_state=42, stratify=labels
    )

    predictions = []
    for text in X_test:
        result = ensemble_model.predict(text)
        predictions.append(result['sentiment'])

    accuracy = accuracy_score(y_test, predictions)

    print("\n" + "=" * 60)
    print("三模型融合评估结果:")
    print(f"  融合模型准确率: {accuracy:.4f} ({accuracy * 100:.2f}%)")
    print("\n详细分类报告:")
    print(classification_report(y_test, predictions, target_names=['正面', '中性', '负面']))

    print("\n混淆矩阵:")
    cm = confusion_matrix(y_test, predictions, labels=['正面', '中性', '负面'])
    print("           预测正面  预测中性  预测负面")
    print(f"实际正面     {cm[0][0]:>6}   {cm[0][1]:>6}   {cm[0][2]:>6}")
    print(f"实际中性     {cm[1][0]:>6}   {cm[1][1]:>6}   {cm[1][2]:>6}")
    print(f"实际负面     {cm[2][0]:>6}   {cm[2][1]:>6}   {cm[2][2]:>6}")

    print("\n" + "=" * 60)
    print("✅ 所有模型训练完成！")
    print(f"模型保存路径: {Config.MODEL_SAVE_PATH}")
    print("=" * 60)

    return ensemble_model


if __name__ == "__main__":
    import numpy as np

    train_all_models()