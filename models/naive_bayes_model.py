# models/naive_bayes_model.py
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import joblib
import os


class NaiveBayesClassifier:
    """朴素贝叶斯分类器"""

    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.95
        )
        self.classifier = MultinomialNB(alpha=1.0)
        self.label_map = {'正面': 0, '中性': 1, '负面': 2}
        self.reverse_label_map = {0: '正面', 1: '中性', 2: '负面'}
        self.is_trained = False

    def train(self, texts, labels):
        """训练模型"""
        print("开始训练朴素贝叶斯模型...")

        # 转换标签
        y = np.array([self.label_map[label] for label in labels])

        # TF-IDF特征提取
        X = self.vectorizer.fit_transform(texts)
        print(f"特征维度: {X.shape}")

        # 分割训练集和验证集
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        # 训练
        self.classifier.fit(X_train, y_train)

        # 验证
        y_pred = self.classifier.predict(X_val)
        accuracy = accuracy_score(y_val, y_pred)
        print(f"朴素贝叶斯模型验证准确率: {accuracy:.4f}")
        print("\n分类报告:")
        print(classification_report(y_val, y_pred,
                                    target_names=['正面', '中性', '负面']))

        self.is_trained = True
        return accuracy

    def predict(self, text):
        """预测单条文本的情感"""
        if not self.is_trained:
            raise Exception("模型未训练，请先调用train方法")

        # 特征提取
        X = self.vectorizer.transform([text])

        # 预测类别和概率
        pred_class = self.classifier.predict(X)[0]
        probabilities = self.classifier.predict_proba(X)[0]

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
        joblib.dump({
            'vectorizer': self.vectorizer,
            'classifier': self.classifier,
            'label_map': self.label_map,
            'reverse_label_map': self.reverse_label_map
        }, filepath)
        print(f"朴素贝叶斯模型已保存到 {filepath}")

    def load(self, filepath):
        """加载模型"""
        data = joblib.load(filepath)
        self.vectorizer = data['vectorizer']
        self.classifier = data['classifier']
        self.label_map = data['label_map']
        self.reverse_label_map = data['reverse_label_map']
        self.is_trained = True
        print(f"朴素贝叶斯模型已从 {filepath} 加载")


# 测试代码
if __name__ == "__main__":
    # 加载数据
    df = pd.read_csv('data/processed_data.csv')

    # 创建模型并训练
    nb_model = NaiveBayesClassifier()
    nb_model.train(df['processed_text'].tolist(), df['sentiment'].tolist())

    # 测试预测
    test_text = "这个产品太棒了，我非常喜欢"
    result = nb_model.predict(test_text)
    print(f"\n测试预测: '{test_text}'")
    print(f"预测结果: {result}")

    # 保存模型
    nb_model.save('models/saved_models/naive_bayes.pkl')