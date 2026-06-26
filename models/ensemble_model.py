# models/ensemble_model.py
import numpy as np
from collections import Counter


class EnsembleModel:
    """多模型融合投票机制"""

    def __init__(self, models):
        """
        初始化融合模型
        models: 字典，包含各个模型的实例
        """
        self.models = models
        self.model_names = list(models.keys())

    def predict(self, text):
        """
        使用投票机制预测
        返回融合后的结果
        """
        predictions = []
        probabilities = {}

        for name, model in self.models.items():
            try:
                result = model.predict(text)
                predictions.append(result['sentiment'])
                probabilities[name] = result
            except Exception as e:
                print(f"模型 {name} 预测失败: {e}")
                predictions.append('中性')  # 默认中性
                probabilities[name] = {'sentiment': '中性', 'probability': 0.33}

        # 投票统计
        vote_counts = Counter(predictions)
        most_common = vote_counts.most_common(1)[0]
        majority_vote = most_common[0]
        votes = most_common[1]

        # 计算置信度
        confidence = votes / len(self.models)

        # 兜底策略：如果三个模型结果各不相同
        if len(set(predictions)) == 3:
            # 选择置信度最高的模型结果
            best_model = max(probabilities.items(),
                             key=lambda x: x[1]['probability'])
            majority_vote = best_model[1]['sentiment']
            confidence = best_model[1]['probability']

        # 计算平均概率
        avg_probabilities = {'正面': 0, '中性': 0, '负面': 0}
        for name, prob in probabilities.items():
            for k in avg_probabilities.keys():
                avg_probabilities[k] += prob['probabilities'][k]
        for k in avg_probabilities.keys():
            avg_probabilities[k] /= len(self.models)

        return {
            'sentiment': majority_vote,
            'confidence': confidence,
            'votes': {name: predictions[i] for i, name in enumerate(self.model_names)},
            'vote_count': dict(vote_counts),
            'average_probabilities': avg_probabilities,
            'individual_results': probabilities
        }

    def predict_batch(self, texts):
        """批量预测"""
        results = []
        for text in texts:
            results.append(self.predict(text))
        return results

    def evaluate(self, texts, true_labels):
        """评估融合模型性能"""
        predictions = []
        for text in texts:
            result = self.predict(text)
            predictions.append(result['sentiment'])

        from sklearn.metrics import classification_report, accuracy_score
        accuracy = accuracy_score(true_labels, predictions)
        report = classification_report(true_labels, predictions,
                                       target_names=['正面', '中性', '负面'])

        print(f"\n融合模型评估结果:")
        print(f"准确率: {accuracy:.4f}")
        print(report)

        return {
            'accuracy': accuracy,
            'classification_report': report
        }