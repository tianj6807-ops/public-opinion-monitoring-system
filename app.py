# app.py - 完整版（三模型融合Web系统）
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
import os
from datetime import datetime
from collections import Counter
import warnings
import random
warnings.filterwarnings('ignore')

from config import Config
from data.process_data import DataPreprocessor
from models.naive_bayes_model import NaiveBayesClassifier
from models.textcnn_model import TextCNNClassifier
from models.ensemble_model import EnsembleModel
from clustering.cluster_analysis import ClusterAnalyzer
from risk.risk_alert import RiskAlertSystem

# BERT相关导入
import torch
from transformers import AutoTokenizer, AutoModel
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

app = Flask(__name__)
app.config.from_object(Config)
CORS(app)

# 全局变量
data_df = None
nb_model = None
textcnn_model = None
bert_model = None
ensemble_model = None
cluster_analyzer = None
risk_system = None


# ==================== BERT模型定义 ====================
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


class BertFineTuner:
    """BERT微调器 - 用于加载和预测"""

    def __init__(self, model_path, max_len=128):
        self.model_path = model_path
        self.max_len = max_len
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        print(f"BERT使用设备: {self.device}")
        print(f"加载本地BERT模型: {model_path}")

        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.bert_model = AutoModel.from_pretrained(model_path)
        self.model = BertSentimentClassifier(self.bert_model).to(self.device)

        self.label_map = {'正面': 0, '中性': 1, '负面': 2}
        self.reverse_label_map = {0: '正面', 1: '中性', 2: '负面'}
        self.is_trained = False

    def load(self, filepath):
        """加载训练好的模型权重"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.label_map = checkpoint.get('label_map', {'正面': 0, '中性': 1, '负面': 2})
        self.reverse_label_map = {v: k for k, v in self.label_map.items()}
        self.is_trained = True
        print(f"✅ BERT模型已从 {filepath} 加载")

    def predict(self, text):
        """预测单条文本"""
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


# ==================== 系统初始化 ====================
def initialize_system():
    """初始化系统 - 加载三个模型"""
    global data_df, nb_model, textcnn_model, bert_model, ensemble_model, cluster_analyzer, risk_system

    print("=" * 60)
    print("正在初始化网络舆情监测系统（三模型融合）...")
    print("=" * 60)

    # 1. 加载数据（优先使用平衡数据）
    print("\n[1/6] 加载数据...")
    balanced_path = r'D:\网络舆情\qimo\data\balanced_data.csv'
    original_path = Config.DATA_PATH

    if os.path.exists(balanced_path):
        data_df = pd.read_csv(balanced_path)
        print(f"✅ 使用平衡数据: {len(data_df)} 条")
        print(f"   情感分布: {data_df['sentiment'].value_counts().to_dict()}")
    elif os.path.exists(original_path):
        data_df = pd.read_csv(original_path)
        print(f"⚠️ 使用原始数据: {len(data_df)} 条")
    else:
        print("❌ 错误: 数据文件不存在")
        return

    # 2. 加载朴素贝叶斯模型
    print("\n[2/6] 加载朴素贝叶斯模型...")
    nb_model = NaiveBayesClassifier()
    nb_path = os.path.join(Config.MODEL_SAVE_PATH, 'naive_bayes_balanced.pkl')
    if os.path.exists(nb_path):
        nb_model.load(nb_path)
        print("✅ 朴素贝叶斯模型加载成功")
    else:
        print("❌ 朴素贝叶斯模型不存在")
        return

    # 3. 加载TextCNN模型
    print("\n[3/6] 加载TextCNN模型...")
    textcnn_model = TextCNNClassifier()
    textcnn_path = os.path.join(Config.MODEL_SAVE_PATH, 'textcnn_balanced.pth')
    if os.path.exists(textcnn_path):
        textcnn_model.load(textcnn_path)
        print("✅ TextCNN模型加载成功")
    else:
        print("❌ TextCNN模型不存在")
        return

    # 4. 加载BERT模型
    print("\n[4/6] 加载BERT模型...")
    bert_path = os.path.join(Config.MODEL_SAVE_PATH, 'bert_finetuned.pth')
    has_bert = False
    if os.path.exists(bert_path):
        try:
            bert_model = BertFineTuner(model_path=Config.BERT_MODEL_PATH)
            bert_model.load(bert_path)
            print("✅ BERT模型加载成功")
            has_bert = True
        except Exception as e:
            print(f"⚠️ BERT模型加载失败: {e}")
            bert_model = None
    else:
        print("⚠️ BERT模型不存在，将使用双模型融合")
        bert_model = None

    # 5. 创建融合模型
    print("\n[5/6] 创建多模型融合引擎...")
    models_dict = {
        '朴素贝叶斯': nb_model,
        'TextCNN': textcnn_model,
    }
    if has_bert and bert_model is not None:
        models_dict['BERT'] = bert_model
        print("✅ 三模型融合（朴素贝叶斯 + TextCNN + BERT）")
    else:
        print("⚠️ 双模型融合（朴素贝叶斯 + TextCNN）")

    ensemble_model = EnsembleModel(models_dict)

    # 6. 初始化聚类和风险预警
    print("\n[6/6] 初始化其他模块...")
    try:
        global cluster_analyzer
        cluster_analyzer = ClusterAnalyzer(n_clusters=5, method='kmeans')
        cluster_analyzer.perform_clustering(data_df['processed_text'].tolist())
        print("✅ 聚类分析器初始化完成")
    except Exception as e:
        print(f"⚠️ 聚类分析器跳过: {e}")
        cluster_analyzer = None

    risk_system = RiskAlertSystem()

    print("\n" + "=" * 60)
    print("✅ 系统初始化完成！")
    print(f"📊 数据量: {len(data_df)} 条")
    print(f"🤖 模型: {'三模型融合' if has_bert else '双模型融合'}")
    print(f"💻 设备: {'GPU' if torch.cuda.is_available() else 'CPU'}")
    print("🌐 访问地址: http://localhost:5000")
    print("=" * 60)


# ==================== 路由 ====================
@app.route('/')
def index():
    """总览看板页面"""
    return render_template('index.html')


@app.route('/cluster')
def cluster_page():
    """聚类分析页面"""
    return render_template('cluster.html')


@app.route('/predict')
def predict_page():
    """智能研判页面"""
    return render_template('predict.html')


@app.route('/api/dashboard_data')
def get_dashboard_data():
    """获取总览看板数据"""
    global data_df, risk_system

    if data_df is None:
        return jsonify({'error': '数据未加载'}), 500

    # 基本统计
    total_count = len(data_df)
    sentiment_counts = data_df['sentiment'].value_counts().to_dict()

    # ========== 修复1：情感趋势数据 ==========
    # 确保日期格式正确
    data_df['date'] = pd.to_datetime(data_df['publish_time']).dt.date

    # 按日期和情感分组
    daily_sentiment = data_df.groupby(['date', 'sentiment']).size().unstack(fill_value=0)

    # 生成连续的日期范围（过去30天）
    from datetime import datetime, timedelta
    end_date = datetime.now().date()
    start_date = end_date - timedelta(days=30)

    # 如果数据日期太旧，使用数据中的日期范围
    if len(daily_sentiment) > 0:
        min_date = min(daily_sentiment.index)
        max_date = max(daily_sentiment.index)
        if max_date < end_date:
            # 使用数据中的日期范围，但确保至少有7天
            start_date = min_date
            end_date = max_date
            if (end_date - start_date).days < 7:
                start_date = end_date - timedelta(days=7)

    # 生成完整日期序列
    date_range = []
    current = start_date
    while current <= end_date:
        date_range.append(current)
        current += timedelta(days=1)

    # 构建趋势数据
    trend_data = []
    for date in date_range:
        if date in daily_sentiment.index:
            row = daily_sentiment.loc[date]
            trend_data.append({
                'date': date.strftime('%m-%d'),
                '正面': int(row.get('正面', 0)),
                '中性': int(row.get('中性', 0)),
                '负面': int(row.get('负面', 0))
            })
        else:
            trend_data.append({
                'date': date.strftime('%m-%d'),
                '正面': 0,
                '中性': 0,
                '负面': 0
            })

    # 如果趋势数据为空，生成模拟数据用于展示
    if len(trend_data) == 0 or sum(d['正面'] + d['中性'] + d['负面'] for d in trend_data) == 0:
        print("⚠️ 趋势数据为空，生成模拟数据")
        trend_data = []
        for i in range(14):
            date = (datetime.now() - timedelta(days=13 - i)).strftime('%m-%d')
            trend_data.append({
                'date': date,
                '正面': random.randint(5, 20),
                '中性': random.randint(10, 30),
                '负面': random.randint(3, 15)
            })

    # ========== 修复2：词云数据 ==========
    # 从 original_text 中提取关键词（而不是 processed_text）
    all_text = ' '.join(data_df['original_text'].tolist())

    # 使用jieba分词
    import jieba
    words = jieba.lcut(all_text)

    # 过滤停用词和短词
    stopwords = set(['的', '了', '是', '在', '我', '有', '和', '就', '不', '人', '都', '一', '一个',
                     '上', '也', '很', '到', '说', '要', '去', '会', '这', '那', '你', '他', '她', '它',
                     '我们', '你们', '他们', 'this', 'that', 'the', 'and', 'of', 'to', 'for', 'with',
                     'on', 'at', 'by', 'in', 'is', 'are', 'was', 'were', 'be', 'been', 'being'])

    filtered_words = [w for w in words if len(w) > 1 and w not in stopwords and not w.isdigit()]

    # 统计词频
    from collections import Counter
    word_counts = Counter(filtered_words).most_common(30)

    # 格式化为ECharts词云格式
    if len(word_counts) > 0:
        wordcloud_data = [{'name': w, 'value': c} for w, c in word_counts]
    else:
        # 如果没有关键词，生成示例数据
        wordcloud_data = [
            {'name': '舆情', 'value': 100},
            {'name': '监测', 'value': 85},
            {'name': '分析', 'value': 78},
            {'name': '情感', 'value': 72},
            {'name': '预警', 'value': 65}
        ]

    # 风险预警
    risk_status = risk_system.evaluate_risk(data_df)

    return jsonify({
        'total_count': total_count,
        'sentiment_distribution': sentiment_counts,
        'trend_data': trend_data,
        'risk_status': risk_status,
        'wordcloud_data': wordcloud_data
    })

@app.route('/api/cluster_data')
def get_cluster_data():
    """获取聚类分析数据"""
    global data_df, cluster_analyzer

    if data_df is None:
        return jsonify({'error': '数据未加载'}), 500

    if cluster_analyzer is None:
        return jsonify({'error': '聚类分析器未初始化'}), 500

    try:
        # 获取可视化数据
        texts = data_df['processed_text'].tolist()
        vis_data = cluster_analyzer.get_visualization_data(texts)

        # 获取聚类摘要
        sentiments = data_df['sentiment'].tolist()
        times = data_df['publish_time'].tolist()
        summary = cluster_analyzer.get_cluster_summary(texts, sentiments, times)

        return jsonify({
            'clusters': summary,
            'visualization': {
                'points': vis_data['coordinates'],
                'labels': vis_data['cluster_labels'],
                'texts': [text[:100] + '...' for text in vis_data['texts']]
            }
        })
    except Exception as e:
        return jsonify({'error': f'聚类分析失败: {str(e)}'}), 500


@app.route('/api/predict', methods=['POST'])
def predict_text():
    """预测单条文本情感"""
    global ensemble_model

    if ensemble_model is None:
        return jsonify({'error': '模型未加载'}), 500

    data = request.get_json()
    text = data.get('text', '')

    if not text:
        return jsonify({'error': '请输入文本'}), 400

    # 预处理
    preprocessor = DataPreprocessor()
    processed_text, _ = preprocessor.preprocess(text)

    # 预测
    result = ensemble_model.predict(processed_text)

    return jsonify({
        'original_text': text,
        'processed_text': processed_text,
        'prediction': result
    })


@app.route('/api/batch_predict', methods=['POST'])
def batch_predict():
    """批量预测"""
    global data_df, ensemble_model

    if ensemble_model is None:
        return jsonify({'error': '模型未加载'}), 500

    data = request.get_json()
    sample_size = min(data.get('sample_size', 100), len(data_df))

    sample_df = data_df.head(sample_size)
    results = []

    for _, row in sample_df.iterrows():
        result = ensemble_model.predict(row['processed_text'])
        results.append({
            'text': row['original_text'][:100],
            'true_sentiment': row['sentiment'],
            'predicted_sentiment': result['sentiment'],
            'confidence': result['confidence']
        })

    # 计算准确率
    correct = sum(1 for r in results if r['true_sentiment'] == r['predicted_sentiment'])
    accuracy = correct / len(results) if results else 0

    return jsonify({
        'results': results,
        'accuracy': accuracy,
        'total': len(results)
    })


@app.route('/api/risk_trend')
def get_risk_trend():
    """获取风险趋势数据"""
    global data_df, risk_system

    if data_df is None:
        return jsonify({'error': '数据未加载'}), 500

    trend_data = risk_system.get_trend_data(data_df)

    return jsonify({
        'trend_data': trend_data,
        'current_risk': risk_system.evaluate_risk(data_df)
    })


@app.route('/api/refresh', methods=['POST'])
def refresh_data():
    """刷新数据"""
    global data_df, cluster_analyzer

    preprocessor = DataPreprocessor()
    new_data = preprocessor.create_sample_data(100)
    data_df = pd.concat([data_df, new_data], ignore_index=True)
    preprocessor.save_data(data_df, Config.DATA_PATH)

    # 重新聚类
    if cluster_analyzer is not None:
        cluster_analyzer.perform_clustering(data_df['processed_text'].tolist())

    return jsonify({
        'message': '数据刷新成功',
        'new_count': len(new_data),
        'total_count': len(data_df)
    })


# ==================== 启动应用 ====================
if __name__ == '__main__':
    # 创建必要的目录
    os.makedirs('data', exist_ok=True)
    os.makedirs('models/saved_models', exist_ok=True)
    os.makedirs('templates', exist_ok=True)
    os.makedirs('static/css', exist_ok=True)
    os.makedirs('static/js', exist_ok=True)

    # 导入numpy
    import numpy as np

    # 初始化系统
    initialize_system()

    # 启动应用
    print("\n🚀 启动Web服务器...")
    app.run(debug=True, host='0.0.0.0', port=5000)