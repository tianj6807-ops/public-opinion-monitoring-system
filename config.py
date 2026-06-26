# config.py
import os


class Config:
    SECRET_KEY = 'your-secret-key-here'

    # 数据路径
    DATA_PATH = 'data/processed_data.csv'
    RAW_DATA_PATH = 'data/raw_data.csv'

    # 20 Newsgroups数据路径
    NEWSGROUPS_PATH = '20news-bydate'

    # BERT模型路径（本地下载的模型）
    BERT_MODEL_PATH = 'paraphrase-multilingual-MiniLM-L12-v2'

    # 模型保存路径
    MODEL_SAVE_PATH = 'models/saved_models/'

    # 聚类参数
    N_CLUSTERS = 5

    # 风险预警阈值
    RISK_THRESHOLDS = {
        'green': 0.2,  # 负面比例低于20%
        'yellow': 0.5,  # 负面比例20%-50%
        'red': 0.5  # 负面比例超过50%
    }

    # 上传配置
    UPLOAD_FOLDER = 'uploads/'
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024