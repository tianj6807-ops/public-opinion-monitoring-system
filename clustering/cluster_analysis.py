# clustering/cluster_analysis.py
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

import jieba
from collections import Counter


class ClusterAnalyzer:
    """聚类分析器"""

    def __init__(self, n_clusters=5, method='kmeans'):
        self.n_clusters = n_clusters
        self.method = method
        self.vectorizer = TfidfVectorizer(
            max_features=2000,
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.95
        )
        self.kmeans = None
        self.dbscan = None
        self.cluster_labels = None
        self.pca = PCA(n_components=2)
        self.tsne = None
        self.is_fitted = False

    def extract_features(self, texts):
        """提取TF-IDF特征"""
        print("提取TF-IDF特征...")
        X = self.vectorizer.fit_transform(texts)
        print(f"特征矩阵形状: {X.shape}")
        return X

    def perform_clustering(self, texts):
        """执行聚类"""
        X = self.extract_features(texts)

        if self.method == 'kmeans':
            print(f"执行K-Means聚类，簇数: {self.n_clusters}")
            self.kmeans = KMeans(n_clusters=self.n_clusters, random_state=42, n_init=10)
            self.cluster_labels = self.kmeans.fit_predict(X)

            # 计算聚类中心的关键词
            self.cluster_keywords = self._get_cluster_keywords(X)

        elif self.method == 'dbscan':
            print("执行DBSCAN聚类")
            self.dbscan = DBSCAN(eps=0.5, min_samples=5, metric='cosine')
            self.cluster_labels = self.dbscan.fit_predict(X)
            self.n_clusters = len(set(self.cluster_labels)) - (1 if -1 in self.cluster_labels else 0)
            print(f"DBSCAN发现 {self.n_clusters} 个簇")
            self.cluster_keywords = self._get_cluster_keywords(X)

        self.is_fitted = True
        return self.cluster_labels

    def _get_cluster_keywords(self, X, top_k=10):
        """获取每个簇的关键词"""
        feature_names = self.vectorizer.get_feature_names_out()
        cluster_keywords = {}

        for cluster_id in range(self.n_clusters):
            # 获取当前簇的样本索引
            cluster_indices = np.where(self.cluster_labels == cluster_id)[0]

            if len(cluster_indices) == 0:
                cluster_keywords[cluster_id] = []
                continue

            # 计算簇中心
            cluster_center = X[cluster_indices].mean(axis=0)
            cluster_center = np.asarray(cluster_center).flatten()

            # 获取最重要的特征
            top_indices = cluster_center.argsort()[-top_k:][::-1]
            keywords = [(feature_names[i], cluster_center[i]) for i in top_indices if cluster_center[i] > 0]
            cluster_keywords[cluster_id] = keywords

        return cluster_keywords

    def reduce_dimension(self, X, method='pca'):
        """降维用于可视化"""
        if method == 'pca':
            reduced = self.pca.fit_transform(X.toarray() if hasattr(X, 'toarray') else X)
            print(f"PCA降维完成，解释方差比: {self.pca.explained_variance_ratio_}")
        elif method == 'tsne':
            tsne = TSNE(n_components=2, random_state=42, perplexity=30)
            reduced = tsne.fit_transform(X.toarray() if hasattr(X, 'toarray') else X)

        return reduced

    def get_cluster_summary(self, texts, sentiments, times):
        """获取聚类摘要"""
        if not self.is_fitted:
            raise Exception("请先执行聚类")

        cluster_summary = []

        for cluster_id in range(self.n_clusters):
            cluster_indices = np.where(self.cluster_labels == cluster_id)[0]

            if len(cluster_indices) == 0:
                continue

            cluster_texts = [texts[i] for i in cluster_indices]
            cluster_sentiments = [sentiments[i] for i in cluster_indices]
            cluster_times = [times[i] for i in cluster_indices]

            # 情感分布
            sentiment_counts = Counter(cluster_sentiments)
            dominant_sentiment = max(sentiment_counts, key=sentiment_counts.get)

            # 生成观点标签
            keywords = self.cluster_keywords.get(cluster_id, [])
            if keywords:
                top_keywords = [kw[0] for kw in keywords[:3]]
                if dominant_sentiment == '正面':
                    tag = f"支持派（{','.join(top_keywords)}）"
                elif dominant_sentiment == '负面':
                    tag = f"反对派（{','.join(top_keywords)}）"
                else:
                    tag = f"中立观望者（{','.join(top_keywords)}）"
            else:
                tag = f"观点组{cluster_id + 1}"

            cluster_summary.append({
                'cluster_id': int(cluster_id),
                'size': len(cluster_indices),
                'percentage': len(cluster_indices) / len(self.cluster_labels) * 100,
                'dominant_sentiment': dominant_sentiment,
                'sentiment_distribution': dict(sentiment_counts),
                'tag': tag,
                'keywords': [{'word': kw[0], 'weight': float(kw[1])} for kw in keywords[:10]],
                'sample_texts': cluster_texts[:5],
                'time_range': {
                    'min': min(cluster_times) if cluster_times else None,
                    'max': max(cluster_times) if cluster_times else None
                }
            })

        return cluster_summary

    def get_visualization_data(self, texts):
        """获取可视化数据"""
        X = self.extract_features(texts)
        reduced = self.reduce_dimension(X, method='pca')

        return {
            'coordinates': reduced.tolist(),
            'cluster_labels': self.cluster_labels.tolist(),
            'texts': texts
        }


# 示例使用
if __name__ == "__main__":
    # 加载数据
    df = pd.read_csv('data/processed_data.csv')
    texts = df['processed_text'].tolist()
    sentiments = df['sentiment'].tolist()
    times = df['publish_time'].tolist()

    # 执行聚类
    cluster_analyzer = ClusterAnalyzer(n_clusters=5, method='kmeans')
    cluster_labels = cluster_analyzer.perform_clustering(texts)

    # 获取聚类摘要
    summary = cluster_analyzer.get_cluster_summary(texts, sentiments, times)

    print("\n聚类结果摘要:")
    for cluster in summary:
        print(f"\n{cluster['tag']}")
        print(f"  样本数: {cluster['size']} ({cluster['percentage']:.1f}%)")
        print(f"  主导情感: {cluster['dominant_sentiment']}")
        print(f"  关键词: {', '.join([kw['word'] for kw in cluster['keywords'][:5]])}")

    # 获取可视化数据
    vis_data = cluster_analyzer.get_visualization_data(texts)
    print(f"\n可视化数据准备完成，共 {len(vis_data['coordinates'])} 个点")