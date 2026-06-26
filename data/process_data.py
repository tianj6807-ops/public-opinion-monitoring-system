# data/process_data.py
import pandas as pd
import numpy as np
import jieba
import re
import os
import random
from datetime import datetime, timedelta
from pathlib import Path


class DataPreprocessor:
    """数据预处理类 - 直接读取本地20 Newsgroups数据"""

    def __init__(self, data_root=None):
        self.stopwords = self._load_stopwords()
        # 设置数据根目录
        if data_root is None:
            # 默认路径：项目根目录下的20news-bydate
            self.data_root = Path(__file__).parent.parent / "20news-bydate"
        else:
            self.data_root = Path(data_root)

    def _load_stopwords(self):
        """加载停用词表"""
        stopwords = set()
        # 中文停用词
        chinese_stopwords = ['的', '了', '是', '在', '我', '有', '和', '就', '不', '人', '都', '一', '一个', '上', '也',
                             '很', '到', '说', '要', '去', '会', '这', '那', '你', '他', '她', '它', '我们', '你们',
                             '他们', '她们', '它们', '这个', '那个', '这些', '那些', '自己', '什么', '怎么', '为什么',
                             '可以', '没有', '但是', '因为', '所以', '如果', '虽然', '然而', '对于', '关于', '以及',
                             '等等', '这样', '那样']
        # 英文停用词
        english_stopwords = ['the', 'a', 'an', 'and', 'or', 'but', 'so', 'for', 'of', 'to', 'in', 'on', 'at', 'with',
                             'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having',
                             'do', 'does', 'did', 'doing', 'will', 'would', 'could', 'should', 'may', 'might', 'must',
                             'this', 'that', 'these', 'those', 'it', 'they', 'we', 'you', 'he', 'she', 'it', 'them',
                             'us', 'your', 'my', 'his', 'her', 'its', 'their', 'our', 'some', 'any', 'no', 'yes']

        stopwords.update(chinese_stopwords)
        stopwords.update(english_stopwords)
        return stopwords

    def clean_text(self, text):
        """清洗文本"""
        if not isinstance(text, str):
            return ""
        # 移除特殊符号，保留中英文和基本标点
        text = re.sub(r'[^\u4e00-\u9fa5a-zA-Z\s\.\,\!\?\'\"]', ' ', text)
        # 移除多余空格
        text = re.sub(r'\s+', ' ', text)
        # 转小写
        text = text.lower()
        return text.strip()

    def tokenize(self, text):
        """分词（支持中英文混合）"""
        words = []
        # 使用jieba进行中文分词
        for word in jieba.cut(text):
            # 过滤停用词和过短的词
            if word not in self.stopwords and len(word) > 1 and not word.isdigit():
                words.append(word)
        return words

    def preprocess(self, text):
        """完整的预处理流程"""
        cleaned = self.clean_text(text)
        tokens = self.tokenize(cleaned)
        return ' '.join(tokens), tokens

    def read_local_20newsgroups(self, sample_size=500):
        """直接读取本地20 Newsgroups数据"""
        print("=" * 60)
        print("正在从本地读取20 Newsgroups数据集...")
        print(f"数据路径: {self.data_root}")
        print("=" * 60)

        train_path = self.data_root / "20news-bydate-train"
        test_path = self.data_root / "20news-bydate-test"

        # 检查目录是否存在
        if not train_path.exists():
            raise FileNotFoundError(f"训练集目录不存在: {train_path}")
        if not test_path.exists():
            raise FileNotFoundError(f"测试集目录不存在: {test_path}")

        texts = []
        categories = []
        file_info = []  # 存储文件路径信息

        # 读取训练集
        print(f"\n[1/2] 读取训练集: {train_path}")
        for category_dir in sorted(train_path.iterdir()):
            if category_dir.is_dir():
                category_name = category_dir.name
                print(f"  处理类别: {category_name}")

                count = 0
                for file_path in category_dir.iterdir():
                    if file_path.is_file():
                        try:
                            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                                content = f.read()
                                # 提取邮件正文（跳过邮件头）
                                content = self._extract_email_body(content)
                                if len(content) > 100:  # 只保留足够长的内容
                                    texts.append(content[:1500])  # 限制长度
                                    categories.append(category_name)
                                    file_info.append(str(file_path))
                                    count += 1
                        except Exception as e:
                            print(f"    读取失败: {file_path.name} - {e}")
                print(f"    读取了 {count} 个文件")

        # 读取测试集
        print(f"\n[2/2] 读取测试集: {test_path}")
        for category_dir in sorted(test_path.iterdir()):
            if category_dir.is_dir():
                category_name = category_dir.name
                print(f"  处理类别: {category_name}")

                count = 0
                for file_path in category_dir.iterdir():
                    if file_path.is_file():
                        try:
                            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                                content = f.read()
                                content = self._extract_email_body(content)
                                if len(content) > 100:
                                    texts.append(content[:1500])
                                    categories.append(category_name)
                                    file_info.append(str(file_path))
                                    count += 1
                        except Exception as e:
                            print(f"    读取失败: {file_path.name} - {e}")
                print(f"    读取了 {count} 个文件")

        print(f"\n总共读取了 {len(texts)} 条数据")

        # 如果数据量超过sample_size，进行采样
        if len(texts) > sample_size:
            indices = random.sample(range(len(texts)), sample_size)
            texts = [texts[i] for i in indices]
            categories = [categories[i] for i in indices]
            file_info = [file_info[i] for i in indices]
            print(f"采样后保留 {sample_size} 条数据")

        return texts, categories, file_info

    def _extract_email_body(self, email_content):
        """从邮件内容中提取正文（去除邮件头）"""
        lines = email_content.split('\n')
        body_lines = []
        in_header = True

        for line in lines:
            # 空行表示邮件头结束
            if in_header and line.strip() == '':
                in_header = False
                continue
            # 跳过邮件头行
            if in_header:
                continue
            # 跳过引用行（以>开头）
            if line.startswith('>'):
                continue
            body_lines.append(line)

        body = ' '.join(body_lines)
        # 清理多余空白
        body = re.sub(r'\s+', ' ', body)
        return body.strip()

    def classify_sentiment(self, text, category):
        """基于文本内容和类别进行情感分类"""
        text_lower = text.lower()

        # 积极情感关键词（英文）
        positive_keywords = [
            'good', 'great', 'excellent', 'amazing', 'wonderful', 'beautiful',
            'love', 'like', 'enjoy', 'awesome', 'fantastic', 'perfect', 'nice',
            'happy', 'pleased', 'satisfied', 'recommend', 'best', 'brilliant',
            'interesting', 'useful', 'helpful', 'success', 'successful'
        ]

        # 消极情感关键词（英文）
        negative_keywords = [
            'bad', 'terrible', 'awful', 'horrible', 'hate', 'dislike', 'poor',
            'worse', 'worst', 'sucks', 'stupid', 'useless', 'disappointing',
            'frustrating', 'annoying', 'wrong', 'problem', 'issue', 'fail',
            'failure', 'difficult', 'hard', 'waste', 'wasted'
        ]

        # 基于类别的先验知识
        category_bias = {
            'comp.graphics': 0,  # 计算机图形 - 中性偏积极
            'rec.sport.baseball': 1,  # 棒球 - 偏积极
            'sci.space': 0,  # 太空 - 中性
            'talk.politics.mideast': -1,  # 中东政治 - 偏消极
            'alt.atheism': 0,  # 无神论 - 中性
            'soc.religion.christian': 0  # 基督教 - 中性
        }

        pos_score = 0
        neg_score = 0

        # 关键词评分
        for kw in positive_keywords:
            pos_score += text_lower.count(kw) * 2
        for kw in negative_keywords:
            neg_score += text_lower.count(kw) * 2

        # 考虑短语
        if 'not good' in text_lower or 'not great' in text_lower:
            neg_score += 5
        if 'very good' in text_lower or 'very nice' in text_lower:
            pos_score += 3

        # 添加类别偏置
        bias = category_bias.get(category, 0)
        if bias > 0:
            pos_score += 5
        elif bias < 0:
            neg_score += 5

        # 归一化（考虑文本长度）
        length_factor = min(len(text_lower) / 500, 1)
        pos_score = pos_score * length_factor
        neg_score = neg_score * length_factor

        # 确定情感
        if pos_score > neg_score + 2:
            return '正面'
        elif neg_score > pos_score + 2:
            return '负面'
        else:
            return '中性'

    def load_and_process_data(self, sample_size=500):
        """加载并处理数据"""
        # 读取本地数据
        texts, categories, file_info = self.read_local_20newsgroups(sample_size)

        if len(texts) == 0:
            raise ValueError("未读取到任何数据，请检查数据路径")

        print("\n" + "=" * 60)
        print("正在处理数据...")
        print("=" * 60)

        base_date = datetime(2024, 1, 1)
        data_list = []

        for i, (text, category, filepath) in enumerate(zip(texts, categories, file_info)):
            # 情感分类
            sentiment = self.classify_sentiment(text, category)

            # 模拟发布时间（根据文件修改时间或随机生成）
            try:
                file_mtime = os.path.getmtime(filepath)
                publish_time = datetime.fromtimestamp(file_mtime)
            except:
                days_offset = random.randint(0, 90)
                publish_time = base_date + timedelta(days=days_offset)

            # 模拟互动数据
            base_likes = random.randint(0, 500)
            base_comments = random.randint(0, 200)

            if sentiment == '正面':
                likes = int(base_likes * random.uniform(0.8, 1.5))
                comments = int(base_comments * random.uniform(0.5, 1.2))
            elif sentiment == '负面':
                likes = int(base_likes * random.uniform(0.3, 0.8))
                comments = int(base_comments * random.uniform(1.0, 2.0))
            else:
                likes = base_likes
                comments = base_comments

            likes = min(likes, 1000)
            comments = min(comments, 500)

            # 预处理文本
            processed_text, tokens = self.preprocess(text)

            data_list.append({
                'id': i,
                'category': category,
                'file_path': filepath,
                'original_text': text[:500],  # 保留前500字符
                'processed_text': processed_text,
                'sentiment': sentiment,
                'publish_time': publish_time.strftime('%Y-%m-%d %H:%M:%S'),
                'likes': likes,
                'comments': comments,
                'shares': random.randint(0, 100)
            })

        df = pd.DataFrame(data_list)

        print("\n" + "=" * 60)
        print("数据处理完成!")
        print(f"总数据量: {len(df)} 条")
        print(f"\n情感分布:")
        for sentiment, count in df['sentiment'].value_counts().items():
            print(f"  {sentiment}: {count} ({count / len(df) * 100:.1f}%)")
        print(f"\n类别分布:")
        for category, count in df['category'].value_counts().items():
            print(f"  {category}: {count}")
        print("=" * 60)

        return df

    def save_data(self, df, filepath):
        """保存数据到CSV"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        df.to_csv(filepath, index=False, encoding='utf-8-sig')
        print(f"\n数据已保存到: {filepath}")


# 主程序
if __name__ == "__main__":
    # 设置数据路径
    DATA_ROOT = r"D:\网络舆情\qimo\20news-bydate"

    # 创建预处理实例
    preprocessor = DataPreprocessor(data_root=DATA_ROOT)

    # 加载并处理数据（至少500条）
    df = preprocessor.load_and_process_data(sample_size=800)

    # 确保至少有500条
    if len(df) < 500:
        print(f"\n警告: 实际数据量 {len(df)} 条，少于要求的500条")
    else:
        print(f"\n数据量满足要求: {len(df)} 条")

    # 保存数据
    save_path = r"D:\网络舆情\qimo\data\processed_data.csv"
    preprocessor.save_data(df, save_path)

    # 显示数据预览
    print("\n数据预览:")
    print(df.head(10))

    print("\n数据信息:")
    print(df.info())