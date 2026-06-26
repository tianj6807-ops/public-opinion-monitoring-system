# balance_data.py - 数据平衡脚本
import pandas as pd
import numpy as np
from sklearn.utils import resample
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import os
import random
from datetime import datetime, timedelta


def balance_dataset():
    """平衡数据集"""
    print("=" * 60)
    print("数据平衡处理")
    print("=" * 60)

    # 加载原始数据
    data_path = r'D:\网络舆情\qimo\data\processed_data.csv'
    df = pd.read_csv(data_path)

    print(f"\n原始数据量: {len(df)} 条")
    print("原始情感分布:")
    for sentiment, count in df['sentiment'].value_counts().items():
        print(f"  {sentiment}: {count} ({count / len(df) * 100:.1f}%)")

    # 分离各类别
    df_positive = df[df['sentiment'] == '正面']
    df_neutral = df[df['sentiment'] == '中性']
    df_negative = df[df['sentiment'] == '负面']

    pos_count = len(df_positive)
    neu_count = len(df_neutral)
    neg_count = len(df_negative)

    print(f"\n各类别数量: 正面={pos_count}, 中性={neu_count}, 负面={neg_count}")

    # 策略1: 下采样中性类，上采样正面和负面类
    # 目标：让三类数量相等（使用最小值或中位数）
    target_count = min(max(pos_count, neg_count) * 2, 300)  # 目标每类300条，不超过原始数据太多

    print(f"\n目标每类数量: {target_count} 条")

    # 处理中性类（下采样）
    if neu_count > target_count:
        df_neutral_balanced = resample(
            df_neutral,
            replace=False,
            n_samples=target_count,
            random_state=42
        )
        print(f"  中性类: {neu_count} → {len(df_neutral_balanced)} (下采样)")
    else:
        df_neutral_balanced = df_neutral
        print(f"  中性类: {neu_count} → {neu_count} (保持不变)")

    # 处理正面类（上采样）
    if pos_count < target_count:
        df_positive_balanced = resample(
            df_positive,
            replace=True,
            n_samples=target_count,
            random_state=42
        )
        print(f"  正面类: {pos_count} → {len(df_positive_balanced)} (上采样)")
    else:
        df_positive_balanced = df_positive[:target_count]
        print(f"  正面类: {pos_count} → {len(df_positive_balanced)} (下采样)")

    # 处理负面类（上采样）
    if neg_count < target_count:
        df_negative_balanced = resample(
            df_negative,
            replace=True,
            n_samples=target_count,
            random_state=42
        )
        print(f"  负面类: {neg_count} → {len(df_negative_balanced)} (上采样)")
    else:
        df_negative_balanced = df_negative[:target_count]
        print(f"  负面类: {neg_count} → {len(df_negative_balanced)} (下采样)")

    # 合并平衡后的数据
    df_balanced = pd.concat([df_positive_balanced, df_neutral_balanced, df_negative_balanced])

    # 打乱顺序
    df_balanced = df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)

    print(f"\n平衡后数据量: {len(df_balanced)} 条")
    print("平衡后情感分布:")
    for sentiment, count in df_balanced['sentiment'].value_counts().items():
        print(f"  {sentiment}: {count} ({count / len(df_balanced) * 100:.1f}%)")

    # 保存平衡后的数据
    balanced_path = r'D:\网络舆情\qimo\data\balanced_data.csv'
    df_balanced.to_csv(balanced_path, index=False, encoding='utf-8-sig')
    print(f"\n✅ 平衡数据已保存到: {balanced_path}")

    return df_balanced


def augment_text_data(df):
    """文本数据增强（可选，用于增加正面和负面样本）"""
    print("\n" + "=" * 60)
    print("文本数据增强处理")
    print("=" * 60)

    # 简单的中文同义词替换
    synonym_dict = {
        '好': ['棒', '赞', '优秀'],
        '坏': ['差', '烂', '糟糕'],
        '喜欢': ['喜爱', '钟爱', '欣赏'],
        '讨厌': ['厌恶', '反感', '排斥'],
        '很好': ['非常好', '极好', '超好'],
        '太差': ['太烂', '太糟', '很差'],
        '推荐': ['建议', '力荐', '安利'],
        '失望': ['失落', '沮丧', '灰心']
    }

    def synonym_replacement(text, n=1):
        """同义词替换增强"""
        words = list(text)
        for old, news in synonym_dict.items():
            if old in text and len(news) > 0:
                new_word = random.choice(news)
                text = text.replace(old, new_word)
                n -= 1
                if n <= 0:
                    break
        return text

    # 为正面和负面类生成增强样本
    augmented_data = []

    for sentiment in ['正面', '负面']:
        df_class = df[df['sentiment'] == sentiment]
        current_count = len(df_class)
        target_count = 300  # 目标数量

        if current_count < target_count:
            need_count = target_count - current_count
            print(f"  为 {sentiment} 类生成 {need_count} 条增强样本")

            # 随机选择要增强的样本
            samples_to_augment = df_class.sample(n=min(need_count, current_count),
                                                 replace=True,
                                                 random_state=42)

            for idx, row in samples_to_augment.iterrows():
                new_row = row.copy()
                # 增强文本
                original_text = new_row['original_text']
                new_text = synonym_replacement(original_text)
                new_row['original_text'] = new_text

                # 重新预处理
                from data.process_data import DataPreprocessor
                preprocessor = DataPreprocessor()
                processed_text, _ = preprocessor.preprocess(new_text)
                new_row['processed_text'] = processed_text

                # 修改ID和时间戳
                new_row['id'] = len(df) + len(augmented_data) + 1

                # 随机调整互动数据
                new_row['likes'] = int(row['likes'] * random.uniform(0.8, 1.2))
                new_row['comments'] = int(row['comments'] * random.uniform(0.8, 1.2))

                augmented_data.append(new_row)

                if len(augmented_data) >= need_count:
                    break

    if augmented_data:
        df_augmented = pd.DataFrame(augmented_data)
        df_balanced = pd.concat([df, df_augmented], ignore_index=True)
        print(f"✅ 增强后总数据量: {len(df_balanced)} 条")
        return df_balanced

    return df


if __name__ == "__main__":
    # 执行数据平衡
    df_balanced = balance_dataset()

    # 可选：文本增强
    # df_balanced = augment_text_data(df_balanced)

    print("\n" + "=" * 60)
    print("数据平衡完成！")
    print("=" * 60)