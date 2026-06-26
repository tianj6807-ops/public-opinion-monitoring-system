# risk/risk_alert.py
import numpy as np
import pandas as pd
from datetime import datetime, timedelta


class RiskAlertSystem:
    """风险预警系统"""

    def __init__(self):
        self.thresholds = {
            'green': 0.2,  # 负面比例低于20%
            'yellow': 0.5,  # 负面比例20%-50%
            'red': 0.5  # 负面比例超过50%
        }

        self.speed_threshold = 1.2  # 传播速度上升阈值
        self.history = []

    def update_history(self, data):
        """更新历史数据"""
        self.history.append(data)
        # 保留最近30天的数据
        if len(self.history) > 30:
            self.history.pop(0)

    def calculate_negative_ratio(self, df, time_window=None):
        """计算负面舆情比例"""
        if time_window:
            # 时间窗口过滤
            cutoff_time = datetime.now() - timedelta(days=time_window)
            df = df[pd.to_datetime(df['publish_time']) >= cutoff_time]

        if len(df) == 0:
            return 0

        negative_count = len(df[df['sentiment'] == '负面'])
        return negative_count / len(df)

    def calculate_propagation_speed(self, df):
        """计算传播速度（单位时间发文量变化）"""
        if len(df) < 10:
            return 1.0

        # 按时间排序
        df_sorted = df.sort_values('publish_time')

        # 分割前后两半
        half = len(df_sorted) // 2
        first_half = df_sorted.iloc[:half]
        second_half = df_sorted.iloc[half:]

        # 计算时间跨度（天）
        first_span = (pd.to_datetime(first_half['publish_time'].max()) -
                      pd.to_datetime(first_half['publish_time'].min())).days or 1
        second_span = (pd.to_datetime(second_half['publish_time'].max()) -
                       pd.to_datetime(second_half['publish_time'].min())).days or 1

        # 计算平均发文密度
        first_density = len(first_half) / first_span
        second_density = len(second_half) / second_span

        speed = second_density / first_density if first_density > 0 else 1.0
        return speed

    def evaluate_risk(self, df):
        """评估风险等级"""
        negative_ratio = self.calculate_negative_ratio(df)
        propagation_speed = self.calculate_propagation_speed(df)

        # 判断风险等级
        if negative_ratio < self.thresholds['green']:
            level = 'green'
            status = '正常'
            color = '#52c41a'
            action = '继续保持日常监测'
        elif negative_ratio < self.thresholds['yellow']:
            level = 'yellow'
            status = '关注'
            color = '#faad14'
            action = '密切关注舆情发展，准备应对预案'
        else:
            if propagation_speed > self.speed_threshold:
                level = 'red'
                status = '预警'
                color = '#f5222d'
                action = '立即启动应急预案，采取应对措施'
            else:
                level = 'yellow'
                status = '关注'
                color = '#faad14'
                action = '负面比例较高，需加强监测'

        return {
            'level': level,
            'status': status,
            'color': color,
            'negative_ratio': round(negative_ratio * 100, 2),
            'propagation_speed': round(propagation_speed, 2),
            'action': action,
            'thresholds': {
                'green': f"<{self.thresholds['green'] * 100}%",
                'yellow': f"{self.thresholds['green'] * 100}%-{self.thresholds['yellow'] * 100}%",
                'red': f">{self.thresholds['yellow'] * 100}%"
            }
        }

    def get_trend_data(self, df, interval_days=7):
        """获取趋势数据用于图表"""
        df['date'] = pd.to_datetime(df['publish_time']).dt.date

        # 按日期分组
        daily_stats = df.groupby('date').agg({
            'sentiment': lambda x: (x == '负面').sum(),
            'id': 'count'
        }).rename(columns={'sentiment': 'negative_count', 'id': 'total_count'})

        daily_stats['negative_ratio'] = daily_stats['negative_count'] / daily_stats['total_count'] * 100

        # 生成完整的时间序列
        date_range = pd.date_range(df['date'].min(), df['date'].max(), freq='D')
        result = []

        for date in date_range:
            date_obj = date.date()
            if date_obj in daily_stats.index:
                row = daily_stats.loc[date_obj]
                result.append({
                    'date': date.strftime('%Y-%m-%d'),
                    'negative_ratio': round(row['negative_ratio'], 2),
                    'total_count': int(row['total_count']),
                    'negative_count': int(row['negative_count'])
                })
            else:
                result.append({
                    'date': date.strftime('%Y-%m-%d'),
                    'negative_ratio': 0,
                    'total_count': 0,
                    'negative_count': 0
                })

        # 计算移动平均
        if len(result) >= 3:
            for i in range(len(result)):
                if i >= 2:
                    avg = (result[i - 2]['negative_ratio'] + result[i - 1]['negative_ratio'] + result[i][
                        'negative_ratio']) / 3
                    result[i]['ma_3'] = round(avg, 2)
                else:
                    result[i]['ma_3'] = result[i]['negative_ratio']

        return result