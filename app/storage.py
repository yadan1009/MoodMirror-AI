import sqlite3
import json
import pandas as pd
from datetime import datetime, timedelta
import os

class EmotionStorage:
    """情绪数据存储类"""
    
    def __init__(self, db_path='data/emotion_diary.db'):
        self.db_path = db_path
        self.ensure_db_directory()
        self.init_database()
    
    def ensure_db_directory(self):
        """确保数据库目录存在"""
        db_dir = os.path.dirname(self.db_path)
        if db_dir and not os.path.exists(db_dir):
            os.makedirs(db_dir)
    
    def init_database(self):
        """初始化数据库和表"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # 创建情绪记录表
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS emotion_records (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    emotion TEXT NOT NULL,
                    emotion_cn TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    all_probabilities TEXT,
                    date TEXT NOT NULL,
                    hour INTEGER NOT NULL,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # 创建索引以提高查询性能
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_date ON emotion_records(date)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_timestamp ON emotion_records(timestamp)')
            
            conn.commit()
    
    def add_emotion_record(self, emotion, emotion_cn, confidence, all_probabilities=None):
        """
        添加一条情绪记录
        
        Args:
            emotion: 英文情绪标签
            emotion_cn: 中文情绪标签
            confidence: 置信度
            all_probabilities: 所有类别的概率
        
        Returns:
            int: 记录ID
        """
        now = datetime.now()
        timestamp = now.strftime('%Y-%m-%d %H:%M:%S')
        date = now.strftime('%Y-%m-%d')
        hour = now.hour
        
        # 将概率数组转换为JSON字符串
        if all_probabilities is not None:
            if hasattr(all_probabilities, 'tolist'):
                prob_json = json.dumps(all_probabilities.tolist())
            elif isinstance(all_probabilities, (list, tuple)):
                prob_json = json.dumps(list(all_probabilities))
            else:
                prob_json = json.dumps(str(all_probabilities))
        else:
            prob_json = None
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO emotion_records 
                (timestamp, emotion, emotion_cn, confidence, all_probabilities, date, hour)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (timestamp, emotion, emotion_cn, confidence, prob_json, date, hour))
            
            record_id = cursor.lastrowid
            conn.commit()
            
        print(f"成功添加情绪记录: {emotion_cn} (置信度: {confidence:.2f})")
        return record_id
    
    def get_recent_records(self, days=7):
        """
        获取最近几天的情绪记录
        
        Args:
            days: 天数，默认7天
            
        Returns:
            pandas.DataFrame: 情绪记录数据框
        """
        start_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
        
        with sqlite3.connect(self.db_path) as conn:
            query = '''
                SELECT * FROM emotion_records 
                WHERE date >= ? 
                ORDER BY timestamp DESC
            '''
            df = pd.read_sql_query(query, conn, params=(start_date,))
            
        return df
    
    def get_daily_emotion_stats(self, days=7):
        """
        获取每日情绪统计
        
        Args:
            days: 天数，默认7天
            
        Returns:
            pandas.DataFrame: 每日情绪统计
        """
        start_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
        
        with sqlite3.connect(self.db_path) as conn:
            query = '''
                SELECT 
                    date,
                    emotion_cn,
                    COUNT(*) as count,
                    AVG(confidence) as avg_confidence
                FROM emotion_records 
                WHERE date >= ? 
                GROUP BY date, emotion_cn
                ORDER BY date DESC, count DESC
            '''
            df = pd.read_sql_query(query, conn, params=(start_date,))
            
        return df
    
    def get_hourly_emotion_pattern(self, days=7):
        """
        获取小时级情绪模式
        
        Args:
            days: 天数，默认7天
            
        Returns:
            pandas.DataFrame: 小时级情绪统计
        """
        start_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
        
        with sqlite3.connect(self.db_path) as conn:
            query = '''
                SELECT 
                    hour,
                    emotion_cn,
                    COUNT(*) as count,
                    AVG(confidence) as avg_confidence
                FROM emotion_records 
                WHERE date >= ? 
                GROUP BY hour, emotion_cn
                ORDER BY hour, count DESC
            '''
            df = pd.read_sql_query(query, conn, params=(start_date,))
            
        return df
    
    def get_emotion_distribution(self, days=7):
        """
        获取情绪分布统计
        
        Args:
            days: 天数，默认7天
            
        Returns:
            pandas.DataFrame: 情绪分布统计
        """
        start_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
        
        with sqlite3.connect(self.db_path) as conn:
            query = '''
                SELECT 
                    emotion_cn,
                    COUNT(*) as count,
                    AVG(confidence) as avg_confidence,
                    COUNT(*) * 100.0 / (SELECT COUNT(*) FROM emotion_records WHERE date >= ?) as percentage
                FROM emotion_records 
                WHERE date >= ? 
                GROUP BY emotion_cn
                ORDER BY count DESC
            '''
            df = pd.read_sql_query(query, conn, params=(start_date, start_date))
            
        return df
    
    def get_mood_trends(self, days=7):
        """
        获取情绪趋势数据
        
        Args:
            days: 天数，默认7天
            
        Returns:
            dict: 包含各种趋势数据的字典
        """
        # 获取每日主要情绪
        daily_stats = self.get_daily_emotion_stats(days)
        
        # 计算每日主要情绪
        daily_main_emotion = daily_stats.groupby('date').first().reset_index()
        
        # 获取情绪分布
        emotion_distribution = self.get_emotion_distribution(days)
        
        # 获取小时模式
        hourly_pattern = self.get_hourly_emotion_pattern(days)
        
        # 获取原始记录
        recent_records = self.get_recent_records(days)
        
        return {
            'daily_main_emotion': daily_main_emotion,
            'emotion_distribution': emotion_distribution,
            'hourly_pattern': hourly_pattern,
            'recent_records': recent_records,
            'daily_stats': daily_stats
        }
    
    def clear_old_records(self, keep_days=30):
        """
        清理旧记录，只保留最近指定天数的数据
        
        Args:
            keep_days: 保留天数，默认30天
        """
        cutoff_date = (datetime.now() - timedelta(days=keep_days)).strftime('%Y-%m-%d')
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('DELETE FROM emotion_records WHERE date < ?', (cutoff_date,))
            deleted_count = cursor.rowcount
            conn.commit()
            
        print(f"清理了 {deleted_count} 条旧记录")
        return deleted_count
    
    def export_to_csv(self, filepath, days=30):
        """
        导出数据到CSV文件
        
        Args:
            filepath: 导出文件路径
            days: 导出天数，默认30天
        """
        df = self.get_recent_records(days)
        df.to_csv(filepath, index=False, encoding='utf-8-sig')
        print(f"数据已导出到: {filepath}")
    
    def get_database_stats(self):
        """
        获取数据库统计信息
        
        Returns:
            dict: 数据库统计信息
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # 总记录数
            cursor.execute('SELECT COUNT(*) FROM emotion_records')
            total_records = cursor.fetchone()[0]
            
            # 最早记录
            cursor.execute('SELECT MIN(timestamp) FROM emotion_records')
            earliest_record = cursor.fetchone()[0]
            
            # 最新记录
            cursor.execute('SELECT MAX(timestamp) FROM emotion_records')
            latest_record = cursor.fetchone()[0]
            
            # 不同日期数
            cursor.execute('SELECT COUNT(DISTINCT date) FROM emotion_records')
            unique_dates = cursor.fetchone()[0]
            
        return {
            'total_records': total_records,
            'earliest_record': earliest_record,
            'latest_record': latest_record,
            'unique_dates': unique_dates,
            'database_path': self.db_path
        }

    def delete_record_by_id(self, record_id):
        """
        通过ID删除单条记录
        
        Args:
            record_id: 记录ID
            
        Returns:
            bool: 删除是否成功
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('DELETE FROM emotion_records WHERE id = ?', (record_id,))
            deleted_count = cursor.rowcount
            conn.commit()
            
        success = deleted_count > 0
        if success:
            print(f"成功删除记录 ID: {record_id}")
        else:
            print(f"未找到记录 ID: {record_id}")
            
        return success

    def delete_records_by_ids(self, record_ids):
        """
        批量删除多条记录
        
        Args:
            record_ids: 记录ID列表
            
        Returns:
            int: 删除的记录数
        """
        if not record_ids:
            return 0
            
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            placeholders = ','.join('?' * len(record_ids))
            cursor.execute(f'DELETE FROM emotion_records WHERE id IN ({placeholders})', record_ids)
            deleted_count = cursor.rowcount
            conn.commit()
            
        print(f"成功删除 {deleted_count} 条记录")
        return deleted_count

    def get_records_with_ids(self, days=7):
        """
        获取带有ID的最近记录，用于删除操作
        
        Args:
            days: 天数，默认7天
            
        Returns:
            pandas.DataFrame: 包含ID的情绪记录数据框
        """
        start_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
        
        with sqlite3.connect(self.db_path) as conn:
            query = '''
                SELECT id, timestamp, emotion, emotion_cn, confidence, date
                FROM emotion_records 
                WHERE date >= ? 
                ORDER BY timestamp DESC
            '''
            df = pd.read_sql_query(query, conn, params=(start_date,))
            
        return df

    def get_record_by_id(self, record_id):
        """
        通过ID获取单条记录详情
        
        Args:
            record_id: 记录ID
            
        Returns:
            dict: 记录详情，如果不存在返回None
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT * FROM emotion_records WHERE id = ?', (record_id,))
            row = cursor.fetchone()
            
            if row:
                columns = [description[0] for description in cursor.description]
                return dict(zip(columns, row))
            return None

# 创建全局存储实例
emotion_storage = EmotionStorage()

def test_storage():
    """测试存储功能"""
    storage = EmotionStorage('data/test_emotion_diary.db')
    
    # 添加测试数据
    import numpy as np
    test_probs = np.random.rand(7)
    
    storage.add_emotion_record('happy', '快乐', 0.85, test_probs)
    storage.add_emotion_record('sad', '悲伤', 0.72, test_probs)
    storage.add_emotion_record('neutral', '中性', 0.68, test_probs)
    
    # 获取统计信息
    stats = storage.get_database_stats()
    print(f"数据库统计: {stats}")
    
    # 获取趋势数据
    trends = storage.get_mood_trends()
    print(f"情绪分布:\n{trends['emotion_distribution']}")

if __name__ == '__main__':
    test_storage() 