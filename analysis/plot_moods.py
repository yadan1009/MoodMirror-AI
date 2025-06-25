import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import os
import sys

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.storage import emotion_storage

class MoodAnalyzer:
    """情绪数据分析器"""
    
    def __init__(self, storage=None):
        self.storage = storage or emotion_storage
        
        # 情绪颜色映射
        self.emotion_colors = {
            '快乐': '#FFD700',    # 金色
            '悲伤': '#4169E1',    # 皇家蓝
            '愤怒': '#DC143C',    # 深红色
            '恐惧': '#9370DB',    # 中紫色
            '惊讶': '#FF6347',    # 番茄色
            '厌恶': '#32CD32',    # 酸橙绿
            '中性': '#808080',    # 灰色
        }
        
        # 英文到中文映射
        self.emotion_map = {
            'happy': '快乐',
            'sad': '悲伤',
            'angry': '愤怒',
            'fear': '恐惧',
            'surprise': '惊讶',
            'disgust': '厌恶',
            'neutral': '中性'
        }
    
    def plot_daily_emotion_trend(self, days=7, save_path=None, show_plot=True):
        """
        绘制每日情绪趋势图
        
        Args:
            days: 分析天数
            save_path: 保存路径
            show_plot: 是否显示图表
            
        Returns:
            matplotlib.figure.Figure: 图表对象
        """
        # 获取数据
        trends = self.storage.get_mood_trends(days)
        daily_stats = trends['daily_stats']
        
        if daily_stats.empty:
            print("没有足够的数据进行分析")
            return None
        
        # 创建图表
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # 1. 每日主要情绪
        daily_main = daily_stats.groupby('date').apply(
            lambda x: x.loc[x['count'].idxmax()]
        ).reset_index(drop=True)
        
        dates = pd.to_datetime(daily_main['date'])
        emotions = daily_main['emotion_cn']
        
        # 绘制散点图
        for emotion in emotions.unique():
            mask = emotions == emotion
            ax1.scatter(dates[mask], [emotion] * mask.sum(), 
                       c=self.emotion_colors.get(emotion, '#808080'), 
                       s=100, alpha=0.7, label=emotion)
        
        ax1.set_title(f'最近{days}天每日主要情绪', fontsize=16, fontweight='bold')
        ax1.set_xlabel('日期', fontsize=12)
        ax1.set_ylabel('情绪', fontsize=12)
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax1.grid(True, alpha=0.3)
        
        # 格式化日期轴
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
        ax1.xaxis.set_major_locator(mdates.DayLocator(interval=1))
        plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)
        
        # 2. 每日情绪计数堆叠柱状图
        daily_pivot = daily_stats.pivot_table(
            index='date', 
            columns='emotion_cn', 
            values='count', 
            fill_value=0
        )
        
        # 绘制堆叠柱状图
        daily_pivot.plot(kind='bar', stacked=True, ax=ax2, 
                        color=[self.emotion_colors.get(col, '#808080') 
                              for col in daily_pivot.columns])
        
        ax2.set_title(f'最近{days}天每日情绪记录数量', fontsize=16, fontweight='bold')
        ax2.set_xlabel('日期', fontsize=12)
        ax2.set_ylabel('记录数量', fontsize=12)
        ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax2.grid(True, alpha=0.3)
        
        # 旋转x轴标签
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)
        
        plt.tight_layout()
        
        # 保存图表
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"图表已保存到: {save_path}")
        
        if show_plot:
            plt.show()
        
        return fig
    
    def plot_emotion_distribution(self, days=7, save_path=None, show_plot=True):
        """
        绘制情绪分布饼图
        
        Args:
            days: 分析天数
            save_path: 保存路径
            show_plot: 是否显示图表
            
        Returns:
            matplotlib.figure.Figure: 图表对象
        """
        # 获取情绪分布数据
        trends = self.storage.get_mood_trends(days)
        distribution = trends['emotion_distribution']
        
        if distribution.empty:
            print("没有足够的数据进行分析")
            return None
        
        # 创建图表
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # 1. 饼图
        colors = [self.emotion_colors.get(emotion, '#808080') 
                 for emotion in distribution['emotion_cn']]
        
        wedges, texts, autotexts = ax1.pie(
            distribution['count'], 
            labels=distribution['emotion_cn'],
            colors=colors,
            autopct='%1.1f%%',
            startangle=90,
            textprops={'fontsize': 10}
        )
        
        ax1.set_title(f'最近{days}天情绪分布', fontsize=16, fontweight='bold')
        
        # 2. 柱状图
        bars = ax2.bar(distribution['emotion_cn'], distribution['count'], 
                      color=colors, alpha=0.7)
        
        # 添加数值标签
        for bar in bars:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height)}',
                    ha='center', va='bottom')
        
        ax2.set_title(f'最近{days}天情绪记录次数', fontsize=16, fontweight='bold')
        ax2.set_xlabel('情绪类型', fontsize=12)
        ax2.set_ylabel('记录次数', fontsize=12)
        ax2.grid(True, alpha=0.3)
        
        # 旋转x轴标签
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)
        
        plt.tight_layout()
        
        # 保存图表
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"图表已保存到: {save_path}")
        
        if show_plot:
            plt.show()
        
        return fig
    
    def plot_hourly_pattern(self, days=7, save_path=None, show_plot=True):
        """
        绘制小时级情绪模式热力图
        
        Args:
            days: 分析天数
            save_path: 保存路径
            show_plot: 是否显示图表
            
        Returns:
            matplotlib.figure.Figure: 图表对象
        """
        # 获取小时模式数据
        trends = self.storage.get_mood_trends(days)
        hourly_pattern = trends['hourly_pattern']
        
        if hourly_pattern.empty:
            print("没有足够的数据进行分析")
            return None
        
        # 创建透视表
        hourly_pivot = hourly_pattern.pivot_table(
            index='emotion_cn', 
            columns='hour', 
            values='count', 
            fill_value=0
        )
        
        # 创建图表
        fig, ax = plt.subplots(figsize=(15, 8))
        
        # 绘制热力图
        sns.heatmap(hourly_pivot, annot=True, fmt='d', cmap='YlOrRd', 
                   ax=ax, cbar_kws={'label': '记录次数'})
        
        ax.set_title(f'最近{days}天小时级情绪模式热力图', fontsize=16, fontweight='bold')
        ax.set_xlabel('小时', fontsize=12)
        ax.set_ylabel('情绪类型', fontsize=12)
        
        plt.tight_layout()
        
        # 保存图表
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"图表已保存到: {save_path}")
        
        if show_plot:
            plt.show()
        
        return fig
    
    def create_interactive_dashboard(self, days=7, save_path=None):
        """
        创建交互式仪表板
        
        Args:
            days: 分析天数
            save_path: 保存HTML文件路径
            
        Returns:
            plotly.graph_objects.Figure: 交互式图表
        """
        # 获取数据
        trends = self.storage.get_mood_trends(days)
        
        if trends['emotion_distribution'].empty:
            print("没有足够的数据创建仪表板")
            return None
        
        # 创建子图
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('情绪分布', '每日趋势', '小时模式', '置信度分布'),
            specs=[[{"type": "pie"}, {"type": "scatter"}],
                   [{"type": "heatmap"}, {"type": "histogram"}]]
        )
        
        # 1. 情绪分布饼图
        distribution = trends['emotion_distribution']
        colors = [self.emotion_colors.get(emotion, '#808080') 
                 for emotion in distribution['emotion_cn']]
        
        fig.add_trace(
            go.Pie(labels=distribution['emotion_cn'], 
                  values=distribution['count'],
                  marker_colors=colors,
                  name="情绪分布"),
            row=1, col=1
        )
        
        # 2. 每日趋势线图
        daily_stats = trends['daily_stats']
        if not daily_stats.empty:
            daily_main = daily_stats.groupby('date').apply(
                lambda x: x.loc[x['count'].idxmax()]
            ).reset_index(drop=True)
            
            for emotion in daily_main['emotion_cn'].unique():
                mask = daily_main['emotion_cn'] == emotion
                fig.add_trace(
                    go.Scatter(x=daily_main[mask]['date'], 
                             y=daily_main[mask]['count'],
                             mode='lines+markers',
                             name=emotion,
                             marker_color=self.emotion_colors.get(emotion, '#808080')),
                    row=1, col=2
                )
        
        # 3. 小时模式热力图
        hourly_pattern = trends['hourly_pattern']
        if not hourly_pattern.empty:
            hourly_pivot = hourly_pattern.pivot_table(
                index='emotion_cn', 
                columns='hour', 
                values='count', 
                fill_value=0
            )
            
            fig.add_trace(
                go.Heatmap(z=hourly_pivot.values,
                          x=hourly_pivot.columns,
                          y=hourly_pivot.index,
                          colorscale='YlOrRd',
                          name="小时模式"),
                row=2, col=1
            )
        
        # 4. 置信度分布直方图
        recent_records = trends['recent_records']
        if not recent_records.empty:
            fig.add_trace(
                go.Histogram(x=recent_records['confidence'],
                           nbinsx=20,
                           name="置信度分布",
                           marker_color='lightblue'),
                row=2, col=2
            )
        
        # 更新布局
        fig.update_layout(
            title_text=f"MoodMirror 情绪分析仪表板 (最近{days}天)",
            title_x=0.5,
            showlegend=False,
            height=800
        )
        
        # 保存HTML文件
        if save_path:
            fig.write_html(save_path)
            print(f"交互式仪表板已保存到: {save_path}")
        
        return fig
    
    def generate_mood_report(self, days=7, save_dir='analysis/reports'):
        """
        生成情绪分析报告
        
        Args:
            days: 分析天数
            save_dir: 保存目录
            
        Returns:
            dict: 分析结果摘要
        """
        # 确保保存目录存在
        os.makedirs(save_dir, exist_ok=True)
        
        # 获取数据
        trends = self.storage.get_mood_trends(days)
        
        if trends['emotion_distribution'].empty:
            print("没有足够的数据生成报告")
            return None
        
        # 生成时间戳
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # 1. 每日趋势图
        daily_path = os.path.join(save_dir, f'daily_trend_{timestamp}.png')
        self.plot_daily_emotion_trend(days, daily_path, show_plot=False)
        
        # 2. 情绪分布图
        dist_path = os.path.join(save_dir, f'emotion_distribution_{timestamp}.png')
        self.plot_emotion_distribution(days, dist_path, show_plot=False)
        
        # 3. 小时模式图
        hourly_path = os.path.join(save_dir, f'hourly_pattern_{timestamp}.png')
        self.plot_hourly_pattern(days, hourly_path, show_plot=False)
        
        # 4. 交互式仪表板
        dashboard_path = os.path.join(save_dir, f'dashboard_{timestamp}.html')
        self.create_interactive_dashboard(days, dashboard_path)
        
        # 5. 生成分析摘要
        distribution = trends['emotion_distribution']
        recent_records = trends['recent_records']
        
        summary = {
            'analysis_period': f'{days}天',
            'total_records': len(recent_records),
            'dominant_emotion': distribution.iloc[0]['emotion_cn'] if not distribution.empty else '无',
            'average_confidence': recent_records['confidence'].mean() if not recent_records.empty else 0,
            'generated_files': {
                'daily_trend': daily_path,
                'distribution': dist_path,
                'hourly_pattern': hourly_path,
                'dashboard': dashboard_path
            },
            'timestamp': timestamp
        }
        
        print(f"情绪分析报告已生成:")
        print(f"- 分析周期: {summary['analysis_period']}")
        print(f"- 总记录数: {summary['total_records']}")
        print(f"- 主要情绪: {summary['dominant_emotion']}")
        print(f"- 平均置信度: {summary['average_confidence']:.2f}")
        print(f"- 报告文件保存在: {save_dir}")
        
        return summary

# 创建全局分析器实例
mood_analyzer = MoodAnalyzer()

def demo_analysis():
    """演示分析功能"""
    analyzer = MoodAnalyzer()
    
    # 检查是否有数据
    stats = emotion_storage.get_database_stats()
    print(f"数据库统计: {stats}")
    
    if stats['total_records'] > 0:
        # 生成分析报告
        summary = analyzer.generate_mood_report(days=7)
        print(f"分析完成: {summary}")
    else:
        print("数据库中没有数据，请先添加一些情绪记录")
        
        # 添加一些示例数据用于演示
        import numpy as np
        from datetime import datetime, timedelta
        
        emotions = ['happy', 'sad', 'angry', 'neutral', 'surprise']
        emotions_cn = ['快乐', '悲伤', '愤怒', '中性', '惊讶']
        
        print("添加示例数据...")
        for i in range(20):
            idx = np.random.randint(0, len(emotions))
            confidence = np.random.uniform(0.6, 0.95)
            probs = np.random.rand(7)
            
            emotion_storage.add_emotion_record(
                emotion=emotions[idx],
                emotion_cn=emotions_cn[idx],
                confidence=confidence,
                all_probabilities=probs
            )
        
        print("示例数据添加完成，重新生成报告...")
        summary = analyzer.generate_mood_report(days=7)
        print(f"分析完成: {summary}")

if __name__ == '__main__':
    demo_analysis() 