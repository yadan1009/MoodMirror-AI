import gradio as gr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import os
import sys
from datetime import datetime, timedelta
import plotly.graph_objects as go

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.predict import emotion_predictor
from app.storage import emotion_storage
from analysis.plot_moods import mood_analyzer
from app.emotion_ai import emotion_ai

# 设置matplotlib后端
plt.switch_backend('Agg')

class MoodMirrorApp:
    """MoodMirror主应用程序"""
    
    def __init__(self):
        self.predictor = emotion_predictor
        self.storage = emotion_storage
        self.analyzer = mood_analyzer
        
        # 情绪emoji映射
        self.emotion_emojis = {
            'happy': '😄',
            'sad': '😢',
            'angry': '😠',
            'fear': '😨',
            'surprise': '😲',
            'disgust': '🤢',
            'neutral': '😐'
        }
    
    def predict_from_image(self, image, enable_ai_explanation=True):
        """
        从上传的图像预测情绪（不保存到数据库）
        
        Args:
            image: PIL Image对象
            enable_ai_explanation: 是否启用AI解释
            
        Returns:
            tuple: (结果文本, 情绪分布图, AI解释文本, 预测结果数据)
        """
        if image is None:
            return "请上传一张图片", None, "", None
        
        try:
            # 预测情绪，但不保存到数据库
            result = self.predictor.predict_emotion_from_image(image, save_to_db=False)
            
            if 'error' in result:
                return f"预测失败: {result['error']}", None, "", None
            
            # 构造结果文本
            emoji = self.emotion_emojis.get(result['emotion'], '❓')
            
            # 检查是否为演示模式
            demo_note = ""
            if result.get('demo_mode', False):
                demo_note = "\n🎭 **演示模式**: 当前使用模拟AI预测，训练完成后将获得真实结果"
            
            result_text = f"""🎯 **情绪识别结果**

{emoji} **检测到的情绪**: {result['emotion_cn']}
📊 **置信度**: {result['confidence']:.2f} ({result['confidence']*100:.1f}%)
👥 **检测到的人脸数**: {result.get('faces_detected', 0)}

⚠️ **请确认结果准确性，点击下方按钮保存到情绪日记**{demo_note}"""
            
            # 创建情绪概率分布图
            prob_chart = self.create_probability_chart(result['all_probabilities'])
            
            # 获取AI解释
            ai_explanation_text = ""
            if enable_ai_explanation:
                try:
                    ai_result = emotion_ai.get_emotion_explanation(
                        probabilities=result['all_probabilities'],
                        confidence_threshold=0.4
                    )
                    
                    if ai_result['success']:
                        selected_emotions = ai_result['selected_emotions']
                        selected_text = "、".join([f"{e['emotion_cn']}({e['confidence']:.1%})" for e in selected_emotions])
                        
                        ai_explanation_text = f"""🤖 AI情绪解释师 (Qwen-Plus)

📋 分析的情绪: {selected_text}

💬 温柔指导:

{ai_result['ai_explanation']}


💡 这是专为自闭症和社交障碍人群设计的情绪理解辅助工具"""
                    else:
                        # 使用备用解释
                        selected_emotions = ai_result['selected_emotions']
                        ai_explanation_text = f"""🤖 情绪理解助手 (备用模式)

📋 检测到的情绪: {selected_emotions[0]['emotion_cn']}

💬 温柔指导:

{ai_result['ai_explanation']}


⚠️ *AI服务暂时不可用，使用备用解释模式*"""
                        
                except Exception as e:
                    ai_explanation_text = f"""🤖 **AI解释服务**

⚠️ 暂时无法获取AI解释: {str(e)}

💡 您可以尝试关闭AI解释功能，或稍后重试"""
            
            return result_text, prob_chart, ai_explanation_text, result
            
        except Exception as e:
            return f"处理图像时出错: {str(e)}", None, "", None
    
    def create_probability_chart(self, probabilities):
        """
        创建情绪概率分布图
        
        Args:
            probabilities: 概率数组
            
        Returns:
            matplotlib.figure.Figure: 图表对象
        """
        emotions_cn = ['愤怒', '厌恶', '恐惧', '快乐', '中性', '悲伤', '惊讶']
        colors = ['#DC143C', '#32CD32', '#9370DB', '#FFD700', '#808080', '#4169E1', '#FF6347']
        
        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.bar(emotions_cn, probabilities, color=colors, alpha=0.7)
        
        # 添加数值标签
        for bar, prob in zip(bars, probabilities):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{prob:.3f}', ha='center', va='bottom', fontsize=10)
        
        ax.set_title('各情绪类别的预测概率', fontsize=14, fontweight='bold')
        ax.set_xlabel('情绪类别', fontsize=12)
        ax.set_ylabel('概率', fontsize=12)
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3)
        
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        return fig
    
    def get_recent_records(self, days=7):
        """
        获取最近的情绪记录
        
        Args:
            days: 天数
            
        Returns:
            pandas.DataFrame: 最近的记录
        """
        records = self.storage.get_recent_records(days)
        
        if records.empty:
            return pd.DataFrame(columns=['时间', '情绪', '置信度'])
        
        # 格式化显示
        display_records = pd.DataFrame({
            '时间': records['timestamp'],
            '情绪': records['emotion_cn'],
            '置信度': records['confidence'].apply(lambda x: f"{x:.2f}")
        })
        
        return display_records.head(20)  # 只显示最近20条
    
    def get_recent_records_with_delete(self, days=7):
        """
        获取最近的情绪记录（包含ID用于删除）
        
        Args:
            days: 天数
            
        Returns:
            pandas.DataFrame: 最近的记录（包含删除选项）
        """
        records = self.storage.get_records_with_ids(days)
        
        if records.empty:
            return pd.DataFrame(columns=['ID', '时间', '情绪', '置信度', '操作'])
        
        # 格式化显示
        display_records = pd.DataFrame({
            'ID': records['id'],
            '时间': records['timestamp'],
            '情绪': records['emotion_cn'],
            '置信度': records['confidence'].apply(lambda x: f"{x:.2f}"),
        })
        
        return display_records.head(20)  # 只显示最近20条

    def delete_emotion_record(self, record_id):
        """
        删除情绪记录
        
        Args:
            record_id: 记录ID
            
        Returns:
            str: 删除状态消息
        """
        if not record_id:
            return "❌ 请提供有效的记录ID"
        
        try:
            record_id = int(record_id)
            # 先获取记录详情
            record = self.storage.get_record_by_id(record_id)
            
            if not record:
                return f"❌ 未找到ID为 {record_id} 的记录"
            
            # 删除记录
            success = self.storage.delete_record_by_id(record_id)
            
            if success:
                return f"""✅ **记录删除成功**

🗑️ **已删除记录ID**: {record_id}
🎯 **情绪**: {record['emotion_cn']}
📊 **置信度**: {record['confidence']:.2f}
⏰ **原记录时间**: {record['timestamp']}

💡 删除操作不可撤销，请谨慎操作"""
            else:
                return f"❌ 删除记录失败，请重试"
                
        except ValueError:
            return "❌ 记录ID必须是数字"
        except Exception as e:
            return f"❌ 删除失败: {str(e)}"

    def batch_delete_records(self, selected_ids_text):
        """
        批量删除记录
        
        Args:
            selected_ids_text: 用逗号分隔的ID字符串
            
        Returns:
            str: 删除状态消息
        """
        if not selected_ids_text or not selected_ids_text.strip():
            return "❌ 请输入要删除的记录ID（用逗号分隔）"
        
        try:
            # 解析ID列表
            id_strings = [s.strip() for s in selected_ids_text.split(',')]
            record_ids = [int(id_str) for id_str in id_strings if id_str]
            
            if not record_ids:
                return "❌ 未找到有效的记录ID"
            
            # 执行批量删除
            deleted_count = self.storage.delete_records_by_ids(record_ids)
            
            if deleted_count > 0:
                return f"""✅ **批量删除成功**

🗑️ **删除记录数**: {deleted_count}
📝 **删除的ID**: {', '.join(map(str, record_ids))}

💡 删除操作不可撤销，请谨慎操作"""
            else:
                return "❌ 没有找到任何匹配的记录"
                
        except ValueError:
            return "❌ ID格式错误，请输入数字，用逗号分隔（例如：1,2,3）"
        except Exception as e:
            return f"❌ 批量删除失败: {str(e)}"

    def quick_delete_recent_records(self, count):
        """
        快速删除最近N条记录
        
        Args:
            count: 要删除的记录数量
            
        Returns:
            str: 删除状态消息
        """
        if count <= 0:
            return "❌ 删除数量必须大于0"
        
        try:
            # 获取最近的记录ID
            records = self.storage.get_records_with_ids(30)  # 获取最近30天的记录
            
            if records.empty:
                return "❌ 没有找到任何记录"
            
            if len(records) < count:
                return f"❌ 只有{len(records)}条记录，无法删除{count}条"
            
            # 获取最近N条记录的ID
            recent_ids = records.head(count)['id'].tolist()
            
            # 执行删除
            deleted_count = self.storage.delete_records_by_ids(recent_ids)
            
            if deleted_count > 0:
                return f"""✅ **快速删除成功**

🗑️ **删除记录数**: {deleted_count}
📝 **删除的ID**: {', '.join(map(str, recent_ids))}

💡 已删除最近{count}条记录"""
            else:
                return "❌ 删除失败，请重试"
                
        except Exception as e:
            return f"❌ 删除失败: {str(e)}"

    def delete_records_by_confidence(self, max_confidence, days=7):
        """
        删除指定置信度以下的记录
        
        Args:
            max_confidence: 最大置信度阈值（删除小于等于此值的记录）
            days: 查找范围天数
            
        Returns:
            str: 删除状态消息
        """
        if max_confidence is None:
            return "❌ 请设置置信度阈值"
        
        try:
            # 获取指定置信度以下的记录
            start_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
            
            import sqlite3
            with sqlite3.connect(self.storage.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    'SELECT id, emotion_cn, confidence FROM emotion_records WHERE confidence <= ? AND date >= ?', 
                    (max_confidence, start_date)
                )
                records = cursor.fetchall()
                record_ids = [row[0] for row in records]
            
            if not record_ids:
                return f"❌ 最近{days}天内没有找到置信度≤{max_confidence:.2f}的记录"
            
            # 统计各情绪的数量
            emotion_counts = {}
            for record in records:
                emotion = record[1]
                emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
            
            # 执行删除
            deleted_count = self.storage.delete_records_by_ids(record_ids)
            
            if deleted_count > 0:
                emotion_summary = "、".join([f"{emotion}({count}条)" for emotion, count in emotion_counts.items()])
                
                return f"""✅ **按置信度删除成功**

📊 **置信度阈值**: ≤{max_confidence:.2f}
📅 **查找范围**: 最近{days}天
🗑️ **删除记录数**: {deleted_count}
😊 **删除明细**: {emotion_summary}
📝 **删除的ID**: {', '.join(map(str, record_ids))}

💡 低置信度记录已清理，提高数据质量"""
            else:
                return "❌ 删除失败，请重试"
                
        except Exception as e:
            return f"❌ 删除失败: {str(e)}"

    def create_trend_analysis(self, days=7):
        """
        创建趋势分析图表
        
        Args:
            days: 分析天数
            
        Returns:
            tuple: (分析文本, 趋势图, 分布图)
        """
        try:
            # 获取数据库统计
            stats = self.storage.get_database_stats()
            
            if stats['total_records'] == 0:
                return "📊 暂无数据可分析，请先添加一些情绪记录", None, None
            
            # 获取趋势数据
            trends = self.storage.get_mood_trends(days)
            
            if trends['emotion_distribution'].empty:
                return f"📊 最近{days}天内没有情绪记录", None, None
            
            # 生成分析文本
            distribution = trends['emotion_distribution']
            recent_records = trends['recent_records']
            
            analysis_text = f"""📊 **最近{days}天情绪分析报告**

📈 **总记录数**: {len(recent_records)}
🎯 **主要情绪**: {distribution.iloc[0]['emotion_cn']} ({distribution.iloc[0]['percentage']:.1f}%)
📊 **平均置信度**: {recent_records['confidence'].mean():.2f}
📅 **分析周期**: {days}天

🔍 **情绪分布**:"""
            
            for _, row in distribution.iterrows():
                analysis_text += f"\n　{row['emotion_cn']}: {row['count']}次 ({row['percentage']:.1f}%)"
            
            # 创建趋势图
            trend_fig = self.analyzer.plot_daily_emotion_trend(days, show_plot=False)
            
            # 创建分布图
            dist_fig = self.analyzer.plot_emotion_distribution(days, show_plot=False)
            
            return analysis_text, trend_fig, dist_fig
            
        except Exception as e:
            return f"生成分析时出错: {str(e)}", None, None

    def generate_emotion_diary(self, days=7):
        """
        生成AI情绪日记
        
        Args:
            days: 分析天数
            
        Returns:
            str: 生成的情绪日记
        """
        try:
            # 获取趋势数据
            trends = self.storage.get_mood_trends(days)
            
            if trends['emotion_distribution'].empty:
                return f"📖 最近{days}天内没有足够的情绪记录来生成日记"
            
            # 准备数据给AI
            emotion_data = {
                'total_records': len(trends['recent_records']),
                'dominant_emotion': trends['emotion_distribution'].iloc[0]['emotion_cn'],
                'average_confidence': trends['recent_records']['confidence'].mean(),
                'emotion_distribution': trends['emotion_distribution']
            }
            
            # 调用AI生成日记
            result = emotion_ai.generate_emotion_diary(emotion_data, days)
            
            if result['success']:
                diary_display = f"""📖 **AI情绪日记** ({result['analysis_period']})

{result['diary_content']}

---
🤖 *由阿里百炼Qwen-Plus生成，专为特殊人群优化*
📊 *基于{result['data_summary']['total_records']}条记录分析*"""
            else:
                diary_display = f"""📖 **情绪日记** ({result['analysis_period']})

{result['diary_content']}

---
💙 *温柔的本地生成内容*"""
            
            return diary_display
            
        except Exception as e:
            return f"生成情绪日记时出错: {str(e)}"

    def get_personalized_advice(self, image, context=""):
        """
        获取个性化情绪建议
        
        Args:
            image: 图像
            context: 额外上下文
            
        Returns:
            str: 个性化建议
        """
        if image is None:
            return "请先上传图片进行情绪识别"
        
        try:
            # 先识别情绪
            result = self.predictor.predict_emotion_from_image(image, save_to_db=False)
            
            if 'error' in result:
                return f"识别情绪失败: {result['error']}"
            
            # 获取AI建议
            advice_result = emotion_ai.generate_personalized_advice(
                current_emotion=result['emotion'],
                confidence=result['confidence'],
                context=context
            )
            
            if advice_result['success']:
                advice_display = f"""🎯 **个性化情绪建议**

📋 **当前情绪**: {advice_result['emotion_cn']} (置信度: {advice_result['confidence']:.1%})

💡 **专业建议**:

{advice_result['advice_content']}

---
🤖 *由阿里百炼Qwen-Plus生成专业建议*"""
            else:
                advice_display = f"""🎯 **情绪管理建议**

📋 **当前情绪**: {advice_result['emotion_cn']} (置信度: {advice_result['confidence']:.1%})

💡 **温和建议**:

{advice_result['advice_content']}

---
💙 *贴心的本地建议*"""
            
            return advice_display
            
        except Exception as e:
            return f"生成建议时出错: {str(e)}"
    
    def export_data(self, days=30):
        """
        导出数据到CSV
        
        Args:
            days: 导出天数
            
        Returns:
            tuple: (导出路径, 状态信息)
        """
        try:
            # 确保导出目录存在
            export_dir = 'data/exports'
            os.makedirs(export_dir, exist_ok=True)
            
            # 生成文件名
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f'emotion_records_{timestamp}.csv'
            filepath = os.path.join(export_dir, filename)
            
            # 导出数据
            self.storage.export_to_csv(filepath, days)
            
            status = f"✅ 数据导出成功!\n📁 文件路径: {filepath}\n📊 导出了最近{days}天的记录"
            
            return filepath, status
            
        except Exception as e:
            return None, f"❌ 导出失败: {str(e)}"
    
    def confirm_and_save_emotion(self, prediction_result):
        """
        确认并保存情绪记录
        
        Args:
            prediction_result: 预测结果数据
            
        Returns:
            str: 保存状态消息
        """
        if prediction_result is None:
            return "❌ 没有待保存的预测结果，请先进行情绪识别"
        
        try:
            # 保存到数据库
            record_id = self.storage.add_emotion_record(
                emotion=prediction_result['emotion'],
                emotion_cn=prediction_result['emotion_cn'],
                confidence=prediction_result['confidence'],
                all_probabilities=prediction_result['all_probabilities']
            )
            
            return f"""✅ **记录已成功保存**

📝 **记录ID**: {record_id}
🎯 **情绪**: {prediction_result['emotion_cn']}
📊 **置信度**: {prediction_result['confidence']:.2f}
⏰ **保存时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

💡 您可以在"记录查看"页面管理所有情绪记录"""
            
        except Exception as e:
            return f"❌ 保存失败: {str(e)}"
    
    def create_interface(self):
        """
        创建Gradio界面
        
        Returns:
            gr.Blocks: Gradio应用程序
        """
        # 检查是否为演示模式
        demo_mode_note = ""
        if not self.predictor.is_loaded:
            demo_mode_note = """
                
                🎭 **当前运行在演示模式**
                
                > 使用模拟AI预测展示功能，训练完成后将获得真实的情绪识别结果
                """
        
        with gr.Blocks(title="MoodMirror - AI情绪日记", theme=gr.themes.Soft()) as demo:
            gr.Markdown(
                f"""
                # 🎭 MoodMirror: AI情绪日记 2.0
                
                ### 基于深度学习的情绪识别与AI解释系统
                ### 🌟 专为自闭症等特殊人群设计的情绪理解辅助工具
                
                📸 **上传照片** → 🧠 **AI识别** → 🤖 **温柔解释** → 📊 **趋势分析**
                
                💫 **全新AI解释功能**: 接入阿里百炼大模型，提供温柔、专业的情绪指导{demo_mode_note}
                """
            )
            
            with gr.Tabs():
                # Tab 1: 情绪识别
                with gr.TabItem("📸 情绪识别"):
                    gr.Markdown("### 上传照片，让AI识别你的情绪")
                    
                    with gr.Row():
                        with gr.Column():
                            image_input = gr.Image(
                                type="pil",
                                label="上传图片",
                                height=300
                            )
                            with gr.Row():
                                predict_btn = gr.Button("🔍 识别情绪", variant="primary")
                                ai_switch = gr.Checkbox(
                                    label="🤖 启用AI情绪解释",
                                    value=True,
                                    info="专为自闭症等特殊人群设计的温柔指导"
                                )
                        
                        with gr.Column():
                            result_text = gr.Markdown(label="识别结果")
                            prob_chart = gr.Plot(label="情绪概率分布")
                    
                    # 确认保存区域
                    with gr.Row():
                        with gr.Column():
                            save_btn = gr.Button("✅ 确认并保存到情绪日记", variant="secondary")
                            save_status = gr.Markdown(
                                label="保存状态",
                                value="💡 识别情绪后点击上方按钮保存记录"
                            )
                        
                        with gr.Column():
                            # 隐藏的状态存储
                            prediction_data = gr.State(value=None)
                    
                    # AI解释区域
                    with gr.Row():
                        ai_explanation = gr.Markdown(
                            label="🤖 AI情绪解释师",
                            value="",
                            visible=True
                        )
                    
                    # 个性化建议区域
                    with gr.Row():
                        with gr.Column():
                            context_input = gr.Textbox(
                                label="额外情况说明（可选）",
                                placeholder="例如：今天工作压力很大，或者刚和朋友聊完天...",
                                lines=2
                            )
                            advice_btn = gr.Button("💡 获取个性化建议", variant="secondary")
                        
                        with gr.Column():
                            advice_result = gr.Markdown(
                                label="🎯 个性化建议",
                                value=""
                            )
                    
                    predict_btn.click(
                        fn=self.predict_from_image,
                        inputs=[image_input, ai_switch],
                        outputs=[result_text, prob_chart, ai_explanation, prediction_data]
                    )
                    
                    save_btn.click(
                        fn=self.confirm_and_save_emotion,
                        inputs=[prediction_data],
                        outputs=[save_status]
                    )
                    
                    advice_btn.click(
                        fn=self.get_personalized_advice,
                        inputs=[image_input, context_input],
                        outputs=[advice_result]
                    )
                
                # Tab 2: 记录查看
                with gr.TabItem("📋 记录查看"):
                    gr.Markdown("### 查看和管理最近的情绪记录")
                    
                    with gr.Row():
                        days_slider = gr.Slider(
                            minimum=1, maximum=30, value=7, step=1,
                            label="查看天数"
                        )
                        refresh_btn = gr.Button("🔄 刷新记录", variant="secondary")
                    
                    records_table = gr.Dataframe(
                        headers=['ID', '时间', '情绪', '置信度'],
                        label="最近的情绪记录"
                    )
                    
                    # 快捷删除功能
                    with gr.Row():
                        with gr.Column():
                            gr.Markdown("""
                            ## 🗑️ 删除记录功能
                            
                            提供多种删除方式，满足不同需求：
                            
                            1. **快速删除**：删除最新的几条记录
                            2. **按置信度删除**：删除置信度较低的不准确记录  
                            3. **精确删除** (高级)：按具体ID删除特定记录
                            """)
                            
                            with gr.Row():
                                with gr.Column():
                                    gr.Markdown("#### 🚀 快速删除")
                                    delete_count = gr.Slider(
                                        minimum=1, maximum=10, value=1, step=1,
                                        label="删除最近几条记录"
                                    )
                                    quick_delete_btn = gr.Button("🗑️ 快速删除", variant="secondary")
                                    
                                with gr.Column():
                                    gr.Markdown("#### 📊 按置信度删除")
                                    gr.Markdown("💡 *删除置信度较低的记录，提高数据质量*")
                                    confidence_input = gr.Slider(
                                        minimum=0.0, maximum=1.0, value=0.5, step=0.01,
                                        label="删除置信度 ≤ 此值的记录"
                                    )
                                    confidence_delete_btn = gr.Button("🗑️ 删除低置信度记录", variant="stop")
                            
                            # 传统删除方式（高级用户）
                            with gr.Accordion("高级删除选项", open=False):
                                with gr.Row():
                                    with gr.Column():
                                        gr.Markdown("#### 🔢 按ID删除")
                                        single_id_input = gr.Textbox(
                                            label="记录ID",
                                            placeholder="输入要删除的记录ID",
                                            lines=1
                                        )
                                        single_delete_btn = gr.Button("🗑️ 删除记录", variant="stop")
                                    
                                    with gr.Column():
                                        gr.Markdown("#### 📦 批量删除")
                                        batch_ids_input = gr.Textbox(
                                            label="记录ID列表",
                                            placeholder="输入多个ID，用逗号分隔（例如：1,2,3）",
                                            lines=1
                                        )
                                        batch_delete_btn = gr.Button("🗑️ 批量删除", variant="stop")
                    
                    # 删除操作状态
                    delete_status = gr.Markdown(
                        label="删除状态",
                        value="💡 请谨慎删除记录，删除操作不可撤销"
                    )
                    
                    refresh_btn.click(
                        fn=self.get_recent_records_with_delete,
                        inputs=[days_slider],
                        outputs=[records_table]
                    )
                    
                    # 快速删除事件
                    quick_delete_btn.click(
                        fn=self.quick_delete_recent_records,
                        inputs=[delete_count],
                        outputs=[delete_status]
                    ).then(  # 删除后自动刷新表格
                        fn=self.get_recent_records_with_delete,
                        inputs=[days_slider],
                        outputs=[records_table]
                    )
                    
                    # 按置信度删除事件
                    confidence_delete_btn.click(
                        fn=self.delete_records_by_confidence,
                        inputs=[confidence_input],
                        outputs=[delete_status]
                    ).then(  # 删除后自动刷新表格
                        fn=self.get_recent_records_with_delete,
                        inputs=[days_slider],
                        outputs=[records_table]
                    )
                    
                    # 传统删除方式
                    single_delete_btn.click(
                        fn=self.delete_emotion_record,
                        inputs=[single_id_input],
                        outputs=[delete_status]
                    ).then(  # 删除后自动刷新表格
                        fn=self.get_recent_records_with_delete,
                        inputs=[days_slider],
                        outputs=[records_table]
                    )
                    
                    batch_delete_btn.click(
                        fn=self.batch_delete_records,
                        inputs=[batch_ids_input],
                        outputs=[delete_status]
                    ).then(  # 删除后自动刷新表格
                        fn=self.get_recent_records_with_delete,
                        inputs=[days_slider],
                        outputs=[records_table]
                    )
                    
                    # 页面加载时自动刷新
                    demo.load(
                        fn=self.get_recent_records_with_delete,
                        inputs=[days_slider],
                        outputs=[records_table]
                    )
                
                # Tab 3: 趋势分析
                with gr.TabItem("📊 趋势分析"):
                    gr.Markdown("### 情绪趋势分析与可视化")
                    
                    with gr.Row():
                        analysis_days = gr.Slider(
                            minimum=1, maximum=30, value=7, step=1,
                            label="分析天数"
                        )
                        with gr.Column():
                            analyze_btn = gr.Button("📈 生成分析", variant="primary")
                            diary_btn = gr.Button("📖 生成AI情绪日记", variant="secondary")
                    
                    analysis_result = gr.Markdown(label="分析结果")
                    
                    with gr.Row():
                        trend_plot = gr.Plot(label="情绪趋势图")
                        dist_plot = gr.Plot(label="情绪分布图")
                    
                    # AI情绪日记区域
                    with gr.Row():
                        emotion_diary = gr.Markdown(
                            label="📖 AI情绪日记",
                            value="点击上方按钮生成专属的温柔情绪日记..."
                        )
                    
                    analyze_btn.click(
                        fn=self.create_trend_analysis,
                        inputs=[analysis_days],
                        outputs=[analysis_result, trend_plot, dist_plot]
                    )
                    
                    diary_btn.click(
                        fn=self.generate_emotion_diary,
                        inputs=[analysis_days],
                        outputs=[emotion_diary]
                    )
                
                # Tab 4: 数据管理
                with gr.TabItem("🔧 数据管理"):
                    gr.Markdown("### 数据导出")
                    
                    with gr.Row():
                        with gr.Column():
                            gr.Markdown("#### 📤 导出情绪记录")
                            export_days = gr.Slider(
                                minimum=1, maximum=365, value=30, step=1,
                                label="导出天数"
                            )
                            export_btn = gr.Button("📤 导出CSV文件", variant="primary")
                        
                        with gr.Column():
                            export_status = gr.Textbox(label="导出状态", interactive=False)
                    
                    gr.Markdown("""
                    💡 **导出说明**:
                    - 导出的CSV文件包含完整的情绪记录数据
                    - 可用于数据分析、备份或在其他工具中查看
                    - 文件保存在 `data/exports/` 目录下
                    """)
                    
                    export_btn.click(
                        fn=self.export_data,
                        inputs=[export_days],
                        outputs=[gr.File(label="下载文件"), export_status]
                    )
            
            # 页脚信息
            gr.Markdown(
                """
                ---
                💡 **使用提示**:
                - 📸 上传清晰的人脸照片获得更准确的识别结果
                - ✅ 识别后确认结果准确性再保存到情绪日记
                - 🗑️ 可以删除错误或不需要的情绪记录
                - 🤖 启用AI解释获得温柔、专业的情绪指导
                - 💡 获取个性化建议了解情绪应对策略
                - 📖 生成AI情绪日记回顾情绪变化历程
                - 📊 定期查看趋势分析了解情绪变化模式
                - 📋 导出数据可用于进一步分析
                
                🌟 **AI增强功能**:
                - **情绪解释师**: 温柔解读每次情绪识别结果
                - **个性化建议**: 基于当前情绪状态提供应对策略
                - **情绪日记**: 分析趋势数据生成鼓励性日记
                - **二次确认**: 确保记录准确性后再保存
                - **记录管理**: 支持单条或批量删除错误记录
                - **专业设计**: 专为自闭症、社交障碍人群优化
                
                🔧 **技术栈**: PyTorch + ResNet18 + 阿里百炼Qwen-Plus + Gradio + SQLite
                """
            )
        
        return demo
    
    def generate_demo_data(self):
        """为演示模式生成一些示例数据"""
        from datetime import datetime, timedelta
        import random
        import numpy as np
        
        stats = self.storage.get_database_stats()
        if stats['total_records'] > 0:
            print(f"📊 数据库中已有 {stats['total_records']} 条记录")
            return
        
        print("🎭 演示模式：生成示例数据...")
        
        emotions = ['happy', 'sad', 'angry', 'neutral', 'surprise', 'fear', 'disgust']
        emotions_cn = ['快乐', '悲伤', '愤怒', '中性', '惊讶', '恐惧', '厌恶']
        
        total_generated = 0
        
        # 生成过去7天的数据
        for day_offset in range(7):
            date = datetime.now() - timedelta(days=day_offset)
            
            # 每天生成3-8条记录
            daily_records = random.randint(3, 8)
            
            for _ in range(daily_records):
                # 随机选择情绪，但倾向于快乐和中性
                weights = [0.3, 0.1, 0.1, 0.25, 0.1, 0.1, 0.05]  # 快乐和中性权重更高
                emotion_idx = random.choices(range(len(emotions)), weights=weights)[0]
                
                # 生成随机时间
                hour = random.randint(7, 23)
                minute = random.randint(0, 59)
                record_time = date.replace(hour=hour, minute=minute, second=0, microsecond=0)
                
                # 生成置信度和概率分布
                confidence = random.uniform(0.6, 0.95)
                probabilities = np.random.dirichlet([0.5] * 7)
                probabilities[emotion_idx] = confidence
                probabilities = probabilities / probabilities.sum()
                
                # 临时修改storage的时间戳
                original_time = record_time.strftime('%Y-%m-%d %H:%M:%S')
                
                # 直接插入数据库
                import sqlite3
                with sqlite3.connect(self.storage.db_path) as conn:
                    cursor = conn.cursor()
                    cursor.execute('''
                        INSERT INTO emotion_records 
                        (timestamp, emotion, emotion_cn, confidence, all_probabilities, date, hour)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        original_time,
                        emotions[emotion_idx],
                        emotions_cn[emotion_idx],
                        confidence,
                        str(probabilities.tolist()),
                        record_time.strftime('%Y-%m-%d'),
                        hour
                    ))
                    conn.commit()
            
            total_generated += daily_records
        
        print(f"✅ 已生成 {total_generated} 条示例数据")

    def launch(self, share=False, server_name="0.0.0.0", server_port=7860):
        """
        启动应用程序
        
        Args:
            share: 是否创建公共链接
            server_name: 服务器地址
            server_port: 服务器端口
        """
        # 如果是演示模式，生成示例数据
        if not self.predictor.is_loaded:
            self.generate_demo_data()
        
        demo = self.create_interface()
        
        print("🚀 MoodMirror正在启动...")
        print(f"📊 数据库状态: {self.storage.get_database_stats()}")
        print(f"🤖 模型状态: {'✅ 已加载' if self.predictor.is_loaded else '🎭 演示模式'}")
        
        try:
            demo.launch(
                share=share,
                server_name=server_name,
                server_port=server_port,
                show_api=False
            )
        except OSError as e:
            if "address already in use" in str(e) or "Cannot find empty port" in str(e):
                print(f"端口 {server_port} 被占用，尝试使用端口 {server_port + 1}")
                demo.launch(
                    share=share,
                    server_name=server_name,
                    server_port=server_port + 1,
                    show_api=False
                )
            else:
                raise e

def main():
    """主函数"""
    app = MoodMirrorApp()
    
    # 启动应用
    app.launch(
        share=False,  # 设为True可创建公共链接
        server_name="127.0.0.1",  # 本地访问
        server_port=7860
    )

if __name__ == "__main__":
    main() 