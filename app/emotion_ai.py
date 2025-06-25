import os
import json
from typing import List, Dict, Tuple, Optional
from openai import OpenAI

class EmotionAI:
    """
    情绪AI解释服务
    专为自闭症等社交障碍人群设计的情绪理解辅助工具
    """
    
    def __init__(self, api_key: str = None):
        """
        初始化情绪AI服务
        
        Args:
            api_key: 阿里百炼API密钥，如果为None则从环境变量获取
        """
        self.api_key = api_key or "sk-98ab3de935d84bbd98da194d89ef97ea"
        
        # 初始化OpenAI客户端（阿里百炼兼容接口）
        self.client = OpenAI(
            api_key=self.api_key,
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        )
        
        # 情绪标签映射
        self.emotion_labels = {
            'angry': '愤怒',
            'disgust': '厌恶', 
            'fear': '恐惧',
            'happy': '快乐',
            'neutral': '平静',
            'sad': '悲伤',
            'surprise': '惊讶'
        }
        
        # 情绪描述
        self.emotion_descriptions = {
            'angry': '愤怒是当我们感到不公平或受到阻碍时的自然反应',
            'disgust': '厌恶帮助我们远离不喜欢或不健康的事物',
            'fear': '恐惧是身体在提醒我们注意安全',
            'happy': '快乐是当好事发生时的美好感受',
            'neutral': '平静是一种放松、没有强烈情绪的状态',
            'sad': '悲伤是当失去或遇到困难时的正常反应',
            'surprise': '惊讶是遇到意外事情时的即时反应'
        }

    def select_emotions(self, probabilities: List[float], emotion_names: List[str], 
                       confidence_threshold: float = 0.4) -> List[Tuple[str, float]]:
        """
        根据置信度选择要传递给AI的情绪
        
        Args:
            probabilities: 情绪概率列表
            emotion_names: 情绪名称列表
            confidence_threshold: 置信度阈值
            
        Returns:
            选中的情绪列表 [(情绪名, 概率), ...]
        """
        # 创建情绪-概率对并排序
        emotion_probs = list(zip(emotion_names, probabilities))
        emotion_probs.sort(key=lambda x: x[1], reverse=True)
        
        # 检查是否有高置信度情绪
        top_emotion = emotion_probs[0]
        if top_emotion[1] >= confidence_threshold:
            return [top_emotion]
        else:
            # 返回前两个情绪
            return emotion_probs[:2]

    def create_prompt(self, selected_emotions: List[Tuple[str, float]], 
                     context: str = "") -> str:
        """
        创建专为自闭症人群设计的prompt
        
        Args:
            selected_emotions: 选中的情绪列表
            context: 额外上下文信息
            
        Returns:
            构建的prompt
        """
        # 基础系统角色
        system_role = """你是一位温柔、专业的情绪辅导师，专门帮助自闭症和社交障碍人群理解情绪。

你的特点：
- 使用简单、清晰的语言
- 语气温和、耐心、不带判断
- 提供具体、实用的建议
- 避免复杂的心理学术语
- 重点关注情绪的正常性和可理解性

回答格式要求：
1. 情绪识别确认（1-2句）
2. 情绪解释（为什么会有这种感受）
3. 正常化表达（这是正常的）
4. 实用建议（可以怎么做）
5. 鼓励语句

回答长度：保持在150-200字，简洁易懂。"""

        # 构建用户消息
        if len(selected_emotions) == 1:
            emotion, confidence = selected_emotions[0]
            emotion_cn = self.emotion_labels[emotion]
            user_message = f"""我刚刚拍了一张照片，系统识别出我的情绪是"{emotion_cn}"，置信度是{confidence:.1%}。

{context}

请帮我理解这种情绪，并给出温柔的指导。"""
        else:
            emotions_str = "、".join([f"{self.emotion_labels[e[0]]}({e[1]:.1%})" for e in selected_emotions])
            user_message = f"""我刚刚拍了一张照片，系统识别出我可能的情绪包括：{emotions_str}。

{context}

请帮我理解这些情绪，并给出温柔的指导。"""

        return system_role, user_message

    def get_emotion_explanation(self, probabilities: List[float], 
                              emotion_names: List[str] = None,
                              context: str = "",
                              confidence_threshold: float = 0.4) -> Dict:
        """
        获取情绪解释
        
        Args:
            probabilities: 情绪概率列表
            emotion_names: 情绪名称列表，默认使用标准7种情绪
            context: 额外上下文信息
            confidence_threshold: 置信度阈值
            
        Returns:
            包含AI解释的字典
        """
        if emotion_names is None:
            emotion_names = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
        
        try:
            # 选择要解释的情绪
            selected_emotions = self.select_emotions(probabilities, emotion_names, confidence_threshold)
            
            # 创建prompt
            system_role, user_message = self.create_prompt(selected_emotions, context)
            
            # 调用大模型
            completion = self.client.chat.completions.create(
                model="qwen-plus",
                messages=[
                    {"role": "system", "content": system_role},
                    {"role": "user", "content": user_message},
                ],
                temperature=0.7,  # 稍微增加创造性
                max_tokens=300,   # 控制回答长度
            )
            
            ai_response = completion.choices[0].message.content
            
            # 构建返回结果
            result = {
                'success': True,
                'selected_emotions': [
                    {
                        'emotion': emotion,
                        'emotion_cn': self.emotion_labels[emotion],
                        'confidence': confidence,
                        'description': self.emotion_descriptions[emotion]
                    }
                    for emotion, confidence in selected_emotions
                ],
                'ai_explanation': ai_response,
                'model_used': 'qwen-plus',
                'tokens_used': completion.usage.total_tokens if completion.usage else 0
            }
            
            return result
            
        except Exception as e:
            # 错误处理，提供备用解释
            return self._get_fallback_explanation(probabilities, emotion_names, str(e))

    def _get_fallback_explanation(self, probabilities: List[float], 
                                 emotion_names: List[str], error: str) -> Dict:
        """
        当AI服务不可用时的备用解释
        
        Args:
            probabilities: 情绪概率列表
            emotion_names: 情绪名称列表
            error: 错误信息
            
        Returns:
            备用解释字典
        """
        # 选择最可能的情绪
        max_idx = probabilities.index(max(probabilities))
        top_emotion = emotion_names[max_idx]
        top_confidence = probabilities[max_idx]
        
        # 生成简单的备用解释
        emotion_cn = self.emotion_labels[top_emotion]
        description = self.emotion_descriptions[top_emotion]
        
        fallback_explanation = f"""我看到你可能正在感受"{emotion_cn}"的情绪。

{description}。这是完全正常的人类情绪反应。

每个人都会经历各种情绪，这些情绪帮助我们理解自己和周围的世界。如果你想进一步了解这种感受，可以尝试深呼吸，或者和信任的人聊聊。

记住，你的感受都是有效的，值得被理解和关爱。"""

        return {
            'success': False,
            'error': error,
            'selected_emotions': [{
                'emotion': top_emotion,
                'emotion_cn': emotion_cn,
                'confidence': top_confidence,
                'description': description
            }],
            'ai_explanation': fallback_explanation,
            'model_used': 'fallback',
            'tokens_used': 0
        }

    def test_connection(self) -> Dict:
        """
        测试与阿里百炼平台的连接
        
        Returns:
            测试结果字典
        """
        try:
            completion = self.client.chat.completions.create(
                model="qwen-plus",
                messages=[
                    {"role": "system", "content": "你是一个测试助手。"},
                    {"role": "user", "content": "请回复'连接成功'"},
                ],
                max_tokens=10
            )
            
            return {
                'success': True,
                'message': '与阿里百炼平台连接成功',
                'response': completion.choices[0].message.content,
                'model': 'qwen-plus'
            }
            
        except Exception as e:
            return {
                'success': False,
                'message': f'连接失败: {str(e)}',
                'error': str(e)
            }

    def generate_emotion_diary(self, emotion_data: Dict, days: int = 7) -> Dict:
        """
        基于情绪趋势数据生成温和的情绪日记
        
        Args:
            emotion_data: 情绪趋势数据字典
            days: 分析天数
            
        Returns:
            包含生成日记的字典
        """
        try:
            # 解析情绪数据
            if not emotion_data or emotion_data.get('total_records', 0) == 0:
                return {
                    'success': False,
                    'error': '暂无足够的情绪数据来生成日记',
                    'diary_content': ''
                }
            
            # 提取关键信息
            total_records = emotion_data['total_records']
            dominant_emotion = emotion_data['dominant_emotion']
            average_confidence = emotion_data['average_confidence']
            emotion_distribution = emotion_data['emotion_distribution']
            daily_patterns = emotion_data.get('daily_patterns', [])
            
            # 构建情绪分布描述
            emotion_summary = []
            for _, row in emotion_distribution.iterrows():
                emotion_summary.append(f"{row['emotion_cn']}({row['count']}次, {row['percentage']:.1f}%)")
            
            # 构建prompt
            system_role = """你是一位温柔的情绪日记助手，专门为自闭症和社交障碍人群撰写鼓励性的情绪日记。

你的写作特点：
- 使用第二人称，温暖亲切的语气
- 语言简单清晰，避免复杂词汇
- 重点关注积极面和成长
- 承认困难但强调正常性
- 提供具体的自我关爱建议
- 长度控制在200-300字

日记结构：
1. 温暖的开头问候
2. 对情绪数据的温和解读
3. 肯定和鼓励
4. 具体的自我关爱建议
5. 充满希望的结尾

避免：
- 复杂的心理学术语
- 过于专业的分析
- 批判性语言
- 过长的句子"""

            user_message = f"""请基于以下情绪数据为我写一篇温柔的情绪日记：

📊 时间周期：过去{days}天
📈 总记录数：{total_records}次
🎯 主要情绪：{dominant_emotion}
📊 平均置信度：{average_confidence:.2f}

🌈 情绪分布：
{', '.join(emotion_summary)}

请用温和、鼓励的语气写一篇个人情绪日记，帮助我理解和接纳自己的情绪变化。"""

            # 调用大模型
            completion = self.client.chat.completions.create(
                model="qwen-plus",
                messages=[
                    {"role": "system", "content": system_role},
                    {"role": "user", "content": user_message},
                ],
                temperature=0.8,  # 增加创造性
                max_tokens=400,   # 足够的空间生成日记
            )
            
            diary_content = completion.choices[0].message.content
            
            return {
                'success': True,
                'diary_content': diary_content,
                'analysis_period': f'{days}天',
                'data_summary': {
                    'total_records': total_records,
                    'dominant_emotion': dominant_emotion,
                    'average_confidence': average_confidence
                },
                'model_used': 'qwen-plus',
                'tokens_used': completion.usage.total_tokens if completion.usage else 0
            }
            
        except Exception as e:
            # 备用日记生成
            return self._generate_fallback_diary(emotion_data, days, str(e))

    def _generate_fallback_diary(self, emotion_data: Dict, days: int, error: str) -> Dict:
        """
        生成备用情绪日记
        
        Args:
            emotion_data: 情绪数据
            days: 天数
            error: 错误信息
            
        Returns:
            备用日记字典
        """
        total_records = emotion_data.get('total_records', 0)
        dominant_emotion = emotion_data.get('dominant_emotion', '未知')
        
        fallback_diary = f"""亲爱的朋友，

在过去的{days}天里，你记录了{total_records}次情绪，这本身就是一件很棒的事情。能够关注和记录自己的感受，说明你正在努力了解自己。

你最常体验到的情绪是{dominant_emotion}，这是完全正常的。每个人都有自己的情绪模式，没有对错之分。重要的是你愿意去观察和接纳它们。

情绪就像天气一样，有晴有雨，有风有雪。它们都是生活的一部分，都有其存在的意义。当你感到困难时，记住这只是暂时的，就像暴风雨后总会有晴天。

建议你今天做一件让自己感到舒适的小事：可能是听一首喜欢的音乐，或者给自己泡一杯温暖的茶。照顾好自己的感受，你值得被温柔对待。

加油，你正在做得很好。💙"""

        return {
            'success': False,
            'error': error,
            'diary_content': fallback_diary,
            'analysis_period': f'{days}天',
            'data_summary': emotion_data,
            'model_used': 'fallback',
            'tokens_used': 0
        }

    def generate_personalized_advice(self, current_emotion: str, confidence: float, 
                                   context: str = "") -> Dict:
        """
        基于当前情绪生成个性化建议
        
        Args:
            current_emotion: 当前情绪
            confidence: 置信度
            context: 额外上下文
            
        Returns:
            建议字典
        """
        try:
            emotion_cn = self.emotion_labels.get(current_emotion, current_emotion)
            
            system_role = """你是一位专业的情绪管理顾问，专门为自闭症和社交障碍人群提供实用的情绪应对策略。

你的建议特点：
- 具体可操作的步骤
- 简单易懂的语言
- 考虑感官敏感性
- 重点关注自我安抚
- 避免社交压力
- 提供多种选择

建议结构：
1. 即时应对策略（3-5个）
2. 自我安抚方法
3. 环境调节建议
4. 长期管理技巧
控制在150-200字内。"""

            user_message = f"""当前情绪：{emotion_cn}（置信度{confidence:.1%}）
{context}

请为这种情绪状态提供专门的应对建议和自我关爱策略。"""

            completion = self.client.chat.completions.create(
                model="qwen-plus",
                messages=[
                    {"role": "system", "content": system_role},
                    {"role": "user", "content": user_message},
                ],
                temperature=0.6,
                max_tokens=250,
            )
            
            advice_content = completion.choices[0].message.content
            
            return {
                'success': True,
                'advice_content': advice_content,
                'emotion': current_emotion,
                'emotion_cn': emotion_cn,
                'confidence': confidence,
                'model_used': 'qwen-plus',
                'tokens_used': completion.usage.total_tokens if completion.usage else 0
            }
            
        except Exception as e:
            # 备用建议
            return self._generate_fallback_advice(current_emotion, confidence, str(e))

    def _generate_fallback_advice(self, emotion: str, confidence: float, error: str) -> Dict:
        """生成备用建议"""
        emotion_cn = self.emotion_labels.get(emotion, emotion)
        
        advice_map = {
            'happy': '继续保持这种美好的感受，可以记录下让你快乐的事情，或者和信任的人分享这份喜悦。',
            'sad': '悲伤是正常的情绪。给自己一些温暖：裹上柔软的毯子，听舒缓的音乐，或者写下感受。',
            'angry': '愤怒说明某些事情对你很重要。尝试深呼吸，或者通过安全的方式表达，比如画画或写字。',
            'fear': '恐惧提醒我们注意安全。找一个安全的地方，做深呼吸，提醒自己现在是安全的。',
            'neutral': '平静是很珍贵的状态。享受这份宁静，做一些让你感到舒适的事情。',
            'surprise': '惊讶说明生活充满未知。给自己时间适应，不急于做判断。',
            'disgust': '厌恶帮助我们远离不舒服的事物。相信自己的感受，寻找让你感到舒适的环境。'
        }
        
        fallback_advice = f"""针对{emotion_cn}情绪的建议：

{advice_map.get(emotion, '每种情绪都有其意义，给自己时间去感受和理解。')}

记住，所有情绪都是正常的，你不需要立即"修复"它们。照顾好自己，做让自己感到安全和舒适的事情。"""

        return {
            'success': False,
            'error': error,
            'advice_content': fallback_advice,
            'emotion': emotion,
            'emotion_cn': emotion_cn,
            'confidence': confidence,
            'model_used': 'fallback',
            'tokens_used': 0
        }

# 创建全局实例
emotion_ai = EmotionAI()

if __name__ == "__main__":
    # 测试功能
    print("🧪 测试情绪AI服务...")
    
    # 测试连接
    test_result = emotion_ai.test_connection()
    print(f"连接测试: {test_result}")
    
    # 测试情绪解释
    if test_result['success']:
        test_probs = [0.1, 0.15, 0.2, 0.45, 0.05, 0.03, 0.02]  # happy最高
        result = emotion_ai.get_emotion_explanation(test_probs)
        print(f"\n情绪解释测试:")
        print(f"选中情绪: {result['selected_emotions']}")
        print(f"AI解释: {result['ai_explanation']}") 