# 🎭 MoodMirror: AI情绪日记 | AI Emotion Diary

[![License](https://img.shields.io/badge/license-MIT-blue)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.8+-blue)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.9+-red)](https://pytorch.org)
[![Accuracy](https://img.shields.io/badge/accuracy-65.67%25-green)](#模型性能--model-performance)
[![Contact](https://img.shields.io/badge/contact-zwan0569@student.monash.edu-orange)](mailto:zwan0569@student.monash.edu)

基于深度学习的智能情绪识别与AI陪伴系统，专为自闭症等特殊人群设计的情绪理解辅助工具

*An intelligent emotion recognition and AI companion system based on deep learning, specifically designed as an emotional understanding assistant for individuals with autism and other special needs*

---

## 🌟 产品设计理念 | Product Design Philosophy

### 中文版

**MoodMirror** 的核心设计理念是"技术向善，温暖陪伴"。我们深信技术应该服务于人的情感需求，特别是那些在情绪理解和表达方面面临挑战的特殊人群。

#### 🎯 设计原则
- **💙 温柔关怀**: 采用温柔、非评判的语言风格，避免任何可能造成压力的表达
- **🧩 理解包容**: 专为自闭症谱系障碍人群设计，理解他们独特的情绪体验方式
- **🎨 简洁直观**: 界面设计遵循简洁明了原则，减少认知负担
- **🔒 隐私保护**: 所有数据本地存储，保护用户隐私安全
- **📊 可视化反馈**: 通过图表和数据让情绪变化"看得见"，帮助用户建立自我认知

#### 🌈 服务对象
- **自闭症谱系障碍人群**: 提供情绪识别和理解支持
- **社交焦虑群体**: 帮助练习和理解情绪表达
- **心理健康关注者**: 长期追踪情绪变化趋势
- **教育工作者**: 辅助特殊教育和心理健康教育

### English Version

**MoodMirror** embodies the core philosophy of "Technology for Good, Warm Companionship." We firmly believe that technology should serve human emotional needs, especially for those who face challenges in understanding and expressing emotions.

#### 🎯 Design Principles
- **💙 Gentle Care**: Uses gentle, non-judgmental language to avoid any expressions that might cause stress
- **🧩 Understanding & Inclusion**: Specifically designed for individuals on the autism spectrum, understanding their unique emotional experiences
- **🎨 Simple & Intuitive**: Interface design follows simplicity principles to reduce cognitive load
- **🔒 Privacy Protection**: All data stored locally to protect user privacy
- **📊 Visual Feedback**: Makes emotional changes "visible" through charts and data to help users build self-awareness

#### 🌈 Target Users
- **Individuals with Autism Spectrum Disorders**: Providing emotion recognition and understanding support
- **Social Anxiety Groups**: Helping practice and understand emotional expression
- **Mental Health Conscious**: Long-term tracking of emotional trends
- **Educators**: Assisting special education and mental health education

---

## 🛠️ 训练技术细节 | Technical Training Details

### 中文版

#### 🏗️ 模型架构
- **基础模型**: ResNet18 (ImageNet预训练)
- **分类器**: 增强版三层神经网络
- **输入尺寸**: 224×224×3 RGB图像
- **输出类别**: 7种基本情绪 (愤怒、厌恶、恐惧、快乐、中性、悲伤、惊讶)

#### 📊 训练配置
```python
# 核心训练参数
训练轮数: 10 epochs
批次大小: 16
学习率: 0.0005 (自适应调整)
优化器: AdamW (weight_decay=0.01)
损失函数: CrossEntropyLoss
梯度裁剪: max_norm=1.0
```

#### 🎯 数据增强策略
- **CutMix增强**: 50%概率应用，α=1.0
- **几何变换**: 随机旋转(±20°)、透视变换(20%概率)
- **颜色空间**: 亮度/对比度/饱和度随机调整
- **随机擦除**: 30%概率应用，提升鲁棒性
- **高斯模糊**: 轻微模糊模拟真实场景

#### 🔄 学习率调度
- **策略**: Cosine Annealing Warm Restarts
- **初始学习率**: 0.0005
- **最小学习率**: 1e-7
- **重启周期**: 动态调整

#### 📁 数据集规模
- **训练集**: 28,709 张图像
- **验证集**: 7,178 张图像
- **数据来源**: FER2013 + 增强数据
- **标注质量**: 人工校验 + 自动筛选

### English Version

#### 🏗️ Model Architecture
- **Base Model**: ResNet18 (ImageNet pretrained)
- **Classifier**: Enhanced three-layer neural network
- **Input Size**: 224×224×3 RGB images
- **Output Categories**: 7 basic emotions (Angry, Disgust, Fear, Happy, Neutral, Sad, Surprise)

#### 📊 Training Configuration
```python
# Core Training Parameters
Epochs: 10
Batch Size: 16
Learning Rate: 0.0005 (adaptive adjustment)
Optimizer: AdamW (weight_decay=0.01)
Loss Function: CrossEntropyLoss
Gradient Clipping: max_norm=1.0
```

#### 🎯 Data Augmentation Strategy
- **CutMix Enhancement**: 50% probability, α=1.0
- **Geometric Transform**: Random rotation(±20°), perspective transform(20% prob)
- **Color Space**: Random brightness/contrast/saturation adjustment
- **Random Erasing**: 30% probability for robustness
- **Gaussian Blur**: Light blur to simulate real scenarios

#### 🔄 Learning Rate Scheduling
- **Strategy**: Cosine Annealing Warm Restarts
- **Initial LR**: 0.0005
- **Minimum LR**: 1e-7
- **Restart Cycles**: Dynamic adjustment

#### 📁 Dataset Scale
- **Training Set**: 28,709 images
- **Validation Set**: 7,178 images
- **Data Source**: FER2013 + augmented data
- **Annotation Quality**: Manual verification + automatic filtering

---

## 🎨 产品功能细节 | Product Feature Details

### 中文版

#### 📸 情绪识别模块
- **智能人脸检测**: 自动定位和裁剪人脸区域
- **实时情绪分析**: 支持图片上传和摄像头实时识别
- **置信度评估**: 显示每种情绪的概率分布
- **AI温柔解释**: 集成阿里百炼大模型，提供温柔的情绪解读

#### 🤖 AI陪伴系统
- **情绪解释师**: 用温柔、简单的语言解释识别结果
- **个性化建议**: 基于当前情绪状态提供应对策略
- **情绪日记生成**: 分析趋势数据，生成鼓励性的情绪日记
- **二次确认机制**: 确保记录准确性后再保存

#### 📊 数据分析与可视化
- **实时图表**: 情绪概率分布的动态可视化
- **趋势分析**: 时间序列情绪变化趋势
- **统计报告**: 多维度情绪模式分析
- **热力图**: 不同时间段的情绪分布模式

#### 🗂️ 记录管理系统
- **智能存储**: SQLite数据库本地存储
- **灵活删除**: 支持按置信度、时间、ID等多种删除方式
- **数据导出**: 支持CSV格式导出进行进一步分析
- **隐私保护**: 所有数据完全本地化，不上传云端

### English Version

#### 📸 Emotion Recognition Module
- **Intelligent Face Detection**: Automatic face region localization and cropping
- **Real-time Emotion Analysis**: Support for image upload and real-time camera recognition
- **Confidence Assessment**: Display probability distribution for each emotion
- **AI Gentle Explanation**: Integrated with Alibaba's Qianwen model for gentle emotion interpretation

#### 🤖 AI Companion System
- **Emotion Interpreter**: Explains recognition results in gentle, simple language
- **Personalized Suggestions**: Provides coping strategies based on current emotional state
- **Emotion Diary Generation**: Analyzes trend data to generate encouraging emotion diaries
- **Double Confirmation**: Ensures accuracy before saving records

#### 📊 Data Analysis & Visualization
- **Real-time Charts**: Dynamic visualization of emotion probability distribution
- **Trend Analysis**: Time series emotional change trends
- **Statistical Reports**: Multi-dimensional emotion pattern analysis
- **Heat Maps**: Emotion distribution patterns across different time periods

#### 🗂️ Record Management System
- **Intelligent Storage**: Local SQLite database storage
- **Flexible Deletion**: Support for deletion by confidence, time, ID, and other criteria
- **Data Export**: CSV format export for further analysis
- **Privacy Protection**: All data completely localized, no cloud upload

---

## 📊 模型性能 | Model Performance

### 中文版

#### 🎯 最新性能指标
- **验证准确率**: **65.67%** ⭐
- **训练准确率**: **55.30%**
- **模型大小**: 129.6 MB
- **推理速度**: ~300ms (CPU)
- **训练轮数**: 10 epochs

#### 📈 训练历史
```
Epoch 1: 验证准确率 63.07%
Epoch 2: 验证准确率 62.41%
Epoch 3: 验证准确率 63.81%
Epoch 4: 验证准确率 64.85%
Epoch 5: 验证准确率 65.67% ⭐ (最佳)
...
Epoch 10: 验证准确率 65.67% (最终)
```

#### 🏆 性能提升对比
| 版本 | 准确率 | 提升幅度 | 主要改进 |
|------|--------|----------|----------|
| 基础版 | 52.09% | - | ResNet18基础架构 |
| 增强版 | 62.33% | +10.24% | CutMix + 数据增强 |
| **最新版** | **65.67%** | **+13.58%** | **优化训练策略** |

### English Version

#### 🎯 Latest Performance Metrics
- **Validation Accuracy**: **65.67%** ⭐
- **Training Accuracy**: **55.30%**
- **Model Size**: 129.6 MB
- **Inference Speed**: ~300ms (CPU)
- **Training Epochs**: 10 epochs

#### 📈 Training History
```
Epoch 1: Validation Accuracy 63.07%
Epoch 2: Validation Accuracy 62.41%
Epoch 3: Validation Accuracy 63.81%
Epoch 4: Validation Accuracy 64.85%
Epoch 5: Validation Accuracy 65.67% ⭐ (Best)
...
Epoch 10: Validation Accuracy 65.67% (Final)
```

#### 🏆 Performance Improvement Comparison
| Version | Accuracy | Improvement | Key Features |
|---------|----------|-------------|--------------|
| Base | 52.09% | - | Basic ResNet18 architecture |
| Enhanced | 62.33% | +10.24% | CutMix + data augmentation |
| **Latest** | **65.67%** | **+13.58%** | **Optimized training strategy** |

---

## 🎭 示例输出 | Example Outputs

### 中文版

#### 📸 情绪识别示例
```
🎯 识别结果:
主要情绪: 快乐 (Happy)
置信度: 87.23%

📊 详细概率分布:
快乐: 87.23% ████████▊
中性: 8.45%  ▊
惊讶: 2.17%  ▎
悲伤: 1.23%  ▎
愤怒: 0.67%  ▎
恐惧: 0.15%  ▎
厌恶: 0.10%  ▎
```

#### 🤖 AI温柔解释
```
🌟 AI情绪解释师说:

亲爱的朋友，我从你的表情中感受到了满满的快乐！😊 
你的笑容真的很温暖，就像春天的阳光一样。这种快乐
的情绪是珍贵的，它表明你现在的状态很好。

建议你：
• 记录下这个美好的时刻
• 与身边的人分享这份快乐
• 保持这样积极的心态

每一个快乐的瞬间都值得被珍惜！ 💝
```

#### 📖 AI情绪日记
```
📅 2024年12月24日 情绪日记

亲爱的自己，

这一周你的情绪就像一幅美丽的画卷。我注意到你有
60%的时间都保持着快乐的状态，这真的很棒！特别是
在上午10点左右，你的笑容最灿烂。

虽然偶尔会有一些悲伤的时刻(15%)，但这是完全正常
的，就像天空中偶尔飘过的云朵。重要的是，你总能
重新找回快乐。

记住，你是独特而珍贵的，每一种情绪都是你内心
世界的一部分。继续保持这份美好吧！

温柔的AI伙伴 🤗
```

### English Version

#### 📸 Emotion Recognition Example
```
🎯 Recognition Result:
Primary Emotion: Happy
Confidence: 87.23%

📊 Detailed Probability Distribution:
Happy:    87.23% ████████▊
Neutral:   8.45% ▊
Surprise:  2.17% ▎
Sad:       1.23% ▎
Angry:     0.67% ▎
Fear:      0.15% ▎
Disgust:   0.10% ▎
```

#### 🤖 AI Gentle Explanation
```
🌟 AI Emotion Interpreter says:

Dear friend, I can feel so much happiness from your expression! 😊 
Your smile is truly warm, like sunshine in spring. This joyful 
emotion is precious and shows that you're in a wonderful state.

I suggest you:
• Record this beautiful moment
• Share this happiness with those around you
• Keep this positive mindset

Every moment of joy deserves to be treasured! 💝
```

#### 📖 AI Emotion Diary
```
📅 December 24, 2024 Emotion Diary

Dear Self,

This week your emotions have been like a beautiful painting. 
I noticed you maintained a happy state 60% of the time, which 
is truly wonderful! Especially around 10 AM, your smile was 
the brightest.

While there were occasional moments of sadness (15%), this is 
completely normal, like clouds occasionally passing through 
the sky. What matters is that you always find your way back 
to happiness.

Remember, you are unique and precious, and every emotion is 
part of your inner world. Keep cherishing this beauty!

Your gentle AI companion 🤗
```

---

## 🚀 快速开始 | Quick Start

### 中文版

#### 方法1: 一键启动 (推荐)
```bash
python run.py --no-train  # 使用已训练模型
```

#### 方法2: 完整安装
```bash
# 1. 克隆项目
git clone <repository-url>
cd emotiondiary

# 2. 安装依赖
pip install -r requirements.txt

# 3. 配置阿里百炼API密钥 (可选，用于AI温柔解释功能)
# 编辑 app/emotion_ai.py 第19行，替换API密钥
# self.api_key = api_key or "your-api-key-here"

# 4. 下载预训练权重 (可选)
python training/download_pretrained.py

# 5. 启动应用
python app/main.py
```

#### 方法3: 从头训练
```bash
# 1. 准备数据集 (将FER2013数据放入dataset文件夹)
# 2. 开始训练
python training/train_model.py

# 3. 启动应用
python run.py
```

**访问地址**: http://127.0.0.1:7860

### English Version

#### Method 1: One-Click Start (Recommended)
```bash
python run.py --no-train  # Use pre-trained model
```

#### Method 2: Complete Installation
```bash
# 1. Clone repository
git clone <repository-url>
cd emotiondiary

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure Alibaba Qianwen API key (optional, for AI gentle explanation)
# Edit app/emotion_ai.py line 19, replace API key
# self.api_key = api_key or "your-api-key-here"

# 4. Download pretrained weights (optional)
python training/download_pretrained.py

# 5. Launch application
python app/main.py
```

#### Method 3: Train from Scratch
```bash
# 1. Prepare dataset (place FER2013 data in dataset folder)
# 2. Start training
python training/train_model.py

# 3. Launch application
python run.py
```

**Access URL**: http://127.0.0.1:7860

---

## ⚙️ 配置说明 | Configuration

### 中文版

#### 🔧 阿里百炼API配置 (可选)
为了使用AI温柔解释功能，需要配置阿里百炼API密钥：

1. **获取API密钥**：
   - 前往 [阿里云百炼平台](https://dashscope.aliyuncs.com/)
   - 注册账号并获取API密钥

2. **配置密钥**：
   - 打开文件：`app/emotion_ai.py`
   - 定位到第19行：`self.api_key = api_key or "sk-enter-your-api-key-here"`
   - 将 `"sk-enter-your-api-key-here"` 替换为您的实际API密钥

3. **功能说明**：
   - ✅ **有API密钥**：完整的AI温柔解释、情绪日记生成、个性化建议
   - 🔄 **无API密钥**：自动使用备用解释机制，功能不受影响

### English Version

#### 🔧 Alibaba Qianwen API Configuration (Optional)
To use AI gentle explanation features, configure Alibaba Qianwen API key:

1. **Get API Key**:
   - Visit [Alibaba Cloud Qianwen Platform](https://dashscope.aliyuncs.com/)
   - Register account and obtain API key

2. **Configure Key**:
   - Open file: `app/emotion_ai.py`
   - Locate line 19: `self.api_key = api_key or "sk-enter-your-api-key-here"`
   - Replace `"sk-enter-your-api-key-here"` with your actual API key

3. **Feature Description**:
   - ✅ **With API Key**: Full AI gentle explanations, emotion diary generation, personalized advice
   - 🔄 **Without API Key**: Automatic fallback mechanism, functionality unaffected

---

## 🏗️ 项目结构 | Project Structure

```
emotiondiary/
├── 📁 dataset/                    # 数据集 | Dataset
│   ├── train/                     # 训练数据 | Training data
│   └── test/                      # 测试数据 | Test data
├── 📁 model/                      # 模型文件 | Model files
│   ├── emotion_classifier.py      # 模型定义 | Model definition
│   ├── emotion_model.pt          # 训练模型 | Trained model (129.6MB)
│   └── emotion_model_history.json # 训练历史 | Training history
├── 📁 app/                        # 应用程序 | Application
│   ├── main.py                   # 主程序 | Main application
│   ├── predict.py                # 预测模块 | Prediction module
│   ├── storage.py                # 数据存储 | Data storage
│   └── emotion_ai.py             # AI服务 | AI service
├── 📁 training/                   # 训练脚本 | Training scripts
│   ├── train_model.py            # 训练脚本 | Training script
│   └── download_pretrained.py    # 权重下载 | Weight download
├── 📁 analysis/                   # 数据分析 | Data analysis
│   └── plot_moods.py             # 可视化 | Visualization
├── 📁 data/                       # 数据文件 | Data files
├── 📄 requirements.txt            # 依赖包 | Dependencies
├── 📄 run.py                     # 启动脚本 | Launch script
└── 📄 README.md                  # 项目说明 | Documentation
```

---

## 🛠️ 技术栈 | Technology Stack

### 深度学习 | Deep Learning
- **PyTorch** 1.9+ - 深度学习框架 | Deep learning framework
- **torchvision** - 计算机视觉库 | Computer vision library
- **OpenCV** - 图像处理 | Image processing
- **PIL/Pillow** - 图像操作 | Image manipulation

### Web界面 | Web Interface
- **Gradio** 3.0+ - Web界面框架 | Web UI framework
- **HTML/CSS/JavaScript** - 前端技术 | Frontend technologies

### 数据处理 | Data Processing
- **pandas** - 数据分析 | Data analysis
- **numpy** - 数值计算 | Numerical computing
- **SQLite** - 数据库 | Database

### 可视化 | Visualization
- **matplotlib** - 基础绘图 | Basic plotting
- **seaborn** - 统计可视化 | Statistical visualization
- **plotly** - 交互式图表 | Interactive charts

### AI服务 | AI Services
- **阿里百炼** - 大语言模型 | Large language model
- **Qwen-Plus** - 情绪解释与建议 | Emotion interpretation & advice
- **API配置** - `app/emotion_ai.py:19` | API configuration at `app/emotion_ai.py:19`

---

## 📞 联系方式 | Contact Information

### 项目维护者 | Project Maintainer
- **邮箱 | Email**: [zwan0569@student.monash.edu](mailto:zwan0569@student.monash.edu)
- **机构 | Institution**: Monash University
- **专业 | Major**: Computer Science

### 支持渠道 | Support Channels
- **📧 技术支持 | Technical Support**: zwan0569@student.monash.edu
- **🐛 Bug报告 | Bug Reports**: GitHub Issues
- **💡 功能建议 | Feature Requests**: GitHub Issues
- **📚 文档问题 | Documentation Issues**: GitHub Issues

### 响应时间 | Response Time
- **邮件回复 | Email Response**: 24-48小时 | 24-48 hours
- **Issue处理 | Issue Processing**: 1-3个工作日 | 1-3 business days
- **紧急问题 | Urgent Issues**: 当日回复 | Same day response

---


## 🙏 致谢 | Acknowledgments

### 中文版
- **FER2013数据集**: 提供情绪分类的基准数据
- **PyTorch团队**: 优秀的深度学习框架
- **Gradio开发者**: 简洁易用的Web界面框架
- **阿里云百炼**: 提供温柔的AI解释服务
- **开源社区**: 无私分享的技术知识和代码
- **特殊教育工作者**: 提供宝贵的需求反馈和建议
- **所有测试用户**: 帮助发现问题和改进产品

### English Version
- **FER2013 Dataset**: Providing benchmark data for emotion classification
- **PyTorch Team**: Excellent deep learning framework
- **Gradio Developers**: Simple and easy-to-use web UI framework
- **Alibaba Cloud Qianwen**: Providing gentle AI interpretation services
- **Open Source Community**: Selflessly sharing technical knowledge and code
- **Special Education Workers**: Providing valuable needs feedback and suggestions
- **All Test Users**: Helping discover issues and improve the product

---

## 🌟 项目亮点 | Project Highlights

### 🏆 技术创新 | Technical Innovation
- ✅ **65.67%** 高准确率情绪识别
- ✅ **CutMix数据增强** 提升模型鲁棒性
- ✅ **温柔AI解释** 专为特殊人群设计
- ✅ **本地化部署** 保护用户隐私
- ✅ **智能记录管理** 多维度数据分析

### 🎯 社会价值 | Social Value
- ✅ **助力特殊教育** 支持自闭症康复
- ✅ **促进心理健康** 情绪认知与管理
- ✅ **技术普惠** 开源免费使用
- ✅ **跨平台支持** 广泛的设备兼容性
- ✅ **持续改进** 基于用户反馈优化

---

**⭐ 如果这个项目对您有帮助，请给我们一个星标！| If this project helps you, please give us a star! ⭐**

**📧 联系我们 | Contact Me**: [zwan0569@student.monash.edu](mailto:zwan0569@student.monash.edu)

**🔗 当前版本 | Current Version**: v3.0 | **模型准确率 | Model Accuracy**: 65.67% | **状态 | Status**: 生产就绪 | Production Ready**

---

*让技术更有温度，让情绪更被理解 | Making technology warmer, making emotions more understood* 💙 
