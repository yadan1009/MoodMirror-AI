#!/usr/bin/env python3
"""
MoodMirror 智能启动脚本
运行此脚本可以快速启动整个情绪识别系统
"""

import os
import sys
import subprocess
import argparse
import time
import socket
from pathlib import Path

def check_dependencies():
    """检查依赖包是否安装"""
    print("🔍 检查依赖包...")
    
    # 包名映射 (安装名 -> 导入名)
    package_mapping = {
        'torch': 'torch',
        'torchvision': 'torchvision', 
        'opencv-python': 'cv2',
        'gradio': 'gradio',
        'pandas': 'pandas',
        'matplotlib': 'matplotlib',
        'plotly': 'plotly',
        'seaborn': 'seaborn',
        'tqdm': 'tqdm'
    }
    
    missing_packages = []
    
    for install_name, import_name in package_mapping.items():
        try:
            __import__(import_name)
            print(f"✅ {install_name}")
        except ImportError:
            print(f"❌ {install_name} - 未安装")
            missing_packages.append(install_name)
    
    if missing_packages:
        print(f"\n⚠️  缺少以下依赖包: {', '.join(missing_packages)}")
        print("请运行以下命令安装:")
        print(f"pip install {' '.join(missing_packages)}")
        return False
    
    print("✅ 所有依赖包已安装")
    return True

def check_model_status():
    """智能检查模型状态"""
    print("\n🤖 检查模型状态...")
    
    model_dir = Path("model")
    best_model = model_dir / "emotion_model.pt"
    latest_model = model_dir / "emotion_model_latest.pt"
    history_file = model_dir / "emotion_model_history.json"
    
    status = {
        'has_best': best_model.exists(),
        'has_latest': latest_model.exists(),
        'has_history': history_file.exists(),
        'can_resume': False,
        'accuracy': None,
        'epochs_completed': 0
    }
    
    if status['has_latest']:
        try:
            import torch
            checkpoint = torch.load(latest_model, map_location='cpu')
            status['can_resume'] = True
            status['accuracy'] = checkpoint.get('accuracy', 0)
            status['epochs_completed'] = checkpoint.get('epoch', 0) + 1
        except Exception as e:
            print(f"⚠️  无法读取checkpoint: {e}")
    
    # 显示状态
    if status['has_best'] and status['has_latest']:
        print(f"✅ 发现完整模型文件")
        if status['can_resume']:
            print(f"   📊 当前进度: {status['epochs_completed']} epochs")
            print(f"   🎯 验证准确率: {status['accuracy']:.2f}%")
        print("   💡 模型已就绪，可直接启动应用")
    elif status['has_latest']:
        print(f"✅ 发现训练中的模型")
        if status['can_resume']:
            print(f"   📊 可从第 {status['epochs_completed']} epoch继续训练")
            print(f"   🎯 当前准确率: {status['accuracy']:.2f}%")
    elif status['has_best']:
        print(f"✅ 发现最佳模型文件")
        print("   💡 可直接启动应用")
    else:
        print("❌ 未发现模型文件")
        print("   🎓 需要先训练模型")
    
    return status

def check_dataset():
    """检查数据集是否存在"""
    print("\n📁 检查数据集...")
    
    dataset_path = Path("dataset")
    train_path = dataset_path / "train"
    test_path = dataset_path / "test"
    
    if not dataset_path.exists():
        print("❌ dataset文件夹不存在")
        return False
    
    if not train_path.exists() or not test_path.exists():
        print("❌ 训练或测试数据文件夹不存在")
        return False
    
    # 检查是否有情绪类别文件夹
    emotion_folders = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
    
    train_count = 0
    test_count = 0
    
    for folder in emotion_folders:
        train_folder = train_path / folder
        test_folder = test_path / folder
        
        if not train_folder.exists():
            print(f"❌ 训练数据中缺少 {folder} 文件夹")
            return False
        
        if not test_folder.exists():
            print(f"❌ 测试数据中缺少 {folder} 文件夹")
            return False
        
        # 统计文件数量
        train_files = len(list(train_folder.glob('*.*')))
        test_files = len(list(test_folder.glob('*.*')))
        train_count += train_files
        test_count += test_files
    
    print(f"✅ 数据集检查通过")
    print(f"   📊 训练样本: {train_count:,}")
    print(f"   📊 测试样本: {test_count:,}")
    return True

def ask_user_choice(model_status):
    """询问用户选择"""
    print("\n🤔 请选择操作：")
    
    if model_status['has_best'] or model_status['has_latest']:
        print("1. 🚀 直接启动应用（推荐）")
        
        if model_status['can_resume']:
            print("2. 📈 继续训练模型（断点续训）")
        
        print("3. 🔄 重新训练模型")
        print("4. 🛠️  仅检查环境")
        print("5. ❌ 退出")
        
        while True:
            choice = input("\n请输入选择 (1-5): ").strip()
            if choice in ['1', '2', '3', '4', '5']:
                return choice
            print("⚠️  请输入有效选择")
    else:
        print("1. 🎓 开始训练模型")
        print("2. 🛠️  仅检查环境")
        print("3. ❌ 退出")
        
        while True:
            choice = input("\n请输入选择 (1-3): ").strip()
            if choice in ['1', '2', '3']:
                # 映射到统一的选择
                mapping = {'1': '3', '2': '4', '3': '5'}
                return mapping[choice]
            print("⚠️  请输入有效选择")

def start_training_background():
    """后台启动训练"""
    print("\n🎓 启动后台训练...")
    print("💡 训练将在后台进行，您可以:")
    print("   - 按 Ctrl+C 查看当前进度")
    print("   - 训练完成后会自动提示")
    print("   - 训练日志保存在终端输出中")
    
    try:
        # 使用 nohup 在后台运行
        if os.name == 'nt':  # Windows
            process = subprocess.Popen([
                sys.executable, "training/train_model.py"
            ], creationflags=subprocess.CREATE_NEW_CONSOLE)
        else:  # Unix/Linux/Mac
            process = subprocess.Popen([
                sys.executable, "training/train_model.py"
            ])
        
        print(f"✅ 训练进程已启动 (PID: {process.pid})")
        print("🔍 可以运行以下命令查看进度:")
        print("   python run.py --check-training")
        
        return True
        
    except Exception as e:
        print(f"❌ 启动训练失败: {e}")
        return False

def start_training_interactive():
    """交互式训练"""
    print("\n🎓 开始交互式训练...")
    print("⏱️  预计需要 2-3 小时，请耐心等待...")
    print("⏹️  按 Ctrl+C 可以随时停止\n")
    
    try:
        subprocess.run([sys.executable, "training/train_model.py"], check=True)
        print("\n✅ 模型训练完成!")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ 模型训练失败: {e}")
        return False
    except KeyboardInterrupt:
        print("\n⏹️  训练被用户中断")
        return False

def check_port_available(port=7860):
    """检查端口是否可用"""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind(('127.0.0.1', port))
            return True
        except OSError:
            return False

def launch_app():
    """启动应用"""
    print("\n🚀 启动 MoodMirror 应用...")
    
    # 检查端口
    port = 7860
    if not check_port_available(port):
        print(f"⚠️  端口 {port} 被占用，尝试寻找可用端口...")
        for test_port in range(7861, 7870):
            if check_port_available(test_port):
                port = test_port
                break
        else:
            print("❌ 无法找到可用端口")
            return False
    
    print(f"🌐 应用将启动在: http://127.0.0.1:{port}")
    print("⏹️  按 Ctrl+C 停止应用")
    print("-" * 50)
    
    try:
        env = os.environ.copy()
        if port != 7860:
            env['GRADIO_SERVER_PORT'] = str(port)
        
        subprocess.run([sys.executable, "app/main.py"], 
                      check=True, env=env)
        
    except subprocess.CalledProcessError as e:
        print(f"❌ 应用启动失败: {e}")
        return False
    except KeyboardInterrupt:
        print("\n👋 应用已停止")
        return True

def check_training_progress():
    """检查训练进度"""
    model_status = check_model_status()
    
    if model_status['can_resume']:
        print(f"\n📊 当前训练进度:")
        print(f"   ✅ 已完成: {model_status['epochs_completed']} epochs")
        print(f"   🎯 验证准确率: {model_status['accuracy']:.2f}%")
        
        # 检查训练进程
        try:
            result = subprocess.run(['ps', 'aux'], capture_output=True, text=True)
            if 'train_model.py' in result.stdout:
                print(f"   🔄 训练进程正在运行")
            else:
                print(f"   ⏸️  训练进程未运行")
        except:
            pass
    else:
        print("\n❌ 未发现训练进度")

def create_directories():
    """创建必要的目录"""
    directories = [
        "data", "data/exports", "analysis/reports", "model"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)

def main():
    parser = argparse.ArgumentParser(description="MoodMirror 智能启动脚本")
    parser.add_argument("--skip-check", action="store_true", help="跳过环境检查")
    parser.add_argument("--check-training", action="store_true", help="检查训练进度")
    parser.add_argument("--background-train", action="store_true", help="后台训练模式")
    parser.add_argument("--auto-launch", action="store_true", help="自动启动应用")
    args = parser.parse_args()
    
    print("🎭 MoodMirror: AI情绪日记 v2.0")
    print("=" * 50)
    
    # 仅检查训练进度
    if args.check_training:
        check_training_progress()
        return 0
    
    # 创建必要目录
    create_directories()
    
    # 检查环境
    if not args.skip_check:
        if not check_dependencies():
            print("\n❌ 环境检查失败，请安装缺少的依赖包")
            return 1
        
        if not check_dataset():
            print("\n❌ 数据集检查失败，请确保dataset文件夹包含正确的数据")
            return 1
    
    # 检查模型状态
    model_status = check_model_status()
    
    # 自动启动模式
    if args.auto_launch:
        if model_status['has_best'] or model_status['has_latest']:
            return 0 if launch_app() else 1
        else:
            print("\n❌ 没有可用模型，无法自动启动")
            return 1
    
    # 后台训练模式
    if args.background_train:
        return 0 if start_training_background() else 1
    
    # 交互模式
    choice = ask_user_choice(model_status)
    
    if choice == '1':  # 启动应用
        return 0 if launch_app() else 1
    elif choice == '2':  # 继续训练
        return 0 if start_training_interactive() else 1
    elif choice == '3':  # 重新训练
        return 0 if start_training_interactive() else 1
    elif choice == '4':  # 仅检查环境
        print("\n✅ 环境检查完成")
        return 0
    elif choice == '5':  # 退出
        print("\n👋 再见!")
        return 0

if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n👋 再见!")
        sys.exit(0)
    except Exception as e:
        print(f"\n💥 意外错误: {e}")
        sys.exit(1) 