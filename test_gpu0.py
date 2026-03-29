import os
import sys

__package__ = "trainer"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
from model.MokioModel import MokioMindConfig
from trainer.trainer_utils import init_model

def test_gpu_inference():
    print("🔍 开始 GPU 模型简单推理测试")
    
    # 检查GPU是否可用
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("⚠️  未检测到 GPU，自动切换到 CPU 模式")
    
    # 模型配置
    lm_config = MokioMindConfig(
        hidden_size=512,
        num_hidden_layers=8,
        use_moe=False,
    )
    
    # 加载模型
    model, tokenizer = init_model(lm_config, "none", device=device)
    
    # 尝试加载训练好的权重
    model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "./out2/pretrain_512_gpu.pth")
    if os.path.exists(model_path):
        print(f"📥 加载权重文件: {model_path}")
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)
    else:
        print("⚠️  未找到训练权重，使用随机初始化模型进行测试")
    
    # 切换到推理模式
    model.eval()
    
    # 测试输入文本
    test_text = "如何提高写作能力"
    print(f"\n📝 测试输入: {test_text}")
    
    # 编码 + 推理（仅一次）
    inputs = tokenizer(test_text, return_tensors="pt").to(device)
    with torch.no_grad():
        output = model.generate(
            inputs.input_ids,
            max_length=150,
            num_return_sequences=1,
            temperature=0.7,
            top_k=50,
            top_p=0.95
        )
    
    # 解码输出结果
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    print(f"\n🤖 生成结果: {generated_text}")
    
    print("\n✅ 简单推理测试完成!")

if __name__ == "__main__":
    test_gpu_inference()