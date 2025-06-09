# 大语言模型评估项目

## 项目概述
本项目包含了对两个大语言模型的评估和部署测试结果：智谱AI的ChatGLM和阿里巴巴的通义千问。项目内容包括部署流程、测试结果以及两个模型的对比分析。

## 目录结构
```
LLM-Evaluation/
├── LLM-Deployment-Test-Report.pdf    # 综合测试报告
├── Model-Deployment/                 # 模型部署相关文件
│   ├── Platform-Startup.png         # 平台启动截图
│   ├── Qwen-Git-Clone.png          # 千问模型克隆过程
│   ├── Zhipu-Git-Clone.png         # 智谱模型克隆过程
│   ├── Download-Complete.png        # 模型下载完成
│   └── Dependencies-Download-Complete.png  # 依赖安装完成
├── Running-Code/                    # 运行代码
│   ├── run_qwen_cpu.py             # 千问模型CPU运行脚本
│   └── run_glm_cpu.py              # ChatGLM CPU运行脚本
├── Qwen-Answers/                    # 千问模型测试结果
│   └── Question-1.png 到 Question-8.png  # 测试用例截图
└── Zhipu-Answers/                   # 智谱模型测试结果
    └── Question-1.png 到 Question-8.png  # 测试用例截图
```

## 测试模型
1. **智谱AI ChatGLM**
   - CPU部署方式
   - 本地模型运行能力
   - 包含测试用例和响应结果

2. **阿里巴巴通义千问**
   - CPU部署方式
   - 本地模型运行能力
   - 包含测试用例和响应结果

## 测试用例
本项目为每个模型包含8个测试用例，涵盖了大语言模型的各个方面：
- 问题一：基础语言理解
- 问题二：复杂推理
- 问题三：代码生成
- 问题四：数学问题求解
- 问题五：文本摘要
- 问题六：创意写作
- 问题七：技术文档
- 问题八：多轮对话

## 运行模型
1. 进入 `Running-Code` 目录
2. 选择要运行的模型：
   - 运行千问：`python run_qwen_cpu.py`
   - 运行ChatGLM：`python run_glm_cpu.py`

## 系统要求
- Python 3.8+
- 相关依赖包（详见各模型要求）
- 足够的CPU资源用于本地模型运行

## 测试报告
详细的测试报告可在 `LLM-Deployment-Test-Report.pdf` 中查看，内容包括：
- 部署流程
- 性能指标
- 对比分析
- 测试结果
- 建议和总结

## 注意事项
- 所有模型均在仅CPU模式下测试
- 测试结果以截图形式保存以供验证
- 部署流程配有截图说明

## 许可证
本项目仅用于评估目的。具体使用权限请参考各模型的许可证说明。

 