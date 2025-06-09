from transformers import TextStreamer, AutoTokenizer, AutoModelForCausalLM
import torch
import time

# 检查IPEX是否可用
try:
    import intel_extension_for_pytorch as ipex
    print(f"IPEX {ipex.__version__} 可用")
    ipex_available = True
except ImportError:
    print("警告: IPEX不可用，将在标准CPU上运行")
    ipex_available = False

# 模型路径配置
MODEL_PATH = "/mnt/data/Qwen-7B-Chat"

# 测试问题集
TEST_PROMPTS = {
    "问题1 (冬天夏天)": "请说出以下两句话区别在哪里? 1、冬天:能穿多少穿多少 2、夏天:能穿多少穿多少",
    "问题2 (单身狗)": "请说出以下两句话区别在哪里?单身狗产生的原因有两个，一是谁都看不上，二是谁都看不上",
    "问题3 (谁不知道)": "他知道我知道你知道他不知道吗? 这句话里,到底谁不知道",
    "问题4 (明明白白)": "明明明明明白白白喜欢他,可她就是不说。 这句话里,明明和白白谁喜欢谁?",
    "问题5 (意思)": "领导:你这是什么意思? 小明:没什么意思。意思意思。 领导:你这就不够意思了。 小明:小意思,小意思。领导:你这人真有意思。 小明:其实也没有别的意思。 领导:那我就不好意思了。 小明:是我不好意思。请问:以上\"意思\"分别是什么意思。",
    "问题6 (数学逻辑)": "一个水池有两个进水口A和B。A单独注满水池需要6小时，B单独注满需要4小时。水池底部有一个排水口C，单独排空满池水需要3小时。如果同时打开A、B、C三个口，需要多少小时才能注满水池？请分步骤解释计算过程。",
    "问题7 (科技伦理)": "近年来人工智能发展迅速，有人认为AI最终会超越人类智能并威胁人类生存。你如何看待这种观点？请从技术发展现状、理论可能性和伦理规范三个角度分析。",
    "问题8 (创作能力)": "请创作一首李白风格的古诗。"
}

# 打印环境信息
print(f"PyTorch版本: {torch.__version__}, CUDA可用: {torch.cuda.is_available()}")

# 加载模型
print(f"加载模型: {MODEL_PATH}")
start_time = time.time()

try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, 
        trust_remote_code=True,
        torch_dtype="auto",
        device_map="cpu"
    ).eval()
    print(f"模型加载成功，耗时: {time.time()-start_time:.2f}秒")
except Exception as e:
    print(f"加载失败: {e}")
    exit()

# 应用IPEX优化
if ipex_available:
    try:
        print("应用IPEX优化...")
        model = ipex.optimize(model, dtype=torch.bfloat16)
    except Exception as e:
        print(f"IPEX优化失败: {e}")

# 设置流式输出
streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id

# 测试模型
print(f"\n开始测试{len(TEST_PROMPTS)}个问题...")
total_start = time.time()

for name, prompt in TEST_PROMPTS.items():
    print(f"\n【{name}】{prompt}")
    print("模型回答:")
    
    start_time = time.time()
    inputs = tokenizer(prompt, return_tensors="pt").input_ids.to("cpu")
    
    try:
        with torch.no_grad():
            model.generate(
                input_ids=inputs,
                streamer=streamer,
                max_new_tokens=7680,
                do_sample=True,
                temperature=0.7,
                top_p=0.8,
                pad_token_id=tokenizer.pad_token_id
            )
        print(f"\n回答耗时: {time.time()-start_time:.2f}秒")
    except Exception as e:
        print(f"生成错误: {e}")

print(f"\n所有问题测试完成，总耗时: {(time.time()-total_start)/60:.2f}分钟")