"""In-domain eval text for MNN-Pocket-0.3B: answers of the bf16 model itself (transformers, sampled at
temperature 0.7 / top-p 0.9, fixed seed) to 48 English and Chinese requests, rendered with the model's
chat template and concatenated. Raw WikiText is a poor yardstick for this model (it is very confident
when wrong, ppl ~4700); this text is what it actually produces.

usage: python -I chat_eval.py MODEL_DIR OUT.txt"""
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

EN = [
    "What causes the seasons on Earth?", "Give three tips for writing a clear email.",
    "Explain what a hash table is to a beginner.", "Summarize the plot of Romeo and Juliet in four sentences.",
    "What is the difference between weather and climate?", "Write a short poem about autumn rain.",
    'Extract the product and price as JSON: "The new Aurora lamp costs 49 dollars."',
    "How do vaccines train the immune system?", "List five healthy breakfast ideas.",
    "Translate into French: I would like a cup of coffee, please.", "Why is the sky blue?",
    "Write a Python function that counts vowels in a string.", "What is photosynthesis?",
    "Give a one-paragraph history of the printing press.", "How should I prepare for a job interview?",
    "Explain recursion with a simple example.", "What are the main causes of inflation?",
    "Write a polite message declining a meeting invitation.", "What is the boiling point of water at sea level?",
    "Describe how a bicycle gear system works.", "What does an operating system do?",
    "Compare cats and dogs as pets in a few sentences.", "Explain the rule of thirds in photography.",
    "Classify the sentiment of: \"The service was slow but the food was wonderful.\"",
]
ZH = [
    "为什么会有四季变化？", "给我三个提高睡眠质量的建议。", "用简单的话解释什么是区块链。",
    "写一首关于月亮的短诗。", "请把下面的句子翻译成英文：今天天气很好，我们去公园散步吧。",
    "介绍一下长城的历史。", "如何学习一门新的编程语言？", "什么是人工智能？", "推荐五本适合初学者阅读的书。",
    "解释一下光合作用的过程。", "写一段请假邮件，原因是身体不舒服。", "火锅的起源是什么？",
    "从这句话中提取人名和城市，用JSON输出：张伟上个月从成都搬到了上海。", "如何保持良好的学习习惯？",
    "简单介绍一下太阳系的行星。", "用Python写一个判断素数的函数。", "为什么要多喝水？", "解释一下什么是通货膨胀。",
    "描述一下你理想中的周末。", "茶和咖啡哪个更健康？", "怎样做一道番茄炒蛋？", "什么是机器学习中的过拟合？",
    "给一家新开的面包店写一句广告语。", "判断这句话的情感倾向：这部电影情节拖沓，但演员表演很出色。",
]

model_dir, out = sys.argv[1], sys.argv[2]
tok = AutoTokenizer.from_pretrained(model_dir)
model = AutoModelForCausalLM.from_pretrained(model_dir, dtype=torch.float32).eval()
torch.manual_seed(1234)
texts = []
for q in [p for pair in zip(EN, ZH) for p in pair]:
    msgs = [{"role": "user", "content": q}]
    prompt = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
    x = tok(prompt, return_tensors="pt", add_special_tokens=False)
    with torch.no_grad():
        o = model.generate(**x, max_new_tokens=256, do_sample=True, temperature=0.7, top_p=0.9)
    ans = tok.decode(o[0][x["input_ids"].shape[1]:], skip_special_tokens=True).strip()
    texts.append(tok.apply_chat_template(msgs + [{"role": "assistant", "content": ans}], tokenize=False))
    print(f"{len(texts):2d} {len(o[0]) - x['input_ids'].shape[1]:3d} tokens  {q[:40]}", flush=True)
open(out, "w", encoding="utf-8").write("".join(texts))
print("wrote", out, sum(len(tok(t, add_special_tokens=False)["input_ids"]) for t in texts), "tokens")
