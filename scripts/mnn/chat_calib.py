"""In-domain CALIBRATION text (other requests than the eval set, seed 4321) -- eval text for MNN-Pocket-0.3B: answers of the bf16 model itself (transformers, sampled at
temperature 0.7 / top-p 0.9, fixed seed) to 48 English and Chinese requests, rendered with the model's
chat template and concatenated. Raw WikiText is a poor yardstick for this model (it is very confident
when wrong, ppl ~4700); this text is what it actually produces.

usage: python -I chat_eval.py MODEL_DIR OUT.txt"""
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

EN = ['How does a refrigerator keep food cold?', 'Give me four ideas for a rainy-day activity with kids.', 'Explain what an API is in simple terms.', 'Write a haiku about the sea.', 'What is the difference between a virus and a bacterium?', 'How can I save money on groceries?', 'Describe the water cycle.', 'Write a SQL query that counts orders per customer.', 'What is the Pythagorean theorem?', 'Give three arguments for learning a second language.', 'How do airplanes stay in the air?', 'Write a short thank-you note to a teacher.', 'What is a black hole?', 'Explain the difference between RAM and storage.', 'Extract the date and place as JSON: "The concert takes place on May 3 in Vienna."', 'What should I pack for a weekend hiking trip?']
ZH = ['冰箱是怎样保持食物低温的？', '给我推荐四个适合下雨天的室内活动。', '用通俗的话解释什么是API。', '写一首关于大海的小诗。', '病毒和细菌有什么区别？', '怎样在买菜时省钱？', '描述一下水循环的过程。', '写一个统计每个客户订单数量的SQL查询。', '什么是勾股定理？', '学习第二外语有哪些好处？', '飞机为什么能在天上飞？', '写一张给老师的感谢卡。', '什么是黑洞？', '内存和硬盘有什么区别？', '从这句话中提取日期和地点，用JSON输出：音乐会将于5月3日在杭州举行。', '周末去爬山需要带哪些东西？']

model_dir, out = sys.argv[1], sys.argv[2]
tok = AutoTokenizer.from_pretrained(model_dir)
model = AutoModelForCausalLM.from_pretrained(model_dir, dtype=torch.float32).eval()
torch.manual_seed(4321)
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
