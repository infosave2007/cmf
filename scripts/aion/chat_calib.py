"""Chat-format calibration text: the model's own greedy answers (HF bf16) to requests of its
use cases, rendered with its template.  python3 -I chat_calib.py MODEL_DIR OUT_TXT"""
import sys, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
md, out = sys.argv[1], sys.argv[2]
tok = AutoTokenizer.from_pretrained(md, trust_remote_code=True)
m = AutoModelForCausalLM.from_pretrained(md, trust_remote_code=True, dtype=torch.bfloat16).cuda().eval()
P = [
 "Summarize in three bullet points: Remote work has changed how companies hire. Firms now recruit across time zones, cut office costs, and rely on written communication, but they struggle with onboarding and team cohesion.",
 "Rewrite this email to sound more polite: 'Send me the report by Friday. You are already late.'",
 "Explain the difference between RAM and storage to a beginner.",
 "Write a short product description for a reusable water bottle.",
 "What are the main causes of inflation? Keep it brief.",
 "Turn these notes into a paragraph: meeting moved to Tuesday; budget approved; need two more testers; release in March.",
 "Give me five tips for better sleep.",
 "Translate into German: The train to Berlin leaves at eight in the morning.",
 "Traduis en français : Our team will present the results next week.",
 "Explica en español qué es la fotosíntesis en dos frases.",
 "用中文简要介绍一下机器学习。",
 "日本語で自己紹介を短く書いてください。",
 "Write a Python function that removes duplicates from a list while keeping the order.",
 "Write a SQL query that returns the ten customers with the highest total order value.",
 "Explain what this does: `grep -rn TODO src | wc -l`",
 "Write a JavaScript function that debounces another function.",
 "Write a bash script that renames all .jpeg files in a folder to .jpg.",
 "What is the time complexity of binary search and why?",
 "Fix the grammar: 'Me and him goes to the store yesterday and buyed apples.'",
 "Shorten this sentence: 'Due to the fact that the weather conditions were not favorable, the event was postponed to a later date.'",
 "Write a haiku about autumn rain.",
 "List the planets of the solar system in order from the Sun.",
 "Solve step by step: a shirt costs $40 and is discounted by 15%. What is the new price?",
 "How do I make a strong password? Give practical advice.",
 "Write a polite reply declining a meeting invitation.",
 "Classify the sentiment of: 'The update made my laptop slower and the battery drains fast.'",
 "Extract the dates from: 'The contract starts on 3 March 2026 and ends on 31 December 2027.'",
 "Explain what a REST API is with a simple example.",
 "Write a Rust function that returns the largest element of a slice.",
 "Describe the water cycle for a ten-year-old.",
]
texts = []
for q in P:
    msgs = [{"role": "user", "content": q}]
    ids = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt", current_date="2026-11-20", return_dict=True)["input_ids"].cuda()
    with torch.no_grad():
        g = m.generate(ids, max_new_tokens=400, do_sample=False)
    a = tok.decode(g[0, ids.shape[1]:], skip_special_tokens=True).strip()
    msgs.append({"role": "assistant", "content": a})
    texts.append(tok.apply_chat_template(msgs, tokenize=False, current_date="2026-11-20"))
    print(len(texts), len(a), flush=True)
open(out, "w", encoding="utf-8").write("\n".join(texts))
