# profile_feed.py URL MODEL N: N code prompts from the calibration corpus (not the eval set) and a set of chat
# questions, each answered with up to 128 tokens: routing for `CMF_QWEN_LAT_PROFILE_SAVE`.
import json, sys, urllib.request
url, model, n = sys.argv[1], sys.argv[2], int(sys.argv[3])
code = open("/root/e0/data/calib_code.txt").read()
chat = ["Explain the difference between a process and a thread.", "Write a SQL query that finds duplicate emails in a users table.",
        "How do I reverse a linked list in C?", "Summarize the causes of the French Revolution in five sentences.",
        "Write a bash script that renames all .jpeg files to .jpg.", "What is the time complexity of quicksort and why?",
        "Translate to French: The meeting has been moved to Thursday afternoon.", "Write a haiku about autumn rain.",
        "Explain how HTTPS protects data in transit.", "Write a Rust function that parses a CSV line with quoted fields.",
        "Give me a weekly workout plan for a beginner.", "What are closures in JavaScript? Show an example.",
        "Draft a polite email asking for a deadline extension.", "Explain gradient descent to a high school student.",
        "Write a Python decorator that retries a function three times.", "Compare TCP and UDP.",
        "Напиши функцию на Python, которая считает частоту слов в тексте.", "Объясни, что такое рекурсия, на простом примере.",
        "用Python写一个二分查找函数。", "解释一下什么是数据库索引。", "Write unit tests for a function that validates IPv4 addresses.",
        "Refactor this to be more readable: def f(a):return [x*2 for x in a if x%2==0 and x>3]",
        "Explain the CAP theorem with examples.", "Write a Go HTTP server that returns the current time as JSON."]
def ask(msg):
    body = {"model": model, "temperature": 0, "max_tokens": 128, "enable_thinking": False,
            "chat_template_kwargs": {"enable_thinking": False}, "messages": [{"role": "user", "content": msg}]}
    req = urllib.request.Request(url.rstrip("/") + "/v1/chat/completions", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    urllib.request.urlopen(req, timeout=3600).read()
step = max(1, (len(code) - 8000) // max(1, n))
for i in range(n):
    ask("Here is some code:\n\n" + code[i * step:i * step + 7000] + "\n\nExplain what it does and suggest one improvement.")
for q in chat:
    ask(q)
print("fed", n, "code prompts and", len(chat), "chat questions")
