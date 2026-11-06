#!/usr/bin/env python3
"""Prompts whose answers train a draft head: q27_mtp_prompts.py WIKI_TRAIN CODE_ANSWERS_JSONL OUT_JSONL [prose|prose3]
With `prose`, a second set instead: essays, explanations, advice and comparisons in English and Russian only.
With `prose3`, a third: every topic of the first two sets in the forms `prose` did not ask it in.
Prose and explanations in English and Russian, code, lists, short questions, work on a text that is in the
prompt (quote, summarize, translate) and edits of code that is in the prompt. The quoted text is wikitext-2
train and the code comes from CODE_ANSWERS_JSONL (rows with `response`), so nothing here repeats the five
prompts of q27_spec_prompts.sh, which measure the result."""
import json, re, sys

wiki, code_answers, out = sys.argv[1:4]
PROSE = len(sys.argv) > 4 and sys.argv[4] in ("prose", "prose3")
PROSE3 = len(sys.argv) > 4 and sys.argv[4] == "prose3"

EN2 = [
    "why the Library of Alexandria mattered", "how coral reefs form and why they bleach", "the Black Death and its effect on wages",
    "how a jet engine produces thrust", "why the Ottoman Empire lasted six centuries", "how money is created by banks",
    "the discovery of penicillin", "how bees make honey", "what the Magna Carta changed", "how tornadoes form",
    "the history of the bicycle", "how the human immune system remembers", "why the Aral Sea dried up",
    "how fibre-optic cables carry data", "the rise and fall of Carthage", "how a refrigerator moves heat",
    "why some countries drive on the left", "how the stock market sets prices", "the invention of the telephone",
    "how deserts form", "what caused the First World War", "how a vaccine is tested before approval",
    "the history of tea", "how volcanoes erupt", "why the dinosaurs died out", "how an electric motor works",
    "the Meiji Restoration in Japan", "how the kidneys filter blood", "what an operating system kernel does",
    "how sailing ships cross against the wind", "the Great Fire of London", "how rainbows form",
    "why the Maya cities were abandoned", "how hash functions protect passwords", "the building of the Panama Canal",
    "how bread rises", "what a central bank does", "how bats navigate in the dark", "the history of the calendar",
    "how noise-cancelling headphones work", "why the Berlin Wall fell", "how a suspension bridge carries its load",
    "the domestication of the horse", "how memory is laid out in a running program", "what the Enlightenment was",
    "how the Moon was formed", "why rivers meander", "how touchscreens sense a finger", "the Hanseatic League",
    "how antibiotics kill bacteria", "why the Sahara was once green", "how a symphony orchestra is organised",
    "the abolition of slavery in the British Empire", "how MRI scanners see inside the body", "what makes soil fertile",
    "how email travels across the internet", "the voyage of Magellan", "how concrete hardens", "why we have leap years",
    "how elections with proportional representation work", "the history of the compass", "how sharks sense prey",
    "what a supply chain is", "how the printing of banknotes prevents forgery", "the Dutch Golden Age",
    "how a microwave oven heats food", "why ice floats", "how bird migration works", "the fall of Constantinople",
    "how version control changed software teams",
]
RU2 = [
    "почему погибла Александрийская библиотека и чем она была важна", "как образуются коралловые рифы",
    "как работает реактивный двигатель", "откуда берутся деньги в банковской системе", "как открыли пенициллин",
    "как пчёлы делают мёд", "как возникают торнадо", "как устроен велосипед и кто его придумал",
    "почему высохло Аральское море", "как данные идут по оптоволокну", "почему в одних странах ездят по левой стороне",
    "как на бирже складывается цена акции", "как появился телефон", "как образуются пустыни",
    "что привело к Первой мировой войне", "как проверяют вакцину перед тем, как разрешить её",
    "как чай попал в Россию", "как извергаются вулканы", "почему вымерли динозавры", "как работает электродвигатель",
    "как почки очищают кровь", "что делает ядро операционной системы", "как парусник идёт против ветра",
    "как возникает радуга", "как хеш-функции защищают пароли", "как строили Транссибирскую магистраль",
    "почему поднимается тесто", "чем занимается центральный банк", "как летучие мыши ориентируются в темноте",
    "как устроен календарь и откуда взялся високосный год", "как работают наушники с шумоподавлением",
    "почему пала Берлинская стена", "как висячий мост держит нагрузку", "как человек приручил лошадь",
    "как устроена память работающей программы", "как образовалась Луна", "почему реки петляют",
    "как сенсорный экран чувствует палец", "как антибиотики убивают бактерии", "как устроен симфонический оркестр",
    "как отменили крепостное право в России", "как томограф видит тело изнутри", "от чего зависит плодородие почвы",
    "как письмо идёт по электронной почте", "как Магеллан обошёл вокруг света", "как твердеет бетон",
    "как работают выборы по пропорциональной системе", "как изобрели компас", "что такое цепочка поставок",
    "как защищают банкноты от подделки", "как микроволновая печь греет еду", "почему лёд не тонет",
    "как птицы находят дорогу при перелётах", "как пал Константинополь", "как системы контроля версий изменили работу программистов",
    "как устроено метро и почему поезда не сталкиваются", "почему небо ночью тёмное", "как работает прививка от гриппа",
    "как Ломоносов повлиял на русскую науку", "как выращивают и обжаривают кофе", "что такое парниковый эффект",
    "как работает лифт и что будет, если оборвётся трос", "как появился русский алфавит", "как делают стекло",
    "почему самолёты летают на высоте десять километров", "как человек научился добывать огонь",
    "как устроена пенсионная система", "что такое иммунитет растений", "как работает турбина гидроэлектростанции",
    "как выбирают имя новому химическому элементу",
]
EN_FORMS2 = [
    ("essay", "Write an essay of several paragraphs about {t}."),
    ("explain", "Explain {t} as you would to a friend who knows nothing about it."),
    ("history", "Tell the story of {t}: what happened, in what order, and why it matters today."),
    ("compare", "Explain {t}, and name two common misconceptions about it."),
]
RU_FORMS2 = [
    ("essay_ru", "Напиши эссе в несколько абзацев о том, {t}."),
    ("explain_ru", "Объясни так, как объяснил бы другу, который ничего об этом не знает: {t}."),
    ("history_ru", "Расскажи по порядку, {t}, и почему это важно сегодня."),
    ("compare_ru", "Объясни, {t}, и назови два распространённых заблуждения на этот счёт."),
]
ADVICE = [
    "How should I prepare for a job interview for a junior developer position?", "What is a sensible way to start running if I have never exercised?",
    "How do I choose between renting and buying a flat? Give the main arguments for each.", "How can I learn a foreign language to a conversational level in a year?",
    "What should a good code review look for? Explain with reasons.", "How do I plan a week of healthy meals on a small budget?",
    "What are the pros and cons of working remotely?", "How do I write a clear bug report?", "How should a small team decide what to build next?",
    "What is the best way to back up family photos, and why?",
    "Как подготовиться к собеседованию на должность младшего разработчика?", "С чего начать бегать, если никогда не занимался спортом?",
    "Как выбрать между арендой и покупкой квартиры? Приведи главные доводы за каждое.", "Как за год выучить иностранный язык до разговорного уровня?",
    "На что смотреть при ревью кода? Объясни почему.", "Как спланировать здоровое питание на неделю при небольшом бюджете?",
    "Какие плюсы и минусы у удалённой работы?", "Как написать понятный отчёт об ошибке?", "Как небольшой команде решить, что делать дальше?",
    "Как лучше хранить семейные фотографии, чтобы не потерять, и почему?",
]

EN_TOPICS = [
    "how vaccines train the immune system", "why the Roman Republic fell", "how a transistor works",
    "the causes of the 2008 financial crisis", "how plate tectonics shapes mountains", "the history of the printing press",
    "why the sky is blue and sunsets are red", "how public-key cryptography works", "the life cycle of a star",
    "how the Silk Road changed trade", "what makes a good database index", "how honeybees decide where to build a hive",
    "the invention of the steam engine", "how glaciers carve valleys", "why antibiotics stop working",
    "how a compiler turns source code into machine code", "the rise of the Mongol Empire", "how photosynthesis stores energy",
    "what inflation is and who it hurts", "how GPS finds your position", "the French Revolution in brief",
    "how neural networks learn by gradient descent", "why oceans are salty", "how the heart pumps blood",
    "the origins of jazz", "how a bill becomes law in a parliament", "what a black hole is",
    "how coffee went from Ethiopia to the world", "why bridges resonate and sometimes fail", "how TCP keeps data in order",
    "the Renaissance and the idea of perspective in painting", "how batteries store electricity", "what causes earthquakes",
    "how the telegraph changed news", "why languages die out", "how garbage collection works in a runtime",
    "the Apollo program and what it took to land on the Moon", "how rice farming shaped Asian societies",
    "what a recession is", "how the eye turns light into signals", "the story of the Rosetta Stone",
    "how containers differ from virtual machines", "why some metals rust", "how tides follow the Moon",
    "the Industrial Revolution and child labour", "how a search engine ranks pages", "what DNA replication is",
    "how medieval cathedrals were built", "why sleep matters for memory", "how wind turbines make power",
]
RU_TOPICS = [
    "как работает иммунитет после прививки", "почему распалась Римская империя", "как устроен транзистор",
    "почему зимой холодно, а летом тепло", "как работает шифрование с открытым ключом", "что такое инфляция",
    "как самолёт держится в воздухе", "как работает поисковая система", "что такое чёрная дыра",
    "как появилась письменность", "почему море солёное", "как растения получают энергию из света",
    "что такое индекс в базе данных", "как устроен двигатель внутреннего сгорания", "что такое рекурсия в программировании",
    "как работает атомная электростанция", "почему возникают землетрясения", "как работает сборщик мусора в языках программирования",
    "как Пётр Первый изменил Россию", "что такое блокчейн", "как работает холодильник",
    "почему Луна вызывает приливы", "как обучается нейронная сеть", "что такое фотоэффект",
    "как устроен интернет и что такое протокол TCP", "почему ржавеет железо", "как работает человеческая память",
    "что такое контейнеры и чем они отличаются от виртуальных машин", "как строили средневековые соборы",
    "что такое естественный отбор", "как компилятор превращает программу в машинный код", "почему важен сон",
    "как работает электронная подпись", "что такое теория относительности простыми словами", "как появились деньги",
    "как работает ветряная электростанция", "что такое многопоточность и гонка данных", "почему листья желтеют осенью",
    "как устроен аккумулятор", "что такое кэш процессора и зачем он нужен",
]
EN_FORMS = [
    ("essay", "Write a short essay about {t}."),
    ("explain", "Explain {t} to a curious beginner in a few paragraphs."),
    ("list", "List the main points about {t} as bullet points with one sentence each."),
    ("steps", "Describe {t} step by step, as a numbered list."),
]
RU_FORMS = [
    ("essay_ru", "Напиши короткое эссе о том, {t}."),
    ("explain_ru", "Объясни простыми словами, {t}. Приведи пример."),
    ("list_ru", "Перечисли главное о том, {t}, в виде списка, по одному предложению на пункт."),
]
CODE_TASKS = [
    ("Python", "merges overlapping intervals in a list"), ("Python", "parses an ISO 8601 date string without libraries"),
    ("Python", "implements an LRU cache as a class"), ("Python", "walks a directory and groups files by extension"),
    ("Python", "finds the longest common subsequence of two strings"), ("Python", "retries a call with exponential backoff"),
    ("Rust", "counts word frequencies in a string and returns the top n"), ("Rust", "implements a ring buffer with push and pop"),
    ("Rust", "validates balanced brackets of three kinds"), ("Rust", "converts a Roman numeral to an integer"),
    ("Rust", "reads key=value lines into a HashMap, skipping comments"), ("Rust", "computes a moving average over a slice"),
    ("Go", "fans out work to n goroutines and collects the results in order"), ("Go", "implements a rate limiter with a token bucket"),
    ("Go", "reverses a singly linked list"), ("Go", "deduplicates a slice while keeping the order"),
    ("TypeScript", "debounces a function"), ("TypeScript", "deep-merges two plain objects"),
    ("TypeScript", "groups an array of objects by a key"), ("TypeScript", "formats a duration in milliseconds as hours, minutes and seconds"),
    ("C++", "implements binary search over a sorted vector"), ("C++", "splits a string by a delimiter into a vector"),
    ("C++", "implements a simple thread-safe queue"), ("Java", "checks whether two strings are anagrams"),
    ("Java", "implements a generic stack backed by an array"), ("Java", "parses a query string into a map"),
    ("C", "reverses the words of a string in place"), ("C", "implements a growable array of ints"),
    ("Bash", "renames every .txt file in a directory to lower case"), ("SQL", "returns the three best-paid employees of every department"),
    ("SQL", "finds customers with no orders in the last 90 days"), ("Kotlin", "flattens a nested list of integers"),
    ("Swift", "computes the edit distance between two strings"), ("Python", "implements Dijkstra's algorithm on an adjacency list"),
    ("Rust", "implements a trie with insert and prefix search"), ("Go", "parses command-line flags into a struct"),
]
SHORT = [
    "What is the difference between a process and a thread?", "Why is the Dead Sea so salty?", "What does HTTP status 409 mean?",
    "Who wrote 'War and Peace' and what is it about, in two sentences?", "How many moons does Mars have, and what are they called?",
    "What is the capital of Australia, and why is it not Sydney?", "Give me three tips for writing clear commit messages.",
    "What is Big O notation, briefly?", "How do I undo the last git commit but keep the changes?", "What is the boiling point of water at altitude?",
    "В чём разница между стеком и кучей?", "Сколько планет в Солнечной системе и какая самая большая?",
    "Что означает код ответа HTTP 502?", "Как отменить последний коммит в git, сохранив изменения?",
    "Чем отличается компилятор от интерпретатора?", "Кто написал «Мастера и Маргариту» и о чём эта книга, в двух предложениях?",
    "Дай три совета, как писать понятные сообщения коммитов.", "Что такое хеш-таблица, коротко?",
    "Почему нельзя делить на ноль?", "Что такое индекс массы тела и как его считать?",
    "Write a JSON object describing a book with title, author, year and three tags.",
    "Make a markdown table comparing arrays, linked lists and hash maps by lookup, insert and memory.",
    "Составь таблицу в markdown: три языка программирования, год появления и главное применение.",
    "Write a polite email declining a meeting invitation because of a schedule conflict.",
    "Напиши вежливое письмо с просьбой перенести встречу на следующую неделю.",
    "Translate into Russian: 'The quick brown fox jumps over the lazy dog, and nobody is surprised any more.'",
    "Переведи на английский: «Утро вечера мудренее, но откладывать всё на завтра тоже не стоит.»",
    "Write a haiku about a server that will not start.", "Explain what a pull request is to a non-programmer.",
    "Что такое рекурсия? Объясни на примере матрёшки.",
]
TEXT_TASKS = [
    ("quote", "Quote the two most important sentences of the text above word for word, then explain why they matter."),
    ("summary", "Summarize the text above in three sentences."),
    ("facts", "List every date, number and name that the text above mentions, each with what it refers to."),
    ("rewrite", "Rewrite the first paragraph of the text above in simpler words, keeping every fact."),
    ("ru", "Перескажи текст выше по-русски в четырёх предложениях."),
    ("qa", "Based only on the text above, what is its main subject and what happened to it? Quote the sentence that supports your answer."),
]
EDITS = [
    "Add a short doc comment to every function and return the full code.",
    "Rename the variables so that their names say what they hold, and return the full code.",
    "Add error handling for invalid input and return the full code.",
    "Add logging at the start and the end of the main function and return the full code.",
    "Split the longest function into two smaller ones and return the full code.",
    "Add one more unit test for an edge case and return the full code.",
]


def passages(path, n, size=2400):
    text = open(path, encoding="utf-8").read()
    paras = [p.strip() for p in text.split("\n") if len(p.strip()) > 400 and not p.strip().startswith("=")]
    step = max(1, len(paras) // n)
    out = []
    for i in range(0, len(paras), step):
        chunk, j = "", i
        while len(chunk) < size and j < len(paras):
            chunk += paras[j] + "\n\n"
            j += 1
        out.append(chunk[:size].rsplit(".", 1)[0] + ".")
        if len(out) == n:
            break
    return out


def code_blocks(path, n):
    out = []
    for line in open(path, encoding="utf-8"):
        m = re.search(r"```(\w+)\n(.*?)```", json.loads(line).get("response", ""), re.S)
        if not m:
            continue
        lang, body = m.group(1), m.group(2).rstrip()
        if 12 <= body.count("\n") <= 45:
            out.append((lang, body))
    step = max(1, len(out) // n)
    return out[::step][:n]


rows = []
add = lambda cat, i, prompt: rows.append({"id": f"{cat}_{i:03d}", "category": cat, "prompt": prompt})
if PROSE3:
    n = 0
    for topics, forms, taken in ((EN2, EN_FORMS2, True), (RU2, RU_FORMS2, True), (EN_TOPICS, EN_FORMS2, False), (RU_TOPICS, RU_FORMS2, False)):
        for i, t in enumerate(topics):
            for f, (cat, form) in enumerate(forms):
                if not (taken and f == i % len(forms)):
                    add(cat + "3", n, form.format(t=t))
                    n += 1
    EN_TOPICS, RU_TOPICS, CODE_TASKS, SHORT = [], [], [], []
elif PROSE:
    for i, t in enumerate(EN2):
        cat, form = EN_FORMS2[i % len(EN_FORMS2)]
        add(cat + "2", i, form.format(t=t))
    for i, t in enumerate(RU2):
        cat, form = RU_FORMS2[i % len(RU_FORMS2)]
        add(cat + "2", i, form.format(t=t))
    for i, q in enumerate(ADVICE):
        add("advice2", i, q)
    EN_TOPICS, RU_TOPICS, CODE_TASKS, SHORT = [], [], [], []
for i, t in enumerate(EN_TOPICS):
    cat, form = EN_FORMS[i % len(EN_FORMS)]
    add(cat, i, form.format(t=t))
for i, t in enumerate(RU_TOPICS):
    cat, form = RU_FORMS[i % len(RU_FORMS)]
    add(cat, i, form.format(t=t))
for i, (lang, task) in enumerate(CODE_TASKS):
    tail = " Then write unit tests for it." if i % 2 == 0 else " Explain the approach in two sentences first."
    add("code", i, f"Write a {lang} function that {task}." + tail if lang not in ("SQL", "Bash") else f"Write a {lang} script that {task}. Explain it briefly.")
for i, q in enumerate(SHORT):
    add("short", i, q)
for i, p in enumerate([] if PROSE else passages(wiki, 60)):
    cat, task = TEXT_TASKS[i % len(TEXT_TASKS)]
    add("text_" + cat, i, p + "\n\n" + task)
for i, (lang, body) in enumerate([] if PROSE else code_blocks(code_answers, 60)):
    add("edit", i, f"Here is some {lang} code:\n\n```{lang}\n{body}\n```\n\n" + EDITS[i % len(EDITS)])
with open(out, "w", encoding="utf-8") as f:
    for r in rows:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
cats = {}
for r in rows:
    cats[r["category"]] = cats.get(r["category"], 0) + 1
print(len(rows), "prompts:", ", ".join(f"{k} {v}" for k, v in sorted(cats.items())))
