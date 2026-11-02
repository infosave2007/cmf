"use strict";
const ASSETS = "https://huggingface.co/infosave/Embryo-56M-cmf/resolve/main/assets/";

const T = {
  en: {
    tagline: "A 56-million-parameter language model in its own architecture and file format, born from scratch on one laptop GPU.",
    note: "This is the first experimental version of a model built entirely on the technologies of Oleg Kirichenko: architecture, file format, router, runtime and trainer. Larger models for different tasks and the mechanisms for training models will follow.",
    linkModel: "Model", tabTests: "Tests", tabChat: "Chat", tabHow: "How it works", tabMeasured: "Measured",
    loading: "The model is loading, it takes a few seconds.",
    runAll: "Run all tests", stop: "Stop", download: "Download results (JSON)",
    filterAll: "All", filterMissed: "Missed only",
    testsFoot: "The questions are fixed-seed samples of the release gates, nothing is picked by hand, so a miss here is a real miss. Results stay in your browser. This Space runs on a free 2-core CPU: speed here is not the speed of a GPU.",
    placeholder: "Ask about a medicinal plant, a country, an element, or just talk…",
    send: "Send", reset: "New dialog",
    chatFoot: "Answers from a record are word for word. Outside the records this is a very small network: grammatical, short, often empty. Do not use it for medical decisions.",
    chatEmpty: "Ask something, or start with an example above. Under each answer you can see which record answered and why.",
    capRouting: "How a message finds its answer: the record with the smallest reconstruction error, or an honest refusal, or the network.",
    capArch: "One token through the network. All memory of the text sits in blocks of fixed size.",
    capAdv: "What this gives. The KV-cache line is calculated, everything else is measured.",
    mSpeed: "Speed, tokens per second", mDevice: "Device", mDecode: "Decode", mDeep: "After 2000 tokens", mPrefill: "Prefill",
    mSpeedNote: "On an Apple M4 a run of 3000 tokens averages 389 tok/s on the GPU and 398 on the CPU: the GPU slows down after a few seconds of continuous work.",
    mQuality: "Quality of the release file", mCheck: "Check", mResult: "Result",
    mQualityNote: "CPU and GPU give the same results. “Held-out” means the phrasing or name was not used in the fit.",
    mPatents: "U.S. patent applications filed with the USPTO", mMethod: "Method", mApp: "Application", mFiled: "Filed",
    mLimits: "Apache-2.0. Patent applications, not granted patents. Details: PATENTS.md in the repository.",
    footer: "Embryo-56M · Apache-2.0 · runs on cortiq, one file of 91 MB",
    pass: "passed", miss: "missed", info: "info",
    of: "of", questions: "questions", dialogs: "dialogs", run: "Run",
    results: "Results", sPassed: "passed", sMissed: "missed", sTime: "median answer time", sRecord: "median time from a record", sSpeed: "network speed on this CPU",
    question: "Question", answer: "Answer",
    you: "You", model: "Embryo",
    whyShow: "why", whyTitle: "Decision of the router", eRecord: "this record", eBase: "the network", eChat: "ordinary chat", eFollowup: "unrelated",
    key: "key", field: "field", entry: "entry", turnsBack: "turns back", expected: "expected", got: "got",
    srcNetwork: "network", srcRefusal: "refusal", noDecision: "No key in the message: no reconstruction error was computed for a record.",
    busy: "The Space is busy, try again in a moment.", failed: "The request failed",
    expRecord: "answer from the record", expFollowup: "the second turn stays on the same record", expNotFollowup: "the second turn leaves the record", expRefusal: "honest refusal", expNotRefusal: "no refusal", expSpeed: "network writes, speed is measured",
    sets: {
      plants: ["Medicinal plants", "Plain questions about popular plants, English and Russian."],
      world: ["World facts", "Countries, elements, science, people, books, places, how-to, in phrasing left out of the fit."],
      followups: ["Follow-ups", "Two-turn dialogs: a question about the plant, or a message that has nothing to do with it."],
      refusal: ["Honest refusal", "Facts the file does not have must be refused, ordinary requests must not."],
      network: ["Network and speed", "No records here: the small network writes, and its speed on this CPU is measured."],
    },
    records: { herbs: "plants", countries: "countries", elements: "elements", people: "people", books: "books", self: "about itself", trivia: "trivia", places: "places", science: "science", howto: "how-to" },
    examples: ["What is valerian used for?", "Is it safe?", "What is the capital of Japan?", "Why is the sky blue?", "What is 2+2?", "Write a poem about the sea."],
    quality: [["Plain questions about plants", "178 / 180"], ["The same, held-out names", "127 / 180"], ["Plain questions, second set", "172 / 180"], ["The same, held-out names", "155 / 180"], ["Plant facts, held-out questions", "218 / 223"], ["World questions", "405 / 445"], ["The same, held-out", "367 / 445"], ["First turn of a dialog", "100 %"], ["Follow-ups in plain words", "900 / 1200"], ["Unrelated messages taken for a follow-up", "41 / 1380"], ["Facts outside the records, refused", "38 / 44"], ["Ordinary requests refused by mistake", "1 / 40"]],
    patents: [["Resonance routing", "No. 19/452,440", "January 19, 2026"], ["Unified execution", "No. 19/731,402", "July 6, 2026"], ["Constant-memory attention", "No. 19/738,763", "July 13, 2026"]],
  },
  ru: {
    tagline: "Языковая модель на 56 миллионов параметров в собственной архитектуре и собственном формате файла, рождена с нуля на GPU одного ноутбука.",
    note: "Это первый экспериментальный вариант модели, полностью основанной на технологиях Олега Кириченко: архитектура, формат файла, роутер, рантайм и обучение. В дальнейшем выйдут более крупные модели для различных задач и механизмы обучения моделей.",
    linkModel: "Модель", tabTests: "Тесты", tabChat: "Чат", tabHow: "Как устроена", tabMeasured: "Измерения",
    loading: "Модель загружается, это займёт несколько секунд.",
    runAll: "Запустить все тесты", stop: "Остановить", download: "Скачать результаты (JSON)",
    filterAll: "Все", filterMissed: "Только промахи",
    testsFoot: "Вопросы взяты из проверок релиза выборкой с фиксированным зерном, ничего не отобрано вручную, поэтому промах здесь настоящий. Результаты остаются в вашем браузере. Этот Space работает на бесплатном CPU с 2 ядрами: скорость здесь не равна скорости GPU.",
    placeholder: "Спросите о лекарственном растении, стране, элементе или просто поговорите…",
    send: "Отправить", reset: "Новый диалог",
    chatFoot: "Ответы из записей дословные. Вне записей это очень маленькая сеть: грамматично, коротко, часто пусто. Не используйте её для медицинских решений.",
    chatEmpty: "Задайте вопрос или начните с примера выше. Под каждым ответом видно, какая запись ответила и почему.",
    capRouting: "Как сообщение находит ответ: запись с наименьшей ошибкой реконструкции, честный отказ или сеть.",
    capArch: "Один токен через сеть. Вся память о тексте лежит в блоках постоянного размера.",
    capAdv: "Что это даёт. Линия KV-кеша рассчитана, всё остальное измерено.",
    mSpeed: "Скорость, токенов в секунду", mDevice: "Устройство", mDecode: "Декодирование", mDeep: "После 2000 токенов", mPrefill: "Префилл",
    mSpeedNote: "На Apple M4 при 3000 токенах подряд GPU в среднем даёт 389 ток/с, CPU 398: после нескольких секунд непрерывной работы GPU замедляется.",
    mQuality: "Качество релизного файла", mCheck: "Проверка", mResult: "Результат",
    mQualityNote: "CPU и GPU дают одинаковые результаты. «Отложенные» значит, что формулировка или название не участвовали в подгонке.",
    mPatents: "Заявки на патенты США, поданные в USPTO", mMethod: "Метод", mApp: "Заявка", mFiled: "Подана",
    mLimits: "Apache-2.0. Это заявки, а не выданные патенты. Подробности: PATENTS.md в репозитории.",
    footer: "Embryo-56M · Apache-2.0 · работает на cortiq, один файл на 91 МБ",
    pass: "верно", miss: "промах", info: "замер",
    of: "из", questions: "вопросов", dialogs: "диалогов", run: "Запустить",
    results: "Результаты", sPassed: "верно", sMissed: "промахов", sTime: "медиана времени ответа", sRecord: "медиана ответа из записи", sSpeed: "скорость сети на этом CPU",
    question: "Вопрос", answer: "Ответ",
    you: "Вы", model: "Embryo",
    whyShow: "почему", whyTitle: "Решение роутера", eRecord: "эта запись", eBase: "сеть", eChat: "обычный чат", eFollowup: "посторонняя",
    key: "ключ", field: "поле", entry: "запись №", turnsBack: "реплик назад", expected: "ожидалось", got: "получено",
    srcNetwork: "сеть", srcRefusal: "отказ", noDecision: "В сообщении нет ключа: ошибка реконструкции для записей не считалась.",
    busy: "Space занят, попробуйте через минуту.", failed: "Запрос не выполнен",
    expRecord: "ответ из записи", expFollowup: "вторая реплика остаётся на той же записи", expNotFollowup: "вторая реплика уходит от записи", expRefusal: "честный отказ", expNotRefusal: "без отказа", expSpeed: "пишет сеть, измеряется скорость",
    sets: {
      plants: ["Лекарственные растения", "Простые вопросы о популярных растениях, на английском и русском."],
      world: ["Факты о мире", "Страны, элементы, наука, люди, книги, места, советы, в формулировках вне подгонки."],
      followups: ["Уточнения", "Диалоги из двух реплик: вопрос о растении или реплика, не связанная с ним."],
      refusal: ["Честный отказ", "На факты, которых нет в файле, нужен отказ, на обычные просьбы не нужен."],
      network: ["Сеть и скорость", "Без записей: пишет маленькая сеть, измеряется её скорость на этом CPU."],
    },
    records: { herbs: "растения", countries: "страны", elements: "элементы", people: "люди", books: "книги", self: "о себе", trivia: "факты", places: "места", science: "наука", howto: "советы" },
    examples: ["Для чего применяют валериану?", "Какие есть противопоказания?", "Какая столица Австралии?", "Что такое фотосинтез?", "Сколько будет 2+2?", "Напиши стихотворение о море."],
    quality: [["Простые вопросы о растениях", "178 / 180"], ["То же, отложенные названия", "127 / 180"], ["Простые вопросы, второй набор", "172 / 180"], ["То же, отложенные названия", "155 / 180"], ["Факты о растениях, отложенные вопросы", "218 / 223"], ["Вопросы о мире", "405 / 445"], ["То же, отложенные", "367 / 445"], ["Первая реплика диалога", "100 %"], ["Уточнения простыми словами", "900 / 1200"], ["Посторонние реплики, принятые за уточнение", "41 / 1380"], ["Факты вне записей, честный отказ", "38 / 44"], ["Обычные просьбы, отказ по ошибке", "1 / 40"]],
    patents: [["Резонансный роутинг", "№ 19/452,440", "19 января 2026"], ["Единое исполнение", "№ 19/731,402", "6 июля 2026"], ["Внимание с постоянной памятью", "№ 19/738,763", "13 июля 2026"]],
  },
  zh: {
    tagline: "一个 5600 万参数的语言模型，采用自有架构和自有文件格式，在一台笔记本的 GPU 上从零诞生。",
    note: "这是第一个实验版本的模型，完全基于 Oleg Kirichenko 的技术：架构、文件格式、路由器、运行时和训练器。之后将发布面向不同任务的更大模型以及模型训练机制。",
    linkModel: "模型", tabTests: "测试", tabChat: "对话", tabHow: "工作原理", tabMeasured: "实测",
    loading: "模型正在加载，需要几秒钟。",
    runAll: "运行全部测试", stop: "停止", download: "下载结果（JSON）",
    filterAll: "全部", filterMissed: "只看未通过",
    testsFoot: "问题来自发布检查，按固定随机种子抽样，没有人工挑选，因此这里的未通过是真实的。结果只保存在你的浏览器中。此 Space 运行在免费的 2 核 CPU 上，这里的速度不代表 GPU 速度。",
    placeholder: "询问药用植物、国家、元素，或者随便聊聊……",
    send: "发送", reset: "新对话",
    chatFoot: "来自记录的回答是逐字的。记录之外它只是一个很小的网络：语法正确，回答简短，常常没有实质内容。请勿用于医疗决策。",
    chatEmpty: "提一个问题，或从上面的示例开始。每个回答下方都能看到由哪条记录回答以及原因。",
    capRouting: "消息如何找到答案：重建误差最小的记录、如实拒答，或由网络生成。",
    capArch: "一个 token 穿过网络。关于文本的全部记忆都放在固定大小的块中。",
    capAdv: "带来什么。KV 缓存那条线是计算值，其余均为实测。",
    mSpeed: "速度，token/秒", mDevice: "设备", mDecode: "解码", mDeep: "2000 token 之后", mPrefill: "预填充",
    mSpeedNote: "在 Apple M4 上连续生成 3000 个 token，GPU 平均 389 tok/s，CPU 398：GPU 连续工作几秒后会变慢。",
    mQuality: "发布文件的质量", mCheck: "检查项", mResult: "结果",
    mQualityNote: "CPU 和 GPU 结果相同。“留出”表示该措辞或名称未用于拟合。",
    mPatents: "已向 USPTO 提交的美国专利申请", mMethod: "方法", mApp: "申请号", mFiled: "提交日期",
    mLimits: "Apache-2.0。这些是专利申请，不是已授权专利。详情见仓库中的 PATENTS.md。",
    footer: "Embryo-56M · Apache-2.0 · 运行于 cortiq，单个 91 MB 文件",
    pass: "通过", miss: "未通过", info: "测量",
    of: "/", questions: "个问题", dialogs: "个对话", run: "运行",
    results: "结果", sPassed: "通过", sMissed: "未通过", sTime: "回答时间中位数", sRecord: "记录回答时间中位数", sSpeed: "此 CPU 上的网络速度",
    question: "问题", answer: "回答",
    you: "你", model: "Embryo",
    whyShow: "原因", whyTitle: "路由器的决定", eRecord: "这条记录", eBase: "网络", eChat: "普通聊天", eFollowup: "无关",
    key: "键", field: "字段", entry: "条目", turnsBack: "轮之前", expected: "预期", got: "实际",
    srcNetwork: "网络", srcRefusal: "拒答", noDecision: "消息中没有键：未计算记录的重建误差。",
    busy: "Space 正忙，请稍后再试。", failed: "请求失败",
    expRecord: "由记录回答", expFollowup: "第二轮仍在同一条记录上", expNotFollowup: "第二轮离开该记录", expRefusal: "如实拒答", expNotRefusal: "不拒答", expSpeed: "由网络生成，测量速度",
    sets: {
      plants: ["药用植物", "关于常见植物的简单问题，英语和俄语。"],
      world: ["世界知识", "国家、元素、科学、人物、书籍、地点、方法，措辞未用于拟合。"],
      followups: ["追问", "两轮对话：关于该植物的问题，或与其无关的消息。"],
      refusal: ["如实拒答", "文件中没有的事实应拒答，普通请求不应拒答。"],
      network: ["网络与速度", "不用记录：由小网络生成，并测量它在此 CPU 上的速度。"],
    },
    records: { herbs: "植物", countries: "国家", elements: "元素", people: "人物", books: "书籍", self: "自我介绍", trivia: "常识", places: "地点", science: "科学", howto: "方法" },
    examples: ["What is valerian used for?", "Is it safe?", "What is the capital of Japan?", "Why is the sky blue?", "What is 2+2?", "Write a poem about the sea."],
    quality: [["关于植物的简单问题", "178 / 180"], ["同上，留出的名称", "127 / 180"], ["简单问题，第二组", "172 / 180"], ["同上，留出的名称", "155 / 180"], ["植物事实，留出的问题", "218 / 223"], ["世界知识问题", "405 / 445"], ["同上，留出", "367 / 445"], ["对话第一轮", "100 %"], ["平实语言的追问", "900 / 1200"], ["被误认为追问的无关消息", "41 / 1380"], ["记录之外的事实，如实拒答", "38 / 44"], ["普通请求被误拒", "1 / 40"]],
    patents: [["共振路由", "No. 19/452,440", "2026 年 1 月 19 日"], ["统一执行", "No. 19/731,402", "2026 年 7 月 6 日"], ["恒定内存注意力", "No. 19/738,763", "2026 年 7 月 13 日"]],
  },
};

const $ = (id) => document.getElementById(id);
const el = (tag, cls, text) => {
  const e = document.createElement(tag);
  if (cls) e.className = cls;
  if (text !== undefined) e.textContent = text;
  return e;
};
let lang = "en";
const t = (k) => T[lang][k];

function storedLang() {
  try { return localStorage.getItem("embryo-lang"); } catch (e) { return null; }
}
function detectLang() {
  const saved = storedLang();
  if (T[saved]) return saved;
  const n = (navigator.language || "en").slice(0, 2);
  return T[n] ? n : "en";
}

// ---- router details

function sourceChip(route) {
  const target = route && route.target;
  if (!target || target === "backbone") return el("span", "chip", t("srcNetwork"));
  if (target === "dontknow") return el("span", "chip", t("srcRefusal"));
  return el("span", "chip rec", T[lang].records[target] || target);
}

function whyPanel(route) {
  const box = el("div", "why");
  box.appendChild(el("h4", "", t("whyTitle")));
  if (!route) return box;
  const { e_skill: a, e_base: b, reason } = route;
  if (typeof a === "number" && typeof b === "number") {
    const refusal = route.target === "dontknow" || /fact outside/.test(reason || "");
    const followup = /^follow-up/.test(reason || "");
    const other = refusal ? t("eChat") : followup ? t("eFollowup") : t("eBase");
    const top = Math.max(a, b, 1e-6);
    const skillWon = route.target !== "backbone";
    for (const [name, v, win] of [[t("eRecord"), a, skillWon], [other, b, !skillWon]]) {
      const row = el("div", "bar");
      row.appendChild(el("span", "", name));
      const track = el("div");
      const fill = el("i", win ? "win" : "");
      fill.style.width = `${Math.max(2, (100 * v) / top)}%`;
      track.appendChild(fill);
      row.appendChild(track);
      row.appendChild(el("b", "", v.toFixed(3)));
      box.appendChild(row);
    }
  } else {
    box.appendChild(el("p", "reason", t("noDecision")));
  }
  const kv = el("dl", "kv");
  const add = (k, v) => { kv.appendChild(el("dt", "", k)); kv.appendChild(el("dd", "", String(v))); };
  if (route.lookup_key) add(t("key"), route.lookup_key);
  if (route.field) add(t("field"), route.field);
  if (route.lookup_turn) add(t("turnsBack"), route.lookup_turn);
  if (kv.childNodes.length) box.appendChild(kv);
  if (reason) box.appendChild(el("p", "reason", reason));
  return box;
}

// ---- requests

async function chat(messages, maxTokens) {
  const t0 = performance.now();
  const res = await fetch("api/chat", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ messages, max_tokens: maxTokens }),
  });
  const data = await res.json();
  if (!res.ok) throw new Error(data.error || t("failed"));
  const choice = data.choices[0];
  return {
    text: choice.message.content || "",
    finish: choice.finish_reason,
    route: data.x_cortiq_route || null,
    tokens: (data.usage && data.usage.completion_tokens) || 0,
    ms: performance.now() - t0,
  };
}

// ---- tests

const state = { sets: [], results: {}, selected: null, filter: "all", running: false, stopAsked: false, ready: false };

function judge(c, turns) {
  const last = turns[turns.length - 1].route || {};
  const first = turns[0].route || {};
  switch (c.expect) {
    case "record": return last.target === c.record;
    case "followup": return last.target === c.record && last.lookup_turn >= 1 && last.lookup_entry === first.lookup_entry;
    case "not_followup": return !(last.target === c.record && last.lookup_turn >= 1);
    case "refusal": return last.target === "dontknow";
    case "not_refusal": return last.target !== "dontknow";
    default: return null;
  }
}

function expectText(c) {
  const name = c.record ? ` (${T[lang].records[c.record] || c.record})` : "";
  return {
    record: t("expRecord") + name, followup: t("expFollowup"), not_followup: t("expNotFollowup"),
    refusal: t("expRefusal"), not_refusal: t("expNotRefusal"), speed: t("expSpeed"),
  }[c.expect];
}

async function runCase(c) {
  const messages = [];
  const turns = [];
  for (const q of c.turns) {
    messages.push({ role: "user", content: q });
    const a = await chat(messages, c.expect === "speed" ? 200 : 120);
    messages.push({ role: "assistant", content: a.text || "…" });
    turns.push({ q, ...a });
  }
  return { case: c, turns, ok: judge(c, turns) };
}

function totals() {
  let done = 0, ok = 0, miss = 0;
  const all = [], rec = [];
  let spTokens = 0, spMs = 0;
  for (const s of state.sets) {
    for (const r of state.results[s.id] || []) {
      done++;
      if (r.ok === true) ok++;
      if (r.ok === false) miss++;
      const lastTurn = r.turns[r.turns.length - 1];
      all.push(lastTurn.ms);
      if (lastTurn.route && lastTurn.route.lookup_hit) rec.push(lastTurn.ms);
      if (r.case.expect === "speed") { spTokens += lastTurn.tokens; spMs += lastTurn.ms; }
    }
  }
  return { done, ok, miss, all, rec, speed: spMs ? (spTokens * 1000) / spMs : null };
}

const median = (xs) => {
  if (!xs.length) return null;
  const s = [...xs].sort((a, b) => a - b);
  return s[Math.floor(s.length / 2)];
};
const fmtMs = (v) => (v === null ? "–" : v < 1000 ? `${Math.round(v)} ms` : `${(v / 1000).toFixed(2)} s`);

function drawSets() {
  const box = $("sets");
  box.replaceChildren();
  for (const s of state.sets) {
    const [name, desc] = T[lang].sets[s.id];
    const results = state.results[s.id] || [];
    const card = el("div", "set" + (state.selected === s.id ? " on" : ""));
    card.tabIndex = 0;
    card.setAttribute("role", "button");
    card.onclick = () => { state.selected = s.id; draw(); };
    card.onkeydown = (e) => { if (e.key === "Enter") { state.selected = s.id; draw(); } };
    card.appendChild(el("h3", "", name));
    card.appendChild(el("p", "", desc));
    const scored = results.filter((r) => r.ok !== null);
    const score = el("div", "score");
    if (s.id === "network") {
      const ms = results.reduce((a, r) => a + r.turns[0].ms, 0);
      const tk = results.reduce((a, r) => a + r.turns[0].tokens, 0);
      score.textContent = results.length ? `${Math.round((tk * 1000) / ms)}` : "–";
      score.appendChild(el("small", "", results.length ? " tok/s" : ` · ${s.cases.length} ${t("questions")}`));
    } else {
      const unit = s.id === "followups" ? t("dialogs") : t("questions");
      score.textContent = scored.length ? `${scored.filter((r) => r.ok).length}` : "–";
      score.appendChild(el("small", "", scored.length ? ` ${t("of")} ${s.cases.length}` : ` · ${s.cases.length} ${unit}`));
    }
    card.appendChild(score);
    const m = el("div", "meter");
    const f = el("div");
    f.style.width = `${(100 * results.length) / s.cases.length}%`;
    m.appendChild(f);
    card.appendChild(m);
    const run = el("button", "", t("run"));
    run.disabled = state.running || !state.ready;
    run.onclick = (e) => { e.stopPropagation(); runSets([s]); };
    card.appendChild(run);
    box.appendChild(card);
  }
}

function drawStats() {
  const tot = totals();
  const box = $("stats");
  box.hidden = tot.done === 0;
  box.replaceChildren();
  const add = (v, label) => {
    const d = el("div", "stat");
    d.appendChild(el("b", "", v));
    d.appendChild(el("span", "", label));
    box.appendChild(d);
  };
  add(String(tot.ok), t("sPassed"));
  add(String(tot.miss), t("sMissed"));
  add(fmtMs(median(tot.rec)), t("sRecord"));
  add(tot.speed ? `${Math.round(tot.speed)} tok/s` : "–", t("sSpeed"));
  $("download").disabled = tot.done === 0;
}

function drawRow(r, i) {
  const row = el("div", "row");
  const head = el("button", "row-head");
  head.setAttribute("aria-expanded", "false");
  const badge = r.ok === null ? el("span", "badge info", t("info")) : r.ok ? el("span", "badge ok", "✓ " + t("pass")) : el("span", "badge miss", "✕ " + t("miss"));
  const statusCell = el("span");
  statusCell.appendChild(badge);
  head.appendChild(statusCell);
  const lastTurn = r.turns[r.turns.length - 1];
  head.appendChild(el("span", "q", lastTurn.q));
  head.appendChild(el("span", "a", lastTurn.text.replace(/\s+/g, " ")));
  const src = el("span", "src");
  src.appendChild(sourceChip(lastTurn.route));
  head.appendChild(src);
  const time = lastTurn.tokens > 1 ? `${Math.round((lastTurn.tokens * 1000) / lastTurn.ms)} t/s` : fmtMs(lastTurn.ms);
  head.appendChild(el("span", "t", time));
  row.appendChild(head);
  head.onclick = () => {
    const open = row.querySelector(".detail");
    if (open) { open.remove(); head.setAttribute("aria-expanded", "false"); return; }
    head.setAttribute("aria-expanded", "true");
    const d = el("div", "detail");
    const left = el("div");
    for (const turn of r.turns) {
      const q = el("div", "turn");
      q.appendChild(el("div", "who", t("you")));
      q.appendChild(el("div", "say", turn.q));
      left.appendChild(q);
      const a = el("div", "turn");
      a.appendChild(el("div", "who", t("model")));
      a.appendChild(el("div", "say", turn.text));
      left.appendChild(a);
    }
    d.appendChild(left);
    const right = whyPanel(lastTurn.route);
    const exp = el("dl", "kv");
    exp.appendChild(el("dt", "", t("expected")));
    exp.appendChild(el("dd", "", expectText(r.case)));
    right.insertBefore(exp, right.children[1]);
    d.appendChild(right);
    row.appendChild(d);
  };
  return row;
}

function drawResults() {
  const id = state.selected;
  const set = state.sets.find((s) => s.id === id);
  const results = (id && state.results[id]) || [];
  $("resultsHead").hidden = !results.length;
  $("results").replaceChildren();
  if (!results.length) return;
  $("resultsTitle").textContent = `${t("results")}: ${T[lang].sets[set.id][0]}`;
  document.querySelectorAll("[data-filter]").forEach((b) => b.classList.toggle("on", b.dataset.filter === state.filter));
  const rows = el("div", "rows");
  results.forEach((r, i) => {
    if (state.filter === "missed" && r.ok !== false) return;
    rows.appendChild(drawRow(r, i));
  });
  if (!rows.childNodes.length) {
    $("results").appendChild(el("div", "empty", "✓"));
    return;
  }
  $("results").appendChild(rows);
}

function draw() {
  drawSets();
  drawStats();
  drawResults();
}

async function runSets(sets) {
  if (state.running) return;
  state.running = true;
  state.stopAsked = false;
  $("runAll").disabled = true;
  $("stop").hidden = false;
  $("progress").hidden = false;
  const total = sets.reduce((a, s) => a + s.cases.length, 0);
  let done = 0;
  try {
    for (const s of sets) {
      state.results[s.id] = [];
      state.selected = s.id;
      draw();
      for (const c of s.cases) {
        if (state.stopAsked) return;
        state.results[s.id].push(await runCase(c));
        done++;
        $("progressBar").style.width = `${(100 * done) / total}%`;
        $("progressText").textContent = `${done} / ${total}`;
        draw();
      }
    }
  } catch (e) {
    $("progressText").textContent = `${t("failed")}: ${e.message}`;
  } finally {
    state.running = false;
    $("runAll").disabled = false;
    $("stop").hidden = true;
    draw();
  }
}

function downloadResults() {
  const dump = {
    model: "infosave/Embryo-56M-cmf",
    when: new Date().toISOString(),
    sets: state.sets.map((s) => ({
      id: s.id,
      results: (state.results[s.id] || []).map((r) => ({
        expect: r.case.expect, record: r.case.record || null, ok: r.ok,
        turns: r.turns.map((x) => ({ question: x.q, answer: x.text, ms: Math.round(x.ms), tokens: x.tokens, finish: x.finish, route: x.route })),
      })),
    })),
  };
  const url = URL.createObjectURL(new Blob([JSON.stringify(dump, null, 1)], { type: "application/json" }));
  const a = document.createElement("a");
  a.href = url;
  a.download = "embryo-56m-results.json";
  a.click();
  URL.revokeObjectURL(url);
}

// ---- chat

const dialog = [];

function drawEmptyLog() {
  if (!dialog.length) {
    $("log").replaceChildren(el("div", "empty", t("chatEmpty")));
  }
}

function botMeta(box, route, ms, tokens) {
  const meta = el("div", "meta");
  meta.appendChild(sourceChip(route));
  meta.appendChild(el("span", "", tokens > 1 ? `${fmtMs(ms)} · ${Math.round((tokens * 1000) / ms)} tok/s` : fmtMs(ms)));
  const why = el("button", "", t("whyShow"));
  why.type = "button";
  why.onclick = () => {
    const open = box.querySelector(".why");
    if (open) open.remove();
    else box.appendChild(whyPanel(route));
  };
  meta.appendChild(why);
  box.appendChild(meta);
}

async function sendChat(text) {
  if (!text.trim()) return;
  if (!dialog.length) $("log").replaceChildren();
  dialog.push({ role: "user", content: text.trim() });
  $("log").appendChild(el("div", "msg user", text.trim()));
  const box = el("div", "msg bot");
  const body = el("span", "", "…");
  box.appendChild(body);
  $("log").appendChild(box);
  $("send").disabled = true;
  box.scrollIntoView({ block: "nearest" });
  const t0 = performance.now();
  try {
    const res = await fetch("api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ messages: dialog.slice(-12), stream: true }),
    });
    if (!res.ok) {
      const d = await res.json().catch(() => ({}));
      throw new Error(d.error || t("failed"));
    }
    const reader = res.body.getReader();
    const dec = new TextDecoder();
    let buf = "", text2 = "", route = null, tokens = 0;
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      buf += dec.decode(value, { stream: true });
      let i;
      while ((i = buf.indexOf("\n")) >= 0) {
        const line = buf.slice(0, i).trim();
        buf = buf.slice(i + 1);
        if (!line.startsWith("data:")) continue;
        const payload = line.slice(5).trim();
        if (payload === "[DONE]") continue;
        const j = JSON.parse(payload);
        if (j.x_cortiq_route) route = j.x_cortiq_route;
        if (j.usage) tokens = j.usage.completion_tokens || tokens;
        const delta = j.choices && j.choices[0] && j.choices[0].delta && j.choices[0].delta.content;
        if (delta) {
          text2 += delta;
          body.textContent = text2;
        }
      }
    }
    dialog.push({ role: "assistant", content: text2 || "…" });
    botMeta(box, route, performance.now() - t0, tokens);
  } catch (e) {
    dialog.pop();
    body.textContent = `${t("failed")}: ${e.message}`;
  } finally {
    $("send").disabled = false;
    box.scrollIntoView({ block: "nearest" });
  }
}

// ---- static parts

function applyLang() {
  document.documentElement.lang = lang;
  document.querySelectorAll("[data-t]").forEach((e) => { e.textContent = t(e.dataset.t); });
  document.querySelectorAll("[data-t-placeholder]").forEach((e) => { e.placeholder = t(e.dataset.tPlaceholder); });
  document.querySelectorAll("[data-lang]").forEach((b) => b.classList.toggle("on", b.dataset.lang === lang));
  $("figRouting").src = `${ASSETS}routing_${lang}.png`;
  $("figRouting").alt = t("capRouting");
  $("figArch").src = `${ASSETS}architecture_${lang}.png`;
  $("figArch").alt = t("capArch");
  $("figAdv").src = `${ASSETS}advantages_${lang}.png`;
  $("figAdv").alt = t("capAdv");
  const ex = $("examples");
  ex.replaceChildren();
  for (const q of t("examples")) {
    const b = el("button", "", q);
    b.type = "button";
    b.onclick = () => sendChat(q);
    ex.appendChild(b);
  }
  const qual = $("quality");
  qual.replaceChildren();
  for (const [a, b] of t("quality")) {
    const tr = el("tr");
    tr.appendChild(el("td", "", a));
    tr.appendChild(el("td", "num", b));
    qual.appendChild(tr);
  }
  const pat = $("patents");
  pat.replaceChildren();
  for (const row of t("patents")) {
    const tr = el("tr");
    row.forEach((v) => tr.appendChild(el("td", "", v)));
    pat.appendChild(tr);
  }
  drawEmptyLog();
  draw();
}

function selectTab(name) {
  document.querySelectorAll("[data-tab]").forEach((b) => {
    const on = b.dataset.tab === name;
    b.classList.toggle("on", on);
    b.setAttribute("aria-selected", on);
    $(b.dataset.tab).hidden = !on;
  });
}

async function pollHealth() {
  try {
    const h = await (await fetch("api/health")).json();
    $("loading").hidden = h.ready;
    state.ready = h.ready;
    $("runAll").disabled = !h.ready || state.running;
    $("send").disabled = !h.ready;
    drawSets();
    if (h.ready) return;
  } catch (e) {
    $("loading").hidden = false;
  }
  setTimeout(pollHealth, 1500);
}

async function init() {
  lang = detectLang();
  document.querySelectorAll("[data-lang]").forEach((b) => {
    b.onclick = () => {
      lang = b.dataset.lang;
      try { localStorage.setItem("embryo-lang", lang); } catch (e) { /* the choice lives for this page only */ }
      applyLang();
    };
  });
  document.querySelectorAll("[data-tab]").forEach((b) => { b.onclick = () => selectTab(b.dataset.tab); });
  document.querySelectorAll("[data-filter]").forEach((b) => { b.onclick = () => { state.filter = b.dataset.filter; drawResults(); }; });
  $("runAll").onclick = () => runSets(state.sets);
  $("stop").onclick = () => { state.stopAsked = true; };
  $("download").onclick = downloadResults;
  $("form").onsubmit = (e) => {
    e.preventDefault();
    const v = $("input").value;
    $("input").value = "";
    sendChat(v);
  };
  $("input").onkeydown = (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      $("form").requestSubmit();
    }
  };
  $("reset").onclick = () => { dialog.length = 0; $("log").replaceChildren(); drawEmptyLog(); };
  state.sets = (await (await fetch("tests.json")).json()).sets;
  state.selected = state.sets[0].id;
  selectTab("tests");
  applyLang();
  pollHealth();
}

init();
