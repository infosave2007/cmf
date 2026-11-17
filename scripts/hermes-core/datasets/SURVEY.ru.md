# Что можно взять для обучения ядра Hermes: обзор объёмов, 14.11.2026

Источники: список наборов на странице `qvac/BotPsy-0.8B` (29 названий),
исходники открытых агентных харнессов на GitHub, наш корпус от 03.11.
Ничего не скачивалось: только метаданные через открытые API
(`survey.py`, результаты — `hf-sizes.json`, `repos.tsv`).

Оценка токенов: байты / 4 для диалогов и трасс, байты / 3,46 для кода
(отношение измерено на исходниках Hermes токенизатором Embryo). Точность
оценки ±30 %; у наборов без parquet-статистики взяты размеры файлов, для
сжатого parquet это занижение в 2–3 раза.

## Наборы со страницы BotPsy

Лицензия — из метаданных самого набора на HF, а не со страницы BotPsy.

### Агентные и инструментальные трассы — 23,9 млрд токенов

| Набор | Лицензия | Объём | Токены |
|---|---|---|---|
| Agent-Ark/Toucan-1.5M | Apache-2.0 | 64,0 ГБ | 16,0 млрд |
| nvidia/Nemotron-SFT-Agentic-v2 | CC-BY-4.0 / Apache-2.0 / MIT | 21,9 ГБ | 5,5 млрд |
| Nanbeige/ToolMind | Apache-2.0 | 3,9 ГБ | 0,97 млрд |
| tuandunghcmut/toolbench-v1 | Apache-2.0 (перезалив ToolBench) | 1,9 ГБ | 0,47 млрд |
| nvidia/…Conversational-Tool-Use-Pivot-v1 | CC-BY-4.0 | 1,6 ГБ | 0,40 млрд |
| inclusionAI/AReaL-tau2-data | Apache-2.0 | 0,9 ГБ | 0,22 млрд |
| LARK-Lab/EnvFactory-SFT-FILTERED | Apache-2.0 | 0,9 ГБ | 0,23 млрд |
| nvidia/…Function-Calling-Pivot-v1 | CC-BY-4.0 | 0,3 ГБ | 0,07 млрд |
| glaiveai/glaive-function-calling-v2 | Apache-2.0 | 0,25 ГБ | 0,06 млрд |
| When2Call, ToolACE, xlam-irrelevance-7.5k, workplace_assistant | CC-BY-4.0 / Apache-2.0 | 0,12 ГБ | 0,03 млрд |

### Общие диалоги, инструкции, рассуждения

| Набор | Лицензия | Объём | Токены |
|---|---|---|---|
| HuggingFaceTB/smoltalk2 | в метаданных не указана; смесь, проверять по поднаборам | 208,7 ГБ | 52 млрд |
| openbmb/UltraData-SFT-2605 | Apache-2.0, доступ по согласию с условиями | 97,6 ГБ файлов | от 24 млрд |
| HuggingFaceH4/ultrachat_200k | MIT | 3,0 ГБ | 0,75 млрд |
| nvidia/Nemotron-RL-knowledge-mcqa | CC-BY-4.0 | 1,3 ГБ | 0,32 млрд |
| наборы на следование инструкциям (4 шт.) | CC-BY-4.0 / ODC-BY | 0,5 ГБ | 0,12 млрд |

В smoltalk2: Mid 125,9 ГБ (в том числе трассы рассуждений OpenThoughts3),
SFT 77,3 ГБ, Preference 5,5 ГБ.

### Не брать или брать с оговорками

| Набор | Причина |
|---|---|
| BAAI/Infinity-Instruct (21 ГБ файлов) | на HF стоит CC-BY-SA-4.0 (страница BotPsy называет CC-BY), доступ по согласию |
| nvidia/Nemotron-RL-coding-competitive_coding (5,9 ГБ) | CC-BY-SA-4.0; большая часть объёма — тестовые данные |
| Skywork/Skywork-OR1-RL-Data (2,0 ГБ) | лицензия не указана |
| nvidia/Nemotron-3-Nano-RL-Training-Blend (6,9 ГБ) | задачи для RL (условия и проверки), а не текст; включает часть с CC-BY-SA |
| BytedTsinghua-SIA/DAPO-Math-17k | 17 тыс. задач, повторённых до 1,8 млн строк |
| Salesforce/xlam-function-calling-60k (0,1 ГБ) | CC-BY-4.0, доступ по согласию с условиями |
| Tool-N1, ToolRL (GitHub) | Apache-2.0, объём не измерялся |

## Исходники харнессов

31 репозиторий, код по подсчёту GitHub (без документации в Markdown).

| Репозиторий | Лицензия | Код | Токены |
|---|---|---|---|
| openclaw/openclaw | MIT | 520 МБ | 150 млн |
| QwenLM/qwen-code (форк gemini-cli) | Apache-2.0 | 223 МБ | 64 млн |
| NousResearch/hermes-agent (сейчас) | MIT | 181 МБ | 52 млн |
| openai/codex | Apache-2.0 | 89 МБ | 26 млн |
| openinterpreter/openinterpreter (по составу языков похож на форк Codex) | Apache-2.0 | 72 МБ | 21 млн |
| anomalyco/opencode | MIT | 64 МБ | 18 млн |
| pydantic/pydantic-ai | MIT | 54 МБ | 16 млн |
| agno-agi/agno | Apache-2.0 | 54 МБ | 16 млн |
| earendil-works/pi | MIT | 33 МБ | 10 млн |
| cline/cline | Apache-2.0 | 32 МБ | 9 млн |
| gemini-cli, crewAI, openai-agents-python, OpenHands, goose, langchain, Roo-Code, continue, camel, langgraph, autogen, MetaGPT, plandex, aider, smolagents, SWE-agent, mini-swe-agent, OpenManus | MIT / Apache-2.0 | 164 МБ | 47 млн |

Всего с разрешительной лицензией 1,49 ГБ кода ≈ 430 млн токенов; без
вероятных форков (qwen-code, Roo-Code, openinterpreter) ≈ 340 млн.
В корпусе от 03.11 исходников Hermes 20,1 млн токенов: репозиторий с тех
пор вырос в 2,6 раза.

Не брать: Significant-Gravitas/AutoGPT (смешанная лицензия),
charmbracelet/crush (не открытая лицензия), Claude Code (исходники закрыты).

## Итог

| Группа | Токены |
|---|---|
| Наш корпус (Hermes, документация ОС, Stack Exchange) | 0,49 млрд |
| Код харнессов | 0,34–0,43 млрд |
| Агентные трассы, открытый доступ, разрешительные лицензии | 23,9 млрд |
| Общие диалоги с ясной лицензией (ultrachat, mcqa, инструкции) | 1,2 млрд |
| UltraData-SFT-2605 (после согласия с условиями) | от 24 млрд |
| smoltalk2 (после проверки лицензий поднаборов) | до 52 млрд |

Сразу и без оговорок доступно около 26 млрд токенов, с двумя большими
общими наборами — около 100 млрд. Под A40 проходит 1,16 млрд токенов в
сутки: данных на порядок-два больше, чем карта успеет прочитать, объём
обучения задаёт время аренды, а не данные.

## Что учесть при сборке

- Форматы вызова инструментов у наборов разные (JSON-схемы, теги, MCP у
  Toucan): нужен перевод в один чат-контракт с нашим форматом команд.
- Почти всё синтетика от моделей-учителей (у Toucan — Kimi-K2, Qwen3,
  GPT-OSS) и почти всё на английском; русского текста в этих наборах мало.
- CC-BY и ODC-BY требуют указания источников (файл атрибуции, как у BotPsy).
- Наборы с доступом «по согласию» открывает владелец аккаунта HF.
- Крупные наборы качать сразу на под (390 ГБ свободно), не на Mac.
