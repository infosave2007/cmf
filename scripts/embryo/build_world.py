#!/usr/bin/env python3
# The `world` lookup record of the Embryo chat model: countries, chemical elements, space, science, computing,
# history, people and books as short cards in English and Russian, and the question sets that fit its reconstruction
# topology. A held-out part of the entities never enters the fit. Usage: build_world.py <out_dir>
import json, os, random, re, sys
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = sys.argv[1]
os.makedirs(OUT, exist_ok=True)
CONT = {'EU': ('Europe', 'Европе'), 'AS': ('Asia', 'Азии'), 'AF': ('Africa', 'Африке'), 'NA': ('North America', 'Северной Америке'),
        'SA': ('South America', 'Южной Америке'), 'OC': ('Oceania', 'Океании')}
# Capital names and words too ambiguous to be keys by themselves
BAD_CAPITAL = {'Male', 'Victoria', 'Luxembourg', 'Monaco', 'Singapore', 'Djibouti', 'San Marino', 'Vatican City', 'Sofia', 'Lima', 'Kingston', 'Georgetown', 'San Jose', 'Santiago', 'Tripoli', 'Washington, D.C.', 'Roseau', 'Bern', 'Accra', 'Praia', 'Sucre (seat of government: La Paz)', 'Nauru', 'Yaren (no official capital)', 'Pretoria, Cape Town and Bloemfontein', 'Sri Jayawardenepura Kotte (largest city: Colombo)'}
BAD_RU_KEY = {'Алжир', 'Монако', 'Сингапур', 'Люксембург', 'Джибути', 'Ватикан', 'Сан-Марино', 'Панама', 'Гватемала', 'Тунис', 'София', 'Виктория', 'Лима', 'Рига', 'Вена', 'Прага', 'Рим', 'Сана', 'Сува', 'Сукре (резиденция правительства: Ла-Пас)', 'Ярен (официальной столицы нет)', 'Претория, Кейптаун и Блумфонтейн', 'Шри-Джаяварденепура-Котте (крупнейший город: Коломбо)', 'Вашингтон'}
GOOD_RU_CITY = {'Вена', 'Прага', 'Рига', 'Рим'}
MAX_KEY_WORDS = 4
QFORM = re.compile(r'^(why|how|who|what|when|where|is|are|do|does|did|can|почему|как|кто|что|сколько|где|когда|зачем|какой|какая|какое|какие|на каких|о чём)\b', re.I)
rows = []
entities = []
forms = []
for line in open(os.path.join(HERE, 'world_countries.tsv'), encoding='utf-8'):
    if line.startswith('#') or not line.strip():
        continue
    en, ru, cont, cap_en, cap_ru, cur_en, cur_ru, lang_en, lang_ru = line.rstrip('\n').split('\t')
    keys = [en, ru]
    if cap_en not in BAD_CAPITAL:
        keys.append(cap_en)
    if cap_ru not in BAD_RU_KEY or cap_ru in GOOD_RU_CITY:
        keys.append(cap_ru)
    if ru == 'Соединённые Штаты Америки':
        keys += ['USA', 'America', 'США', 'Америка', 'United States of America']
    if ru == 'Великобритания':
        keys += ['UK', 'Britain', 'Great Britain', 'England', 'Англия', 'Британия']
    if ru == 'Объединённые Арабские Эмираты':
        keys += ['UAE', 'ОАЭ', 'Emirates']
    if ru == 'Тоголезская Республика':
        keys += ['Того Африка']
    if ru == 'Южная Африка':
        keys += ['ЮАР', 'Южно-Африканская Республика']
    if ru == 'Чехия':
        keys += ['Czechia']
    if ru == 'Демократическая Республика Конго':
        keys += ['DR Congo', 'ДР Конго']
    card_en = f"{en} is a country in {CONT[cont][0]}. Capital: {cap_en}. Currency: {cur_en}. Official language: {lang_en}."
    card_ru = f"{ru} — страна в {CONT[cont][1]}. Столица: {cap_ru}. Валюта: {cur_ru}. Официальный язык: {lang_ru}."
    rows.append({'keys': keys, 'en': {'card': card_en, 'fields': {}}, 'ru': {'card': card_ru, 'fields': {}}})
    entities.append(('country', en, ru))
    forms.append(([k for k in keys if not re.search('[а-яё]', k, re.I)], [k for k in keys if re.search('[а-яё]', k, re.I)]))
ELEMENTS = """Hydrogen Водород H;Helium Гелий He;Lithium Литий Li;Beryllium Бериллий Be;Boron Бор B;Carbon Углерод C;Nitrogen Азот N;Oxygen Кислород O;Fluorine Фтор F;Neon Неон Ne;Sodium Натрий Na;Magnesium Магний Mg;Aluminium Алюминий Al;Silicon Кремний Si;Phosphorus Фосфор P;Sulfur Сера S;Chlorine Хлор Cl;Argon Аргон Ar;Potassium Калий K;Calcium Кальций Ca;Scandium Скандий Sc;Titanium Титан Ti;Vanadium Ванадий V;Chromium Хром Cr;Manganese Марганец Mn;Iron Железо Fe;Cobalt Кобальт Co;Nickel Никель Ni;Copper Медь Cu;Zinc Цинк Zn;Gallium Галлий Ga;Germanium Германий Ge;Arsenic Мышьяк As;Selenium Селен Se;Bromine Бром Br;Krypton Криптон Kr;Rubidium Рубидий Rb;Strontium Стронций Sr;Yttrium Иттрий Y;Zirconium Цирконий Zr;Niobium Ниобий Nb;Molybdenum Молибден Mo;Technetium Технеций Tc;Ruthenium Рутений Ru;Rhodium Родий Rh;Palladium Палладий Pd;Silver Серебро Ag;Cadmium Кадмий Cd;Indium Индий In;Tin Олово Sn;Antimony Сурьма Sb;Tellurium Теллур Te;Iodine Йод I;Xenon Ксенон Xe;Caesium Цезий Cs;Barium Барий Ba;Lanthanum Лантан La;Cerium Церий Ce;Praseodymium Празеодим Pr;Neodymium Неодим Nd;Promethium Прометий Pm;Samarium Самарий Sm;Europium Европий Eu;Gadolinium Гадолиний Gd;Terbium Тербий Tb;Dysprosium Диспрозий Dy;Holmium Гольмий Ho;Erbium Эрбий Er;Thulium Тулий Tm;Ytterbium Иттербий Yb;Lutetium Лютеций Lu;Hafnium Гафний Hf;Tantalum Тантал Ta;Tungsten Вольфрам W;Rhenium Рений Re;Osmium Осмий Os;Iridium Иридий Ir;Platinum Платина Pt;Gold Золото Au;Mercury Ртуть Hg;Thallium Таллий Tl;Lead Свинец Pb;Bismuth Висмут Bi;Polonium Полоний Po;Astatine Астат At;Radon Радон Rn;Francium Франций Fr;Radium Радий Ra;Actinium Актиний Ac;Thorium Торий Th;Protactinium Протактиний Pa;Uranium Уран U;Neptunium Нептуний Np;Plutonium Плутоний Pu;Americium Америций Am;Curium Кюрий Cm;Berkelium Берклий Bk;Californium Калифорний Cf;Einsteinium Эйнштейний Es;Fermium Фермий Fm;Mendelevium Менделевий Md;Nobelium Нобелий No;Lawrencium Лоуренсий Lr;Rutherfordium Резерфордий Rf;Dubnium Дубний Db;Seaborgium Сиборгий Sg;Bohrium Борий Bh;Hassium Хассий Hs;Meitnerium Мейтнерий Mt;Darmstadtium Дармштадтий Ds;Roentgenium Рентгений Rg;Copernicium Коперниций Cn;Nihonium Нихоний Nh;Flerovium Флеровий Fl;Moscovium Московий Mc;Livermorium Ливерморий Lv;Tennessine Теннессин Ts;Oganesson Оганесон Og""".split(';')
assert len(ELEMENTS) == 118
NOBLE = {'He', 'Ne', 'Ar', 'Kr', 'Xe', 'Rn', 'Og'}
HALO = {'F', 'Cl', 'Br', 'I', 'At', 'Ts'}
NONMET = {'H', 'C', 'N', 'O', 'P', 'S', 'Se'}
METALLOID = {'B', 'Si', 'Ge', 'As', 'Sb', 'Te'}
BAD_EL_KEY = {'Tin', 'Lead', 'Mercury', 'Olово', 'Боr'}
BAD_EL_RU = {'Олово', 'Бор', 'Уран', 'Сера', 'Титан', 'Франций'}
for i, e in enumerate(ELEMENTS):
    en, ru, sym = e.split(' ')
    kind_en, kind_ru = ('a noble gas', 'благородный газ') if sym in NOBLE else ('a halogen', 'галоген') if sym in HALO else ('a nonmetal', 'неметалл') if sym in NONMET else ('a metalloid', 'полуметалл') if sym in METALLOID else ('a metal', 'металл')
    keys = []
    keys += [en] if en not in BAD_EL_KEY else [f'{en} element', f'element {en.lower()}', f'chemical element {en.lower()}']
    keys += [ru] if ru not in BAD_EL_RU else [f'элемент {ru.lower()}', f'химический элемент {ru.lower()}']
    if en == 'Aluminium':
        keys.append('Aluminum')
    if en == 'Sulfur':
        keys.append('Sulphur')
    if en == 'Caesium':
        keys.append('Cesium')
    rows.append({'keys': keys, 'en': {'card': f"{en} ({sym}) is a chemical element with atomic number {i + 1}. It is {kind_en}.", 'fields': {}},
                 'ru': {'card': f"{ru} ({sym}) — химический элемент с атомным номером {i + 1}, {kind_ru}.", 'fields': {}}})
    entities.append(('element', en, ru))
    forms.append(([k for k in keys if not re.search('[а-яё]', k, re.I)], [k for k in keys if re.search('[а-яё]', k, re.I)]))
for line in open(os.path.join(HERE, 'world_facts.tsv'), encoding='utf-8'):
    if line.startswith('#') or not line.strip():
        continue
    kind, ken, kru, cen, cru = line.rstrip('\n').split('\t')
    # A key of more than MAX_KEY_WORDS words cannot match (the table searches n-grams up to that): it stays as the natural
    # phrasing of the fit, and a shorter fragment marked `~` is the key.
    keys = [k.rstrip('~') for k in ken.split(';') + kru.split(';') if len(k.split()) <= MAX_KEY_WORDS]
    assert keys, line
    rows.append({'keys': keys, 'en': {'card': cen, 'fields': {}}, 'ru': {'card': cru, 'fields': {}}})
    entities.append((kind, ken.split(';')[0], kru.split(';')[0]))
    forms.append((ken.split(';'), kru.split(';')))
# One record per kind of knowledge: each gets its own isolated topology in the router.
GROUPS = {'countries': ['country'], 'elements': ['element'], 'people': ['person'], 'books': ['work'], 'self': ['self'], 'trivia': ['trivia'],
          'howto': ['howto'], 'places': ['history', 'geo'], 'science': ['space', 'science', 'math', 'computing', 'body', 'unit', 'econ', 'animal']}
GROUP_OF = {k: g for g, v in GROUPS.items() for k in v}
for g in GROUPS:
    with open(os.path.join(OUT, f'entries-{g}.jsonl'), 'w', encoding='utf-8') as f:
        for r, e in zip(rows, entities):
            if GROUP_OF[e[0]] == g:
                f.write(json.dumps(r, ensure_ascii=False) + '\n')
# Question sets: templates per kind; 15 % of the entities only in the held-out test set
def gen_ru(n):
    # genitive for the fit questions; a wrong form is harmless: the table's stem index and the router see real inflections
    if ' ' in n or '-' in n or n[-1] in 'оуиеыю':
        return n
    if n.endswith('ия'):
        return n[:-1] + 'и'
    if n.endswith('я'):
        return n[:-1] + 'и'
    if n.endswith('а'):
        return n[:-1] + ('и' if n[-2] in 'кгхжшчщ' else 'ы')
    if n.endswith('ь'):
        return n[:-1] + 'и'
    return n + 'а'
T = {
 'country': {'en': ['What is the capital of {}?', 'Which continent is {} in?', 'What currency is used in {}?', 'What language do people speak in {}?', 'Tell me about {}.', 'Where is {}?', '{}: what is its capital?', 'Name the capital city of {}.'],
             'ru': ['Какая столица у страны {n}?', 'Столица {g}?', 'На каком континенте находится {n}?', 'Какая валюта в стране {n}?', 'На каком языке говорят в стране {n}?', 'Расскажи про страну {n}.', '{n}: какая столица?', 'Назови столицу {g}.', 'Какой город является столицей {g}?']},
 'element': {'en': ['What is {}?', 'What is the atomic number of {}?', 'What is the chemical symbol of {}?', 'Tell me about the element {}.', 'Is {} a metal?'],
             'ru': ['Что такое {n}?', 'Какой атомный номер у элемента {n}?', 'Какой химический символ у элемента {n}?', 'Расскажи про элемент {n}.', '{n} это металл?']},
 'space': {'en': ['What is {}?', 'Tell me about {}.', 'Explain {} briefly.', 'How far is {}?'], 'ru': ['Что такое {n}?', 'Расскажи про {n}.', 'Кратко объясни, что такое {n}.', '{n}: что это?']},
 'science': {'en': ['What is {}?', 'Explain {}.', 'Tell me about {}.', 'Explain {} in simple terms.', 'How does {} work?'], 'ru': ['Что такое {n}?', 'Объясни, что такое {n}.', 'Расскажи про {n}.', 'Простыми словами: {n}?', 'Как работает {n}?']},
 'math': {'en': ['What is {}?', 'Explain {}.', 'Define {}.'], 'ru': ['Что такое {n}?', 'Объясни, что такое {n}.', 'Дай определение: {n}.']},
 'computing': {'en': ['What is {}?', 'Explain {}.', 'Explain {} to a beginner.', 'What is {} used for?'], 'ru': ['Что такое {n}?', 'Объясни, что такое {n}.', 'Объясни новичку, что такое {n}.', 'Зачем нужен {n}?']},
 'history': {'en': ['What was {}?', 'When did {} happen?', 'Tell me about {}.', 'What is {}?'], 'ru': ['Что такое {n}?', 'Когда была {n}?', 'Расскажи про {n}.', '{n}: что это?']},
 'person': {'en': ['Who was {}?', 'What is {} famous for?', 'Tell me about {}.', 'When did {} live?'], 'ru': ['Кто такой {n}?', 'Чем известен {n}?', 'Расскажи про {n}.', 'Когда жил {n}?']},
 'work': {'en': ['Who wrote {}?', 'Who is the author of {}?', 'What is {} about?', 'Tell me about {}.', 'When was {} written?'], 'ru': ['Кто написал «{n}»?', 'Кто автор произведения «{n}»?', 'О чём «{n}»?', 'Расскажи про «{n}».', 'Когда написали «{n}»?']},
 'geo': {'en': ['What is {}?', 'Where is {}?', 'Tell me about {}.', 'How big is {}?'], 'ru': ['Что такое {n}?', 'Где находится {n}?', 'Расскажи про {n}.', 'Какой размер у объекта {n}?']},
 'body': {'en': ['Tell me about {}.', 'What does {} do?', 'Explain {}.'], 'ru': ['Расскажи про {n}.', 'Что делает {n}?', 'Объясни, что такое {n}.']},
 'unit': {'en': ['What is {}?', 'Explain {}.'], 'ru': ['Что такое {n}?', 'Объясни: {n}.']},
 'econ': {'en': ['What is {}?', 'Explain {}.', 'Explain {} in simple terms.', 'How does {} work?'], 'ru': ['Что такое {n}?', 'Объясни, что такое {n}.', 'Простыми словами: {n}?', 'Расскажи про {n}.']},
 'animal': {'en': ['What is {}?', 'Tell me about {}.', '{}?'], 'ru': ['Что такое {n}?', 'Расскажи про {n}.', '{n}?']},
}
# Test phrasings the fit never saw
TEST_T = {
 'country': {'en': ['What city is the capital of {}?', 'I am going to {}. What money do they use?', 'What do they speak in {}?'], 'ru': ['Какой город столица {g}?', 'Я еду в страну {n}. Какие там деньги?', 'Какой язык официальный в стране {n}?']},
 'element': {'en': ['What kind of substance is {}?', 'Give me the symbol for {}.'], 'ru': ['Какое вещество {n}?', 'Какой символ в таблице у {n}?']},
 'space': {'en': ['Can you describe {}?', 'What should I know about {}?'], 'ru': ['Опиши {n}.', 'Что нужно знать про {n}?']},
 'science': {'en': ['Can you describe {}?', 'What should I know about {}?'], 'ru': ['Опиши {n}.', 'Что нужно знать про {n}?']},
 'math': {'en': ['What does {} mean?'], 'ru': ['Что означает {n}?']},
 'computing': {'en': ['What does {} mean?', 'Give me a short definition of {}.'], 'ru': ['Что означает {n}?', 'Дай короткое определение: {n}.']},
 'history': {'en': ['What happened in {}?', 'Give me a short summary of {}.'], 'ru': ['Что произошло: {n}?', 'Коротко: {n}.']},
 'person': {'en': ['Who is {}?', 'Give me a short biography of {}.'], 'ru': ['Кто это: {n}?', 'Краткая биография: {n}.']},
 'work': {'en': ['Which author wrote {}?', 'Who created {}?'], 'ru': ['Какой автор написал «{n}»?', 'Кто создал «{n}»?']},
 'geo': {'en': ['Can you describe {}?', 'Where can I find {}?'], 'ru': ['Опиши {n}.', 'Где искать {n}?']},
 'body': {'en': ['Can you describe {}?'], 'ru': ['Опиши: {n}.']},
 'unit': {'en': ['What does {} mean?'], 'ru': ['Что означает {n}?']},
 'econ': {'en': ['What does {} mean?', 'Give me a short definition of {}.'], 'ru': ['Что означает {n}?', 'Дай короткое определение: {n}.']},
 'animal': {'en': ['What should I know about {}?'], 'ru': ['Что нужно знать про {n}?']},
}
# self: the keys themselves are the questions
T['self'] = {'en': [], 'ru': []}; TEST_T['self'] = {'en': [], 'ru': []}
T['howto'] = {'en': ['How do I {}?', 'How can I {}?', 'How to {}?', 'Any advice on how to {}?', 'Give me a tip: how do I {}?'],
              'ru': ['Как {n}?', 'Как мне {n}?', 'Посоветуй, как {n}.', 'Подскажи, как {n}.', 'Как можно {n}?']}
TEST_T['howto'] = {'en': ['What is the best way to {}?', 'I want to {}. What should I do?'], 'ru': ['Как лучше {n}?', 'Хочу {n}. Что делать?']}
T['trivia'] = {'en': ['What is the {}?', '{}?', 'Tell me the {}.'], 'ru': ['{n}?', 'Какая {n}?', 'Скажи: {n}.']}
TEST_T['trivia'] = {'en': ['Do you know the {}?'], 'ru': ['Ты знаешь: {n}?']}
rng = random.Random(29)
ids = list(range(len(entities)))
rng.shuffle(ids)
held = set(ids[:len(ids) * 15 // 100])
fit, fit_all, test_held, test_seen = [], [], [], []
def render(t, lang, en, ru):
    return t.format(en) if lang == 'en' else t.format(n=ru, g=gen_ru(ru))
# A key that is a question by itself is its own phrasing; every other name of an entity gets the first templates of its kind.
ALT = {'country': {'en': ['Tell me about {}.', 'Where is {}?'], 'ru': ['Расскажи про {n}.', 'Где находится {n}?']}}
NATURAL = {'en': ['{K}?', '{k}', 'Hey, {k}?', 'Tell me, {k}?', 'Do you know {k}?'], 'ru': ['{K}?', '{k}', 'Слушай, {k}?', 'Скажи, {k}?', 'Ты знаешь, {k}?']}
NATURAL_TEST = {'en': ['Can you tell me {k}?'], 'ru': ['Подскажи, {k}?']}
for i, ((kind, en, ru), (fen, fru)) in enumerate(zip(entities, forms)):
    for lang, names in (('en', fen), ('ru', fru)):
        natural = [k for k in names if QFORM.match(k) and not k.endswith('~')]
        plain = [k for k in names if not QFORM.match(k) and not k.endswith('~')]
        rows_fit = []
        for j, k in enumerate(plain):
            ts = T[kind][lang] if j == 0 else ALT.get(kind, T[kind])[lang][:2]
            rows_fit += [render(t, lang, k, k) for t in ts]
        for k in natural + (plain if kind == 'self' else []):
            K = k[0].upper() + k[1:]
            rows_fit += [t.format(k=k, K=K) for t in NATURAL[lang]]
            # a short message is the key and its punctuation: greetings, thanks
            if kind == 'self':
                rows_fit += [K, K + '!', K + '.', k + '!', K + ', please', K + '...']
        for p in rows_fit:
            if i not in held:
                fit.append({'lang': lang, 'prompt': p, 'src': en, 'kind': kind})
            fit_all.append({'lang': lang, 'prompt': p, 'src': en, 'kind': kind})
        tests = [render(t, lang, plain[0], plain[0]) for t in TEST_T[kind][lang]] if plain else []
        for k in natural + (plain if kind == 'self' else []):
            tests += [t.format(k=k) for t in NATURAL_TEST[lang]]
        for p in tests:
            (test_held if i in held else test_seen).append({'lang': lang, 'prompt': p, 'src': en, 'kind': kind})
rng.shuffle(fit); rng.shuffle(fit_all)
for name, data in (('world-fit', fit), ('world-fit-all', fit_all), ('world-test-held', test_held), ('world-test-seen', test_seen)):
    with open(os.path.join(OUT, name + '.jsonl'), 'w', encoding='utf-8') as f:
        for r in data:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')
# The card of a question for a fact outside the records (keyless record: the key is a placeholder, the router decides)
with open(os.path.join(OUT, 'entries-dontknow.jsonl'), 'w', encoding='utf-8') as f:
    f.write(json.dumps({'keys': ['cortiq keyless record card'],
        'en': {'card': "I don't know that: it is not in my records, and the small network behind me would only guess. Ask me about medicinal plants, countries, chemical elements, science, computing, history or books.", 'fields': {}},
        'ru': {'card': "Этого я не знаю: в моих записях такого нет, а маленькая сеть внутри меня могла бы только угадывать. Спросите о лекарственных растениях, странах, химических элементах, науке, компьютерах, истории или книгах.", 'fields': {}}}, ensure_ascii=False) + '\n')
with open(os.path.join(OUT, 'groups.json'), 'w') as f:
    json.dump({'groups': GROUPS, 'of': GROUP_OF}, f)
print('entries', len(rows), 'held entities', len(held), 'fit', len(fit), 'test held', len(test_held), 'test seen', len(test_seen))
