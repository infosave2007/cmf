#!/usr/bin/env python3
# The two classes of the keyless decision on the whole-message φ: questions for a specific fact (the small network has no facts)
# and ordinary chat. The test sets use phrasing the fit never saw. usage: mk_keyless.py <dir>
import json, os, random, re, sys
D = sys.argv[1]
rng=random.Random(9)
WH_EN="""What is {a} plus {b}?
What is {a}+{b}?
How much is {a} times {b}?
What is {a} minus {b}?
Calculate {a} * {b}.
What is the square root of {c}?
How many {u1} are in a {u2}?
What is the population of {place}?
What is the area of {place}?
What is the largest city in {place}?
Who is the president of {place}?
Who is the prime minister of {place}?
What is the national dish of {place}?
What is the time zone of {place}?
When was {thing} invented?
Who invented {thing}?
How does {thing} work exactly and who made it?
How tall is {tall}?
How old is {tall}?
Where is {tall} located?
When was {person} born?
When did {person} die?
What did {person} win?
Who was the wife of {person}?
How many children did {person} have?
What is the boiling point of {subst}?
What is the melting point of {subst}?
What is the chemical formula of {subst}?
What is the density of {subst}?
Who won the {sport} world cup in {year}?
Who won the {sport} championship in {year}?
What happened in {year}?
Who was the king of {place} in {year}?
What is the distance between {place} and {place2}?
How far is {place} from {place2}?
What is the exchange rate of the {cur}?
What is the stock price of {firm}?
Who founded {firm}?
When was {firm} founded?
What is the phone number of {firm}?
What is the weather in {place}?
What is the name of the {nth} president?
What is the {nth} planet from the sun?
What is the {nth} element of the periodic table?
Which is bigger, {place} or {place2}?
What is the speed of {animal}?
How long does {animal} live?
How much does {animal} weigh?
Can I take {drug} with {drug2}?
Can I take {drug} with alcohol?
Is {drug} safe during pregnancy?
What is the dose of {drug}?
What is the maximum dose of {drug} per day?
What are the side effects of {drug}?
What are the symptoms of {disease}?
How do I treat {disease}?
What is the best medicine for {disease}?
Is {disease} dangerous?
Is {disease} contagious?
Should I see a doctor for {disease}?
What should I take for {symptom}?
What helps with {symptom}?
What medicine is good for {symptom}?
How do I get rid of {symptom}?
How can I cure {symptom}?
Is it legal to {act} in {place}?
What is the fine for {act} in {place}?""".split('\n')
WH_RU="""Сколько будет {a} плюс {b}?
Сколько будет {a}+{b}?
Сколько будет {a} умножить на {b}?
Чему равно {a} минус {b}?
Посчитай {a} * {b}.
Чему равен квадратный корень из {c}?
Какое население в {rplace}?
Какая площадь у {rplace}?
Какой самый большой город в {rplace}?
Кто президент в {rplace}?
Кто премьер-министр в {rplace}?
Когда изобрели {rthing}?
Кто изобрёл {rthing}?
Какая высота у {rtall}?
Где находится {rtall}?
Когда родился {rperson}?
Когда умер {rperson}?
Что получил {rperson}?
Сколько детей было у {rperson}?
Какая температура кипения у {rsubst}?
Какая температура плавления у {rsubst}?
Какая химическая формула у {rsubst}?
Какая плотность у {rsubst}?
Кто выиграл чемпионат мира по {rsport} в {year} году?
Что произошло в {year} году?
Кто правил в {rplace} в {year} году?
Какое расстояние от {rplace} до {rplace2}?
Какой курс {rcur}?
Какая цена акций {firm}?
Кто основал {firm}?
Когда основали {firm}?
Какая погода в {rplace}?
Какая скорость у {ranimal}?
Сколько живёт {ranimal}?
Сколько весит {ranimal}?
Можно ли принимать {rdrug} вместе с {rdrug2}?
Можно ли пить {rdrug} с алкоголем?
Можно ли {rdrug} при беременности?
Какая дозировка у лекарства {rdrug}?
Какая максимальная доза в сутки для {rdrug}?
Какие побочные эффекты у лекарства {rdrug}?
Какие симптомы у болезни {rdisease}?
Как лечить {rdisease}?
Какое лекарство лучше при болезни {rdisease}?
Опасна ли болезнь {rdisease}?
Заразна ли болезнь {rdisease}?
Нужно ли идти к врачу при болезни {rdisease}?
Что принять от {rsymptom}?
Что помогает от {rsymptom}?
Какое лекарство от {rsymptom}?
Чем лечить {rsymptom2}?
Как избавиться от {rsymptom}?""".split('\n')
V=dict(place="Bavaria;Texas;Siberia;Tokyo;Moscow;Quebec;Sicily;Bali;Lagos;Mumbai;Kyoto;Crimea;Tuscany;Alaska;Ontario".split(';'),
 thing="the bicycle;the zipper;the microwave;the ballpoint pen;the dishwasher;the elevator;the umbrella;the radio;the steam engine;the helicopter;the printing press;the sewing machine".split(';'),
 tall="the Eiffel Tower;the Burj Khalifa;Big Ben;the Statue of Liberty;Mont Blanc;Kilimanjaro;the Great Pyramid;Elbrus".split(';'),
 person="Mozart;Beethoven;Napoleon;Picasso;Churchill;Lincoln;Pushkin;Gagarin;Marie Antoinette;Rembrandt;Tesla;Edison".split(';'),
 subst="ethanol;ammonia;methane;acetone;glycerin;salt;sugar;steel;bronze".split(';'),
 sport="football;hockey;tennis;basketball;chess".split(';'),
 year=[str(y) for y in (1812,1905,1917,1945,1969,1986,1991,2001,2008,2014,2018)],
 cur="dollar;yuan;peso;rupee;pound".split(';'), firm="Toyota;Samsung;Nokia;IKEA;Boeing;Sony;Tesla;Adobe".split(';'),
 nth="third;fifth;seventh;tenth;sixteenth".split(';'), animal="a horse;a lion;an elephant;a dolphin;a parrot;a whale;a tiger".split(';'),
 u1="feet;inches;ounces;cups;grams;yards".split(';'), u2="mile;yard;pound;gallon;kilogram;meter".split(';'),
 symptom="a headache;a cough;a fever;heartburn;nausea;a sore throat;insomnia;back pain;a toothache;a runny nose".split(';'),
 rsymptom="головной боли;кашля;температуры;изжоги;тошноты;боли в горле;бессонницы;боли в спине;зубной боли;насморка".split(';'),
 rsymptom2="кашель;температуру;изжогу;тошноту;боль в горле;бессонницу;боль в спине;зубную боль;насморк".split(';'),
 drug="aspirin;ibuprofen;paracetamol;amoxicillin;metformin;warfarin;melatonin;omeprazole".split(';'),
 disease="the flu;diabetes;asthma;migraine;gout;hypertension;pneumonia;measles".split(';'),
 act="drive at night;record a call;carry a knife;drink in public;fly a drone".split(';'),
 rdrug="аспирин;ибупрофен;парацетамол;амоксициллин;метформин;варфарин;мелатонин;омепразол".split(';'),
 rdisease="грипп;диабет;астма;мигрень;подагра;гипертония;пневмония;корь".split(';'),
 rplace="Баварии;Техасе;Сибири;Токио;Москве;Крыму;Татарстане;Якутии;Мумбаи".split(';'),
 rthing="велосипед;молнию;микроволновку;лифт;зонт;радио;паровоз;вертолёт;швейную машинку".split(';'),
 rtall="Эйфелева башня;Останкинская башня;Эльбрус;Биг-Бен;Статуя Свободы".split(';'),
 rperson="Моцарт;Наполеон;Пушкин;Гагарин;Толстой;Менделеев;Королёв".split(';'),
 rsubst="этанола;аммиака;метана;ацетона;глицерина;соли;стали".split(';'),
 rsport="футболу;хоккею;теннису;шахматам".split(';'),rcur="доллара;юаня;евро;рубля".split(';'),
 ranimal="лошадь;лев;слон;дельфин;попугай;кит;тигр".split(';'))
def fill(t):
    d={k:rng.choice(v) for k,v in V.items()}
    d['place2']=rng.choice(V['place']); d['rplace2']=rng.choice(V['rplace']); d['drug2']=rng.choice(V['drug']); d['rdrug2']=rng.choice(V['rdrug'])
    d['a']=rng.randint(0,99); d['b']=rng.randint(0,99); d['c']=rng.choice([4,9,16,25,36,49,64,81,100,144,225,2,3,10])
    return t.format(**d)
pos=[]
for t in WH_EN+WH_RU:
    pos+=list({fill(t) for _ in range(40)})
rng.shuffle(pos)
# the test uses phrasing not in the fit
TEST_POS="""What is 2+2?
What is 7 times 8?
How much is 15 plus 27?
What's 100 divided by 4?
2+2=?
Сколько будет 2+2?
Сколько будет 7 на 8?
Сколько получится, если 15 разделить на 3?
How many people live in Berlin?
What is the GDP of Brazil?
Who is the richest person in the world?
Who is the current president of the United States?
What year did the Titanic sink?
When did the Roman Empire fall?
What is the tallest building in the world?
How many countries are in Africa?
What is the fastest car in the world?
Who wrote the song Imagine?
Who directed the movie Titanic?
What is the name of Harry Potter's owl?
How many strings does a violin have?
What is the freezing point of mercury?
Сколько человек живёт в Берлине?
Какой ВВП у Бразилии?
Кто самый богатый человек в мире?
В каком году утонул Титаник?
Когда пала Римская империя?
Какое самое высокое здание в мире?
Сколько стран в Африке?
Кто снял фильм Титаник?
Кто поёт песню Yesterday?
Сколько струн у скрипки?
Какой сегодня курс доллара?
Кто выиграл Евровидение в 2015 году?
Какая столица Баварии?
Какая столица Сибири?
Is it safe?
Can I take aspirin with alcohol?
How much vitamin C can I take per day?
What medicine can I use for stomach cramps?
What are the symptoms of covid?
Можно ли пить ибупрофен на голодный желудок?
Какие симптомы у ангины?
Чем снять спазмы в животе?""".split('\n')
TEST_NEG="""Write a poem about the sea.
Tell me a story about a dragon.
Write an email to my boss asking for a day off.
Give me some advice on how to save money.
How can I become more confident?
How do I make friends in a new city?
I feel sad today.
I am bored, what should I do?
What do you think about music?
Can you help me write a resume?
Write a short motivational speech.
Suggest a name for my cat.
Write a Python function that reverses a string.
Explain how to tie a tie.
What should I cook tonight?
Do you like cats?
Let's talk about books.
Can you be my friend?
Why are people afraid of the dark?
How can I sleep better?
Напиши стихотворение о море.
Расскажи сказку про дракона.
Напиши письмо начальнику с просьбой об отгуле.
Дай совет, как экономить деньги.
Как стать увереннее?
Как найти друзей в новом городе?
Мне сегодня грустно.
Мне скучно, чем заняться?
Что ты думаешь о музыке?
Помоги составить резюме.
Напиши короткую мотивирующую речь.
Придумай имя для кота.
Напиши функцию на Python, которая переворачивает строку.
Объясни, как завязать галстук.
Что приготовить на ужин?
Ты любишь кошек?
Давай поговорим о книгах.
Посоветуй, как быстрее выучить английский.
Почему люди боятся темноты?
Как лучше спать?""".split('\n')
R=os.path.expanduser('~/embryo-data/o1/chat-sft/')
gen=[json.loads(l)['prompt'] for l in open(R+'router-fit2/general-balanced.jsonl')]
FACT=re.compile(r"^\s*(what is|what are|who|when|where|how many|how much|how far|how tall|how old|which|сколько|кто|когда|где|какой|какая|какое|какие|чему равн)\b",re.I)
neg=[g for g in gen if len(g)<400 and not FACT.match(g)]
factual_in_chat=[g for g in gen if len(g)<120 and FACT.match(g)]
rng.shuffle(neg); rng.shuffle(factual_in_chat)
def w(n,rows):
    with open(os.path.join(D,n),'w') as f:
        for p in rows: f.write(json.dumps({'prompt':p},ensure_ascii=False)+'\n')
w('kl-fact.jsonl',pos); w('kl-chat.jsonl',neg[:3000]); w('kl-test-fact.jsonl',TEST_POS); w('kl-test-chat.jsonl',TEST_NEG)
