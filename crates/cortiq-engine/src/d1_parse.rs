//! d1's strict readings of written values (`parsing.py`): a read number's
//! magnitude (currency and percent set apart, scale words applied, the
//! decimal mark settled from evidence) and a read date as ISO — no
//! arithmetic, conversion or invented date components.

use fancy_regex::Regex;
use serde_json::{Value, json};
use std::collections::{HashMap, HashSet};
use std::sync::LazyLock;
use unicode_normalization::UnicodeNormalization;

const MONTH_NAMES: [&str; 12] = [
    "january janvier januar enero gennaio janeiro januari stycznia styczen января",
    "february fevrier februar febrero febbraio fevereiro februari lutego luty февраля",
    "march mars marz marzo marco maart marca марта",
    "april avril abril aprile kwietnia kwiecien апреля",
    "may mai mayo maggio maio mei maj maja мая",
    "june juin juni junio giugno junho czerwca czerwiec июня",
    "july juillet juli julio luglio julho lipca lipiec июля",
    "august aout agosto augustus augusti sierpnia sierpien августа",
    "september septembre septiembre settembre setembro wrzesnia wrzesien сентября",
    "october octobre oktober octubre ottobre outubro pazdziernika pazdziernik октября",
    "november novembre noviembre novembro listopada listopad ноября",
    "december decembre dezember diciembre dicembre dezembro grudnia grudzien декабря",
];

static MONTHS: LazyLock<HashMap<String, u32>> = LazyLock::new(|| {
    let mut m = HashMap::new();
    for (i, names) in MONTH_NAMES.iter().enumerate() {
        for name in names.split_whitespace() {
            m.insert(name.to_string(), i as u32 + 1);
        }
    }
    let mut abbr: HashMap<String, HashSet<u32>> = HashMap::new();
    for (name, &month) in &m {
        let prefix: String = name.chars().take(3).collect();
        abbr.entry(prefix).or_default().insert(month);
    }
    for (name, months) in abbr {
        if months.len() == 1 {
            m.entry(name).or_insert(*months.iter().next().unwrap());
        }
    }
    for (k, v) in [("janv", 1), ("fevr", 2), ("juil", 7), ("sept", 9)] {
        m.insert(k.into(), v);
    }
    m
});

static WEEKDAYS: LazyLock<HashSet<&'static str>> = LazyLock::new(|| {
    "monday tuesday wednesday thursday friday saturday sunday mon tue tues wed thu thur thurs fri sat sun \
     lundi mardi mercredi jeudi vendredi samedi dimanche montag dienstag mittwoch donnerstag freitag samstag sonntag \
     lunes martes miercoles jueves viernes sabado domingo lunedi martedi mercoledi giovedi venerdi sabato domenica \
     segunda terca quarta quinta sexta"
        .split_whitespace()
        .collect()
});

fn re(p: &str) -> Regex {
    Regex::new(p).expect("d1_parse regex")
}

static NUMERIC_DATE: LazyLock<Regex> = LazyLock::new(|| re(r"(?<!\d)(\d{1,4})([/.-])(\d{1,2})\2(\d{2,4})(?!\d)"));
static NUMERIC_DATE_FULL: LazyLock<Regex> = LazyLock::new(|| re(r"^(\d{1,4})([/.-])(\d{1,2})\2(\d{2,4})$"));
static CJK_DATE_FULL: LazyLock<Regex> =
    LazyLock::new(|| re(r"^(\d{4})\s*[年년]\s*(\d{1,2})\s*[月월]\s*(\d{1,2})\s*[日일]$"));
static WORD_DATE: LazyLock<Regex> = LazyLock::new(|| {
    let mut names: Vec<&String> = MONTHS.keys().collect();
    names.sort_by_key(|n| std::cmp::Reverse(n.chars().count()));
    let alt = names.iter().map(|n| fancy_regex::escape(n).into_owned()).collect::<Vec<_>>().join("|");
    re(&format!(
        r"\b(?:\d{{1,2}}\w*[ .]+(?:de\s+)?(?:{alt})[ ,.]+(?:de\s+)?\d{{4}}|(?:{alt})[ .]+\d{{1,2}}\w*[ ,.]+\d{{4}})\b"
    ))
});

/// Currencies by the decimal mark their amounts are written with: `""` for
/// none, `None` for both in use.
const CURRENCIES: [(Option<&str>, &[&str]); 4] = [
    (Some("."), &["$", "£", "₹", "USD", "GBP", "CAD", "AUD", "NZD", "INR", "Rs.", "Rs", "MXN", "CNY", "RMB", "元", "HKD", "SGD"]),
    (Some(","), &["R$", "BRL", "₽", "RUB", "zł", "PLN", "kr", "SEK", "NOK", "DKK", "₺", "TRY"]),
    (Some(""), &["¥", "￥", "円", "₩", "₫", "Rp.", "Rp", "IDR", "JPY", "KRW", "VND", "CLP"]),
    (None, &["€", "EUR", "CHF"]),
];

fn casefold(s: &str) -> String {
    s.to_lowercase()
}

static NAMES: LazyLock<HashMap<String, &'static str>> = LazyLock::new(|| {
    CURRENCIES.iter().flat_map(|(_, cs)| cs.iter().map(|c| (casefold(c), *c))).collect()
});
static MARK: LazyLock<HashMap<String, Option<&'static str>>> = LazyLock::new(|| {
    CURRENCIES.iter().flat_map(|(m, cs)| cs.iter().map(move |c| (casefold(c), *m))).collect()
});
static CURRENCY: LazyLock<String> = LazyLock::new(|| {
    let mut names: Vec<&str> = NAMES.values().copied().collect();
    names.sort_by_key(|n| std::cmp::Reverse(n.chars().count()));
    names.dedup();
    let alts: Vec<String> = names
        .iter()
        .map(|c| {
            let first = c.chars().next().unwrap();
            if first.is_ascii() && first.is_alphabetic() {
                format!(r"\b{}(?![a-z])", fancy_regex::escape(c))
            } else {
                fancy_regex::escape(c).into_owned()
            }
        })
        .collect();
    format!("(?i:{})", alts.join("|"))
});
static CURRENCY_RE: LazyLock<Regex> = LazyLock::new(|| re(&CURRENCY));

const SCALE_WORDS: &[(&str, u64)] = &[
    ("thousand", 1_000), ("mil", 1_000), ("tausend", 1_000), ("тыс", 1_000), ("千", 1_000),
    ("万", 10_000), ("萬", 10_000), ("lakh", 100_000), ("lakhs", 100_000), ("crore", 10_000_000),
    ("crores", 10_000_000), ("million", 1_000_000), ("millions", 1_000_000), ("mio", 1_000_000),
    ("mln", 1_000_000), ("millón", 1_000_000), ("millones", 1_000_000), ("milhão", 1_000_000),
    ("milhões", 1_000_000), ("milione", 1_000_000), ("milioni", 1_000_000), ("млн", 1_000_000),
    ("亿", 100_000_000), ("億", 100_000_000), ("billion", 1_000_000_000), ("billions", 1_000_000_000),
    ("milliard", 1_000_000_000), ("milliards", 1_000_000_000), ("milliarde", 1_000_000_000),
    ("milliarden", 1_000_000_000), ("mrd", 1_000_000_000), ("miliardo", 1_000_000_000),
    ("miliardi", 1_000_000_000), ("млрд", 1_000_000_000), ("trillion", 1_000_000_000_000),
];

static SCALE_RE: LazyLock<Regex> = LazyLock::new(|| {
    let mut words: Vec<&str> = SCALE_WORDS.iter().map(|(w, _)| *w).collect();
    words.sort_by_key(|w| std::cmp::Reverse(w.chars().count()));
    words.dedup();
    let alt = words.iter().map(|w| fancy_regex::escape(w).into_owned()).collect::<Vec<_>>().join("|");
    re(&format!(r"(?i)\s*({alt})\.?$"))
});
static SUFFIX_RE: LazyLock<Regex> = LazyLock::new(|| re(r"(?i)(?<=\d)(k|m|bn|b)$"));
static COUNTED: LazyLock<Regex> = LazyLock::new(|| {
    re(r"(?i)\b(?:money|price|invoice|revenue|cost|count|quantity|population|people|items|units|users|downloads)\b")
});
static GROUPED: LazyLock<Regex> = LazyLock::new(|| {
    re(r"(?<![\d.,])\d{1,3}(?:([,. \x{00a0}\x{202f}'’])\d{3})+(?!\1)([.,])\d+(?![\d.,])")
});
static SHORT: LazyLock<Regex> = LazyLock::new(|| re(r"(?<![\d.,])\d+([.,])\d{1,2}(?![\d.,])"));
static REPEATED: LazyLock<Regex> = LazyLock::new(|| re(r"(?<![\d.,])\d{1,3}(?:([.,])\d{3}){2,}(?![\d.,])"));
static LOCALE: LazyLock<Regex> = LazyLock::new(|| {
    re(r#"\b(?:locale|language|lang)["']?\s*[=:]\s*["']?(en|fr|de|es|it|pt|nl|pl|ru|sv|ja|zh|ko)(?:[-_][a-z]{2})?\b"#)
});

fn scale_of(word: &str) -> u64 {
    let w = word.to_lowercase();
    SCALE_WORDS.iter().find(|(k, _)| *k == w).map(|(_, v)| *v).unwrap_or(0)
}

fn one<T: Clone + Eq + std::hash::Hash>(values: &HashSet<T>) -> Option<T> {
    if values.len() == 1 { values.iter().next().cloned() } else { None }
}

/// Only an explicitly written percent or currency marker.
pub fn written_unit(text: &str) -> Option<String> {
    if text.trim_end().ends_with('%') {
        return Some("%".into());
    }
    let markers: Vec<String> = CURRENCY_RE
        .find_iter(text)
        .filter_map(|m| m.ok())
        .map(|m| m.as_str().to_string())
        .collect();
    if markers.len() == 1 {
        NAMES.get(&casefold(&markers[0])).map(|s| s.to_string())
    } else {
        None
    }
}

fn currency_marks(text: &str) -> HashSet<String> {
    CURRENCY_RE
        .find_iter(text)
        .filter_map(|m| m.ok())
        .filter_map(|m| MARK.get(&casefold(m.as_str())).copied().flatten())
        .map(str::to_string)
        .collect()
}

fn decimal_mark(written: &str, context: &str) -> Option<String> {
    let context = re(r"<\?[^>]*\?>").replace_all(context, "").into_owned();
    let mut numbers: HashSet<String> = HashSet::new();
    for c in GROUPED.captures_iter(&context).flatten() {
        numbers.insert(c.get(2).unwrap().as_str().to_string());
    }
    for c in SHORT.captures_iter(&context).flatten() {
        numbers.insert(c.get(1).unwrap().as_str().to_string());
    }
    for c in REPEATED.captures_iter(&context).flatten() {
        numbers.insert(if c.get(1).unwrap().as_str() == "." { ",".into() } else { ".".into() });
    }
    let folded = casefold(&context);
    let locales: HashSet<String> = LOCALE
        .captures_iter(&folded)
        .flatten()
        .map(|c| {
            let lang = c.get(1).unwrap().as_str();
            if ["en", "ja", "zh", "ko"].contains(&lang) { ".".into() } else { ",".into() }
        })
        .collect();
    let candidates = [currency_marks(written), numbers, currency_marks(&context), locales];
    let first = candidates.iter().find(|m| !m.is_empty()).cloned().unwrap_or_default();
    one(&first)
}

fn digits(text: &str, written: &str, context: &str) -> Option<String> {
    if !re(r"^\d[\d., '’]*$").is_match(text).unwrap_or(false) {
        return None;
    }
    if re(r"[ '’]").is_match(text).unwrap_or(false) {
        let g = re(r"^(\d{1,3}(?:[ '’]\d{3})+)([.,]\d+)?$").captures(text).ok().flatten()?;
        let joined = re(r"[ '’]").replace_all(g.get(1)?.as_str(), "").into_owned()
            + g.get(2).map_or("", |m| m.as_str());
        return digits(&joined, written, context);
    }
    let both = format!("{written}\n{context}");
    if re(r"^\d{1,2}(?:,\d{2})+,\d{3}(?:\.\d+)?$").is_match(text).unwrap_or(false)
        && re(r"(?i)₹|\bINR\b|\bRs\b|lakh|crore").is_match(&both).unwrap_or(false)
    {
        return Some(text.replace(',', ""));
    }
    if text.contains('.') && text.contains(',') {
        let (point, group) = if text.rfind('.') > text.rfind(',') { (".", ",") } else { (",", ".") };
        let pat = format!(r"^\d{{1,3}}(?:{}\d{{3}})+{}\d+$", fancy_regex::escape(group), fancy_regex::escape(point));
        if !re(&pat).is_match(text).unwrap_or(false) {
            return None;
        }
        return Some(text.replace(group, "").replace(point, "."));
    }
    if !text.contains('.') && !text.contains(',') {
        return Some(text.to_string());
    }
    let mark = if text.contains('.') { '.' } else { ',' };
    let parts: Vec<&str> = text.split(mark).collect();
    if parts.iter().any(|p| p.is_empty()) {
        return None;
    }
    let point = decimal_mark(written, context);
    if parts.len() > 2 || point.as_deref().is_some_and(|p| p != mark.to_string()) {
        let ok = (1..=3).contains(&parts[0].len()) && parts[1..].iter().all(|p| p.len() == 3);
        return ok.then(|| parts.concat());
    }
    if point.is_none() && parts[1].len() == 3 && parts[0].len() <= 3 {
        return None;
    }
    Some(parts.join("."))
}

/// The written magnitude (percent points included): currency and percent
/// set apart, a scale word applied; null when the writing does not settle
/// one value.
pub fn parse_number(text: &str, context: &str, unit: Option<&str>) -> Value {
    let mut context = context.to_string();
    if let Some(u) = unit {
        let full = format!("^(?:{})$", *CURRENCY);
        if re(&full).is_match(u.trim()).unwrap_or(false) {
            context.push('\n');
            context.push_str(u);
        }
    }
    let written = text;
    let mut t = text.trim().replace('\u{2212}', "-").replace(['\u{00a0}', '\u{202f}'], " ");
    if let Some(s) = t.strip_suffix('%') {
        t = s.to_string();
    }
    let mut t = t.trim_end().to_string();
    let negative = t.starts_with('(') && t.ends_with(')');
    if negative {
        t = t[1..t.len() - 1].trim().to_string();
    }
    let (mut sign, mut currency): (Option<char>, Option<String>) = (None, None);
    let both = re(&format!(r"^(?:({c})\s*(.*)|(.*?)\s*({c}))$", c = *CURRENCY));
    for _ in 0..2 {
        if currency.is_none() {
            if let Ok(Some(m)) = both.captures(&t) {
                if let Some(c) = m.get(1) {
                    currency = Some(c.as_str().to_string());
                    t = m.get(2).map_or("", |x| x.as_str()).trim().to_string();
                } else if let Some(c) = m.get(4) {
                    currency = Some(c.as_str().to_string());
                    t = m.get(3).map_or("", |x| x.as_str()).trim().to_string();
                }
            }
        }
        if sign.is_none() && (t.starts_with('-') || t.starts_with('+')) {
            sign = t.chars().next();
            t = t[1..].trim().to_string();
        }
    }
    if negative && sign.is_some() {
        return Value::Null;
    }
    let mut scale: u64 = 1;
    if let Ok(Some(w)) = SCALE_RE.captures(&t) {
        let word = w.get(1).unwrap().as_str();
        scale = if word == "Billion" { 0 } else { scale_of(word) };
        t = t[..w.get(0).unwrap().start()].to_string();
    } else if let Ok(Some(s)) = SUFFIX_RE.captures(&t) {
        let both = format!("{written}\n{context}");
        if !(CURRENCY_RE.is_match(&both).unwrap_or(false) || COUNTED.is_match(&context).unwrap_or(false)) {
            return Value::Null;
        }
        scale = match s.get(1).unwrap().as_str().to_lowercase().as_str() {
            "k" => 1_000,
            "m" => 1_000_000,
            _ => 1_000_000_000,
        };
        t = t[..s.get(0).unwrap().start()].to_string();
    }
    let Some(d) = digits(&t, written, &context) else {
        return Value::Null;
    };
    if scale == 0 {
        return Value::Null;
    }
    let sign = if negative { "-" } else if sign == Some('-') { "-" } else { "" };
    // Decimal arithmetic on the digit string: an integer stays one.
    let (int_part, frac) = d.split_once('.').unwrap_or((&d, ""));
    let frac_trim = frac.trim_end_matches('0');
    let shift = scale.to_string().len() - 1; // scales are powers of ten
    if frac_trim.len() <= shift {
        let mut s = format!("{int_part}{frac_trim}");
        s.push_str(&"0".repeat(shift - frac_trim.len()));
        let s = s.trim_start_matches('0');
        if let Ok(v) = s.parse::<i64>() {
            return json!(if sign == "-" { -v } else { v });
        }
    }
    let v: f64 = format!("{sign}{int_part}.{frac}").parse::<f64>().unwrap_or(f64::NAN) * scale as f64;
    if v.is_finite() { json!(v) } else { Value::Null }
}

fn fold(text: &str) -> String {
    casefold(text).nfkd().filter(|c| !unicode_normalization::char::is_combining_mark(*c)).collect()
}

fn valid_date(y: i32, m: u32, d: u32) -> Option<String> {
    let days = match m {
        1 | 3 | 5 | 7 | 8 | 10 | 12 => 31,
        4 | 6 | 9 | 11 => 30,
        2 if (y % 4 == 0 && y % 100 != 0) || y % 400 == 0 => 29,
        2 => 28,
        _ => return None,
    };
    (y >= 1 && (1..=days).contains(&d)).then(|| format!("{y:04}-{m:02}-{d:02}"))
}

fn date_order(context: &str, folded: &str) -> Option<String> {
    let (day, month, year) = (
        r"(?:dd?|jj?|tt?|gg?|day|jour|tag|dia|giorno)",
        r"(?:mm?|month|mois|monat|mes|mese)",
        r"(?:yy(?:yy)?|aa(?:aa)?|jj(?:jj)?|year|annee|jahr|ano|anno)",
    );
    let formats = [
        (format!(r"\b{day}\s*[/.-]\s*{month}\s*[/.-]\s*{year}\b"), "dmy"),
        (format!(r"\b{month}\s*[/.-]\s*{day}\s*[/.-]\s*{year}\b"), "mdy"),
    ];
    let explicit: HashSet<String> = formats
        .iter()
        .filter(|(p, _)| re(p).is_match(folded).unwrap_or(false))
        .map(|(_, o)| o.to_string())
        .collect();
    if !explicit.is_empty() {
        return one(&explicit);
    }
    let loc = re(r#"\b(?:locale|language|lang|langue|sprache|idioma|lingua)["']?\s*[=:]\s*["']?(en(?:[-_](?:us|gb|au|ca))?|fr(?:[-_]fr)?|de(?:[-_]de)?|es(?:[-_]es)?|it(?:[-_]it)?)\b"#);
    let locales: HashSet<String> = loc
        .captures_iter(folded)
        .flatten()
        .map(|c| {
            let l = c.get(1).unwrap().as_str().replace('_', "-");
            if l == "en" || l == "en-us" { "mdy".into() } else { "dmy".into() }
        })
        .collect();
    if !locales.is_empty() {
        return one(&locales);
    }
    let mut orders = HashSet::new();
    for m in NUMERIC_DATE.captures_iter(context).flatten() {
        let (a, b): (u32, u32) = (m[1].parse().unwrap_or(0), m[3].parse().unwrap_or(0));
        let year: i32 = m[4].parse().unwrap_or(0);
        if m[4].len() != 4 || a > 31 || b > 31 || a < 1 || b < 1 {
            continue;
        }
        for (d, mo, order) in [(a, b, "dmy"), (b, a, "mdy")] {
            if valid_date(year, mo, d).is_some() && ((a > 12) != (b > 12)) {
                orders.insert(order.to_string());
            }
        }
    }
    one(&orders)
}

/// ISO value and an unresolved-ambiguity flag (`parse_date`).
pub fn parse_date(text: &str, context: &str) -> (Option<String>, bool) {
    let folded = fold(context);
    let order = date_order(context, &folded);
    parse_date_in(text, context, &folded, order.as_deref())
}

fn parse_date_in(text: &str, context: &str, folded: &str, order: Option<&str>) -> (Option<String>, bool) {
    let text = text.trim();
    if text.is_empty() {
        return (None, false);
    }
    let timed = re(r"^(.+?)[T\s]+(\d{1,2}:\d{2}(?::\d{2}(?:\.\d+)?)?(?:Z|[+-]\d{2}:\d{2})?)$");
    if let Ok(Some(c)) = timed.captures(text) {
        let (date, ambiguous) = parse_date_in(&c[1], context, folded, order);
        let Some(date) = date else {
            return (None, ambiguous);
        };
        // datetime.fromisoformat(...).isoformat(): HH:MM:SS, the zone as ±HH:MM.
        let mut clock = c[2].to_string();
        if clock.as_bytes().get(1) == Some(&b':') {
            clock = format!("0{clock}");
        }
        let split = clock.find(['Z', '+', '-']).unwrap_or(clock.len());
        let (time, zone) = clock.split_at(split);
        let zone = if zone == "Z" { "+00:00" } else { zone };
        let time = if time.len() == 5 { format!("{time}:00") } else { time.to_string() };
        return (Some(format!("{date}T{time}{zone}")), false);
    }
    if let Ok(Some(m)) = NUMERIC_DATE_FULL.captures(text) {
        let (a, b, c): (i32, i32, i32) = (m[1].parse().unwrap(), m[3].parse().unwrap(), m[4].parse().unwrap());
        let candidates: Vec<(i32, i32, i32)> = if m[1].len() == 4 {
            vec![(a, b, c)]
        } else if m[4].len() == 4 {
            cands(c, a, b, order)
        } else {
            if m[4].len() != 2 {
                return (None, false);
            }
            let mut complete: Vec<String> = NUMERIC_DATE
                .captures_iter(context)
                .flatten()
                .filter(|x| x[1].len() == 4 || x[4].len() == 4)
                .map(|x| x[0].to_string())
                .collect();
            complete.extend(WORD_DATE.find_iter(folded).flatten().map(|x| x.as_str().to_string()));
            let mut centuries = HashSet::new();
            for cand in complete {
                if let (Some(v), false) = parse_date_in(&cand, context, folded, order) {
                    centuries.insert(v[..4].parse::<i32>().unwrap() / 100 * 100);
                }
            }
            if centuries.len() != 1 {
                return (None, false);
            }
            cands(centuries.into_iter().next().unwrap() + c, a, b, order)
        };
        let values: HashSet<String> = candidates
            .iter()
            .filter_map(|&(y, mo, d)| (mo > 0 && d > 0).then(|| valid_date(y, mo as u32, d as u32)).flatten())
            .collect();
        if values.len() > 1 {
            return (None, true);
        }
        return (values.into_iter().next(), false);
    }
    if let Ok(Some(c)) = CJK_DATE_FULL.captures(text) {
        return (valid_date(c[1].parse().unwrap(), c[2].parse().unwrap(), c[3].parse().unwrap()), false);
    }
    let mut words = fold(text);
    words = re(r"(?<=\d)(?:st|nd|rd|th|er)\b").replace_all(&words, "").into_owned();
    words = re(r"(?<=\d)\.?(?:o|a|°)\b").replace_all(&words, "").into_owned();
    words = re(r"\b(?:de|of|the|le|el|il|den)\b").replace_all(&words, " ").into_owned();
    words = re(r"[,.]").replace_all(&words, " ").into_owned();
    let mut w: Vec<&str> = words.split_whitespace().collect();
    if let Some(first) = w.first() {
        if WEEKDAYS.contains(first.trim_end_matches("-feira")) {
            w.remove(0);
        }
    }
    if w.len() != 3 {
        return (None, false);
    }
    let (month, day, year) = if let Some(&m) = MONTHS.get(w[0]) {
        (m, w[1], w[2])
    } else if let Some(&m) = MONTHS.get(w[1]) {
        (m, w[0], w[2])
    } else {
        return (None, false);
    };
    if !day.chars().all(|c| c.is_ascii_digit()) || !year.chars().all(|c| c.is_ascii_digit()) || year.len() != 4 {
        return (None, false);
    }
    (valid_date(year.parse().unwrap(), month, day.parse().unwrap_or(0)), false)
}

fn cands(year: i32, a: i32, b: i32, order: Option<&str>) -> Vec<(i32, i32, i32)> {
    let (dmy, mdy) = ((year, b, a), (year, a, b));
    match order {
        None => vec![dmy, mdy],
        Some("dmy") => vec![dmy],
        Some(_) => vec![mdy],
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const INVOICE: &str = "ACME SA\nInvoice 2026-117\nTotal due: 1 234,50 EUR\nPay before 03/04/2026 (day/month/year)";

    #[test]
    fn invoice_values_read_like_the_reference() {
        assert_eq!(parse_number("1 234,50 EUR", INVOICE, None), json!(1234.5));
        assert_eq!(written_unit("1 234,50 EUR").as_deref(), Some("EUR"));
        assert_eq!(parse_date("03/04/2026", INVOICE), (Some("2026-04-03".into()), false));
        assert_eq!(parse_number("4471", "", None), json!(4471));
    }

    #[test]
    fn numbers_and_dates() {
        assert_eq!(parse_number("$1.5M", "", None), json!(1500000));
        assert_eq!(parse_number("12%", "", None), json!(12));
        assert_eq!(parse_number("1,250", "", None), Value::Null);
        assert_eq!(parse_number("-3.25", "", None), json!(-3.25));
        assert_eq!(parse_date("20 octobre 2026", "").0.as_deref(), Some("2026-10-20"));
        assert_eq!(parse_date("2026年10月20日", "").0.as_deref(), Some("2026-10-20"));
        assert_eq!(parse_date("03/04/2026", ""), (None, true));
    }
}
