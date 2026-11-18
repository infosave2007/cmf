//! Built-in tools: exact functions the engine runs itself when a model calls
//! them in the Hermes function-calling contract of the Embryo cores
//! (`<tool_call>{"name": …, "arguments": …}</tool_call>` in an assistant
//! turn, the result back as `<tool_response>…</tool_response>` in a `tool`
//! turn). A small model fills the call's fields the way it fills a card's
//! placeholders; the counting and the arithmetic, which it does badly, are
//! done here, exactly.
//!
//! - `calc`: an expression over exact rationals (`i128` numerator and
//!   denominator, overflow is an error, never a rounded result): `+ - * / % ^`,
//!   parentheses, `sum min max mean abs round floor ceil`.
//! - `table`: the last command output read as a [`Table`]: `count sum min max
//!   mean get argmax argmin diff` over a column, with a row-name filter
//!   (`rows`) and a cell test (`where`).
//!
//! [`round`] runs the calls of one assistant turn when every one of them is
//! built in and renders the tool turns that follow it; the server loops on it
//! (`/v1/completions` with `builtin_tools`). The descriptions the model is
//! shown are `tools.json` ([`descriptions`]); the trace generators read the
//! same file.

use serde_json::Value;
use std::cmp::Ordering;

/// The model-facing descriptions (`{"name", "description", "parameters"}`).
pub const TOOLS_JSON: &str = include_str!("tools.json");

pub fn descriptions() -> Vec<Value> {
    serde_json::from_str(TOOLS_JSON).expect("tools.json is valid JSON")
}

pub fn is_builtin(name: &str) -> bool {
    matches!(name, "calc" | "table")
}

/// Runs a built-in tool. `input` is the text `table` reads (see
/// [`last_output`]); `calc` ignores it.
pub fn call(name: &str, args: &Value, input: &str) -> Result<String, String> {
    match name {
        "calc" => {
            let e = args
                .get("expression")
                .or_else(|| args.get("expr"))
                .and_then(Value::as_str)
                .ok_or("calc needs an expression")?;
            Ok(calc(e)?.show())
        }
        "table" => table(args, input),
        other => Err(format!("{other} is not a built-in tool")),
    }
}

// ---------------------------------------------------------------- rationals

/// An exact rational: `n / d`, `d > 0`, reduced.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Q {
    n: i128,
    d: i128,
}

fn gcd(mut a: i128, mut b: i128) -> i128 {
    a = a.abs();
    b = b.abs();
    while b != 0 {
        (a, b) = (b, a % b);
    }
    a
}

const OVERFLOW: &str = "the number is too large for exact arithmetic";

impl Q {
    pub fn int(n: i128) -> Q {
        Q { n, d: 1 }
    }

    pub fn new(n: i128, d: i128) -> Result<Q, String> {
        if d == 0 {
            return Err("division by zero".into());
        }
        let g = gcd(n, d).max(1);
        let (mut n, mut d) = (n / g, d / g);
        if d < 0 {
            n = n.checked_neg().ok_or(OVERFLOW)?;
            d = d.checked_neg().ok_or(OVERFLOW)?;
        }
        Ok(Q { n, d })
    }

    fn add(self, o: Q) -> Result<Q, String> {
        let n = self
            .n
            .checked_mul(o.d)
            .and_then(|a| o.n.checked_mul(self.d).and_then(|b| a.checked_add(b)))
            .ok_or(OVERFLOW)?;
        Q::new(n, self.d.checked_mul(o.d).ok_or(OVERFLOW)?)
    }

    fn neg(self) -> Result<Q, String> {
        Ok(Q { n: self.n.checked_neg().ok_or(OVERFLOW)?, d: self.d })
    }

    fn sub(self, o: Q) -> Result<Q, String> {
        self.add(o.neg()?)
    }

    fn mul(self, o: Q) -> Result<Q, String> {
        let g1 = gcd(self.n, o.d).max(1);
        let g2 = gcd(o.n, self.d).max(1);
        let n = (self.n / g1).checked_mul(o.n / g2).ok_or(OVERFLOW)?;
        let d = (self.d / g2).checked_mul(o.d / g1).ok_or(OVERFLOW)?;
        Q::new(n, d)
    }

    fn div(self, o: Q) -> Result<Q, String> {
        if o.n == 0 {
            return Err("division by zero".into());
        }
        self.mul(Q::new(o.d, o.n)?)
    }

    fn floor(self) -> Q {
        Q::int(self.n.div_euclid(self.d))
    }

    fn ceil(self) -> Q {
        let f = self.n.div_euclid(self.d);
        Q::int(if f * self.d == self.n { f } else { f + 1 })
    }

    fn rem(self, o: Q) -> Result<Q, String> {
        if o.n == 0 {
            return Err("modulo by zero".into());
        }
        self.sub(o.mul(self.div(o)?.floor())?)
    }

    fn pow(self, e: Q) -> Result<Q, String> {
        if e.d != 1 {
            return Err("the exponent must be a whole number".into());
        }
        if e.n.abs() > 256 {
            return Err(OVERFLOW.into());
        }
        let mut r = Q::int(1);
        for _ in 0..e.n.abs() {
            r = r.mul(self)?;
        }
        if e.n < 0 { Q::int(1).div(r) } else { Ok(r) }
    }

    fn abs(self) -> Q {
        Q { n: self.n.abs(), d: self.d }
    }

    /// Half away from zero to `digits` decimals.
    fn round(self, digits: i128) -> Result<Q, String> {
        if !(0..=18).contains(&digits) {
            return Err("round takes 0 to 18 decimals".into());
        }
        let s = 10i128.pow(digits as u32);
        let x = self.mul(Q::int(s))?;
        let half = Q::new(1, 2)?;
        let r = if x.n >= 0 { x.add(half)?.floor() } else { x.neg()?.add(half)?.floor().neg()? };
        r.div(Q::int(s))
    }

    fn to_f64(self) -> f64 {
        self.n as f64 / self.d as f64
    }

    /// An integer as itself, a finite decimal exactly, any other fraction as
    /// `n/d ≈ decimal`.
    pub fn show(self) -> String {
        if self.d == 1 {
            return self.n.to_string();
        }
        let (mut d, mut k2, mut k5) = (self.d, 0u32, 0u32);
        while d % 2 == 0 {
            d /= 2;
            k2 += 1;
        }
        while d % 5 == 0 {
            d /= 5;
            k5 += 1;
        }
        let k = k2.max(k5);
        if d == 1 && k <= 30 {
            if let Some(scaled) = 10i128.checked_pow(k).and_then(|p| self.n.checked_mul(p / self.d)) {
                let neg = scaled < 0;
                let digits = scaled.unsigned_abs().to_string();
                let digits = format!("{digits:0>width$}", width = k as usize + 1);
                let (int, frac) = digits.split_at(digits.len() - k as usize);
                let frac = frac.trim_end_matches('0');
                return format!("{}{int}{}{frac}", if neg { "-" } else { "" }, if frac.is_empty() { "" } else { "." });
            }
        }
        format!("{}/{} ≈ {}", self.n, self.d, trim_f64(self.to_f64()))
    }
}

fn trim_f64(x: f64) -> String {
    let s = format!("{x:.10}");
    let s = s.trim_end_matches('0').trim_end_matches('.');
    s.to_string()
}

impl PartialOrd for Q {
    fn partial_cmp(&self, o: &Q) -> Option<Ordering> {
        Some(self.cmp(o))
    }
}

impl Ord for Q {
    fn cmp(&self, o: &Q) -> Ordering {
        // n1/d1 vs n2/d2 with d > 0: n1*d2 vs n2*d1 (i128 overflow falls back to f64)
        match (self.n.checked_mul(o.d), o.n.checked_mul(self.d)) {
            (Some(a), Some(b)) => a.cmp(&b),
            _ => self.to_f64().partial_cmp(&o.to_f64()).unwrap_or(Ordering::Equal),
        }
    }
}

/// A decimal literal (`12`, `-3.25`, `.5`) as an exact rational.
pub fn parse_number(s: &str) -> Option<Q> {
    let s = s.trim();
    let (neg, body) = match s.strip_prefix('-') {
        Some(b) => (true, b),
        None => (false, s.strip_prefix('+').unwrap_or(s)),
    };
    if body.is_empty() || !body.chars().all(|c| c.is_ascii_digit() || c == '.') || body.matches('.').count() > 1 {
        return None;
    }
    let (int, frac) = body.split_once('.').unwrap_or((body, ""));
    if int.is_empty() && frac.is_empty() {
        return None;
    }
    let digits = format!("{int}{frac}");
    let n: i128 = digits.parse().ok()?;
    let d = 10i128.checked_pow(frac.len() as u32)?;
    let q = Q::new(n, d).ok()?;
    if neg { q.neg().ok() } else { Some(q) }
}

// --------------------------------------------------------------------- calc

#[derive(Clone, Debug, PartialEq)]
enum Tok {
    Num(Q),
    Id(String),
    Op(char),
}

fn lex(s: &str) -> Result<Vec<Tok>, String> {
    let cs: Vec<char> = s.chars().collect();
    let mut out = Vec::new();
    let mut i = 0;
    while i < cs.len() {
        let c = cs[i];
        if c.is_whitespace() {
            i += 1;
        } else if c.is_ascii_digit() || c == '.' {
            let j = (i..cs.len()).find(|&j| !(cs[j].is_ascii_digit() || cs[j] == '.')).unwrap_or(cs.len());
            let lit: String = cs[i..j].iter().collect();
            out.push(Tok::Num(parse_number(&lit).ok_or_else(|| format!("not a number: {lit}"))?));
            i = j;
        } else if c.is_alphabetic() || c == '_' {
            let j = (i..cs.len()).find(|&j| !(cs[j].is_alphanumeric() || cs[j] == '_')).unwrap_or(cs.len());
            out.push(Tok::Id(cs[i..j].iter().collect::<String>().to_lowercase()));
            i = j;
        } else {
            let op = match c {
                '×' => '*',
                '÷' => '/',
                '−' => '-',
                '*' if cs.get(i + 1) == Some(&'*') => {
                    i += 1;
                    '^'
                }
                '+' | '-' | '*' | '/' | '%' | '^' | '(' | ')' | ',' => c,
                _ => return Err(format!("unexpected character {c:?}")),
            };
            out.push(Tok::Op(op));
            i += 1;
        }
    }
    Ok(out)
}

struct Parser {
    t: Vec<Tok>,
    i: usize,
}

impl Parser {
    fn peek(&self) -> Option<&Tok> {
        self.t.get(self.i)
    }

    fn eat(&mut self, op: char) -> bool {
        if self.peek() == Some(&Tok::Op(op)) {
            self.i += 1;
            true
        } else {
            false
        }
    }

    fn expr(&mut self) -> Result<Q, String> {
        let mut v = self.term()?;
        loop {
            if self.eat('+') {
                v = v.add(self.term()?)?;
            } else if self.eat('-') {
                v = v.sub(self.term()?)?;
            } else {
                return Ok(v);
            }
        }
    }

    fn term(&mut self) -> Result<Q, String> {
        let mut v = self.unary()?;
        loop {
            if self.eat('*') {
                v = v.mul(self.unary()?)?;
            } else if self.eat('/') {
                v = v.div(self.unary()?)?;
            } else if self.eat('%') {
                v = v.rem(self.unary()?)?;
            } else {
                return Ok(v);
            }
        }
    }

    fn unary(&mut self) -> Result<Q, String> {
        if self.eat('-') {
            return self.unary()?.neg();
        }
        if self.eat('+') {
            return self.unary();
        }
        let base = self.primary()?;
        if self.eat('^') {
            // right-associative, binds tighter than unary minus on its right
            let e = self.unary()?;
            return base.pow(e);
        }
        Ok(base)
    }

    fn primary(&mut self) -> Result<Q, String> {
        match self.t.get(self.i).cloned() {
            Some(Tok::Num(q)) => {
                self.i += 1;
                Ok(q)
            }
            Some(Tok::Op('(')) => {
                self.i += 1;
                let v = self.expr()?;
                if !self.eat(')') {
                    return Err("a ( is not closed".into());
                }
                Ok(v)
            }
            Some(Tok::Id(f)) => {
                self.i += 1;
                if !self.eat('(') {
                    return Err(format!("{f} needs (…)"));
                }
                let mut args = Vec::new();
                if !self.eat(')') {
                    loop {
                        args.push(self.expr()?);
                        if self.eat(')') {
                            break;
                        }
                        if !self.eat(',') {
                            return Err(format!("{f}(…): a , or ) is missing"));
                        }
                    }
                }
                func(&f, &args)
            }
            Some(t) => Err(format!("unexpected {t:?}")),
            None => Err("the expression ends too early".into()),
        }
    }
}

fn func(f: &str, a: &[Q]) -> Result<Q, String> {
    let one = |a: &[Q]| -> Result<Q, String> {
        if a.len() == 1 { Ok(a[0]) } else { Err(format!("{f} takes one value")) }
    };
    let some = |a: &[Q]| -> Result<(), String> {
        if a.is_empty() { Err(format!("{f} needs at least one value")) } else { Ok(()) }
    };
    match f {
        "sum" => a.iter().try_fold(Q::int(0), |s, x| s.add(*x)),
        "min" => {
            some(a)?;
            Ok(*a.iter().min().unwrap())
        }
        "max" => {
            some(a)?;
            Ok(*a.iter().max().unwrap())
        }
        "mean" | "avg" | "average" => {
            some(a)?;
            a.iter().try_fold(Q::int(0), |s, x| s.add(*x))?.div(Q::int(a.len() as i128))
        }
        "abs" => Ok(one(a)?.abs()),
        "floor" => Ok(one(a)?.floor()),
        "ceil" => Ok(one(a)?.ceil()),
        "round" => match a {
            [x] => x.round(0),
            [x, d] if d.d == 1 => x.round(d.n),
            _ => Err("round takes a value and a whole number of decimals".into()),
        },
        other => Err(format!("unknown function {other}")),
    }
}

/// Evaluates an arithmetic expression exactly.
pub fn calc(expr: &str) -> Result<Q, String> {
    let t = lex(expr)?;
    if t.is_empty() {
        return Err("the expression is empty".into());
    }
    let mut p = Parser { t, i: 0 };
    let v = p.expr()?;
    if p.i != p.t.len() {
        return Err(format!("unexpected {:?} after the expression", p.t[p.i]));
    }
    Ok(v)
}

// -------------------------------------------------------------------- table

/// A command output as a table: one row a line, cells split by whitespace,
/// an optional header row; total lines (`1425 total`, `total 48`) are not rows.
#[derive(Debug)]
pub struct Table {
    pub header: Option<Vec<String>>,
    pub rows: Vec<Vec<String>>,
}

/// A cell's number: `8345`, `27G`, `1.5Gi`, `46%`, `3KB` (the unit or `%` dropped).
pub fn cell_number(s: &str) -> Option<Q> {
    let mut t = s.trim().trim_end_matches('%');
    for unit in ["iB", "B", "i"] {
        if let Some(x) = t.strip_suffix(unit) {
            if x.ends_with(|c: char| "KMGTPEkmgtpe".contains(c)) || unit == "B" {
                t = x;
                break;
            }
        }
    }
    let t = t.strip_suffix(|c: char| "KMGTPEkmgtpe".contains(c)).unwrap_or(t);
    parse_number(t)
}

fn is_aggregate(row: &[String]) -> bool {
    row.first().map(|c| c == "total").unwrap_or(false) || row.last().map(|c| c == "total").unwrap_or(false)
}

impl Table {
    /// The table in `text`: of the blank-line separated blocks (a leading
    /// `Output of …:` line and `Question:`/`Task:` lines left out), the
    /// longest one whose lines mostly have two or more cells.
    pub fn parse(text: &str) -> Result<Table, String> {
        let mut best: Vec<Vec<String>> = Vec::new();
        for block in text.split("\n\n") {
            let mut lines: Vec<&str> = block
                .lines()
                .map(str::trim_end)
                .filter(|l| !l.trim().is_empty())
                .filter(|l| {
                    let t = l.trim_start();
                    !(t.starts_with("Question:") || t.starts_with("Task:") || t.starts_with("Answer "))
                })
                .collect();
            if lines.first().map(|l| l.trim_end().ends_with(':')).unwrap_or(false) {
                lines.remove(0);
            }
            let rows: Vec<Vec<String>> =
                lines.iter().map(|l| l.split_whitespace().map(String::from).collect()).collect();
            let wide = rows.iter().filter(|r| r.len() >= 2).count();
            if !rows.is_empty() && 2 * wide >= rows.len() && rows.len() > best.len() {
                best = rows;
            }
        }
        if best.is_empty() {
            return Err("no table in the last output".into());
        }
        let mut rows: Vec<Vec<String>> = best.into_iter().filter(|r| !is_aggregate(r)).collect();
        let header_like = |r: &[String]| r.iter().all(|c| cell_number(c).is_none());
        let mut header = None;
        if rows.len() >= 2 && header_like(&rows[0]) && rows[1..].iter().any(|r| !header_like(r)) {
            let mut h = rows.remove(0);
            // the usual width of a row; trailing header words of a wider header name one column ("Mounted on")
            let mut widths: Vec<usize> = rows.iter().map(Vec::len).collect();
            widths.sort_unstable();
            let w = widths[widths.len() / 2];
            if h.len() > w && w >= 1 {
                let tail = h.split_off(w - 1).join(" ");
                h.push(tail);
            }
            header = Some(h);
        }
        if rows.is_empty() {
            return Err("the table has no rows".into());
        }
        Ok(Table { header, rows })
    }

    /// A column given as its header name or as a number from 1.
    fn col(&self, c: &Value) -> Result<usize, String> {
        let width = self.rows.iter().map(Vec::len).max().unwrap_or(0);
        let by_number = |n: i64| -> Result<usize, String> {
            if n >= 1 && (n as usize) <= width {
                Ok(n as usize - 1)
            } else {
                Err(format!("no column {n}: the rows have {width}"))
            }
        };
        match c {
            Value::Number(n) => by_number(n.as_i64().ok_or("a column number is a whole number")?),
            Value::String(s) => {
                if let Ok(n) = s.trim().parse::<i64>() {
                    return by_number(n);
                }
                let h = self.header.as_ref().ok_or_else(|| format!("no column named {s}: the table has no header"))?;
                h.iter()
                    .position(|x| x.eq_ignore_ascii_case(s.trim()))
                    .ok_or_else(|| format!("no column named {s} (columns: {})", h.join(", ")))
            }
            _ => Err("column: a header name or a number from 1".into()),
        }
    }

    fn name(row: &[String]) -> &str {
        row.last().map(String::as_str).unwrap_or("")
    }

    fn cell<'a>(row: &'a [String], c: usize) -> Result<&'a str, String> {
        row.get(c)
            .map(String::as_str)
            .ok_or_else(|| format!("row {} has no column {}", Table::name(row), c + 1))
    }

    fn number(row: &[String], c: usize) -> Result<Q, String> {
        let s = Table::cell(row, c)?;
        cell_number(s).ok_or_else(|| format!("{s} (row {}) is not a number", Table::name(row)))
    }
}

fn value_text(v: &Value) -> String {
    match v {
        Value::String(s) => s.clone(),
        other => other.to_string(),
    }
}

fn passes(t: &Table, row: &[String], test: &Value) -> Result<bool, String> {
    let c = t.col(test.get("column").ok_or("where needs a column")?)?;
    let op = test.get("op").and_then(Value::as_str).unwrap_or("=");
    let want = value_text(test.get("value").ok_or("where needs a value")?).trim().to_string();
    let cell = Table::cell(row, c).unwrap_or("");
    let nums = (cell_number(cell), cell_number(&want));
    Ok(match op {
        "=" | "==" => match nums {
            (Some(a), Some(b)) => a == b,
            _ => cell == want,
        },
        "!=" => match nums {
            (Some(a), Some(b)) => a != b,
            _ => cell != want,
        },
        "contains" => cell.contains(&want),
        ">" | "<" | ">=" | "<=" => {
            let (Some(a), Some(b)) = nums else {
                return Err(format!("{op} compares numbers: {cell} and {want}"));
            };
            match op {
                ">" => a > b,
                "<" => a < b,
                ">=" => a >= b,
                _ => a <= b,
            }
        }
        other => return Err(format!("unknown test {other}")),
    })
}

/// The `table` tool over `input` (see [`Table::parse`]).
pub fn table(args: &Value, input: &str) -> Result<String, String> {
    let t = Table::parse(input)?;
    let op = args.get("op").and_then(Value::as_str).ok_or("table needs an op")?;
    // rows by name, in the order asked (diff reads the order)
    let mut rows: Vec<&Vec<String>> = match args.get("rows") {
        None | Some(Value::Null) => t.rows.iter().collect(),
        Some(Value::Array(names)) if !names.is_empty() => {
            let mut out = Vec::new();
            for n in names {
                let n = value_text(n).trim().to_string();
                let r = t
                    .rows
                    .iter()
                    .find(|r| Table::name(r) == n)
                    .ok_or_else(|| format!("no row named {n}"))?;
                out.push(r);
            }
            out
        }
        Some(Value::String(n)) => {
            let n = n.trim();
            vec![t.rows.iter().find(|r| Table::name(r) == n).ok_or_else(|| format!("no row named {n}"))?]
        }
        _ => return Err("rows: a list of row names".into()),
    };
    if let Some(w) = args.get("where").filter(|w| !w.is_null()) {
        let tests: Vec<&Value> = match w {
            Value::Array(a) => a.iter().collect(),
            other => vec![other],
        };
        let mut kept = Vec::new();
        for r in rows {
            let mut ok = true;
            for test in &tests {
                ok &= passes(&t, r, test)?;
            }
            if ok {
                kept.push(r);
            }
        }
        rows = kept;
    }
    if op == "count" {
        return Ok(rows.len().to_string());
    }
    let c = t.col(args.get("column").ok_or_else(|| format!("{op} needs a column"))?)?;
    if rows.is_empty() {
        return Err("no row passes".into());
    }
    match op {
        "sum" | "min" | "max" | "mean" => {
            let v: Vec<Q> = rows.iter().map(|r| Table::number(r, c)).collect::<Result<_, _>>()?;
            Ok(func(op, &v)?.show())
        }
        "get" => {
            if rows.len() != 1 {
                return Err(format!("{} rows pass; get reads one", rows.len()));
            }
            let s = Table::cell(rows[0], c)?;
            Ok(cell_number(s).map(Q::show).unwrap_or_else(|| s.to_string()))
        }
        "argmax" | "argmin" => {
            let v: Vec<Q> = rows.iter().map(|r| Table::number(r, c)).collect::<Result<_, _>>()?;
            let best = if op == "argmax" { v.iter().max() } else { v.iter().min() }.copied().unwrap();
            let names: Vec<&str> = rows.iter().zip(&v).filter(|(_, x)| **x == best).map(|(r, _)| Table::name(r)).collect();
            Ok(names.join(", "))
        }
        "diff" => {
            if rows.len() != 2 {
                return Err("diff reads two rows: the first minus the second".into());
            }
            Ok(Table::number(rows[0], c)?.sub(Table::number(rows[1], c)?)?.show())
        }
        other => Err(format!("unknown op {other}")),
    }
}

// ------------------------------------------------------------- the contract

/// The text `table` reads at the end of `prompt` (a cmf-im-v1 transcript):
/// the last tool response after the last user turn, else that user turn;
/// a prompt without turns is read whole.
pub fn last_output(prompt: &str) -> String {
    const USER: &str = "<|im_start|>user\n";
    const TOOL: &str = "<|im_start|>tool\n";
    const END: &str = "<|im_end|>";
    let Some(u) = prompt.rfind(USER) else {
        return prompt.to_string();
    };
    let after = &prompt[u..];
    if let Some(t) = after.rfind(TOOL) {
        let body = &after[t + TOOL.len()..];
        let body = body.split(END).next().unwrap_or(body);
        let body = body.trim();
        let body = body.strip_prefix("<tool_response>").unwrap_or(body);
        let body = body.strip_suffix("</tool_response>").unwrap_or(body);
        return body.trim().to_string();
    }
    let body = &after[USER.len()..];
    body.split(END).next().unwrap_or(body).trim().to_string()
}

/// The texts `table` may read at the end of `prompt`, newest first: the tool
/// responses after the last user turn, then that user turn (a prompt without
/// turns is one text). A second `table` call after a result reads the table
/// again, not the previous result.
pub fn outputs(prompt: &str) -> Vec<String> {
    const USER: &str = "<|im_start|>user\n";
    const TOOL: &str = "<|im_start|>tool\n";
    const END: &str = "<|im_end|>";
    let Some(u) = prompt.rfind(USER) else {
        return vec![prompt.to_string()];
    };
    let after = &prompt[u..];
    let mut out: Vec<String> = after
        .split(TOOL)
        .skip(1)
        .map(|t| {
            let body = t.split(END).next().unwrap_or(t).trim();
            let body = body.strip_prefix("<tool_response>").unwrap_or(body);
            body.strip_suffix("</tool_response>").unwrap_or(body).trim().to_string()
        })
        .collect();
    out.reverse();
    let body = &after[USER.len()..];
    out.push(body.split(END).next().unwrap_or(body).trim().to_string());
    out
}

/// `table` over the newest of `texts` that holds a table (the error of the
/// newest when none does); any other tool over the newest text.
fn call_on(name: &str, args: &Value, texts: &[String]) -> Result<String, String> {
    if name == "table" {
        // a tool's own error line is not an output to read
        if let Some(t) = texts.iter().find(|t| !t.starts_with("error:") && Table::parse(t).is_ok()) {
            return call(name, args, t);
        }
    }
    call(name, args, texts.first().map(String::as_str).unwrap_or(""))
}

/// The `<tool_call>` blocks of an assistant turn: (name, arguments), `None`
/// for a block that is not a call object.
pub fn turn_calls(text: &str) -> Vec<Option<(String, Value)>> {
    let mut out = Vec::new();
    let mut rest = text;
    while let Some(s) = rest.find("<tool_call>") {
        let after = &rest[s + "<tool_call>".len()..];
        let Some(e) = after.find("</tool_call>") else {
            out.push(None);
            break;
        };
        let body = after[..e].trim();
        out.push(serde_json::from_str::<Value>(body).ok().and_then(|v| {
            let name = v.get("name")?.as_str()?.to_string();
            let args = match v.get("arguments") {
                Some(Value::String(s)) => serde_json::from_str(s).unwrap_or(Value::Null),
                Some(a) => a.clone(),
                None => Value::Object(Default::default()),
            };
            Some((name, args))
        }));
        rest = &after[e + "</tool_call>".len()..];
    }
    out
}

/// One built-in round: the calls run and the tool turns that answer them.
#[derive(Debug)]
pub struct Round {
    /// (name, arguments, result) of every call, in order.
    pub calls: Vec<(String, Value, String)>,
    /// `<|im_start|>tool\n<tool_response>\n…\n</tool_response><|im_end|>\n` per call.
    pub tool_turns: String,
}

/// Runs the calls of the assistant turn `generated` (continuing `prompt`)
/// when it has calls and every one is built in; `None` otherwise (no call,
/// a malformed block, or a call the client serves). A failing call answers
/// `error: …`, which the model reads like any tool result.
pub fn round(prompt: &str, generated: &str) -> Option<Round> {
    let calls = turn_calls(generated);
    if calls.is_empty() || calls.iter().any(|c| !matches!(c, Some((n, _)) if is_builtin(n))) {
        return None;
    }
    let texts = outputs(prompt);
    let mut done = Vec::new();
    let mut tool_turns = String::new();
    for (name, args) in calls.into_iter().flatten() {
        let r = call_on(&name, &args, &texts).unwrap_or_else(|e| format!("error: {e}"));
        tool_turns.push_str("<|im_start|>tool\n<tool_response>\n");
        tool_turns.push_str(&r);
        tool_turns.push_str("\n</tool_response><|im_end|>\n");
        done.push((name, args, r));
    }
    Some(Round { calls: done, tool_turns })
}

/// The prompt that continues after a built-in round: the assistant turn
/// closed, the tool turns, a new assistant turn opened.
pub fn continue_prompt(prompt: &str, generated: &str, r: &Round) -> String {
    format!("{prompt}{}<|im_end|>\n{}<|im_start|>assistant\n", generated.trim_end(), r.tool_turns)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn c(e: &str) -> String {
        calc(e).map(Q::show).unwrap_or_else(|x| format!("ERR {x}"))
    }

    #[test]
    fn calc_is_exact() {
        assert_eq!(c("7243 + 2719"), "9962");
        assert_eq!(c("75066 - 22301"), "52765");
        assert_eq!(c("(3 + 4) * 12 / 5"), "16.8");
        assert_eq!(c("1/3"), "1/3 ≈ 0.3333333333");
        assert_eq!(c("1/3 * 3"), "1");
        assert_eq!(c("0.1 + 0.2"), "0.3");
        assert_eq!(c("2^10"), "1024");
        assert_eq!(c("2**-2"), "0.25");
        assert_eq!(c("-2^2"), "-4");
        assert_eq!(c("7 % 3"), "1");
        assert_eq!(c("-7 % 3"), "2");
        assert_eq!(c("sum(1, 2, 3.5)"), "6.5");
        assert_eq!(c("mean(542, 391, 272, 220)"), "356.25");
        assert_eq!(c("max(3, 9, 4) - min(3, 9, 4)"), "6");
        assert_eq!(c("round(2.345, 2)"), "2.35");
        assert_eq!(c("round(-2.5)"), "-3");
        assert_eq!(c("12 × 3 ÷ 4"), "9");
        assert_eq!(c("170141183460469231731687303715884105727 + 1"), format!("ERR {OVERFLOW}"));
        assert_eq!(c("1/0"), "ERR division by zero");
        assert_eq!(c("2 +"), "ERR the expression ends too early");
        assert_eq!(c("foo(1)"), "ERR unknown function foo");
    }

    const LS: &str = "Output of `ls -l /var/log/app`:\n-rw-r--r-- 1 user user  8345 Jan 15 10:00 error.log\n\
-rw-r--r-- 1 user user   602 Jan 15 10:00 debug.log\n-rw-r--r-- 1 user user  7369 Jan 15 10:00 cron.log\n\
-rw-r--r-- 1 user user  4072 Jan 15 10:00 mail.log\n-rw-r--r-- 1 user user  7243 Jan 15 10:00 db.log\n\
-rw-r--r-- 1 user user  2719 Jan 15 10:00 boot.log\n\nQuestion: How many of these files are larger than 2692 bytes?\nAnswer with a number.";
    const WC: &str = "Output of `wc -l *.log`:\n  542 db.log\n  391 debug.log\n  272 mail.log\n  220 boot.log\n 1425 total\n\n\
Question: Which file has the most lines?\nAnswer with the file name.";
    const DF: &str = "Output of `df -h`:\nFilesystem      Size  Used Avail Use% Mounted on\n/dev/sda1        50G   23G   27G  46% /var\n\
/dev/sdb1       200G   32G  168G  16% /data\n/dev/sdc1       250G   87G  163G  34% /\n/dev/sdd1       100G   12G   88G  12% /backup\n\n\
Question: How many GB are available on /data?\nAnswer with a number.";
    const DU: &str = "Output of `du -sk /opt/app /srv/www /var/log`:\n75066\t/opt/app\n22301\t/srv/www\n60284\t/var/log\n\n\
Question: How many KB larger is /opt/app than /srv/www?\nAnswer with a number.";
    const PS: &str = "Output of `ps -eo user,pid,comm`:\nUSER       PID COMMAND\nroot       3973 nginx\nwww-data   2346 python3\n\
boris      5364 nginx\npostgres   4772 bash\nanna       6633 nginx\nroot       8017 redis-server\nroot        303 node\n\
postgres   5471 nginx\nwww-data   6020 node\n\nQuestion: How many of these processes belong to the user postgres?\nAnswer with a number.";

    fn t(args: Value, input: &str) -> String {
        table(&args, input).unwrap_or_else(|e| format!("ERR {e}"))
    }

    #[test]
    fn table_answers_the_eight_families_of_set_r() {
        assert_eq!(t(json!({"op": "count", "where": {"column": 5, "op": ">", "value": 2692}}), LS), "5");
        assert_eq!(t(json!({"op": "sum", "column": 5, "rows": ["db.log", "boot.log"]}), LS), "9962");
        assert_eq!(t(json!({"op": "sum", "column": 1, "rows": ["boot.log", "debug.log"]}), WC), "611");
        assert_eq!(t(json!({"op": "argmax", "column": 1}), WC), "db.log");
        assert_eq!(t(json!({"op": "get", "column": "Avail", "rows": ["/data"]}), DF), "168");
        assert_eq!(t(json!({"op": "sum", "column": "Used", "rows": ["/", "/var"]}), DF), "110");
        assert_eq!(t(json!({"op": "diff", "column": 1, "rows": ["/opt/app", "/srv/www"]}), DU), "52765");
        assert_eq!(t(json!({"op": "count", "where": {"column": "USER", "op": "=", "value": "postgres"}}), PS), "2");
    }

    #[test]
    fn table_reads_headers_totals_and_reports_mistakes() {
        let df = Table::parse(DF).unwrap();
        assert_eq!(df.header.as_ref().unwrap().last().unwrap(), "Mounted on");
        assert_eq!(Table::parse(WC).unwrap().rows.len(), 4, "the total line is not a row");
        assert_eq!(t(json!({"op": "mean", "column": 1}), WC), "356.25");
        assert_eq!(t(json!({"op": "count", "where": [{"column": "USER", "value": "root"}, {"column": 3, "op": "=", "value": "nginx"}]}), PS), "1");
        assert_eq!(t(json!({"op": "get", "column": "Use%", "rows": ["/"]}), DF), "34");
        assert_eq!(t(json!({"op": "get", "column": "Size", "rows": ["/home"]}), DF), "ERR no row named /home");
        assert_eq!(t(json!({"op": "sum", "column": "Free", "rows": ["/"]}), DF).starts_with("ERR no column named Free"), true);
        assert_eq!(t(json!({"op": "get", "column": 1}), PS), "ERR 9 rows pass; get reads one");
        assert_eq!(t(json!({"op": "sum", "column": 1}), PS), "ERR root (row nginx) is not a number");
        // a stray space in what the model writes does not change the answer
        assert_eq!(t(json!({"op": "count", "where": {"column": "USER", "op": "=", "value": " anna"}}), PS), "1");
        assert_eq!(t(json!({"op": "get", "column": "Avail", "rows": [" /data "]}), DF), "168");
    }

    #[test]
    fn a_round_runs_built_in_calls_and_leaves_others_to_the_client() {
        let prompt = format!("<|im_start|>system\nS<|im_end|>\n<|im_start|>user\n{PS}<|im_end|>\n<|im_start|>assistant\n");
        let turn = "<think>\nA table: count the rows of postgres.\n</think>\n<tool_call>\n{\"name\": \"table\", \"arguments\": \
{\"op\": \"count\", \"where\": {\"column\": \"USER\", \"op\": \"=\", \"value\": \"postgres\"}}}\n</tool_call>";
        let r = round(&prompt, turn).expect("a built-in turn");
        assert_eq!(r.calls[0].2, "2");
        let next = continue_prompt(&prompt, turn, &r);
        assert!(next.ends_with("</tool_call><|im_end|>\n<|im_start|>tool\n<tool_response>\n2\n</tool_response><|im_end|>\n<|im_start|>assistant\n"));
        // the tool's answer is now the last output
        assert_eq!(last_output(&next), "2");
        assert!(round(&prompt, "<tool_call>\n{\"name\": \"lookup\", \"arguments\": {\"query\": \"x\"}}\n</tool_call>").is_none());
        assert!(round(&prompt, "The answer is 2.").is_none());
        // a second table call after the result reads the table again, not the result
        let again = round(&next, turn).expect("a built-in turn");
        assert_eq!(again.calls[0].2, "2");
        // nor the error of a failed call
        let miss = "<tool_call>\n{\"name\": \"table\", \"arguments\": {\"op\": \"get\", \"column\": 2, \"rows\": [\"x\"]}}\n</tool_call>";
        let failed = round(&prompt, miss).unwrap();
        assert_eq!(failed.calls[0].2, "error: no row named x");
        let after = continue_prompt(&prompt, miss, &failed);
        assert_eq!(round(&after, turn).unwrap().calls[0].2, "2");
        let bad = round(&prompt, "<tool_call>\n{\"name\": \"calc\", \"arguments\": {\"expression\": \"1/0\"}}\n</tool_call>").unwrap();
        assert_eq!(bad.calls[0].2, "error: division by zero");
        assert_eq!(descriptions().len(), 2);
    }
}
