//! PII redaction of the state before egress (spec §5.3).
//!
//! Two passes. The first is a port of cortiq-router `pii.rs`: the text is
//! split into runs of token characters (`[A-Za-z0-9@._+-]`) and runs of
//! everything else; a token run is replaced by `[REDACTED]` when it looks like
//! * an e-mail address: an `@` after the first character with a `.` after it;
//! * a secret: at least 20 bytes of `[A-Za-z0-9_-]` with a digit and a letter;
//! * a long number: at least 9 digits and only digits, `-` and `+`.
//!
//! The second pass catches the numbers the router's split misses, because it
//! splits at spaces and parentheses: digit groups joined by up to
//! [`MAX_GROUP_GAP`] separators (space, tab, `-`, `.`, `/`, `(`, `)`) are one
//! number, and a number of at least [`MIN_NUMBER_DIGITS`] digits in total is
//! replaced from its first digit (a `+` or `(` just before it included) to its
//! last — `4111 1111 1111 1111`, `+1 (555) 123-4567`, the digits of
//! `DE89 3704 0044 0532 0130 00`. Numbers are redacted by their digit count
//! alone (no Luhn check), so a long run of plain numbers is redacted too.
//!
//! This is a heuristic: names, postal addresses, numbers written in words and
//! identifiers with letters between short digit groups are not detected.
//!
//! Every other byte is kept, so the shape of the text survives. Non-ASCII bytes
//! are never token or separator characters, so replacements always end on
//! character boundaries.
//!
//! [`redact_value`] applies [`redact`] to every string leaf of a JSON state
//! (object keys are kept). The cascade redacts when `oracle.redact_pii` is on and
//! the request did not set `cmf.allow_pii_egress` — the state and, since 0.8.8
//! (DESIGN B4), each question's instructions and criteria (the option ids are
//! keys, so they are kept); a redacted question carries the flag
//! [`FLAG_PII_REDACTED`].

use serde_json::{Map, Value};

/// The replacement of a redacted token.
pub const REDACTED: &str = "[REDACTED]";
/// Flag of a question whose state was redacted before it left the machine.
pub const FLAG_PII_REDACTED: &str = "pii_redacted";

/// Fewest digits of a number that is redacted.
pub const MIN_NUMBER_DIGITS: usize = 9;
/// Most separator bytes between two digit groups of one number (`") "`,
/// `" - "`).
pub const MAX_GROUP_GAP: usize = 3;

/// `(redacted text, whether anything was replaced)`.
pub fn redact(text: &str) -> (String, bool) {
    let mut out = String::with_capacity(text.len());
    let mut changed = false;
    for token in split_keep_delims(text) {
        if looks_like_email(token) || looks_like_secret(token) || looks_like_long_number(token) {
            out.push_str(REDACTED);
            changed = true;
        } else {
            out.push_str(token);
        }
    }
    let (out, grouped) = redact_number_groups(&out);
    (out, changed || grouped)
}

#[inline]
fn is_group_separator(b: u8) -> bool {
    matches!(b, b' ' | b'\t' | b'-' | b'.' | b'/' | b'(' | b')')
}

/// The second pass (see the module notes): numbers whose digit groups are
/// joined by separators.
fn redact_number_groups(s: &str) -> (String, bool) {
    let b = s.as_bytes();
    let mut out = String::with_capacity(s.len());
    let mut last = 0; // end of the text already copied
    let mut changed = false;
    let mut i = 0;
    while i < b.len() {
        if !b[i].is_ascii_digit() {
            i += 1;
            continue;
        }
        // A number starts here: its groups, and the gaps between them.
        let mut j = i;
        let mut digits = 0;
        let end = loop {
            while j < b.len() && b[j].is_ascii_digit() {
                digits += 1;
                j += 1;
            }
            let group_end = j;
            let mut k = j;
            while k < b.len() && k - group_end < MAX_GROUP_GAP && is_group_separator(b[k]) {
                k += 1;
            }
            if k > group_end && k < b.len() && b[k].is_ascii_digit() {
                j = k;
                continue;
            }
            break group_end;
        };
        if digits >= MIN_NUMBER_DIGITS {
            let mut start = i;
            while start > last && matches!(b[start - 1], b'+' | b'(') {
                start -= 1;
            }
            out.push_str(&s[last..start]);
            out.push_str(REDACTED);
            last = end;
            changed = true;
        }
        i = end;
    }
    out.push_str(&s[last..]);
    (out, changed)
}

/// [`redact`] on every string leaf of a JSON value; keys are kept.
pub fn redact_value(v: &Value) -> (Value, bool) {
    match v {
        Value::String(s) => {
            let (r, c) = redact(s);
            (Value::String(r), c)
        }
        Value::Array(a) => {
            let mut changed = false;
            let out = a
                .iter()
                .map(|x| {
                    let (r, c) = redact_value(x);
                    changed |= c;
                    r
                })
                .collect();
            (Value::Array(out), changed)
        }
        Value::Object(m) => {
            let mut changed = false;
            let mut out = Map::with_capacity(m.len());
            for (k, x) in m {
                let (r, c) = redact_value(x);
                changed |= c;
                out.insert(k.clone(), r);
            }
            (Value::Object(out), changed)
        }
        other => (other.clone(), false),
    }
}

/// Runs of token and non-token bytes, in order (their concatenation is `s`).
fn split_keep_delims(s: &str) -> Vec<&str> {
    let mut parts = Vec::new();
    let bytes = s.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        let start = i;
        let word_like = is_token_char(bytes[i]);
        while i < bytes.len() && is_token_char(bytes[i]) == word_like {
            i += 1;
        }
        parts.push(&s[start..i]);
    }
    parts
}

#[inline]
fn is_token_char(b: u8) -> bool {
    b.is_ascii_alphanumeric() || b == b'@' || b == b'.' || b == b'_' || b == b'-' || b == b'+'
}

fn looks_like_email(t: &str) -> bool {
    let at = t.find('@');
    matches!(at, Some(p) if p > 0 && t[p + 1..].contains('.'))
}

/// Long alphanumeric blobs that look like keys or tokens (≥ 20 bytes, mixed).
fn looks_like_secret(t: &str) -> bool {
    if t.len() < 20 {
        return false;
    }
    let has_digit = t.bytes().any(|b| b.is_ascii_digit());
    let has_alpha = t.bytes().any(|b| b.is_ascii_alphabetic());
    let alnum = t
        .bytes()
        .all(|b| b.is_ascii_alphanumeric() || b == b'_' || b == b'-');
    has_digit && has_alpha && alnum
}

/// Long digit runs (phone, card or account numbers): ≥ 9 digits.
fn looks_like_long_number(t: &str) -> bool {
    let digits = t.bytes().filter(|b| b.is_ascii_digit()).count();
    digits >= 9
        && t.bytes()
            .all(|b| b.is_ascii_digit() || b == b'-' || b == b'+')
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn masks_email_and_keeps_shape() {
        let (r, c) = redact("contact me at john.doe@example.com please");
        assert!(c);
        assert_eq!(r, "contact me at [REDACTED] please");
    }

    #[test]
    fn masks_secret_and_phone() {
        assert!(redact("key sk_live_abcd1234EFGH5678ijkl").1);
        assert_eq!(redact("call +15551234567 now").0, "call [REDACTED] now");
        // The trailing '.' is a token character, so it goes with the number.
        assert_eq!(
            redact("card 4111-1111-1111-1111 ok").0,
            "card [REDACTED] ok"
        );
    }

    #[test]
    fn masks_numbers_written_in_groups() {
        for (t, want) in [
            (
                "card 4111 1111 1111 1111 phone +1 555 123 4567",
                "card [REDACTED] phone [REDACTED]",
            ),
            ("call +1 (555) 123-4567 today", "call [REDACTED] today"),
            ("tel (495) 123-45-67.", "tel [REDACTED]."),
            (
                "iban DE89 3704 0044 0532 0130 00 ok",
                "iban DE[REDACTED] ok",
            ),
            ("acct 1234 5678 9", "acct [REDACTED]"),
            (
                "номер 4111 1111 1111 1111, спасибо",
                "номер [REDACTED], спасибо",
            ),
            ("a/c 12/34/56/78/90", "a/c [REDACTED]"),
        ] {
            let (r, c) = redact(t);
            assert!(c, "{t}");
            assert_eq!(r, want, "{t}");
        }
    }

    #[test]
    fn short_groups_stay() {
        for t in [
            "on 2026-09-27 at 10:30",
            "call 555 1234 later",
            "rooms 12 and 34, floors 5 and 6",
            "price 1,234,567 dollars",
        ] {
            let (r, c) = redact(t);
            assert!(!c, "{t}");
            assert_eq!(r, t);
        }
    }

    #[test]
    fn leaves_clean_text_untouched() {
        for t in [
            "Solve x^2 - 5x + 6 = 0 and explain",
            "I still have not received my new card",
            "привет, мир — 12345678 is short",
            "",
        ] {
            let (r, c) = redact(t);
            assert!(!c, "{t}");
            assert_eq!(r, t);
        }
    }

    #[test]
    fn non_ascii_runs_stay_whole() {
        let (r, c) = redact("письмо на a.b@c.de спасибо");
        assert!(c);
        assert_eq!(r, "письмо на [REDACTED] спасибо");
    }

    #[test]
    fn json_leaves_are_redacted_keys_kept() {
        let v = json!({"user@x.io": "mail user@x.io", "n": [1, "tel 123456789"], "ok": "fine"});
        let (r, c) = redact_value(&v);
        assert!(c);
        assert_eq!(
            r,
            json!({"user@x.io": "mail [REDACTED]", "n": [1, "tel [REDACTED]"], "ok": "fine"})
        );
        let (same, c2) = redact_value(&json!({"a": "b"}));
        assert!(!c2);
        assert_eq!(same, json!({"a": "b"}));
    }
}
