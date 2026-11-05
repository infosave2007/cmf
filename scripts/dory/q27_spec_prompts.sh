#!/bin/bash
# The five prompts of the draft measurements on a Mac: q27_spec_prompts.sh WIKI_TEST DIR
# prose, code, ru and edit are written here; quote is 3200 bytes of wikitext-2 test and a question.
set -e
wiki=$1; d=$2; mkdir -p "$d"
printf '%s\n' "Write a detailed essay about how the Tang dynasty shaped Chinese poetry, with examples of the major poets and their styles." > "$d/p_prose.txt"
printf '%s\n' "Write a Rust function that parses one line of CSV into a vector of fields, handling quoted fields and escaped quotes. Then write unit tests for it." > "$d/p_code.txt"
printf '%s\n' "Объясни простыми словами, как работает квантование нейросетей в 4 бита и почему модель после него почти не теряет качество. Приведи пример." > "$d/p_ru.txt"
cat > "$d/p_edit.txt" <<'P'
Here is a Rust function:

fn parse_csv_line(line: &str) -> Vec<String> {
    let mut fields = Vec::new();
    let mut current = String::new();
    let mut in_quotes = false;
    let mut chars = line.chars().peekable();
    while let Some(c) = chars.next() {
        match c {
            '"' if in_quotes && chars.peek() == Some(&'"') => {
                current.push('"');
                chars.next();
            }
            '"' => in_quotes = !in_quotes,
            ',' if !in_quotes => {
                fields.push(current.clone());
                current.clear();
            }
            _ => current.push(c),
        }
    }
    fields.push(current);
    fields
}

Rewrite it so that the separator is a parameter `sep: char` instead of the hard-coded comma, and trim whitespace around unquoted fields. Return the full function.
P
{ tail -c +5005 "$wiki" | head -c 3200; printf '\n\n%s\n' "Quote the three most important sentences of the text above word for word, then say in one sentence what the text is about."; } > "$d/p_quote.txt"
