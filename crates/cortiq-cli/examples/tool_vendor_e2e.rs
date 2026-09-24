//! Tool calling end to end with a model's ORIGINAL tokenizer and chat
//! template swapped in — the check that a file converted from the vendor
//! checkpoint as-is (special tool-markup tokens, `tojson(ensure_ascii=…)`)
//! works, without converting one.
//!
//!   cargo run --release -p cortiq-cli --example tool_vendor_e2e -- \
//!       model.cmf tokenizer.json chat_template.jinja
//!
//! Six tool conversations (the tooleval set): turn 1 must parse into a
//! call of the right tool — through the non-streaming extractor AND the
//! streaming holdback, which must agree and leak no markup — and turn 2,
//! with the tool result, must answer from it.
use cortiq_core::CmfModel;
use cortiq_engine::tokenizer::Tokenizer;
use cortiq_engine::{Pipeline, SamplerConfig};
use cortiq_server::tool_calls::{ToolHoldback, extract_tool_calls};
use serde_json::{Value, json};
use std::sync::{Arc, Mutex};

fn tool(name: &str, desc: &str, props: Value, req: &[&str]) -> Value {
    json!({"type": "function", "function": {"name": name, "description": desc,
        "parameters": {"type": "object", "properties": props, "required": req}}})
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let a: Vec<String> = std::env::args().skip(1).collect();
    let m = Arc::new(CmfModel::open_sharded(&a[0])?);
    let cfg = SamplerConfig {
        temperature: 0.0,
        top_p: 1.0,
        top_k: 1,
        repetition_penalty: 1.0,
        presence_penalty: 0.0,
        min_p: 0.0,
        seed: Some(1),
        suppress_tokens: Vec::new(),
    };
    let mut p = Pipeline::from_model(&m, cfg)?;
    let mut tok = Tokenizer::from_file(&a[1])?;
    tok.chat_template = Some(std::fs::read_to_string(&a[2])?);
    tok.extra_eos = p.tokenizer.extra_eos.clone();
    p.tokenizer = Arc::new(tok);

    let weather = tool(
        "get_weather",
        "Get the current weather for a city",
        json!({"city": {"type": "string", "description": "City name"}}),
        &["city"],
    );
    let calc = tool(
        "calculator",
        "Evaluate an arithmetic expression",
        json!({"expression": {"type": "string"}}),
        &["expression"],
    );
    let search = tool(
        "web_search",
        "Search the web and return the top result",
        json!({"query": {"type": "string"}}),
        &["query"],
    );
    let time = tool(
        "get_time",
        "Get the current local time in a timezone",
        json!({"timezone": {"type": "string", "description": "IANA timezone, e.g. Europe/Berlin"}}),
        &["timezone"],
    );
    let email = tool(
        "send_email",
        "Send an email",
        json!({"to": {"type": "string"}, "subject": {"type": "string"}, "body": {"type": "string"}}),
        &["to", "subject", "body"],
    );
    let scenarios: Vec<(&str, Vec<Value>, Value, Vec<&str>)> = vec![
        (
            "What's the weather in Paris right now?",
            vec![weather.clone(), time.clone()],
            json!({"temp_c": 18, "sky": "cloudy"}),
            vec!["18"],
        ),
        (
            "What is 1234 * 5678? Use the calculator.",
            vec![calc],
            json!({"result": 7006652}),
            vec!["7006652", "7,006,652"],
        ),
        (
            "Who won the 2022 FIFA World Cup? Search the web.",
            vec![search, weather.clone()],
            json!({"top_result": "Argentina won the 2022 FIFA World Cup, beating France on penalties."}),
            vec!["Argentina"],
        ),
        (
            "What time is it in Tokyo?",
            vec![time, weather.clone()],
            json!({"time": "21:37", "timezone": "Asia/Tokyo"}),
            vec!["21:37"],
        ),
        (
            "Email bob@example.com that the meeting moved to 3pm.",
            vec![email],
            json!({"status": "sent"}),
            vec!["sent", "Bob", "bob"],
        ),
        (
            "北京现在天气怎么样？",
            vec![weather],
            json!({"temp_c": 25, "sky": "sunny"}),
            vec!["25"],
        ),
    ];

    let (mut ok1, mut ok2, mut agree) = (0, 0, 0);
    for (q, tools, result, expect) in &scenarios {
        let mut msgs = vec![json!({"role": "user", "content": q})];
        let ids = p
            .tokenizer
            .try_apply_chat_template_json(&msgs, Some(tools), None)?;
        // Streaming path: every decoded piece through the holdback.
        let hold = Arc::new(Mutex::new(ToolHoldback::new()));
        let streamed = Arc::new(Mutex::new(String::new()));
        let (h2, s2) = (hold.clone(), streamed.clone());
        let cb: cortiq_engine::TokenCallback = Box::new(move |t: &str| {
            let out = h2.lock().unwrap().push(t);
            s2.lock().unwrap().push_str(&out);
            true
        });
        let r = p.generate_from_ids(&ids, 600, None, Some(cb))?;
        let (plain, calls) = extract_tool_calls(&r.text, Some(tools));
        let (rest, s_calls) = std::mem::take(&mut *hold.lock().unwrap()).finish(Some(tools));
        let s_content = format!("{}{}", streamed.lock().unwrap(), rest);
        let same = calls.len() == s_calls.len()
            && calls
                .iter()
                .zip(&s_calls)
                .all(|(x, y)| x["function"] == y["function"])
            && !s_content.contains("<function");
        agree += same as usize;
        let want = tools[0]["function"]["name"].as_str().unwrap();
        let good1 = calls.first().map(|c| c["function"]["name"] == want) == Some(true)
            && !plain.contains("<function");
        ok1 += good1 as usize;
        let mut answer = String::new();
        let mut good2 = false;
        if let Some(c) = calls.first() {
            msgs.push(json!({"role": "assistant", "content": plain, "tool_calls": [{
                "id": "call_0", "type": "function",
                "function": {"name": c["function"]["name"],
                    "arguments": serde_json::from_str::<Value>(c["function"]["arguments"].as_str().unwrap())?}}]}));
            msgs.push(
                json!({"role": "tool", "tool_call_id": "call_0", "content": result.to_string()}),
            );
            let ids2 = p
                .tokenizer
                .try_apply_chat_template_json(&msgs, Some(tools), None)?;
            let r2 = p.generate_from_ids(&ids2, 600, None, None)?;
            let (plain2, calls2) = extract_tool_calls(&r2.text, Some(tools));
            answer = plain2
                .rsplit("</think>")
                .next()
                .unwrap_or("")
                .trim()
                .to_string();
            good2 = calls2.is_empty() && expect.iter().any(|e| answer.contains(e));
            ok2 += good2 as usize;
        }
        println!(
            "[{}|{}|{}] {q:?} calls={:?} answer={:?}",
            if good1 { "OK" } else { "--" },
            if good2 { "OK" } else { "--" },
            if same { "same" } else { "DIFF" },
            calls
                .iter()
                .map(|c| c["function"].clone())
                .collect::<Vec<_>>(),
            answer.chars().take(90).collect::<String>()
        );
    }
    let n = scenarios.len();
    println!(
        "vendor tokenizer+template: turn1 call {ok1}/{n}; stream==non-stream {agree}/{n}; turn2 answered {ok2}/{n}"
    );
    Ok(())
}
