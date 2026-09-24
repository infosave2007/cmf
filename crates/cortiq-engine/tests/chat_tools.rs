//! Tool calling through the FILE's own chat template — the contract the
//! server relies on, held against the real Qwen3 template (extracted
//! verbatim from a converted .cmf, `fixtures/qwen3_chat_template.jinja`).
//!
//! The format never lacked tools: this template has carried the
//! `{%- if tools %}` branch, the `<tool_call>` output grammar, the
//! `tool_call.function` unwrapping and the `<tool_response>` rendering
//! of `role: "tool"` since the first convert. What was missing was a
//! renderer that passed the fields through — these tests pin that it
//! now does, in the shapes the OpenAI protocol actually sends.

use cortiq_engine::tokenizer::Tokenizer;

fn tok_with_template() -> Tokenizer {
    let mut t = Tokenizer::byte_level();
    t.chat_template = Some(include_str!("fixtures/qwen3_chat_template.jinja").to_string());
    t
}

fn j(v: serde_json::Value) -> serde_json::Value {
    v
}

#[test]
fn tools_render_into_the_system_block() {
    let t = tok_with_template();
    let tools = vec![j(serde_json::json!({
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get current weather for a city",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"]
            }
        }
    }))];
    let msgs = vec![j(
        serde_json::json!({"role": "user", "content": "Weather in Paris?"}),
    )];
    let text = t.render_chat_json(&msgs, Some(&tools), None).unwrap();

    assert!(text.contains("<tools>"), "tool signatures section missing");
    assert!(
        text.contains("\"get_weather\""),
        "the function declaration must reach the prompt"
    );
    assert!(
        text.contains("<tool_call>"),
        "the output-grammar instruction must reach the prompt"
    );
    // And without tools the block must not appear — the template guards
    // on `tools` being defined, which is exactly what tool_choice:"none"
    // relies on server-side.
    let bare = t.render_chat_json(&msgs, None, None).unwrap();
    assert!(
        !bare.contains("<tools>"),
        "tools block must be absent without tools"
    );
}

/// OpenAI history: an assistant turn whose tool_calls carry `function`
/// wrappers and STRING arguments — both must round-trip through the
/// template into the `<tool_call>` grammar the model was trained on.
#[test]
fn assistant_history_with_tool_calls_renders() {
    let t = tok_with_template();
    let msgs = vec![
        j(serde_json::json!({"role": "user", "content": "Weather in Paris?"})),
        j(serde_json::json!({
            "role": "assistant",
            "content": "",
            "tool_calls": [{
                "id": "call_1",
                "type": "function",
                "function": {"name": "get_weather", "arguments": "{\"city\": \"Paris\"}"}
            }]
        })),
        j(serde_json::json!({
            "role": "tool",
            "tool_call_id": "call_1",
            "content": "{\"temp_c\": 18, \"sky\": \"clear\"}"
        })),
    ];
    let text = t.render_chat_json(&msgs, None, None).unwrap();

    assert!(
        text.contains(
            "<tool_call>\n{\"name\": \"get_weather\", \"arguments\": {\"city\": \"Paris\"}}"
        ),
        "assistant tool_calls must render in the trained grammar, got:\n{text}"
    );
    assert!(
        text.contains("<tool_response>\n{\"temp_c\": 18"),
        "role:tool results must render as <tool_response>, got:\n{text}"
    );
}

/// Consecutive tool results share one user turn — the template merges
/// them, and a renderer that broke the merge would double the im_start
/// tokens and shift every position after.
#[test]
fn consecutive_tool_results_share_one_user_turn() {
    let t = tok_with_template();
    let msgs = vec![
        j(serde_json::json!({"role": "user", "content": "Two cities."})),
        j(serde_json::json!({
            "role": "assistant", "content": "",
            "tool_calls": [
                {"type": "function", "function": {"name": "get_weather", "arguments": "{\"city\": \"Paris\"}"}},
                {"type": "function", "function": {"name": "get_weather", "arguments": "{\"city\": \"Oslo\"}"}}
            ]
        })),
        j(serde_json::json!({"role": "tool", "content": "18C"})),
        j(serde_json::json!({"role": "tool", "content": "7C"})),
    ];
    let text = t.render_chat_json(&msgs, None, None).unwrap();
    let user_starts = text.matches("<|im_start|>user").count();
    assert_eq!(
        user_starts, 2,
        "the two tool results must share one user turn:\n{text}"
    );
    assert_eq!(text.matches("<tool_response>").count(), 2);
}

// ── Nanbeige 4.2 — the template that found three renderer gaps ──

fn nanbeige_tok() -> Tokenizer {
    let mut t = Tokenizer::byte_level();
    t.chat_template = Some(include_str!("fixtures/nanbeige42_chat_template.jinja").to_string());
    t
}

/// Nanbeige's tools branch calls transformers' `visible_text()` helper,
/// reads dicts with python `.get`, and picks its grammar from a
/// `tool_call_format` variable whose UNDEFINED value selects the XML
/// branch. Any of the three silently produced a toolless prompt before;
/// this pins all of them at once, on the template byte-for-byte from a
/// user's converted file.
#[test]
fn nanbeige_tools_render_in_json_grammar() {
    let t = nanbeige_tok();
    let tools = vec![j(serde_json::json!({
        "type": "function",
        "function": {"name": "get_weather", "description": "d",
                      "parameters": {"type": "object", "properties": {}}}
    }))];
    let msgs = vec![j(
        serde_json::json!({"role": "user", "content": "Weather in Paris?"}),
    )];
    let text = t.render_chat_json(&msgs, Some(&tools), None).unwrap();
    assert!(
        text.contains("<tools>"),
        "tools section missing:\n{}",
        &text[..text.len().min(400)]
    );
    assert!(text.contains("\"get_weather\""));
    assert!(
        text.contains("\"arguments\": <args-json-object>"),
        "the JSON grammar must be selected, not the XML default"
    );
    assert!(
        !text.contains("<function=example_function_name>"),
        "the XML branch must not be the one rendered"
    );
}

#[test]
fn nanbeige_tool_history_round_trips() {
    let t = nanbeige_tok();
    let msgs = vec![
        j(serde_json::json!({"role": "user", "content": "Weather?"})),
        j(serde_json::json!({"role": "assistant", "content": "",
            "tool_calls": [{"type": "function",
                "function": {"name": "get_weather", "arguments": "{\"city\": \"Paris\"}"}}]})),
        j(serde_json::json!({"role": "tool", "content": "18C"})),
    ];
    let text = t.render_chat_json(&msgs, None, None).unwrap();
    assert!(text.contains("get_weather"), "call history lost:\n{text}");
    assert!(
        text.contains("<tool_response>"),
        "tool result lost:\n{text}"
    );
}

/// The server asks `try_apply_chat_template_json` when tools are present
/// and answers 400 on Err: a render failure must surface there, while the
/// lenient call keeps its ChatML fallback for plain chat.
#[test]
fn a_failing_template_is_reported_not_swallowed() {
    let mut t = Tokenizer::byte_level();
    t.chat_template = Some(
        "{% for m in messages %}{{ m.content | no_such_filter }}{% endfor %}".to_string(),
    );
    let msgs = vec![serde_json::json!({"role": "user", "content": "hi"})];
    let tools = vec![serde_json::json!({"type": "function", "function": {"name": "f"}})];
    let err = t
        .try_apply_chat_template_json(&msgs, Some(&tools), None)
        .unwrap_err();
    assert!(err.contains("no_such_filter"), "{err}");
    assert!(!t.apply_chat_template_json(&msgs, None, None).is_empty());
    // `raise_exception` carries the template's own message out.
    t.chat_template = Some("{{ raise_exception('Conversation roles must alternate') }}".into());
    let err = t.try_apply_chat_template_json(&msgs, None, None).unwrap_err();
    assert!(err.contains("roles must alternate"), "{err}");
}

// ── MiniCPM5 — the vendor template, byte-for-byte ──

/// The ORIGINAL openbmb/MiniCPM5-2B template calls
/// `tojson(ensure_ascii=False)`, which minijinja's built-in rejected:
/// every request with tools failed to render and the server served a
/// toolless ChatML prompt instead. The fixture holds transformers'
/// renders of the same conversations — tool declarations with
/// non-ASCII text, `<>&"`, non-alphabetical key order, tool-call
/// history with CDATA, int and bool arguments, a dict tool result —
/// and every one must come out identical.
#[test]
fn minicpm5_vendor_template_renders_like_transformers() {
    let mut t = Tokenizer::byte_level();
    t.chat_template = Some(include_str!("fixtures/minicpm5_chat_template.jinja").to_string());
    let fx: serde_json::Value =
        serde_json::from_str(include_str!("fixtures/minicpm5_template_renders.json")).unwrap();
    let cases = fx["cases"].as_array().unwrap();
    let mut with_tools = 0;
    for (i, case) in cases.iter().enumerate() {
        let msgs: Vec<serde_json::Value> = case["messages"].as_array().unwrap().clone();
        let tools: Option<Vec<serde_json::Value>> =
            case.get("tools").and_then(|t| t.as_array()).cloned();
        with_tools += tools.is_some() as usize;
        let think = case.get("enable_thinking").and_then(|v| v.as_bool());
        if let Err(e) = t.try_apply_chat_template_json(&msgs, tools.as_deref(), think) {
            panic!("case {i}: render error {e}");
        }
        let got = t.render_chat_json(&msgs, tools.as_deref(), think).unwrap();
        assert_eq!(got, case["hf_text"].as_str().unwrap(), "case {i}");
    }
    assert!(with_tools >= 3, "the fixture must exercise the tools branch");
}
