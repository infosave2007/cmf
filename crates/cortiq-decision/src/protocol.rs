//! Decisions request and response protocol, Jev/OpenRouter shape (spec §4.4,
//! §4.7, §4.8).
//!
//! **Request** `{model, state, questions}` plus OpenRouter's optional
//! `provider` (object), `user` (≤ 256 characters), `session_id` (≤ 256
//! characters) and `trace` (object), which are accepted and ignored, plus the
//! `cmf` extension `{skill?, oracle?, allow_pii_egress?, round?, explain?,
//! profile?}` (`null` for any optional field = absent). Any other key at the top
//! level or inside `cmf` → 400. Rules ([`parse_request`]):
//!
//! * the body is at most `limits.body_bytes` (1 MiB; else 413), UTF-8 JSON, and
//!   **no object anywhere has a duplicate key** (400, e.g. a criteria key sent
//!   twice) — serde would otherwise keep one of them silently;
//! * `model`: `"cortiq/decision"` or `"cortiq/decision@<12 lowercase hex>"`
//!   (the served model's `model_sha[..12]`, checked by the service); anything
//!   else, a Jev name included → 404 `MODEL_NOT_FOUND`;
//! * `state`: a non-empty string, object or array; a string is used as is, an
//!   object or array as its canonical JSON (answers are then not certified);
//!   at most `limits.state_bytes` (32 KiB) of that text;
//! * `questions`: an object of 1..`limits.questions` (32) questions in request
//!   order, ids 1..128 characters; a question is `{type, instructions,
//!   criteria}` with `type` ∈ {choice, score, noul} and `instructions` a
//!   string, object or array (required);
//!   * choice: `criteria` an object of 2..255 options, option ids 1..256 bytes,
//!     descriptions a string, object, array or null of at most 24,000 bytes
//!     (text or canonical JSON); key order is kept;
//!   * score: `criteria` an array of 2..10 levels (string, object or array,
//!     ≤ 24,000 bytes each), lowest first;
//!   * noul: `criteria` optional; when present an object with only `true`
//!     and/or `false` (descriptions as for choice).
//!
//! **Errors** ([`ApiError`]) have OpenRouter's body with cortiq-router reason
//! codes: `{"error":{"code":<HTTP>,"message":"…","metadata":{"reason":"<code>",
//! "retriable":bool,"request_id":"…","details":{}}}}`.
//!
//! **Validator** ([`validate_decisions_response`]): a line-by-line port of
//! `openrouter_bench.validate_oracle_response` (`openrouter_bench.py:155-199`),
//! the check every stored Jev answer passed, with the model rule as a
//! parameter ([`ModelRule::Jev`] reproduces the original regex).

use crate::answer::Rounding;
use crate::canonical;
use crate::config;
use serde_json::{Map, Value, json};

/// The public model id (spec §4.4).
pub const MODEL_ID: &str = "cortiq/decision";
/// `provider` of every response.
pub const PROVIDER: &str = "Cortiq";
/// Hex characters of the model sha in a pinned model id.
pub const MODEL_SHA_CHARS: usize = 12;
pub const MAX_QUESTION_ID_CHARS: usize = 128;
pub const MIN_CHOICE_OPTIONS: usize = 2;
pub const MAX_CHOICE_OPTIONS: usize = 255;
pub const MAX_OPTION_ID_BYTES: usize = 256;
pub const MAX_DESCRIPTION_BYTES: usize = 24_000;
pub const MIN_SCORE_LEVELS: usize = 2;
pub const MAX_SCORE_LEVELS: usize = 10;
/// `user` and `session_id` (OpenRouter).
pub const MAX_OPENROUTER_STRING_CHARS: usize = 256;
/// Feedback labels (the label rule of the training data).
pub const MAX_LABEL_BYTES: usize = 256;

const TOP_KEYS: [&str; 8] = [
    "model",
    "state",
    "questions",
    "provider",
    "user",
    "session_id",
    "trace",
    "cmf",
];
const CMF_KEYS: [&str; 6] = [
    "skill",
    "oracle",
    "allow_pii_egress",
    "round",
    "explain",
    "profile",
];
const QUESTION_KEYS: [&str; 3] = ["type", "instructions", "criteria"];

// ------------------------------------------------------------------ errors

/// Reason codes (cortiq-router `api.rs:207-284` plus the cascade's).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Reason {
    InvalidRequest,
    Unauthorized,
    QuotaExceeded,
    ModelNotFound,
    AdminDisabled,
    PayloadTooLarge,
    UnsupportedQuestion,
    RateLimited,
    Overloaded,
    Internal,
    OracleUnavailable,
    OracleBudgetExhausted,
    OracleDisabled,
}

impl Reason {
    pub fn code(self) -> &'static str {
        match self {
            Reason::InvalidRequest => "INVALID_REQUEST",
            Reason::Unauthorized => "UNAUTHORIZED",
            Reason::QuotaExceeded => "QUOTA_EXCEEDED",
            Reason::ModelNotFound => "MODEL_NOT_FOUND",
            Reason::AdminDisabled => "ADMIN_DISABLED",
            Reason::PayloadTooLarge => "PAYLOAD_TOO_LARGE",
            Reason::UnsupportedQuestion => "UNSUPPORTED_QUESTION",
            Reason::RateLimited => "RATE_LIMITED",
            Reason::Overloaded => "OVERLOADED",
            Reason::Internal => "INTERNAL",
            Reason::OracleUnavailable => "ORACLE_UNAVAILABLE",
            Reason::OracleBudgetExhausted => "ORACLE_BUDGET_EXHAUSTED",
            Reason::OracleDisabled => "ORACLE_DISABLED",
        }
    }

    /// The HTTP status of the reason (spec §4.8); `INVALID_REQUEST` is 400
    /// except for a missing feedback decision or skill (404,
    /// [`ApiError::not_found`]).
    pub fn status(self) -> u16 {
        match self {
            Reason::InvalidRequest => 400,
            Reason::Unauthorized => 401,
            Reason::QuotaExceeded => 402,
            Reason::ModelNotFound | Reason::AdminDisabled => 404,
            Reason::PayloadTooLarge => 413,
            Reason::UnsupportedQuestion => 422,
            Reason::RateLimited | Reason::Overloaded => 429,
            Reason::Internal => 500,
            Reason::OracleUnavailable => 502,
            Reason::OracleBudgetExhausted | Reason::OracleDisabled => 503,
        }
    }
}

/// An error answer: status, reason, message, optional details and
/// `Retry-After` seconds.
#[derive(Clone, Debug, PartialEq)]
pub struct ApiError {
    pub status: u16,
    pub reason: Reason,
    pub message: String,
    pub details: Option<Box<Map<String, Value>>>,
    pub retry_after: Option<u64>,
}

impl std::fmt::Display for ApiError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{} {}: {}",
            self.status,
            self.reason.code(),
            self.message
        )
    }
}

impl std::error::Error for ApiError {}

impl ApiError {
    pub fn new(reason: Reason, message: impl Into<String>) -> Self {
        Self {
            status: reason.status(),
            reason,
            message: message.into(),
            details: None,
            retry_after: None,
        }
    }

    /// 400 `INVALID_REQUEST`.
    pub fn invalid(message: impl Into<String>) -> Self {
        Self::new(Reason::InvalidRequest, message)
    }

    /// 400 `INVALID_REQUEST` naming the offending field.
    pub fn invalid_field(field: &str, message: impl Into<String>) -> Self {
        Self::invalid(message).with_detail("field", json!(field))
    }

    /// 404 `INVALID_REQUEST` (a feedback decision or a skill that does not exist).
    pub fn not_found(message: impl Into<String>) -> Self {
        let mut e = Self::invalid(message);
        e.status = 404;
        e
    }

    /// 500 `INTERNAL` (the message is logged by the caller, not returned verbatim
    /// when it could carry data; callers pass a generic text).
    pub fn internal(message: impl Into<String>) -> Self {
        Self::new(Reason::Internal, message)
    }

    pub fn with_detail(mut self, key: &str, value: Value) -> Self {
        self.details
            .get_or_insert_with(Default::default)
            .insert(key.to_string(), value);
        self
    }

    pub fn with_retry_after(mut self, seconds: u64) -> Self {
        self.retry_after = Some(seconds);
        self
    }

    /// Worth retrying unchanged: 429, 500, 502.
    pub fn retriable(&self) -> bool {
        matches!(self.status, 429 | 500 | 502)
    }

    /// The OpenRouter-shaped body.
    pub fn body(&self, request_id: &str) -> Value {
        json!({
            "error": {
                "code": self.status,
                "message": self.message,
                "metadata": {
                    "reason": self.reason.code(),
                    "retriable": self.retriable(),
                    "request_id": request_id,
                    "details": Value::Object(self.details.as_deref().cloned().unwrap_or_default()),
                }
            }
        })
    }
}

// ------------------------------------------------------------------ ids

/// `cmf-dec-<unix>-<20 alphanumerics from the OS RNG>` (spec §4.7).
pub fn new_request_id(now: u64) -> String {
    use rand_core::{OsRng, RngCore};
    const ALNUM: &[u8; 62] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789";
    let mut out = String::with_capacity(20);
    let mut buf = [0u8; 32];
    while out.len() < 20 {
        if OsRng.try_fill_bytes(&mut buf).is_err() {
            // Fall back to a hash of the time and a counter; ids only need to
            // be unique, not secret.
            use sha2::{Digest, Sha256};
            static N: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
            let n = N.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            let t = std::time::SystemTime::now()
                .duration_since(std::time::SystemTime::UNIX_EPOCH)
                .map_or(0, |d| d.as_nanos());
            buf = Sha256::digest(format!("{t}-{n}-{}", std::process::id())).into();
        }
        for &b in &buf {
            // Rejection sampling keeps the 62 characters uniform.
            if b < 248 && out.len() < 20 {
                out.push(ALNUM[(b % 62) as usize] as char);
            }
        }
    }
    format!("cmf-dec-{now}-{out}")
}

/// `cortiq/decision@<model_sha[..12]>`.
pub fn model_name(model_sha: &str) -> String {
    format!(
        "{MODEL_ID}@{}",
        &model_sha[..MODEL_SHA_CHARS.min(model_sha.len())]
    )
}

// ------------------------------------------------------------------ request

/// Size limits of a request.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RequestLimits {
    pub body_bytes: usize,
    pub state_bytes: usize,
    pub questions: usize,
}

impl Default for RequestLimits {
    fn default() -> Self {
        (&config::Limits::default()).into()
    }
}

impl From<&config::Limits> for RequestLimits {
    fn from(l: &config::Limits) -> Self {
        Self {
            body_bytes: l.body_bytes,
            state_bytes: l.state_bytes,
            questions: l.questions,
        }
    }
}

/// The requested model.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ModelRef {
    /// `cortiq/decision`: the current generation.
    Latest,
    /// `cortiq/decision@<12 hex>`.
    Pinned(String),
}

/// The `state` of a request.
#[derive(Clone, Debug, PartialEq)]
pub enum State {
    Text(String),
    /// An object or an array (answers are not certified).
    Json(Value),
}

impl State {
    pub fn is_text(&self) -> bool {
        matches!(self, State::Text(_))
    }

    /// The text the model reads: the string, or canonical JSON.
    pub fn text(&self) -> String {
        match self {
            State::Text(s) => s.clone(),
            State::Json(v) => canonical::to_string(v),
        }
    }

    /// The state as JSON.
    pub fn to_value(&self) -> Value {
        match self {
            State::Text(s) => Value::String(s.clone()),
            State::Json(v) => v.clone(),
        }
    }
}

/// Question type.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum QuestionKind {
    Choice,
    Score,
    Noul,
}

impl QuestionKind {
    pub fn as_str(self) -> &'static str {
        match self {
            QuestionKind::Choice => "choice",
            QuestionKind::Score => "score",
            QuestionKind::Noul => "noul",
        }
    }

    pub fn parse(s: &str) -> Option<Self> {
        match s {
            "choice" => Some(QuestionKind::Choice),
            "score" => Some(QuestionKind::Score),
            "noul" => Some(QuestionKind::Noul),
            _ => None,
        }
    }
}

/// One question, as sent.
#[derive(Clone, Debug, PartialEq)]
pub struct Question {
    pub id: String,
    pub kind: QuestionKind,
    pub instructions: Value,
    /// `criteria` as sent (key order kept); `None` for a noul question without it.
    pub criteria: Option<Value>,
}

impl Question {
    /// Choice option ids in request order (empty for other types).
    pub fn options(&self) -> Vec<&str> {
        match (&self.kind, &self.criteria) {
            (QuestionKind::Choice, Some(Value::Object(m))) => {
                m.keys().map(String::as_str).collect()
            }
            _ => Vec::new(),
        }
    }

    /// Score levels, lowest first (empty for other types).
    pub fn levels(&self) -> &[Value] {
        match (&self.kind, &self.criteria) {
            (QuestionKind::Score, Some(Value::Array(a))) => a,
            _ => &[],
        }
    }

    /// The question contract `{type, instructions, criteria?}` (criteria in
    /// request order): the oracle's view of the question and the key of
    /// contract-scoped cache entries and token memo.
    pub fn contract(&self) -> Value {
        let mut m = Map::new();
        m.insert("type".into(), Value::String(self.kind.as_str().into()));
        m.insert("instructions".into(), self.instructions.clone());
        if let Some(c) = &self.criteria {
            m.insert("criteria".into(), c.clone());
        }
        Value::Object(m)
    }
}

/// The policy profile (spec §4.7b).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum Profile {
    /// Only the θ gate (τ not applied); never certified.
    CostSaver,
    /// The certified gate.
    #[default]
    Balanced,
    /// The certified gate and the router's quality-first thresholds (margin ≥
    /// 0.08, novelty ≤ min(θ, 0.50)).
    QualityFirst,
}

impl Profile {
    pub fn as_str(self) -> &'static str {
        match self {
            Profile::CostSaver => "cost-saver",
            Profile::Balanced => "balanced",
            Profile::QualityFirst => "quality-first",
        }
    }

    pub fn parse(s: &str) -> Option<Self> {
        match s {
            "cost-saver" => Some(Profile::CostSaver),
            "balanced" => Some(Profile::Balanced),
            "quality-first" => Some(Profile::QualityFirst),
            _ => None,
        }
    }
}

/// The `cmf` extension.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct CmfOptions {
    /// Force the skill.
    pub skill: Option<String>,
    /// Consent to the oracle for this request (`None`: the configured default).
    pub oracle: Option<bool>,
    pub allow_pii_egress: bool,
    /// `None`: the configured `response.round`; `Some(Exact)`: `null` sent.
    pub round: Option<Rounding>,
    pub explain: bool,
    pub profile: Profile,
}

/// A validated decisions request.
#[derive(Clone, Debug, PartialEq)]
pub struct DecisionRequest {
    pub model: ModelRef,
    pub state: State,
    /// [`State::text`], computed once.
    pub state_text: String,
    pub questions: Vec<Question>,
    pub cmf: CmfOptions,
    pub user: Option<String>,
    pub session_id: Option<String>,
}

impl DecisionRequest {
    pub fn question(&self, id: &str) -> Option<&Question> {
        self.questions.iter().find(|q| q.id == id)
    }
}

fn type_name(v: &Value) -> &'static str {
    match v {
        Value::Null => "null",
        Value::Bool(_) => "boolean",
        Value::Number(_) => "number",
        Value::String(_) => "string",
        Value::Array(_) => "array",
        Value::Object(_) => "object",
    }
}

/// A description (choice option, score level, noul value): string, object or
/// array (or null where `allow_null`), at most 24,000 bytes of text.
fn check_description(field: &str, v: &Value, allow_null: bool) -> Result<(), ApiError> {
    let bytes = match v {
        Value::String(s) => s.len(),
        Value::Object(_) | Value::Array(_) => canonical::to_string(v).len(),
        Value::Null if allow_null => 0,
        other => {
            return Err(ApiError::invalid_field(
                field,
                format!(
                    "{field} must be a string, object or array{}, not {}",
                    if allow_null { " or null" } else { "" },
                    type_name(other)
                ),
            ));
        }
    };
    if bytes > MAX_DESCRIPTION_BYTES {
        return Err(ApiError::invalid_field(
            field,
            format!("{field} is {bytes} bytes (at most {MAX_DESCRIPTION_BYTES})"),
        ));
    }
    Ok(())
}

fn parse_question(id: &str, v: &Value) -> Result<Question, ApiError> {
    let base = format!("questions.{id}");
    let Value::Object(q) = v else {
        return Err(ApiError::invalid_field(
            &base,
            format!("{base} must be an object {{type, instructions, criteria}}"),
        ));
    };
    for k in q.keys() {
        if !QUESTION_KEYS.contains(&k.as_str()) {
            return Err(ApiError::invalid_field(
                &format!("{base}.{k}"),
                format!("unknown field '{k}' in {base} (expected type, instructions, criteria)"),
            ));
        }
    }
    let kind = match q.get("type") {
        Some(Value::String(t)) => QuestionKind::parse(t).ok_or_else(|| {
            ApiError::invalid_field(
                &format!("{base}.type"),
                format!("{base}.type must be choice, score or noul, not '{t}'"),
            )
        })?,
        _ => {
            return Err(ApiError::invalid_field(
                &format!("{base}.type"),
                format!("{base}.type is required: choice, score or noul"),
            ));
        }
    };
    let instructions = match q.get("instructions") {
        Some(v @ (Value::String(_) | Value::Object(_) | Value::Array(_))) => v.clone(),
        Some(other) => {
            return Err(ApiError::invalid_field(
                &format!("{base}.instructions"),
                format!(
                    "{base}.instructions must be a string, object or array, not {}",
                    type_name(other)
                ),
            ));
        }
        None => {
            return Err(ApiError::invalid_field(
                &format!("{base}.instructions"),
                format!("{base}.instructions is required"),
            ));
        }
    };
    let cfield = format!("{base}.criteria");
    let criteria = q.get("criteria");
    let criteria = match kind {
        QuestionKind::Choice => {
            let Some(Value::Object(m)) = criteria else {
                return Err(ApiError::invalid_field(
                    &cfield,
                    format!(
                        "{cfield} must be an object of {MIN_CHOICE_OPTIONS}..{MAX_CHOICE_OPTIONS} options"
                    ),
                ));
            };
            if !(MIN_CHOICE_OPTIONS..=MAX_CHOICE_OPTIONS).contains(&m.len()) {
                return Err(ApiError::invalid_field(
                    &cfield,
                    format!(
                        "a choice has {MIN_CHOICE_OPTIONS}..{MAX_CHOICE_OPTIONS} options, {cfield} has {}",
                        m.len()
                    ),
                ));
            }
            for (k, v) in m {
                if k.is_empty() || k.len() > MAX_OPTION_ID_BYTES {
                    return Err(ApiError::invalid_field(
                        &cfield,
                        format!("option ids are 1..{MAX_OPTION_ID_BYTES} bytes ({cfield})"),
                    ));
                }
                check_description(&format!("{cfield}.{k}"), v, true)?;
            }
            Some(Value::Object(m.clone()))
        }
        QuestionKind::Score => {
            let Some(Value::Array(a)) = criteria else {
                return Err(ApiError::invalid_field(
                    &cfield,
                    format!(
                        "{cfield} must be an array of {MIN_SCORE_LEVELS}..{MAX_SCORE_LEVELS} levels"
                    ),
                ));
            };
            if !(MIN_SCORE_LEVELS..=MAX_SCORE_LEVELS).contains(&a.len()) {
                return Err(ApiError::invalid_field(
                    &cfield,
                    format!(
                        "a score has {MIN_SCORE_LEVELS}..{MAX_SCORE_LEVELS} levels, {cfield} has {}",
                        a.len()
                    ),
                ));
            }
            for (i, v) in a.iter().enumerate() {
                check_description(&format!("{cfield}[{i}]"), v, false)?;
            }
            Some(Value::Array(a.clone()))
        }
        QuestionKind::Noul => match criteria {
            None | Some(Value::Null) => None,
            Some(Value::Object(m)) => {
                for (k, v) in m {
                    if k != "true" && k != "false" {
                        return Err(ApiError::invalid_field(
                            &cfield,
                            format!("noul criteria may only contain true and false, not '{k}'"),
                        ));
                    }
                    check_description(&format!("{cfield}.{k}"), v, true)?;
                }
                Some(Value::Object(m.clone()))
            }
            Some(other) => {
                return Err(ApiError::invalid_field(
                    &cfield,
                    format!(
                        "{cfield} must be an object {{true, false}}, not {}",
                        type_name(other)
                    ),
                ));
            }
        },
    };
    Ok(Question {
        id: id.to_string(),
        kind,
        instructions,
        criteria,
    })
}

fn opt_bool(m: &Map<String, Value>, k: &str, field: &str) -> Result<Option<bool>, ApiError> {
    match m.get(k) {
        None | Some(Value::Null) => Ok(None),
        Some(Value::Bool(b)) => Ok(Some(*b)),
        Some(other) => Err(ApiError::invalid_field(
            field,
            format!("{field} must be a boolean, not {}", type_name(other)),
        )),
    }
}

fn parse_cmf(v: Option<&Value>) -> Result<CmfOptions, ApiError> {
    let m = match v {
        None | Some(Value::Null) => return Ok(CmfOptions::default()),
        Some(Value::Object(m)) => m,
        Some(other) => {
            return Err(ApiError::invalid_field(
                "cmf",
                format!("cmf must be an object, not {}", type_name(other)),
            ));
        }
    };
    for k in m.keys() {
        if !CMF_KEYS.contains(&k.as_str()) {
            return Err(ApiError::invalid_field(
                &format!("cmf.{k}"),
                format!(
                    "unknown field '{k}' in cmf (expected {})",
                    CMF_KEYS.join(", ")
                ),
            ));
        }
    }
    let skill = match m.get("skill") {
        None | Some(Value::Null) => None,
        Some(Value::String(s)) => Some(s.clone()),
        Some(other) => {
            return Err(ApiError::invalid_field(
                "cmf.skill",
                format!("cmf.skill must be a string, not {}", type_name(other)),
            ));
        }
    };
    let round = match m.get("round") {
        None => None,
        Some(Value::Null) => Some(Rounding::Exact),
        Some(Value::Number(n)) if n.as_u64() == Some(u64::from(config::ROUND_HUNDREDTHS)) => {
            Some(Rounding::Hundredths)
        }
        Some(_) => {
            return Err(ApiError::invalid_field(
                "cmf.round",
                "cmf.round must be 2 or null",
            ));
        }
    };
    let profile = match m.get("profile") {
        None | Some(Value::Null) => Profile::default(),
        Some(Value::String(s)) => Profile::parse(s).ok_or_else(|| {
            ApiError::invalid_field(
                "cmf.profile",
                format!("cmf.profile must be cost-saver, balanced or quality-first, not '{s}'"),
            )
        })?,
        Some(other) => {
            return Err(ApiError::invalid_field(
                "cmf.profile",
                format!("cmf.profile must be a string, not {}", type_name(other)),
            ));
        }
    };
    Ok(CmfOptions {
        skill,
        oracle: opt_bool(m, "oracle", "cmf.oracle")?,
        allow_pii_egress: opt_bool(m, "allow_pii_egress", "cmf.allow_pii_egress")?.unwrap_or(false),
        round,
        explain: opt_bool(m, "explain", "cmf.explain")?.unwrap_or(false),
        profile,
    })
}

fn opt_string(
    top: &Map<String, Value>,
    k: &str,
    max_chars: usize,
) -> Result<Option<String>, ApiError> {
    match top.get(k) {
        None | Some(Value::Null) => Ok(None),
        Some(Value::String(s)) => {
            let n = s.chars().count();
            if n > max_chars {
                Err(ApiError::invalid_field(
                    k,
                    format!("{k} is {n} characters (at most {max_chars})"),
                ))
            } else {
                Ok(Some(s.clone()))
            }
        }
        Some(other) => Err(ApiError::invalid_field(
            k,
            format!("{k} must be a string, not {}", type_name(other)),
        )),
    }
}

/// The model id rule.
pub fn parse_model(v: Option<&Value>) -> Result<ModelRef, ApiError> {
    let m = match v {
        Some(Value::String(s)) => s,
        Some(other) => {
            return Err(ApiError::invalid_field(
                "model",
                format!("model must be a string, not {}", type_name(other)),
            ));
        }
        None => return Err(ApiError::invalid_field("model", "model is required")),
    };
    if m == MODEL_ID {
        return Ok(ModelRef::Latest);
    }
    if let Some(sha) = m.strip_prefix(MODEL_ID).and_then(|r| r.strip_prefix('@'))
        && sha.len() == MODEL_SHA_CHARS
        && sha.bytes().all(|b| matches!(b, b'0'..=b'9' | b'a'..=b'f'))
    {
        return Ok(ModelRef::Pinned(sha.to_string()));
    }
    Err(ApiError::new(
        Reason::ModelNotFound,
        format!(
            "model '{m}' is not served here; use \"{MODEL_ID}\" or \"{MODEL_ID}@<model_sha[..12]>\""
        ),
    )
    .with_detail("model", Value::String(m.clone())))
}

/// Parse and validate a decisions request (see the module notes).
pub fn parse_request(body: &[u8], limits: &RequestLimits) -> Result<DecisionRequest, ApiError> {
    if body.len() > limits.body_bytes {
        return Err(ApiError::new(
            Reason::PayloadTooLarge,
            format!(
                "request body is {} bytes (at most {})",
                body.len(),
                limits.body_bytes
            ),
        ));
    }
    let v = parse_json(body)?;
    let Value::Object(top) = v else {
        return Err(ApiError::invalid(
            "the body must be a JSON object {model, state, questions}",
        ));
    };
    for k in top.keys() {
        if !TOP_KEYS.contains(&k.as_str()) {
            return Err(ApiError::invalid_field(
                k,
                format!(
                    "unknown field '{k}' (expected model, state, questions; optional provider, user, session_id, trace, cmf)"
                ),
            ));
        }
    }
    let model = parse_model(top.get("model"))?;
    // OpenRouter's optional fields: accepted, checked for shape, ignored.
    for k in ["provider", "trace"] {
        match top.get(k) {
            None | Some(Value::Null) | Some(Value::Object(_)) => {}
            Some(other) => {
                return Err(ApiError::invalid_field(
                    k,
                    format!("{k} must be an object, not {}", type_name(other)),
                ));
            }
        }
    }
    let user = opt_string(&top, "user", MAX_OPENROUTER_STRING_CHARS)?;
    let session_id = opt_string(&top, "session_id", MAX_OPENROUTER_STRING_CHARS)?;
    let state = match top.get("state") {
        Some(Value::String(s)) if !s.is_empty() => State::Text(s.clone()),
        Some(v @ Value::Object(m)) if !m.is_empty() => State::Json(v.clone()),
        Some(v @ Value::Array(a)) if !a.is_empty() => State::Json(v.clone()),
        Some(Value::String(_) | Value::Object(_) | Value::Array(_)) => {
            return Err(ApiError::invalid_field("state", "state must not be empty"));
        }
        Some(other) => {
            return Err(ApiError::invalid_field(
                "state",
                format!(
                    "state must be a string, object or array, not {}",
                    type_name(other)
                ),
            ));
        }
        None => return Err(ApiError::invalid_field("state", "state is required")),
    };
    let state_text = state.text();
    if state_text.len() > limits.state_bytes {
        return Err(ApiError::invalid_field(
            "state",
            format!(
                "state is {} bytes (at most {})",
                state_text.len(),
                limits.state_bytes
            ),
        ));
    }
    let Some(Value::Object(qs)) = top.get("questions") else {
        return Err(ApiError::invalid_field(
            "questions",
            "questions must be an object {id: question}",
        ));
    };
    if qs.is_empty() || qs.len() > limits.questions {
        return Err(ApiError::invalid_field(
            "questions",
            format!(
                "a request has 1..{} questions, this one {}",
                limits.questions,
                qs.len()
            ),
        ));
    }
    let mut questions = Vec::with_capacity(qs.len());
    for (id, q) in qs {
        let n = id.chars().count();
        if n == 0 || n > MAX_QUESTION_ID_CHARS {
            return Err(ApiError::invalid_field(
                "questions",
                format!("question ids are 1..{MAX_QUESTION_ID_CHARS} characters"),
            ));
        }
        questions.push(parse_question(id, q)?);
    }
    let cmf = parse_cmf(top.get("cmf"))?;
    Ok(DecisionRequest {
        model,
        state,
        state_text,
        questions,
        cmf,
        user,
        session_id,
    })
}

/// `POST /v1/feedback` body `{"id","question","label"}` (spec §5.11).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FeedbackRequest {
    /// The decision's response id.
    pub id: String,
    /// The question id.
    pub question: String,
    /// The correct label.
    pub label: String,
}

/// Parse a feedback body.
pub fn parse_feedback(body: &[u8], limits: &RequestLimits) -> Result<FeedbackRequest, ApiError> {
    if body.len() > limits.body_bytes {
        return Err(ApiError::new(
            Reason::PayloadTooLarge,
            "request body too large",
        ));
    }
    let v = parse_json(body)?;
    let Value::Object(m) = v else {
        return Err(ApiError::invalid(
            "feedback must be an object {id, question, label}",
        ));
    };
    for k in m.keys() {
        if !["id", "question", "label"].contains(&k.as_str()) {
            return Err(ApiError::invalid_field(
                k,
                format!("unknown field '{k}' (expected id, question, label)"),
            ));
        }
    }
    let get = |k: &str, max_bytes: usize| -> Result<String, ApiError> {
        match m.get(k) {
            Some(Value::String(s)) if !s.is_empty() && s.len() <= max_bytes => Ok(s.clone()),
            _ => Err(ApiError::invalid_field(
                k,
                format!("{k} must be a string of 1..{max_bytes} bytes"),
            )),
        }
    };
    Ok(FeedbackRequest {
        id: get("id", 256)?,
        question: get("question", 4 * MAX_QUESTION_ID_CHARS)?,
        label: get("label", MAX_LABEL_BYTES)?,
    })
}

/// Parse JSON (correctly rounded floats) and refuse a duplicate object key.
pub fn parse_json(body: &[u8]) -> Result<Value, ApiError> {
    let v = canonical::parse(body)
        .map_err(|e| ApiError::invalid(format!("the body is not valid JSON: {e}")))?;
    if let Some((path, key)) = find_duplicate_key(body) {
        let at = if path.is_empty() {
            "the top-level object".to_string()
        } else {
            path.clone()
        };
        return Err(ApiError::invalid(format!("duplicate key '{key}' in {at}"))
            .with_detail("field", Value::String(path))
            .with_detail("key", Value::String(key)));
    }
    Ok(v)
}

// ------------------------------------------------------------------ duplicate keys

/// A walk over JSON text that is already known to be valid, reporting the
/// first object that holds one key twice: `(path, key)`, the path in dotted
/// form (`questions.team.criteria`, arrays as `[i]`).
pub fn find_duplicate_key(text: &[u8]) -> Option<(String, String)> {
    let mut s = Scan { s: text, i: 0 };
    let mut path = Vec::new();
    s.ws();
    s.value(&mut path, 0)
}

struct Scan<'a> {
    s: &'a [u8],
    i: usize,
}

fn render_path(path: &[String]) -> String {
    let mut out = String::new();
    for p in path {
        if p.starts_with('[') {
            out.push_str(p);
        } else {
            if !out.is_empty() {
                out.push('.');
            }
            out.push_str(p);
        }
    }
    out
}

impl Scan<'_> {
    fn peek(&self) -> u8 {
        self.s.get(self.i).copied().unwrap_or(0)
    }

    fn ws(&mut self) {
        while matches!(self.peek(), b' ' | b'\t' | b'\n' | b'\r') {
            self.i += 1;
        }
    }

    fn value(&mut self, path: &mut Vec<String>, depth: usize) -> Option<(String, String)> {
        if depth > canonical::MAX_DEPTH + 1 {
            return None;
        }
        match self.peek() {
            b'{' => {
                self.i += 1;
                let mut seen = std::collections::HashSet::new();
                self.ws();
                if self.peek() == b'}' {
                    self.i += 1;
                    return None;
                }
                loop {
                    self.ws();
                    let key = self.string();
                    if !seen.insert(key.clone()) {
                        return Some((render_path(path), key));
                    }
                    self.ws();
                    self.i += 1; // ':'
                    self.ws();
                    path.push(key);
                    let dup = self.value(path, depth + 1);
                    path.pop();
                    if dup.is_some() {
                        return dup;
                    }
                    self.ws();
                    let c = self.peek();
                    self.i += 1;
                    if c != b',' {
                        return None;
                    }
                }
            }
            b'[' => {
                self.i += 1;
                self.ws();
                if self.peek() == b']' {
                    self.i += 1;
                    return None;
                }
                let mut k = 0usize;
                loop {
                    self.ws();
                    path.push(format!("[{k}]"));
                    let dup = self.value(path, depth + 1);
                    path.pop();
                    if dup.is_some() {
                        return dup;
                    }
                    k += 1;
                    self.ws();
                    let c = self.peek();
                    self.i += 1;
                    if c != b',' {
                        return None;
                    }
                }
            }
            b'"' => {
                self.string();
                None
            }
            _ => {
                while !matches!(
                    self.peek(),
                    b',' | b'}' | b']' | b' ' | b'\t' | b'\n' | b'\r' | 0
                ) {
                    self.i += 1;
                }
                None
            }
        }
    }

    fn hex4(&mut self) -> u32 {
        let mut v = 0u32;
        for _ in 0..4 {
            let c = self.peek();
            self.i += 1;
            v = v * 16 + (c as char).to_digit(16).unwrap_or(0);
        }
        v
    }

    /// The decoded string at `self.i` (which is a `"`).
    fn string(&mut self) -> String {
        let mut out: Vec<u8> = Vec::new();
        self.i += 1;
        loop {
            let c = self.peek();
            if c == 0 && self.i >= self.s.len() {
                break;
            }
            self.i += 1;
            match c {
                b'"' => break,
                b'\\' => {
                    let e = self.peek();
                    self.i += 1;
                    let ch = match e {
                        b'"' => '"',
                        b'\\' => '\\',
                        b'/' => '/',
                        b'b' => '\u{8}',
                        b'f' => '\u{c}',
                        b'n' => '\n',
                        b'r' => '\r',
                        b't' => '\t',
                        b'u' => {
                            let hi = self.hex4();
                            if (0xD800..0xDC00).contains(&hi)
                                && self.peek() == b'\\'
                                && self.s.get(self.i + 1) == Some(&b'u')
                            {
                                let save = self.i;
                                self.i += 2;
                                let lo = self.hex4();
                                if (0xDC00..0xE000).contains(&lo) {
                                    char::from_u32(0x10000 + ((hi - 0xD800) << 10) + (lo - 0xDC00))
                                        .unwrap_or('\u{FFFD}')
                                } else {
                                    self.i = save;
                                    '\u{FFFD}'
                                }
                            } else {
                                char::from_u32(hi).unwrap_or('\u{FFFD}')
                            }
                        }
                        _ => '\u{FFFD}',
                    };
                    let mut b = [0u8; 4];
                    out.extend_from_slice(ch.encode_utf8(&mut b).as_bytes());
                }
                _ => out.push(c),
            }
        }
        String::from_utf8_lossy(&out).into_owned()
    }
}

// ------------------------------------------------------------------ validator

/// The model a response must name.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ModelRule {
    /// `typesafe/jev-1\.13(?:-\d{8})?` (the original regex; `\d` as ASCII digits).
    Jev,
    /// `cortiq/decision(@[0-9a-f]{12})?`.
    Cortiq,
    /// Exactly this id.
    Exact(String),
}

impl ModelRule {
    pub fn matches(&self, model: &str) -> bool {
        match self {
            ModelRule::Jev => match model.strip_prefix("typesafe/jev-1.13") {
                Some("") => true,
                Some(rest) => {
                    rest.len() == 9
                        && rest.starts_with('-')
                        && rest[1..].bytes().all(|b| b.is_ascii_digit())
                }
                None => false,
            },
            ModelRule::Cortiq => parse_model(Some(&Value::String(model.to_string()))).is_ok(),
            ModelRule::Exact(m) => m == model,
        }
    }
}

/// `_probability`: an int or float (not a bool), finite, in [0, 1].
fn probability(v: Option<&Value>) -> Option<f64> {
    let x = v?.as_f64()?;
    (x.is_finite() && (0.0..=1.0).contains(&x)).then_some(x)
}

/// Port of `openrouter_bench.validate_oracle_response(result, questions)`:
/// `questions` is the wire `questions` object (`{qid: {type, criteria, …}}`).
/// Returns the Python error message on failure.
pub fn validate_decisions_response(
    result: &Value,
    questions: &Map<String, Value>,
    rule: &ModelRule,
) -> Result<(), String> {
    let err = |m: &str| Err(m.to_string());
    let Value::Object(r) = result else {
        return err("unexpected oracle model");
    };
    let model_ok = r
        .get("model")
        .and_then(Value::as_str)
        .is_some_and(|m| rule.matches(m));
    if !model_ok {
        return err("unexpected oracle model");
    }
    let Some(Value::Object(answers)) = r.get("answers") else {
        return err("oracle question IDs mismatch");
    };
    if answers.len() != questions.len() || !questions.keys().all(|k| answers.contains_key(k)) {
        return err("oracle question IDs mismatch");
    }
    for (qid, q) in questions {
        let a = &answers[qid];
        let qtype = q.get("type").and_then(Value::as_str);
        let Value::Object(a) = a else {
            return err("oracle answer type mismatch");
        };
        if a.get("type").and_then(Value::as_str) != qtype || qtype.is_none() {
            return err("oracle answer type mismatch");
        }
        let qtype = qtype.unwrap_or_default();
        if qtype == "noul" {
            if probability(a.get("noul")).is_none() {
                return err("invalid noul");
            }
            continue;
        }
        let expected: Vec<String> = if qtype == "choice" {
            match q.get("criteria") {
                Some(Value::Object(c)) => c.keys().cloned().collect(),
                _ => return err("invalid oracle probabilities"),
            }
        } else {
            let n = q
                .get("criteria")
                .and_then(Value::as_array)
                .map_or(0, Vec::len);
            (0..n).map(|i| i.to_string()).collect()
        };
        let Some(Value::Object(p)) = a.get("probabilities") else {
            return err("invalid oracle probabilities");
        };
        let same_keys = p.len() == expected.len() && expected.iter().all(|k| p.contains_key(k));
        if !same_keys || !p.values().all(|v| probability(Some(v)).is_some()) {
            return err("invalid oracle probabilities");
        }
        let vals: Vec<f64> = p.values().map(|v| v.as_f64().unwrap_or(f64::NAN)).collect();
        let quantized = vals
            .iter()
            .all(|v| (v * 100.0 - (v * 100.0).round()).abs() < 1e-7);
        let tolerance = if quantized {
            0.005 * vals.len() as f64 + 1e-8
        } else {
            0.01000001
        };
        let sum: f64 = vals.iter().fold(0.0, |s, v| s + v);
        if sum <= 0.0 || (sum - 1.0).abs() > tolerance || probability(a.get("confidence")).is_none()
        {
            return err("invalid oracle distribution/confidence");
        }
        if qtype == "choice" {
            let max = vals.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            let ok = a
                .get("choice")
                .and_then(Value::as_str)
                .and_then(|c| p.get(c))
                .and_then(Value::as_f64)
                .is_some_and(|pc| pc + 1e-8 >= max);
            if !ok {
                return err("invalid oracle choice");
            }
        } else {
            let n = p.len();
            let score = a.get("score").and_then(Value::as_f64);
            let Some(score) =
                score.filter(|s| s.is_finite() && *s >= 0.0 && *s <= (n as f64 - 1.0))
            else {
                return err("invalid oracle score");
            };
            let legend_ok = match a.get("legend") {
                Some(Value::Object(l)) => {
                    l.len() == expected.len() && expected.iter().all(|k| l.contains_key(k))
                }
                _ => false,
            };
            if !legend_ok {
                return err("invalid oracle legend");
            }
            let weighted = p.iter().fold(0.0f64, |s, (k, v)| {
                s + k.parse::<f64>().unwrap_or(f64::NAN) * v.as_f64().unwrap_or(f64::NAN)
            });
            let spread: f64 = (0..n).map(|i| i as f64).sum();
            if (score - weighted).abs() > 0.005 * spread + 0.005 + 1e-7 {
                return err("oracle score inconsistent with distribution");
            }
        }
    }
    let usage = match r.get("usage") {
        None => Map::new(),
        Some(Value::Object(u)) => u.clone(),
        Some(_) => return err("invalid oracle cost"),
    };
    let cost_ok = usage
        .get("cost")
        .and_then(Value::as_f64)
        .is_some_and(|c| c.is_finite() && c >= 0.0);
    if !cost_ok {
        return err("invalid oracle cost");
    }
    for k in ["input_tokens", "output_tokens"] {
        let ok = usage.get(k).is_some_and(|v| match v {
            Value::Number(n) => n.is_u64() || n.as_i64().is_some_and(|i| i >= 0),
            _ => false,
        });
        if !ok {
            return err("invalid oracle token usage");
        }
    }
    Ok(())
}

/// The wire `questions` object of a request.
pub fn wire_questions(req: &DecisionRequest) -> Map<String, Value> {
    req.questions
        .iter()
        .map(|q| (q.id.clone(), q.contract()))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn duplicate_keys_are_found_with_their_path() {
        let d = |s: &str| find_duplicate_key(s.as_bytes());
        assert_eq!(d(r#"{"a":1,"b":{"c":[1,{"x":1}]}}"#), None);
        assert_eq!(d(r#"{"a":1,"a":2}"#), Some((String::new(), "a".into())));
        assert_eq!(
            d(r#"{"q":{"t":{"criteria":{"x":"1","y":null,"x":"3"}}}}"#),
            Some(("q.t.criteria".into(), "x".into()))
        );
        assert_eq!(
            d(r#"{"s":[{"k":1},{"k":1,"k":2}]}"#),
            Some(("s[1]".into(), "k".into()))
        );
        // Escapes decode before comparison.
        assert_eq!(d(r#"{"ab":1,"ab":2}"#), Some((String::new(), "ab".into())));
        assert_eq!(d(r#"{"a\"":1,"a":2}"#), None);
        assert_eq!(d(r#"{"😀":1,"😀":2}"#), Some((String::new(), "😀".into())));
    }

    #[test]
    fn request_ids_have_the_documented_shape() {
        let id = new_request_id(1_790_500_000);
        let rest = id.strip_prefix("cmf-dec-1790500000-").unwrap();
        assert_eq!(rest.len(), 20);
        assert!(rest.bytes().all(|b| b.is_ascii_alphanumeric()));
        assert_ne!(new_request_id(1), new_request_id(1));
    }

    #[test]
    fn model_rules() {
        assert!(ModelRule::Jev.matches("typesafe/jev-1.13"));
        assert!(ModelRule::Jev.matches("typesafe/jev-1.13-20260917"));
        assert!(!ModelRule::Jev.matches("typesafe/jev-1.13-2026091"));
        assert!(!ModelRule::Jev.matches("typesafe/jev-1.14"));
        assert!(ModelRule::Cortiq.matches("cortiq/decision"));
        assert!(ModelRule::Cortiq.matches("cortiq/decision@3f2a9c1b7d44"));
        assert!(!ModelRule::Cortiq.matches("cortiq/decision@3F2A9C1B7D44"));
        assert!(!ModelRule::Cortiq.matches("typesafe/jev-1.13"));
    }
}
